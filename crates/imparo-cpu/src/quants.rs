//! Block interiors: one row codec per block-quantised GGML type.
//!
//! `imparo_gguf::tensor_layout` describes every type's OUTSIDE -- how many elements a block
//! holds and how many bytes it takes. This module is the INSIDE: where the scales sit, how
//! the quants are packed, and what arithmetic turns them back into floats. The two together
//! are the whole description of a quantised weight, and adding a format to the engine means
//! adding a row here and a row there -- not a new kernel family and not a new code path.
//!
//! Every codec is ported from llama.cpp's `dequantize_row_*` (ggml-quants.c) with the byte
//! offsets read off ggml-common.h, because the reference implementation IS the format:
//!
//! ```text
//!     Q4_0    18 B   d f16 @0, qs[16] @2                     (q - 8) * d
//!     Q4_1    20 B   d f16 @0, m f16 @2, qs[16] @4           q * d + m
//!     Q5_0    22 B   d f16 @0, qh[4] @2, qs[16] @6           (q|hi - 16) * d
//!     Q5_1    24 B   d f16 @0, m f16 @2, qh[4] @4, qs[16] @8 (q|hi) * d + m
//!     Q8_0    34 B   d f16 @0, qs[32] i8 @2                  q * d
//!     Q2_K    84 B   scales[16] @0, qs[64] @16, d f16 @80, dmin f16 @82
//!     Q3_K   110 B   hmask[32] @0, qs[64] @32, scales[12] @96, d f16 @108
//!     Q4_K   144 B   d f16 @0, dmin f16 @2, scales[12] @4, qs[128] @16
//!     Q5_K   176 B   d f16 @0, dmin f16 @2, scales[12] @4, qh[32] @16, qs[128] @48
//!     Q6_K   210 B   ql[128] @0, qh[64] @128, scales[16] i8 @192, d f16 @208
//!     IQ4_NL  18 B   d f16 @0, qs[16] @2                     codebook[q] * d
//!     IQ4_XS 136 B   d f16 @0, scales_h u16 @2, scales_l[4] @4, qs[128] @8
//!     IQ3_S  110 B   d f16 @0, qs[64] @2, qh[8] @66, signs[32] @74, scales[4] @106
//!     IQ2_XS  74 B   d f16 @0, qs[32] u16 @2, scales[8] @66
//!     IQ2_S   82 B   d f16 @0, qs[32] @2, signs[32] @34, qh[8] @66, scales[8] @74
//!     IQ3_XXS 98 B   d f16 @0, qs[64] @2, scales+signs[8] u32 @66
//!     IQ2_XXS 66 B   d f16 @0, qs[32] u16 @2 (per 32 values: 4 index bytes, then 4 x 7-bit signs + 4-bit scale)
//!     IQ1_S   50 B   d f16 @0, qs[32] @2, qh[8] u16 @34 (per 32 values: 4 index bytes; qh = 3-bit scale, delta sign, 4 x 3-bit index tops)
//!     IQ1_M   56 B   qs[32] @0, qh[16] @32, scales[4] u16 @48 (two 3-bit scales per 32 values; d f16 in the words' top nibbles)
//! ```
//!
//! Three families, three different mistakes to make. The legacy types (Q4_0..Q8_0) carry one
//! scale per 32 values and differ only in whether they are symmetric ((q-8)*d) or asymmetric
//! (q*d+m); a k-quant carries a 256-element SUPER-BLOCK whose sub-blocks each get their own
//! 4- or 6-bit scale packed against a super-block scale; an IQ type replaces the quant value
//! itself with an index into a fixed codebook, so the bits are not a number at all.
//!
//! Getting a sign convention or a nibble pairing wrong reads as plausible garbage, not as a
//! crash -- the same failure class as the rotated KV basis in #156 -- so every codec here is
//! gated against the reference dequantiser, never eyeballed.

use imparo_gguf::weights::f16_to_f32;

/// A row codec: `(row bytes, output row)`. The row's length is a whole number of blocks by
/// construction (`row_bytes_len` derives it from `tensor_layout`), so a codec never has a
/// partial block to reason about.
pub type RowCodec = fn(&[u8], &mut [f32]);

/// One row of a TILE-MAJOR tensor, gathered back into row-major blocks and decoded.
///
/// The rule moves a block's bytes; it does not change what they mean. So a TM reader is
/// the row codec plus the rule's addresses -- gather the scale spans and the payload spans
/// of one block back into a block-shaped buffer, then hand it to the same codec the
/// row-major path uses. One decode implementation for both layouts is the point: a second
/// one would be a second chance to disagree about a block interior.
///
/// `data` is the WHOLE tensor, `n_in` its row width, `r` the row.
///
/// # Panics
/// When `kind` is not a tile-major type this engine knows.
pub fn tm_row(data: &[u8], kind: u32, n_in: usize, r: usize, out: &mut [f32]) {
    let rule = imparo_gguf::weights::tm_rule_to(kind)
        .unwrap_or_else(|| panic!("tm_row: {kind} is not a tile-major type"));
    let codec = row_codec(rule.from)
        .unwrap_or_else(|| panic!("tm_row: no codec for {}", rule.from_name));
    let blocks = n_in / rule.block_elems;
    let bb = rule.block_bytes();
    let mut block = vec![0_u8; bb];
    let mut vals = vec![0.0_f32; rule.block_elems];
    for b in 0..blocks {
        let mut at = rule.scale_offset(r, b, blocks, 0);
        for &(off, len) in rule.scale_spans {
            block[off..off + len].copy_from_slice(&data[at..at + len]);
            at += len;
        }
        let mut at = rule.payload_offset(r, b, blocks);
        for (off, len) in rule.payload_spans() {
            block[off..off + len].copy_from_slice(&data[at..at + len]);
            at += len;
        }
        codec(&block, &mut vals);
        out[b * rule.block_elems..(b + 1) * rule.block_elems].copy_from_slice(&vals);
    }
}

/// The codec for a row-major block type, or `None` when the engine cannot read it.
///
/// This is the ONE place a quantised type becomes readable. A caller that wants to know
/// whether a file will load asks here; a caller that wants the numbers calls what comes back.
/// Tile-major kinds are absent on purpose: their bytes are not a row slice, so they are read
/// through the `TmRule` in imparo-gguf, not through a row codec.
#[must_use]
pub fn row_codec(ggml_type: u32) -> Option<RowCodec> {
    Some(match ggml_type {
        2 => q4_0,
        3 => q4_1,
        6 => q5_0,
        7 => q5_1,
        8 => q8_0,
        10 => q2_k,
        11 => q3_k,
        12 => q4_k,
        13 => q5_k,
        14 => q6_k,
        20 => iq4_nl,
        16 => iq2_xxs,
        17 => iq2_xs,
        18 => iq3_xxs,
        19 => iq1_s,
        21 => iq3_s,
        22 => iq2_s,
        23 => iq4_xs,
        29 => iq1_m,
        _ => return None,
    })
}

const QK: usize = 32;
const QK_K: usize = 256;
const K_SCALE_SIZE: usize = 12;

fn f16_at(b: &[u8], off: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[off], b[off + 1]]))
}

// ---- legacy: one scale per 32 values ------------------------------------------------

/// The LOW nibble of byte i is element i and the HIGH nibble is element i+16 -- the halves
/// of a block are interleaved in the bytes, not consecutive. Getting that pairing wrong
/// produces plausible garbage.
fn q4_0(row: &[u8], out: &mut [f32]) {
    for (b, blk) in row.chunks_exact(18).enumerate() {
        let d = f16_at(blk, 0);
        let o = b * QK;
        for i in 0..QK / 2 {
            let byte = blk[2 + i];
            out[o + i] = (f32::from(byte & 0x0F) - 8.0) * d;
            out[o + i + QK / 2] = (f32::from(byte >> 4) - 8.0) * d;
        }
    }
}

fn q4_1(row: &[u8], out: &mut [f32]) {
    for (b, blk) in row.chunks_exact(20).enumerate() {
        let (d, m) = (f16_at(blk, 0), f16_at(blk, 2));
        let o = b * QK;
        for i in 0..QK / 2 {
            let byte = blk[4 + i];
            out[o + i] = f32::from(byte & 0x0F) * d + m;
            out[o + i + QK / 2] = f32::from(byte >> 4) * d + m;
        }
    }
}

fn q5_0(row: &[u8], out: &mut [f32]) {
    for (b, blk) in row.chunks_exact(22).enumerate() {
        let d = f16_at(blk, 0);
        let qh = u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]);
        let o = b * QK;
        for i in 0..QK / 2 {
            let byte = blk[6 + i];
            let h0 = ((qh >> i) << 4) & 0x10;
            let h1 = (qh >> (i + 12)) & 0x10;
            out[o + i] = ((f32::from(byte & 0x0F) + h0 as f32) - 16.0) * d;
            out[o + i + QK / 2] = ((f32::from(byte >> 4) + h1 as f32) - 16.0) * d;
        }
    }
}

fn q5_1(row: &[u8], out: &mut [f32]) {
    for (b, blk) in row.chunks_exact(24).enumerate() {
        let (d, m) = (f16_at(blk, 0), f16_at(blk, 2));
        let qh = u32::from_le_bytes([blk[4], blk[5], blk[6], blk[7]]);
        let o = b * QK;
        for i in 0..QK / 2 {
            let byte = blk[8 + i];
            let h0 = ((qh >> i) << 4) & 0x10;
            let h1 = (qh >> (i + 12)) & 0x10;
            out[o + i] = (f32::from(byte & 0x0F) + h0 as f32) * d + m;
            out[o + i + QK / 2] = (f32::from(byte >> 4) + h1 as f32) * d + m;
        }
    }
}

/// One Q8_0 block is an f16 scale followed by 32 SIGNED bytes, element i at byte 2+i. No
/// nibble pairing and no -8 bias -- reusing the Q4_0 reader here would read 16 plausible
/// values instead of 32 correct ones.
fn q8_0(row: &[u8], out: &mut [f32]) {
    for (b, blk) in row.chunks_exact(34).enumerate() {
        let d = f16_at(blk, 0);
        let o = b * QK;
        for i in 0..QK {
            out[o + i] = f32::from(blk[2 + i] as i8) * d;
        }
    }
}

// ---- k-quants: a 256-element super-block of sub-blocks -------------------------------

/// The 6-bit sub-block scale and min for sub-block `j`, unpacked from the 12 packed bytes.
/// llama.cpp's `get_scale_min_k4`: the first four pairs are plain 6-bit fields, the last
/// four steal their top two bits from the first four's spare bits.
fn k_scale_min(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// Q2_K: 16 sub-blocks of 16 values, each with a 4-bit scale and a 4-bit min in one byte.
/// Two bits per value, four values per byte, read by shifting 0/2/4/6 across 32-byte halves.
fn q2_k(row: &[u8], out: &mut [f32]) {
    for (b, blk) in row.chunks_exact(84).enumerate() {
        let scales = &blk[0..QK_K / 16];
        let qs = &blk[16..16 + QK_K / 4];
        let (d, dmin) = (f16_at(blk, 80), f16_at(blk, 82));
        let mut o = b * QK_K;
        let mut is = 0;
        for n in 0..2 {
            let q = &qs[n * 32..n * 32 + 32];
            for j in 0..4 {
                let shift = j * 2;
                for half in 0..2 {
                    let sc = scales[is];
                    is += 1;
                    let dl = d * f32::from(sc & 0xF);
                    let ml = dmin * f32::from(sc >> 4);
                    for l in 0..16 {
                        out[o + l] =
                            dl * f32::from((q[half * 16 + l] >> shift) & 3) - ml;
                    }
                    o += 16;
                }
            }
        }
    }
}

/// Q3_K: 16 sub-blocks of 16 values with 6-bit scales packed into 12 bytes, two low bits per
/// value in `qs` and the third bit in `hmask`. The high bit is INVERTED -- a clear hmask bit
/// subtracts 4 -- which is the sign convention that reads as garbage if you flip it.
fn q3_k(row: &[u8], out: &mut [f32]) {
    for (b, blk) in row.chunks_exact(110).enumerate() {
        let hm = &blk[0..QK_K / 8];
        let qs = &blk[32..32 + QK_K / 4];
        let d_all = f16_at(blk, 108);

        // The 12 scale bytes hold 16 six-bit fields: four bytes of low nibbles per group of
        // four, with the top two bits of every field living in the last four bytes.
        // Only THREE words are read: the 12 scale bytes. aux[3] is produced by the
        // shuffle below (llama.cpp memcpy's 12 bytes into a 4-word array and overwrites
        // the fourth), so reading a fourth word here would run off the block.
        let mut aux = [0_u32; 4];
        for (i, a) in aux.iter_mut().take(3).enumerate() {
            let j = 96 + i * 4;
            *a = u32::from_le_bytes([blk[j], blk[j + 1], blk[j + 2], blk[j + 3]]);
        }
        let (kmask1, kmask2) = (0x0303_0303_u32, 0x0f0f_0f0f_u32);
        let tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | ((tmp & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        let scale = |i: usize| aux[i / 4].to_le_bytes()[i % 4] as i8;

        let mut o = b * QK_K;
        let mut is = 0;
        let mut m = 1_u8;
        for n in 0..2 {
            let q = &qs[n * 32..n * 32 + 32];
            for j in 0..4 {
                let shift = j * 2;
                for half in 0..2 {
                    let dl = d_all * f32::from(scale(is) - 32);
                    is += 1;
                    for l in 0..16 {
                        let i = half * 16 + l;
                        let hi = if hm[i] & m != 0 { 0.0 } else { 4.0 };
                        out[o + l] = dl * (f32::from((q[i] >> shift) & 3) - hi);
                    }
                    o += 16;
                }
                m <<= 1;
            }
        }
    }
}

/// Q4_K / Q5_K are ASYMMETRIC: value = d*sc*q - dmin*m, unlike Q4_0's symmetric (q-8)*d.
fn q4_k(row: &[u8], out: &mut [f32]) {
    for (b, blk) in row.chunks_exact(144).enumerate() {
        let (d, dmin) = (f16_at(blk, 0), f16_at(blk, 2));
        let scales = &blk[4..4 + K_SCALE_SIZE];
        let qs = &blk[16..16 + QK_K / 2];
        let base = b * QK_K;
        for half in 0..4 {
            let (sc1, m1) = k_scale_min(half * 2, scales);
            let (sc2, m2) = k_scale_min(half * 2 + 1, scales);
            let (d1, off1) = (d * f32::from(sc1), dmin * f32::from(m1));
            let (d2, off2) = (d * f32::from(sc2), dmin * f32::from(m2));
            let q = &qs[half * 32..half * 32 + 32];
            let o = base + half * 64;
            for l in 0..32 {
                out[o + l] = d1 * f32::from(q[l] & 0xF) - off1;
                out[o + 32 + l] = d2 * f32::from(q[l] >> 4) - off2;
            }
        }
    }
}

fn q5_k(row: &[u8], out: &mut [f32]) {
    for (b, blk) in row.chunks_exact(176).enumerate() {
        let (d, dmin) = (f16_at(blk, 0), f16_at(blk, 2));
        let scales = &blk[4..4 + K_SCALE_SIZE];
        let qh = &blk[16..16 + QK_K / 8];
        let qs = &blk[48..48 + QK_K / 2];
        let base = b * QK_K;
        for half in 0..4 {
            let (sc1, m1) = k_scale_min(half * 2, scales);
            let (sc2, m2) = k_scale_min(half * 2 + 1, scales);
            let (d1, off1) = (d * f32::from(sc1), dmin * f32::from(m1));
            let (d2, off2) = (d * f32::from(sc2), dmin * f32::from(m2));
            // The high bit walks two bit positions per 64 values, so sub-block `half`
            // reads bits 2*half and 2*half+1 of every qh byte.
            let (u1, u2) = (1_u8 << (half * 2), 1_u8 << (half * 2 + 1));
            let q = &qs[half * 32..half * 32 + 32];
            let o = base + half * 64;
            for l in 0..32 {
                let hi1 = if qh[l] & u1 != 0 { 16.0 } else { 0.0 };
                let hi2 = if qh[l] & u2 != 0 { 16.0 } else { 0.0 };
                out[o + l] = d1 * (f32::from(q[l] & 0xF) + hi1) - off1;
                out[o + 32 + l] = d2 * (f32::from(q[l] >> 4) + hi2) - off2;
            }
        }
    }
}

/// Q6_K is symmetric around 32 with SIGNED 8-bit sub-scales -- no min at all.
fn q6_k(row: &[u8], out: &mut [f32]) {
    for (b, blk) in row.chunks_exact(210).enumerate() {
        let ql = &blk[0..QK_K / 2];
        let qh = &blk[QK_K / 2..QK_K / 2 + QK_K / 4];
        let sc = &blk[192..192 + QK_K / 16];
        let d = f16_at(blk, 208);
        let base = b * QK_K;
        for n in 0..2 {
            let (ql, qh, sc) = (&ql[n * 64..], &qh[n * 32..], &sc[n * 8..]);
            let o = base + n * 128;
            for l in 0..32 {
                let is = l / 16;
                let q1 = i32::from((ql[l] & 0xF) | ((qh[l] & 3) << 4)) - 32;
                let q2 = i32::from((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                let q3 = i32::from((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                let q4 = i32::from((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                let s = |i: usize| f32::from(sc[i] as i8);
                out[o + l] = d * s(is) * q1 as f32;
                out[o + 32 + l] = d * s(is + 2) * q2 as f32;
                out[o + 64 + l] = d * s(is + 4) * q3 as f32;
                out[o + 96 + l] = d * s(is + 6) * q4 as f32;
            }
        }
    }
}

// ---- IQ: the bits are an INDEX, not a number ----------------------------------------

/// The 16 non-linear levels IQ4_NL and IQ4_XS quantise to. A 4-bit field is a position in
/// this table, so the values are not evenly spaced and no arithmetic reproduces them.
const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

fn iq4_nl(row: &[u8], out: &mut [f32]) {
    for (b, blk) in row.chunks_exact(18).enumerate() {
        let d = f16_at(blk, 0);
        let o = b * QK;
        for i in 0..QK / 2 {
            let byte = blk[2 + i];
            out[o + i] = d * f32::from(KVALUES_IQ4NL[(byte & 0xF) as usize]);
            out[o + i + QK / 2] = d * f32::from(KVALUES_IQ4NL[(byte >> 4) as usize]);
        }
    }
}

/// IQ4_XS is IQ4_NL's codebook with a k-quant's super-block: eight 6-bit sub-block scales
/// split across a nibble in `scales_l` and two bits in `scales_h`, biased by 32.
fn iq4_xs(row: &[u8], out: &mut [f32]) {
    for (b, blk) in row.chunks_exact(136).enumerate() {
        let d = f16_at(blk, 0);
        let scales_h = u16::from_le_bytes([blk[2], blk[3]]);
        let scales_l = &blk[4..8];
        let qs = &blk[8..8 + QK_K / 2];
        let base = b * QK_K;
        for ib in 0..QK_K / 32 {
            let ls = usize::from((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF)
                | ((usize::from(scales_h >> (2 * ib)) & 3) << 4);
            let dl = d * (ls as f32 - 32.0);
            let q = &qs[ib * 16..ib * 16 + 16];
            let o = base + ib * 32;
            for j in 0..16 {
                out[o + j] = dl * f32::from(KVALUES_IQ4NL[(q[j] & 0xF) as usize]);
                out[o + j + 16] = dl * f32::from(KVALUES_IQ4NL[(q[j] >> 4) as usize]);
            }
        }
    }
}

/// IQ3_S: every 8 values are ONE grid entry -- a 9-bit index (8 bits in `qs`, the ninth in
/// `qh`) selecting four bytes of the codebook, twice, with a separate sign bit per value in
/// `signs` and an odd scale (1 + 2*s) per 32 values. Three packings in one block, which is
/// why it is the easiest of these to get subtly wrong.
fn iq3_s(row: &[u8], out: &mut [f32]) {
    for (b, blk) in row.chunks_exact(110).enumerate() {
        let d = f16_at(blk, 0);
        let qs = &blk[2..2 + QK_K / 4];
        let qh = &blk[66..66 + QK_K / 32];
        let signs = &blk[74..74 + QK_K / 8];
        let scales = &blk[106..106 + QK_K / 64];
        let mut o = b * QK_K;
        for ib32 in (0..QK_K / 32).step_by(2) {
            let sc = scales[ib32 / 2];
            let db = [
                d * (1.0 + 2.0 * f32::from(sc & 0xF)),
                d * (1.0 + 2.0 * f32::from(sc >> 4)),
            ];
            for (h, &dbh) in db.iter().enumerate() {
                let qs = &qs[(ib32 + h) * 8..(ib32 + h) * 8 + 8];
                let sg = &signs[(ib32 + h) * 4..(ib32 + h) * 4 + 4];
                let qhb = u32::from(qh[ib32 + h]);
                for l in 0..4 {
                    let g1 = IQ3S_GRID[usize::from(qs[2 * l])
                        | ((qhb << (8 - 2 * l)) & 256) as usize];
                    let g2 = IQ3S_GRID[usize::from(qs[2 * l + 1])
                        | ((qhb << (7 - 2 * l)) & 256) as usize];
                    let (g1, g2) = (g1.to_le_bytes(), g2.to_le_bytes());
                    for j in 0..4 {
                        let s1 = if sg[l] & (1 << j) != 0 { -1.0 } else { 1.0 };
                        let s2 = if sg[l] & (1 << (j + 4)) != 0 {
                            -1.0
                        } else {
                            1.0
                        };
                        out[o + j] = dbh * f32::from(g1[j]) * s1;
                        out[o + j + 4] = dbh * f32::from(g2[j]) * s2;
                    }
                    o += 8;
                }
            }
        }
    }
}

/// IQ3_S's codebook: 512 entries of four 3-bit-ish levels packed one per byte.
#[rustfmt::skip]
#[allow(clippy::unreadable_literal)]
const IQ3S_GRID: [u32; 512] = [
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
];

/// The sign table the IQ2/IQ3 codebook formats share. llama.cpp ships it as a written
/// `ksigns_iq2xs[128]`; it is one expression -- the eighth sign is the PARITY of the other
/// seven -- so it is derived here. A 128-entry copy is 128 chances to mistype a number that
/// would read as a plausible weight, not as a failure.
const fn ksign(i: u8) -> u8 {
    i | (((i.count_ones() as u8) & 1) << 7)
}

/// IQ2_XS: eight values per codebook entry. Each `u16` in `qs` carries a 9-bit grid index
/// in its low bits and a 7-bit SIGN index in its top bits; one 4-bit sub-scale per 16
/// values, read as `(0.5 + s) * 0.25` -- the half-step bias is part of the format.
fn iq2_xs(row: &[u8], out: &mut [f32]) {
    for (b, blk) in row.chunks_exact(74).enumerate() {
        let d = f16_at(blk, 0);
        let qs = &blk[2..2 + QK_K / 4];
        let scales = &blk[2 + QK_K / 4..2 + QK_K / 4 + QK_K / 32];
        let base = b * QK_K;
        for (ib32, &sc) in scales.iter().enumerate() {
            let db = [
                d * (0.5 + f32::from(sc & 0xF)) * 0.25,
                d * (0.5 + f32::from(sc >> 4)) * 0.25,
            ];
            for l in 0..4 {
                let at = ib32 * 8 + l * 2;
                let q = u16::from_le_bytes([qs[at], qs[at + 1]]);
                let g = IQ2XS_GRID[usize::from(q & 511)].to_le_bytes();
                let signs = ksign((q >> 9) as u8);
                let o = base + ib32 * 32 + l * 8;
                for j in 0..8 {
                    let s = if signs & (1 << j) != 0 { -1.0 } else { 1.0 };
                    out[o + j] = db[l / 2] * f32::from(g[j]) * s;
                }
            }
        }
    }
}

/// IQ2_XXS: the leanest codebook. Every 32 values are two `u32` words -- four 8-bit grid
/// indices (eight levels each), then four 7-bit sign indices with the 4-bit sub-scale in
/// the top nibble, read as `(0.5 + s) * 0.25` like IQ2_XS. One span: d, then the words.
fn iq2_xxs(row: &[u8], out: &mut [f32]) {
    for (b, blk) in row.chunks_exact(66).enumerate() {
        let d = f16_at(blk, 0);
        let qs = &blk[2..66];
        let base = b * QK_K;
        for ib32 in 0..QK_K / 32 {
            let at = ib32 * 8;
            let w0 = u32::from_le_bytes([qs[at], qs[at + 1], qs[at + 2], qs[at + 3]]);
            let w1 =
                u32::from_le_bytes([qs[at + 4], qs[at + 5], qs[at + 6], qs[at + 7]]);
            let db = d * (0.5 + (w1 >> 28) as f32) * 0.25;
            for l in 0..4 {
                let g = IQ2XXS_GRID[((w0 >> (8 * l)) & 0xFF) as usize].to_le_bytes();
                let signs = ksign(((w1 >> (7 * l)) & 127) as u8);
                let o = base + ib32 * 32 + l * 8;
                for j in 0..8 {
                    let s = if signs & (1 << j) != 0 { -1.0 } else { 1.0 };
                    out[o + j] = db * f32::from(g[j]) * s;
                }
            }
        }
    }
}

/// IQ1_S: a 1.56-bit codebook with an offset. Every 32 values are four 11-bit grid indices
/// (eight bits in `qs`, three in the sub-block's `qh` word) selecting eight levels in
/// {-1, 0, 1}; the same word holds an odd 3-bit scale `2s + 1` and the sign of a +-0.125
/// delta added to every level before the scale.
fn iq1_s(row: &[u8], out: &mut [f32]) {
    for (b, blk) in row.chunks_exact(50).enumerate() {
        let d = f16_at(blk, 0);
        let base = b * QK_K;
        for ib32 in 0..QK_K / 32 {
            let qh =
                u32::from(u16::from_le_bytes([blk[34 + 2 * ib32], blk[35 + 2 * ib32]]));
            let dl = d * (2 * ((qh >> 12) & 7) + 1) as f32;
            let delta = if qh & 0x8000 != 0 {
                -IQ1S_DELTA
            } else {
                IQ1S_DELTA
            };
            for l in 0..4 {
                let idx = usize::from(blk[2 + 4 * ib32 + l])
                    | ((((qh >> (3 * l)) & 7) as usize) << 8);
                let g = IQ1S_GRID[idx];
                let o = base + ib32 * 32 + l * 8;
                for j in 0..8 {
                    out[o + j] =
                        dl * (f32::from(((g >> (2 * j)) & 3) as u8) - 1.0 + delta);
                }
            }
        }
    }
}

/// IQ1_M: IQ1_S's codebook with two scales per 32 values (one per half) and a delta sign
/// per 8. The super-block scale has no field of its own: its sixteen bits are the top
/// nibbles of the four scale words that close the block, whose other twelve bits hold the
/// 3-bit scales of sub-blocks 2k and 2k + 1. Each `qh` byte carries two indices' top three
/// bits and their delta signs.
fn iq1_m(row: &[u8], out: &mut [f32]) {
    for (b, blk) in row.chunks_exact(56).enumerate() {
        let sc = |k: usize| {
            u32::from(u16::from_le_bytes([blk[48 + 2 * k], blk[49 + 2 * k]]))
        };
        let packed = (sc(0) >> 12)
            | ((sc(1) >> 8) & 0x00F0)
            | ((sc(2) >> 4) & 0x0F00)
            | (sc(3) & 0xF000);
        let d = f16_to_f32(packed as u16);
        let base = b * QK_K;
        for ib32 in 0..QK_K / 32 {
            let s = sc(ib32 / 2);
            let sh = 6 * (ib32 % 2);
            let dl = [
                d * (2 * ((s >> sh) & 7) + 1) as f32,
                d * (2 * ((s >> (sh + 3)) & 7) + 1) as f32,
            ];
            for l in 0..4 {
                let hb = u32::from(blk[32 + 2 * ib32 + l / 2]) >> (4 * (l % 2));
                let idx = usize::from(blk[4 * ib32 + l]) | (((hb & 7) as usize) << 8);
                let g = IQ1S_GRID[idx];
                let delta = if hb & 8 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA };
                let o = base + ib32 * 32 + l * 8;
                for j in 0..8 {
                    out[o + j] = dl[l / 2]
                        * (f32::from(((g >> (2 * j)) & 3) as u8) - 1.0 + delta);
                }
            }
        }
    }
}

/// IQ2_S: IQ2_XS's scale rule over a 10-bit grid index -- eight bits in `qs`, the top two
/// from `qh` -- with the signs stored OUTRIGHT in the block (the second half of `qs`)
/// rather than as a parity-coded index. Same codebook shape, four packings.
fn iq2_s(row: &[u8], out: &mut [f32]) {
    for (b, blk) in row.chunks_exact(82).enumerate() {
        let d = f16_at(blk, 0);
        let qs = &blk[2..2 + QK_K / 8];
        let signs = &blk[2 + QK_K / 8..2 + QK_K / 4];
        let qh = &blk[2 + QK_K / 4..2 + QK_K / 4 + QK_K / 32];
        let scales = &blk[2 + QK_K / 4 + QK_K / 32..];
        let base = b * QK_K;
        for ib32 in 0..QK_K / 32 {
            let db = [
                d * (0.5 + f32::from(scales[ib32] & 0xF)) * 0.25,
                d * (0.5 + f32::from(scales[ib32] >> 4)) * 0.25,
            ];
            for l in 0..4 {
                let idx = usize::from(qs[ib32 * 4 + l])
                    | ((usize::from(qh[ib32]) << (8 - 2 * l)) & 0x300);
                let g = IQ2S_GRID[idx].to_le_bytes();
                let sg = signs[ib32 * 4 + l];
                let o = base + ib32 * 32 + l * 8;
                for j in 0..8 {
                    let s = if sg & (1 << j) != 0 { -1.0 } else { 1.0 };
                    out[o + j] = db[l / 2] * f32::from(g[j]) * s;
                }
            }
        }
    }
}

/// IQ3_XXS: FOUR values per codebook entry, two entries per eight values, and one 32-bit
/// word per 32 values holding four 7-bit sign indices and, in its top nibble, the
/// sub-block scale `(0.5 + s) * 0.5`. Scale and signs share a word, so they travel
/// together wherever the block's bytes are moved.
fn iq3_xxs(row: &[u8], out: &mut [f32]) {
    for (b, blk) in row.chunks_exact(98).enumerate() {
        let d = f16_at(blk, 0);
        let qs = &blk[2..2 + QK_K / 4];
        let sas = &blk[2 + QK_K / 4..2 + 3 * (QK_K / 8)];
        let base = b * QK_K;
        for ib32 in 0..QK_K / 32 {
            let w = ib32 * 4;
            let aux32 =
                u32::from_le_bytes([sas[w], sas[w + 1], sas[w + 2], sas[w + 3]]);
            let db = d * (0.5 + (aux32 >> 28) as f32) * 0.5;
            for l in 0..4 {
                let signs = ksign(((aux32 >> (7 * l)) & 127) as u8);
                let at = ib32 * 8 + l * 2;
                let g1 = IQ3XXS_GRID[usize::from(qs[at])].to_le_bytes();
                let g2 = IQ3XXS_GRID[usize::from(qs[at + 1])].to_le_bytes();
                let o = base + ib32 * 32 + l * 8;
                for j in 0..4 {
                    let s1 = if signs & (1 << j) != 0 { -1.0 } else { 1.0 };
                    let s2 = if signs & (1 << (j + 4)) != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    out[o + j] = db * f32::from(g1[j]) * s1;
                    out[o + j + 4] = db * f32::from(g2[j]) * s2;
                }
            }
        }
    }
}

/// IQ2_XXS's codebook: 256 entries of eight packed byte levels (ggml-common.h iq2xxs_grid).
#[allow(clippy::unreadable_literal)]
const IQ2XXS_GRID: [u64; 256] = [
    0x0808080808080808,
    0x080808080808082b,
    0x0808080808081919,
    0x0808080808082b08,
    0x0808080808082b2b,
    0x0808080808190819,
    0x0808080808191908,
    0x08080808082b0808,
    0x08080808082b082b,
    0x08080808082b2b08,
    0x08080808082b2b2b,
    0x0808080819080819,
    0x0808080819081908,
    0x0808080819190808,
    0x0808080819192b08,
    0x08080808192b0819,
    0x08080808192b1908,
    0x080808082b080808,
    0x080808082b08082b,
    0x080808082b082b2b,
    0x080808082b2b082b,
    0x0808081908080819,
    0x0808081908081908,
    0x0808081908190808,
    0x0808081908191919,
    0x0808081919080808,
    0x080808192b081908,
    0x080808192b192b08,
    0x0808082b08080808,
    0x0808082b0808082b,
    0x0808082b082b082b,
    0x0808082b2b08082b,
    0x0808190808080819,
    0x0808190808081908,
    0x0808190808190808,
    0x08081908082b0819,
    0x08081908082b1908,
    0x0808190819080808,
    0x080819081908082b,
    0x0808190819082b08,
    0x08081908192b0808,
    0x080819082b080819,
    0x080819082b081908,
    0x080819082b190808,
    0x080819082b2b1908,
    0x0808191908080808,
    0x080819190808082b,
    0x0808191908082b08,
    0x08081919082b0808,
    0x080819191908192b,
    0x08081919192b2b19,
    0x080819192b080808,
    0x080819192b190819,
    0x0808192b08082b19,
    0x0808192b08190808,
    0x0808192b19080808,
    0x0808192b2b081908,
    0x0808192b2b2b1908,
    0x08082b0808080808,
    0x08082b0808081919,
    0x08082b0808082b08,
    0x08082b0808191908,
    0x08082b08082b2b08,
    0x08082b0819080819,
    0x08082b0819081908,
    0x08082b0819190808,
    0x08082b081919082b,
    0x08082b082b082b08,
    0x08082b1908081908,
    0x08082b1919080808,
    0x08082b2b0808082b,
    0x08082b2b08191908,
    0x0819080808080819,
    0x0819080808081908,
    0x0819080808190808,
    0x08190808082b0819,
    0x0819080819080808,
    0x08190808192b0808,
    0x081908082b081908,
    0x081908082b190808,
    0x081908082b191919,
    0x0819081908080808,
    0x0819081908082b08,
    0x08190819082b0808,
    0x0819081919190808,
    0x0819081919192b2b,
    0x081908192b080808,
    0x0819082b082b1908,
    0x0819082b19081919,
    0x0819190808080808,
    0x0819190808082b08,
    0x08191908082b0808,
    0x08191908082b1919,
    0x0819190819082b19,
    0x081919082b080808,
    0x0819191908192b08,
    0x08191919192b082b,
    0x0819192b08080808,
    0x0819192b0819192b,
    0x08192b0808080819,
    0x08192b0808081908,
    0x08192b0808190808,
    0x08192b0819080808,
    0x08192b082b080819,
    0x08192b1908080808,
    0x08192b1908081919,
    0x08192b192b2b0808,
    0x08192b2b19190819,
    0x082b080808080808,
    0x082b08080808082b,
    0x082b080808082b2b,
    0x082b080819081908,
    0x082b0808192b0819,
    0x082b08082b080808,
    0x082b08082b08082b,
    0x082b0819082b2b19,
    0x082b081919082b08,
    0x082b082b08080808,
    0x082b082b0808082b,
    0x082b190808080819,
    0x082b190808081908,
    0x082b190808190808,
    0x082b190819080808,
    0x082b19081919192b,
    0x082b191908080808,
    0x082b191919080819,
    0x082b1919192b1908,
    0x082b192b2b190808,
    0x082b2b0808082b08,
    0x082b2b08082b0808,
    0x082b2b082b191908,
    0x082b2b2b19081908,
    0x1908080808080819,
    0x1908080808081908,
    0x1908080808190808,
    0x1908080808192b08,
    0x19080808082b0819,
    0x19080808082b1908,
    0x1908080819080808,
    0x1908080819082b08,
    0x190808081919192b,
    0x19080808192b0808,
    0x190808082b080819,
    0x190808082b081908,
    0x190808082b190808,
    0x1908081908080808,
    0x19080819082b0808,
    0x19080819192b0819,
    0x190808192b080808,
    0x190808192b081919,
    0x1908082b08080819,
    0x1908082b08190808,
    0x1908082b19082b08,
    0x1908082b1919192b,
    0x1908082b192b2b08,
    0x1908190808080808,
    0x1908190808082b08,
    0x19081908082b0808,
    0x190819082b080808,
    0x190819082b192b19,
    0x190819190819082b,
    0x19081919082b1908,
    0x1908192b08080808,
    0x19082b0808080819,
    0x19082b0808081908,
    0x19082b0808190808,
    0x19082b0819080808,
    0x19082b0819081919,
    0x19082b1908080808,
    0x19082b1919192b08,
    0x19082b19192b0819,
    0x19082b192b08082b,
    0x19082b2b19081919,
    0x19082b2b2b190808,
    0x1919080808080808,
    0x1919080808082b08,
    0x1919080808190819,
    0x1919080808192b19,
    0x19190808082b0808,
    0x191908082b080808,
    0x191908082b082b08,
    0x1919081908081908,
    0x191908191908082b,
    0x191908192b2b1908,
    0x1919082b2b190819,
    0x191919082b190808,
    0x191919082b19082b,
    0x1919191908082b2b,
    0x1919192b08080819,
    0x1919192b19191908,
    0x19192b0808080808,
    0x19192b0808190819,
    0x19192b0808192b19,
    0x19192b08192b1908,
    0x19192b1919080808,
    0x19192b2b08082b08,
    0x192b080808081908,
    0x192b080808190808,
    0x192b080819080808,
    0x192b0808192b2b08,
    0x192b081908080808,
    0x192b081919191919,
    0x192b082b08192b08,
    0x192b082b192b0808,
    0x192b190808080808,
    0x192b190808081919,
    0x192b191908190808,
    0x192b19190819082b,
    0x192b19192b081908,
    0x192b2b081908082b,
    0x2b08080808080808,
    0x2b0808080808082b,
    0x2b08080808082b2b,
    0x2b08080819080819,
    0x2b0808082b08082b,
    0x2b08081908081908,
    0x2b08081908192b08,
    0x2b08081919080808,
    0x2b08082b08190819,
    0x2b08190808080819,
    0x2b08190808081908,
    0x2b08190808190808,
    0x2b08190808191919,
    0x2b08190819080808,
    0x2b081908192b0808,
    0x2b08191908080808,
    0x2b0819191908192b,
    0x2b0819192b191908,
    0x2b08192b08082b19,
    0x2b08192b19080808,
    0x2b08192b192b0808,
    0x2b082b080808082b,
    0x2b082b1908081908,
    0x2b082b2b08190819,
    0x2b19080808081908,
    0x2b19080808190808,
    0x2b190808082b1908,
    0x2b19080819080808,
    0x2b1908082b2b0819,
    0x2b1908190819192b,
    0x2b1908192b080808,
    0x2b19082b19081919,
    0x2b19190808080808,
    0x2b191908082b082b,
    0x2b19190819081908,
    0x2b19191919190819,
    0x2b192b082b080819,
    0x2b192b19082b0808,
    0x2b2b08080808082b,
    0x2b2b080819190808,
    0x2b2b08082b081919,
    0x2b2b081908082b19,
    0x2b2b082b08080808,
    0x2b2b190808192b08,
    0x2b2b2b0819190808,
    0x2b2b2b1908081908,
];

/// IQ2_XS's codebook: 512 entries of eight packed byte levels.
#[rustfmt::skip]
#[allow(clippy::unreadable_literal)]
const IQ2XS_GRID: [u64; 512] = [
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x080808080819192b,
    0x0808080808192b19, 0x08080808082b0808, 0x08080808082b082b, 0x08080808082b1919,
    0x08080808082b2b08, 0x0808080819080819, 0x0808080819081908, 0x080808081908192b,
    0x0808080819082b19, 0x0808080819190808, 0x080808081919082b, 0x0808080819191919,
    0x0808080819192b08, 0x08080808192b0819, 0x08080808192b1908, 0x080808082b080808,
    0x080808082b08082b, 0x080808082b081919, 0x080808082b082b08, 0x080808082b190819,
    0x080808082b191908, 0x080808082b192b19, 0x080808082b2b0808, 0x0808081908080819,
    0x0808081908081908, 0x080808190808192b, 0x0808081908082b19, 0x0808081908190808,
    0x080808190819082b, 0x0808081908191919, 0x0808081908192b08, 0x0808081908192b2b,
    0x08080819082b0819, 0x08080819082b1908, 0x0808081919080808, 0x080808191908082b,
    0x0808081919081919, 0x0808081919082b08, 0x0808081919190819, 0x0808081919191908,
    0x08080819192b0808, 0x08080819192b2b08, 0x080808192b080819, 0x080808192b081908,
    0x080808192b190808, 0x0808082b08080808, 0x0808082b0808082b, 0x0808082b08081919,
    0x0808082b08082b08, 0x0808082b08190819, 0x0808082b08191908, 0x0808082b082b0808,
    0x0808082b19080819, 0x0808082b19081908, 0x0808082b19190808, 0x0808082b19191919,
    0x0808082b2b080808, 0x0808082b2b082b2b, 0x0808190808080819, 0x0808190808081908,
    0x080819080808192b, 0x0808190808082b19, 0x0808190808190808, 0x080819080819082b,
    0x0808190808191919, 0x0808190808192b08, 0x08081908082b0819, 0x08081908082b1908,
    0x0808190819080808, 0x080819081908082b, 0x0808190819081919, 0x0808190819082b08,
    0x0808190819190819, 0x0808190819191908, 0x080819081919192b, 0x08081908192b0808,
    0x080819082b080819, 0x080819082b081908, 0x080819082b190808, 0x0808191908080808,
    0x080819190808082b, 0x0808191908081919, 0x0808191908082b08, 0x0808191908190819,
    0x0808191908191908, 0x08081919082b0808, 0x0808191919080819, 0x0808191919081908,
    0x0808191919190808, 0x08081919192b0819, 0x080819192b080808, 0x0808192b08080819,
    0x0808192b08081908, 0x0808192b08190808, 0x0808192b082b192b, 0x0808192b19080808,
    0x0808192b1908082b, 0x0808192b2b081908, 0x08082b0808080808, 0x08082b080808082b,
    0x08082b0808081919, 0x08082b0808082b08, 0x08082b0808082b2b, 0x08082b0808190819,
    0x08082b0808191908, 0x08082b08082b0808, 0x08082b08082b1919, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b0819192b08, 0x08082b082b080808,
    0x08082b082b2b0808, 0x08082b082b2b2b2b, 0x08082b1908080819, 0x08082b1908081908,
    0x08082b1908190808, 0x08082b1919080808, 0x08082b192b080819, 0x08082b192b082b19,
    0x08082b2b08080808, 0x08082b2b082b0808, 0x08082b2b082b2b08, 0x08082b2b2b19192b,
    0x08082b2b2b2b0808, 0x0819080808080819, 0x0819080808081908, 0x081908080808192b,
    0x0819080808082b19, 0x0819080808190808, 0x081908080819082b, 0x0819080808191919,
    0x0819080808192b08, 0x08190808082b0819, 0x08190808082b1908, 0x0819080819080808,
    0x081908081908082b, 0x0819080819081919, 0x0819080819082b08, 0x0819080819190819,
    0x0819080819191908, 0x08190808192b0808, 0x08190808192b2b2b, 0x081908082b080819,
    0x081908082b081908, 0x081908082b190808, 0x0819081908080808, 0x081908190808082b,
    0x0819081908081919, 0x0819081908082b08, 0x0819081908190819, 0x0819081908191908,
    0x08190819082b0808, 0x0819081919080819, 0x0819081919081908, 0x0819081919190808,
    0x081908192b080808, 0x081908192b191908, 0x081908192b19192b, 0x0819082b08080819,
    0x0819082b08081908, 0x0819082b0808192b, 0x0819082b08190808, 0x0819082b19080808,
    0x0819082b192b0808, 0x0819190808080808, 0x081919080808082b, 0x0819190808081919,
    0x0819190808082b08, 0x0819190808190819, 0x0819190808191908, 0x08191908082b0808,
    0x0819190819080819, 0x0819190819081908, 0x0819190819082b19, 0x0819190819190808,
    0x08191908192b1908, 0x081919082b080808, 0x0819191908080819, 0x0819191908081908,
    0x0819191908190808, 0x0819191919080808, 0x0819192b08080808, 0x0819192b08191908,
    0x0819192b19082b19, 0x08192b0808080819, 0x08192b0808081908, 0x08192b0808190808,
    0x08192b080819082b, 0x08192b0819080808, 0x08192b0819191908, 0x08192b082b08192b,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b19192b192b, 0x08192b2b19190819,
    0x08192b2b2b2b2b19, 0x082b080808080808, 0x082b08080808082b, 0x082b080808081919,
    0x082b080808082b08, 0x082b080808082b2b, 0x082b080808190819, 0x082b080808191908,
    0x082b0808082b0808, 0x082b080819080819, 0x082b080819081908, 0x082b080819190808,
    0x082b08082b080808, 0x082b08082b2b0808, 0x082b081908080819, 0x082b081908081908,
    0x082b081908190808, 0x082b081919080808, 0x082b081919082b08, 0x082b0819192b1919,
    0x082b082b08080808, 0x082b082b082b082b, 0x082b082b2b080808, 0x082b082b2b2b2b08,
    0x082b190808080819, 0x082b190808081908, 0x082b190808190808, 0x082b1908082b2b19,
    0x082b190819080808, 0x082b191908080808, 0x082b191919080819, 0x082b19191919082b,
    0x082b19192b192b19, 0x082b192b08080819, 0x082b192b08192b2b, 0x082b192b2b2b192b,
    0x082b2b0808080808, 0x082b2b0808082b08, 0x082b2b0808082b2b, 0x082b2b08082b0808,
    0x082b2b0819191919, 0x082b2b082b082b08, 0x082b2b082b2b082b, 0x082b2b19192b2b08,
    0x082b2b192b190808, 0x082b2b2b08082b08, 0x082b2b2b082b0808, 0x082b2b2b2b08082b,
    0x082b2b2b2b082b08, 0x082b2b2b2b082b2b, 0x1908080808080819, 0x1908080808081908,
    0x190808080808192b, 0x1908080808082b19, 0x1908080808190808, 0x190808080819082b,
    0x1908080808191919, 0x1908080808192b08, 0x19080808082b0819, 0x19080808082b1908,
    0x1908080819080808, 0x190808081908082b, 0x1908080819081919, 0x1908080819082b08,
    0x1908080819082b2b, 0x1908080819190819, 0x1908080819191908, 0x19080808192b0808,
    0x19080808192b1919, 0x190808082b080819, 0x190808082b081908, 0x190808082b190808,
    0x1908081908080808, 0x190808190808082b, 0x1908081908081919, 0x1908081908082b08,
    0x1908081908190819, 0x1908081908191908, 0x19080819082b0808, 0x1908081919080819,
    0x1908081919081908, 0x1908081919190808, 0x190808192b080808, 0x190808192b081919,
    0x190808192b2b082b, 0x1908082b08080819, 0x1908082b08081908, 0x1908082b08190808,
    0x1908082b0819082b, 0x1908082b082b2b19, 0x1908082b19080808, 0x1908190808080808,
    0x190819080808082b, 0x1908190808081919, 0x1908190808082b08, 0x1908190808190819,
    0x1908190808191908, 0x1908190808192b19, 0x19081908082b0808, 0x1908190819080819,
    0x1908190819081908, 0x1908190819190808, 0x190819082b080808, 0x190819082b191908,
    0x1908191908080819, 0x1908191908081908, 0x1908191908190808, 0x19081919082b1908,
    0x1908191919080808, 0x190819192b192b2b, 0x1908192b08080808, 0x1908192b08082b2b,
    0x1908192b19081908, 0x1908192b19190808, 0x19082b0808080819, 0x19082b0808081908,
    0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919, 0x19082b0819191908,
    0x19082b08192b082b, 0x19082b1908080808, 0x19082b1908190819, 0x19082b1919081908,
    0x19082b1919190808, 0x19082b19192b2b19, 0x19082b2b08081908, 0x1919080808080808,
    0x191908080808082b, 0x1919080808081919, 0x1919080808082b08, 0x1919080808190819,
    0x1919080808191908, 0x19190808082b0808, 0x19190808082b2b08, 0x1919080819080819,
    0x1919080819081908, 0x1919080819190808, 0x191908082b080808, 0x1919081908080819,
    0x1919081908081908, 0x1919081908190808, 0x1919081908191919, 0x1919081919080808,
    0x191908191908082b, 0x1919082b08080808, 0x1919082b19081908, 0x1919082b2b2b2b2b,
    0x1919190808080819, 0x1919190808081908, 0x1919190808190808, 0x19191908082b0819,
    0x1919190819080808, 0x19191908192b0808, 0x191919082b080819, 0x191919082b2b0819,
    0x1919191908080808, 0x1919191908082b08, 0x191919192b080808, 0x191919192b082b08,
    0x1919192b082b0819, 0x1919192b192b2b08, 0x1919192b2b2b0819, 0x19192b0808080808,
    0x19192b0808191908, 0x19192b0819080819, 0x19192b0819190808, 0x19192b082b192b19,
    0x19192b1908192b2b, 0x19192b1919080808, 0x19192b191908082b, 0x19192b2b2b081919,
    0x192b080808080819, 0x192b080808081908, 0x192b080808190808, 0x192b080819080808,
    0x192b080819191908, 0x192b0808192b082b, 0x192b08082b08192b, 0x192b08082b2b2b19,
    0x192b081908080808, 0x192b082b082b1908, 0x192b082b19082b2b, 0x192b082b2b19082b,
    0x192b190808080808, 0x192b19080819192b, 0x192b191908190808, 0x192b191919080808,
    0x192b191919081919, 0x192b19192b2b1908, 0x192b2b0808080819, 0x192b2b08192b2b2b,
    0x192b2b19082b1919, 0x192b2b2b0808192b, 0x192b2b2b19191908, 0x192b2b2b192b082b,
    0x2b08080808080808, 0x2b0808080808082b, 0x2b08080808081919, 0x2b08080808082b08,
    0x2b08080808190819, 0x2b08080808191908, 0x2b080808082b0808, 0x2b080808082b2b2b,
    0x2b08080819080819, 0x2b08080819081908, 0x2b08080819190808, 0x2b0808082b080808,
    0x2b0808082b08082b, 0x2b0808082b2b2b08, 0x2b0808082b2b2b2b, 0x2b08081908080819,
    0x2b08081908081908, 0x2b0808190808192b, 0x2b08081908190808, 0x2b08081919080808,
    0x2b08081919190819, 0x2b08081919192b19, 0x2b08082b08080808, 0x2b08082b082b0808,
    0x2b08082b2b080808, 0x2b08082b2b08082b, 0x2b08082b2b2b0808, 0x2b08082b2b2b2b08,
    0x2b08190808080819, 0x2b08190808081908, 0x2b08190808190808, 0x2b0819080819082b,
    0x2b08190808191919, 0x2b08190819080808, 0x2b081908192b0808, 0x2b0819082b082b19,
    0x2b08191908080808, 0x2b08191919081908, 0x2b0819192b2b1919, 0x2b08192b08192b08,
    0x2b08192b192b2b2b, 0x2b082b0808080808, 0x2b082b0808082b08, 0x2b082b08082b1919,
    0x2b082b0819192b2b, 0x2b082b082b080808, 0x2b082b082b08082b, 0x2b082b082b2b2b08,
    0x2b082b190808192b, 0x2b082b2b082b082b, 0x2b082b2b2b080808, 0x2b082b2b2b082b08,
    0x2b082b2b2b19192b, 0x2b082b2b2b2b2b08, 0x2b19080808080819, 0x2b19080808081908,
    0x2b19080808190808, 0x2b19080819080808, 0x2b1908081919192b, 0x2b1908082b081908,
    0x2b19081908080808, 0x2b190819082b082b, 0x2b190819192b1908, 0x2b19082b1919192b,
    0x2b19082b2b082b19, 0x2b19190808080808, 0x2b19190808081919, 0x2b19190819081908,
    0x2b19190819190808, 0x2b19190819192b08, 0x2b191919082b2b19, 0x2b1919192b190808,
    0x2b1919192b19082b, 0x2b19192b19080819, 0x2b192b0819190819, 0x2b192b082b2b192b,
    0x2b192b1919082b19, 0x2b192b2b08191919, 0x2b192b2b192b0808, 0x2b2b080808080808,
    0x2b2b08080808082b, 0x2b2b080808082b08, 0x2b2b080808082b2b, 0x2b2b0808082b0808,
    0x2b2b0808082b2b2b, 0x2b2b08082b2b0808, 0x2b2b081919190819, 0x2b2b081919192b19,
    0x2b2b08192b2b192b, 0x2b2b082b08080808, 0x2b2b082b0808082b, 0x2b2b082b08082b08,
    0x2b2b082b082b2b2b, 0x2b2b082b2b080808, 0x2b2b082b2b2b0808, 0x2b2b190819080808,
    0x2b2b19082b191919, 0x2b2b192b192b1919, 0x2b2b192b2b192b08, 0x2b2b2b0808082b2b,
    0x2b2b2b08082b0808, 0x2b2b2b08082b082b, 0x2b2b2b08082b2b08, 0x2b2b2b082b2b0808,
    0x2b2b2b082b2b2b08, 0x2b2b2b1908081908, 0x2b2b2b192b081908, 0x2b2b2b192b08192b,
    0x2b2b2b2b082b2b08, 0x2b2b2b2b082b2b2b, 0x2b2b2b2b2b190819, 0x2b2b2b2b2b2b2b2b,
];

/// IQ2_S's codebook: 1024 entries of eight packed byte levels (a 10-bit index).
#[rustfmt::skip]
#[allow(clippy::unreadable_literal)]
const IQ2S_GRID: [u64; 1024] = [
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x080808080819192b,
    0x0808080808192b19, 0x08080808082b0808, 0x08080808082b082b, 0x08080808082b1919,
    0x08080808082b2b08, 0x0808080819080819, 0x0808080819081908, 0x080808081908192b,
    0x0808080819082b19, 0x0808080819190808, 0x080808081919082b, 0x0808080819191919,
    0x0808080819192b08, 0x08080808192b0819, 0x08080808192b1908, 0x08080808192b192b,
    0x08080808192b2b19, 0x080808082b080808, 0x080808082b08082b, 0x080808082b081919,
    0x080808082b082b08, 0x080808082b190819, 0x080808082b191908, 0x080808082b2b0808,
    0x080808082b2b1919, 0x080808082b2b2b2b, 0x0808081908080819, 0x0808081908081908,
    0x080808190808192b, 0x0808081908082b19, 0x0808081908190808, 0x080808190819082b,
    0x0808081908191919, 0x0808081908192b08, 0x08080819082b0819, 0x08080819082b1908,
    0x0808081919080808, 0x080808191908082b, 0x0808081919081919, 0x0808081919082b08,
    0x0808081919190819, 0x0808081919191908, 0x080808191919192b, 0x0808081919192b19,
    0x08080819192b0808, 0x08080819192b1919, 0x08080819192b2b08, 0x080808192b080819,
    0x080808192b081908, 0x080808192b190808, 0x080808192b19082b, 0x080808192b191919,
    0x080808192b2b0819, 0x080808192b2b1908, 0x0808082b08080808, 0x0808082b0808082b,
    0x0808082b08081919, 0x0808082b08082b08, 0x0808082b08190819, 0x0808082b08191908,
    0x0808082b082b0808, 0x0808082b082b2b2b, 0x0808082b19080819, 0x0808082b19081908,
    0x0808082b1908192b, 0x0808082b19082b19, 0x0808082b19190808, 0x0808082b19191919,
    0x0808082b2b080808, 0x0808082b2b081919, 0x0808082b2b082b2b, 0x0808082b2b191908,
    0x0808082b2b2b082b, 0x0808190808080819, 0x0808190808081908, 0x080819080808192b,
    0x0808190808082b19, 0x0808190808190808, 0x080819080819082b, 0x0808190808191919,
    0x0808190808192b08, 0x08081908082b0819, 0x08081908082b1908, 0x08081908082b192b,
    0x08081908082b2b19, 0x0808190819080808, 0x080819081908082b, 0x0808190819081919,
    0x0808190819082b08, 0x0808190819082b2b, 0x0808190819190819, 0x0808190819191908,
    0x080819081919192b, 0x0808190819192b19, 0x08081908192b0808, 0x08081908192b082b,
    0x08081908192b1919, 0x080819082b080819, 0x080819082b081908, 0x080819082b08192b,
    0x080819082b082b19, 0x080819082b190808, 0x080819082b191919, 0x080819082b192b08,
    0x080819082b2b0819, 0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b,
    0x0808191908081919, 0x0808191908082b08, 0x0808191908082b2b, 0x0808191908190819,
    0x0808191908191908, 0x080819190819192b, 0x0808191908192b19, 0x08081919082b0808,
    0x08081919082b1919, 0x08081919082b2b08, 0x0808191919080819, 0x0808191919081908,
    0x080819191908192b, 0x0808191919082b19, 0x0808191919190808, 0x080819191919082b,
    0x0808191919191919, 0x0808191919192b08, 0x08081919192b0819, 0x08081919192b1908,
    0x080819192b080808, 0x080819192b08082b, 0x080819192b081919, 0x080819192b082b08,
    0x080819192b190819, 0x080819192b191908, 0x080819192b2b0808, 0x0808192b08080819,
    0x0808192b08081908, 0x0808192b0808192b, 0x0808192b08082b19, 0x0808192b08190808,
    0x0808192b08191919, 0x0808192b19080808, 0x0808192b19081919, 0x0808192b19082b08,
    0x0808192b19190819, 0x0808192b19191908, 0x0808192b192b0808, 0x0808192b2b080819,
    0x0808192b2b081908, 0x0808192b2b190808, 0x08082b0808080808, 0x08082b080808082b,
    0x08082b0808081919, 0x08082b0808082b08, 0x08082b0808190819, 0x08082b0808191908,
    0x08082b080819192b, 0x08082b0808192b19, 0x08082b08082b0808, 0x08082b08082b1919,
    0x08082b08082b2b2b, 0x08082b0819080819, 0x08082b0819081908, 0x08082b081908192b,
    0x08082b0819082b19, 0x08082b0819190808, 0x08082b081919082b, 0x08082b0819191919,
    0x08082b0819192b08, 0x08082b08192b0819, 0x08082b08192b1908, 0x08082b082b080808,
    0x08082b082b081919, 0x08082b082b191908, 0x08082b082b2b2b2b, 0x08082b1908080819,
    0x08082b1908081908, 0x08082b1908190808, 0x08082b190819082b, 0x08082b1908191919,
    0x08082b1908192b08, 0x08082b19082b0819, 0x08082b1919080808, 0x08082b1919081919,
    0x08082b1919082b08, 0x08082b1919190819, 0x08082b1919191908, 0x08082b19192b0808,
    0x08082b192b080819, 0x08082b192b190808, 0x08082b2b08080808, 0x08082b2b08190819,
    0x08082b2b08191908, 0x08082b2b082b082b, 0x08082b2b082b2b08, 0x08082b2b082b2b2b,
    0x08082b2b19190808, 0x08082b2b2b192b19, 0x0819080808080819, 0x0819080808081908,
    0x081908080808192b, 0x0819080808082b19, 0x0819080808190808, 0x081908080819082b,
    0x0819080808191919, 0x0819080808192b08, 0x08190808082b0819, 0x08190808082b1908,
    0x08190808082b192b, 0x0819080819080808, 0x081908081908082b, 0x0819080819081919,
    0x0819080819082b08, 0x0819080819190819, 0x0819080819191908, 0x081908081919192b,
    0x0819080819192b19, 0x08190808192b0808, 0x08190808192b082b, 0x08190808192b1919,
    0x08190808192b2b08, 0x081908082b080819, 0x081908082b081908, 0x081908082b08192b,
    0x081908082b190808, 0x081908082b191919, 0x081908082b192b08, 0x081908082b2b0819,
    0x081908082b2b1908, 0x0819081908080808, 0x081908190808082b, 0x0819081908081919,
    0x0819081908082b08, 0x0819081908082b2b, 0x0819081908190819, 0x0819081908191908,
    0x081908190819192b, 0x0819081908192b19, 0x08190819082b0808, 0x08190819082b082b,
    0x08190819082b1919, 0x08190819082b2b08, 0x0819081919080819, 0x0819081919081908,
    0x081908191908192b, 0x0819081919082b19, 0x0819081919190808, 0x081908191919082b,
    0x0819081919191919, 0x0819081919192b08, 0x08190819192b0819, 0x08190819192b1908,
    0x081908192b080808, 0x081908192b08082b, 0x081908192b081919, 0x081908192b082b08,
    0x081908192b190819, 0x081908192b191908, 0x0819082b08080819, 0x0819082b08081908,
    0x0819082b08082b19, 0x0819082b08190808, 0x0819082b08191919, 0x0819082b082b0819,
    0x0819082b082b1908, 0x0819082b19080808, 0x0819082b19081919, 0x0819082b19190819,
    0x0819082b19191908, 0x0819082b2b080819, 0x0819082b2b081908, 0x0819082b2b190808,
    0x0819190808080808, 0x081919080808082b, 0x0819190808081919, 0x0819190808082b08,
    0x0819190808190819, 0x0819190808191908, 0x081919080819192b, 0x0819190808192b19,
    0x08191908082b0808, 0x08191908082b1919, 0x08191908082b2b08, 0x0819190819080819,
    0x0819190819081908, 0x081919081908192b, 0x0819190819082b19, 0x0819190819190808,
    0x081919081919082b, 0x0819190819191919, 0x0819190819192b08, 0x08191908192b0819,
    0x08191908192b1908, 0x081919082b080808, 0x081919082b08082b, 0x081919082b081919,
    0x081919082b082b08, 0x081919082b190819, 0x081919082b191908, 0x081919082b2b0808,
    0x0819191908080819, 0x0819191908081908, 0x081919190808192b, 0x0819191908082b19,
    0x0819191908190808, 0x081919190819082b, 0x0819191908191919, 0x0819191908192b08,
    0x08191919082b0819, 0x08191919082b1908, 0x0819191919080808, 0x081919191908082b,
    0x0819191919081919, 0x0819191919082b08, 0x0819191919190819, 0x0819191919191908,
    0x08191919192b0808, 0x081919192b080819, 0x081919192b081908, 0x081919192b190808,
    0x0819192b08080808, 0x0819192b08081919, 0x0819192b08082b08, 0x0819192b08190819,
    0x0819192b08191908, 0x0819192b082b0808, 0x0819192b19080819, 0x0819192b19081908,
    0x0819192b19190808, 0x0819192b2b080808, 0x0819192b2b2b2b2b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b080808192b, 0x08192b0808082b19, 0x08192b0808190808,
    0x08192b0808191919, 0x08192b0808192b08, 0x08192b08082b0819, 0x08192b0819080808,
    0x08192b081908082b, 0x08192b0819081919, 0x08192b0819082b08, 0x08192b0819190819,
    0x08192b0819191908, 0x08192b08192b0808, 0x08192b082b080819, 0x08192b082b081908,
    0x08192b1908080808, 0x08192b190808082b, 0x08192b1908081919, 0x08192b1908082b08,
    0x08192b1908190819, 0x08192b1908191908, 0x08192b19082b0808, 0x08192b1919080819,
    0x08192b1919081908, 0x08192b1919190808, 0x08192b19192b2b19, 0x08192b192b2b082b,
    0x08192b2b08081908, 0x08192b2b08190808, 0x08192b2b19080808, 0x08192b2b1919192b,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808081919, 0x082b080808082b08,
    0x082b080808190819, 0x082b080808191908, 0x082b08080819192b, 0x082b080808192b19,
    0x082b0808082b0808, 0x082b0808082b1919, 0x082b0808082b2b2b, 0x082b080819080819,
    0x082b080819081908, 0x082b080819190808, 0x082b08081919082b, 0x082b080819191919,
    0x082b0808192b1908, 0x082b08082b080808, 0x082b08082b082b2b, 0x082b08082b191908,
    0x082b08082b2b2b2b, 0x082b081908080819, 0x082b081908081908, 0x082b081908190808,
    0x082b08190819082b, 0x082b081908191919, 0x082b0819082b0819, 0x082b081919080808,
    0x082b08191908082b, 0x082b081919081919, 0x082b081919190819, 0x082b081919191908,
    0x082b0819192b0808, 0x082b08192b080819, 0x082b08192b081908, 0x082b08192b190808,
    0x082b082b08080808, 0x082b082b08082b2b, 0x082b082b082b082b, 0x082b082b082b2b08,
    0x082b082b082b2b2b, 0x082b082b19081908, 0x082b082b19190808, 0x082b082b2b082b08,
    0x082b082b2b082b2b, 0x082b082b2b2b2b08, 0x082b190808080819, 0x082b190808081908,
    0x082b19080808192b, 0x082b190808082b19, 0x082b190808190808, 0x082b190808191919,
    0x082b190808192b08, 0x082b1908082b0819, 0x082b1908082b1908, 0x082b190819080808,
    0x082b19081908082b, 0x082b190819081919, 0x082b190819082b08, 0x082b190819190819,
    0x082b190819191908, 0x082b1908192b0808, 0x082b19082b080819, 0x082b19082b081908,
    0x082b19082b190808, 0x082b191908080808, 0x082b191908081919, 0x082b191908082b08,
    0x082b191908190819, 0x082b191908191908, 0x082b1919082b0808, 0x082b191919080819,
    0x082b191919081908, 0x082b191919190808, 0x082b1919192b192b, 0x082b19192b080808,
    0x082b192b08080819, 0x082b192b08081908, 0x082b192b08190808, 0x082b192b19080808,
    0x082b192b19192b19, 0x082b2b0808080808, 0x082b2b0808081919, 0x082b2b0808190819,
    0x082b2b0808191908, 0x082b2b0819080819, 0x082b2b0819081908, 0x082b2b0819190808,
    0x082b2b082b082b2b, 0x082b2b082b2b2b2b, 0x082b2b1908080819, 0x082b2b1908081908,
    0x082b2b1908190808, 0x082b2b192b191919, 0x082b2b2b08082b2b, 0x082b2b2b082b082b,
    0x082b2b2b192b1908, 0x082b2b2b2b082b08, 0x082b2b2b2b082b2b, 0x1908080808080819,
    0x1908080808081908, 0x190808080808192b, 0x1908080808082b19, 0x1908080808190808,
    0x190808080819082b, 0x1908080808191919, 0x1908080808192b08, 0x1908080808192b2b,
    0x19080808082b0819, 0x19080808082b1908, 0x19080808082b192b, 0x1908080819080808,
    0x190808081908082b, 0x1908080819081919, 0x1908080819082b08, 0x1908080819082b2b,
    0x1908080819190819, 0x1908080819191908, 0x190808081919192b, 0x1908080819192b19,
    0x19080808192b0808, 0x19080808192b082b, 0x19080808192b1919, 0x190808082b080819,
    0x190808082b081908, 0x190808082b190808, 0x190808082b191919, 0x190808082b192b08,
    0x190808082b2b0819, 0x190808082b2b1908, 0x1908081908080808, 0x190808190808082b,
    0x1908081908081919, 0x1908081908082b08, 0x1908081908190819, 0x1908081908191908,
    0x190808190819192b, 0x1908081908192b19, 0x19080819082b0808, 0x19080819082b082b,
    0x19080819082b1919, 0x1908081919080819, 0x1908081919081908, 0x190808191908192b,
    0x1908081919082b19, 0x1908081919190808, 0x190808191919082b, 0x1908081919191919,
    0x1908081919192b08, 0x19080819192b0819, 0x19080819192b1908, 0x190808192b080808,
    0x190808192b08082b, 0x190808192b081919, 0x190808192b082b08, 0x190808192b190819,
    0x190808192b191908, 0x190808192b2b0808, 0x1908082b08080819, 0x1908082b08081908,
    0x1908082b08190808, 0x1908082b0819082b, 0x1908082b08191919, 0x1908082b08192b08,
    0x1908082b082b1908, 0x1908082b19080808, 0x1908082b19081919, 0x1908082b19082b08,
    0x1908082b19190819, 0x1908082b19191908, 0x1908082b192b0808, 0x1908082b2b080819,
    0x1908082b2b081908, 0x1908190808080808, 0x190819080808082b, 0x1908190808081919,
    0x1908190808082b08, 0x1908190808082b2b, 0x1908190808190819, 0x1908190808191908,
    0x190819080819192b, 0x1908190808192b19, 0x19081908082b0808, 0x19081908082b082b,
    0x19081908082b1919, 0x19081908082b2b08, 0x1908190819080819, 0x1908190819081908,
    0x190819081908192b, 0x1908190819082b19, 0x1908190819190808, 0x190819081919082b,
    0x1908190819191919, 0x1908190819192b08, 0x19081908192b0819, 0x19081908192b1908,
    0x190819082b080808, 0x190819082b08082b, 0x190819082b081919, 0x190819082b082b08,
    0x190819082b190819, 0x190819082b191908, 0x190819082b2b0808, 0x1908191908080819,
    0x1908191908081908, 0x190819190808192b, 0x1908191908082b19, 0x1908191908190808,
    0x190819190819082b, 0x1908191908191919, 0x1908191908192b08, 0x19081919082b0819,
    0x19081919082b1908, 0x1908191919080808, 0x190819191908082b, 0x1908191919081919,
    0x1908191919082b08, 0x1908191919190819, 0x1908191919191908, 0x19081919192b0808,
    0x19081919192b2b2b, 0x190819192b080819, 0x190819192b081908, 0x190819192b190808,
    0x1908192b08080808, 0x1908192b0808082b, 0x1908192b08081919, 0x1908192b08082b08,
    0x1908192b08190819, 0x1908192b08191908, 0x1908192b082b0808, 0x1908192b19080819,
    0x1908192b19081908, 0x1908192b19190808, 0x1908192b2b080808, 0x1908192b2b2b1919,
    0x19082b0808080819, 0x19082b0808081908, 0x19082b0808082b19, 0x19082b0808190808,
    0x19082b080819082b, 0x19082b0808191919, 0x19082b0808192b08, 0x19082b08082b0819,
    0x19082b08082b1908, 0x19082b0819080808, 0x19082b081908082b, 0x19082b0819081919,
    0x19082b0819082b08, 0x19082b0819190819, 0x19082b0819191908, 0x19082b08192b0808,
    0x19082b082b081908, 0x19082b082b190808, 0x19082b1908080808, 0x19082b190808082b,
    0x19082b1908081919, 0x19082b1908082b08, 0x19082b1908190819, 0x19082b1908191908,
    0x19082b19082b0808, 0x19082b1919080819, 0x19082b1919081908, 0x19082b1919190808,
    0x19082b192b080808, 0x19082b192b19192b, 0x19082b2b08080819, 0x19082b2b08081908,
    0x19082b2b08190808, 0x19082b2b19080808, 0x1919080808080808, 0x191908080808082b,
    0x1919080808081919, 0x1919080808082b08, 0x1919080808190819, 0x1919080808191908,
    0x191908080819192b, 0x1919080808192b19, 0x19190808082b0808, 0x19190808082b082b,
    0x19190808082b1919, 0x19190808082b2b08, 0x1919080819080819, 0x1919080819081908,
    0x191908081908192b, 0x1919080819082b19, 0x1919080819190808, 0x191908081919082b,
    0x1919080819191919, 0x1919080819192b08, 0x19190808192b0819, 0x19190808192b1908,
    0x191908082b080808, 0x191908082b08082b, 0x191908082b081919, 0x191908082b082b08,
    0x191908082b190819, 0x191908082b191908, 0x1919081908080819, 0x1919081908081908,
    0x191908190808192b, 0x1919081908082b19, 0x1919081908190808, 0x191908190819082b,
    0x1919081908191919, 0x1919081908192b08, 0x19190819082b0819, 0x19190819082b1908,
    0x1919081919080808, 0x191908191908082b, 0x1919081919081919, 0x1919081919082b08,
    0x1919081919190819, 0x1919081919191908, 0x19190819192b0808, 0x191908192b080819,
    0x191908192b081908, 0x191908192b190808, 0x1919082b08080808, 0x1919082b08081919,
    0x1919082b08082b08, 0x1919082b08190819, 0x1919082b08191908, 0x1919082b082b0808,
    0x1919082b19080819, 0x1919082b19081908, 0x1919082b19190808, 0x1919082b192b2b19,
    0x1919082b2b080808, 0x1919190808080819, 0x1919190808081908, 0x191919080808192b,
    0x1919190808082b19, 0x1919190808190808, 0x191919080819082b, 0x1919190808191919,
    0x1919190808192b08, 0x19191908082b0819, 0x19191908082b1908, 0x1919190819080808,
    0x191919081908082b, 0x1919190819081919, 0x1919190819082b08, 0x1919190819190819,
    0x1919190819191908, 0x19191908192b0808, 0x191919082b080819, 0x191919082b081908,
    0x191919082b190808, 0x1919191908080808, 0x191919190808082b, 0x1919191908081919,
    0x1919191908082b08, 0x1919191908190819, 0x1919191908191908, 0x19191919082b0808,
    0x1919191919080819, 0x1919191919081908, 0x1919191919190808, 0x191919192b080808,
    0x1919192b08080819, 0x1919192b08081908, 0x1919192b08190808, 0x1919192b082b192b,
    0x1919192b19080808, 0x19192b0808080808, 0x19192b080808082b, 0x19192b0808081919,
    0x19192b0808082b08, 0x19192b0808190819, 0x19192b0808191908, 0x19192b08082b0808,
    0x19192b0819080819, 0x19192b0819081908, 0x19192b0819190808, 0x19192b0819192b2b,
    0x19192b082b080808, 0x19192b1908080819, 0x19192b1908081908, 0x19192b1908190808,
    0x19192b1919080808, 0x19192b2b08080808, 0x19192b2b08192b19, 0x19192b2b2b081919,
    0x19192b2b2b2b2b08, 0x192b080808080819, 0x192b080808081908, 0x192b08080808192b,
    0x192b080808190808, 0x192b08080819082b, 0x192b080808191919, 0x192b080808192b08,
    0x192b0808082b0819, 0x192b0808082b1908, 0x192b080819080808, 0x192b080819081919,
    0x192b080819082b08, 0x192b080819190819, 0x192b080819191908, 0x192b0808192b0808,
    0x192b08082b081908, 0x192b08082b190808, 0x192b081908080808, 0x192b08190808082b,
    0x192b081908081919, 0x192b081908082b08, 0x192b081908190819, 0x192b081908191908,
    0x192b0819082b0808, 0x192b081919080819, 0x192b081919081908, 0x192b081919190808,
    0x192b08192b080808, 0x192b08192b192b19, 0x192b082b08081908, 0x192b082b08190808,
    0x192b082b19080808, 0x192b082b1919192b, 0x192b082b2b2b0819, 0x192b190808080808,
    0x192b190808081919, 0x192b190808082b08, 0x192b190808190819, 0x192b190808191908,
    0x192b1908082b0808, 0x192b190819080819, 0x192b190819081908, 0x192b190819190808,
    0x192b19082b080808, 0x192b191908080819, 0x192b191908081908, 0x192b191908190808,
    0x192b191919080808, 0x192b191919082b2b, 0x192b1919192b2b08, 0x192b19192b19082b,
    0x192b192b08080808, 0x192b192b2b191908, 0x192b2b0808080819, 0x192b2b0808081908,
    0x192b2b0808190808, 0x192b2b08192b1919, 0x192b2b082b192b08, 0x192b2b1908080808,
    0x192b2b19082b2b2b, 0x192b2b2b1908082b, 0x192b2b2b2b2b0819, 0x2b08080808080808,
    0x2b0808080808082b, 0x2b08080808081919, 0x2b08080808082b08, 0x2b08080808190819,
    0x2b08080808191908, 0x2b08080808192b19, 0x2b080808082b0808, 0x2b080808082b1919,
    0x2b08080819080819, 0x2b08080819081908, 0x2b08080819190808, 0x2b0808081919082b,
    0x2b08080819191919, 0x2b08080819192b08, 0x2b080808192b0819, 0x2b0808082b080808,
    0x2b0808082b081919, 0x2b0808082b190819, 0x2b0808082b191908, 0x2b08081908080819,
    0x2b08081908081908, 0x2b08081908082b19, 0x2b08081908190808, 0x2b0808190819082b,
    0x2b08081908191919, 0x2b08081908192b08, 0x2b080819082b0819, 0x2b080819082b1908,
    0x2b08081919080808, 0x2b0808191908082b, 0x2b08081919081919, 0x2b08081919082b08,
    0x2b08081919190819, 0x2b08081919191908, 0x2b0808192b080819, 0x2b0808192b081908,
    0x2b0808192b190808, 0x2b0808192b2b2b19, 0x2b08082b08080808, 0x2b08082b08081919,
    0x2b08082b08082b2b, 0x2b08082b08190819, 0x2b08082b08191908, 0x2b08082b19080819,
    0x2b08082b19081908, 0x2b08082b19190808, 0x2b08190808080819, 0x2b08190808081908,
    0x2b0819080808192b, 0x2b08190808082b19, 0x2b08190808190808, 0x2b0819080819082b,
    0x2b08190808191919, 0x2b08190808192b08, 0x2b081908082b0819, 0x2b08190819080808,
    0x2b0819081908082b, 0x2b08190819081919, 0x2b08190819082b08, 0x2b08190819190819,
    0x2b08190819191908, 0x2b081908192b0808, 0x2b0819082b080819, 0x2b0819082b081908,
    0x2b0819082b190808, 0x2b08191908080808, 0x2b0819190808082b, 0x2b08191908081919,
    0x2b08191908082b08, 0x2b08191908190819, 0x2b08191908191908, 0x2b081919082b0808,
    0x2b08191919080819, 0x2b08191919081908, 0x2b08191919190808, 0x2b0819192b080808,
    0x2b0819192b082b2b, 0x2b08192b08080819, 0x2b08192b08081908, 0x2b08192b08190808,
    0x2b08192b082b2b19, 0x2b08192b19080808, 0x2b082b0808080808, 0x2b082b0808081919,
    0x2b082b0808190819, 0x2b082b0808191908, 0x2b082b0819080819, 0x2b082b0819081908,
    0x2b082b0819190808, 0x2b082b082b2b082b, 0x2b082b1908080819, 0x2b082b1908081908,
    0x2b082b1919080808, 0x2b082b19192b1919, 0x2b082b2b082b082b, 0x2b082b2b19192b08,
    0x2b082b2b19192b2b, 0x2b082b2b2b08082b, 0x2b082b2b2b2b082b, 0x2b19080808080819,
    0x2b19080808081908, 0x2b19080808082b19, 0x2b19080808190808, 0x2b1908080819082b,
    0x2b19080808191919, 0x2b19080808192b08, 0x2b190808082b1908, 0x2b19080819080808,
    0x2b1908081908082b, 0x2b19080819081919, 0x2b19080819082b08, 0x2b19080819190819,
    0x2b19080819191908, 0x2b190808192b0808, 0x2b1908082b080819, 0x2b1908082b081908,
    0x2b1908082b190808, 0x2b19081908080808, 0x2b19081908081919, 0x2b19081908190819,
    0x2b19081908191908, 0x2b19081919080819, 0x2b19081919081908, 0x2b19081919190808,
    0x2b19081919192b2b, 0x2b19082b08080819, 0x2b19082b08081908, 0x2b19082b08190808,
    0x2b19082b19080808, 0x2b19082b2b2b192b, 0x2b19190808080808, 0x2b1919080808082b,
    0x2b19190808081919, 0x2b19190808082b08, 0x2b19190808190819, 0x2b19190808191908,
    0x2b191908082b0808, 0x2b19190819080819, 0x2b19190819081908, 0x2b19190819190808,
    0x2b1919082b080808, 0x2b1919082b19192b, 0x2b19191908080819, 0x2b19191908081908,
    0x2b19191908190808, 0x2b19191919080808, 0x2b1919192b192b08, 0x2b1919192b2b0819,
    0x2b19192b08080808, 0x2b19192b1908192b, 0x2b19192b192b1908, 0x2b192b0808080819,
    0x2b192b0808081908, 0x2b192b0808190808, 0x2b192b08082b192b, 0x2b192b0819080808,
    0x2b192b082b2b2b19, 0x2b192b1908080808, 0x2b192b1919082b19, 0x2b192b191919082b,
    0x2b192b2b2b190808, 0x2b2b080808080808, 0x2b2b080808081919, 0x2b2b080808082b2b,
    0x2b2b080808191908, 0x2b2b0808082b082b, 0x2b2b0808082b2b2b, 0x2b2b080819080819,
    0x2b2b080819081908, 0x2b2b080819190808, 0x2b2b08082b2b082b, 0x2b2b08082b2b2b2b,
    0x2b2b081919080808, 0x2b2b0819192b1919, 0x2b2b082b0808082b, 0x2b2b082b08082b2b,
    0x2b2b082b082b082b, 0x2b2b082b082b2b08, 0x2b2b082b082b2b2b, 0x2b2b082b2b08082b,
    0x2b2b082b2b082b08, 0x2b2b082b2b082b2b, 0x2b2b082b2b2b2b08, 0x2b2b190808080819,
    0x2b2b190808081908, 0x2b2b190808190808, 0x2b2b190819080808, 0x2b2b19082b082b19,
    0x2b2b19082b2b1908, 0x2b2b191908080808, 0x2b2b191908192b19, 0x2b2b192b19190819,
    0x2b2b2b0808082b2b, 0x2b2b2b08082b2b08, 0x2b2b2b082b2b082b, 0x2b2b2b1919191908,
    0x2b2b2b192b08192b, 0x2b2b2b2b08082b08, 0x2b2b2b2b08082b2b, 0x2b2b2b2b082b0808,
    0x2b2b2b2b082b082b, 0x2b2b2b2b082b2b08, 0x2b2b2b2b2b082b08, 0x2b2b2b2b2b2b2b2b,
];

/// IQ3_XXS's codebook: 256 entries of four packed byte levels.
#[rustfmt::skip]
#[allow(clippy::unreadable_literal)]
const IQ3XXS_GRID: [u32; 256] = [
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
];

/// The +-0.125 offset IQ1_S and IQ1_M add to every level (ggml-common.h IQ1S_DELTA).
const IQ1S_DELTA: f32 = 0.125;

/// IQ1_S / IQ1_M's codebook: 2048 entries of eight levels in {-1, 0, 1}, two bits each
/// (level + 1), from ggml-common.h's iq1s_grid; the same numbers as gguf-py's grid_hex.
const IQ1S_GRID: [u16; 2048] = [
    0x0000, 0x0002, 0x0005, 0x0008, 0x000a, 0x0011, 0x0015, 0x0020, 0x0022, 0x0028,
    0x002a, 0x0045, 0x0051, 0x0054, 0x0056, 0x0065, 0x0080, 0x0082, 0x0088, 0x008a,
    0x0095, 0x00a0, 0x00a2, 0x00a8, 0x00aa, 0x0104, 0x0105, 0x0111, 0x0114, 0x0116,
    0x0119, 0x011a, 0x0125, 0x0141, 0x0146, 0x0149, 0x0152, 0x0155, 0x015a, 0x0161,
    0x0164, 0x0166, 0x0168, 0x0185, 0x0191, 0x0194, 0x0196, 0x01a5, 0x0200, 0x0202,
    0x0208, 0x020a, 0x0215, 0x0220, 0x0222, 0x0228, 0x022a, 0x0245, 0x0251, 0x0259,
    0x0264, 0x0269, 0x0280, 0x0282, 0x0288, 0x028a, 0x0291, 0x0295, 0x0299, 0x02a0,
    0x02a2, 0x02a8, 0x02aa, 0x0411, 0x0414, 0x0416, 0x0425, 0x0441, 0x0449, 0x0455,
    0x045a, 0x0464, 0x0465, 0x0491, 0x0499, 0x04a5, 0x0501, 0x0504, 0x0505, 0x0506,
    0x0515, 0x0518, 0x051a, 0x0529, 0x0540, 0x0545, 0x054a, 0x0550, 0x0551, 0x0554,
    0x0555, 0x0556, 0x0559, 0x0560, 0x0562, 0x0565, 0x0568, 0x056a, 0x0581, 0x0591,
    0x0595, 0x0598, 0x059a, 0x05a1, 0x05a4, 0x05a5, 0x05a6, 0x05a9, 0x0614, 0x0619,
    0x0641, 0x0644, 0x0650, 0x0652, 0x0655, 0x0658, 0x0660, 0x0661, 0x0666, 0x0669,
    0x0685, 0x0691, 0x0694, 0x0699, 0x0800, 0x0802, 0x0808, 0x080a, 0x0815, 0x0820,
    0x0822, 0x0828, 0x082a, 0x0845, 0x0851, 0x0856, 0x0865, 0x0880, 0x0882, 0x0888,
    0x088a, 0x0895, 0x08a0, 0x08a2, 0x08a8, 0x08aa, 0x0905, 0x0911, 0x0914, 0x0919,
    0x0924, 0x0925, 0x0941, 0x0950, 0x0951, 0x0955, 0x0961, 0x0964, 0x0969, 0x0991,
    0x0994, 0x0996, 0x0999, 0x09a5, 0x0a00, 0x0a02, 0x0a08, 0x0a0a, 0x0a15, 0x0a20,
    0x0a22, 0x0a28, 0x0a2a, 0x0a45, 0x0a51, 0x0a59, 0x0a61, 0x0a65, 0x0a80, 0x0a82,
    0x0a85, 0x0a88, 0x0a8a, 0x0a95, 0x0aa0, 0x0aa2, 0x0aa8, 0x0aaa, 0x1010, 0x1011,
    0x1014, 0x1019, 0x1024, 0x1025, 0x1041, 0x1044, 0x1050, 0x1055, 0x1058, 0x1061,
    0x1064, 0x1065, 0x1069, 0x1091, 0x1094, 0x1096, 0x10a1, 0x10a5, 0x1101, 0x1104,
    0x1106, 0x1109, 0x1110, 0x1112, 0x1115, 0x1118, 0x1121, 0x1124, 0x1129, 0x1145,
    0x114a, 0x1150, 0x1151, 0x1152, 0x1154, 0x1155, 0x1156, 0x1159, 0x1160, 0x1165,
    0x1184, 0x1192, 0x1195, 0x11a1, 0x11a4, 0x1211, 0x1214, 0x1216, 0x1225, 0x1240,
    0x1246, 0x1249, 0x1252, 0x1255, 0x1258, 0x125a, 0x1264, 0x1266, 0x1285, 0x1291,
    0x1294, 0x1296, 0x12a5, 0x1401, 0x1406, 0x1409, 0x1414, 0x1415, 0x1418, 0x1419,
    0x1421, 0x1426, 0x1441, 0x1445, 0x1446, 0x1448, 0x144a, 0x1451, 0x1454, 0x1455,
    0x1456, 0x1459, 0x1462, 0x1465, 0x1468, 0x1484, 0x1489, 0x1490, 0x1494, 0x1495,
    0x1498, 0x1499, 0x149a, 0x14a1, 0x14a4, 0x14a5, 0x14a9, 0x1502, 0x1505, 0x150a,
    0x1511, 0x1514, 0x1515, 0x1516, 0x1519, 0x1520, 0x1522, 0x1525, 0x1528, 0x152a,
    0x1541, 0x1544, 0x1545, 0x1546, 0x1551, 0x1552, 0x1554, 0x1555, 0x1556, 0x1559,
    0x155a, 0x1561, 0x1564, 0x1565, 0x1566, 0x1569, 0x1580, 0x1582, 0x1584, 0x1585,
    0x1588, 0x158a, 0x1590, 0x1591, 0x1594, 0x1595, 0x1596, 0x1599, 0x159a, 0x15a0,
    0x15a2, 0x15a5, 0x1601, 0x1604, 0x1605, 0x1606, 0x1615, 0x1616, 0x1618, 0x161a,
    0x1621, 0x1626, 0x1640, 0x1642, 0x1644, 0x1645, 0x1648, 0x164a, 0x1651, 0x1655,
    0x1656, 0x1658, 0x1659, 0x1661, 0x1664, 0x1665, 0x1668, 0x1669, 0x166a, 0x1686,
    0x168a, 0x1692, 0x1695, 0x16a4, 0x16a9, 0x1811, 0x1816, 0x1825, 0x1841, 0x1844,
    0x1846, 0x1849, 0x1850, 0x1855, 0x1858, 0x185a, 0x1860, 0x1861, 0x1864, 0x1866,
    0x1869, 0x1885, 0x1891, 0x1894, 0x18a5, 0x1910, 0x1912, 0x1915, 0x191a, 0x1921,
    0x1925, 0x1942, 0x1944, 0x1945, 0x1948, 0x1951, 0x1954, 0x1955, 0x1956, 0x1959,
    0x195a, 0x1960, 0x1965, 0x196a, 0x1989, 0x1991, 0x1992, 0x1995, 0x1998, 0x19a1,
    0x19a6, 0x19a9, 0x1a09, 0x1a16, 0x1a24, 0x1a26, 0x1a44, 0x1a46, 0x1a49, 0x1a50,
    0x1a52, 0x1a55, 0x1a58, 0x1a61, 0x1a66, 0x1a69, 0x1a85, 0x1a91, 0x1a96, 0x1a9a,
    0x2000, 0x2002, 0x2008, 0x200a, 0x2015, 0x2020, 0x2022, 0x2025, 0x2028, 0x202a,
    0x2045, 0x2051, 0x2059, 0x2061, 0x2065, 0x2080, 0x2082, 0x2088, 0x208a, 0x2095,
    0x20a0, 0x20a2, 0x20a5, 0x20a8, 0x20aa, 0x2105, 0x2111, 0x2114, 0x2119, 0x2125,
    0x2142, 0x2144, 0x2149, 0x2155, 0x2158, 0x215a, 0x2161, 0x2164, 0x2165, 0x2166,
    0x2185, 0x2190, 0x2196, 0x2199, 0x21a5, 0x2201, 0x2208, 0x220a, 0x2211, 0x2215,
    0x2220, 0x2222, 0x2228, 0x222a, 0x2245, 0x2251, 0x2256, 0x2259, 0x2265, 0x2281,
    0x2288, 0x228a, 0x2291, 0x2295, 0x22a0, 0x22a2, 0x22a8, 0x22aa, 0x2405, 0x2414,
    0x2416, 0x2419, 0x2425, 0x2444, 0x2445, 0x2446, 0x2449, 0x2452, 0x2455, 0x2458,
    0x245a, 0x2466, 0x2485, 0x2491, 0x2494, 0x2499, 0x24a1, 0x24a5, 0x2509, 0x2515,
    0x2521, 0x2529, 0x2540, 0x2545, 0x2548, 0x2551, 0x2554, 0x2555, 0x2559, 0x2562,
    0x2565, 0x2568, 0x2589, 0x2590, 0x2594, 0x2595, 0x2598, 0x259a, 0x25a1, 0x25a4,
    0x25a6, 0x25a9, 0x2605, 0x2610, 0x2612, 0x2619, 0x2625, 0x2641, 0x2649, 0x2655,
    0x2660, 0x2661, 0x2669, 0x2684, 0x2686, 0x2690, 0x269a, 0x2800, 0x2802, 0x2808,
    0x280a, 0x2815, 0x2820, 0x2822, 0x2828, 0x282a, 0x2845, 0x2851, 0x2854, 0x2865,
    0x2880, 0x2882, 0x2888, 0x288a, 0x28a0, 0x28a2, 0x28a8, 0x28aa, 0x2909, 0x2911,
    0x2914, 0x2919, 0x2925, 0x2946, 0x2949, 0x2952, 0x2955, 0x2961, 0x2964, 0x2966,
    0x2969, 0x2985, 0x2990, 0x2996, 0x2999, 0x29a4, 0x29a5, 0x2a00, 0x2a02, 0x2a08,
    0x2a0a, 0x2a20, 0x2a22, 0x2a28, 0x2a2a, 0x2a45, 0x2a51, 0x2a56, 0x2a59, 0x2a65,
    0x2a80, 0x2a82, 0x2a88, 0x2a8a, 0x2a95, 0x2aa0, 0x2aa2, 0x2aa8, 0x2aaa, 0x4005,
    0x4011, 0x4016, 0x4025, 0x4049, 0x4052, 0x4055, 0x4058, 0x405a, 0x4061, 0x4064,
    0x4066, 0x4094, 0x4099, 0x40a1, 0x40a6, 0x4100, 0x4101, 0x4104, 0x4106, 0x4109,
    0x4112, 0x4115, 0x4116, 0x4118, 0x411a, 0x4121, 0x4126, 0x4129, 0x4145, 0x4148,
    0x414a, 0x4151, 0x4154, 0x4155, 0x4156, 0x4159, 0x415a, 0x4165, 0x4168, 0x416a,
    0x4181, 0x4184, 0x4186, 0x4190, 0x4192, 0x4195, 0x41a0, 0x41a1, 0x41a2, 0x4205,
    0x4211, 0x4214, 0x4216, 0x4225, 0x4241, 0x4252, 0x4255, 0x425a, 0x4264, 0x4269,
    0x4289, 0x4294, 0x42a5, 0x4401, 0x4415, 0x4419, 0x4429, 0x4445, 0x4448, 0x444a,
    0x4451, 0x4454, 0x4455, 0x4456, 0x4461, 0x4462, 0x4465, 0x4468, 0x446a, 0x4481,
    0x4486, 0x4489, 0x4490, 0x4492, 0x4495, 0x44a0, 0x44a1, 0x44a9, 0x4501, 0x4502,
    0x4505, 0x450a, 0x4511, 0x4514, 0x4515, 0x4516, 0x4519, 0x4520, 0x4525, 0x452a,
    0x4541, 0x4544, 0x4545, 0x4546, 0x4549, 0x4550, 0x4551, 0x4554, 0x4555, 0x4556,
    0x4558, 0x4559, 0x4561, 0x4564, 0x4565, 0x4566, 0x4569, 0x4582, 0x4584, 0x4585,
    0x4588, 0x4591, 0x4594, 0x4595, 0x4596, 0x4599, 0x459a, 0x45a5, 0x45a8, 0x45aa,
    0x4601, 0x4605, 0x4609, 0x4614, 0x4615, 0x4618, 0x461a, 0x4621, 0x4624, 0x4629,
    0x4640, 0x4642, 0x4645, 0x4648, 0x4650, 0x4651, 0x4652, 0x4655, 0x4656, 0x4659,
    0x4662, 0x4665, 0x4668, 0x4681, 0x4685, 0x468a, 0x4694, 0x4695, 0x46a1, 0x46a4,
    0x46a6, 0x4805, 0x4811, 0x4815, 0x481a, 0x4825, 0x4842, 0x4849, 0x4850, 0x4855,
    0x4858, 0x4861, 0x4864, 0x4866, 0x4869, 0x4885, 0x4891, 0x4894, 0x4896, 0x4899,
    0x48a5, 0x4901, 0x4905, 0x4906, 0x490a, 0x4910, 0x4914, 0x4915, 0x4918, 0x4921,
    0x4924, 0x4926, 0x4940, 0x4945, 0x494a, 0x4951, 0x4952, 0x4954, 0x4955, 0x4956,
    0x4959, 0x4960, 0x4962, 0x4965, 0x4966, 0x496a, 0x4986, 0x4989, 0x4992, 0x4995,
    0x4996, 0x4998, 0x49a1, 0x49a4, 0x49a6, 0x49a9, 0x4a16, 0x4a44, 0x4a46, 0x4a49,
    0x4a55, 0x4a58, 0x4a5a, 0x4a64, 0x4a69, 0x4a94, 0x4aa5, 0x5001, 0x5004, 0x5005,
    0x5006, 0x5009, 0x5012, 0x5015, 0x501a, 0x5021, 0x5024, 0x5029, 0x5040, 0x5045,
    0x5048, 0x5051, 0x5054, 0x5055, 0x5056, 0x5059, 0x5065, 0x5068, 0x5086, 0x5089,
    0x5095, 0x5098, 0x50a0, 0x50a1, 0x50a6, 0x50a9, 0x5105, 0x5108, 0x5109, 0x510a,
    0x5111, 0x5114, 0x5115, 0x5116, 0x5118, 0x5119, 0x5120, 0x5125, 0x5126, 0x5128,
    0x512a, 0x5141, 0x5144, 0x5145, 0x5146, 0x5149, 0x5150, 0x5151, 0x5152, 0x5154,
    0x5155, 0x5156, 0x5158, 0x5159, 0x515a, 0x5161, 0x5164, 0x5165, 0x5166, 0x5169,
    0x5182, 0x5185, 0x5191, 0x5194, 0x5195, 0x5196, 0x5199, 0x51a0, 0x51a5, 0x51aa,
    0x5201, 0x5206, 0x5212, 0x5215, 0x521a, 0x5221, 0x5224, 0x5242, 0x5245, 0x524a,
    0x5251, 0x5254, 0x5255, 0x5256, 0x5259, 0x5262, 0x5265, 0x5285, 0x5290, 0x5292,
    0x5295, 0x5299, 0x529a, 0x52a4, 0x5404, 0x5405, 0x5411, 0x5414, 0x5415, 0x5416,
    0x5418, 0x5419, 0x5421, 0x5425, 0x5428, 0x542a, 0x5441, 0x5444, 0x5445, 0x5446,
    0x5449, 0x544a, 0x5450, 0x5451, 0x5454, 0x5455, 0x5456, 0x5458, 0x5459, 0x545a,
    0x5461, 0x5462, 0x5464, 0x5465, 0x5466, 0x5469, 0x5480, 0x5488, 0x548a, 0x5491,
    0x5494, 0x5495, 0x5496, 0x5499, 0x54a1, 0x54a4, 0x54a5, 0x54aa, 0x5501, 0x5502,
    0x5504, 0x5505, 0x5506, 0x5509, 0x5510, 0x5511, 0x5512, 0x5514, 0x5515, 0x5516,
    0x5519, 0x551a, 0x5521, 0x5524, 0x5525, 0x5526, 0x5529, 0x5540, 0x5541, 0x5542,
    0x5544, 0x5545, 0x5546, 0x5548, 0x5549, 0x5550, 0x5551, 0x5552, 0x5554, 0x5555,
    0x5556, 0x5558, 0x5559, 0x555a, 0x5560, 0x5561, 0x5564, 0x5565, 0x5566, 0x5568,
    0x5569, 0x556a, 0x5581, 0x5584, 0x5585, 0x5589, 0x558a, 0x5590, 0x5591, 0x5594,
    0x5595, 0x5596, 0x5598, 0x5599, 0x55a1, 0x55a4, 0x55a5, 0x55a6, 0x55a9, 0x5600,
    0x5601, 0x5602, 0x5604, 0x5606, 0x5608, 0x5609, 0x5611, 0x5614, 0x5615, 0x5618,
    0x5619, 0x5620, 0x5621, 0x5622, 0x5624, 0x5625, 0x5626, 0x5628, 0x5629, 0x5641,
    0x5645, 0x5646, 0x5648, 0x5649, 0x564a, 0x5650, 0x5651, 0x5652, 0x5654, 0x5655,
    0x5656, 0x5658, 0x5659, 0x565a, 0x5661, 0x5664, 0x5665, 0x5669, 0x5682, 0x5685,
    0x5686, 0x5688, 0x5689, 0x568a, 0x5691, 0x5695, 0x569a, 0x56a2, 0x56a5, 0x56a6,
    0x56a8, 0x56a9, 0x5804, 0x5805, 0x5806, 0x5809, 0x5810, 0x5815, 0x5818, 0x5821,
    0x582a, 0x5845, 0x5848, 0x584a, 0x5851, 0x5854, 0x5855, 0x5856, 0x5858, 0x5859,
    0x5860, 0x5862, 0x5864, 0x5865, 0x5882, 0x5889, 0x5890, 0x5892, 0x5895, 0x5898,
    0x58a1, 0x58a9, 0x5901, 0x5902, 0x5905, 0x590a, 0x5911, 0x5914, 0x5915, 0x5916,
    0x5919, 0x5925, 0x5941, 0x5944, 0x5945, 0x5946, 0x5949, 0x5950, 0x5951, 0x5952,
    0x5954, 0x5955, 0x5956, 0x5958, 0x5959, 0x595a, 0x5961, 0x5964, 0x5965, 0x5966,
    0x5969, 0x5981, 0x5985, 0x5989, 0x5991, 0x5994, 0x5995, 0x5996, 0x5998, 0x5999,
    0x59a5, 0x5a04, 0x5a08, 0x5a15, 0x5a1a, 0x5a20, 0x5a25, 0x5a26, 0x5a29, 0x5a45,
    0x5a48, 0x5a49, 0x5a51, 0x5a55, 0x5a56, 0x5a58, 0x5a59, 0x5a62, 0x5a65, 0x5a68,
    0x5a6a, 0x5a81, 0x5a8a, 0x5a92, 0x5a95, 0x5a96, 0x5a98, 0x5a9a, 0x5aa1, 0x6005,
    0x6014, 0x6016, 0x6019, 0x6025, 0x6044, 0x6050, 0x6055, 0x6056, 0x6058, 0x605a,
    0x6061, 0x6064, 0x6066, 0x6069, 0x6081, 0x6096, 0x60a5, 0x6101, 0x6104, 0x6106,
    0x6109, 0x6112, 0x6115, 0x6121, 0x6122, 0x6126, 0x6129, 0x6145, 0x6149, 0x6151,
    0x6155, 0x6156, 0x6159, 0x6165, 0x6166, 0x616a, 0x6184, 0x618a, 0x6192, 0x6195,
    0x61a1, 0x61a6, 0x61a9, 0x6211, 0x6216, 0x6219, 0x6240, 0x6241, 0x6246, 0x6255,
    0x6256, 0x6258, 0x6260, 0x6285, 0x6291, 0x6296, 0x62a5, 0x6411, 0x6412, 0x6415,
    0x6416, 0x641a, 0x6421, 0x6426, 0x6429, 0x6440, 0x6442, 0x6445, 0x6448, 0x644a,
    0x6451, 0x6454, 0x6455, 0x6456, 0x6459, 0x645a, 0x6460, 0x6462, 0x6465, 0x6484,
    0x6485, 0x6489, 0x6490, 0x6492, 0x6494, 0x6495, 0x6496, 0x6498, 0x649a, 0x64a1,
    0x64a4, 0x64a9, 0x6505, 0x6508, 0x650a, 0x6511, 0x6515, 0x6516, 0x6519, 0x6544,
    0x6545, 0x6546, 0x6549, 0x6550, 0x6551, 0x6554, 0x6555, 0x6556, 0x6559, 0x6561,
    0x6564, 0x6565, 0x6566, 0x6569, 0x6586, 0x6589, 0x658a, 0x6591, 0x6595, 0x6596,
    0x6599, 0x659a, 0x65a2, 0x65a5, 0x65a6, 0x65a8, 0x6602, 0x6609, 0x6615, 0x6620,
    0x6626, 0x6628, 0x6629, 0x6640, 0x6645, 0x6648, 0x664a, 0x6651, 0x6654, 0x6655,
    0x6656, 0x6658, 0x665a, 0x6660, 0x6665, 0x6668, 0x6680, 0x6682, 0x6685, 0x668a,
    0x6694, 0x6696, 0x6698, 0x6699, 0x66a0, 0x66a4, 0x66a6, 0x66aa, 0x6816, 0x6819,
    0x6825, 0x6841, 0x6852, 0x6855, 0x685a, 0x6861, 0x6869, 0x6885, 0x6891, 0x6898,
    0x68a6, 0x6901, 0x6904, 0x6910, 0x6915, 0x6921, 0x6924, 0x6926, 0x6929, 0x6940,
    0x6941, 0x6945, 0x6946, 0x6948, 0x6951, 0x6954, 0x6955, 0x6956, 0x6959, 0x6960,
    0x6965, 0x696a, 0x6982, 0x6984, 0x698a, 0x6995, 0x69a1, 0x69a4, 0x69a5, 0x69a9,
    0x6a11, 0x6a16, 0x6a18, 0x6a41, 0x6a44, 0x6a49, 0x6a50, 0x6a55, 0x6a58, 0x6a5a,
    0x6a64, 0x6a65, 0x6a69, 0x6a86, 0x6a94, 0x6a98, 0x6a9a, 0x6aa6, 0x8000, 0x8002,
    0x8008, 0x800a, 0x8020, 0x8022, 0x8028, 0x802a, 0x8045, 0x8050, 0x8051, 0x8054,
    0x8056, 0x8059, 0x8065, 0x8080, 0x8082, 0x8088, 0x808a, 0x8095, 0x80a0, 0x80a2,
    0x80a8, 0x80aa, 0x8105, 0x8111, 0x8114, 0x8116, 0x8119, 0x8125, 0x8141, 0x8144,
    0x8149, 0x8150, 0x8152, 0x8155, 0x8156, 0x8158, 0x8159, 0x8164, 0x8166, 0x8169,
    0x8185, 0x8189, 0x8194, 0x8196, 0x8199, 0x81a5, 0x8200, 0x8202, 0x8208, 0x820a,
    0x8215, 0x8220, 0x8222, 0x8228, 0x822a, 0x8251, 0x8254, 0x8259, 0x8265, 0x8280,
    0x8282, 0x8288, 0x828a, 0x8295, 0x82a0, 0x82a2, 0x82a8, 0x82aa, 0x8414, 0x8419,
    0x8441, 0x8444, 0x8451, 0x8455, 0x845a, 0x8461, 0x8464, 0x8469, 0x8494, 0x8499,
    0x8501, 0x8509, 0x8512, 0x8515, 0x851a, 0x8526, 0x8529, 0x8540, 0x8541, 0x8545,
    0x8548, 0x8551, 0x8554, 0x8555, 0x8556, 0x8559, 0x855a, 0x8565, 0x8566, 0x8568,
    0x856a, 0x8581, 0x8584, 0x8586, 0x8589, 0x8590, 0x8592, 0x8595, 0x8598, 0x85a6,
    0x8611, 0x8616, 0x8619, 0x8625, 0x8641, 0x8644, 0x8649, 0x864a, 0x8650, 0x8655,
    0x8659, 0x865a, 0x8661, 0x8666, 0x866a, 0x8685, 0x8691, 0x869a, 0x86a4, 0x8800,
    0x8802, 0x8808, 0x880a, 0x8815, 0x8820, 0x8822, 0x8828, 0x882a, 0x8841, 0x8845,
    0x8851, 0x8854, 0x8859, 0x8865, 0x8869, 0x8880, 0x8882, 0x8888, 0x888a, 0x8895,
    0x88a0, 0x88a2, 0x88a8, 0x88aa, 0x8905, 0x8906, 0x8911, 0x8914, 0x8916, 0x8925,
    0x8941, 0x8944, 0x8946, 0x8949, 0x8950, 0x8952, 0x8955, 0x895a, 0x8961, 0x8964,
    0x8985, 0x8996, 0x8999, 0x89a5, 0x8a00, 0x8a02, 0x8a08, 0x8a0a, 0x8a15, 0x8a20,
    0x8a22, 0x8a28, 0x8a2a, 0x8a45, 0x8a51, 0x8a54, 0x8a56, 0x8a80, 0x8a82, 0x8a88,
    0x8a8a, 0x8a95, 0x8aa0, 0x8aa2, 0x8aa8, 0x8aaa, 0x9005, 0x9011, 0x9016, 0x9018,
    0x9019, 0x9025, 0x9041, 0x9046, 0x9049, 0x9055, 0x9058, 0x905a, 0x9069, 0x906a,
    0x9085, 0x9091, 0x9094, 0x9096, 0x9099, 0x90a5, 0x9101, 0x9104, 0x9106, 0x9109,
    0x9110, 0x9115, 0x9118, 0x911a, 0x9121, 0x9124, 0x9126, 0x9129, 0x9140, 0x9145,
    0x9150, 0x9151, 0x9154, 0x9155, 0x9156, 0x9159, 0x9162, 0x9165, 0x9184, 0x9186,
    0x9192, 0x9195, 0x9198, 0x91a1, 0x91a4, 0x91a6, 0x91a9, 0x9205, 0x9211, 0x9214,
    0x9219, 0x9225, 0x9244, 0x9246, 0x9249, 0x9250, 0x9252, 0x9255, 0x9258, 0x9266,
    0x9269, 0x9285, 0x9294, 0x9296, 0x92a9, 0x9401, 0x9404, 0x9406, 0x9410, 0x9415,
    0x9418, 0x9426, 0x9440, 0x944a, 0x9451, 0x9454, 0x9455, 0x9456, 0x9458, 0x9459,
    0x9460, 0x9461, 0x9462, 0x9465, 0x9484, 0x9486, 0x9492, 0x9494, 0x9495, 0x9498,
    0x94a1, 0x94a9, 0x9500, 0x9505, 0x9508, 0x950a, 0x9510, 0x9511, 0x9514, 0x9515,
    0x9516, 0x9519, 0x9521, 0x9525, 0x9529, 0x952a, 0x9541, 0x9544, 0x9545, 0x9546,
    0x9549, 0x9550, 0x9551, 0x9552, 0x9554, 0x9555, 0x9556, 0x9558, 0x9559, 0x955a,
    0x9561, 0x9564, 0x9565, 0x9566, 0x9569, 0x9581, 0x9585, 0x9588, 0x9591, 0x9592,
    0x9594, 0x9595, 0x9596, 0x9599, 0x959a, 0x95a0, 0x95a2, 0x95a5, 0x95a8, 0x95aa,
    0x9601, 0x9604, 0x9610, 0x9615, 0x9619, 0x9620, 0x9626, 0x9629, 0x9645, 0x9648,
    0x9649, 0x9651, 0x9652, 0x9655, 0x9656, 0x9659, 0x9665, 0x9668, 0x9682, 0x9684,
    0x9689, 0x968a, 0x9692, 0x9694, 0x9695, 0x96a4, 0x96a6, 0x96a9, 0x9805, 0x9816,
    0x9819, 0x9825, 0x9841, 0x9846, 0x9850, 0x9852, 0x9855, 0x9856, 0x985a, 0x9864,
    0x9865, 0x9885, 0x9891, 0x9896, 0x9899, 0x98a5, 0x9904, 0x9906, 0x9909, 0x9910,
    0x9912, 0x9915, 0x9918, 0x991a, 0x9920, 0x9921, 0x9924, 0x9926, 0x9940, 0x9942,
    0x9945, 0x9948, 0x994a, 0x9951, 0x9954, 0x9955, 0x9956, 0x9959, 0x9962, 0x9965,
    0x9966, 0x996a, 0x9981, 0x9984, 0x9990, 0x9992, 0x9995, 0x999a, 0x99a1, 0x99a6,
    0x9a05, 0x9a15, 0x9a25, 0x9a44, 0x9a46, 0x9a49, 0x9a50, 0x9a55, 0x9a58, 0x9a61,
    0x9a85, 0x9a91, 0x9a94, 0x9a95, 0x9a96, 0xa000, 0xa002, 0xa008, 0xa00a, 0xa015,
    0xa020, 0xa022, 0xa028, 0xa02a, 0xa045, 0xa051, 0xa054, 0xa056, 0xa059, 0xa080,
    0xa082, 0xa088, 0xa08a, 0xa095, 0xa0a0, 0xa0a2, 0xa0a8, 0xa0aa, 0xa105, 0xa109,
    0xa111, 0xa114, 0xa116, 0xa119, 0xa11a, 0xa146, 0xa149, 0xa151, 0xa155, 0xa158,
    0xa15a, 0xa161, 0xa164, 0xa185, 0xa190, 0xa192, 0xa196, 0xa199, 0xa202, 0xa208,
    0xa20a, 0xa210, 0xa219, 0xa222, 0xa228, 0xa22a, 0xa245, 0xa251, 0xa256, 0xa259,
    0xa265, 0xa280, 0xa282, 0xa288, 0xa28a, 0xa295, 0xa2a0, 0xa2a2, 0xa2a8, 0xa2aa,
    0xa419, 0xa425, 0xa441, 0xa444, 0xa450, 0xa454, 0xa455, 0xa458, 0xa45a, 0xa461,
    0xa465, 0xa466, 0xa468, 0xa469, 0xa485, 0xa506, 0xa509, 0xa510, 0xa512, 0xa515,
    0xa518, 0xa526, 0xa529, 0xa542, 0xa545, 0xa551, 0xa554, 0xa555, 0xa556, 0xa559,
    0xa565, 0xa56a, 0xa581, 0xa584, 0xa585, 0xa586, 0xa589, 0xa592, 0xa595, 0xa598,
    0xa605, 0xa611, 0xa616, 0xa61a, 0xa621, 0xa625, 0xa644, 0xa646, 0xa64a, 0xa652,
    0xa655, 0xa656, 0xa658, 0xa660, 0xa662, 0xa686, 0xa690, 0xa695, 0xa696, 0xa699,
    0xa6a1, 0xa6a4, 0xa6a6, 0xa800, 0xa802, 0xa808, 0xa80a, 0xa820, 0xa822, 0xa828,
    0xa82a, 0xa851, 0xa854, 0xa856, 0xa859, 0xa880, 0xa882, 0xa888, 0xa88a, 0xa895,
    0xa8a0, 0xa8a2, 0xa8a8, 0xa8aa, 0xa905, 0xa914, 0xa919, 0xa921, 0xa925, 0xa941,
    0xa950, 0xa955, 0xa95a, 0xa961, 0xa966, 0xa969, 0xa990, 0xa996, 0xaa00, 0xaa02,
    0xaa08, 0xaa0a, 0xaa20, 0xaa22, 0xaa28, 0xaa2a, 0xaa51, 0xaa54, 0xaa56, 0xaa80,
    0xaa82, 0xaa88, 0xaa8a, 0xaa95, 0xaaa0, 0xaaa2, 0xaaa8, 0xaaaa,
];
