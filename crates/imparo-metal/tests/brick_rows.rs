//! The MSL decode brick against the CPU row codec, per format.
//!
//! `tm_sub32` in `native/imparo.metal` is a TRANSCRIPTION of the same block interiors
//! `crates/imparo-cpu/src/quants.rs` implements, and that Rust codec is pinned bit-exact
//! against llama.cpp's own dequantiser (`imparo-cpu/tests/quant_rows.rs`, worst relative
//! error 0e0). So the shader can be checked against it directly, one format at a time,
//! instead of being inferred from whether a whole model produces sensible logits.
//!
//! Why this test exists: a wrong nibble pairing or a flipped sign in one arm reads as
//! plausible logits, never as a crash. Without this gate the only signal is "the model
//! disagrees with llama.cpp", which names no format, no sub-block and no line.
//!
//! The brick has two outputs and both are checked. Half is what the prefill GEMM stages for
//! its half matrix multiply, so its tolerance is half's rounding. Float is what the decode
//! GEMV, the row gather and the mega-kernel unit stage, so its tolerance is f32's.
//!
//! The second test is about precision, not transcription. The brick may round a weight to
//! half ONCE, at the end; it may not round an intermediate and then scale it back up.
//! llama.cpp's Metal dequantisers for Q4_0, Q4_1 and Q4_K do that (`xb->d / 16.h`), and the
//! error only shows where the block scale is small enough for d / 16 to be subnormal in half
//! AND the block's min is small too. Trained rows have such blocks
//! (docs/dequant-precision.md); the fixture's random bytes almost never do, because a random
//! min is usually large and the final rounding at its magnitude hides the intermediate's
//! error. So that test rewrites every block's scale fields to small values first.

use imparo_cpu::quants::row_codec;

const FIXTURE: &[u8] = include_bytes!("../../imparo-cpu/tests/data/quant_rows.bin");

/// Every format `tm_sub32` has an arm for. The multi-span formats (Q2_K 10, IQ2_XS 17,
/// IQ3_XXS 18, IQ3_S 21, IQ2_S 22) are here too: the probe takes the scale run and the payload run
/// separately, which is the one shape that can express a format whose scales are two or
/// three spans of the source block.
const BRICK_FORMATS: &[u32] = &[
    3, 6, 7, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 29,
];

fn u32_at(b: &[u8], o: usize) -> usize {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]) as usize
}

/// The fixture's cases of brick formats: (case index, ggml type, element count, block bytes).
fn brick_cases() -> Vec<(usize, u32, usize, &'static [u8])> {
    assert_eq!(&FIXTURE[0..4], b"IQFX", "fixture magic");
    let cases = u32_at(FIXTURE, 4);
    let mut off = 8;
    let mut out = Vec::new();
    for case in 0..cases {
        let (kind, n_elems, n_bytes) = (
            u32_at(FIXTURE, off),
            u32_at(FIXTURE, off + 4),
            u32_at(FIXTURE, off + 8),
        );
        off += 12;
        let raw = &FIXTURE[off..off + n_bytes];
        off += n_bytes;
        off += n_elems * 4; // the reference floats: the CPU codec is the reference here
        let kind = kind as u32;
        if BRICK_FORMATS.contains(&kind) {
            out.push((case, kind, n_elems, raw));
        }
    }
    out
}

/// Decode `raw` with the Metal brick's half output (`half`) or float output and with the CPU
/// codec, assert they agree to that output's rounding, and return the worst
/// sub-block-relative error. `case` names the case in failures.
fn check_brick(kind: u32, raw: &[u8], n_elems: usize, case: &str, half: bool) -> f32 {
    let codec = row_codec(kind).expect("a brick format has a row codec");
    let mut want = vec![0.0_f32; n_elems];
    codec(raw, &mut want);

    // Split each block into its scale run and payload run, by the SAME rule table the
    // repack moves bytes with. A second copy of the layout here would be a second
    // chance to disagree with the first, and the disagreement reads as a slightly
    // wrong weight rather than a failure.
    let rule = imparo_gguf::weights::tm_rule_for(kind)
        .expect("a brick format has a tile-major rule");
    let bb = rule.block_bytes();
    let n_bytes = raw.len();
    assert_eq!(
        n_bytes % bb,
        0,
        "ggml type {kind}: {n_bytes} is not whole blocks"
    );
    let (mut scales, mut payload) = (Vec::new(), Vec::new());
    for blk in raw.chunks_exact(bb) {
        for &(off, len) in rule.scale_spans {
            scales.extend_from_slice(&blk[off..off + len]);
        }
        for (off, len) in rule.payload_spans() {
            payload.extend_from_slice(&blk[off..off + len]);
        }
    }

    let got = match imparo_metal::decode_probe_for_tests(
        kind, &scales, &payload, n_elems, half,
    ) {
        Ok(v) => v,
        Err(rc) => panic!("case {case} (ggml type {kind}): decode probe rc={rc}"),
    };

    // THE ERROR MEASURE IS PER SUB-BLOCK, NOT PER ELEMENT. An affine format computes
    // `scale * q - min`, so when two similar halves cancel the RESULT is small while
    // the rounding that produced it is set by the operands: Q4_K reaches 2.3e-3
    // relative-to-result on a legitimately correct decode. Half's error is bounded
    // relative to the magnitudes in the expression, so that is what to divide by --
    // the largest value the sub-block's own scale can produce. A real transcription
    // defect (a swapped nibble, a wrong sub-scale, a flipped sign) moves an element by
    // ORDER of that scale, so this still separates the two by a thousand-fold.
    let scale_of = |sub: usize| -> f32 {
        want[sub * 32..(sub + 1) * 32]
            .iter()
            .fold(1e-3_f32, |a, v| a.max(v.abs()))
    };
    let mut worst = 0.0_f32;
    let mut worst_at = 0;
    let mut unrepresentable = 0_usize;
    for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
        // The half output cannot hold a value past half's range. The synthetic cases are random BYTES, which puts a
        // few blocks' scales far outside anything a trained weight holds; those
        // elements say nothing about the transcription. Real rows never reach here.
        if half && (w.abs() > 65504.0 || !w.is_finite()) {
            unrepresentable += 1;
            assert!(
                !g.is_finite() || g.abs() > 60000.0,
                "ggml type {kind}, case {case} element {i}: reference {w} is outside \
                 half, so the shader should saturate -- it returned {g}"
            );
            continue;
        }
        // half carries ~11 bits of mantissa, so a relative 1e-3 is the format's own
        // rounding; the float output's 1e-5 leaves room only for the codec and the shader
        // associating the same terms differently. Anything past that is a transcription
        // defect, and the element index names the sub-block: element i sits in sub-block i / 32.
        let err = (g - w).abs() / scale_of(i / 32);
        if err > worst {
            worst = err;
            worst_at = i;
        }
    }
    let (output, bound) = if half {
        ("half", 1e-3_f32)
    } else {
        ("float", 1e-5_f32)
    };
    assert!(
        worst < bound,
        "ggml type {kind}, case {case}, {output} output: worst sub-block-relative error \
         {worst:e} at element {worst_at} (sub-block {}, lane {}) -- shader {} vs codec {}",
        worst_at / 32,
        worst_at % 32,
        got[worst_at],
        want[worst_at],
    );
    if unrepresentable > 0 {
        println!(
            "  ggml type {kind}, case {case}: {unrepresentable} of {n_elems} \
                  elements are outside half's range (synthetic block bytes)"
        );
    }
    worst
}

#[test]
fn the_metal_brick_decodes_every_format_the_cpu_codec_does() {
    let mut checked: Vec<(u32, f32)> = Vec::new();
    for (case, kind, n_elems, raw) in brick_cases() {
        for half in [true, false] {
            let worst = check_brick(kind, raw, n_elems, &case.to_string(), half);
            checked.push((kind, worst));
        }
    }

    let mut kinds: Vec<u32> = checked.iter().map(|&(k, _)| k).collect();
    kinds.sort_unstable();
    kinds.dedup();
    assert_eq!(
        kinds, BRICK_FORMATS,
        "the fixture must cover every format the brick claims to read"
    );
    let worst = checked.iter().fold(0.0_f32, |a, &(_, e)| a.max(e));
    println!(
        "brick vs codec: {} cases, {} formats, half and float outputs, worst \
              sub-block-relative error {worst:e}",
        checked.len() / 2,
        kinds.len()
    );
}

/// A block row split into its scale runs and payload runs by the repack's own rule table.
fn split_runs(kind: u32, raw: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let rule = imparo_gguf::weights::tm_rule_for(kind)
        .expect("a brick format has a tile-major rule");
    let (mut scales, mut payload) = (Vec::new(), Vec::new());
    for blk in raw.chunks_exact(rule.block_bytes()) {
        for &(off, len) in rule.scale_spans {
            scales.extend_from_slice(&blk[off..off + len]);
        }
        for (off, len) in rule.payload_spans() {
            payload.extend_from_slice(&blk[off..off + len]);
        }
    }
    (scales, payload)
}

/// THE RUN FETCH'S GATE. `tm_run8` gives a rows-matmul lane the eight values it holds in the
/// 8x8 fragments; it must give each the float `tm_sub32` gives it, bit for bit, in both the
/// byte-load and the aligned word-load forms -- a rows kernel may add its products in its own
/// order, but a weight it decodes differently is a transcription defect, not rounding. The
/// pair fetch (`tm_run8_pair`, Q4_K and Q5_K: both sub-blocks of a pair from one load) and the
/// half-block fetch (`tm_quad_scales` with `tm_run8_pair_q` / `tm_run8_q`, Q4_K, Q5_K and Q3_K:
/// four sub-blocks' scales unpacked together) are held to the same rule. The probe writes each of the four column pairs' values back at its k, so a
/// value that lands at the wrong k shows here as a mismatch too. The two forms of `tm_sub32`
/// itself are held to the same rule.
#[test]
fn the_run_fetch_gives_every_value_the_bricks_float_bits() {
    let mut formats: Vec<u32> = Vec::new();
    let mut values = 0_usize;
    for (case, kind, n_elems, raw) in brick_cases() {
        let (scales, payload) = split_runs(kind, raw);
        let probe = |mode: u32| {
            imparo_metal::decode_probe_mode_for_tests(
                kind, &scales, &payload, n_elems, mode,
            )
            .unwrap_or_else(|rc| {
                panic!(
                    "case {case} (ggml type {kind}): decode probe mode {mode} rc={rc}"
                )
            })
        };
        let want = probe(0);
        for (mode, what) in [
            (4_u32, "tm_sub32, word loads"),
            (2, "tm_run8, byte loads"),
            (6, "tm_run8, word loads"),
            (
                14,
                "tm_run8_pair (Q4_K, Q5_K; tm_run8 elsewhere), word loads",
            ),
            (
                22,
                "tm_quad_scales + the half-block fetch (Q4_K, Q5_K, Q3_K; tm_run8 elsewhere)",
            ),
        ] {
            let got = probe(mode);
            for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    g.to_bits() == w.to_bits(),
                    "ggml type {kind}, case {case}, {what}: element {i} (sub-block {}, k {}) \
                     is {g:e}, tm_sub32 gives {w:e}",
                    i / 32,
                    i % 32
                );
            }
        }
        values += n_elems;
        formats.push(kind);
    }
    formats.sort_unstable();
    formats.dedup();
    assert_eq!(formats, BRICK_FORMATS, "every brick format has a run fetch");
    println!(
        "run fetch vs brick: {} formats, {values} values, bit-equal in both load forms, the \
         pair fetch and the half-block fetch",
        formats.len()
    );
}

/// Half bit patterns of three block scales, d = 1025 * 2^-21, 1030 * 2^-22 and 1028 * 2^-23.
/// Each is small enough that d / 16 is below half's smallest normal value (2^-14), and each
/// puts d / 16 exactly halfway between two subnormals, so a shader that divides the scale in
/// half and multiplies back up is off by 16 * q * 2^-25 -- several units of half's final
/// rounding at these magnitudes -- while a shader that rounds once is within one.
const SMALL_SCALES: [u16; 3] = [0x1001, 0x0C06, 0x0804];

/// The f16 scale fields of each brick format's SOURCE block (ggml's layout, the offsets the
/// codec reads), each as (byte offset, power of two times d, negated). An affine legacy
/// block's min is set to -2^(bits-1) d so its weights straddle zero the way a trained row's
/// do; a k-quant's min scale is set equal to d. IQ1_M has no field for d (its sixteen bits
/// are the top nibbles of the four scale words at 48..56) and is written by `set_iq1m_scale`.
const SCALE_FIELDS: &[(u32, &[(usize, u16, bool)])] = &[
    (3, &[(0, 0, false), (2, 3, true)]), // Q4_1: d, m = -8 d
    (6, &[(0, 0, false)]),               // Q5_0: d
    (7, &[(0, 0, false), (2, 4, true)]), // Q5_1: d, m = -16 d
    (10, &[(80, 0, false), (82, 0, false)]), // Q2_K: d, dmin = d
    (11, &[(108, 0, false)]),            // Q3_K: d
    (12, &[(0, 0, false), (2, 0, false)]), // Q4_K: d, dmin = d
    (13, &[(0, 0, false), (2, 0, false)]), // Q5_K: d, dmin = d
    (14, &[(208, 0, false)]),            // Q6_K: d
    (16, &[(0, 0, false)]),              // IQ2_XXS
    (17, &[(0, 0, false)]),              // IQ2_XS
    (18, &[(0, 0, false)]),              // IQ3_XXS
    (19, &[(0, 0, false)]),              // IQ1_S
    (20, &[(0, 0, false)]),              // IQ4_NL
    (21, &[(0, 0, false)]),              // IQ3_S
    (22, &[(0, 0, false)]),              // IQ2_S
    (23, &[(0, 0, false)]),              // IQ4_XS
];

/// `bits` (a normal half) times 2^shift, negated when asked: only the exponent field moves.
fn scaled_half(bits: u16, shift: u16, negate: bool) -> u16 {
    let exp = (bits >> 10) & 0x1F;
    assert!(
        exp > 0 && exp + shift < 31,
        "a normal half that stays normal"
    );
    (u16::from(negate) << 15) | ((exp + shift) << 10) | (bits & 0x3FF)
}

/// IQ1_M's d: bits 4k..4k+3 are the top nibble of the scale word at 48 + 2k.
fn set_iq1m_scale(blk: &mut [u8], bits: u16) {
    for k in 0..4 {
        let o = 48 + 2 * k;
        let word = u16::from_le_bytes([blk[o], blk[o + 1]]);
        let word = (word & 0x0FFF) | (((bits >> (4 * k)) & 0xF) << 12);
        blk[o..o + 2].copy_from_slice(&word.to_le_bytes());
    }
}

#[test]
fn the_metal_brick_rounds_once_where_the_block_scale_is_small() {
    let mut checked = 0_usize;
    let mut worst = 0.0_f32;
    for (case, kind, n_elems, raw) in brick_cases() {
        let bb = imparo_gguf::weights::tm_rule_for(kind)
            .expect("a brick format has a tile-major rule")
            .block_bytes();
        let mut small = raw.to_vec();
        for (b, blk) in small.chunks_exact_mut(bb).enumerate() {
            let d = SMALL_SCALES[b % SMALL_SCALES.len()];
            if kind == 29 {
                set_iq1m_scale(blk, d);
                continue;
            }
            let fields = SCALE_FIELDS
                .iter()
                .find(|(k, _)| *k == kind)
                .unwrap_or_else(|| panic!("ggml type {kind} has no scale-field entry"))
                .1;
            for &(off, shift, negate) in fields {
                blk[off..off + 2]
                    .copy_from_slice(&scaled_half(d, shift, negate).to_le_bytes());
            }
        }
        for half in [true, false] {
            let case = format!("{case} at small scales");
            worst = worst.max(check_brick(kind, &small, n_elems, &case, half));
        }
        checked += 1;
    }
    println!(
        "brick vs codec at small block scales: {checked} cases, worst sub-block-relative \
              error {worst:e}"
    );
}
