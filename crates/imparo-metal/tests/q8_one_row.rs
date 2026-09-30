//! The one-row Q8_0 GEMV against a host sum in f64, in both layouts -- row-major Q8_0 (kind 2) and
//! tile-major Q8_0_TM (kind 3) -- on rows from one K block to 64, under every seat the tuner can
//! set. A row takes the simdgroups its K blocks need when that is fewer than the seat, so the shapes
//! include the widths where a row needs one simdgroup more (9 and 33 blocks row-major, 3 and 17
//! tile-major) and the DSpark drafters' Markov projections: rank 256 to LFM2.5's 128000 and
//! Qwen3.8-27B's 248320 vocabulary entries. Every seat at or above what a row needs dispatches the
//! same simdgroups, and the row-major rows per threadgroup (1, 2 or 4) must not move a row's bits,
//! so those seats must agree bit for bit. One test, because the device is process-global state.
#![cfg(target_os = "macos")]

use imparo_backend::{Backend as _, BufId};
use imparo_gguf::weights::{
    q8_0_tm_convertible, q8_0_tm_payload_offset, q8_0_tm_scale_offset,
};
use imparo_metal::MetalBackend;

const PAGE: usize = 16384;
const BLOCK: usize = 32;
const BLOCK_BYTES: usize = 34;
/// The backend's weight kinds for the two layouts.
const ROW_MAJOR: u32 = 2;
const TILE_MAJOR: u32 = 3;

/// A half-precision float's bits for `v`, exact for the small multiples of 1/256 used here.
fn half_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = ((bits >> 13) & 0x3ff) as u16;
    assert!(exp > 0 && exp < 31, "scale {v} out of half range");
    sign | ((exp as u16) << 10) | mant
}

struct Matrix {
    kind: u32,
    n_in: usize,
    n_out: usize,
    off: usize,
}

impl Matrix {
    fn blocks(&self) -> usize {
        self.n_in / BLOCK
    }

    /// Byte offsets of row `r`'s half scale and first int8 value for K block `b`.
    fn at(&self, r: usize, b: usize) -> (usize, usize) {
        let blocks = self.blocks();
        if self.kind == TILE_MAJOR {
            (
                self.off + q8_0_tm_scale_offset(r, b, blocks, self.n_out),
                self.off + q8_0_tm_payload_offset(r, b, blocks),
            )
        } else {
            let at = self.off + (r * blocks + b) * BLOCK_BYTES;
            (at, at + 2)
        }
    }

    /// The simdgroups the row's K blocks need: a simdgroup's lanes start 8 blocks in the
    /// row-major kernel and 2 in the tile-major one.
    fn sgs_needed(&self) -> u32 {
        let starts = if self.kind == TILE_MAJOR { 2 } else { 8 };
        self.blocks().div_ceil(starts) as u32
    }
}

/// Scale of row `r`, block `b`: a multiple of 1/256 between 1/256 and 13/256.
fn scale(r: usize, b: usize) -> f32 {
    (1 + (r * 7 + b * 3) % 13) as f32 / 256.0
}

fn quant(r: usize, b: usize, i: usize) -> i8 {
    (((r * 31 + b * 17 + i * 7) % 255) as i32 - 127) as i8
}

#[test]
#[ignore = "requires a physical Metal device; run explicitly with --ignored"]
fn one_row_q8_matches_the_host_under_every_seat() {
    let shapes = [
        (ROW_MAJOR, 32_usize, 7_usize),
        (ROW_MAJOR, 64, 4097),
        (ROW_MAJOR, 128, 33),
        (ROW_MAJOR, 256, 1000),
        (ROW_MAJOR, 256, 3001),
        (ROW_MAJOR, 288, 999),
        (ROW_MAJOR, 512, 999),
        (ROW_MAJOR, 1024, 300),
        (ROW_MAJOR, 1056, 77),
        (ROW_MAJOR, 2048, 65),
        (ROW_MAJOR, 256, 128_000),
        (ROW_MAJOR, 256, 248_320),
        (TILE_MAJOR, 32, 8),
        (TILE_MAJOR, 64, 4096),
        (TILE_MAJOR, 96, 1000),
        (TILE_MAJOR, 256, 1000),
        (TILE_MAJOR, 480, 64),
        (TILE_MAJOR, 544, 88),
        (TILE_MAJOR, 2048, 64),
        (TILE_MAJOR, 256, 128_000),
        (TILE_MAJOR, 256, 248_320),
    ];
    let mut matrices = Vec::new();
    let mut len = 0;
    for &(kind, n_in, n_out) in &shapes {
        assert!(kind == ROW_MAJOR || q8_0_tm_convertible(n_in, n_out));
        matrices.push(Matrix {
            kind,
            n_in,
            n_out,
            off: len,
        });
        // Both layouts take the row-major size.
        len += n_out * (n_in / BLOCK) * BLOCK_BYTES;
    }
    let len = len.div_ceil(PAGE) * PAGE;
    let layout = std::alloc::Layout::from_size_align(len, PAGE).unwrap();
    // Leaked on purpose: Metal wraps the mapping without copying, and it must outlive every kernel.
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!base.is_null());
    let bytes = unsafe { std::slice::from_raw_parts_mut(base, len) };
    for m in &matrices {
        for r in 0..m.n_out {
            for b in 0..m.blocks() {
                let (scale_at, values_at) = m.at(r, b);
                bytes[scale_at..scale_at + 2]
                    .copy_from_slice(&half_bits(scale(r, b)).to_le_bytes());
                for i in 0..BLOCK {
                    bytes[values_at + i] = quant(r, b, i) as u8;
                }
            }
        }
    }
    imparo_metal::set_rt(true);
    unsafe { imparo_metal::init(base.cast_const(), len as u64) }.unwrap();
    let be = MetalBackend;
    let (was_sgs, was_rows, was_tm_sgs) = (
        imparo_metal::q8_decode_sgs(),
        imparo_metal::q8_decode_rows(),
        imparo_metal::q8_tm_decode_sgs(),
    );
    for m in &matrices {
        let x: Vec<f32> = (0..m.n_in)
            .map(|i| ((i * 37) % 97) as f32 / 64.0 - 0.75)
            .collect();
        let host: Vec<f64> = (0..m.n_out)
            .map(|r| {
                (0..m.blocks())
                    .map(|b| {
                        f64::from(scale(r, b))
                            * (0..BLOCK)
                                .map(|i| {
                                    f64::from(quant(r, b, i))
                                        * f64::from(x[b * BLOCK + i])
                                })
                                .sum::<f64>()
                    })
                    .sum()
            })
            .collect();
        be.alloc(BufId::X, (m.n_in * 4) as u64).unwrap();
        be.alloc(BufId::O, (m.n_out * 4) as u64).unwrap();
        be.write(BufId::X, 0, &x);
        // Row-major seats are (simdgroups, rows per threadgroup); a tile-major unit fixes 8 rows.
        let seats: Vec<(u32, u32)> = if m.kind == TILE_MAJOR {
            [1, 2, 4, 8, 16].iter().map(|&sgs| (sgs, 8)).collect()
        } else {
            [1, 2, 4, 8]
                .iter()
                .flat_map(|&sgs| [1, 2, 4].map(|rows| (sgs, rows)))
                .collect()
        };
        let (n_seats, mut bit_equal) = (seats.len(), 0);
        let mut first: Option<Vec<u32>> = None;
        for (sgs, rows) in seats {
            if m.kind == TILE_MAJOR {
                imparo_metal::set_q8_tm_decode_sgs(sgs);
            } else {
                imparo_metal::set_q8_decode_sgs(sgs);
                imparo_metal::set_q8_decode_rows(rows);
            }
            be.write(BufId::O, 0, &vec![f32::NAN; m.n_out]);
            be.begin();
            imparo_metal::matmat_from(
                m.kind,
                m.off as u64,
                m.n_in as u32,
                m.n_out as u32,
                BufId::X as u32,
                BufId::O as u32,
                1,
                0,
            );
            be.end().unwrap();
            let mut got = vec![0.0_f32; m.n_out];
            be.read(BufId::O, 0, &mut got);
            for (r, (&g, &h)) in got.iter().zip(&host).enumerate() {
                let delta = (f64::from(g) - h).abs();
                assert!(
                    delta <= 1e-5 * h.abs().max(1.0),
                    "kind={} n_in={} n_out={} sgs={sgs} rows={rows} row={r}: device {g} host {h}",
                    m.kind,
                    m.n_in,
                    m.n_out
                );
            }
            if sgs >= m.sgs_needed() {
                bit_equal += 1;
                let bits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
                match &first {
                    None => first = Some(bits),
                    Some(want) => assert!(
                        *want == bits,
                        "kind={} n_in={} n_out={}: sgs={sgs} rows={rows} changed a row's bits",
                        m.kind,
                        m.n_in,
                        m.n_out
                    ),
                }
            }
        }
        eprintln!(
            "kind={} n_in={} n_out={} sgs_needed={} seats={n_seats} bits_equal_across={bit_equal}",
            m.kind,
            m.n_in,
            m.n_out,
            m.sgs_needed()
        );
    }
    imparo_metal::set_q8_decode_sgs(was_sgs);
    imparo_metal::set_q8_decode_rows(was_rows);
    imparo_metal::set_q8_tm_decode_sgs(was_tm_sgs);
}
