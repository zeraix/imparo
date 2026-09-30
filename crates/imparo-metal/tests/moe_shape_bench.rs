//! IS THE GEMM SHAPE WORTH BUILDING FOR THE ROUTED MATMUL? One expert's real shape, the two
//! kernels, the same weights and the same activations.
//!
//! The routed matmul is 78.9% of a 512-token prefill chunk and runs at 1.15 TMAC/s, where
//! matmat_prefill -- our own dense GEMM, measured in the same run -- reaches 2.23. Giving the
//! routed path the GEMM's shape means padding each expert's rows to a tile boundary, gathering
//! the activations by `perm`, and a per-tile expert id: a real build. So before building it,
//! measure whether the GEMM actually wins AT ONE EXPERT'S SHAPE, which is a thin 64-row
//! problem and not the 512-row one the dense number came from.
//!
//! Both arms read the SAME row-major Q4_K weights, so this isolates the kernel shape and not
//! the layout.
//!
//!   dense    matmat_from(n_tok = 64)      the prefill GEMM over 64 contiguous rows
//!   routed   moe_grouped(1 expert)        the rows kernel over the same 64 work rows
//!
//! Run: cargo test --release -p imparo-metal --test moe_shape_bench -- --ignored --nocapture
#![cfg(target_os = "macos")]

use imparo_backend::{Backend as _, BufId};
#[allow(unused_imports)]
use imparo_gguf as _;
use imparo_metal::MetalBackend;

const FIXTURE: &[u8] = include_bytes!("../../imparo-cpu/tests/data/quant_rows.bin");
const PAGE: usize = 16384;
const ALIGN: usize = 256;
const Q4_K_GGML: u32 = 12;
const Q4_K_KIND: u32 = 5;
/// Q8_0: ggml type 8, wire kind 2. IMPARO_BENCH_Q8=1 runs the weights as Q8_0 instead, which
/// is the ONLY format with two GEMM designs (IMPARO_Q8_DESIGN 0 = staged, >=1 = register
/// tiled) -- so it is how "does staging win at a THIN shape" gets answered without writing a
/// staged k-quant GEMM first.
const Q8_0_GGML: u32 = 8;
const Q8_0_KIND: u32 = 2;
fn use_q8() -> bool {
    std::env::var("IMPARO_BENCH_Q8").is_ok_and(|v| v == "1")
}

/// One expert of LFM2.5-8B-A1B's gate projection, and the work rows it holds at a 512-token
/// chunk: 2048 tokens x 4 picks over 32 experts is 64 rows each.
const N_IN: usize = 2048;
const N_OUT: usize = 1792;
/// Work rows per expert. 64 is what a 512-token chunk gives (2048 work rows / 32 experts);
/// IMPARO_BENCH_ROWS sweeps it to separate 'this kernel is wrong' from 'this shape is thin'.
fn rows() -> usize {
    std::env::var("IMPARO_BENCH_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64)
}
const ITERS: usize = 20;

fn u32_at(b: &[u8], o: usize) -> usize {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]) as usize
}

/// Every fixture block of `ggml`, concatenated. The same reader moe_routed_ffn.rs uses --
/// the fixture is a header of (kind, n_elems, n_bytes) triples, not a flat table.
fn fixture_blocks(ggml: u32) -> Vec<u8> {
    assert_eq!(&FIXTURE[0..4], b"IQFX", "fixture magic");
    let cases = u32_at(FIXTURE, 4);
    let mut off = 8;
    let mut out = Vec::new();
    for _ in 0..cases {
        let (kind, n_elems, n_bytes) = (
            u32_at(FIXTURE, off),
            u32_at(FIXTURE, off + 4),
            u32_at(FIXTURE, off + 8),
        );
        off += 12;
        if kind as u32 == ggml {
            out.extend_from_slice(&FIXTURE[off..off + n_bytes]);
        }
        off += n_bytes + n_elems * 4;
    }
    assert!(
        !out.is_empty(),
        "the fixture has no case of ggml type {ggml}"
    );
    out
}

fn mix(a: usize, b: usize) -> u32 {
    let mut h = 0x9e37_79b9_u32 ^ (a as u32).wrapping_mul(0x85eb_ca6b);
    h ^= (b as u32).wrapping_mul(0xc2b2_ae35);
    h ^= h >> 15;
    h.wrapping_mul(0x27d4_eb2f)
}

#[test]
#[ignore = "requires a physical Metal device; run explicitly with --ignored"]
fn the_gemm_shape_against_the_rows_shape_at_one_experts_size() {
    let rows_n = rows();
    let q8 = use_q8();
    let blocks = fixture_blocks(if q8 { Q8_0_GGML } else { Q4_K_GGML });
    let bb = if q8 { 34usize } else { 144usize };
    let nblk = blocks.len() / bb;
    assert!(nblk > 0, "the fixture carries no whole Q4_K block");
    let blocks_per_row = N_IN / if q8 { 32 } else { 256 };

    // One expert's slice, row-major: every row's blocks picked by a hash so no two rows match.
    let mut w = vec![0u8; N_OUT * blocks_per_row * bb];
    for r in 0..N_OUT {
        for b in 0..blocks_per_row {
            let src = (mix(r, b) as usize % nblk) * bb;
            let dst = (r * blocks_per_row + b) * bb;
            w[dst..dst + bb].copy_from_slice(&blocks[src..src + bb]);
        }
    }
    let w_off = ALIGN; // never offset 0: the backend reads that as "this weight is absent"
    let len = (w_off + w.len()).div_ceil(PAGE) * PAGE;
    let layout = std::alloc::Layout::from_size_align(len, PAGE).unwrap();
    // Leaked on purpose: Metal wraps the mapping without copying and it must outlive the run.
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!base.is_null());
    let bytes = unsafe { std::slice::from_raw_parts_mut(base, len) };
    bytes[w_off..w_off + w.len()].copy_from_slice(&w);
    imparo_metal::set_weight_kind_types(&imparo_gguf::weights::wire_kind_types());
    unsafe { imparo_metal::init(base.cast_const(), len as u64) }.unwrap();
    let be = MetalBackend;

    // Activations: rows_n rows of N_IN, and the permutation that sends work row i to row i.
    let x: Vec<f32> = (0..rows_n * N_IN)
        .map(|i| ((mix(i, 7) >> 9) as f32 / 8.388_608e6) - 1.0)
        .collect();
    let perm: Vec<u32> = (0..rows_n as u32).collect();
    // One expert, holding every row, and it is the only active one.
    let seg: Vec<u32> = vec![0, rows_n as u32, 1, 0];

    let f = |n: usize| (n * 4) as u64;
    be.alloc(BufId::X, f(rows_n * N_IN)).unwrap();
    be.alloc(BufId::G, f(rows_n * N_OUT)).unwrap();
    be.alloc(BufId::Model5, f(rows_n)).unwrap();
    // seg: n_expert+1 offsets, the active count, then the active ids (2*n_expert+2).
    be.alloc(BufId::Model7, f(4)).unwrap();
    be.write(BufId::X, 0, &x);
    be.write_u32(BufId::Model5, 0, &perm);
    be.write_u32(BufId::Model7, 0, &seg);

    let mut best = [f64::MAX; 2];
    for round in 0..3 {
        // ---- the dense GEMM over rows_n contiguous rows ----------------------------------
        be.begin();
        for _ in 0..ITERS {
            imparo_metal::matmat_from(
                if q8 { Q8_0_KIND } else { Q4_K_KIND },
                w_off as u64,
                N_IN as u32,
                N_OUT as u32,
                BufId::X as u32,
                BufId::G as u32,
                rows_n as u32,
                0,
            );
        }
        let t0 = std::time::Instant::now();
        be.end().unwrap();
        let dense = t0.elapsed().as_secs_f64() * 1e3 / ITERS as f64;

        // ---- the routed rows kernel over the same rows_n work rows, one expert ------------
        // The routed kernel is built only for the formats the MoE model carries, so the Q8
        // arm -- which exists to compare the two DENSE designs at a thin shape -- skips it.
        if q8 {
            if round > 0 {
                best[0] = best[0].min(dense);
                best[1] = 0.0;
            }
            continue;
        }
        be.begin();
        for _ in 0..ITERS {
            assert!(be.moe_grouped(
                if q8 { Q8_0_KIND } else { Q4_K_KIND },
                w_off as u64,
                (N_OUT * blocks_per_row * bb) as u64,
                BufId::X,
                BufId::G,
                BufId::Model5,
                BufId::Model7,
                N_IN as u32,
                N_OUT as u32,
                1,
                rows_n as u32,
                rows_n as u32,
                false,
                false,
            ));
        }
        let t1 = std::time::Instant::now();
        be.end().unwrap();
        let routed = t1.elapsed().as_secs_f64() * 1e3 / ITERS as f64;

        if round > 0 {
            best[0] = best[0].min(dense);
            best[1] = best[1].min(routed);
        }
    }

    let gmac = (rows_n * N_IN * N_OUT) as f64 / 1e9;
    eprintln!(
        "{} one expert {N_IN}->{N_OUT}, {rows_n} rows, {gmac:.3} GMAC, best of 2 rounds x {ITERS}:\n  \
         dense GEMM  {:.4} ms  {:.2} TMAC/s\n  \
         routed rows {:.4} ms  {:.2} TMAC/s\n  \
         the GEMM shape is {:.2}x the rows shape here",
        if q8 { "Q8_0 " } else { "Q4_K " },
        best[0],
        gmac / best[0],
        best[1],
        gmac / best[1],
        best[1] / best[0],
    );
}
