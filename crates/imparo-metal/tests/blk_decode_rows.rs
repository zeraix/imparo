//! Decode rows for the block formats, in each of Qwen3.8-27B's ten block formats, tile-major. On
//! the fast route a co-batched step's rows take the decode-rows GEMV (`imparo_blk_gemv_rows`) up
//! to a crossing that is PER TENSOR when the tune file seats one and per format otherwise, the
//! matrix-unit rows kernel (`imparo_blk_rows_mma`) up to `blk_rows_mma_max`, and the GEMM above
//! both. This test overrides the crossing for every format so it can put them all on one kernel,
//! checks the served per-format value at the end, and checks that a per-tensor seat moves the
//! DISPATCH -- seating a tensor to the opposite of its format's value must flip the route, which
//! is what proves `matmat_impl` reads the seat rather than only the accessor answering with it. Each
//! kernel is checked at every row count: each row within a tolerance of its one-row GEMV value, two
//! runs bit-equal, and the kernel counted as taken by its route name (a route that never ran would
//! pass on the one-row GEMV's values). The exact
//! route takes neither -- their rows differ from their lone decodes in the last bits, and the test
//! prints which formats came out bit-equal -- and keeps every row's one-row bits. B covers each
//! GEMV form (2, 4, 8 tokens) and the padded counts (3 on the 4-token form, 5 on the 8-token one),
//! whose dead rows must not be stored.
//!
//! The weights are the brick fixture's blocks (crates/imparo-cpu/tests/data/quant_rows.bin, the
//! same bytes `brick_rows.rs` checks the decoder on), placed by a hash so the rows differ, and
//! moved to tile-major by the repack's own rule. Every input width gives each simdgroup's slice of
//! a row two or more sub-blocks (8 and 20 at 8 simdgroups): with one, no slice adds two parts, and
//! a sum rounded in another order could not show.
#![cfg(target_os = "macos")]

use imparo_backend::{Backend as _, BufId, RowRoute};
use imparo_metal::MetalBackend;

const FIXTURE: &[u8] = include_bytes!("../../imparo-cpu/tests/data/quant_rows.bin");
const PAGE: usize = 16384;
/// Tensor starts: the repack keeps a GGUF's 32-byte alignment; 256 covers it.
const ALIGN: usize = 256;
const MAX_ROWS: usize = 8;
/// Floats after the outputs that no kernel may write, and the value they hold.
const GUARD: usize = 64;
const SENTINEL: f32 = 12_345.678;

/// Qwen3.8-27B's block formats: (ggml type, the wire kind of its tile-major form, name).
const FORMATS: &[(u32, u32, &str)] = &[
    (11, 20, "Q3_K"),
    (12, 21, "Q4_K"),
    (13, 22, "Q5_K"),
    (14, 23, "Q6_K"),
    (20, 24, "IQ4_NL"),
    (23, 25, "IQ4_XS"),
    (21, 26, "IQ3_S"),
    (17, 30, "IQ2_XS"),
    (18, 31, "IQ3_XXS"),
    (22, 32, "IQ2_S"),
];

/// (n_in, n_out): 64 and 160 sub-blocks per row; a partial last threadgroup (1000 rows, 32 a
/// threadgroup, 8 live lanes in the last) and a narrow projection.
const SHAPES: &[(usize, usize)] = &[(2048, 1000), (5120, 64)];

fn u32_at(b: &[u8], o: usize) -> usize {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]) as usize
}

/// Every fixture block of `ggml`, concatenated.
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

/// How many times the matmul route `name` ran since `routes` was read.
fn grew(routes: &[(&'static str, u64)], name: &str) -> u64 {
    let count = |rs: &[(&'static str, u64)]| {
        rs.iter().find(|(n, _)| *n == name).map_or(0, |&(_, c)| c)
    };
    count(&imparo_metal::matmul_routes()) - count(routes)
}

/// A deterministic hash of three indices, for data that is not exactly representable.
fn mix(a: usize, b: usize, c: usize) -> u32 {
    let mut h = (a as u32).wrapping_mul(0x9E37_79B1)
        ^ (b as u32).wrapping_mul(0x85EB_CA77)
        ^ (c as u32).wrapping_mul(0xC2B2_AE3D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^ (h >> 12)
}

/// Row `t`'s activations: every row different, every value inexact.
fn activation(t: usize, i: usize) -> f32 {
    (mix(t, i, 2) as f32 / u32::MAX as f32) * 2.0 - 1.0
}

/// A row-major tensor of `n_out` rows, `n_in` wide, from the fixture's blocks of `ggml`, then
/// moved to tile-major.
fn tensor(ggml: u32, n_in: usize, n_out: usize) -> Vec<u8> {
    let rule =
        imparo_gguf::weights::tm_rule_for(ggml).expect("a block format has a rule");
    rule.convert(&row_major_tensor(ggml, n_in, n_out), n_in, n_out)
}

/// The same tensor as `tensor`, left row-major (a drafter's tensors and an lm head that the
/// load-time transform does not move).
fn row_major_tensor(ggml: u32, n_in: usize, n_out: usize) -> Vec<u8> {
    let rule =
        imparo_gguf::weights::tm_rule_for(ggml).expect("a block format has a rule");
    assert!(rule.convertible(n_in, n_out));
    let bb = rule.block_bytes();
    let pool = fixture_blocks(ggml);
    let n_pool = pool.len() / bb;
    let blocks = n_in / rule.block_elems;
    let mut rows = vec![0_u8; n_out * blocks * bb];
    for r in 0..n_out {
        for b in 0..blocks {
            let pick = mix(r, b, ggml as usize) as usize % n_pool;
            rows[(r * blocks + b) * bb..][..bb]
                .copy_from_slice(&pool[pick * bb..][..bb]);
        }
    }
    rows
}

/// Runs the projection over `rows` rows starting at activation row `src_row`, into output rows
/// 0..rows, and returns every output row's bits (MAX_ROWS rows, NaN where nothing was stored).
fn project(
    kind: u32,
    off: usize,
    n_in: usize,
    n_out: usize,
    rows: usize,
    src_row: usize,
) -> Vec<u32> {
    let be = MetalBackend;
    be.write(BufId::O, 0, &vec![f32::NAN; MAX_ROWS * n_out]);
    be.begin();
    imparo_metal::matmat_from(
        kind,
        off as u64,
        n_in as u32,
        n_out as u32,
        BufId::X as u32,
        BufId::O as u32,
        rows as u32,
        src_row as u32,
    );
    be.end().unwrap();
    let mut got = vec![0.0_f32; MAX_ROWS * n_out];
    be.read(BufId::O, 0, &mut got);
    got.iter().map(|v| v.to_bits()).collect()
}

/// One shape of one format: every route and row count. Returns the failures as lines.
fn check(
    kind: u32,
    off: usize,
    n_in: usize,
    n_out: usize,
    what: &str,
    routes_too: bool,
) -> Vec<String> {
    let mut failures = Vec::new();
    // The crossings this check moves, restored at its end.
    let (gemv_max0, mma_max0) = (
        imparo_metal::blk_rows_gemv_max(),
        imparo_metal::blk_rows_mma_max(),
    );
    let be = MetalBackend;
    let x: Vec<f32> = (0..MAX_ROWS)
        .flat_map(|t| (0..n_in).map(move |i| activation(t, i)))
        .collect();
    be.alloc(BufId::X, (x.len() * 4) as u64).unwrap();
    be.alloc(BufId::O, ((MAX_ROWS * n_out + GUARD) * 4) as u64)
        .unwrap();
    be.write(BufId::O, (MAX_ROWS * n_out) as u64, &[SENTINEL; GUARD]);
    be.write(BufId::X, 0, &x);
    // Each row alone: the one-row GEMV, reading row t, writing output row 0.
    imparo_metal::set_decode_rows(None);
    let alone: Vec<Vec<u32>> = (0..MAX_ROWS)
        .map(|t| project(kind, off, n_in, n_out, 1, t)[..n_out].to_vec())
        .collect();
    let finite = alone
        .iter()
        .flatten()
        .filter(|&&b| f32::from_bits(b).is_finite())
        .count();
    if finite != MAX_ROWS * n_out {
        failures.push(format!(
            "{what} n_in={n_in} n_out={n_out}: {} one-row outputs are not finite",
            MAX_ROWS * n_out - finite
        ));
        return failures;
    }
    // Each fast-route kernel at every row count: the GEMV with its crossing at 8, then the
    // matrix-unit kernel with the GEMV's at 1 (and so never).
    let mut worst = [0.0_f64; 2];
    let mut exact_bits = true;
    for (k, (kernel, gemv_max, mma_max)) in
        [("blk.gemv_rows", 8, 0), ("blk.rows_mma", 1, 8)]
            .into_iter()
            .enumerate()
    {
        for route in [RowRoute::Exact, RowRoute::Fast] {
            imparo_metal::set_decode_rows(Some(route));
            imparo_metal::set_blk_rows_gemv_max(gemv_max);
            imparo_metal::set_blk_rows_mma_max(mma_max);
            for rows in [2_usize, 3, 4, 5, 8] {
                let routes = imparo_metal::matmul_routes();
                let got = project(kind, off, n_in, n_out, rows, 0);
                let taken = grew(&routes, kernel);
                let want_taken = u64::from(route == RowRoute::Fast);
                if taken != want_taken {
                    failures.push(format!(
                    "{what} {route:?} n_in={n_in} n_out={n_out} rows={rows}: {kernel} ran \
                     {taken} times, not {want_taken}"
                ));
                }
                for (t, want) in alone.iter().enumerate().take(rows) {
                    let row = &got[t * n_out..(t + 1) * n_out];
                    if route == RowRoute::Exact {
                        let differ =
                            row.iter().zip(want).filter(|(g, w)| g != w).count();
                        if differ > 0 {
                            failures.push(format!(
                            "{what} Exact n_in={n_in} n_out={n_out} rows={rows}: row {t} \
                             differs from its one-row bits in {differ} of {n_out} outputs"
                        ));
                        }
                        continue;
                    }
                    exact_bits &= row == want.as_slice();
                    let squares: f64 = want
                        .iter()
                        .map(|&w| f64::from(f32::from_bits(w)).powi(2))
                        .sum();
                    let rms = (squares / n_out as f64).sqrt().max(1e-30);
                    let err = row
                        .iter()
                        .zip(want)
                        .map(|(&g, &w)| {
                            (f64::from(f32::from_bits(g))
                                - f64::from(f32::from_bits(w)))
                            .abs()
                                / rms
                        })
                        .fold(0.0_f64, |a, e| {
                            if e.is_nan() { f64::INFINITY } else { a.max(e) }
                        });
                    worst[k] = worst[k].max(err);
                    if err > 1e-4 {
                        failures.push(format!(
                        "{what} Fast {kernel} n_in={n_in} n_out={n_out} rows={rows}: row {t} \
                         is {err:.2e} of its rms from its one-row GEMV value"
                    ));
                    }
                }
                // Rows that start past activation row 0: a run of a longer step reads rows
                // src_row.. (the second run of 9 rows is rows 5..8).
                if route == RowRoute::Fast && rows <= 4 {
                    let from = MAX_ROWS - rows;
                    let shifted = project(kind, off, n_in, n_out, rows, from);
                    for t in 0..rows {
                        let want = &alone[from + t];
                        let row = &shifted[t * n_out..(t + 1) * n_out];
                        let squares: f64 = want
                            .iter()
                            .map(|&w| f64::from(f32::from_bits(w)).powi(2))
                            .sum();
                        let rms = (squares / n_out as f64).sqrt().max(1e-30);
                        let err = row
                            .iter()
                            .zip(want)
                            .map(|(&g, &w)| {
                                (f64::from(f32::from_bits(g)) - f64::from(f32::from_bits(w)))
                                    .abs()
                                    / rms
                            })
                            .fold(0.0_f64, |a, e| if e.is_nan() { f64::INFINITY } else { a.max(e) });
                        if err > 1e-4 {
                            failures.push(format!(
                                "{what} Fast {kernel} n_in={n_in} n_out={n_out} rows={rows} \
                                 src_row={from}: row {t} is {err:.2e} of its rms from activation \
                                 row {}'s one-row value",
                                from + t
                            ));
                        }
                    }
                }
                if route == RowRoute::Fast
                    && project(kind, off, n_in, n_out, rows, 0) != got
                {
                    failures.push(format!(
                    "{what} Fast n_in={n_in} n_out={n_out} rows={rows}: two runs gave \
                     different bits"
                ));
                }
                let dead = got[rows * n_out..]
                    .iter()
                    .filter(|b| !f32::from_bits(**b).is_nan())
                    .count();
                if dead > 0 {
                    failures.push(format!(
                    "{what} {route:?} n_in={n_in} n_out={n_out} rows={rows}: {dead} outputs \
                     of dead rows were written"
                ));
                }
                let mut guard = vec![0.0_f32; GUARD];
                be.read(BufId::O, (MAX_ROWS * n_out) as u64, &mut guard);
                if guard.iter().any(|v| v.to_bits() != SENTINEL.to_bits()) {
                    failures.push(format!(
                    "{what} {route:?} n_in={n_in} n_out={n_out} rows={rows}: a store landed \
                     past the outputs"
                ));
                }
            }
        }
    }
    eprintln!(
        "{what} n_in={n_in} n_out={n_out}: fast-route rows within {:.2e} (decode-rows GEMV) and \
         {:.2e} (matrix-unit rows) of each row's rms from its one-row GEMV{}",
        worst[0],
        worst[1],
        if exact_bits { ", bit-equal" } else { "" }
    );
    failures.extend(check_runs(kind, off, n_in, n_out, what));
    // The route and seat checks below are the tile-major tensors' (the ones a tune file seats);
    // a row-major tensor is checked for its values only.
    if !routes_too {
        imparo_metal::set_blk_rows_gemv_max(gemv_max0);
        imparo_metal::set_blk_rows_mma_max(mma_max0);
        imparo_metal::set_decode_rows(None);
        return failures;
    }
    // Where the GEMM has no room for its padded tile (these buffers hold 8 rows -- the lm head's
    // case) the rows take the kernel their crossings name, never the one-row kernel looping them:
    // 4 rows past a GEMV crossing of 2 take the matrix-unit kernel, and past a matrix-unit crossing
    // of 3 the decode-rows GEMV.
    imparo_metal::set_decode_rows(Some(RowRoute::Fast));
    imparo_metal::set_blk_rows_gemv_max(2);
    for (mma_max, want) in [(8, "blk.rows_mma"), (3, "blk.gemv_rows")] {
        imparo_metal::set_blk_rows_mma_max(mma_max);
        let routes = imparo_metal::matmul_routes();
        let _ = project(kind, off, n_in, n_out, 4, 0);
        let (taken, looped) = (
            grew(&routes, want),
            grew(&routes, "blk.gemv_tokens_fallback"),
        );
        if taken != 1 || looped != 0 {
            failures.push(format!(
                "{what} n_in={n_in} n_out={n_out}: 4 rows with no room for the GEMM's tile and a \
                 matrix-unit crossing of {mma_max} ran {want} {taken} times and the one-row loop \
                 {looped} times, not once and never"
            ));
        }
    }
    // With room (64 rows of activations and outputs) 4 rows above the crossing take the GEMM.
    be.alloc(BufId::X, (64 * n_in * 4) as u64).unwrap();
    be.alloc(BufId::O, ((64 * n_out + GUARD) * 4) as u64)
        .unwrap();
    let x64: Vec<f32> = (0..64)
        .flat_map(|t| (0..n_in).map(move |i| activation(t % MAX_ROWS, i)))
        .collect();
    be.write(BufId::X, 0, &x64);
    // With room, 4 rows past a GEMV crossing of 2 take the matrix-unit kernel, and past a
    // matrix-unit crossing of 3 the GEMM.
    for (mma_max, want) in [(8, "blk.rows_mma"), (3, "blk.gemm")] {
        imparo_metal::set_blk_rows_mma_max(mma_max);
        let routes = imparo_metal::matmul_routes();
        be.begin();
        imparo_metal::matmat_from(
            kind,
            off as u64,
            n_in as u32,
            n_out as u32,
            BufId::X as u32,
            BufId::O as u32,
            4,
            0,
        );
        be.end().unwrap();
        let ran: Vec<(&str, u64)> = ["blk.gemv_rows", "blk.rows_mma", "blk.gemm"]
            .into_iter()
            .map(|n| (n, grew(&routes, n)))
            .collect();
        if ran.iter().any(|&(n, c)| c != u64::from(n == want)) {
            failures.push(format!(
                "{what} n_in={n_in} n_out={n_out}: 4 rows with room, crossings 2 and {mma_max}, \
                 ran {ran:?}, not {want} once"
            ));
        }
    }
    // A PER-TENSOR SEAT MOVES THE DISPATCH, not just the accessor. With no override the format's
    // crossing decides which kernel 2 rows take; seating THIS tensor to the opposite value must
    // flip the route. Without this, the two call sites in `matmat_impl` could ignore the seat and
    // the query check below would still pass.
    imparo_metal::set_blk_rows_gemv_max(0);
    imparo_metal::set_blk_rows_mma_max(8);
    imparo_metal::clear_blk_rows_seats();
    let fmt_max = imparo_metal::blk_rows_gemv_max_for(kind);
    for (seat, want) in [(2_u32, "blk.gemv_rows"), (1_u32, "blk.rows_mma")] {
        imparo_metal::clear_blk_rows_seats();
        if seat != fmt_max {
            // Only worth asserting where the seat DIFFERS from the format value; where they agree
            // the route would be right either way and the check would have no teeth.
            assert!(imparo_metal::set_blk_rows_seat(
                kind,
                n_in as u32,
                n_out as u32,
                seat
            ));
        }
        let routes = imparo_metal::matmul_routes();
        be.begin();
        imparo_metal::matmat_from(
            kind,
            off as u64,
            n_in as u32,
            n_out as u32,
            BufId::X as u32,
            BufId::O as u32,
            2,
            0,
        );
        be.end().unwrap();
        let ran: Vec<(&str, u64)> = ["blk.gemv_rows", "blk.rows_mma", "blk.gemm"]
            .into_iter()
            .map(|n| (n, grew(&routes, n)))
            .collect();
        if ran.iter().any(|&(n, c)| c != u64::from(n == want)) {
            failures.push(format!(
                "{what} n_in={n_in} n_out={n_out}: 2 rows seated to {seat} (this format's \
                 crossing is {fmt_max}) ran {ran:?}, not {want} once -- the dispatch is not \
                 reading the per-tensor seat"
            ));
        }
    }
    imparo_metal::clear_blk_rows_seats();
    imparo_metal::set_blk_rows_gemv_max(gemv_max0);
    imparo_metal::set_blk_rows_mma_max(mma_max0);
    imparo_metal::set_decode_rows(None);
    // THE SERVED CROSSING, with the override dropped: 2 for the formats measured short of the
    // bandwidth wall, 1 for the rest. A build that still has one crossing for every format
    // answers 1 here for Q3_K, IQ2_XS, IQ3_XXS, IQ3_S and IQ2_S, and fails.
    let want = u32::from(SHORT_OF_WALL.contains(&what)) + 1;
    let got = imparo_metal::blk_rows_gemv_max_for(kind);
    if got != want {
        failures.push(format!(
            "{what}: the served GEMV crossing is {got}, not the {want} measured for this format"
        ));
    }
    // A PER-TENSOR SEAT OVERRIDES THAT FORMAT VALUE, and only for the tensor it names. The seat
    // exists because the two kernels' winner follows the tensor, not the format: on the served
    // 27B the GEMV beats the matrix unit on six of Q4_K_TM's seven shapes and loses on
    // ffn_down 17408->5120 (evidence section 35).
    imparo_metal::clear_blk_rows_seats();
    let before = imparo_metal::blk_rows_seat_count();
    let unseated = imparo_metal::blk_rows_gemv_max_at(kind, n_in as u32, n_out as u32);
    // Pick a value the format's own crossing does NOT already have, so a build that ignores the
    // seat and answers with the format value fails here instead of passing by coincidence.
    let seated_val = if want == 2 { 1 } else { 2 };
    if !imparo_metal::set_blk_rows_seat(kind, n_in as u32, n_out as u32, seated_val) {
        failures.push(format!("{what} {n_in}->{n_out}: the seat was refused"));
    }
    let seated = imparo_metal::blk_rows_gemv_max_at(kind, n_in as u32, n_out as u32);
    // A DIFFERENT shape of the same kind must be untouched: that is the whole point of the key.
    let other =
        imparo_metal::blk_rows_gemv_max_at(kind, n_in as u32, n_out as u32 + 128);
    // AN EXPLICIT OVERRIDE OUTRANKS THE SEAT. The override exists to put every format on ONE
    // kernel so a test or the bench can time that kernel alone; a seat winning here leaves some
    // tensors on the other kernel and labels the mixture as one. That is not hypothetical: with
    // 37 seats loaded, IMPARO_BENCH_BLK_ROWS_GEMV_MAX=2 ran 28 tensors on the GEMV and 38 on the
    // matrix unit while reporting that it had forced the GEMV.
    let forced = if seated_val == 2 { 1 } else { 2 };
    imparo_metal::set_blk_rows_gemv_max(forced);
    let overridden =
        imparo_metal::blk_rows_gemv_max_at(kind, n_in as u32, n_out as u32);
    imparo_metal::set_blk_rows_gemv_max(0);
    if overridden != forced {
        failures.push(format!(
            "{what} {n_in}->{n_out}: a seat of {seated_val} outranked an explicit override of \
             {forced} (answered {overridden}) -- the bench can no longer force one kernel"
        ));
    }
    imparo_metal::clear_blk_rows_seats();
    let after_clear =
        imparo_metal::blk_rows_gemv_max_at(kind, n_in as u32, n_out as u32);
    if unseated != want
        || seated != seated_val
        || other != want
        || after_clear != want
        || before != 0
    {
        failures.push(format!(
            "{what} {n_in}->{n_out}: seat lookup is wrong -- unseated {unseated} (want {want}), \
             seated {seated} (want {seated_val}), a neighbouring shape {other} (want {want}), \
             after clear {after_clear} (want {want}), count before {before} (want 0)"
        ));
    }
    failures
}

/// Rows past one run of the rows kernels (9..16) on the fast route: one two-fragment dispatch of
/// the matrix unit, or runs of at most 8 rows as even as they can be, each row within the
/// tolerance of its one-row value -- where the GEMM has no
/// room for its padded tile (the lm head's case) and where it has (a DSpark drafter's 9-row
/// block). Every run is a rows kernel: no GEMM, no one-row loop.
fn check_runs(kind: u32, off: usize, n_in: usize, n_out: usize, what: &str) -> Vec<String> {
    const RUN_ROWS: usize = 16;
    const ROOM: usize = 64;
    let mut failures = Vec::new();
    let be = MetalBackend;
    let x: Vec<f32> = (0..ROOM)
        .flat_map(|t| (0..n_in).map(move |i| activation(t % RUN_ROWS, i)))
        .collect();
    let run = |rows: usize, out_rows: usize| -> Vec<u32> {
        be.alloc(BufId::O, ((out_rows * n_out + GUARD) * 4) as u64)
            .unwrap();
        be.write(BufId::O, 0, &vec![f32::NAN; out_rows * n_out]);
        be.write(BufId::O, (out_rows * n_out) as u64, &[SENTINEL; GUARD]);
        be.begin();
        imparo_metal::matmat_from(
            kind,
            off as u64,
            n_in as u32,
            n_out as u32,
            BufId::X as u32,
            BufId::O as u32,
            rows as u32,
            0,
        );
        be.end().unwrap();
        let mut got = vec![0.0_f32; out_rows * n_out + GUARD];
        be.read(BufId::O, 0, &mut got);
        got.iter().map(|v| v.to_bits()).collect()
    };
    // Each row alone on the one-row GEMV.
    be.alloc(BufId::X, (x.len() * 4) as u64).unwrap();
    be.write(BufId::X, 0, &x);
    imparo_metal::set_decode_rows(None);
    let alone: Vec<Vec<u32>> = (0..RUN_ROWS)
        .map(|t| {
            be.alloc(BufId::O, ((n_out + GUARD) * 4) as u64).unwrap();
            be.begin();
            imparo_metal::matmat_from(
                kind,
                off as u64,
                n_in as u32,
                n_out as u32,
                BufId::X as u32,
                BufId::O as u32,
                1,
                t as u32,
            );
            be.end().unwrap();
            let mut got = vec![0.0_f32; n_out];
            be.read(BufId::O, 0, &mut got);
            got.iter().map(|v| v.to_bits()).collect()
        })
        .collect();
    imparo_metal::set_decode_rows(Some(RowRoute::Fast));
    for rows in [9_usize, 12, 16] {
        // No room: the outputs hold the step's rows only. Room: a padded tile's worth.
        for (room, out_rows) in [(false, RUN_ROWS), (true, ROOM)] {
            let routes = imparo_metal::matmul_routes();
            let got = run(rows, out_rows);
            let kernels = grew(&routes, "blk.rows_mma") + grew(&routes, "blk.gemv_rows");
            let (gemm, looped) = (
                grew(&routes, "blk.gemm"),
                grew(&routes, "blk.gemv_tokens_fallback"),
            );
            // One dispatch of two 8-row fragments where a run of 8 is the matrix unit's, else
            // the runs.
            let two = 8 > imparo_metal::blk_rows_gemv_max_for(kind)
                && 8 <= imparo_metal::blk_rows_mma_max();
            let want = if two { 1 } else { rows.div_ceil(8) as u64 };
            if kernels != want || gemm != 0 || looped != 0 {
                failures.push(format!(
                    "{what} n_in={n_in} n_out={n_out} rows={rows} room={room}: {kernels} rows-kernel \
                     runs (want {want}), GEMM {gemm}, one-row loop {looped}"
                ));
            }
            for (t, want) in alone.iter().enumerate().take(rows) {
                let row = &got[t * n_out..(t + 1) * n_out];
                let squares: f64 = want
                    .iter()
                    .map(|&w| f64::from(f32::from_bits(w)).powi(2))
                    .sum();
                let rms = (squares / n_out as f64).sqrt().max(1e-30);
                let err = row
                    .iter()
                    .zip(want)
                    .map(|(&g, &w)| {
                        (f64::from(f32::from_bits(g)) - f64::from(f32::from_bits(w))).abs() / rms
                    })
                    .fold(0.0_f64, |a, e| if e.is_nan() { f64::INFINITY } else { a.max(e) });
                if err > 1e-4 {
                    failures.push(format!(
                        "{what} n_in={n_in} n_out={n_out} rows={rows} room={room}: row {t} is \
                         {err:.2e} of its rms from its one-row value"
                    ));
                }
            }
            let written_dead = got[rows * n_out..out_rows * n_out]
                .iter()
                .filter(|b| !f32::from_bits(**b).is_nan())
                .count();
            let guard_hit = got[out_rows * n_out..]
                .iter()
                .any(|&b| b != SENTINEL.to_bits());
            if written_dead > 0 || guard_hit {
                failures.push(format!(
                    "{what} n_in={n_in} n_out={n_out} rows={rows} room={room}: {written_dead} \
                     dead-row outputs written, guard {}",
                    if guard_hit { "overwritten" } else { "intact" }
                ));
            }
        }
    }
    // A RESIDUAL MATMUL (`matmat_resid`: resid += W . x) on the fast route, 4 and 12 rows: the rows
    // kernels store W . x, so they must refuse it -- taking it overwrote the residual. Whichever
    // route serves it, each row must come back as its residual plus its one-row value.
    // Each row's residual is scaled to that row's own W . x, so a kernel that drops the residual
    // is wrong by the whole of it, not by a sliver of a large output.
    let rms_of = |row: &[u32]| -> f64 {
        (row.iter()
            .map(|&w| f64::from(f32::from_bits(w)).powi(2))
            .sum::<f64>()
            / row.len() as f64)
            .sqrt()
            .max(1e-30)
    };
    for rows in [4_usize, 12] {
        let resid: Vec<f32> = (0..ROOM * n_out)
            .map(|i| {
                let scale = alone.get(i / n_out).map_or(1.0, |r| rms_of(r));
                (f64::from(activation(i / n_out + 100, i % n_out)) * scale) as f32
            })
            .collect();
        be.alloc(BufId::O, ((ROOM * n_out + GUARD) * 4) as u64)
            .unwrap();
        be.write(BufId::O, 0, &resid);
        be.begin();
        let taken = imparo_metal::matmat_resid(
            kind,
            off as u64,
            n_in as u32,
            n_out as u32,
            BufId::X as u32,
            BufId::O as u32,
            rows as u32,
        );
        be.end().unwrap();
        let mut got = vec![0.0_f32; rows * n_out];
        be.read(BufId::O, 0, &mut got);
        if !taken {
            // Refused: nothing may have been written.
            if got.iter().zip(&resid).any(|(g, r)| g.to_bits() != r.to_bits()) {
                failures.push(format!(
                    "{what} n_in={n_in} n_out={n_out} rows={rows}: a refused residual matmul \
                     wrote its output"
                ));
            }
            continue;
        }
        for (t, want) in alone.iter().enumerate().take(rows) {
            let scale = rms_of(want);
            let mut worst = 0.0_f64;
            for i in 0..n_out {
                let w = f64::from(resid[t * n_out + i]) + f64::from(f32::from_bits(want[i]));
                worst = worst.max((f64::from(got[t * n_out + i]) - w).abs());
            }
            let err = worst / scale;
            if err.is_nan() || err > 1e-4 {
                failures.push(format!(
                    "{what} n_in={n_in} n_out={n_out} rows={rows}: residual matmul row {t} is \
                     {err:.2e} of its rms from residual + its one-row value"
                ));
            }
        }
    }
    imparo_metal::set_decode_rows(None);
    // Back to the sizes `check` runs with.
    be.alloc(BufId::X, (MAX_ROWS * n_in * 4) as u64).unwrap();
    be.alloc(BufId::O, ((MAX_ROWS * n_out + GUARD) * 4) as u64)
        .unwrap();
    failures
}

/// The formats whose unpack leaves them short of the bandwidth wall, so the matrix unit's products
/// land on top of an already ALU-limited kernel and the decode-rows GEMV is cheaper at 2 rows
/// (docs/evidence/cobatch/2026-09-20-27b-rows-no-tile.md section 24).
///
/// This is the FALLBACK, by format. Section 35 measured the two kernels per tensor and found the
/// winner follows the tensor: Q4_K TILE-MAJOR wants the GEMV where Q4_K ROW-MAJOR wants the
/// matrix unit, and one format's own shapes disagree. A tune file's per-tensor seats override
/// this list; nothing here says the list is the last word.
const SHORT_OF_WALL: [&str; 5] = ["Q3_K", "IQ2_XS", "IQ3_XXS", "IQ3_S", "IQ2_S"];

/// Formats with no row-major one-row GEMV: their one-row outputs are not a reference.
const ROW_MAJOR_UNSERVED: [&str; 4] = ["IQ3_S", "IQ2_XS", "IQ3_XXS", "IQ2_S"];

/// Every format tile-major, then the same checks on ROW-MAJOR weights (the wire kind of the ggml
/// type itself): a DSpark drafter's Q4_K / Q6_K tensors and a target's Q6_K lm head take those
/// arms of the kernels. One test, one mapping: a process initialises Metal once.
#[test]
#[ignore = "requires a physical Metal device; run explicitly with --ignored"]
fn blk_decode_rows_give_every_row_its_one_row_bits() {
    let wire = imparo_gguf::weights::wire_kind_types();
    let mut offs = Vec::new();
    let mut names = Vec::new();
    let mut tensors = Vec::new();
    let mut len = 0;
    // The IQ formats below IQ4 have no row-major one-row GEMV (every one is served tile-major).
    let formats: Vec<_> = [false, true]
        .into_iter()
        .flat_map(|row_major| FORMATS.iter().map(move |f| (row_major, f)))
        .filter(|(row_major, (_, _, name))| !row_major || !ROW_MAJOR_UNSERVED.contains(name))
        .collect();
    for &(row_major, &(ggml, tm_kind, name)) in &formats {
        // A ggml type's row-major wire kind is the one whose ggml type is the type itself.
        let kind = if row_major {
            wire.iter()
                .find(|&&(_, g)| g == ggml)
                .map(|&(w, _)| w)
                .expect("a row-major wire kind for every block format")
        } else {
            tm_kind
        };
        for &(n_in, n_out) in SHAPES {
            let t = if row_major {
                row_major_tensor(ggml, n_in, n_out)
            } else {
                tensor(ggml, n_in, n_out)
            };
            offs.push((kind, n_in, n_out, len));
            names.push((name, row_major));
            len = (len + t.len()).div_ceil(ALIGN) * ALIGN;
            tensors.push(t);
        }
    }
    let len = len.div_ceil(PAGE) * PAGE;
    let layout = std::alloc::Layout::from_size_align(len, PAGE).unwrap();
    // Leaked on purpose: Metal wraps the mapping without copying, and it must outlive every kernel.
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!base.is_null());
    let bytes = unsafe { std::slice::from_raw_parts_mut(base, len) };
    for (t, &(_, _, _, off)) in tensors.iter().zip(&offs) {
        bytes[off..off + t.len()].copy_from_slice(t);
    }
    imparo_metal::set_weight_kind_types(&imparo_gguf::weights::wire_kind_types());
    imparo_metal::set_rt(true);
    unsafe { imparo_metal::init(base.cast_const(), len as u64) }.unwrap();
    let mut failures = Vec::new();
    for (&(kind, n_in, n_out, off), &(name, row_major)) in offs.iter().zip(&names) {
        failures.extend(check(kind, off, n_in, n_out, name, !row_major));
    }
    for f in &failures {
        eprintln!("FAIL {f}");
    }
    assert!(failures.is_empty(), "{} checks failed", failures.len());
}
