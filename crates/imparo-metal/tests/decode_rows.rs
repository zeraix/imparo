//! Decode rows: a projection of B independent rows (one co-batched step over B conversations) must
//! give every row the bits the one-row decode GEMV gives it alone -- in Q4_0 (kind 1) at every
//! lane count the tuner can pick, row-major Q8_0 (kind 2) and tile-major Q8_0_TM (kind 3). A row's
//! K/V bytes may not depend on who else is in its step. B covers a full pipeline (2, 8) and the
//! padded counts (3 on the 4-row pipeline, 5 on the 8-row one), whose dead rows must neither
//! change a live row nor be stored. On the fast route a row count above the family's crossing
//! takes the GEMM instead: its rows must equal that GEMM's (the same projection as one prompt's
//! chunk), and a count at or below it keeps the GEMV's bits. An 8-row GEMM must also write
//! nothing past its outputs, and each output it writes must be its row's one-row GEMV value to a
//! tolerance. On a tile-major weight the rows matmul takes 2..24 rows first: each row within a
//! tolerance of its one-row GEMV value, and read from the half mirror when a producer left only
//! that. A GEMM's half route must turn down a mirror too small for its padded rows. One test,
//! because the device is process-global state.
#![cfg(target_os = "macos")]

use imparo_backend::{Backend as _, BufId, RowRoute};
use imparo_gguf::weights::{
    q8_0_tm_convertible, q8_0_tm_payload_offset, q8_0_tm_scale_offset,
};
use imparo_metal::MetalBackend;

const PAGE: usize = 16384;
const BLOCK: usize = 32;
const Q4_0: u32 = 1;
const ROW_MAJOR: u32 = 2;
const TILE_MAJOR: u32 = 3;
const MAX_ROWS: usize = 8;
/// Floats after the outputs that no kernel may write, and the value they hold.
const GUARD: usize = 64;
const SENTINEL: f32 = 12_345.678;

fn block_bytes(kind: u32) -> usize {
    if kind == Q4_0 { 18 } else { 34 }
}

/// A half-precision float's bits for `v` (truncated to the half's mantissa).
fn half_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = ((bits >> 13) & 0x3ff) as u16;
    assert!(exp > 0 && exp < 31, "scale {v} out of half range");
    sign | ((exp as u16) << 10) | mant
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

/// A block scale with a full half mantissa. Exact values (small multiples of a power of two)
/// make every product and sum exact, so they cannot show a row summed in another order.
fn scale(r: usize, b: usize) -> f32 {
    0.004 + (mix(r, b, 1) % 4096) as f32 * (0.012 / 4096.0)
}

/// Payload byte `i` of block `b` in row `r`: a Q8 value, or two Q4 nibbles.
fn payload(r: usize, b: usize, i: usize) -> u8 {
    ((r * 31 + b * 17 + i * 7) % 255) as u8
}

/// Row `t`'s activations: every row different, so a row reading its neighbour shows, and
/// every value inexact, so a row summed in another order shows too.
fn activation(t: usize, i: usize) -> f32 {
    (mix(t, i, 2) as f32 / u32::MAX as f32) * 2.0 - 1.0
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

/// Every row count's result for one shape; returns the failures as lines, empty when all pass.
fn check(kind: u32, off: usize, n_in: usize, n_out: usize, what: &str) -> Vec<String> {
    let mut failures = Vec::new();
    let be = MetalBackend;
    let x: Vec<f32> = (0..MAX_ROWS)
        .flat_map(|t| (0..n_in).map(move |i| activation(t, i)))
        .collect();
    be.alloc(BufId::X, (x.len() * 4) as u64).unwrap();
    // The outputs, then a guard the kernels must never write.
    be.alloc(BufId::O, ((MAX_ROWS * n_out + GUARD) * 4) as u64)
        .unwrap();
    be.write(BufId::O, (MAX_ROWS * n_out) as u64, &[SENTINEL; GUARD]);
    be.write(BufId::X, 0, &x);
    // Each row alone: the one-row GEMV, reading row t, writing output row 0.
    imparo_metal::set_decode_rows(None);
    let alone: Vec<Vec<u32>> = (0..MAX_ROWS)
        .map(|t| project(kind, off, n_in, n_out, 1, t)[..n_out].to_vec())
        .collect();
    // The GEMM writes only its outputs. Eight rows fill the buffer to its last output, so a
    // store past n_out in the last row lands in the guard; one in any other row lands on the
    // next row's first outputs, where it shows only when the race goes that way.
    project(kind, off, n_in, n_out, MAX_ROWS, 0);
    let mut all = vec![0.0_f32; MAX_ROWS * n_out + GUARD];
    be.read(BufId::O, 0, &mut all);
    let past = all[MAX_ROWS * n_out..]
        .iter()
        .filter(|v| v.to_bits() != SENTINEL.to_bits())
        .count();
    if past > 0 {
        failures.push(format!(
            "{what} n_in={n_in} n_out={n_out}: the 8-row GEMM wrote {past} floats past its outputs"
        ));
    }
    // And every output it wrote is its own: against the row's one-row GEMV, to a tolerance. The
    // two sum in different orders and precisions, so their bits differ, but a value stored in
    // the wrong place differs by the size of the values themselves.
    for (t, want) in alone.iter().enumerate() {
        let got = &all[t * n_out..(t + 1) * n_out];
        let want: Vec<f32> = want.iter().map(|&b| f32::from_bits(b)).collect();
        let squares: f64 = want.iter().map(|&w| f64::from(w).powi(2)).sum();
        let rms = (squares / n_out as f64).sqrt();
        if let Some(i) = got
            .iter()
            .zip(&want)
            .position(|(&g, &w)| (f64::from(g) - f64::from(w)).abs() > 1e-2 * rms)
        {
            failures.push(format!(
                "{what} n_in={n_in} n_out={n_out}: the 8-row GEMM's row {t} output {i} is {} \
                 where its one-row GEMV gives {}",
                got[i], want[i]
            ));
        }
    }
    // The same rows as one prompt's chunk: the GEMM the fast route takes above its crossing.
    let chunk: Vec<Vec<u32>> = [2_usize, 3, 5, 8]
        .iter()
        .map(|&rows| project(kind, off, n_in, n_out, rows, 0))
        .collect();
    imparo_metal::set_decode_rows(Some(RowRoute::Exact));
    for rows in [2_usize, 3, 5, 8] {
        let got = project(kind, off, n_in, n_out, rows, 0);
        for (t, want) in alone.iter().enumerate().take(rows) {
            let row = &got[t * n_out..(t + 1) * n_out];
            if let Some(i) = row.iter().zip(want).position(|(a, b)| a != b) {
                failures.push(format!(
                    "{what} n_in={n_in} n_out={n_out} rows={rows}: row {t} differs from its \
                     one-row bits at output {i} ({} vs {})",
                    f32::from_bits(row[i]),
                    f32::from_bits(want[i])
                ));
                break;
            }
        }
        for (t, chunk) in got.chunks(n_out).enumerate().skip(rows) {
            if !chunk.iter().all(|&v| f32::from_bits(v).is_nan()) {
                failures.push(format!(
                    "{what} n_in={n_in} n_out={n_out} rows={rows}: dead row {t} was written"
                ));
            }
        }
    }
    // The fast route with its crossing at 3: rows 2 and 3 keep the GEMV, 5 and 8 take the GEMM.
    let (set_max, get_max): (fn(u32), fn() -> u32) = if kind == Q4_0 {
        (
            imparo_metal::set_q4_rows_gemv_max,
            imparo_metal::q4_rows_gemv_max,
        )
    } else {
        (
            imparo_metal::set_q8_rows_gemv_max,
            imparo_metal::q8_rows_gemv_max,
        )
    };
    let seated = get_max();
    set_max(3);
    // The GEMM side of the crossing: the rows matmul, which takes those rows first on a
    // tile-major or Q4_0 weight, is checked on its own in check_mma.
    let mma_seated = imparo_metal::q8_tm_rows_mma_max();
    imparo_metal::set_q8_tm_rows_mma_max(0);
    let q4_mma_seated = imparo_metal::q4_rows_mma_max();
    imparo_metal::set_q4_rows_mma_max(0);
    imparo_metal::set_decode_rows(Some(RowRoute::Fast));
    for (i, rows) in [2_usize, 3, 5, 8].into_iter().enumerate() {
        let got = project(kind, off, n_in, n_out, rows, 0);
        for t in 0..rows {
            let row = &got[t * n_out..(t + 1) * n_out];
            let (want, route) = if rows <= 3 {
                (&alone[t][..], "its one-row GEMV bits")
            } else {
                (
                    &chunk[i][t * n_out..(t + 1) * n_out],
                    "the chunk GEMM's bits",
                )
            };
            if let Some(o) = row.iter().zip(want).position(|(a, b)| a != b) {
                failures.push(format!(
                    "{what} n_in={n_in} n_out={n_out} fast rows={rows}: row {t} is not \
                     {route} at output {o}"
                ));
                break;
            }
        }
    }
    set_max(seated);
    imparo_metal::set_q8_tm_rows_mma_max(mma_seated);
    imparo_metal::set_q4_rows_mma_max(q4_mma_seated);
    imparo_metal::set_decode_rows(None);
    if failures.is_empty() {
        eprintln!(
            "{what} n_in={n_in} n_out={n_out}: rows 2/3/5/8 bit-equal to one row; fast route \
             at crossing 3 takes the GEMV to 3 rows and the GEMM above"
        );
    }
    failures
}

/// The rows matmul (the fast route's rows past the GEMV's, tile-major only), 2..24 rows: every
/// live row within a tolerance of its one-row GEMV value, dead rows and the guard unwritten, two
/// runs bit-equal, and the route counted as taken -- a row the GEMM or GEMV served instead would
/// pass the value check and prove nothing.
fn check_mma(
    kind: u32,
    off: usize,
    n_in: usize,
    n_out: usize,
    what: &str,
) -> Vec<String> {
    const ROWS: usize = 24;
    let mut failures = Vec::new();
    let be = MetalBackend;
    let x: Vec<f32> = (0..ROWS)
        .flat_map(|t| (0..n_in).map(move |i| activation(t, i)))
        .collect();
    be.alloc(BufId::X, (x.len() * 4) as u64).unwrap();
    be.write(BufId::X, 0, &x);
    be.alloc(BufId::O, ((ROWS * n_out + GUARD) * 4) as u64)
        .unwrap();
    let run = |rows: usize, src_row: usize| -> Vec<f32> {
        be.write(BufId::O, 0, &vec![f32::NAN; ROWS * n_out]);
        be.write(BufId::O, (ROWS * n_out) as u64, &[SENTINEL; GUARD]);
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
        let mut got = vec![0.0_f32; ROWS * n_out + GUARD];
        be.read(BufId::O, 0, &mut got);
        got
    };
    imparo_metal::set_decode_rows(None);
    let alone: Vec<Vec<f32>> = (0..ROWS).map(|t| run(1, t)[..n_out].to_vec()).collect();
    // The format's own crossings: the GEMV's and the rows matmul's.
    let (set_gemv, get_gemv, set_mma, get_mma): (
        fn(u32),
        fn() -> u32,
        fn(u32),
        fn() -> u32,
    ) = if kind == Q4_0 {
        (
            imparo_metal::set_q4_rows_gemv_max,
            imparo_metal::q4_rows_gemv_max,
            imparo_metal::set_q4_rows_mma_max,
            imparo_metal::q4_rows_mma_max,
        )
    } else {
        (
            imparo_metal::set_q8_rows_gemv_max,
            imparo_metal::q8_rows_gemv_max,
            imparo_metal::set_q8_tm_rows_mma_max,
            imparo_metal::q8_tm_rows_mma_max,
        )
    };
    let (gemv_max, mma_max) = (get_gemv(), get_mma());
    set_gemv(1);
    set_mma(ROWS as u32);
    imparo_metal::set_decode_rows(Some(RowRoute::Fast));
    let mut worst = 0.0_f64;
    // Each count of 8-token columns, full and with dead rows: 1 (2, 3, 5, 8), 2 (9, 16) and
    // 3 (17, 20, 24).
    for rows in [2_usize, 3, 5, 8, 9, 16, 17, 20, 24] {
        let before = imparo_metal::rows_mma_dispatches();
        let got = run(rows, 0);
        let again = run(rows, 0);
        let taken = imparo_metal::rows_mma_dispatches() - before;
        if taken != 2 {
            failures.push(format!(
                "{what} n_in={n_in} n_out={n_out} rows={rows}: the rows matmul ran {taken} of 2 times"
            ));
            continue;
        }
        if got
            .iter()
            .zip(&again)
            .any(|(a, b)| a.to_bits() != b.to_bits())
        {
            failures.push(format!(
                "{what} n_in={n_in} n_out={n_out} rows={rows}: two runs gave different bits"
            ));
        }
        for (t, want) in alone.iter().enumerate().take(rows) {
            let row = &got[t * n_out..(t + 1) * n_out];
            let squares: f64 = want.iter().map(|&w| f64::from(w).powi(2)).sum();
            let rms = (squares / n_out as f64).sqrt().max(1e-30);
            let err = row
                .iter()
                .zip(want)
                .map(|(&g, &w)| (f64::from(g) - f64::from(w)).abs() / rms)
                .fold(
                    0.0_f64,
                    |a, e| if e.is_nan() { f64::INFINITY } else { a.max(e) },
                );
            worst = worst.max(err);
            if err > 1e-4 {
                failures.push(format!(
                    "{what} n_in={n_in} n_out={n_out} rows={rows}: row {t} is {err:.2e} of its rms \
                     from its one-row GEMV value"
                ));
            }
        }
        if got[rows * n_out..ROWS * n_out].iter().any(|v| !v.is_nan()) {
            failures.push(format!(
                "{what} n_in={n_in} n_out={n_out} rows={rows}: a dead row was written"
            ));
        }
        if got[ROWS * n_out..]
            .iter()
            .any(|v| v.to_bits() != SENTINEL.to_bits())
        {
            failures.push(format!(
                "{what} n_in={n_in} n_out={n_out} rows={rows}: written past the outputs"
            ));
        }
    }
    set_gemv(gemv_max);
    set_mma(mma_max);
    imparo_metal::set_decode_rows(None);
    eprintln!(
        "{what} n_in={n_in} n_out={n_out}: rows matmul at 2/3/5/8/9/16/17/20/24 rows within \
         {worst:.2e} of each row's rms from its one-row GEMV, two runs bit-equal"
    );
    failures
}

/// THE MIRROR'S SIZE: a GEMM's half route converts whole token tiles of its input into the
/// mirror, so a mirror that cannot hold the padded rows must turn the route down instead of
/// converting past its end. The gated pair rides only that route, so it shows the decision:
/// refused with a mirror of exactly five rows (every token tile is 8 or more), run with one
/// of `MAX_GEMM_TOKEN_TILE` rows. A build that checked only that the mirror exists runs the
/// pair on the small one and writes past it.
fn check_xh_guard(kind: u32, off: usize, n_in: usize, n_out: usize) -> Vec<String> {
    const ROWS: usize = 5;
    let mut failures = Vec::new();
    let be = MetalBackend;
    let x: Vec<f32> = (0..ROWS)
        .flat_map(|t| (0..n_in).map(move |i| activation(t, i)))
        .collect();
    be.alloc(BufId::X, (x.len() * 4) as u64).unwrap();
    be.write(BufId::X, 0, &x);
    be.alloc(BufId::G, (ROWS * n_out * 4) as u64).unwrap();
    let pair = |mirror_rows: usize| -> bool {
        be.alloc(BufId::Xh, (mirror_rows * n_in * 2) as u64)
            .unwrap();
        be.begin();
        imparo_metal::set_decode_rows(None);
        imparo_metal::set_epilogue(1);
        let ran = imparo_metal::matmat_gated(
            kind,
            off as u64,
            kind,
            off as u64,
            n_in as u32,
            n_out as u32,
            BufId::X as u32,
            BufId::G as u32,
            ROWS as u32,
        );
        imparo_metal::set_epilogue(0);
        be.end().unwrap();
        ran
    };
    // Small first: the allocation keeps a buffer that is already big enough.
    if pair(ROWS) {
        failures.push(format!(
            "kind={kind} n_in={n_in} n_out={n_out}: the gated pair ran on a mirror of {ROWS} \
             rows, smaller than its padded token tile"
        ));
    }
    if !pair(imparo_backend::MAX_GEMM_TOKEN_TILE) {
        failures.push(format!(
            "kind={kind} n_in={n_in} n_out={n_out}: the gated pair was refused with a mirror \
             that holds a whole token tile, so the small case proved nothing"
        ));
    }
    if failures.is_empty() {
        eprintln!(
            "kind={kind} n_in={n_in} n_out={n_out}: the gated pair is refused on a mirror of \
             {ROWS} rows and runs on one of {} rows",
            imparo_backend::MAX_GEMM_TOKEN_TILE
        );
    }
    failures
}

/// STALE FLOATS: the gated pair GEMM in mirror mode writes only the half mirror of G, so G's
/// floats are whatever was there before -- NaN here, written from the host first. The rows
/// matmul on G must read the mirror: its rows match the GEMM that reads the same mirror, to the
/// GEMM's own precision, and hold no NaN. LFM2.5's FFN shapes: the pair 2048 -> 10752, then the
/// down projection 10752 -> 2048.
fn check_mma_stale(pair_off: usize, down_off: usize) -> Vec<String> {
    const ROWS: usize = 8;
    let (n_embd, n_ff) = (2048_usize, 10752_usize);
    let mut failures = Vec::new();
    let be = MetalBackend;
    let x: Vec<f32> = (0..ROWS)
        .flat_map(|t| (0..n_embd).map(move |i| activation(t, i)))
        .collect();
    be.alloc(BufId::X, (x.len() * 4) as u64).unwrap();
    be.write(BufId::X, 0, &x);
    be.alloc(BufId::G, (ROWS * n_ff * 4) as u64).unwrap();
    be.alloc(BufId::Xh, (64 * n_ff * 2) as u64).unwrap();
    be.alloc(BufId::Xh2, (64 * n_ff * 2) as u64).unwrap();
    be.alloc(BufId::O, (ROWS * n_embd * 4) as u64).unwrap();
    be.alloc(BufId::Logits, (ROWS * n_embd * 4) as u64).unwrap();
    let (gemv_max, mma_max) = (
        imparo_metal::q8_rows_gemv_max(),
        imparo_metal::q8_tm_rows_mma_max(),
    );
    imparo_metal::set_q8_rows_gemv_max(1);
    imparo_metal::set_q8_tm_rows_mma_max(16);
    // One region: the pair, then the down projection on its G, as the forward issues them.
    let run = |dst: BufId, rows_route: bool| -> (bool, u64, Vec<f32>) {
        be.write(BufId::G, 0, &vec![f32::NAN; ROWS * n_ff]);
        be.begin();
        imparo_metal::set_decode_rows(None);
        imparo_metal::set_epilogue(1);
        let pair = imparo_metal::matmat_gated(
            TILE_MAJOR,
            pair_off as u64,
            TILE_MAJOR,
            pair_off as u64,
            n_embd as u32,
            n_ff as u32,
            BufId::X as u32,
            BufId::G as u32,
            ROWS as u32,
        );
        imparo_metal::set_epilogue(0);
        if rows_route {
            imparo_metal::set_decode_rows(Some(RowRoute::Fast));
        }
        let before = imparo_metal::rows_mma_dispatches();
        imparo_metal::matmat_from(
            TILE_MAJOR,
            down_off as u64,
            n_ff as u32,
            n_embd as u32,
            BufId::G as u32,
            dst as u32,
            ROWS as u32,
            0,
        );
        let taken = imparo_metal::rows_mma_dispatches() - before;
        be.end().unwrap();
        imparo_metal::set_decode_rows(None);
        let mut got = vec![0.0_f32; ROWS * n_embd];
        be.read(dst, 0, &mut got);
        (pair, taken, got)
    };
    let (pair_gemm, _, want) = run(BufId::Logits, false);
    let (pair_rows, taken, got) = run(BufId::O, true);
    imparo_metal::set_q8_rows_gemv_max(gemv_max);
    imparo_metal::set_q8_tm_rows_mma_max(mma_max);
    if !pair_gemm || !pair_rows {
        failures.push(
            "stale floats: the gated pair did not run, so nothing was tested".into(),
        );
        return failures;
    }
    if taken != 1 {
        failures.push(format!(
            "stale floats: the rows matmul ran {taken} of 1 times"
        ));
        return failures;
    }
    let squares: f64 = want.iter().map(|&w| f64::from(w).powi(2)).sum();
    let rms = (squares / want.len() as f64).sqrt().max(1e-30);
    let err = got
        .iter()
        .zip(&want)
        .map(|(&g, &w)| (f64::from(g) - f64::from(w)).abs() / rms)
        .fold(
            0.0_f64,
            |a, e| if e.is_nan() { f64::INFINITY } else { a.max(e) },
        );
    if !want.iter().all(|v| v.is_finite()) || err > 1e-2 {
        failures.push(format!(
            "stale floats: the rows matmul on the pair's G is {err:.2e} of its rms from the GEMM \
             reading the same mirror"
        ));
    } else {
        eprintln!(
            "stale floats: the rows matmul read the pair's half mirror, within {err:.2e} of the \
             GEMM on the same mirror, no NaN from the floats"
        );
    }
    failures
}

#[test]
#[ignore = "requires a physical Metal device; run explicitly with --ignored"]
fn decode_rows_give_every_row_its_one_row_bits() {
    let shapes = [
        // Q4_0: fewer blocks than lanes, a partial last threadgroup, E4B's widths.
        (Q4_0, 64_usize, 4097_usize),
        (Q4_0, 256, 1000),
        (Q4_0, 2560, 2050),
        (Q4_0, 10240, 70),
        // The rows matmul needs whole 8-row tiles: E4B's attention and FFN widths.
        (Q4_0, 2560, 2048),
        (Q4_0, 2560, 10240),
        (Q4_0, 10240, 2560),
        (ROW_MAJOR, 64, 4097),
        (ROW_MAJOR, 256, 1000),
        (ROW_MAJOR, 2048, 65),
        (TILE_MAJOR, 64, 4096),
        (TILE_MAJOR, 256, 1000),
        (TILE_MAJOR, 2048, 64),
        (TILE_MAJOR, 2048, 10752),
        // LFM2.5-2.6B's projections as served: q/o, k/v, the convolution's input, the FFN
        // down projection (the one input 10752 wide) and the tied lm head, in both layouts.
        (TILE_MAJOR, 2048, 2048),
        (TILE_MAJOR, 2048, 512),
        (TILE_MAJOR, 2048, 6144),
        (TILE_MAJOR, 10752, 2048),
        (TILE_MAJOR, 2048, 65536),
        (ROW_MAJOR, 10752, 2048),
        (ROW_MAJOR, 2048, 65536),
    ];
    let mut offs = Vec::new();
    let mut len = 0;
    for &(kind, n_in, n_out) in &shapes {
        assert!(kind != TILE_MAJOR || q8_0_tm_convertible(n_in, n_out));
        offs.push(len);
        len += n_out * (n_in / BLOCK) * block_bytes(kind);
    }
    let len = len.div_ceil(PAGE) * PAGE;
    let layout = std::alloc::Layout::from_size_align(len, PAGE).unwrap();
    // Leaked on purpose: Metal wraps the mapping without copying, and it must outlive every kernel.
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!base.is_null());
    let bytes = unsafe { std::slice::from_raw_parts_mut(base, len) };
    for (&(kind, n_in, n_out), &off) in shapes.iter().zip(&offs) {
        let blocks = n_in / BLOCK;
        let bb = block_bytes(kind);
        for r in 0..n_out {
            for b in 0..blocks {
                let (scale_at, values_at) = if kind == TILE_MAJOR {
                    (
                        off + q8_0_tm_scale_offset(r, b, blocks, n_out),
                        off + q8_0_tm_payload_offset(r, b, blocks),
                    )
                } else {
                    let at = off + (r * blocks + b) * bb;
                    (at, at + 2)
                };
                bytes[scale_at..scale_at + 2]
                    .copy_from_slice(&half_bits(scale(r, b)).to_le_bytes());
                for i in 0..bb - 2 {
                    bytes[values_at + i] = payload(r, b, i);
                }
            }
        }
    }
    imparo_metal::set_rt(true);
    unsafe { imparo_metal::init(base.cast_const(), len as u64) }.unwrap();
    let mut failures = Vec::new();
    for (&(kind, n_in, n_out), &off) in shapes.iter().zip(&offs) {
        if kind == Q4_0 {
            // The one-row kernel's K split is its lane count, a tuned value: every one it can take.
            for lanes in [4_u32, 8, 16, 32] {
                imparo_metal::set_lanes(lanes);
                failures.extend(check(
                    kind,
                    off,
                    n_in,
                    n_out,
                    &format!("kind=1 lanes={lanes}"),
                ));
            }
            imparo_metal::set_lanes(16);
            if n_out % 8 == 0 {
                failures.extend(check_mma(kind, off, n_in, n_out, "kind=1 mma"));
            }
        } else {
            failures.extend(check(kind, off, n_in, n_out, &format!("kind={kind}")));
            if kind == TILE_MAJOR {
                failures.extend(check_mma(kind, off, n_in, n_out, "kind=3 mma"));
            }
        }
    }
    let off_of = |kind: u32, n_in: usize, n_out: usize| {
        shapes
            .iter()
            .zip(&offs)
            .find(|(s, _)| **s == (kind, n_in, n_out))
            .map(|(_, &o)| o)
            .unwrap()
    };
    // Before check_mma_stale, whose mirror is big enough for every shape here.
    failures.extend(check_xh_guard(
        TILE_MAJOR,
        off_of(TILE_MAJOR, 2048, 10752),
        2048,
        10752,
    ));
    failures.extend(check_xh_guard(Q4_0, off_of(Q4_0, 2560, 2050), 2560, 2050));
    failures.extend(check_mma_stale(
        off_of(TILE_MAJOR, 2048, 10752),
        off_of(TILE_MAJOR, 10752, 2048),
    ));
    for f in &failures {
        eprintln!("FAIL {f}");
    }
    assert!(
        failures.is_empty(),
        "{} shapes differ from their one-row bits",
        failures.len()
    );
}
