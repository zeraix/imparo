//! FA must preserve scores and row rescaling across KV blocks and partial query tiles.
#![cfg(target_os = "macos")]

use imparo_backend::{Backend as _, BufId};
use imparo_metal::MetalBackend;

#[repr(align(16384))]
struct WeightPage([u8; 16384]);
static WEIGHTS: WeightPage = WeightPage([0; 16384]);

const HD: usize = 64;
const SCALE: f32 = 0.125;
// (absolute query start, number of queries, attention window).
const CASES: &[(usize, usize, usize)] = &[
    (0, 2, 0),
    (0, 8, 0),
    (0, 9, 0),
    (0, 65, 0),
    (63, 2, 0),
    (63, 8, 0),
    (63, 9, 0),
    (63, 65, 0),
    (64, 2, 0),
    (64, 8, 0),
    (64, 9, 0),
    (64, 65, 0),
    // scan_lo = 72 + 1 - 9 = 64: nonzero and aligned. The second
    // query's lower mask advances to 65, within the same query tile.
    (72, 2, 9),
];

fn half_exact_nonzero(seed: usize) -> f32 {
    let signed = i16::try_from(seed % 31).unwrap() - 15;
    f32::from(if signed == 0 { 1 } else { signed }) / 16.0
}

fn reference_row(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    pos: usize,
    window: usize,
) -> Vec<f64> {
    let first = if window == 0 {
        0
    } else {
        (pos + 1).saturating_sub(window)
    };
    let scores: Vec<f64> = (first..=pos)
        .map(|key| {
            q.iter()
                .zip(&k[key * HD..(key + 1) * HD])
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum::<f64>()
                * f64::from(SCALE)
        })
        .collect();
    let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let probabilities: Vec<f64> = scores.iter().map(|s| (s - maximum).exp()).collect();
    let denominator: f64 = probabilities.iter().sum();
    let mut out = vec![0.0_f64; HD];
    for (key, probability) in (first..=pos).zip(probabilities) {
        for (d, value) in out.iter_mut().enumerate() {
            *value += probability * f64::from(v[key * HD + d]);
        }
    }
    for value in &mut out {
        *value /= denominator;
    }
    out
}

// One test owns the process-global Metal context; cases do not initialize or run in parallel.
#[test]
#[ignore = "requires a physical Metal device; run explicitly with --ignored"]
fn fa_matches_softmax_across_blocks_and_query_tails() {
    assert!(
        !std::env::var("IMPARO_ATTN_FA").is_ok_and(|value| value == "0"),
        "this regression fixture requires the FA route"
    );
    imparo_metal::set_attention_head_dims(&[64]);
    imparo_metal::set_attention_kv_widths(&[64]);
    imparo_metal::set_fa_nsg(4);
    imparo_metal::set_kv_types(1, 1);
    // The aligned static page is wrapped by the backend; attention does not read weights.
    unsafe { imparo_metal::init(WEIGHTS.0.as_ptr(), WEIGHTS.0.len() as u64) }.unwrap();
    let backend = MetalBackend;
    let kv_rows = CASES
        .iter()
        .map(|&(start, queries, _)| start + queries)
        .max()
        .unwrap()
        .div_ceil(8)
        * 8;
    // Whole 8-row K/V tiles may be loaded beyond the last live key; populate the
    // padding too so causal masking, rather than zero padding, must discard it.
    let elems = kv_rows * HD;
    for id in [BufId::Q, BufId::K, BufId::V, BufId::Attn] {
        backend
            .alloc(id, u64::try_from(elems * 4).unwrap())
            .unwrap();
    }
    backend
        .alloc_kv(&[u64::try_from(elems * 2).unwrap()])
        .unwrap();
    let k: Vec<f32> = (0..elems)
        .map(|i| half_exact_nonzero((i / HD) * 19 + (i % HD) * 7 + 3))
        .collect();
    let v: Vec<f32> = (0..elems)
        .map(|i| half_exact_nonzero((i / HD) * 13 + (i % HD) * 11 + 9))
        .collect();
    backend.begin();
    backend.write(BufId::K, 0, &k);
    backend.write(BufId::V, 0, &v);
    backend.kv_store(
        BufId::K,
        0,
        64,
        0,
        u32::try_from(kv_rows).unwrap(),
        false,
        0,
    );
    backend.kv_store(BufId::V, 0, 64, 0, u32::try_from(kv_rows).unwrap(), true, 0);
    backend.end().unwrap();

    for &(start, queries, window) in CASES {
        let query_rows = queries.div_ceil(8) * 8;
        let q: Vec<f32> = (0..query_rows * HD)
            .map(|i| half_exact_nonzero((start + i / HD) * 17 + (i % HD) * 5 + 1))
            .collect();
        let mut actual = vec![f32::NAN; elems];
        backend.begin();
        backend.write(BufId::Q, 0, &q);
        backend.write(BufId::Attn, 0, &actual);
        backend.attention(
            0,
            64,
            1,
            1,
            64,
            u32::try_from(start).unwrap(),
            SCALE,
            u32::try_from(window).unwrap(),
            u32::try_from(queries).unwrap(),
            u32::try_from(start + queries).unwrap(),
            0,
        );
        backend.end().unwrap();
        backend.read(BufId::Attn, 0, &mut actual);
        let mut max_error = 0.0_f64;
        for row in 0..queries {
            let expected = reference_row(
                &q[row * HD..(row + 1) * HD],
                &k,
                &v,
                start + row,
                window,
            );
            for (d, &want) in expected.iter().enumerate() {
                let got = actual[row * HD + d];
                let error = (f64::from(got) - want).abs();
                assert!(
                    got.is_finite() && error <= 2e-5,
                    "start={start} queries={queries} window={window} row={row} dim={d}: \
                     got={got} expected={want} error={error}"
                );
                max_error = max_error.max(error);
            }
        }
        assert!(
            actual[queries * HD..].iter().all(|value| value.is_nan()),
            "FA wrote past live query rows: start={start} queries={queries} window={window}"
        );
        eprintln!(
            "FA blocks PASS start={start} queries={queries} window={window} max_error={max_error:.3e}"
        );
    }
}
