//! The row-layout FA entry must match a softmax over exactly the keys each row's layout names:
//! the cache below the batch from the key floor up and the batch rows whose bits are set --
//! rows after the row's own included (a drafted block's rows see the whole block) -- at starts
//! and floors on and off the 8-key tile and 64-key block edges.
#![cfg(target_os = "macos")]

use imparo_backend::{Backend as _, BufId, ROW_LAYOUT_MAX_ROWS, ROW_LAYOUT_WORDS};
use imparo_metal::MetalBackend;

#[repr(align(16384))]
struct WeightPage([u8; 16384]);
static WEIGHTS: WeightPage = WeightPage([0; 16384]);

const HD: usize = 64;
const HEADS: usize = 2;
const SCALE: f32 = 0.125;
/// The long-context cases' batch start: past 1024 keys, so a key split has slices to cut.
const LONG_START: usize = 1500;
/// Query heads a KV head serves in the long-context cases (what the packed-heads grid needs).
const LONG_HEADS: usize = 4;

fn half_exact_nonzero(seed: usize) -> f32 {
    let signed = i16::try_from(seed % 31).unwrap() - 15;
    f32::from(if signed == 0 { 1 } else { signed }) / 16.0
}

fn all_rows(n: usize) -> u64 {
    if n >= 64 { u64::MAX } else { (1_u64 << n) - 1 }
}

/// Each row sees every row of the batch.
fn block(n: usize) -> Vec<u64> {
    vec![all_rows(n); n]
}

/// Row t sees rows 0..=t.
fn chain(n: usize) -> Vec<u64> {
    (1..=n).map(all_rows).collect()
}

/// Node i sees itself and its ancestors.
fn tree(parents: &[i32]) -> Vec<u64> {
    let mut seen = vec![0_u64; parents.len()];
    for (i, &p) in parents.iter().enumerate() {
        seen[i] = 1 << i;
        if let Ok(p) = usize::try_from(p) {
            seen[i] |= seen[p];
        }
    }
    seen
}

/// Row t sees itself and the row `ahead` rows later (or the last row).
fn look_ahead(n: usize, ahead: usize) -> Vec<u64> {
    (0..n)
        .map(|t| (1_u64 << t) | (1_u64 << (t + ahead).min(n - 1)))
        .collect()
}

/// Layout words holding what the attention entry reads: the two visibility words per row.
fn words(start: usize, seen: &[u64]) -> Vec<u32> {
    let mut words = vec![0_u32; seen.len() * ROW_LAYOUT_WORDS];
    for (t, (row, &mask)) in words
        .chunks_exact_mut(ROW_LAYOUT_WORDS)
        .zip(seen)
        .enumerate()
    {
        row[0] = u32::try_from(start + t).unwrap();
        row[1] = u32::try_from(t).unwrap();
        row[2] = mask as u32;
        row[3] = (mask >> 32) as u32;
    }
    words
}

/// `x` rounded to the nearest half (ties to even), as a float-to-half conversion stores it.
fn to_half(x: f32) -> f64 {
    let v = f64::from(x);
    let a = v.abs();
    if a == 0.0 || !a.is_finite() {
        return v;
    }
    let e = a.log2().floor().max(-14.0);
    let step = 2f64.powf(e - 10.0);
    v.signum() * (a / step).round_ties_even() * step
}

fn reference_row(q: &[f32], k: &[f32], v: &[f32], keys: &[usize]) -> Vec<f64> {
    reference_row_q(q, k, v, keys, false)
}

/// The softmax over `keys`, from Q as given or from Q rounded to half.
fn reference_row_q(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    keys: &[usize],
    half_q: bool,
) -> Vec<f64> {
    let scores: Vec<f64> = keys
        .iter()
        .map(|&key| {
            q.iter()
                .zip(&k[key * HD..(key + 1) * HD])
                .map(|(&a, &b)| if half_q { to_half(a) } else { f64::from(a) } * f64::from(b))
                .sum::<f64>()
                * f64::from(SCALE)
        })
        .collect();
    let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let probabilities: Vec<f64> = scores.iter().map(|s| (s - maximum).exp()).collect();
    let denominator: f64 = probabilities.iter().sum();
    let mut out = vec![0.0_f64; HD];
    for (&key, probability) in keys.iter().zip(probabilities) {
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
fn row_layout_attention_sees_exactly_the_named_rows() {
    let parents16 = [-1, 0, 1, 2, 0, 4, 5, 1, 7, 2, 9, 10, 3, 12, 13, 14];
    let cases: Vec<(&str, usize, Vec<u64>)> = vec![
        ("block", 31, block(9)),
        ("block", 32, block(9)),
        ("block", 33, block(9)),
        ("block", 56, block(9)),
        ("block", 63, block(9)),
        ("block", 64, block(9)),
        ("block", 136, block(9)),
        ("block", 32, block(16)),
        ("block", 40, block(33)),
        ("block", 0, block(64)),
        ("block", 64, block(64)),
        ("look_ahead", 60, look_ahead(33, 10)),
        ("chain", 32, chain(16)),
        ("chain", 63, chain(9)),
        ("tree", 32, tree(&parents16)),
        ("tree", 57, tree(&parents16)),
    ];
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
    assert!(
        backend.supports_row_layout(64),
        "no row-layout entry at head dim 64"
    );
    // Whole 64-key blocks, padding populated, so masking rather than zeros must discard it.
    let kv_rows = cases
        .iter()
        .map(|(_, start, seen)| start + seen.len())
        .max()
        .unwrap()
        .max(LONG_START + 3 + 16)
        .div_ceil(64)
        * 64;
    let elems = kv_rows * HD;
    let act = ROW_LAYOUT_MAX_ROWS * HEADS * HD;
    for (id, bytes) in [
        (BufId::Q, act),
        (BufId::Attn, act),
        (BufId::K, elems),
        (BufId::V, elems),
    ] {
        backend
            .alloc(id, u64::try_from(bytes * 4).unwrap())
            .unwrap();
    }
    backend
        .alloc(
            BufId::RowLayout,
            u64::try_from(ROW_LAYOUT_MAX_ROWS * ROW_LAYOUT_WORDS * 4).unwrap(),
        )
        .unwrap();
    backend
        .alloc(
            BufId::AttnPart,
            u64::try_from(LONG_HEADS * 256 * (HD + 2) * 4).unwrap(),
        )
        .unwrap();
    backend
        .alloc_kv(&[u64::try_from(elems * 2).unwrap()])
        .unwrap();
    let k: Vec<f32> = (0..elems)
        .map(|i| half_exact_nonzero((i / HD) * 19 + (i % HD) * 7 + 3))
        .collect();
    let v: Vec<f32> = (0..elems)
        .map(|i| half_exact_nonzero((i / HD) * 13 + (i % HD) * 11 + 9))
        .collect();
    let rows = u32::try_from(kv_rows).unwrap();
    backend.begin();
    backend.write(BufId::K, 0, &k);
    backend.write(BufId::V, 0, &v);
    backend.kv_store(BufId::K, 0, 64, 0, rows, false, 0);
    backend.kv_store(BufId::V, 0, 64, 0, rows, true, 0);
    backend.end().unwrap();

    for (name, start, seen) in &cases {
        let (start, n) = (*start, seen.len());
        let q: Vec<f32> = (0..n * HEADS * HD)
            .map(|i| half_exact_nonzero((start + i / HD) * 17 + (i % HD) * 5 + 1))
            .collect();
        let mut actual = vec![f32::NAN; act];
        backend.begin();
        backend.write(BufId::Q, 0, &q);
        backend.write(BufId::Attn, 0, &actual);
        backend.write_u32(BufId::RowLayout, 0, &words(start, seen));
        let served = backend.attention_rows(
            0,
            64,
            u32::try_from(HEADS).unwrap(),
            1,
            64,
            u32::try_from(start).unwrap(),
            0,
            SCALE,
            u32::try_from(n).unwrap(),
            false,
        );
        backend.end().unwrap();
        assert!(
            served,
            "{name} start={start} rows={n}: the entry refused the layout"
        );
        backend.read(BufId::Attn, 0, &mut actual);
        let mut max_error = 0.0_f64;
        for (t, &mask) in seen.iter().enumerate() {
            let keys: Vec<usize> = (0..start)
                .chain((0..n).filter(|j| (mask >> j) & 1 == 1).map(|j| start + j))
                .collect();
            for h in 0..HEADS {
                let at = (t * HEADS + h) * HD;
                let expected = reference_row(&q[at..at + HD], &k, &v, &keys);
                for (d, &want) in expected.iter().enumerate() {
                    let got = actual[at + d];
                    let error = (f64::from(got) - want).abs();
                    assert!(
                        got.is_finite() && error <= 2e-5,
                        "{name} start={start} rows={n} row={t} head={h} dim={d}: \
                         got={got} expected={want} error={error}"
                    );
                    max_error = max_error.max(error);
                }
            }
        }
        assert!(
            actual[n * HEADS * HD..].iter().all(|value| value.is_nan()),
            "{name} start={start} rows={n}: wrote past the batch rows"
        );
        eprintln!(
            "rows layout PASS {name} start={start} rows={n} max_error={max_error:.3e}"
        );
    }

    // A KEY FLOOR: no row sees a cache key below it (a drafter rebuilt from a restore point).
    // Floors on and off the 8-key tile and 64-key block edges, and at the batch start itself,
    // where the rows see only each other. The keys below the floor hold real values, so an
    // entry that ignored the floor would read them and miss the reference.
    let floor_cases: Vec<(&str, usize, Vec<u64>, usize)> = vec![
        ("block", 136, block(9), 1),
        ("block", 136, block(9), 7),
        ("block", 136, block(9), 63),
        ("block", 136, block(9), 64),
        ("block", 136, block(9), 65),
        ("block", 136, block(9), 100),
        ("block", 136, block(9), 128),
        ("block", 136, block(9), 136),
        ("block", 63, block(9), 8),
        ("block", 64, block(64), 33),
        ("chain", 32, chain(16), 31),
        ("tree", 57, tree(&parents16), 17),
    ];
    for (name, start, seen, floor) in &floor_cases {
        let (start, n, floor) = (*start, seen.len(), *floor);
        let q: Vec<f32> = (0..n * HEADS * HD)
            .map(|i| half_exact_nonzero((start + i / HD) * 23 + (i % HD) * 3 + 2))
            .collect();
        let mut actual = vec![f32::NAN; act];
        backend.begin();
        backend.write(BufId::Q, 0, &q);
        backend.write(BufId::Attn, 0, &actual);
        backend.write_u32(BufId::RowLayout, 0, &words(start, seen));
        let served = backend.attention_rows(
            0,
            64,
            u32::try_from(HEADS).unwrap(),
            1,
            64,
            u32::try_from(start).unwrap(),
            u32::try_from(floor).unwrap(),
            SCALE,
            u32::try_from(n).unwrap(),
            false,
        );
        backend.end().unwrap();
        assert!(
            served,
            "{name} start={start} rows={n} floor={floor}: the entry refused the layout"
        );
        backend.read(BufId::Attn, 0, &mut actual);
        let mut max_error = 0.0_f64;
        for (t, &mask) in seen.iter().enumerate() {
            let keys: Vec<usize> = (floor..start)
                .chain((0..n).filter(|j| (mask >> j) & 1 == 1).map(|j| start + j))
                .collect();
            for h in 0..HEADS {
                let at = (t * HEADS + h) * HD;
                let expected = reference_row(&q[at..at + HD], &k, &v, &keys);
                for (d, &want) in expected.iter().enumerate() {
                    let got = actual[at + d];
                    let error = (f64::from(got) - want).abs();
                    assert!(
                        got.is_finite() && error <= 2e-5,
                        "{name} start={start} rows={n} floor={floor} row={t} head={h} \
                         dim={d}: got={got} expected={want} error={error}"
                    );
                    max_error = max_error.max(error);
                }
            }
        }
        eprintln!(
            "rows layout PASS {name} start={start} rows={n} floor={floor} max_error={max_error:.3e}"
        );
    }

    // A floor above the batch start is refused before anything is encoded.
    backend.begin();
    backend.write_u32(BufId::RowLayout, 0, &words(32, &block(9)));
    let served = backend.attention_rows(
        0,
        64,
        u32::try_from(HEADS).unwrap(),
        1,
        64,
        32,
        33,
        SCALE,
        9,
        false,
    );
    backend.end().unwrap();
    assert!(
        !served,
        "a floor of 33 above a batch starting at 32 was served"
    );
    eprintln!("rows layout PASS refused a floor above the batch start");

    // A layout naming a row outside the batch is refused before anything is encoded.
    let mut outside = block(9);
    outside[0] |= 1 << 9;
    backend.begin();
    backend.write_u32(BufId::RowLayout, 0, &words(32, &outside));
    let served = backend.attention_rows(
        0,
        64,
        u32::try_from(HEADS).unwrap(),
        1,
        64,
        32,
        0,
        SCALE,
        9,
        false,
    );
    backend.end().unwrap();
    assert!(!served, "a layout naming row 9 of a 9-row batch was served");
    eprintln!("rows layout PASS refused a row outside the batch");

    // The entry with Q kept in float (function constant 26) against the one that rounds Q to half,
    // on Q values a half cannot hold: each must match the reference built from its own Q, and the
    // two must sit further apart than either sits from its reference, so a run that silently used
    // one pipeline for both fails.
    let float_cases: Vec<(&str, usize, Vec<u64>)> = vec![
        ("block", 32, block(9)),
        ("chain", 32, chain(16)),
        ("tree", 57, tree(&parents16)),
    ];
    for (name, start, seen) in &float_cases {
        let (start, n) = (*start, seen.len());
        let q: Vec<f32> = (0..n * HEADS * HD)
            .map(|i| {
                half_exact_nonzero((start + i / HD) * 17 + (i % HD) * 5 + 1) * 1.0137
                    + 0.00071
            })
            .collect();
        let mut outputs = [vec![f32::NAN; act], vec![f32::NAN; act]];
        for (arm, float_q) in [false, true].into_iter().enumerate() {
            backend.begin();
            backend.write(BufId::Q, 0, &q);
            backend.write(BufId::Attn, 0, &outputs[arm]);
            backend.write_u32(BufId::RowLayout, 0, &words(start, seen));
            let served = backend.attention_rows(
                0,
                64,
                u32::try_from(HEADS).unwrap(),
                1,
                64,
                u32::try_from(start).unwrap(),
                0,
                SCALE,
                u32::try_from(n).unwrap(),
                float_q,
            );
            backend.end().unwrap();
            assert!(
                served,
                "{name} start={start} float_q={float_q}: the entry refused the layout"
            );
            backend.read(BufId::Attn, 0, &mut outputs[arm]);
        }
        let (mut worst_half, mut worst_float, mut apart) = (0.0_f64, 0.0_f64, 0.0_f64);
        for (t, &mask) in seen.iter().enumerate() {
            let keys: Vec<usize> = (0..start)
                .chain((0..n).filter(|j| (mask >> j) & 1 == 1).map(|j| start + j))
                .collect();
            for h in 0..HEADS {
                let at = (t * HEADS + h) * HD;
                let half_ref = reference_row_q(&q[at..at + HD], &k, &v, &keys, true);
                let float_ref = reference_row_q(&q[at..at + HD], &k, &v, &keys, false);
                for d in 0..HD {
                    worst_half = worst_half
                        .max((f64::from(outputs[0][at + d]) - half_ref[d]).abs());
                    worst_float = worst_float
                        .max((f64::from(outputs[1][at + d]) - float_ref[d]).abs());
                    apart = apart.max(f64::from(
                        (outputs[0][at + d] - outputs[1][at + d]).abs(),
                    ));
                }
            }
        }
        assert!(
            worst_half <= 2e-5
                && worst_float <= 2e-5
                && apart > 10.0 * worst_half.max(worst_float),
            "{name} start={start}: half-Q entry vs half-Q reference {worst_half:.3e}, float-Q entry \
             vs float-Q reference {worst_float:.3e}, entries apart by {apart:.3e}"
        );
        eprintln!(
            "rows layout PASS float_q {name} start={start} rows={n} half_entry_err={worst_half:.3e} \
             float_entry_err={worst_float:.3e} entries_apart={apart:.3e}"
        );
    }

    // THE VERIFY'S GRID. At long context the entry may pack the four query heads of a KV head into
    // one threadgroup (ROW_HEADS) and cut the keys into slices merged afterwards (ROW_SPLIT). Every
    // arm is held to the softmax over the named keys, on values with full half mantissas and
    // scores wide enough that the softmax is not flat. Packing keeps each row's arithmetic, so it
    // must equal the plain arm bit for bit; a split reassociates the softmax, so it is held to the
    // reference only.
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        f32::from(u16::try_from(state >> 48).unwrap()) / 32768.0 - 1.0
    };
    let k2: Vec<f32> = (0..elems).map(|_| to_half(uniform()) as f32).collect();
    let v2: Vec<f32> = (0..elems).map(|_| to_half(uniform()) as f32).collect();
    backend.begin();
    backend.write(BufId::K, 0, &k2);
    backend.write(BufId::V, 0, &v2);
    backend.kv_store(BufId::K, 0, 64, 0, rows, false, 0);
    backend.kv_store(BufId::V, 0, 64, 0, rows, true, 0);
    backend.end().unwrap();
    let long_cases: Vec<(&str, usize, Vec<u64>, usize)> = vec![
        ("chain", LONG_START, chain(16), 0),
        ("tree", LONG_START + 3, tree(&parents16), 0),
        ("block", LONG_START, block(9), 700),
        ("chain", 40, chain(9), 0),
    ];
    // (threadgroups the split aims for, packed heads): plain, packed, then at 1500 keys a split
    // into 2 slices (4 head threadgroups), 4 slices and 16 (packed: one threadgroup a slice).
    let arms: [(u32, u32); 5] = [(0, 0), (0, 1), (8, 0), (4, 1), (64, 1)];
    for (name, start, seen, floor) in &long_cases {
        let (start, n, floor) = (*start, seen.len(), *floor);
        let q: Vec<f32> = (0..n * LONG_HEADS * HD).map(|_| uniform() * 4.0).collect();
        let mut outputs = Vec::new();
        for &(tgs, heads) in &arms {
            imparo_metal::set_verify_attention(tgs, heads);
            let mut actual = vec![f32::NAN; act];
            backend.begin();
            backend.write(BufId::Q, 0, &q);
            backend.write(BufId::Attn, 0, &actual);
            backend.write_u32(BufId::RowLayout, 0, &words(start, seen));
            let served = backend.attention_rows(
                0,
                64,
                u32::try_from(LONG_HEADS).unwrap(),
                1,
                64,
                u32::try_from(start).unwrap(),
                u32::try_from(floor).unwrap(),
                SCALE,
                u32::try_from(n).unwrap(),
                false,
            );
            backend.end().unwrap();
            assert!(
                served,
                "{name} start={start} floor={floor} arm=({tgs},{heads}): the entry refused the layout"
            );
            backend.read(BufId::Attn, 0, &mut actual);
            outputs.push(actual);
        }
        imparo_metal::set_verify_attention(0, 0);
        let mut errors = vec![0.0_f64; arms.len()];
        for (t, &mask) in seen.iter().enumerate() {
            let keys: Vec<usize> = (floor..start)
                .chain((0..n).filter(|j| (mask >> j) & 1 == 1).map(|j| start + j))
                .collect();
            for h in 0..LONG_HEADS {
                let at = (t * LONG_HEADS + h) * HD;
                let expected = reference_row_q(&q[at..at + HD], &k2, &v2, &keys, true);
                for (arm, out) in outputs.iter().enumerate() {
                    for (d, &want) in expected.iter().enumerate() {
                        let got = out[at + d];
                        assert!(
                            got.is_finite(),
                            "{name} start={start} arm={:?} row={t} head={h} dim={d}: got={got}",
                            arms[arm]
                        );
                        errors[arm] = errors[arm].max((f64::from(got) - want).abs());
                    }
                }
            }
        }
        let live = n * LONG_HEADS * HD;
        assert!(
            outputs[1][..live]
                .iter()
                .zip(&outputs[0][..live])
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "{name} start={start}: packed heads differ from the plain arm"
        );
        for out in &outputs {
            assert!(
                out[live..].iter().all(|value| value.is_nan()),
                "{name} start={start}: wrote past the batch rows"
            );
        }
        eprintln!(
            "rows layout long {name} start={start} rows={n} floor={floor} errors plain={:.3e} \
             packed={:.3e} split2={:.3e} split4={:.3e} split16={:.3e}",
            errors[0], errors[1], errors[2], errors[3], errors[4]
        );
    }
}
