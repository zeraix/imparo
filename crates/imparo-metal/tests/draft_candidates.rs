//! The drafter's two row entries against a host computation: `top_k_rows` against a host sort,
//! ids and value bits, on rows built to put equal values in different chunks; `logistic_rows`
//! against a host sum in f64. The shapes reach both top K kernels (k up to 8 and above), lists with
//! empty slots, and confidence blocks that do not divide the widths. One test, because the device
//! is process-global state.
#![cfg(target_os = "macos")]

use imparo_backend::{Backend as _, BufId};
use imparo_metal::MetalBackend;

#[repr(align(16384))]
struct WeightPage([f32; 4096]);

static WEIGHTS: WeightPage = WeightPage([0.0; 4096]);

/// Few distinct values, so equal values land in different chunks; also -inf, -0 and +0.
fn tied(row: usize, i: usize) -> f32 {
    match (i.wrapping_mul(2_654_435_761) ^ row.wrapping_mul(40_503)) % 97 {
        0 => f32::NEG_INFINITY,
        1 => -0.0,
        2 => 0.0,
        x => (x as f32 - 48.0) / 8.0,
    }
}

/// Mostly distinct values, as a row of logits holds.
fn spread(row: usize, i: usize) -> f32 {
    ((i * 7_919 + row * 104_729) % 1_000_003) as f32 / 1_000.0 - 500.0
}

/// The host's order: larger values first, the smaller index first among equal values.
fn host_top_k(row: &[f32], k: usize) -> Vec<(u32, u32)> {
    let mut order: Vec<usize> = (0..row.len()).collect();
    order.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap().then(a.cmp(&b)));
    order
        .into_iter()
        .take(k)
        .map(|i| (i as u32, row[i].to_bits()))
        .collect()
}

#[test]
#[ignore = "requires a physical Metal device; run explicitly with --ignored"]
fn candidates_and_confidence_match_the_host() {
    imparo_metal::set_rt(true);
    unsafe { imparo_metal::init(WEIGHTS.0.as_ptr().cast(), 16384) }.unwrap();
    let be = MetalBackend;
    let max = imparo_metal::top_k_rows_max();
    for (width, rows, k) in [
        (128_000_u32, 9_u32, 8_u32),
        (128_000, 9, max),
        (1_000, 3, 1),
        (1_001, 2, 7),
        (65, 4, max),
        (64, 1, max),
        (5, 2, 5),
        (3, 1, 1),
        (9, 2, 8),
        (16, 3, 8),
        (8, 1, 8),
        (130, 2, 3),
    ] {
        for (name, fill) in [
            ("tied", tied as fn(usize, usize) -> f32),
            ("spread", spread),
        ] {
            let (w, r, kk) = (width as usize, rows as usize, k as usize);
            let src: Vec<f32> = (0..r * w).map(|x| fill(x / w, x % w)).collect();
            let len = be.top_k_rows_len(width, rows, k);
            be.alloc(BufId::DraftLogits, (r * w) as u64 * 4).unwrap();
            be.alloc(BufId::DraftTop, len * 4).unwrap();
            be.write(BufId::DraftLogits, 0, &src);
            be.begin();
            assert!(be.top_k_rows(BufId::DraftLogits, BufId::DraftTop, width, rows, k));
            be.end().unwrap();
            let mut out = vec![0.0_f32; 2 * r * kk];
            be.read(BufId::DraftTop, 0, &mut out);
            for row in 0..r {
                let want = host_top_k(&src[row * w..(row + 1) * w], kk);
                let got: Vec<(u32, u32)> = (0..kk)
                    .map(|j| {
                        (
                            out[row * kk + j].to_bits(),
                            out[r * kk + row * kk + j].to_bits(),
                        )
                    })
                    .collect();
                assert_eq!(
                    got, want,
                    "{name} width={width} rows={rows} k={k} row={row}"
                );
            }
        }
    }
    // Refusals write nothing and say so.
    be.alloc(BufId::DraftLogits, 4 * 64).unwrap();
    be.alloc(BufId::DraftTop, be.top_k_rows_len(64, 1, max) * 4)
        .unwrap();
    be.begin();
    assert!(!be.top_k_rows(BufId::DraftLogits, BufId::DraftTop, 64, 1, max + 1));
    assert!(!be.top_k_rows(BufId::DraftLogits, BufId::DraftTop, 3, 1, 4));
    be.end().unwrap();

    for (a_width, b_width, rows, w_off, dst_off, shared) in [
        (2048_u32, 256_u32, 9_u32, 0_u32, 2305_u32, true),
        (2048, 256, 9, 3, 2311, true),
        (100, 30, 3, 1, 132, true),
        (65, 1, 2, 0, 0, false),
        (3, 5, 1, 0, 0, false),
        (7, 1, 4, 2, 1, false),
    ] {
        let (aw, bw, r) = (a_width as usize, b_width as usize, rows as usize);
        let a: Vec<f32> = (0..r * aw)
            .map(|i| ((i % 251) as f32 - 125.0) / 97.0)
            .collect();
        let b: Vec<f32> = (0..r * bw)
            .map(|i| ((i % 139) as f32 - 69.0) / 53.0)
            .collect();
        let mut w = vec![0.0_f32; w_off as usize];
        w.extend((0..aw + bw).map(|j| ((j * 37 % 101) as f32 - 50.0) / 2_500.0));
        w.push(-0.375);
        let dst = if shared { BufId::Model0 } else { BufId::Model1 };
        let dst_elems =
            dst_off as usize + be.logistic_rows_len(a_width, b_width, rows) as usize;
        be.alloc(BufId::X, (r * aw) as u64 * 4).unwrap();
        be.alloc(BufId::O, (r * bw) as u64 * 4).unwrap();
        be.alloc(BufId::Model0, (w.len().max(dst_elems) as u64 + 4) * 4)
            .unwrap();
        be.alloc(BufId::Model1, dst_elems as u64 * 4).unwrap();
        be.write(BufId::X, 0, &a);
        be.write(BufId::O, 0, &b);
        be.write(BufId::Model0, 0, &w);
        be.begin();
        assert!(be.logistic_rows(
            BufId::X,
            a_width,
            BufId::O,
            b_width,
            BufId::Model0,
            w_off,
            dst,
            dst_off,
            rows
        ));
        be.end().unwrap();
        let mut got = vec![0.0_f32; r];
        be.read(dst, u64::from(dst_off), &mut got);
        let head = &w[w_off as usize..];
        for row in 0..r {
            let z = a[row * aw..(row + 1) * aw]
                .iter()
                .chain(&b[row * bw..(row + 1) * bw])
                .zip(head)
                .map(|(x, w)| f64::from(*x) * f64::from(*w))
                .sum::<f64>()
                + f64::from(head[aw + bw]);
            let want = 1.0 / (1.0 + (-z).exp());
            let delta = (want - f64::from(got[row])).abs();
            assert!(
                delta <= 1e-5,
                "a_width={a_width} b_width={b_width} row={row}: device {} host {want} delta {delta}",
                got[row]
            );
        }
    }
    // Refusals write nothing: an output over the weights it reads, and a dst without room for the
    // working space. The inputs are sized for the call, so each refusal has one cause.
    let len = be.logistic_rows_len(3, 5, 1);
    be.alloc(BufId::X, 3 * 4).unwrap();
    be.alloc(BufId::O, 5 * 4).unwrap();
    be.alloc(BufId::Model0, (9 + len) * 4).unwrap();
    be.alloc(BufId::Model1, (len - 1) * 4).unwrap();
    be.begin();
    assert!(!be.logistic_rows(
        BufId::X,
        3,
        BufId::O,
        5,
        BufId::Model0,
        0,
        BufId::Model0,
        8,
        1
    ));
    assert!(be.logistic_rows(
        BufId::X,
        3,
        BufId::O,
        5,
        BufId::Model0,
        0,
        BufId::Model0,
        9,
        1
    ));
    assert!(!be.logistic_rows(
        BufId::X,
        3,
        BufId::O,
        5,
        BufId::Model0,
        0,
        BufId::Model1,
        0,
        1
    ));
    be.end().unwrap();
}
