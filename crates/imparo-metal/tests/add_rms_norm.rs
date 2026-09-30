//! The combined residual/norm operation must preserve the split transaction.
#![cfg(target_os = "macos")]

use imparo_backend::{Backend as _, BufId};
use imparo_metal::MetalBackend;

#[repr(align(16384))]
struct WeightPage([f32; 4096]);

const fn norm_weights() -> WeightPage {
    let mut values = [0.0; 4096];
    let mut i = 0;
    while i < values.len() {
        values[i] = 0.5 + (i % 31) as f32 / 64.0;
        i += 1;
    }
    WeightPage(values)
}

static WEIGHTS: WeightPage = norm_weights();

fn read_bits(backend: MetalBackend, id: BufId, count: usize) -> Vec<u32> {
    let mut values = vec![0.0; count];
    backend.read(id, 0, &mut values);
    values.into_iter().map(f32::to_bits).collect()
}

#[test]
#[ignore = "requires a physical Metal device; run explicitly with --ignored"]
fn add_rms_norm_preserves_residual_output_and_half_mirror() {
    imparo_metal::set_rt(true);
    // This aligned, static page backs both the weighted and unweighted cases.
    unsafe { imparo_metal::init(WEIGHTS.0.as_ptr().cast(), 16384) }.unwrap();
    let backend = MetalBackend;
    // Exercise wide and narrow reductions, decode, a non-power-of-two width,
    // and an unaligned weight offset through the ordinary backend operation.
    let cases = [
        (2048_u32, 8_u32, 0_u64),
        (128, 3, u64::MAX),
        (2048, 1, 0),
        (4092, 3, 0),
        (2048, 3, 4),
    ];
    // Repeat through distinct, non-overlapping slices of one arena resource.
    for (placed, width, rows, weight) in [false, true].into_iter().flat_map(|placed| {
        cases.map(|(width, rows, weight)| (placed, width, rows, weight))
    }) {
        let count = (width * rows) as usize;
        if placed {
            let span = backend.page_round(count as u64 * 4);
            let mirror_span = backend.page_round(count as u64 * 2);
            backend.arena(3 * span + mirror_span).unwrap();
            for (slot, id) in [BufId::X, BufId::O, BufId::Cur].into_iter().enumerate() {
                backend
                    .place(id, slot as u64 * span, count as u64 * 4)
                    .unwrap();
            }
            backend
                .place(BufId::Xh, 3 * span, count as u64 * 2)
                .unwrap();
        } else {
            for id in [BufId::X, BufId::O, BufId::Cur] {
                backend.alloc(id, count as u64 * 4).unwrap();
            }
            backend.alloc(BufId::Xh, count as u64 * 2).unwrap();
        }
        let residual: Vec<f32> = (0..count)
            .map(|i| ((i % 251) as f32 - 125.0) / 97.0)
            .collect();
        let other: Vec<f32> = (0..count)
            .map(|i| ((i % 139) as f32 - 69.0) / 113.0)
            .collect();
        let initial = vec![f32::from_bits(0x7e00_7e00); count / 2];
        let reset = || {
            backend.write(BufId::X, 0, &residual);
            backend.write(BufId::O, 0, &other);
            backend.write(BufId::Xh, 0, &initial);
        };
        reset();
        backend.begin();
        backend.add(BufId::X, BufId::O, width * rows);
        backend.rms_norm_from(
            BufId::Cur,
            BufId::X,
            weight,
            width,
            1e-5,
            rows,
            width,
            0,
        );
        backend.end().unwrap();
        let expected_residual = read_bits(backend, BufId::X, count);
        let expected_output = read_bits(backend, BufId::Cur, count);
        // Read packed half bytes without converting them through f32 arithmetic.
        let expected_mirror = read_bits(backend, BufId::Xh, count / 2);
        if rows > 1 {
            assert!(
                expected_mirror.iter().any(|&v| v != 0x7e00_7e00),
                "fixture must exercise the half mirror"
            );
        }
        for repeat in 0..3 {
            reset();
            backend.begin();
            assert!(backend.add_rms_norm(
                BufId::Cur,
                BufId::X,
                BufId::O,
                weight,
                width,
                1e-5,
                rows,
                width,
                0,
            ));
            backend.end().unwrap();
            let context = format!(
                "placed={placed} width={width} rows={rows} weight={weight} repeat={repeat}"
            );
            assert_eq!(
                read_bits(backend, BufId::X, count),
                expected_residual,
                "residual {context}"
            );
            assert_eq!(
                read_bits(backend, BufId::Cur, count),
                expected_output,
                "output {context}"
            );
            assert_eq!(
                read_bits(backend, BufId::Xh, count / 2),
                expected_mirror,
                "mirror {context}"
            );
            assert_eq!(
                read_bits(backend, BufId::O, count),
                other.iter().copied().map(f32::to_bits).collect::<Vec<_>>(),
                "source {context}"
            );
        }
    }
}
