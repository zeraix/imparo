//! Attention must round Q before applying the score scale on the FA path.
#![cfg(target_os = "macos")]

use imparo_backend::{Backend as _, BufId};
use imparo_metal::MetalBackend;

#[repr(align(16384))]
struct WeightPage([u8; 16384]);
static WEIGHTS: WeightPage = WeightPage([0; 16384]);

#[test]
#[ignore = "requires a physical Metal device; run explicitly with --ignored"]
fn fa_preserves_small_half_queries_before_score_scaling() {
    imparo_metal::set_attention_head_dims(&[64]);
    imparo_metal::set_attention_kv_widths(&[64]);
    imparo_metal::set_fa_nsg(4);
    imparo_metal::set_kv_types(1, 1);
    // The backend wraps this aligned static page; attention never reads weights.
    unsafe { imparo_metal::init(WEIGHTS.0.as_ptr(), WEIGHTS.0.len() as u64) }.unwrap();
    let backend = MetalBackend;
    for id in [BufId::Q, BufId::K, BufId::V, BufId::Attn] {
        backend.alloc(id, 8 * 64 * 4).unwrap();
    }
    backend.alloc_kv(&[8 * 64 * 2]).unwrap();
    let mut q = vec![0.0_f32; 8 * 64];
    let mut k = vec![0.0_f32; 8 * 64];
    let mut v = vec![0.0_f32; 8 * 64];
    // Nine half subnormal units are exact. Prescaling by 1/8 rounds them to one
    // unit, whereas scaling the float dot product retains all nine eighths.
    q[0] = 9.0 * 2.0_f32.powi(-24);
    q[64] = q[0];
    k[0] = 16384.0;
    k[64] = -16384.0;
    v[..64].fill(1.0);
    v[64..128].fill(-1.0);
    backend.begin();
    backend.write(BufId::K, 0, &k);
    backend.write(BufId::V, 0, &v);
    backend.kv_store(BufId::K, 0, 64, 0, 8, false, 0);
    backend.kv_store(BufId::V, 0, 64, 0, 8, true, 0);
    backend.end().unwrap();
    for scale in [1.0_f32, 0.125] {
        backend.begin();
        backend.write(BufId::Q, 0, &q);
        backend.attention(0, 64, 1, 1, 64, 0, scale, 0, 2, 2, 0);
        backend.end().unwrap();
        let mut actual = vec![0.0_f32; 128];
        backend.read(BufId::Attn, 0, &mut actual);
        let expected = (f64::from(q[0]) * 16384.0 * f64::from(scale)).tanh();
        assert!(actual[..64].iter().all(|&x| (x - 1.0).abs() < 1e-6));
        for &x in &actual[64..] {
            assert!(
                (f64::from(x) - expected).abs() < 1e-7,
                "scale={scale}: {x} != {expected}; Q was rounded after scaling"
            );
        }
        if scale.to_bits() == 0.125_f32.to_bits() {
            // Reproduce the old order through the retained prescaled low-level API.
            backend.begin();
            backend.write(BufId::Q, 0, &q);
            backend.scale(BufId::Q, scale, 2 * 64);
            imparo_metal::attention(0, 64, 1, 1, 64, 0, 0, 2, 2, 0);
            backend.end().unwrap();
            backend.read(BufId::Attn, 0, &mut actual);
            assert!(
                (f64::from(actual[64]) - expected).abs() > 1e-5,
                "fixture must distinguish the old prescale-then-half order"
            );
        }
    }
}
