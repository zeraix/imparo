//! The runtime adapter must match the independently checked single-shape lab.
use crate::{CudaBackend, context::CudaContext, ffi::*};
use imparo_backend::{Backend, BackendKnobs, BufId};

fn write(id: BufId, x: &[f32]) { crate::correctness::write_f32_checked(id, 0, x).unwrap(); }
fn read(id: BufId, n: usize) -> Vec<f32> {
    let mut x = vec![0.; n];
    crate::correctness::read_f32_checked(id, 0, &mut x).unwrap(); x
}
fn bits(x: &[f32]) -> Vec<u32> { x.iter().map(|v| v.to_bits()).collect() }

#[test]
#[ignore = "requires idle SM86 and IMPARO_TEST_DOWN_MMVQ_FIXTURE; owns CUDA context"]
fn gpu_moe_down_mmvq() {
    const NI: usize = 1792; const NO: usize = 2048; const NE: usize = 32;
    const ROWS: usize = 4; const GUARD: usize = 16;
    const STRIDE: u64 = (NI / 32 * 18 * NO) as u64;
    const POISON: f32 = 1234567.;
    let dir = std::path::PathBuf::from(std::env::var("IMPARO_TEST_DOWN_MMVQ_FIXTURE").unwrap());
    let weights_path = dir.parent().unwrap().join("v2-cuda-moe-down-mmq-2026-10-02/down.q4.bin");
    let weights = Box::leak(std::fs::read(weights_path).unwrap().into_boxed_slice());
    assert_eq!(weights.len(), STRIDE as usize * NE);
    let floats = |name: &str| -> Vec<f32> {
        std::fs::read(dir.join("micro-001").join(name)).unwrap().chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
    };
    let input = floats("input.f32.bin"); let seg = floats("seg.u32.bin");
    assert_eq!(input.len(), ROWS * NI); assert_eq!(seg.len(), NE + 1);
    unsafe { CudaContext::get().init_weights(weights.as_ptr(), weights.len() as u64, &[]) }.unwrap();
    let be = CudaBackend;
    let knob = be.knob_registry().iter().find(|k| k.name == "moe_down_mmvq").unwrap();
    struct Restore { apply: fn(u32), old: u32 }
    impl Drop for Restore {
        fn drop(&mut self) { (self.apply)(self.old); let _ = crate::down_mmvq_api::apply_at_boundary(); }
    }
    let _restore = Restore { apply: knob.apply, old: (knob.current)() };
    assert_eq!(unsafe { imparo_cuda_arena(1 << 20) }, 0);
    let mut offset = 0_u64;
    for (id, n) in [(BufId::Cur, ROWS * NI), (BufId::U, ROWS * NI),
        (BufId::G, ROWS * NO + GUARD), (BufId::O, ROWS * NO + GUARD),
        (BufId::Model5, ROWS), (BufId::Model7, NE + 1)] {
        assert_eq!(unsafe { imparo_cuda_place(id as u32, offset, (n * 4) as u64) }, 0);
        offset = (offset + (n * 4) as u64 + 255) / 256 * 256;
    }
    write(BufId::Cur, &input); write(BufId::U, &input); write(BufId::Model7, &seg);
    write(BufId::Model5, &vec![0.; ROWS]);
    let call = |nt: u32, dst: BufId| unsafe {
        crate::moe_api::imparo_cuda_moe_grouped(1, 0, STRIDE, BufId::Cur as u32,
            dst as u32, BufId::Model5 as u32, BufId::Model7 as u32,
            NI as u32, NO as u32, NE as u32, nt, ROWS as u32, 1)
    };
    for selected in [0, 1] {
        (knob.apply)(selected); crate::down_mmvq_api::apply_at_boundary().unwrap();
        write(BufId::O, &vec![POISON; ROWS * NO + GUARD]);
        be.begin_forward(true); assert_eq!(call(1, BufId::O), 0); be.end().unwrap();
        let file = if selected == 0 { "baseline.f32.bin" } else { "candidate.f32.bin" };
        assert_eq!(bits(&read(BufId::O, ROWS * NO + GUARD)), bits(&floats(file)[GUARD..]),
            "runtime versus independently checked full lab output and suffix guard");
    }
    // An invalid pick does not invalidate the remaining experts. Both policies
    // must compute the valid SEG prefix and leave unused work rows untouched.
    let partial_seg: Vec<f32> = (0..=NE).map(|i| f32::from_bits(u32::from(i > 0))).collect();
    write(BufId::Model7, &partial_seg);
    for selected in [0, 1] {
        (knob.apply)(selected); crate::down_mmvq_api::apply_at_boundary().unwrap();
        write(BufId::O, &vec![POISON; ROWS * NO + GUARD]);
        be.begin_forward(true); assert_eq!(call(1, BufId::O), 0); be.end().unwrap();
        let got = read(BufId::O, ROWS * NO + GUARD);
        let file = if selected == 0 { "baseline.f32.bin" } else { "candidate.f32.bin" };
        assert_eq!(bits(&got[..NO]), bits(&floats(file)[GUARD..GUARD + NO]));
        assert!(got[NO..].iter().all(|v| *v == POISON));
    }
    write(BufId::Model7, &seg);
    // Dense and MoE share Q8 storage in one owner; MoE must revoke dense's cache.
    be.begin_forward(true);
    be.matmat(1, 0, NI as u32, NO as u32, BufId::U, BufId::G, 2);
    let dense = read(BufId::G, 2 * NO);
    assert_eq!(call(1, BufId::O), 0);
    be.matmat(1, 0, NI as u32, NO as u32, BufId::U, BufId::G, 2);
    assert_eq!(bits(&read(BufId::G, 2 * NO)), bits(&dense), "shared Q8 cache invalidation");
    be.end().unwrap();
    assert_eq!(bits(&read(BufId::Cur, input.len())), bits(&input));
    assert_eq!(bits(&read(BufId::Model7, seg.len())), bits(&seg));
    // NT2 lies outside this Decode policy and must preserve the old provider.
    be.begin_forward(false); assert_eq!(call(2, BufId::O), 0); be.end().unwrap();
    assert_eq!(bits(&read(BufId::O, ROWS * NO)), bits(&floats("baseline.f32.bin")[GUARD..GUARD + ROWS * NO]));
    // A valid-capacity alias must fail before overwriting source or destination.
    assert_eq!(unsafe { imparo_cuda_place(BufId::Model0 as u32, 0, ((ROWS * NO + GUARD) * 4) as u64) }, 0);
    let before = read(BufId::Model0, ROWS * NO + GUARD);
    be.begin_forward(true); assert_ne!(call(1, BufId::Model0), 0);
    assert!(be.end().is_err(), "native rejection must propagate pending error");
    assert_eq!(bits(&read(BufId::Model0, before.len())), bits(&before));
    for invalid in [0, 2] {
        be.begin_forward(true);
        assert_ne!(unsafe { crate::down_mmvq_api::imparo_cuda_moe_down_mmvq_v1(invalid) }, 0);
        assert!(be.end().is_err(), "invalid policy change cannot be silently ignored");
    }
    println!("Down MMVQ: full native/lab equality, guards, source/SEG unchanged, shared Q8 cache, NT2 fallback, physical alias refusal, boundary errors pass");
}
