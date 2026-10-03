//! Real-weight runtime wiring against the independent, frozen GateUp lab.
#![cfg(feature = "cuda-static")]
use crate::{CudaBackend, context::CudaContext, ffi::*};
use imparo_backend::{Backend, BackendKnobs, BufId, Epilogue};

fn write(id: BufId, x: &[f32]) { crate::correctness::write_f32_checked(id, 0, x).unwrap(); }
fn read(id: BufId, n: usize) -> Vec<f32> {
    let mut x = vec![0.; n];
    crate::correctness::read_f32_checked(id, 0, &mut x).unwrap(); x
}
fn bits(x: &[f32]) -> Vec<u32> { x.iter().map(|v| v.to_bits()).collect() }

#[test]
#[ignore = "requires idle SM86 and IMPARO_TEST_GATEUP_MMVQ_FIXTURE; owns CUDA context"]
fn gpu_moe_gateup_mmvq() {
    const NI: usize = 2048; const NO: usize = 1792; const NE: usize = 32;
    const ROWS: usize = 4; const GUARD: usize = 16;
    const STRIDE: u64 = (NI / 32 * 18 * NO) as u64;
    const POISON: f32 = 1234567.;
    let dir = std::path::PathBuf::from(std::env::var("IMPARO_TEST_GATEUP_MMVQ_FIXTURE").unwrap());
    let weights_dir = dir.parent().unwrap().join("v2-cuda-moe-gateup-mmq-2026-10-02");
    let mut weights = std::fs::read(weights_dir.join("gate.q4.bin")).unwrap();
    let up_off = weights.len() as u64; assert_eq!(up_off, STRIDE * NE as u64);
    weights.extend(std::fs::read(weights_dir.join("up.q4.bin")).unwrap());
    let weights = Box::leak(weights.into_boxed_slice());
    let floats = |name: &str| -> Vec<f32> {
        std::fs::read(dir.join("micro-001").join(name)).unwrap().chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
    };
    let input = floats("input.f32.bin"); let seg = floats("seg.u32.bin");
    let perm = floats("perm.u32.bin");
    assert_eq!(input.len(), NI); assert_eq!(seg.len(), NE + 1);
    unsafe { CudaContext::get().init_weights(weights.as_ptr(), weights.len() as u64, &[]) }.unwrap();
    let be = CudaBackend; be.set_activation(Epilogue::Silu);
    let knob = be.knob_registry().iter().find(|k| k.name == "moe_gateup_mmvq").unwrap();
    let pair = be.knob_registry().iter().find(|k| k.name == "moe_grouped_pair").unwrap();
    struct Restore { apply: fn(u32), old: u32, pair: fn(u32), pair_old: u32 }
    impl Drop for Restore {
        fn drop(&mut self) {
            (self.apply)(self.old); (self.pair)(self.pair_old);
            let _ = crate::gateup_mmvq_api::apply_at_boundary();
            CudaBackend.set_activation(Epilogue::Gelu);
        }
    }
    let _restore = Restore { apply: knob.apply, old: (knob.current)(), pair: pair.apply, pair_old: (pair.current)() };
    (pair.apply)(1);
    assert_eq!(unsafe { imparo_cuda_arena(1 << 20) }, 0);
    let mut offset = 0_u64;
    for (id, n) in [(BufId::Cur, 2 * NI), (BufId::U, 2 * NI),
        (BufId::G, ROWS * NO + GUARD), (BufId::O, ROWS * NO + GUARD),
        (BufId::Model5, ROWS), (BufId::Model7, NE + 1)] {
        assert_eq!(unsafe { imparo_cuda_place(id as u32, offset, (n * 4) as u64) }, 0);
        offset = (offset + (n * 4) as u64 + 255) / 256 * 256;
    }
    let doubled = [input.clone(), input.clone()].concat();
    write(BufId::Cur, &doubled); write(BufId::U, &vec![POISON; 2 * NI]);
    write(BufId::Model7, &seg); write(BufId::Model5, &perm);
    let call = |nt: u32, dst: BufId| be.moe_grouped_pair_with_scratch(
        1, 0, up_off, STRIDE, BufId::Cur, dst, BufId::U, BufId::Model5,
        BufId::Model7, NI as u32, NO as u32, NE as u32, nt, ROWS as u32, true);
    for selected in [0, 1] {
        (knob.apply)(selected); crate::gateup_mmvq_api::apply_at_boundary().unwrap();
        write(BufId::G, &vec![POISON; ROWS * NO + GUARD]);
        be.begin_forward(true); assert!(call(1, BufId::G)); be.end().unwrap();
        let file = if selected == 0 { "baseline.f32.bin" } else { "candidate.f32.bin" };
        assert_eq!(bits(&read(BufId::G, ROWS * NO + GUARD)), bits(&floats(file)[GUARD..]),
            "runtime versus independent full lab output and suffix guard");
    }
    // Partial SEG must compute its valid prefix, preserving unused work rows.
    let partial_seg: Vec<f32> = (0..=NE).map(|i| f32::from_bits(u32::from(i > 0))).collect();
    write(BufId::Model7, &partial_seg);
    for selected in [0, 1] {
        (knob.apply)(selected); crate::gateup_mmvq_api::apply_at_boundary().unwrap();
        write(BufId::G, &vec![POISON; ROWS * NO + GUARD]);
        be.begin_forward(true); assert!(call(1, BufId::G)); be.end().unwrap();
        let got = read(BufId::G, ROWS * NO + GUARD);
        let file = if selected == 0 { "baseline.f32.bin" } else { "candidate.f32.bin" };
        assert_eq!(bits(&got[..NO]), bits(&floats(file)[GUARD..GUARD + NO]));
        assert!(got[NO..].iter().all(|v| *v == POISON));
    }
    write(BufId::Model7, &seg);
    // An invalid token in a covered work row must write zero, not stale data.
    let mut invalid_perm = perm.clone(); invalid_perm[0] = f32::from_bits(u32::MAX);
    write(BufId::Model5, &invalid_perm);
    for selected in [0, 1] {
        (knob.apply)(selected); crate::gateup_mmvq_api::apply_at_boundary().unwrap();
        write(BufId::G, &vec![POISON; ROWS * NO + GUARD]);
        be.begin_forward(true); assert!(call(1, BufId::G)); be.end().unwrap();
        let got = read(BufId::G, ROWS * NO + GUARD);
        assert!(got[..NO].iter().all(|v| v.to_bits() == 0));
        let file = if selected == 0 { "baseline.f32.bin" } else { "candidate.f32.bin" };
        assert_eq!(bits(&got[NO..]), bits(&floats(file)[GUARD + NO..]));
    }
    write(BufId::Model5, &perm);
    (pair.apply)(0); // The independent policy does not require the old pair knob.
    be.begin_forward(true); assert!(call(1, BufId::G)); be.end().unwrap();
    assert_eq!(bits(&read(BufId::G, ROWS * NO)), bits(&floats("candidate.f32.bin")[GUARD..GUARD + ROWS * NO]));
    (pair.apply)(1);
    // Dense and MoE share Q8 storage; the overwrite must invalidate dense cache.
    be.begin_forward(true);
    be.matmat(1, 0, NI as u32, NO as u32, BufId::Cur, BufId::O, 2);
    let dense = read(BufId::O, 2 * NO);
    assert!(call(1, BufId::G));
    be.matmat(1, 0, NI as u32, NO as u32, BufId::Cur, BufId::O, 2);
    assert_eq!(bits(&read(BufId::O, 2 * NO)), bits(&dense), "shared Q8 cache invalidation");
    be.end().unwrap();
    assert_eq!(bits(&read(BufId::Cur, doubled.len())), bits(&doubled));
    assert!(read(BufId::U, 2 * NI).iter().all(|v| *v == POISON));
    assert_eq!(bits(&read(BufId::Model7, seg.len())), bits(&seg));
    assert_eq!(bits(&read(BufId::Model5, perm.len())), bits(&perm));
    // NT2 is outside the policy and must fall back to the original provider.
    be.begin_forward(false); assert!(call(2, BufId::G)); be.end().unwrap();
    assert_eq!(bits(&read(BufId::G, ROWS * NO)), bits(&floats("baseline.f32.bin")[GUARD..GUARD + ROWS * NO]));
    assert_eq!(unsafe { imparo_cuda_place(BufId::Model0 as u32, 0, ((ROWS * NO + GUARD) * 4) as u64) }, 0);
    let before = read(BufId::Model0, ROWS * NO + GUARD);
    be.begin_forward(true); assert!(call(1, BufId::Model0));
    assert!(be.end().is_err(), "handled alias failure must propagate, without fallback");
    assert_eq!(bits(&read(BufId::Model0, before.len())), bits(&before));
    for invalid in [0, 2] {
        be.begin_forward(true);
        assert_ne!(unsafe { crate::gateup_mmvq_api::imparo_cuda_moe_gateup_mmvq_v1(invalid) }, 0);
        assert!(be.end().is_err(), "invalid policy changes must propagate pending errors");
    }
    println!("GateUp MMVQ: full lab equality, partial SEG, invalid PERM zero, independent knob, untouched U, Q8 cache, NT2 fallback, physical alias and boundary errors pass");
}
