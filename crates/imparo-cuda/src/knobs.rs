//! The CUDA knob registry (task #19's demonstration). Candidate VALUES ARE
//! PLACEHOLDERS a CUDA developer refines on real hardware; the point is the shape:
//! a backend declares its knobs here -- name, category, candidates, hooks, whether
//! the tiny-size screen must gate them -- and the shared tuner machinery sweeps
//! them without any CUDA-specific code in shared paths.
//!
//! Slots map to the .cu's knob table (imparo_cuda_set_knob); "consumed by" names
//! the kernel launch a CUDA dev wires each slot into.

use imparo_backend::{
    BackendKnobs, KnobCategory, KnobDecl, SweepKind as Sw,
    Workload as Wl,
};

unsafe extern "C" {
    fn imparo_cuda_set_knob(idx: u32, v: u32);
    fn imparo_cuda_knob(idx: u32) -> u32;
}

macro_rules! slot {
    ($idx:literal) => {
        (
            |v| unsafe { imparo_cuda_set_knob($idx, v) },
            || unsafe { imparo_cuda_knob($idx) },
        )
    };
}

// The GEMM knob set. CUDA's category-4 space is far larger than Metal's -- block
// geometry, k-split, pipeline stages, vector widths, tensor-core routing -- which
// is exactly why the registry exists: adding a knob is one entry here.
pub static CUDA_KNOBS: &[KnobDecl] = &[
    KnobDecl {
        name: "gemm_threads",
        legal: None,
        bit_affecting: false,
        derive: None,
        cross_check: None,
        applies: None,
        tuple: None,
        category: KnobCategory::Benched,
        values: &[64, 128, 256, 512],
        apply: slot!(0).0,
        current: slot!(0).1,
        // block geometry can spill registers exactly like Metal's shape 7
        screened: true,
        sweep: Sw::Values,
        workload: Wl::DecodeMix,
    },
    KnobDecl {
        name: "gemm_rows_per_block",
        legal: None,
        bit_affecting: false,
        derive: None,
        cross_check: None,
        applies: None,
        tuple: None,
        category: KnobCategory::Benched,
        values: &[32, 64, 128],
        apply: slot!(1).0,
        current: slot!(1).1,
        screened: true,
        sweep: Sw::Values,
        workload: Wl::DecodeMix,
    },
    KnobDecl {
        name: "gemm_k_chunk",
        legal: None,
        bit_affecting: false,
        derive: None,
        cross_check: None,
        applies: None,
        tuple: None,
        category: KnobCategory::Benched,
        values: &[32, 64, 128],
        apply: slot!(2).0,
        current: slot!(2).1,
        screened: true,
        sweep: Sw::Values,
        workload: Wl::DecodeMix,
    },
    KnobDecl {
        name: "gemm_stages",
        legal: None,
        bit_affecting: false,
        derive: None,
        cross_check: None,
        applies: None,
        tuple: None,
        category: KnobCategory::Benched,
        values: &[1, 2, 3],
        apply: slot!(3).0,
        current: slot!(3).1,
        screened: true,
        sweep: Sw::Values,
        workload: Wl::DecodeMix,
    },
    KnobDecl {
        name: "gemm_vec_width",
        legal: None,
        bit_affecting: false,
        derive: None,
        cross_check: None,
        applies: None,
        tuple: None,
        category: KnobCategory::Benched,
        values: &[1, 2, 4],
        apply: slot!(4).0,
        current: slot!(4).1,
        screened: false,
        sweep: Sw::Values,
        workload: Wl::DecodeMix,
    },
    KnobDecl {
        name: "gemm_tensor_cores",
        legal: None,
        bit_affecting: false,
        derive: None,
        cross_check: None,
        applies: None,
        tuple: None,
        category: KnobCategory::Benched,
        values: &[0, 1],
        apply: slot!(5).0,
        current: slot!(5).1,
        screened: true,
        sweep: Sw::Values,
        workload: Wl::DecodeMix,
    },
    KnobDecl {
        name: "gemv_warps",
        legal: None,
        bit_affecting: false,
        derive: None,
        cross_check: None,
        applies: None,
        tuple: None,
        category: KnobCategory::Benched,
        values: &[2, 4, 8],
        apply: slot!(6).0,
        current: slot!(6).1,
        screened: false,
        sweep: Sw::Values,
        workload: Wl::DecodeMix,
    },
    KnobDecl {
        name: "attn_threads",
        legal: None,
        bit_affecting: false,
        derive: None,
        cross_check: None,
        applies: None,
        tuple: None,
        category: KnobCategory::Benched,
        values: &[64, 128, 256],
        apply: slot!(7).0,
        current: slot!(7).1,
        screened: true,
        sweep: Sw::Values,
        workload: Wl::AttentionDecode,
    },
];

impl BackendKnobs for crate::CudaBackend {
    fn knob_registry(&self) -> &'static [KnobDecl] {
        CUDA_KNOBS
    }
    /// The CUDA search space starts at 1 and versions INDEPENDENTLY of Metal's:
    /// a bump here never invalidates a Metal config (the stored fingerprint
    /// carries device_tag + this number).
    fn space_version(&self) -> u32 {
        1
    }
}
