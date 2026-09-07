//! Source-level contract checks for the ABI-25-compatible Step 1 extension.
//!
//! Live CUDA tests are separately gated to SM86; these checks keep static exports,
//! the Windows export table, optional dynamic loading and graph isolation in lockstep.

#[cfg(any(feature = "cuda-static", feature = "cuda-dynamic"))]
use imparo_backend::BackendKnobs as _;

const FFI: &str = include_str!("../src/ffi.rs");
const BACKEND: &str = include_str!("../src/backend_impl.rs");
const NATIVE: &str = include_str!("../native/imparo_cuda.cu");
const EXPORTS: &str = include_str!("../native/imparo_cuda.def");
const CATALOG: &str = include_str!("../cuda-sm.json");
const PROGRAM_PROFILE_IDENTITY: &str =
    include_str!("../src/program_pack/profile_identity.rs");

#[test]
fn step1_surface_is_optional_without_pretending_partial_support() {
    for symbol in [
        "imparo_cuda_last_gpu_us",
        "imparo_cuda_set_tuner_mode",
        "imparo_cuda_device_profile",
        "imparo_cuda_probe",
        "imparo_cuda_dispatch_proof_reset",
        "imparo_cuda_dispatch_expectation",
        "imparo_cuda_dispatch_proof",
    ] {
        assert!(FFI.contains(symbol));
        assert!(NATIVE.contains(symbol));
        assert!(EXPORTS.contains(symbol));
    }
    assert!(FFI.contains("dispatch_proof_reset"));
    assert!(FFI.contains("_ => (None, None, None, None, None, None, None)"));
    assert!(FFI.contains("map_or(LOAD_ERROR"));
    assert!(NATIVE.contains("proof.expected_knob_mask = g.proof_expected_knob_mask"));
    assert!(NATIVE.contains("proof.observed_knob_mask = g.proof_observed_knob_mask"));
    assert!(NATIVE.contains("uint32_t tuner_knob(uint32_t slot)"));
    assert!(BACKEND.contains("proof.observed_knob_mask & proof.expected_knob_mask"));
}

#[test]
fn tuner_mode_is_fail_closed_and_graph_isolated() {
    assert!(BACKEND.contains("fn tuner_requires_device_timing(&self) -> bool"));
    assert!(NATIVE.contains("!g.tuner_mode && g.decode_prepared"));
    let setter = NATIVE
        .split("extern \"C\" int imparo_cuda_set_tuner_mode")
        .nth(1)
        .expect("native tuner-mode setter");
    assert!(setter.contains("cudaStreamSynchronize"));
    assert!(setter.contains("destroy_decode_graph"));
    assert!(NATIVE.contains("cudaStreamCreateWithPriority"));
    assert!(NATIVE.contains("least_priority"));
}

#[test]
fn step1_preserves_current_e4b_identity_and_dormant_program_pack() {
    let catalog: serde_json::Value = serde_json::from_str(CATALOG).unwrap();
    assert_eq!(catalog["backend_abi"].as_u64(), Some(26));
    assert_eq!(imparo_cuda::CUDA_BACKEND_ABI, 26);
    #[cfg(any(feature = "cuda-static", feature = "cuda-dynamic"))]
    assert_eq!(imparo_cuda::CudaBackend.space_version(), 34);
    assert!(PROGRAM_PROFILE_IDENTITY.contains("identity_ready: false"));
}
