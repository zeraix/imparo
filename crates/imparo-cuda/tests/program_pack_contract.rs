//! Source-level public contract for the PR-E CUDA Program bridge.
//!
//! Live Driver/Graph tests remain SM86 hardware gates. These checks make ABI drift,
//! accidental per-kernel exports, split CUDA ownership, and premature PR-G identity
//! activation fail on every host, including the detached public-tree build.

const FFI: &str = include_str!("../src/ffi.rs");
const RUST_BRIDGE: &str = include_str!("../src/program_pack/mod.rs");
const PROFILE_IDENTITY: &str = include_str!("../src/program_pack/profile_identity.rs");
const REGISTRY: &str = include_str!("../src/program_pack/registry.rs");
const HEADER: &str = include_str!("../native/program_pack.h");
const NATIVE: &str = include_str!("../native/program_pack.cu");
const UNITY: &str = include_str!("../native/imparo_cuda.cu");
const EXPORTS: &str = include_str!("../native/imparo_cuda.def");
const CATALOG: &str = include_str!("../cuda-sm.json");

const PROGRAM_EXPORTS: [&str; 6] = [
    "imparo_cuda_program_pack_install",
    "imparo_cuda_program_catalog_identity",
    "imparo_cuda_program_bind",
    "imparo_cuda_program_freeze",
    "imparo_cuda_program_launch",
    "imparo_cuda_program_reset",
];

#[test]
fn backend_abi_26_has_exactly_six_generic_program_exports() {
    let catalog: serde_json::Value = serde_json::from_str(CATALOG).unwrap();
    assert_eq!(catalog["backend_abi"].as_u64(), Some(26));
    assert_eq!(imparo_cuda::CUDA_BACKEND_ABI, 26);

    let actual: Vec<_> = EXPORTS
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with("imparo_cuda_program_"))
        .collect();
    assert_eq!(actual, PROGRAM_EXPORTS);
    for symbol in PROGRAM_EXPORTS {
        assert!(FFI.contains(symbol), "Rust FFI is missing {symbol}");
        assert!(NATIVE.contains(symbol), "native bridge is missing {symbol}");
    }
}

#[test]
fn program_v1_wire_sizes_and_offsets_are_frozen_on_both_sides() {
    for marker in [
        "IMPARO_CUDA_PROGRAM_ABI_V1 = 1",
        "sizeof(ImparoCudaProgramModuleWire) == 56",
        "sizeof(ImparoCudaProgramArgumentDescriptorWire) == 16",
        "sizeof(ImparoCudaProgramFunctionWire) == 200",
        "sizeof(ImparoCudaProgramPackWire) == 160",
        "sizeof(ImparoCudaProgramCatalogIdentityWire) == 128",
        "sizeof(ImparoCudaProgramArgumentWire) == 16",
        "sizeof(ImparoCudaProgramLaunchWire) == 72",
        "offsetof(ImparoCudaProgramFunctionWire, symbol) == 104",
        "offsetof(ImparoCudaProgramFunctionWire, argument_schema) == 120",
        "offsetof(ImparoCudaProgramFunctionWire, local_memory_bytes_max) == 152",
        "offsetof(ImparoCudaProgramFunctionWire, builtin_variant_id) == 168",
        "offsetof(ImparoCudaProgramPackWire, modules) == 32",
        "offsetof(ImparoCudaProgramPackWire, functions) == 48",
        "offsetof(ImparoCudaProgramLaunchWire, arguments) == 56",
    ] {
        assert!(
            HEADER.contains(marker),
            "native wire contract lost {marker}"
        );
    }
    for marker in [
        "pub(crate) const CUDA_PROGRAM_ABI_V1: u32 = 1",
        "size_of::<ProgramModuleWire>()",
        "size_of::<ProgramFunctionWire>()",
        "size_of::<ProgramPackWire>()",
        "size_of::<ProgramCatalogIdentityWire>()",
        "size_of::<ProgramArgumentWire>()",
        "size_of::<ProgramLaunchWire>()",
    ] {
        assert!(FFI.contains(marker), "Rust wire contract lost {marker}");
    }
}

#[test]
fn graph_update_source_one_is_closed_and_engine_owned() {
    for marker in [
        "IMPARO_CUDA_PROGRAM_UPDATE_NONE = 0",
        "IMPARO_CUDA_PROGRAM_UPDATE_DECODE_START_POS_U32 = 1",
        "descriptor.update_source != kProgramUpdateDecodeStartPosU32",
        "descriptor.kind != IMPARO_CUDA_PROGRAM_SCALAR_U32",
    ] {
        assert!(
            HEADER.contains(marker) || NATIVE.contains(marker),
            "Program graph-update contract lost {marker}"
        );
    }
}

#[test]
fn bridge_is_one_data_only_unity_boundary() {
    assert!(UNITY.contains("#include \"program_pack.cu\""));
    for marker in [
        "cuDevicePrimaryCtxRetain",
        "ensure_cuda_runtime()",
        "current != catalog.primary_context",
        "cuModuleLoadDataEx",
        "cuModuleGetFunction",
        "cuFuncGetAttribute",
        "cuLaunchKernel",
        "reinterpret_cast<CUstream>(g.stream)",
        "destroy_decode_graph_checked()",
    ] {
        assert!(NATIVE.contains(marker), "native bridge lost {marker}");
    }
    for forbidden in ["Py_", "torch", "triton", "LoadLibrary(pack", "dlopen(pack"] {
        assert!(
            !NATIVE.contains(forbidden),
            "native Program bridge gained executable host dependency {forbidden}"
        );
    }
}

#[test]
fn pre_pr_g_identity_is_fail_closed_and_native_fallback_remains() {
    for marker in [
        "identity_ready: false",
        "pack_set_sha256: [0; 32]",
        "candidate_catalog_sha256: [0; 32]",
        "eligible_candidate_set_sha256: [0; 32]",
    ] {
        assert!(
            PROFILE_IDENTITY.contains(marker),
            "PR-G identity was activated without its frozen contract: {marker}"
        );
    }
    assert!(REGISTRY.contains("profile_identity::pending_identity()"));
    assert!(RUST_BRIDGE.contains("pub fn shutdown() -> Result<(), String>"));
    assert!(NATIVE.contains("pack->identity_ready != 0"));
    assert!(NATIVE.contains("imparo_cuda_program_reset(void) noexcept"));
}

#[test]
fn empty_dormant_catalog_does_not_block_runtime_cuda_graph_nodes() {
    let classifier = NATIVE
        .split_once("bool add_program_dynamic_graph_node(")
        .expect("Program graph classifier")
        .1
        .split_once("int update_program_dynamic_graph_node(")
        .expect("Program graph updater boundary")
        .0;
    assert!(classifier.contains("if (program_catalog.poisoned) return false;"));
    assert!(classifier.contains("return program_catalog.modules.empty()"));
    assert!(classifier.contains("&& program_catalog.functions.empty()"));
    assert!(classifier.contains("&& program_catalog.bindings.empty();"));
}
