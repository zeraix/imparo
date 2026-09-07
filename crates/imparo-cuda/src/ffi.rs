//! One Rust call surface for statically linked and runtime-loaded CUDA backends.

use core::ffi::c_void;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct RuntimeIdentityWire {
    pub struct_bytes: u32,
    pub backend_abi: u32,
    pub device_sm: u32,
    pub driver_version: u32,
    pub runtime_version: u32,
    pub device_uuid: [u8; 16],
    pub backend_build_sha256: [u8; 32],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct HostProfileWire {
    pub struct_bytes: u32,
    pub reserved: u32,
    pub available_host_bytes: u64,
    pub pinned_h2d_bytes_per_second: u64,
    pub pinned_d2h_bytes_per_second: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct DeviceProfileWire {
    pub struct_bytes: u32,
    pub max_threads: u32,
    pub threadgroup_bytes: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct DispatchProofWire {
    pub struct_bytes: u32,
    pub family: u32,
    pub choice_epoch: u64,
    pub observed_choice_epoch: u64,
    pub dispatches: u64,
    pub variant: u64,
    pub expected_knob_mask: u64,
    pub observed_knob_mask: u64,
}

/// Data-only CUDA Program ABI implemented by native ABI 26.
pub(crate) const CUDA_PROGRAM_ABI_V1: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ProgramModuleWire {
    pub struct_size: u32,
    pub abi_version: u32,
    pub module_id: [u8; 32],
    pub image: *const u8,
    pub image_bytes: u64,
}

pub(crate) const PROGRAM_GRAPH_CAPTURE_STATIC: u32 = 0;
pub(crate) const PROGRAM_GRAPH_REPLAY_UPDATE: u32 = 1;
pub(crate) const PROGRAM_UPDATE_NONE: u32 = 0;
pub(crate) const PROGRAM_UPDATE_DECODE_START_POS_U32: u32 = 1;

/// Borrowed install metadata. Native ABI26 must validate and copy this POD before the
/// synchronous install call returns; `update_source` is an adapter-owned stable ID, not
/// a callback or host address.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ProgramArgumentDescriptorWire {
    pub kind: u32,
    pub graph_mutability: u32,
    pub update_source: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ProgramFunctionWire {
    pub struct_size: u32,
    pub abi_version: u32,
    pub choice_group_id: [u8; 32],
    pub variant_id: [u8; 32],
    pub module_id: [u8; 32],
    pub symbol: *const u8,
    pub symbol_bytes: u32,
    pub argument_count: u32,
    /// Borrowed for synchronous install; points to exactly `argument_count` descriptors.
    pub argument_schema: *const ProgramArgumentDescriptorWire,
    pub block_x: u32,
    pub block_y: u32,
    pub block_z: u32,
    pub dynamic_shared_bytes_max: u32,
    pub registers_per_thread_max: u32,
    pub static_shared_bytes_max: u32,
    pub local_memory_bytes_max: u64,
    pub threads_per_block_max: u32,
    pub graph_capture: u32,
    /// Exact engine-owned built-in fallback marker for this function's choice group.
    pub builtin_variant_id: [u8; 32],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ProgramPackWire {
    pub struct_size: u32,
    pub abi_version: u32,
    pub program_pack_abi: u32,
    pub kernel_contract_abi: u32,
    pub backend_abi_min: u32,
    pub backend_abi_max_exclusive: u32,
    pub target_sm: u32,
    pub driver_min: u32,
    pub modules: *const ProgramModuleWire,
    pub module_count: u32,
    pub functions: *const ProgramFunctionWire,
    pub function_count: u32,
    /// Reserved for PR-G's frozen identity encoding. Must be zero in PR-E.
    pub identity_ready: u32,
    pub pack_set_sha256: [u8; 32],
    pub candidate_catalog_sha256: [u8; 32],
    pub eligible_candidate_set_sha256: [u8; 32],
}

#[repr(C)]
#[allow(dead_code)] // ABI26 surface; PR-G constructs this for frozen catalog identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ProgramCatalogIdentityWire {
    pub struct_size: u32,
    pub abi_version: u32,
    pub generation: u64,
    pub frozen: u32,
    pub module_count: u32,
    pub function_count: u32,
    pub reserved: u32,
    pub pack_set_sha256: [u8; 32],
    pub candidate_catalog_sha256: [u8; 32],
    pub eligible_candidate_set_sha256: [u8; 32],
}

#[allow(dead_code)] // Frozen ABI values are required before the first launch adapter.
pub(crate) const PROGRAM_ARGUMENT_TENSOR_PTR: u32 = 1;
#[allow(dead_code)]
pub(crate) const PROGRAM_ARGUMENT_STATE_PTR: u32 = 2;
#[allow(dead_code)]
pub(crate) const PROGRAM_ARGUMENT_SCRATCH_PTR: u32 = 3;
#[allow(dead_code)]
pub(crate) const PROGRAM_ARGUMENT_SCALAR_I32: u32 = 4;
#[allow(dead_code)]
pub(crate) const PROGRAM_ARGUMENT_SCALAR_U32: u32 = 5;
#[allow(dead_code)]
pub(crate) const PROGRAM_ARGUMENT_SCALAR_U64: u32 = 6;
#[allow(dead_code)]
pub(crate) const PROGRAM_ARGUMENT_SCALAR_F32: u32 = 7;
#[allow(dead_code)]
pub(crate) const PROGRAM_ARGUMENT_MANIFEST_U32: u32 = 8;

#[repr(C)]
#[allow(dead_code)] // Constructed by contract adapters added after the bridge.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ProgramArgumentWire {
    pub kind: u32,
    pub reserved: u32,
    pub value: u64,
}

#[repr(C)]
#[allow(dead_code)] // Constructed by contract adapters added after the bridge.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ProgramLaunchWire {
    pub struct_size: u32,
    pub abi_version: u32,
    pub variant_id: [u8; 32],
    pub grid_x: u32,
    pub grid_y: u32,
    pub grid_z: u32,
    pub dynamic_shared_bytes: u32,
    pub arguments: *const ProgramArgumentWire,
    pub argument_count: u32,
    pub reserved: u32,
}

#[cfg(test)]
mod wire_layout_tests {
    use super::{
        CUDA_PROGRAM_ABI_V1, DeviceProfileWire, DispatchProofWire, HostProfileWire,
        PROGRAM_ARGUMENT_MANIFEST_U32, PROGRAM_ARGUMENT_SCALAR_F32,
        PROGRAM_ARGUMENT_SCALAR_I32, PROGRAM_ARGUMENT_SCALAR_U32,
        PROGRAM_ARGUMENT_SCALAR_U64, PROGRAM_ARGUMENT_SCRATCH_PTR,
        PROGRAM_ARGUMENT_STATE_PTR, PROGRAM_ARGUMENT_TENSOR_PTR,
        PROGRAM_GRAPH_CAPTURE_STATIC, PROGRAM_GRAPH_REPLAY_UPDATE,
        PROGRAM_UPDATE_DECODE_START_POS_U32, PROGRAM_UPDATE_NONE,
        ProgramArgumentDescriptorWire, ProgramArgumentWire, ProgramCatalogIdentityWire,
        ProgramFunctionWire, ProgramLaunchWire, ProgramModuleWire, ProgramPackWire,
    };

    #[test]
    fn dispatch_proof_wire_layout_is_stable_optional_abi_25_extension() {
        assert_eq!(size_of::<DispatchProofWire>(), 56);
        assert_eq!(std::mem::offset_of!(DispatchProofWire, struct_bytes), 0);
        assert_eq!(std::mem::offset_of!(DispatchProofWire, family), 4);
        assert_eq!(std::mem::offset_of!(DispatchProofWire, choice_epoch), 8);
        assert_eq!(
            std::mem::offset_of!(DispatchProofWire, observed_choice_epoch),
            16
        );
        assert_eq!(std::mem::offset_of!(DispatchProofWire, dispatches), 24);
        assert_eq!(std::mem::offset_of!(DispatchProofWire, variant), 32);
        assert_eq!(
            std::mem::offset_of!(DispatchProofWire, expected_knob_mask),
            40
        );
        assert_eq!(
            std::mem::offset_of!(DispatchProofWire, observed_knob_mask),
            48
        );
    }

    #[test]
    fn device_profile_wire_layout_is_stable_optional_abi_25_extension() {
        assert_eq!(size_of::<DeviceProfileWire>(), 16);
        assert_eq!(std::mem::offset_of!(DeviceProfileWire, struct_bytes), 0);
        assert_eq!(std::mem::offset_of!(DeviceProfileWire, max_threads), 4);
        assert_eq!(
            std::mem::offset_of!(DeviceProfileWire, threadgroup_bytes),
            8
        );
    }

    #[test]
    fn host_profile_wire_layout_remains_stable_in_abi_25() {
        assert_eq!(size_of::<HostProfileWire>(), 32);
        assert_eq!(std::mem::offset_of!(HostProfileWire, struct_bytes), 0);
        assert_eq!(std::mem::offset_of!(HostProfileWire, reserved), 4);
        assert_eq!(
            std::mem::offset_of!(HostProfileWire, available_host_bytes),
            8
        );
        assert_eq!(
            std::mem::offset_of!(HostProfileWire, pinned_h2d_bytes_per_second),
            16
        );
        assert_eq!(
            std::mem::offset_of!(HostProfileWire, pinned_d2h_bytes_per_second),
            24
        );
    }

    #[test]
    fn program_module_and_function_wire_layouts_match_native_v1() {
        assert_eq!(CUDA_PROGRAM_ABI_V1, 1);
        assert_eq!(
            (
                size_of::<ProgramModuleWire>(),
                align_of::<ProgramModuleWire>()
            ),
            (56, 8)
        );
        for (actual, expected) in [
            (std::mem::offset_of!(ProgramModuleWire, struct_size), 0),
            (std::mem::offset_of!(ProgramModuleWire, abi_version), 4),
            (std::mem::offset_of!(ProgramModuleWire, module_id), 8),
            (std::mem::offset_of!(ProgramModuleWire, image), 40),
            (std::mem::offset_of!(ProgramModuleWire, image_bytes), 48),
        ] {
            assert_eq!(actual, expected);
        }

        assert_eq!(PROGRAM_GRAPH_CAPTURE_STATIC, 0);
        assert_eq!(PROGRAM_GRAPH_REPLAY_UPDATE, 1);
        assert_eq!(PROGRAM_UPDATE_NONE, 0);
        assert_eq!(PROGRAM_UPDATE_DECODE_START_POS_U32, 1);
        assert_eq!(
            (
                size_of::<ProgramArgumentDescriptorWire>(),
                align_of::<ProgramArgumentDescriptorWire>()
            ),
            (16, 4)
        );
        for (actual, expected) in [
            (std::mem::offset_of!(ProgramArgumentDescriptorWire, kind), 0),
            (
                std::mem::offset_of!(ProgramArgumentDescriptorWire, graph_mutability),
                4,
            ),
            (
                std::mem::offset_of!(ProgramArgumentDescriptorWire, update_source),
                8,
            ),
            (
                std::mem::offset_of!(ProgramArgumentDescriptorWire, reserved),
                12,
            ),
        ] {
            assert_eq!(actual, expected);
        }

        // The declared fields end at byte 200. Keeping every offset here catches
        // accidental implicit padding as well as Rust/C field-order drift.
        assert_eq!(
            (
                size_of::<ProgramFunctionWire>(),
                align_of::<ProgramFunctionWire>()
            ),
            (200, 8)
        );
        for (actual, expected) in [
            (std::mem::offset_of!(ProgramFunctionWire, struct_size), 0),
            (std::mem::offset_of!(ProgramFunctionWire, abi_version), 4),
            (
                std::mem::offset_of!(ProgramFunctionWire, choice_group_id),
                8,
            ),
            (std::mem::offset_of!(ProgramFunctionWire, variant_id), 40),
            (std::mem::offset_of!(ProgramFunctionWire, module_id), 72),
            (std::mem::offset_of!(ProgramFunctionWire, symbol), 104),
            (std::mem::offset_of!(ProgramFunctionWire, symbol_bytes), 112),
            (
                std::mem::offset_of!(ProgramFunctionWire, argument_count),
                116,
            ),
            (
                std::mem::offset_of!(ProgramFunctionWire, argument_schema),
                120,
            ),
            (std::mem::offset_of!(ProgramFunctionWire, block_x), 128),
            (std::mem::offset_of!(ProgramFunctionWire, block_y), 132),
            (std::mem::offset_of!(ProgramFunctionWire, block_z), 136),
            (
                std::mem::offset_of!(ProgramFunctionWire, dynamic_shared_bytes_max),
                140,
            ),
            (
                std::mem::offset_of!(ProgramFunctionWire, registers_per_thread_max),
                144,
            ),
            (
                std::mem::offset_of!(ProgramFunctionWire, static_shared_bytes_max),
                148,
            ),
            (
                std::mem::offset_of!(ProgramFunctionWire, local_memory_bytes_max),
                152,
            ),
            (
                std::mem::offset_of!(ProgramFunctionWire, threads_per_block_max),
                160,
            ),
            (
                std::mem::offset_of!(ProgramFunctionWire, graph_capture),
                164,
            ),
            (
                std::mem::offset_of!(ProgramFunctionWire, builtin_variant_id),
                168,
            ),
        ] {
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn program_pack_and_catalog_wire_layouts_match_native_v1() {
        assert_eq!(
            (size_of::<ProgramPackWire>(), align_of::<ProgramPackWire>()),
            (160, 8)
        );
        for (actual, expected) in [
            (std::mem::offset_of!(ProgramPackWire, struct_size), 0),
            (std::mem::offset_of!(ProgramPackWire, abi_version), 4),
            (std::mem::offset_of!(ProgramPackWire, program_pack_abi), 8),
            (
                std::mem::offset_of!(ProgramPackWire, kernel_contract_abi),
                12,
            ),
            (std::mem::offset_of!(ProgramPackWire, backend_abi_min), 16),
            (
                std::mem::offset_of!(ProgramPackWire, backend_abi_max_exclusive),
                20,
            ),
            (std::mem::offset_of!(ProgramPackWire, target_sm), 24),
            (std::mem::offset_of!(ProgramPackWire, driver_min), 28),
            (std::mem::offset_of!(ProgramPackWire, modules), 32),
            (std::mem::offset_of!(ProgramPackWire, module_count), 40),
            (std::mem::offset_of!(ProgramPackWire, functions), 48),
            (std::mem::offset_of!(ProgramPackWire, function_count), 56),
            (std::mem::offset_of!(ProgramPackWire, identity_ready), 60),
            (std::mem::offset_of!(ProgramPackWire, pack_set_sha256), 64),
            (
                std::mem::offset_of!(ProgramPackWire, candidate_catalog_sha256),
                96,
            ),
            (
                std::mem::offset_of!(ProgramPackWire, eligible_candidate_set_sha256),
                128,
            ),
        ] {
            assert_eq!(actual, expected);
        }

        assert_eq!(
            (
                size_of::<ProgramCatalogIdentityWire>(),
                align_of::<ProgramCatalogIdentityWire>()
            ),
            (128, 8)
        );
        for (actual, expected) in [
            (
                std::mem::offset_of!(ProgramCatalogIdentityWire, struct_size),
                0,
            ),
            (
                std::mem::offset_of!(ProgramCatalogIdentityWire, abi_version),
                4,
            ),
            (
                std::mem::offset_of!(ProgramCatalogIdentityWire, generation),
                8,
            ),
            (std::mem::offset_of!(ProgramCatalogIdentityWire, frozen), 16),
            (
                std::mem::offset_of!(ProgramCatalogIdentityWire, module_count),
                20,
            ),
            (
                std::mem::offset_of!(ProgramCatalogIdentityWire, function_count),
                24,
            ),
            (
                std::mem::offset_of!(ProgramCatalogIdentityWire, reserved),
                28,
            ),
            (
                std::mem::offset_of!(ProgramCatalogIdentityWire, pack_set_sha256),
                32,
            ),
            (
                std::mem::offset_of!(
                    ProgramCatalogIdentityWire,
                    candidate_catalog_sha256
                ),
                64,
            ),
            (
                std::mem::offset_of!(
                    ProgramCatalogIdentityWire,
                    eligible_candidate_set_sha256
                ),
                96,
            ),
        ] {
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn program_argument_and_launch_wire_layouts_match_native_v1() {
        assert_eq!(
            [
                PROGRAM_ARGUMENT_TENSOR_PTR,
                PROGRAM_ARGUMENT_STATE_PTR,
                PROGRAM_ARGUMENT_SCRATCH_PTR,
                PROGRAM_ARGUMENT_SCALAR_I32,
                PROGRAM_ARGUMENT_SCALAR_U32,
                PROGRAM_ARGUMENT_SCALAR_U64,
                PROGRAM_ARGUMENT_SCALAR_F32,
                PROGRAM_ARGUMENT_MANIFEST_U32,
            ],
            [1, 2, 3, 4, 5, 6, 7, 8]
        );
        assert_eq!(
            (
                size_of::<ProgramArgumentWire>(),
                align_of::<ProgramArgumentWire>()
            ),
            (16, 8)
        );
        assert_eq!(std::mem::offset_of!(ProgramArgumentWire, kind), 0);
        assert_eq!(std::mem::offset_of!(ProgramArgumentWire, reserved), 4);
        assert_eq!(std::mem::offset_of!(ProgramArgumentWire, value), 8);

        assert_eq!(
            (
                size_of::<ProgramLaunchWire>(),
                align_of::<ProgramLaunchWire>()
            ),
            (72, 8)
        );
        for (actual, expected) in [
            (std::mem::offset_of!(ProgramLaunchWire, struct_size), 0),
            (std::mem::offset_of!(ProgramLaunchWire, abi_version), 4),
            (std::mem::offset_of!(ProgramLaunchWire, variant_id), 8),
            (std::mem::offset_of!(ProgramLaunchWire, grid_x), 40),
            (std::mem::offset_of!(ProgramLaunchWire, grid_y), 44),
            (std::mem::offset_of!(ProgramLaunchWire, grid_z), 48),
            (
                std::mem::offset_of!(ProgramLaunchWire, dynamic_shared_bytes),
                52,
            ),
            (std::mem::offset_of!(ProgramLaunchWire, arguments), 56),
            (std::mem::offset_of!(ProgramLaunchWire, argument_count), 64),
            (std::mem::offset_of!(ProgramLaunchWire, reserved), 68),
        ] {
            assert_eq!(actual, expected);
        }
    }
}

#[cfg(feature = "cuda-static")]
mod imp {
    use super::c_void;
    unsafe extern "C" {
        pub(crate) fn imparo_cuda_init(
            weights: *const c_void,
            len: u64,
            streamed: *const c_void,
            streamed_count: u32,
        ) -> i32;
        pub(crate) fn imparo_cuda_prepare_quantized_weight_cache(
            specs: *const imparo_backend::QuantizedWeightPrepack,
            count: u32,
        ) -> i32;
        pub(crate) fn imparo_cuda_memory_info(
            free: *mut u64,
            total: *mut u64,
            allocated: *mut u64,
        ) -> i32;
        pub(crate) fn imparo_cuda_weights_resident() -> u32;
        pub(crate) fn imparo_cuda_runtime_identity(
            out: *mut super::RuntimeIdentityWire,
            out_bytes: u32,
        ) -> i32;
        pub(crate) fn imparo_cuda_set_kv_types(k: u32, v: u32);
        pub(crate) fn imparo_cuda_device_tag(dst: *mut u8, len: u32) -> u32;
        pub(crate) fn imparo_cuda_begin();
        pub(crate) fn imparo_cuda_begin_forward(decode: u32);
        pub(crate) fn imparo_cuda_set_batch_geometry(
            absolute_start: u64,
            active_tokens: u32,
            phase: u32,
        ) -> i32;
        pub(crate) fn imparo_cuda_decode_prepare(
            token: u32,
            start_pos: u32,
            argmax: u32,
        ) -> i32;
        pub(crate) fn imparo_cuda_prefill_prepare(
            tokens: *const u32,
            count: u32,
            start_pos: u32,
            argmax: u32,
        ) -> i32;
        pub(crate) fn imparo_cuda_flush();
        pub(crate) fn imparo_cuda_end() -> i32;
        pub(crate) fn imparo_cuda_last_gpu_us() -> f64;
        pub(crate) fn imparo_cuda_set_tuner_mode(enabled: u32) -> i32;
        pub(crate) fn imparo_cuda_device_profile(
            out: *mut super::DeviceProfileWire,
            out_bytes: u32,
        ) -> i32;
        pub(crate) fn imparo_cuda_probe(
            kind: u32,
            a: u64,
            b: u32,
            c: u32,
            d: u32,
            e: u32,
        ) -> f64;
        pub(crate) fn imparo_cuda_dispatch_proof_reset();
        pub(crate) fn imparo_cuda_dispatch_expectation(knob_mask: u64) -> i32;
        pub(crate) fn imparo_cuda_dispatch_proof(
            out: *mut super::DispatchProofWire,
            out_bytes: u32,
        ) -> i32;
        pub(crate) fn imparo_cuda_program_pack_install(
            pack: *const super::ProgramPackWire,
            struct_size: u32,
        ) -> i32;
        #[allow(dead_code)]
        pub(crate) fn imparo_cuda_program_catalog_identity(
            out: *mut super::ProgramCatalogIdentityWire,
            struct_size: u32,
        ) -> i32;
        pub(crate) fn imparo_cuda_program_bind(
            group_id: *const u8,
            variant_id: *const u8,
        ) -> i32;
        pub(crate) fn imparo_cuda_program_freeze() -> i32;
        #[allow(dead_code)]
        pub(crate) fn imparo_cuda_program_launch(
            launch: *const super::ProgramLaunchWire,
            struct_size: u32,
        ) -> i32;
        #[allow(dead_code)]
        pub(crate) fn imparo_cuda_program_reset() -> i32;
        #[cfg(all(test, imparo_cuda_program_smoke))]
        pub(crate) fn imparo_cuda_program_test_alloc(
            bytes: u64,
            pointer_out: *mut u64,
        ) -> i32;
        #[cfg(all(test, imparo_cuda_program_smoke))]
        pub(crate) fn imparo_cuda_program_test_free(pointer: u64) -> i32;
        #[cfg(all(test, imparo_cuda_program_smoke))]
        pub(crate) fn imparo_cuda_program_test_write_u32(
            pointer: u64,
            value: u32,
        ) -> i32;
        #[cfg(all(test, imparo_cuda_program_smoke))]
        pub(crate) fn imparo_cuda_program_test_read_u32(
            pointer: u64,
            value_out: *mut u32,
        ) -> i32;
        #[cfg(all(test, imparo_cuda_program_smoke))]
        pub(crate) fn imparo_cuda_program_test_synchronize() -> i32;
        #[cfg(all(test, imparo_cuda_program_smoke))]
        pub(crate) fn imparo_cuda_program_test_graph_begin() -> i32;
        #[cfg(all(test, imparo_cuda_program_smoke))]
        pub(crate) fn imparo_cuda_program_test_graph_end_replay(
            replay_count: u32,
        ) -> i32;
        #[cfg(all(test, imparo_cuda_program_smoke))]
        pub(crate) fn imparo_cuda_program_test_graph_replay(
            decode_start_pos: u32,
        ) -> i32;
        #[cfg(all(test, imparo_cuda_program_smoke))]
        pub(crate) fn imparo_cuda_program_test_graph_alive() -> u32;
        pub(crate) fn imparo_cuda_alloc(id: u32, bytes: u64) -> i32;
        pub(crate) fn imparo_cuda_arena(bytes: u64) -> i32;
        pub(crate) fn imparo_cuda_place(id: u32, offset: u64, bytes: u64) -> i32;
        pub(crate) fn imparo_cuda_page_round(n: u64) -> u64;
        pub(crate) fn imparo_cuda_alloc_kv(n_layers: u32, bytes: *const u64) -> i32;
        pub(crate) fn imparo_cuda_grow_kv(n_layers: u32, bytes: *const u64) -> i32;
        pub(crate) fn imparo_cuda_alloc_kv_layout(
            n_layers: u32,
            bytes: *const u64,
            layouts: *const imparo_backend::KvLayout,
            layout_count: u32,
        ) -> i32;
        pub(crate) fn imparo_cuda_grow_kv_layout(
            n_layers: u32,
            bytes: *const u64,
            layouts: *const imparo_backend::KvLayout,
            layout_count: u32,
        ) -> i32;
        pub(crate) fn imparo_cuda_set_kv_pages(
            layer: u32,
            entries: *const u32,
            n: u32,
        ) -> i32;
        pub(crate) fn imparo_cuda_write(id: u32, off: u64, src: *const f32, n: u64);
        pub(crate) fn imparo_cuda_write_u32(id: u32, off: u64, src: *const u32, n: u64);
        pub(crate) fn imparo_cuda_read(id: u32, off: u64, dst: *mut f32, n: u64);
        pub(crate) fn imparo_cuda_read_kv(
            layer: u32,
            is_v: u32,
            off: u64,
            dst: *mut u8,
            n: u64,
        ) -> i32;
        pub(crate) fn imparo_cuda_write_kv(
            layer: u32,
            is_v: u32,
            off: u64,
            src: *const u8,
            n: u64,
        ) -> i32;
        pub(crate) fn imparo_cuda_host_alloc(bytes: u64, handle_out: *mut u64) -> i32;
        pub(crate) fn imparo_cuda_host_free(handle: u64) -> i32;
        pub(crate) fn imparo_cuda_host_read(
            handle: u64,
            off: u64,
            dst: *mut u8,
            n: u64,
        ) -> i32;
        pub(crate) fn imparo_cuda_host_write(
            handle: u64,
            off: u64,
            src: *const u8,
            n: u64,
        ) -> i32;
        pub(crate) fn imparo_cuda_host_allocated_bytes(out: *mut u64) -> i32;
        pub(crate) fn imparo_cuda_host_profile(
            out: *mut super::HostProfileWire,
            out_bytes: u32,
        ) -> i32;
        pub(crate) fn imparo_cuda_kv_demote(
            spans: *const imparo_backend::KvTransferSpan,
            count: u32,
        ) -> i32;
        pub(crate) fn imparo_cuda_kv_promote(
            spans: *const imparo_backend::KvTransferSpan,
            count: u32,
        ) -> i32;
        pub(crate) fn imparo_cuda_kv_advise_free(
            layer: u32,
            is_v: u32,
            off: u64,
            len: u64,
        ) -> i32;
        pub(crate) fn imparo_cuda_kv_advise_reuse(
            layer: u32,
            is_v: u32,
            off: u64,
            len: u64,
        ) -> i32;
        pub(crate) fn imparo_cuda_kv_live_bytes(out: *mut u64) -> i32;
        pub(crate) fn imparo_cuda_set_epilogue(on: u32);
        pub(crate) fn imparo_cuda_set_knob(idx: u32, value: u32);
        pub(crate) fn imparo_cuda_knob(idx: u32) -> u32;
        pub(crate) fn imparo_cuda_buf_count() -> u32;
        pub(crate) fn imparo_cuda_matmat(
            wkind: u32,
            w_off: u64,
            n_in: u32,
            n_out: u32,
            src: u32,
            dst: u32,
            n_tok: u32,
            src_row: u32,
        );
        pub(crate) fn imparo_cuda_matmat_gated(
            gate_kind: u32,
            gate_off: u64,
            up_kind: u32,
            up_off: u64,
            n_in: u32,
            n_out: u32,
            src: u32,
            dst: u32,
            tmp: u32,
            n_tok: u32,
            fused_epilogue: u32,
        );
        pub(crate) fn imparo_cuda_ffn_gated_down(
            gate_kind: u32,
            gate_off: u64,
            up_kind: u32,
            up_off: u64,
            down_kind: u32,
            down_off: u64,
            n_in: u32,
            n_mid: u32,
            n_out: u32,
            src: u32,
            gated_tmp: u32,
            dst: u32,
            n_tok: u32,
        ) -> u32;
        pub(crate) fn imparo_cuda_exact128_route_hits_lab() -> u32;
        pub(crate) fn imparo_cuda_enable_tuner_lab();
        pub(crate) fn imparo_cuda_matmat_pair(
            first_kind: u32,
            first_off: u64,
            first_dst: u32,
            second_kind: u32,
            second_off: u64,
            second_dst: u32,
            n_in: u32,
            n_out: u32,
            src: u32,
            n_tok: u32,
        );
        pub(crate) fn imparo_cuda_ple_project(
            gate_kind: u32,
            gate_off: u64,
            proj_kind: u32,
            proj_off: u64,
            n_embd: u32,
            ple_width: u32,
            src: u32,
            gate: u32,
            per_layer: u32,
            per_layer_off: u32,
            per_layer_stride: u32,
            back: u32,
            n_tok: u32,
        );
        pub(crate) fn imparo_cuda_rms_norm(
            buf: u32,
            src: u32,
            w_off: u64,
            width: u32,
            eps: f32,
            n_row: u32,
            row_stride: u32,
            base_off: u32,
            has_w: u32,
        );
        pub(crate) fn imparo_cuda_rms_norm_project(
            buf: u32,
            src: u32,
            w_off: u64,
            width: u32,
            eps: f32,
            n_row: u32,
            row_stride: u32,
            base_off: u32,
        );
        pub(crate) fn imparo_cuda_rms_norm_add(
            dst: u32,
            src: u32,
            w_off: u64,
            width: u32,
            eps: f32,
            n_row: u32,
            add_buf: u32,
            output_scale: f32,
        );
        pub(crate) fn imparo_cuda_rms_norm_add_project(
            buf: u32,
            w_off: u64,
            width: u32,
            eps: f32,
            n_row: u32,
            add_buf: u32,
        );
        pub(crate) fn imparo_cuda_add_rms_norm_project(
            dst: u32,
            resid: u32,
            other: u32,
            w_off: u64,
            width: u32,
            eps: f32,
            n_row: u32,
        ) -> u32;
        pub(crate) fn imparo_cuda_rms_norm_add_dual_project(
            src: u32,
            residual: u32,
            first_w_off: u64,
            mid: u32,
            second_w_off: u64,
            out: u32,
            width: u32,
            eps: f32,
            n_row: u32,
        ) -> u32;
        pub(crate) fn imparo_cuda_rope(
            buf: u32,
            n_rot: u32,
            base: f32,
            head_dim: u32,
            n_heads: u32,
            start_pos: u32,
            n_tok: u32,
            freqs: *const f32,
        );
        pub(crate) fn imparo_cuda_hadamard(buf: u32, n: u32, nrot: u32);
        pub(crate) fn imparo_cuda_head_norm_rope_hadamard(
            buf: u32,
            w_off: u64,
            head_dim: u32,
            eps: f32,
            n_heads: u32,
            start_pos: u32,
            n_tok: u32,
            rope_dim: u32,
            rope_base: f32,
            freqs: *const f32,
            hadamard_nrot: u32,
        );
        pub(crate) fn imparo_cuda_kv_head_postprocess(
            k_buf: u32,
            v_buf: u32,
            k_norm_off: u64,
            head_dim: u32,
            eps: f32,
            n_kv: u32,
            start_pos: u32,
            n_tok: u32,
            rope_dim: u32,
            rope_base: f32,
            freqs: *const f32,
            k_hadamard_nrot: u32,
            v_hadamard_nrot: u32,
        );
        pub(crate) fn imparo_cuda_kv_store(
            src: u32,
            layer: u32,
            width: u32,
            start_pos: u32,
            n_tok: u32,
            is_v: u32,
            ring: u32,
        );
        pub(crate) fn imparo_cuda_kv_dequant(
            layer: u32,
            width: u32,
            slots: u32,
            is_v: u32,
            scratch: u32,
            ring: u32,
        );
        pub(crate) fn imparo_cuda_attention(
            kv_layer: u32,
            head_dim: u32,
            n_heads: u32,
            n_kv: u32,
            kv_width: u32,
            start_pos: u32,
            qk_scale: f32,
            window: u32,
            n_tok: u32,
            ring: u32,
            q: u32,
            out: u32,
            kdq: u32,
            vdq: u32,
        );
        pub(crate) fn imparo_cuda_shortconv(
            bcx: u32,
            w_off: u64,
            state: u32,
            state_off: u32,
            out: u32,
            width: u32,
            kernel: u32,
            n_tok: u32,
        );
        pub(crate) fn imparo_cuda_shortconv_snapshot(
            bcx: u32,
            state: u32,
            state_off: u32,
            snap: u32,
            snap_off: u32,
            width: u32,
            kernel: u32,
            n_tok: u32,
        );
        pub(crate) fn imparo_cuda_silu(a: u32, n: u32);
        pub(crate) fn imparo_cuda_silu_mul(a: u32, b: u32, n: u32);
        pub(crate) fn imparo_cuda_gelu(a: u32, n: u32);
        pub(crate) fn imparo_cuda_gelu_mul(a: u32, b: u32, n: u32);
        pub(crate) fn imparo_cuda_add(a: u32, b: u32, n: u32);
        pub(crate) fn imparo_cuda_add_scale(a: u32, b: u32, k: f32, n: u32);
        pub(crate) fn imparo_cuda_scale(a: u32, k: f32, n: u32);
        pub(crate) fn imparo_cuda_copy(dst: u32, src: u32, n: u32);
        pub(crate) fn imparo_cuda_copy_range(
            dst: u32,
            dst_off: u32,
            src: u32,
            src_off: u32,
            n: u32,
        );
        pub(crate) fn imparo_cuda_mul_strided(
            a: u32,
            b: u32,
            n: u32,
            b_off: u32,
            b_stride: u32,
            a_stride: u32,
            n_tok: u32,
        );
        pub(crate) fn imparo_cuda_softcap(a: u32, cap: f32, n: u32);
        pub(crate) fn imparo_cuda_argmax(src: u32, dst: u32, n: u32);
        pub(crate) fn imparo_cuda_row(
            wkind: u32,
            w_off: u64,
            width: u32,
            index: u32,
            scale: f32,
            dst: u32,
            dst_off: u32,
        );
        pub(crate) fn imparo_cuda_rows(
            wkind: u32,
            w_off: u64,
            width: u32,
            table_rows: u32,
            tokens_buf: u32,
            scale: f32,
            dst: u32,
            n_tok: u32,
        );
        pub(crate) fn imparo_cuda_ple_gather_combine(
            proj: u32,
            tokens_buf: u32,
            w_offset: u64,
            width: u32,
            emb_scale: f32,
            comb_scale: f32,
            n_tok: u32,
        );
    }

    pub(crate) fn backend_artifact_sha256() -> Result<Option<[u8; 32]>, String> {
        Ok(None)
    }
}

#[cfg(feature = "cuda-static")]
pub(crate) use imp::*;

#[cfg(feature = "cuda-dynamic")]
mod imp_dynamic {
    use super::c_void;
    use crate::dylib::Library;
    use std::sync::OnceLock;

    const LOAD_ERROR: i32 = -70;

    macro_rules! fields {
        ($m:ident) => { $m! {
            imparo_cuda_init(weights: *const c_void, len: u64, streamed: *const c_void, streamed_count: u32) -> i32 = LOAD_ERROR;
            imparo_cuda_prepare_quantized_weight_cache(specs: *const imparo_backend::QuantizedWeightPrepack, count: u32) -> i32 = 0;
            imparo_cuda_memory_info(free: *mut u64, total: *mut u64, allocated: *mut u64) -> i32 = LOAD_ERROR;
            imparo_cuda_weights_resident() -> u32 = 0;
            imparo_cuda_runtime_identity(out: *mut super::RuntimeIdentityWire, out_bytes: u32) -> i32 = LOAD_ERROR;
            imparo_cuda_set_kv_types(k: u32, v: u32) -> () = ();
            imparo_cuda_device_tag(dst: *mut u8, len: u32) -> u32 = 0;
            imparo_cuda_begin() -> () = ();
            imparo_cuda_begin_forward(decode: u32) -> () = ();
            imparo_cuda_set_batch_geometry(absolute_start: u64, active_tokens: u32, phase: u32) -> i32 = LOAD_ERROR;
            imparo_cuda_decode_prepare(token: u32, start_pos: u32, argmax: u32) -> i32 = 0;
            imparo_cuda_prefill_prepare(tokens: *const u32, count: u32, start_pos: u32, argmax: u32) -> i32 = 0;
            imparo_cuda_flush() -> () = ();
            imparo_cuda_end() -> i32 = LOAD_ERROR;
            imparo_cuda_program_pack_install(pack: *const super::ProgramPackWire, struct_size: u32) -> i32 = LOAD_ERROR;
            imparo_cuda_program_catalog_identity(out: *mut super::ProgramCatalogIdentityWire, struct_size: u32) -> i32 = LOAD_ERROR;
            imparo_cuda_program_bind(group_id: *const u8, variant_id: *const u8) -> i32 = LOAD_ERROR;
            imparo_cuda_program_freeze() -> i32 = LOAD_ERROR;
            imparo_cuda_program_launch(launch: *const super::ProgramLaunchWire, struct_size: u32) -> i32 = LOAD_ERROR;
            imparo_cuda_program_reset() -> i32 = LOAD_ERROR;
            imparo_cuda_alloc(id: u32, bytes: u64) -> i32 = LOAD_ERROR;
            imparo_cuda_arena(bytes: u64) -> i32 = LOAD_ERROR;
            imparo_cuda_place(id: u32, offset: u64, bytes: u64) -> i32 = LOAD_ERROR;
            imparo_cuda_page_round(n: u64) -> u64 = 0;
            imparo_cuda_alloc_kv(n_layers: u32, bytes: *const u64) -> i32 = LOAD_ERROR;
            imparo_cuda_grow_kv(n_layers: u32, bytes: *const u64) -> i32 = LOAD_ERROR;
            imparo_cuda_alloc_kv_layout(n_layers: u32, bytes: *const u64, layouts: *const imparo_backend::KvLayout, layout_count: u32) -> i32 = LOAD_ERROR;
            imparo_cuda_grow_kv_layout(n_layers: u32, bytes: *const u64, layouts: *const imparo_backend::KvLayout, layout_count: u32) -> i32 = LOAD_ERROR;
            imparo_cuda_set_kv_pages(layer: u32, entries: *const u32, n: u32) -> i32 = LOAD_ERROR;
            imparo_cuda_write(id: u32, off: u64, src: *const f32, n: u64) -> () = ();
            imparo_cuda_write_u32(id: u32, off: u64, src: *const u32, n: u64) -> () = ();
            imparo_cuda_read(id: u32, off: u64, dst: *mut f32, n: u64) -> () = ();
            imparo_cuda_read_kv(layer: u32, is_v: u32, off: u64, dst: *mut u8, n: u64) -> i32 = LOAD_ERROR;
            imparo_cuda_write_kv(layer: u32, is_v: u32, off: u64, src: *const u8, n: u64) -> i32 = LOAD_ERROR;
            imparo_cuda_host_alloc(bytes: u64, handle_out: *mut u64) -> i32 = LOAD_ERROR;
            imparo_cuda_host_free(handle: u64) -> i32 = LOAD_ERROR;
            imparo_cuda_host_read(handle: u64, off: u64, dst: *mut u8, n: u64) -> i32 = LOAD_ERROR;
            imparo_cuda_host_write(handle: u64, off: u64, src: *const u8, n: u64) -> i32 = LOAD_ERROR;
            imparo_cuda_host_allocated_bytes(out: *mut u64) -> i32 = LOAD_ERROR;
            imparo_cuda_host_profile(out: *mut super::HostProfileWire, out_bytes: u32) -> i32 = LOAD_ERROR;
            imparo_cuda_kv_demote(spans: *const imparo_backend::KvTransferSpan, count: u32) -> i32 = LOAD_ERROR;
            imparo_cuda_kv_promote(spans: *const imparo_backend::KvTransferSpan, count: u32) -> i32 = LOAD_ERROR;
            imparo_cuda_kv_advise_free(layer: u32, is_v: u32, off: u64, len: u64) -> i32 = LOAD_ERROR;
            imparo_cuda_kv_advise_reuse(layer: u32, is_v: u32, off: u64, len: u64) -> i32 = LOAD_ERROR;
            imparo_cuda_kv_live_bytes(out: *mut u64) -> i32 = LOAD_ERROR;
            imparo_cuda_set_epilogue(on: u32) -> () = ();
            imparo_cuda_set_knob(idx: u32, value: u32) -> () = ();
            imparo_cuda_knob(idx: u32) -> u32 = 0;
            imparo_cuda_buf_count() -> u32 = 0;
            imparo_cuda_matmat(wkind: u32, w_off: u64, n_in: u32, n_out: u32, src: u32, dst: u32, n_tok: u32, src_row: u32) -> () = ();
            imparo_cuda_matmat_gated(gate_kind: u32, gate_off: u64, up_kind: u32, up_off: u64, n_in: u32, n_out: u32, src: u32, dst: u32, tmp: u32, n_tok: u32, fused_epilogue: u32) -> () = ();
            imparo_cuda_ffn_gated_down(gate_kind: u32, gate_off: u64, up_kind: u32, up_off: u64, down_kind: u32, down_off: u64, n_in: u32, n_mid: u32, n_out: u32, src: u32, gated_tmp: u32, dst: u32, n_tok: u32) -> u32 = 0;
            imparo_cuda_matmat_pair(first_kind: u32, first_off: u64, first_dst: u32, second_kind: u32, second_off: u64, second_dst: u32, n_in: u32, n_out: u32, src: u32, n_tok: u32) -> () = ();
            imparo_cuda_ple_project(gate_kind: u32, gate_off: u64, proj_kind: u32, proj_off: u64, n_embd: u32, ple_width: u32, src: u32, gate: u32, per_layer: u32, per_layer_off: u32, per_layer_stride: u32, back: u32, n_tok: u32) -> () = ();
            imparo_cuda_rms_norm(buf: u32, src: u32, w_off: u64, width: u32, eps: f32, n_row: u32, row_stride: u32, base_off: u32, has_w: u32) -> () = ();
            imparo_cuda_rms_norm_project(buf: u32, src: u32, w_off: u64, width: u32, eps: f32, n_row: u32, row_stride: u32, base_off: u32) -> () = ();
            imparo_cuda_rms_norm_add(dst: u32, src: u32, w_off: u64, width: u32, eps: f32, n_row: u32, add_buf: u32, output_scale: f32) -> () = ();
            imparo_cuda_rms_norm_add_project(buf: u32, w_off: u64, width: u32, eps: f32, n_row: u32, add_buf: u32) -> () = ();
            imparo_cuda_add_rms_norm_project(dst: u32, resid: u32, other: u32, w_off: u64, width: u32, eps: f32, n_row: u32) -> u32 = 0;
            imparo_cuda_rms_norm_add_dual_project(src: u32, residual: u32, first_w_off: u64, mid: u32, second_w_off: u64, out: u32, width: u32, eps: f32, n_row: u32) -> u32 = 0;
            imparo_cuda_rope(buf: u32, n_rot: u32, base: f32, head_dim: u32, n_heads: u32, start_pos: u32, n_tok: u32, freqs: *const f32) -> () = ();
            imparo_cuda_hadamard(buf: u32, n: u32, nrot: u32) -> () = ();
            imparo_cuda_head_norm_rope_hadamard(buf: u32, w_off: u64, head_dim: u32, eps: f32, n_heads: u32, start_pos: u32, n_tok: u32, rope_dim: u32, rope_base: f32, freqs: *const f32, hadamard_nrot: u32) -> () = ();
            imparo_cuda_kv_head_postprocess(k_buf: u32, v_buf: u32, k_norm_off: u64, head_dim: u32, eps: f32, n_kv: u32, start_pos: u32, n_tok: u32, rope_dim: u32, rope_base: f32, freqs: *const f32, k_hadamard_nrot: u32, v_hadamard_nrot: u32) -> () = ();
            imparo_cuda_kv_store(src: u32, layer: u32, width: u32, start_pos: u32, n_tok: u32, is_v: u32, ring: u32) -> () = ();
            imparo_cuda_kv_dequant(layer: u32, width: u32, slots: u32, is_v: u32, scratch: u32, ring: u32) -> () = ();
            imparo_cuda_attention(kv_layer: u32, head_dim: u32, n_heads: u32, n_kv: u32, kv_width: u32, start_pos: u32, qk_scale: f32, window: u32, n_tok: u32, ring: u32, q: u32, out: u32, kdq: u32, vdq: u32) -> () = ();
            imparo_cuda_shortconv(bcx: u32, w_off: u64, state: u32, state_off: u32, out: u32, width: u32, kernel: u32, n_tok: u32) -> () = ();
            imparo_cuda_shortconv_snapshot(bcx: u32, state: u32, state_off: u32, snap: u32, snap_off: u32, width: u32, kernel: u32, n_tok: u32) -> () = ();
            imparo_cuda_silu(a: u32, n: u32) -> () = ();
            imparo_cuda_silu_mul(a: u32, b: u32, n: u32) -> () = ();
            imparo_cuda_gelu(a: u32, n: u32) -> () = ();
            imparo_cuda_gelu_mul(a: u32, b: u32, n: u32) -> () = ();
            imparo_cuda_add(a: u32, b: u32, n: u32) -> () = ();
            imparo_cuda_add_scale(a: u32, b: u32, k: f32, n: u32) -> () = ();
            imparo_cuda_scale(a: u32, k: f32, n: u32) -> () = ();
            imparo_cuda_copy(dst: u32, src: u32, n: u32) -> () = ();
            imparo_cuda_copy_range(dst: u32, dst_off: u32, src: u32, src_off: u32, n: u32) -> () = ();
            imparo_cuda_mul_strided(a: u32, b: u32, n: u32, b_off: u32, b_stride: u32, a_stride: u32, n_tok: u32) -> () = ();
            imparo_cuda_softcap(a: u32, cap: f32, n: u32) -> () = ();
            imparo_cuda_argmax(src: u32, dst: u32, n: u32) -> () = ();
            imparo_cuda_row(wkind: u32, w_off: u64, width: u32, index: u32, scale: f32, dst: u32, dst_off: u32) -> () = ();
            imparo_cuda_rows(wkind: u32, w_off: u64, width: u32, table_rows: u32, tokens_buf: u32, scale: f32, dst: u32, n_tok: u32) -> () = ();
            imparo_cuda_ple_gather_combine(proj: u32, tokens_buf: u32, w_offset: u64, width: u32, emb_scale: f32, comb_scale: f32, n_tok: u32) -> () = ();
        } };
    }

    macro_rules! declare_api {
        ($( $name:ident($($arg:ident: $ty:ty),*) -> $ret:ty = $fallback:expr; )*) => {
            #[allow(dead_code)] // ABI26 symbols are mandatory before every caller lands.
            struct Api {
                _library: Library,
                artifact_sha256: [u8; 32],
                $( $name: unsafe extern "C" fn($($ty),*) -> $ret, )*
            }
            impl Api {
                fn load() -> Result<Self, String> {
                    let path = crate::loader::resolve_backend()?;
                    let library = Library::open(&path)?;
                    type Abi = unsafe extern "C" fn() -> u32;
                    let abi: Abi = unsafe {
                        std::mem::transmute::<*mut c_void, Abi>(
                            library.symbol(b"imparo_cuda_abi_version\0")?,
                        )
                    };
                    let actual = unsafe { abi() };
                    if actual != crate::CUDA_BACKEND_ABI {
                        return Err(format!("CUDA backend ABI {actual}, runtime expects {}: {}", crate::CUDA_BACKEND_ABI, path.display()));
                    }
                    let artifact_sha256 = crate::loader::backend_artifact_sha256(&path)?;
                    Ok(Self {
                        artifact_sha256,
                        $( $name: unsafe {
                            std::mem::transmute::<
                                *mut c_void,
                                unsafe extern "C" fn($($ty),*) -> $ret,
                            >(library.symbol(concat!(stringify!($name), "\0").as_bytes())?)
                        }, )*
                        _library: library,
                    })
                }
            }
            static API: OnceLock<Result<Api, String>> = OnceLock::new();
            fn api() -> Result<&'static Api, &'static str> {
                match API.get_or_init(Api::load) {
                    Ok(api) => Ok(api),
                    Err(error) => { eprintln!("[imparo] CUDA backend load failed: {error}"); Err(error) }
                }
            }
            $(
                #[allow(dead_code)]
                pub(crate) unsafe fn $name($($arg: $ty),*) -> $ret {
                    let Ok(api) = api() else { return $fallback; };
                    unsafe { (api.$name)($($arg),*) }
                }
            )*
        };
    }
    fields!(declare_api);

    struct Step1Api {
        last_gpu_us: Option<unsafe extern "C" fn() -> f64>,
        set_tuner_mode: Option<unsafe extern "C" fn(u32) -> i32>,
        device_profile:
            Option<unsafe extern "C" fn(*mut super::DeviceProfileWire, u32) -> i32>,
        probe: Option<unsafe extern "C" fn(u32, u64, u32, u32, u32, u32) -> f64>,
        dispatch_proof_reset: Option<unsafe extern "C" fn()>,
        dispatch_expectation: Option<unsafe extern "C" fn(u64) -> i32>,
        dispatch_proof:
            Option<unsafe extern "C" fn(*mut super::DispatchProofWire, u32) -> i32>,
    }

    fn step1_api() -> &'static Step1Api {
        static STEP1: OnceLock<Step1Api> = OnceLock::new();
        STEP1.get_or_init(|| {
            let Ok(api) = api() else {
                return Step1Api {
                    last_gpu_us: None,
                    set_tuner_mode: None,
                    device_profile: None,
                    probe: None,
                    dispatch_proof_reset: None,
                    dispatch_expectation: None,
                    dispatch_proof: None,
                };
            };
            let last_gpu_us = api
                ._library
                .symbol(b"imparo_cuda_last_gpu_us\0")
                .ok()
                .map(|p| unsafe { std::mem::transmute(p) });
            let set_tuner_mode = api
                ._library
                .symbol(b"imparo_cuda_set_tuner_mode\0")
                .ok()
                .map(|p| unsafe { std::mem::transmute(p) });
            let device_profile = api
                ._library
                .symbol(b"imparo_cuda_device_profile\0")
                .ok()
                .map(|p| unsafe { std::mem::transmute(p) });
            let probe = api
                ._library
                .symbol(b"imparo_cuda_probe\0")
                .ok()
                .map(|p| unsafe { std::mem::transmute(p) });
            let dispatch_proof_reset = api
                ._library
                .symbol(b"imparo_cuda_dispatch_proof_reset\0")
                .ok()
                .map(|p| unsafe { std::mem::transmute(p) });
            let dispatch_expectation = api
                ._library
                .symbol(b"imparo_cuda_dispatch_expectation\0")
                .ok()
                .map(|p| unsafe { std::mem::transmute(p) });
            let dispatch_proof = api
                ._library
                .symbol(b"imparo_cuda_dispatch_proof\0")
                .ok()
                .map(|p| unsafe { std::mem::transmute(p) });
            let (
                last_gpu_us,
                set_tuner_mode,
                device_profile,
                probe,
                dispatch_proof_reset,
                dispatch_expectation,
                dispatch_proof,
            ) = match (
                last_gpu_us,
                set_tuner_mode,
                device_profile,
                probe,
                dispatch_proof_reset,
                dispatch_expectation,
                dispatch_proof,
            ) {
                (
                    Some(last),
                    Some(mode),
                    Some(profile),
                    Some(probe),
                    Some(reset),
                    Some(expect),
                    Some(proof),
                ) => (
                    Some(last),
                    Some(mode),
                    Some(profile),
                    Some(probe),
                    Some(reset),
                    Some(expect),
                    Some(proof),
                ),
                _ => (None, None, None, None, None, None, None),
            };
            Step1Api {
                last_gpu_us,
                set_tuner_mode,
                device_profile,
                probe,
                dispatch_proof_reset,
                dispatch_expectation,
                dispatch_proof,
            }
        })
    }

    pub(crate) unsafe fn imparo_cuda_last_gpu_us() -> f64 {
        step1_api().last_gpu_us.map_or(0.0, |f| unsafe { f() })
    }

    pub(crate) unsafe fn imparo_cuda_set_tuner_mode(enabled: u32) -> i32 {
        step1_api()
            .set_tuner_mode
            .map_or(LOAD_ERROR, |f| unsafe { f(enabled) })
    }

    pub(crate) unsafe fn imparo_cuda_device_profile(
        out: *mut super::DeviceProfileWire,
        out_bytes: u32,
    ) -> i32 {
        step1_api()
            .device_profile
            .map_or(-70, |f| unsafe { f(out, out_bytes) })
    }

    pub(crate) unsafe fn imparo_cuda_probe(
        kind: u32,
        a: u64,
        b: u32,
        c: u32,
        d: u32,
        e: u32,
    ) -> f64 {
        step1_api()
            .probe
            .map_or(0.0, |f| unsafe { f(kind, a, b, c, d, e) })
    }

    pub(crate) unsafe fn imparo_cuda_dispatch_proof_reset() {
        if let Some(f) = step1_api().dispatch_proof_reset {
            unsafe { f() };
        }
    }

    pub(crate) unsafe fn imparo_cuda_dispatch_expectation(knob_mask: u64) -> i32 {
        step1_api()
            .dispatch_expectation
            .map_or(LOAD_ERROR, |f| unsafe { f(knob_mask) })
    }

    pub(crate) unsafe fn imparo_cuda_dispatch_proof(
        out: *mut super::DispatchProofWire,
        out_bytes: u32,
    ) -> i32 {
        step1_api()
            .dispatch_proof
            .map_or(LOAD_ERROR, |f| unsafe { f(out, out_bytes) })
    }

    #[cfg(not(feature = "cuda-static"))]
    pub(crate) unsafe fn imparo_cuda_exact128_route_hits_lab() -> u32 {
        0
    }

    #[cfg(not(feature = "cuda-static"))]
    pub(crate) unsafe fn imparo_cuda_enable_tuner_lab() {}

    pub(crate) fn backend_artifact_sha256() -> Result<Option<[u8; 32]>, String> {
        api()
            .map(|api| Some(api.artifact_sha256))
            .map_err(str::to_owned)
    }

    #[cfg(test)]
    mod tests {
        #[test]
        fn resolved_plugin_loads_and_matches_abi() {
            if std::env::var_os("IMPARO_CUDA_BACKEND").is_some()
                || std::env::var_os("IMPARO_CUDA_SM").is_some()
            {
                assert!(super::api().is_ok());
            }
        }
    }
}

#[cfg(feature = "cuda-dynamic")]
pub(crate) use imp_dynamic::*;

#[cfg(all(test, feature = "cuda-static"))]
mod native_kv_smoke_tests {
    use super::*;

    #[test]
    fn transactional_kv_arena_roundtrip_and_ownership() {
        if std::env::var_os("IMPARO_CUDA_KV_SMOKE").is_none() {
            return;
        }
        let weights = [0_u8];
        assert_eq!(
            unsafe {
                imparo_cuda_init(
                    weights.as_ptr().cast(),
                    weights.len() as u64,
                    core::ptr::null(),
                    0,
                )
            },
            0
        );
        let initial = [0_u64, 64 * 3, 64 * 5];
        let initial_layout = [
            imparo_backend::KvLayout::default(),
            imparo_backend::KvLayout {
                layer: 1,
                reserved: 0,
                logical_slots: initial[1],
                k_stride: 1,
                v_stride: 1,
            },
            imparo_backend::KvLayout {
                layer: 2,
                reserved: 0,
                logical_slots: initial[2],
                k_stride: 1,
                v_stride: 1,
            },
        ];
        assert_eq!(
            unsafe {
                imparo_cuda_alloc_kv_layout(
                    3,
                    initial.as_ptr(),
                    initial_layout.as_ptr(),
                    initial_layout.len() as u32,
                )
            },
            0
        );
        let short = [2_u32, 1];
        assert_eq!(unsafe { imparo_cuda_set_kv_pages(1, short.as_ptr(), 2) }, 0);
        let bad_entry = [3_u32];
        assert_ne!(
            unsafe { imparo_cuda_set_kv_pages(1, bad_entry.as_ptr(), 1) },
            0
        );
        let too_long = [0_u32, 1, 2, 0];
        assert_ne!(
            unsafe { imparo_cuda_set_kv_pages(1, too_long.as_ptr(), 4) },
            0
        );
        assert_eq!(
            unsafe { imparo_cuda_set_kv_pages(1, core::ptr::null(), 0) },
            0
        );
        let (mut free, mut total, mut allocated) = (0, 0, 0);
        assert_eq!(
            unsafe {
                imparo_cuda_memory_info(
                    &raw mut free,
                    &raw mut total,
                    &raw mut allocated,
                )
            },
            0
        );
        assert!(total > free);
        // Four aligned non-empty slices: K/V for layers 1 and 2. Layer 0 remains
        // null and contributes neither padding nor a phantom allocation.
        assert!(allocated >= 4 * 4096);
        let mut zero_layer = 0_u8;
        assert_ne!(
            unsafe { imparo_cuda_read_kv(0, 0, 0, &raw mut zero_layer, 1) },
            0
        );

        let source: Vec<u8> = (0..initial[1]).map(|n| n as u8).collect();
        let mut output = vec![0_u8; source.len()];
        // Five queued chunks exceed the three-slot staging ring. The subsequent
        // read must observe every chunk after wrap-around, not an overwritten slot.
        for chunk in 0..6_u64 {
            let off = chunk * 32;
            assert_eq!(
                unsafe {
                    imparo_cuda_write_kv(1, 0, off, source[off as usize..].as_ptr(), 32)
                },
                0
            );
        }
        assert_eq!(
            unsafe {
                imparo_cuda_read_kv(1, 0, 0, output.as_mut_ptr(), output.len() as u64)
            },
            0
        );
        assert_eq!(source, output);
        assert_ne!(
            unsafe {
                imparo_cuda_read_kv(1, 0, initial[1] - 1, output.as_mut_ptr(), 2)
            },
            0
        );

        let v_source: Vec<u8> = (0..initial[2]).map(|n| 255 - n as u8).collect();
        let mut v_output = vec![0_u8; v_source.len()];
        assert_eq!(
            unsafe {
                imparo_cuda_write_kv(2, 1, 0, v_source.as_ptr(), v_source.len() as u64)
            },
            0
        );
        assert_eq!(
            unsafe {
                imparo_cuda_read_kv(
                    2,
                    1,
                    0,
                    v_output.as_mut_ptr(),
                    v_output.len() as u64,
                )
            },
            0
        );
        assert_eq!(v_source, v_output);

        assert_eq!(
            unsafe { imparo_cuda_kv_advise_reuse(1, 0, 0, initial[1]) },
            0
        );
        assert_ne!(
            unsafe { imparo_cuda_kv_advise_reuse(1, 0, 0, initial[1]) },
            0
        );
        assert_eq!(
            unsafe { imparo_cuda_kv_advise_reuse(2, 1, 0, initial[2]) },
            0
        );
        let mut live = 0;
        assert_eq!(unsafe { imparo_cuda_kv_live_bytes(&raw mut live) }, 0);
        assert_eq!(live, initial[1] + initial[2]);

        assert_eq!(
            unsafe { imparo_cuda_kv_advise_free(1, 0, 0, initial[1]) },
            0
        );
        assert_ne!(
            unsafe { imparo_cuda_kv_advise_free(1, 0, 0, initial[1]) },
            0
        );
        assert_ne!(
            unsafe {
                imparo_cuda_read_kv(1, 0, 0, output.as_mut_ptr(), output.len() as u64)
            },
            0
        );
        let mut allocated_after_free = 0;
        assert_eq!(
            unsafe {
                imparo_cuda_memory_info(
                    core::ptr::null_mut(),
                    core::ptr::null_mut(),
                    &raw mut allocated_after_free,
                )
            },
            0
        );
        assert_eq!(allocated_after_free, allocated);
        assert_eq!(
            unsafe { imparo_cuda_kv_advise_reuse(1, 0, 0, initial[1]) },
            0
        );

        // Shrink is rejected transactionally; the old allocation and bytes remain.
        let shrink = [0_u64, 64 * 2, 64 * 5];
        let shrink_layout = [
            imparo_backend::KvLayout::default(),
            imparo_backend::KvLayout {
                logical_slots: shrink[1],
                ..initial_layout[1]
            },
            initial_layout[2],
        ];
        assert_ne!(
            unsafe {
                imparo_cuda_grow_kv_layout(
                    3,
                    shrink.as_ptr(),
                    shrink_layout.as_ptr(),
                    shrink_layout.len() as u32,
                )
            },
            0
        );
        output.fill(0);
        assert_eq!(
            unsafe {
                imparo_cuda_read_kv(1, 0, 0, output.as_mut_ptr(), output.len() as u64)
            },
            0
        );
        assert_eq!(source, output);

        // Cross an arena-alignment boundary so the physical reservation must grow.
        let grown = [0_u64, 8192, 64 * 5];
        let grown_layout = [
            imparo_backend::KvLayout::default(),
            imparo_backend::KvLayout {
                logical_slots: grown[1],
                ..initial_layout[1]
            },
            initial_layout[2],
        ];
        assert_eq!(
            unsafe {
                imparo_cuda_grow_kv_layout(
                    3,
                    grown.as_ptr(),
                    grown_layout.as_ptr(),
                    grown_layout.len() as u32,
                )
            },
            0
        );
        output.fill(0);
        assert_eq!(
            unsafe {
                imparo_cuda_read_kv(1, 0, 0, output.as_mut_ptr(), output.len() as u64)
            },
            0
        );
        assert_eq!(source, output);
        let mut allocated_after_grow = 0;
        assert_eq!(
            unsafe {
                imparo_cuda_memory_info(
                    core::ptr::null_mut(),
                    core::ptr::null_mut(),
                    &raw mut allocated_after_grow,
                )
            },
            0
        );
        assert!(allocated_after_grow >= allocated + 2 * 4096);
        assert_eq!(unsafe { imparo_cuda_kv_live_bytes(&raw mut live) }, 0);
        assert_eq!(live, initial[1] + initial[2]);

        // Step 8.4: one batch moves multiple content units and handles. Layer 1 K
        // deliberately leaves block 1 resident-but-unused between physical blocks
        // 0 and 2, while layer 2 V proves the batch is not one contiguous arena copy.
        let k_unit = initial[1];
        let v_unit = initial[2];
        let second_k_off = 2 * k_unit;
        assert!(second_k_off + k_unit <= grown[1]);
        assert_eq!(
            unsafe { imparo_cuda_kv_advise_reuse(1, 0, second_k_off, k_unit) },
            0
        );
        let pattern = |len: u64, salt: u8| -> Vec<u8> {
            (0..len)
                .map(|index| salt.wrapping_add((index as u8).wrapping_mul(17)))
                .collect()
        };
        let first_k = pattern(k_unit, 0x11);
        let first_v = pattern(v_unit, 0x47);
        let second_k = pattern(k_unit, 0x83);
        assert_eq!(
            unsafe {
                imparo_cuda_write_kv(1, 0, 0, first_k.as_ptr(), first_k.len() as u64)
            },
            0
        );
        assert_eq!(
            unsafe {
                imparo_cuda_write_kv(2, 1, 0, first_v.as_ptr(), first_v.len() as u64)
            },
            0
        );
        assert_eq!(
            unsafe {
                imparo_cuda_write_kv(
                    1,
                    0,
                    second_k_off,
                    second_k.as_ptr(),
                    second_k.len() as u64,
                )
            },
            0
        );

        let mut host_before = 0_u64;
        assert_eq!(
            unsafe { imparo_cuda_host_allocated_bytes(&raw mut host_before) },
            0
        );
        let first_host_bytes = k_unit + v_unit;
        let mut first_handle = 0_u64;
        let mut second_handle = 0_u64;
        assert_eq!(
            unsafe { imparo_cuda_host_alloc(first_host_bytes, &raw mut first_handle) },
            0
        );
        assert_ne!(first_handle, 0);
        assert_eq!(
            unsafe { imparo_cuda_host_alloc(k_unit, &raw mut second_handle) },
            0
        );
        assert_ne!(second_handle, 0);
        assert_ne!(first_handle, second_handle);
        let mut host_allocated = 0_u64;
        assert_eq!(
            unsafe { imparo_cuda_host_allocated_bytes(&raw mut host_allocated) },
            0
        );
        assert_eq!(host_allocated, host_before + first_host_bytes + k_unit);

        let host_spans = [
            imparo_backend::KvTransferSpan {
                host_handle: first_handle,
                layer: 1,
                is_v: 0,
                device_offset: 0,
                host_offset: 0,
                len: k_unit,
            },
            imparo_backend::KvTransferSpan {
                host_handle: second_handle,
                layer: 1,
                is_v: 0,
                device_offset: second_k_off,
                host_offset: 0,
                len: k_unit,
            },
            imparo_backend::KvTransferSpan {
                host_handle: first_handle,
                layer: 2,
                is_v: 1,
                device_offset: 0,
                host_offset: k_unit,
                len: v_unit,
            },
        ];
        assert_eq!(
            unsafe {
                imparo_cuda_kv_demote(host_spans.as_ptr(), host_spans.len() as u32)
            },
            0
        );
        let mut host_first_k = vec![0_u8; first_k.len()];
        let mut host_first_v = vec![0_u8; first_v.len()];
        let mut host_second_k = vec![0_u8; second_k.len()];
        assert_eq!(
            unsafe {
                imparo_cuda_host_read(
                    first_handle,
                    0,
                    host_first_k.as_mut_ptr(),
                    host_first_k.len() as u64,
                )
            },
            0
        );
        assert_eq!(
            unsafe {
                imparo_cuda_host_read(
                    first_handle,
                    k_unit,
                    host_first_v.as_mut_ptr(),
                    host_first_v.len() as u64,
                )
            },
            0
        );
        assert_eq!(
            unsafe {
                imparo_cuda_host_read(
                    second_handle,
                    0,
                    host_second_k.as_mut_ptr(),
                    host_second_k.len() as u64,
                )
            },
            0
        );
        assert_eq!(host_first_k, first_k);
        assert_eq!(host_first_v, first_v);
        assert_eq!(host_second_k, second_k);

        let zero_k = vec![0_u8; k_unit as usize];
        let zero_v = vec![0_u8; v_unit as usize];
        for (layer, is_v, off, zeros) in [
            (1_u32, 0_u32, 0_u64, zero_k.as_slice()),
            (1, 0, second_k_off, zero_k.as_slice()),
            (2, 1, 0, zero_v.as_slice()),
        ] {
            assert_eq!(
                unsafe {
                    imparo_cuda_write_kv(
                        layer,
                        is_v,
                        off,
                        zeros.as_ptr(),
                        zeros.len() as u64,
                    )
                },
                0
            );
            let mut cleared = vec![0xff_u8; zeros.len()];
            assert_eq!(
                unsafe {
                    imparo_cuda_read_kv(
                        layer,
                        is_v,
                        off,
                        cleared.as_mut_ptr(),
                        cleared.len() as u64,
                    )
                },
                0
            );
            assert_eq!(cleared, zeros);
        }
        assert_eq!(
            unsafe {
                imparo_cuda_kv_promote(host_spans.as_ptr(), host_spans.len() as u32)
            },
            0
        );
        for (span, expected) in [
            (&host_spans[0], first_k.as_slice()),
            (&host_spans[1], second_k.as_slice()),
            (&host_spans[2], first_v.as_slice()),
        ] {
            let mut restored = vec![0_u8; expected.len()];
            assert_eq!(
                unsafe {
                    imparo_cuda_read_kv(
                        span.layer,
                        span.is_v,
                        span.device_offset,
                        restored.as_mut_ptr(),
                        restored.len() as u64,
                    )
                },
                0
            );
            assert_eq!(restored, expected);
        }

        let mut one = 0_u8;
        assert_ne!(
            unsafe {
                imparo_cuda_host_read(first_handle, first_host_bytes, &raw mut one, 1)
            },
            0
        );
        assert_ne!(
            unsafe {
                imparo_cuda_host_write(
                    first_handle,
                    first_host_bytes,
                    &raw const one,
                    1,
                )
            },
            0
        );
        let invalid_host_span = [imparo_backend::KvTransferSpan {
            host_offset: first_host_bytes - 1,
            len: 2,
            ..host_spans[0]
        }];
        assert_ne!(
            unsafe { imparo_cuda_kv_demote(invalid_host_span.as_ptr(), 1) },
            0
        );
        assert_ne!(
            unsafe { imparo_cuda_kv_promote(invalid_host_span.as_ptr(), 1) },
            0
        );
        let invalid_device_span = [imparo_backend::KvTransferSpan {
            device_offset: grown[1] - 1,
            host_offset: 0,
            len: 2,
            ..host_spans[0]
        }];
        assert_ne!(
            unsafe { imparo_cuda_kv_demote(invalid_device_span.as_ptr(), 1) },
            0
        );
        assert_ne!(
            unsafe { imparo_cuda_kv_promote(invalid_device_span.as_ptr(), 1) },
            0
        );

        assert_eq!(unsafe { imparo_cuda_host_free(first_handle) }, 0);
        assert_ne!(unsafe { imparo_cuda_host_free(first_handle) }, 0);
        let mut replacement_handle = 0_u64;
        assert_eq!(
            unsafe { imparo_cuda_host_alloc(1, &raw mut replacement_handle) },
            0
        );
        assert_ne!(replacement_handle, first_handle);
        assert_ne!(
            unsafe { imparo_cuda_host_read(first_handle, 0, &raw mut one, 1) },
            0
        );
        assert_ne!(unsafe { imparo_cuda_kv_demote(host_spans.as_ptr(), 1) }, 0);
        assert_eq!(unsafe { imparo_cuda_host_free(replacement_handle) }, 0);
        assert_eq!(
            unsafe { imparo_cuda_host_allocated_bytes(&raw mut host_allocated) },
            0
        );
        assert_eq!(host_allocated, host_before + k_unit);
        assert_eq!(unsafe { imparo_cuda_host_free(second_handle) }, 0);
        assert_eq!(
            unsafe { imparo_cuda_host_allocated_bytes(&raw mut host_allocated) },
            0
        );
        assert_eq!(host_allocated, host_before);
        assert_eq!(
            unsafe { imparo_cuda_kv_advise_free(1, 0, second_k_off, k_unit) },
            0
        );
        assert_eq!(unsafe { imparo_cuda_kv_live_bytes(&raw mut live) }, 0);
        assert_eq!(live, initial[1] + initial[2]);

        // Step 8.3-B: exercise K and V stores for all cache types and both
        // quantized dequant implementations. The partial-domain table maps only
        // logical block 0 to physical block 1, while dequant visits 64 logical
        // rows. An implementation that ignores the table or maps its block id a
        // second time cannot satisfy both physical-row assertions.
        const WIDTH: u32 = 32;
        const SLOTS: u64 = 576;
        const SRC: u32 = 0;
        const SCRATCH: u32 = 1;
        const SCRATCH_ALT: u32 = 2;
        let partial_domain = [1_u32];
        let source: Vec<f32> = (1..=WIDTH).map(|value| value as f32 / 7.0).collect();
        let zero_words = vec![0_f32; (SLOTS * u64::from(WIDTH) / 2) as usize];
        for (cache_type, row_bytes) in [(1_u32, 64_u64), (2, 18), (8, 34)] {
            unsafe { imparo_cuda_set_kv_types(cache_type, cache_type) };
            let bytes = [SLOTS * row_bytes];
            let layout = [imparo_backend::KvLayout {
                layer: 0,
                reserved: 0,
                logical_slots: SLOTS,
                k_stride: row_bytes,
                v_stride: row_bytes,
            }];
            assert_eq!(
                unsafe {
                    imparo_cuda_alloc_kv_layout(1, bytes.as_ptr(), layout.as_ptr(), 1)
                },
                0
            );
            assert_eq!(
                unsafe { imparo_cuda_set_kv_pages(0, partial_domain.as_ptr(), 1) },
                0
            );
            assert_eq!(unsafe { imparo_cuda_alloc(SRC, u64::from(WIDTH) * 4) }, 0);
            assert_eq!(
                unsafe { imparo_cuda_alloc(SCRATCH, SLOTS * u64::from(WIDTH) * 2) },
                0
            );
            assert_eq!(
                unsafe { imparo_cuda_alloc(SCRATCH_ALT, SLOTS * u64::from(WIDTH) * 2) },
                0
            );
            for is_v in 0..=1_u32 {
                unsafe {
                    imparo_cuda_begin();
                    imparo_cuda_write(SRC, 0, source.as_ptr(), source.len() as u64);
                    imparo_cuda_kv_store(SRC, 0, WIDTH, 0, 1, is_v, 0);
                }
                assert_eq!(unsafe { imparo_cuda_end() }, 0);
                let mut physical_0 = vec![0_u8; row_bytes as usize];
                let mut physical_64 = vec![0_u8; row_bytes as usize];
                assert_eq!(
                    unsafe {
                        imparo_cuda_read_kv(
                            0,
                            is_v,
                            0,
                            physical_0.as_mut_ptr(),
                            row_bytes,
                        )
                    },
                    0
                );
                assert_eq!(
                    unsafe {
                        imparo_cuda_read_kv(
                            0,
                            is_v,
                            64 * row_bytes,
                            physical_64.as_mut_ptr(),
                            row_bytes,
                        )
                    },
                    0
                );
                assert!(physical_0.iter().all(|byte| *byte == 0));
                assert!(physical_64.iter().any(|byte| *byte != 0));

                if cache_type != 1 {
                    unsafe {
                        imparo_cuda_begin();
                        imparo_cuda_write(
                            SCRATCH,
                            0,
                            zero_words.as_ptr(),
                            zero_words.len() as u64,
                        );
                        imparo_cuda_write(
                            SCRATCH_ALT,
                            0,
                            zero_words.as_ptr(),
                            zero_words.len() as u64,
                        );
                        imparo_cuda_kv_dequant(0, WIDTH, 64, is_v, SCRATCH, 0);
                        // Same source key but a different destination must not hit
                        // the memoized dequant result.
                        imparo_cuda_kv_dequant(0, WIDTH, 64, is_v, SCRATCH_ALT, 0);
                    }
                    assert_eq!(unsafe { imparo_cuda_end() }, 0);
                    for scratch in [SCRATCH, SCRATCH_ALT] {
                        let mut half_0 = [0_f32; (WIDTH / 2) as usize];
                        let mut half_64 = [0_f32; (WIDTH / 2) as usize];
                        unsafe {
                            imparo_cuda_read(
                                scratch,
                                0,
                                half_0.as_mut_ptr(),
                                u64::from(WIDTH / 2),
                            );
                            imparo_cuda_read(
                                scratch,
                                u64::from(64 * WIDTH / 2),
                                half_64.as_mut_ptr(),
                                u64::from(WIDTH / 2),
                            );
                        }
                        assert!(half_0.iter().all(|pair| pair.to_bits() == 0));
                        assert!(half_64.iter().any(|pair| pair.to_bits() != 0));
                    }
                }

                // Recreate zeroed physical storage, keep a non-identity table, and
                // prove device ring addressing wins in both store and dequant.
                assert_eq!(
                    unsafe {
                        imparo_cuda_alloc_kv_layout(
                            1,
                            bytes.as_ptr(),
                            layout.as_ptr(),
                            1,
                        )
                    },
                    0
                );
                assert_eq!(
                    unsafe { imparo_cuda_set_kv_pages(0, partial_domain.as_ptr(), 1) },
                    0
                );
                unsafe {
                    imparo_cuda_begin();
                    imparo_cuda_write(SRC, 0, source.as_ptr(), source.len() as u64);
                    imparo_cuda_kv_store(SRC, 0, WIDTH, 0, 1, is_v, 511);
                }
                assert_eq!(unsafe { imparo_cuda_end() }, 0);
                physical_0.fill(0);
                physical_64.fill(0);
                assert_eq!(
                    unsafe {
                        imparo_cuda_read_kv(
                            0,
                            is_v,
                            0,
                            physical_0.as_mut_ptr(),
                            row_bytes,
                        )
                    },
                    0
                );
                assert_eq!(
                    unsafe {
                        imparo_cuda_read_kv(
                            0,
                            is_v,
                            64 * row_bytes,
                            physical_64.as_mut_ptr(),
                            row_bytes,
                        )
                    },
                    0
                );
                assert!(physical_0.iter().any(|byte| *byte != 0));
                assert!(physical_64.iter().all(|byte| *byte == 0));

                if cache_type != 1 {
                    unsafe {
                        imparo_cuda_begin();
                        imparo_cuda_write(
                            SCRATCH,
                            0,
                            zero_words.as_ptr(),
                            zero_words.len() as u64,
                        );
                        // Same layer/width/slots/type/scratch as the page case;
                        // changing ring must invalidate the memo key.
                        imparo_cuda_kv_dequant(0, WIDTH, 64, is_v, SCRATCH, 511);
                    }
                    assert_eq!(unsafe { imparo_cuda_end() }, 0);
                    let mut half_0 = [0_f32; (WIDTH / 2) as usize];
                    let mut half_64 = [0_f32; (WIDTH / 2) as usize];
                    unsafe {
                        imparo_cuda_read(
                            SCRATCH,
                            0,
                            half_0.as_mut_ptr(),
                            u64::from(WIDTH / 2),
                        );
                        imparo_cuda_read(
                            SCRATCH,
                            u64::from(64 * WIDTH / 2),
                            half_64.as_mut_ptr(),
                            u64::from(WIDTH / 2),
                        );
                    }
                    assert!(half_0.iter().any(|pair| pair.to_bits() != 0));
                    assert!(half_64.iter().all(|pair| pair.to_bits() == 0));
                }
            }
        }

        // A workflow that forgets its backend-owned half scratch must fail before
        // launch.  The old behavior launched at an undersized/null pointer and only
        // surfaced later as cudaErrorIllegalAddress.
        unsafe { imparo_cuda_set_kv_types(2, 2) };
        assert_eq!(unsafe { imparo_cuda_alloc(SCRATCH, 8) }, 0);
        unsafe {
            imparo_cuda_begin();
            imparo_cuda_kv_dequant(0, WIDTH, 64, 0, SCRATCH, 0);
        }
        assert_ne!(unsafe { imparo_cuda_end() }, 0);
    }

    #[test]
    fn exact128_ple_ffn_candidate_matches_the_safe_route_and_reports_all_hits() {
        if std::env::var_os("IMPARO_CUDA_EXACT128_SMOKE").is_none() {
            return;
        }

        const N_TOK: u32 = 128;
        const N_EMBD: u32 = 2560;
        const N_FF: u32 = 10240;
        const PLE: u32 = 256;
        const N_LAYERS: u32 = 42;
        const Q4: u32 = 1;
        // This native smoke runs one FFN and one PLE but no Attention. The packed
        // evidence therefore proves both final commits in addition to the five
        // stage bits it exercises.
        const EXACT128_MASK: u32 = 0x1f | (1 << 12) | (1 << 18);
        const CUR: u32 = imparo_backend::BufId::Cur as u32;
        const G: u32 = imparo_backend::BufId::G as u32;
        const X: u32 = imparo_backend::BufId::X as u32;
        const PLE_GATE: u32 = imparo_backend::BufId::Model0 as u32;
        const PLE_BACK: u32 = imparo_backend::BufId::Model1 as u32;
        const PER_LAYER: u32 = imparo_backend::BufId::Model2 as u32;

        let q4_bytes = |n_in: u32, n_out: u32| -> usize {
            (u64::from(n_in / 32) * 18 * u64::from(n_out)) as usize
        };
        let align = |n: usize| (n + 16_383) & !16_383;
        let mut cursor = 0_usize;
        let mut reserve = |bytes: usize| {
            let off = cursor;
            cursor = align(cursor + bytes);
            off
        };
        let ffn_gate_off = reserve(q4_bytes(N_EMBD, N_FF));
        let ffn_up_off = reserve(q4_bytes(N_EMBD, N_FF));
        let ffn_down_off = reserve(q4_bytes(N_FF, N_EMBD));
        let ple_gate_off = reserve(q4_bytes(N_EMBD, PLE));
        let ple_proj_off = reserve(q4_bytes(PLE, N_EMBD));
        let mut weights = vec![0_u8; cursor];
        let mut fill_q4 = |off: usize, bytes: usize, salt: usize| {
            for (block_index, block) in
                weights[off..off + bytes].chunks_exact_mut(18).enumerate()
            {
                // 2^-6 keeps the three-projection synthetic transaction finite while
                // remaining well above the activation quantizer's zero floor.
                block[..2].copy_from_slice(&0x2400_u16.to_le_bytes());
                for (pair, packed) in block[2..].iter_mut().enumerate() {
                    let low = (block_index + pair + salt) & 15;
                    let high = (3 * block_index + 5 * pair + salt + 1) & 15;
                    *packed = low as u8 | ((high as u8) << 4);
                }
            }
        };
        fill_q4(ffn_gate_off, q4_bytes(N_EMBD, N_FF), 1);
        fill_q4(ffn_up_off, q4_bytes(N_EMBD, N_FF), 3);
        fill_q4(ffn_down_off, q4_bytes(N_FF, N_EMBD), 5);
        fill_q4(ple_gate_off, q4_bytes(N_EMBD, PLE), 7);
        fill_q4(ple_proj_off, q4_bytes(PLE, N_EMBD), 9);

        unsafe { imparo_cuda_enable_tuner_lab() };
        assert_eq!(
            unsafe {
                imparo_cuda_init(
                    weights.as_ptr().cast(),
                    weights.len() as u64,
                    core::ptr::null(),
                    0,
                )
            },
            0
        );
        for (id, elements) in [
            (CUR, N_TOK * N_EMBD),
            (G, N_TOK * N_FF),
            (X, N_TOK * N_EMBD),
            (PLE_GATE, N_TOK * PLE),
            (PLE_BACK, N_TOK * N_EMBD),
            (PER_LAYER, N_TOK * PLE * N_LAYERS),
        ] {
            assert_eq!(unsafe { imparo_cuda_alloc(id, u64::from(elements) * 4) }, 0);
        }
        let cur: Vec<f32> = (0..N_TOK * N_EMBD)
            .map(|index| {
                let bits = index.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (((bits >> 8) & 0xffff) as f32 / 65_535.0 - 0.5) * 0.2
            })
            .collect();
        let per_layer: Vec<f32> = (0..N_TOK * PLE * N_LAYERS)
            .map(|index| 0.2 + (index % 31) as f32 / 512.0)
            .collect();

        let run = |enabled: bool| -> (Vec<f32>, Vec<f32>) {
            unsafe {
                imparo_cuda_set_knob(39, u32::from(enabled));
                assert_eq!(imparo_cuda_set_batch_geometry(0, N_TOK, 0), 0);
                imparo_cuda_begin();
                imparo_cuda_write(CUR, 0, cur.as_ptr(), cur.len() as u64);
                imparo_cuda_write(
                    PER_LAYER,
                    0,
                    per_layer.as_ptr(),
                    per_layer.len() as u64,
                );
                let fused = imparo_cuda_ffn_gated_down(
                    Q4,
                    ffn_gate_off as u64,
                    Q4,
                    ffn_up_off as u64,
                    Q4,
                    ffn_down_off as u64,
                    N_EMBD,
                    N_FF,
                    N_EMBD,
                    CUR,
                    G,
                    X,
                    N_TOK,
                ) != 0;
                if enabled {
                    assert!(fused, "exact-128 FFN candidate was not reached");
                } else {
                    assert!(!fused, "safe selector unexpectedly reached the candidate");
                    imparo_cuda_matmat(
                        Q4,
                        ffn_gate_off as u64,
                        N_EMBD,
                        N_FF,
                        CUR,
                        G,
                        N_TOK,
                        0,
                    );
                    imparo_cuda_set_epilogue(1);
                    imparo_cuda_matmat(
                        Q4,
                        ffn_up_off as u64,
                        N_EMBD,
                        N_FF,
                        CUR,
                        G,
                        N_TOK,
                        0,
                    );
                    imparo_cuda_set_epilogue(0);
                    imparo_cuda_matmat(
                        Q4,
                        ffn_down_off as u64,
                        N_FF,
                        N_EMBD,
                        G,
                        X,
                        N_TOK,
                        0,
                    );
                }
                imparo_cuda_ple_project(
                    Q4,
                    ple_gate_off as u64,
                    Q4,
                    ple_proj_off as u64,
                    N_EMBD,
                    PLE,
                    CUR,
                    PLE_GATE,
                    PER_LAYER,
                    0,
                    PLE * N_LAYERS,
                    PLE_BACK,
                    N_TOK,
                );
                assert_eq!(imparo_cuda_end(), 0);
                assert_eq!(
                    imparo_cuda_exact128_route_hits_lab(),
                    if enabled { EXACT128_MASK } else { 0 }
                );
            }
            let mut ffn = vec![0_f32; (N_TOK * N_EMBD) as usize];
            let mut ple = vec![0_f32; (N_TOK * N_EMBD) as usize];
            unsafe {
                imparo_cuda_read(X, 0, ffn.as_mut_ptr(), ffn.len() as u64);
                imparo_cuda_read(PLE_BACK, 0, ple.as_mut_ptr(), ple.len() as u64);
            }
            (ffn, ple)
        };
        let (safe_ffn, safe_ple) = run(false);
        let (candidate_ffn, candidate_ple) = run(true);

        let assert_close = |name: &str, expected: &[f32], actual: &[f32]| {
            let (mut dot, mut expected2, mut actual2, mut diff2) =
                (0_f64, 0_f64, 0_f64, 0_f64);
            for (&a, &b) in expected.iter().zip(actual) {
                assert!(
                    a.is_finite() && b.is_finite(),
                    "{name} produced non-finite output"
                );
                let (a, b) = (f64::from(a), f64::from(b));
                dot += a * b;
                expected2 += a * a;
                actual2 += b * b;
                diff2 += (a - b) * (a - b);
            }
            assert!(
                expected2 > 0.0 && actual2 > 0.0,
                "{name} was not exercised: safe_norm2={expected2} candidate_norm2={actual2}"
            );
            let cosine = dot / (expected2 * actual2).sqrt();
            let relative_l2 = (diff2 / expected2).sqrt();
            assert!(cosine >= 0.995, "{name} cosine={cosine}");
            assert!(relative_l2 <= 0.15, "{name} relative_l2={relative_l2}");
            eprintln!("{name}: cosine={cosine:.9} relative_l2={relative_l2:.9}");
        };
        assert_close("exact-128 FFN", &safe_ffn, &candidate_ffn);
        assert_close("exact-128 PLE", &safe_ple, &candidate_ple);

        // A selector value alone is not proof. Misaligning the absolute geometry keeps
        // the FFN specialization eligible but forces PLE back to its safe route; the
        // incomplete mask is exactly what the tuner must reject.
        unsafe {
            imparo_cuda_set_knob(39, 1);
            assert_eq!(imparo_cuda_set_batch_geometry(1, N_TOK, 0), 0);
            imparo_cuda_begin();
            imparo_cuda_write(CUR, 0, cur.as_ptr(), cur.len() as u64);
            imparo_cuda_write(PER_LAYER, 0, per_layer.as_ptr(), per_layer.len() as u64);
            assert_ne!(
                imparo_cuda_ffn_gated_down(
                    Q4,
                    ffn_gate_off as u64,
                    Q4,
                    ffn_up_off as u64,
                    Q4,
                    ffn_down_off as u64,
                    N_EMBD,
                    N_FF,
                    N_EMBD,
                    CUR,
                    G,
                    X,
                    N_TOK,
                ),
                0
            );
            imparo_cuda_ple_project(
                Q4,
                ple_gate_off as u64,
                Q4,
                ple_proj_off as u64,
                N_EMBD,
                PLE,
                CUR,
                PLE_GATE,
                PER_LAYER,
                0,
                PLE * N_LAYERS,
                PLE_BACK,
                N_TOK,
            );
            assert_eq!(imparo_cuda_end(), 0);
            let partial = imparo_cuda_exact128_route_hits_lab();
            assert_ne!(partial, EXACT128_MASK);
            assert_eq!(partial & 0x3, 0, "misaligned PLE must not claim a hit");
        }
    }
}

#[cfg(all(test, feature = "cuda-static"))]
mod native_step1_smoke_tests {
    use super::*;

    #[test]
    fn sm86_events_profile_and_all_probe_kinds_are_live() {
        if std::env::var_os("IMPARO_CUDA_STEP1_SMOKE").is_none() {
            return;
        }
        assert_eq!(unsafe { imparo_cuda_set_tuner_mode(1) }, 0);
        let weights = vec![0_u8; 256 * (256 / 32) * 18];
        assert_eq!(
            unsafe {
                imparo_cuda_init(
                    weights.as_ptr().cast(),
                    weights.len() as u64,
                    core::ptr::null(),
                    0,
                )
            },
            0
        );

        let mut profile = DeviceProfileWire {
            struct_bytes: size_of::<DeviceProfileWire>() as u32,
            ..DeviceProfileWire::default()
        };
        assert_eq!(
            unsafe {
                imparo_cuda_device_profile(
                    &raw mut profile,
                    size_of::<DeviceProfileWire>() as u32,
                )
            },
            0
        );
        assert!(profile.max_threads >= 256);
        assert!(profile.threadgroup_bytes >= 32 << 10);
        eprintln!(
            "step1 device max_threads={} threadgroup_bytes={}",
            profile.max_threads, profile.threadgroup_bytes
        );

        for (kind, a, b, c, d, e) in [
            (1, 0, 4, 128, 128, 0),
            (2, 1 << 20, 2, 16, 128, 0),
            (3, 8, 4, 128, 512, 3),
            (4, 16, 0, 0, 0, 0),
            (5, 8, 0, 0, 0, 0),
            (6, 32, 0, 0, 0, 0),
        ] {
            let value = unsafe { imparo_cuda_probe(kind, a, b, c, d, e) };
            assert!(
                value.is_finite() && value > 0.0,
                "probe kind {kind} returned {value}"
            );
            eprintln!("step1 probe kind={kind} value={value:.6}");
        }

        const WORDS: usize = 1024;
        assert_eq!(unsafe { imparo_cuda_alloc(0, (WORDS * 4) as u64) }, 0);
        assert_eq!(unsafe { imparo_cuda_alloc(1, (WORDS * 4) as u64) }, 0);
        let values = vec![1.0_f32; WORDS];
        unsafe {
            imparo_cuda_write(0, 0, values.as_ptr(), WORDS as u64);
            imparo_cuda_write(1, 0, values.as_ptr(), WORDS as u64);
            imparo_cuda_begin();
            imparo_cuda_add(0, 1, WORDS as u32);
        }
        assert_eq!(unsafe { imparo_cuda_end() }, 0);
        let gpu_us = unsafe { imparo_cuda_last_gpu_us() };
        assert!(gpu_us.is_finite() && gpu_us > 0.0);
        eprintln!("step1 event gpu_us={gpu_us:.6}");

        // Positive and negative proof: a measured narrow matmat must report the MMVQ slot it
        // actually consulted, and must not accidentally satisfy an attention-slot
        // expectation merely because some CUDA dispatch happened.
        unsafe {
            imparo_cuda_set_knob(19, 2);
        }
        assert_eq!(unsafe { imparo_cuda_dispatch_expectation(1_u64 << 19) }, 0);
        unsafe {
            imparo_cuda_dispatch_proof_reset();
            imparo_cuda_begin();
            imparo_cuda_matmat(1, 0, 256, 256, 0, 1, 1, 0);
        }
        assert_eq!(unsafe { imparo_cuda_end() }, 0);
        let mut proof = DispatchProofWire {
            struct_bytes: size_of::<DispatchProofWire>() as u32,
            ..DispatchProofWire::default()
        };
        assert_eq!(
            unsafe {
                imparo_cuda_dispatch_proof(
                    &raw mut proof,
                    size_of::<DispatchProofWire>() as u32,
                )
            },
            0
        );
        assert_ne!(proof.observed_knob_mask & (1_u64 << 19), 0);

        assert_eq!(unsafe { imparo_cuda_dispatch_expectation(1_u64 << 7) }, 0);
        unsafe {
            imparo_cuda_dispatch_proof_reset();
            imparo_cuda_begin();
            imparo_cuda_matmat(1, 0, 256, 256, 0, 1, 1, 0);
        }
        assert_eq!(unsafe { imparo_cuda_end() }, 0);
        let mut wrong_family = DispatchProofWire {
            struct_bytes: size_of::<DispatchProofWire>() as u32,
            ..DispatchProofWire::default()
        };
        assert_eq!(
            unsafe {
                imparo_cuda_dispatch_proof(
                    &raw mut wrong_family,
                    size_of::<DispatchProofWire>() as u32,
                )
            },
            0
        );
        assert_eq!(wrong_family.observed_knob_mask & (1_u64 << 7), 0);
        assert_eq!(unsafe { imparo_cuda_set_tuner_mode(0) }, 0);
    }
}
