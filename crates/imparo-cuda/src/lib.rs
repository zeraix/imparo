//! Direct CUDA backend for Imparo.
//!
//! `context` owns the device, stream, resident/paged weights, and memory reporting;
//! `backend_impl` bridges the backend contract to native kernels; `knobs` exposes
//! parameters consumed by those kernels to the adaptive tuner.
//!
//! With the feature off, the crate remains buildable on hosts without CUDA.

include!(concat!(env!("OUT_DIR"), "/imparo_cuda_abi.rs"));

/// Canonical bytes authenticated for every independently published CUDA backend.
#[doc(hidden)]
pub fn cuda_backend_signed_message(
    platform: &str,
    sm: u32,
    abi: u32,
    bytes: u64,
    sha256: &str,
) -> Vec<u8> {
    format!("imparo-cuda-backend-v1\0{platform}\0{sm}\0{abi}\0{bytes}\0{sha256}")
        .into_bytes()
}

#[cfg(test)]
mod release_contract_tests {
    #[test]
    fn signed_message_binds_the_current_backend_abi() {
        let message = super::cuda_backend_signed_message(
            "windows-x86_64",
            86,
            super::CUDA_BACKEND_ABI,
            123,
            "abcd",
        );
        assert_eq!(
            message,
            b"imparo-cuda-backend-v1\x00windows-x86_64\x0086\x0026\x00123\x00abcd"
        );
    }

    #[test]
    fn abi_26_exports_paging_host_transfer_lfm2_program_and_prefill_cache_surfaces() {
        let exports = include_str!("../native/imparo_cuda.def");
        for symbol in [
            "imparo_cuda_alloc_kv_layout",
            "imparo_cuda_grow_kv_layout",
            "imparo_cuda_set_kv_pages",
            "imparo_cuda_host_alloc",
            "imparo_cuda_host_free",
            "imparo_cuda_host_read",
            "imparo_cuda_host_write",
            "imparo_cuda_host_allocated_bytes",
            "imparo_cuda_host_profile",
            "imparo_cuda_kv_demote",
            "imparo_cuda_kv_promote",
            "imparo_cuda_shortconv",
            "imparo_cuda_shortconv_snapshot",
            "imparo_cuda_silu",
            "imparo_cuda_silu_mul",
            "imparo_cuda_program_pack_install",
            "imparo_cuda_program_catalog_identity",
            "imparo_cuda_program_bind",
            "imparo_cuda_program_freeze",
            "imparo_cuda_program_launch",
            "imparo_cuda_program_reset",
            "imparo_cuda_prepare_quantized_weight_cache",
            "imparo_cuda_ffn_gated_down",
        ] {
            assert!(exports.lines().any(|line| line.trim() == symbol));
        }
    }

    #[test]
    fn per_sm_release_plugins_exclude_the_cublas_kernel_lab() {
        let native = include_str!("../native/imparo_cuda.cu");
        let source_build = include_str!("../build.rs");
        let windows_release =
            include_str!("../../../.github/workflows/release-windows.yml");
        let linux_release =
            include_str!("../../../.github/workflows/release-linux.yml");

        assert!(native.contains("#if defined(IMPARO_CUDA_ENABLE_CUBLAS_LAB)"));
        assert!(source_build.contains("-DIMPARO_CUDA_ENABLE_CUBLAS_LAB=1"));
        assert!(source_build.contains("cargo:rustc-link-lib=dylib=stdc++"));
        for release in [windows_release, linux_release] {
            assert!(!release.contains("IMPARO_CUDA_ENABLE_CUBLAS_LAB"));
            assert!(!release.contains("-lcublas"));
        }
    }

    #[cfg(feature = "cuda-dynamic")]
    #[test]
    fn release_catalog_matches_the_runtime_abi() {
        let catalog: serde_json::Value =
            serde_json::from_str(include_str!("../cuda-sm.json")).unwrap();
        assert_eq!(
            catalog["backend_abi"].as_u64(),
            Some(u64::from(super::CUDA_BACKEND_ABI))
        );
        let sms = catalog["sms"].as_array().unwrap();
        assert!(!sms.is_empty());
        assert!(
            sms.windows(2)
                .all(|pair| pair[0].as_u64() < pair[1].as_u64())
        );
        assert_eq!(
            sms.iter()
                .map(|sm| sm.as_u64().unwrap() as u32)
                .collect::<Vec<_>>(),
            super::CUDA_SUPPORTED_SMS
        );
    }
}

#[cfg(all(feature = "cuda-static", feature = "cuda-dynamic"))]
compile_error!("select exactly one of imparo-cuda/cuda-static and cuda-dynamic");

#[cfg(any(feature = "cuda-static", feature = "cuda-dynamic"))]
mod backend_impl;
#[cfg(any(feature = "cuda-static", feature = "cuda-dynamic"))]
pub mod context;
#[cfg(any(feature = "cuda-static", feature = "cuda-dynamic"))]
pub mod correctness;
#[cfg(any(feature = "cuda-static", feature = "cuda-dynamic"))]
pub mod program_pack;
#[cfg(any(feature = "cuda-static", feature = "cuda-dynamic"))]
pub use context::CudaRuntimeIdentity;

/// Query the selected CUDA device and backend artifact/build identity before model init.
#[cfg(any(feature = "cuda-static", feature = "cuda-dynamic"))]
pub fn runtime_identity() -> Result<CudaRuntimeIdentity, String> {
    context::CudaContext::get().runtime_identity()
}
#[cfg(feature = "cuda-dynamic")]
mod dylib;
#[cfg(any(feature = "cuda-static", feature = "cuda-dynamic"))]
mod ffi;
#[cfg(any(feature = "cuda-static", feature = "cuda-dynamic"))]
pub mod knobs;
#[cfg(feature = "cuda-dynamic")]
mod loader;
#[cfg(any(feature = "cuda-static", feature = "cuda-dynamic"))]
pub use backend_impl::{
    CudaBackend, correctness_receipt_template, install_correctness_identity,
    prepare_tuner_lab,
};
