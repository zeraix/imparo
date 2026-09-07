//! Process-global handle for the native CUDA state.
//!
//! Device ownership, streams, weights, activation slots and KV allocations remain in
//! `native/imparo_cuda.cu`. Rust owns only the safe composition surface used by the
//! current v2 backend trait.

use crate::ffi::{
    RuntimeIdentityWire, backend_artifact_sha256, imparo_cuda_device_tag,
    imparo_cuda_init, imparo_cuda_kv_live_bytes, imparo_cuda_memory_info,
    imparo_cuda_runtime_identity, imparo_cuda_set_kv_types,
    imparo_cuda_weights_resident,
};
use imparo_backend::StreamedWeightSpan;
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, Default)]
pub struct MemoryInfo {
    pub free: u64,
    pub total: u64,
    pub allocated: u64,
    pub weights_resident: bool,
}

/// Complete CUDA runtime/backend identity used by correctness receipts.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaRuntimeIdentity {
    pub device_uuid: [u8; 16],
    pub device_sm: u32,
    pub driver_version: u32,
    pub runtime_version: u32,
    pub backend_abi: u32,
    /// Semantic hash of native sources and every code-generation input.
    pub backend_build_sha256: [u8; 32],
    /// Exact selected plugin bytes for dynamic builds; absent for a statically linked
    /// backend because the backend is not a separable file after final linking.
    pub backend_artifact_sha256: Option<[u8; 32]>,
}

impl CudaRuntimeIdentity {
    /// Receipt identity: exact plugin artifact when one exists, otherwise the stable
    /// native build identity. This never hashes the common server/tuner executable.
    #[must_use]
    pub fn backend_fingerprint_sha256(&self) -> [u8; 32] {
        self.backend_artifact_sha256
            .unwrap_or(self.backend_build_sha256)
    }

    #[must_use]
    pub fn device_uuid_string(&self) -> String {
        use std::fmt::Write as _;
        let mut output = String::with_capacity(36);
        for (index, byte) in self.device_uuid.iter().enumerate() {
            if matches!(index, 4 | 6 | 8 | 10) {
                output.push('-');
            }
            write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
        }
        output
    }
}

#[derive(Debug, Default)]
pub struct CudaContext;

static CONTEXT: OnceLock<CudaContext> = OnceLock::new();

impl CudaContext {
    /// One handle per process. Merely selecting the backend does not touch the GPU.
    pub fn get() -> &'static Self {
        CONTEXT.get_or_init(Self::default)
    }

    /// Initialize the native state from the live GGUF mapping.
    ///
    /// # Safety
    /// `base..base+len` must remain readable for the process lifetime. The native
    /// bounded-weight path may stage slices from that mapping after this call returns.
    pub unsafe fn init_weights(
        &self,
        base: *const u8,
        len: u64,
        streamed: &[StreamedWeightSpan],
    ) -> Result<(), i32> {
        let count = u32::try_from(streamed.len()).map_err(|_| 1)?;
        let spans = if streamed.is_empty() {
            core::ptr::null()
        } else {
            streamed.as_ptr().cast()
        };
        let rc = unsafe { imparo_cuda_init(base.cast(), len, spans, count) };
        if rc == 0 { Ok(()) } else { Err(rc) }
    }

    /// Query identity before weights/native execution are initialized.
    ///
    /// This uses the same `IMPARO_CUDA_DEVICE` selection as `init_weights` and is safe
    /// for a composition root to call while constructing a correctness fingerprint.
    pub fn runtime_identity(&self) -> Result<CudaRuntimeIdentity, String> {
        let mut wire = RuntimeIdentityWire::default();
        let wire_bytes = u32::try_from(core::mem::size_of::<RuntimeIdentityWire>())
            .map_err(|_| "CUDA runtime identity wire is too large".to_string())?;
        let rc = unsafe { imparo_cuda_runtime_identity(&raw mut wire, wire_bytes) };
        if rc != 0 {
            return Err(format!("CUDA runtime identity query failed with code {rc}"));
        }
        if wire.struct_bytes != wire_bytes
            || wire.backend_abi != crate::CUDA_BACKEND_ABI
            || wire.device_sm < 50
            || wire.driver_version == 0
            || wire.runtime_version == 0
            || wire.device_uuid == [0; 16]
            || wire.backend_build_sha256 == [0; 32]
        {
            return Err(
                "CUDA runtime returned an incomplete or incompatible identity".into(),
            );
        }
        let artifact = backend_artifact_sha256()?;
        if artifact == Some([0; 32]) {
            return Err("CUDA backend artifact identity is zero".into());
        }
        Ok(CudaRuntimeIdentity {
            device_uuid: wire.device_uuid,
            device_sm: wire.device_sm,
            driver_version: wire.driver_version,
            runtime_version: wire.runtime_version,
            backend_abi: wire.backend_abi,
            backend_build_sha256: wire.backend_build_sha256,
            backend_artifact_sha256: artifact,
        })
    }

    #[must_use]
    pub fn memory_info(&self) -> MemoryInfo {
        let mut out = MemoryInfo::default();
        let rc = unsafe {
            imparo_cuda_memory_info(
                &raw mut out.free,
                &raw mut out.total,
                &raw mut out.allocated,
            )
        };
        if rc == 0 {
            out.weights_resident = unsafe { imparo_cuda_weights_resident() != 0 };
        }
        out
    }

    /// Logical bytes currently owned inside the fixed-offset KV arena. This is
    /// separate from `allocated`: releasing a request page returns suballocator
    /// ownership without synchronizing the CUDA driver or freeing the arena.
    pub fn kv_live_bytes(&self) -> Result<u64, i32> {
        let mut bytes = 0;
        let rc = unsafe { imparo_cuda_kv_live_bytes(&raw mut bytes) };
        if rc == 0 { Ok(bytes) } else { Err(rc) }
    }

    pub fn set_kv_types(&self, k: u32, v: u32) {
        unsafe { imparo_cuda_set_kv_types(k, v) };
    }

    #[must_use]
    pub fn device_tag(&self) -> String {
        let mut bytes = [0_u8; 192];
        let n =
            unsafe { imparo_cuda_device_tag(bytes.as_mut_ptr(), bytes.len() as u32) };
        let n = usize::try_from(n).unwrap_or(0).min(bytes.len());
        if n == 0 {
            "cuda".to_string()
        } else {
            format!("cuda:{}", String::from_utf8_lossy(&bytes[..n]))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_wire_layout_matches_native_abi() {
        assert_eq!(core::mem::size_of::<RuntimeIdentityWire>(), 68);
        assert_eq!(core::mem::align_of::<RuntimeIdentityWire>(), 4);
    }

    #[test]
    fn dynamic_artifact_takes_precedence_over_static_build_identity() {
        let mut identity = CudaRuntimeIdentity {
            device_uuid: [1; 16],
            device_sm: 86,
            driver_version: 1,
            runtime_version: 1,
            backend_abi: crate::CUDA_BACKEND_ABI,
            backend_build_sha256: [2; 32],
            backend_artifact_sha256: None,
        };
        assert_eq!(identity.backend_fingerprint_sha256(), [2; 32]);
        identity.backend_artifact_sha256 = Some([3; 32]);
        assert_eq!(identity.backend_fingerprint_sha256(), [3; 32]);
        assert_eq!(
            identity.device_uuid_string(),
            "01010101-0101-0101-0101-010101010101"
        );
    }
}
