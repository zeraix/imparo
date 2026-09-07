use crate::ffi::{
    CUDA_PROGRAM_ABI_V1, ProgramCatalogIdentityWire,
    imparo_cuda_program_catalog_identity,
};
use imparo_backend::ProgramCatalogIdentity;

pub const PROGRAM_PACK_ABI: u32 = imparo_program_pack::PROGRAM_PACK_ABI;

/// PR-E deliberately does not invent the PR-G catalog encodings.
#[must_use]
pub const fn pending_identity() -> ProgramCatalogIdentity {
    ProgramCatalogIdentity {
        program_pack_abi: PROGRAM_PACK_ABI,
        identity_ready: false,
        pack_set_sha256: [0; 32],
        candidate_catalog_sha256: [0; 32],
        eligible_candidate_set_sha256: [0; 32],
    }
}

pub fn query_native_catalog() -> Result<(u64, bool, u32, u32), String> {
    let mut wire = ProgramCatalogIdentityWire {
        struct_size: 0,
        abi_version: 0,
        generation: 0,
        frozen: 0,
        module_count: 0,
        function_count: 0,
        reserved: 0,
        pack_set_sha256: [0; 32],
        candidate_catalog_sha256: [0; 32],
        eligible_candidate_set_sha256: [0; 32],
    };
    let bytes = u32::try_from(core::mem::size_of::<ProgramCatalogIdentityWire>())
        .map_err(|_| "Program catalog identity wire is too large")?;
    // SAFETY: writable POD output of the exact advertised size.
    let rc = unsafe { imparo_cuda_program_catalog_identity(&raw mut wire, bytes) };
    if rc != 0
        || wire.struct_size != bytes
        || wire.abi_version != CUDA_PROGRAM_ABI_V1
        || wire.reserved != 0
        || wire.pack_set_sha256 != [0; 32]
        || wire.candidate_catalog_sha256 != [0; 32]
        || wire.eligible_candidate_set_sha256 != [0; 32]
    {
        return Err(
            "native Program catalog returned an incompatible PR-E identity".into(),
        );
    }
    Ok((
        wire.generation,
        wire.frozen != 0,
        wire.module_count,
        wire.function_count,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pr_e_never_activates_reserved_pr_g_catalog_hashes() {
        let identity = pending_identity();
        assert_eq!(identity.program_pack_abi, PROGRAM_PACK_ABI);
        assert!(!identity.identity_ready);
        assert_eq!(identity.pack_set_sha256, [0; 32]);
        assert_eq!(identity.candidate_catalog_sha256, [0; 32]);
        assert_eq!(identity.eligible_candidate_set_sha256, [0; 32]);
    }
}
