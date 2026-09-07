//! Conversion from authenticated Rust objects to the synchronous native Program ABI.

use crate::ffi::{
    CUDA_PROGRAM_ABI_V1, PROGRAM_ARGUMENT_MANIFEST_U32, PROGRAM_ARGUMENT_SCALAR_F32,
    PROGRAM_ARGUMENT_SCALAR_I32, PROGRAM_ARGUMENT_SCALAR_U32,
    PROGRAM_ARGUMENT_SCALAR_U64, PROGRAM_ARGUMENT_SCRATCH_PTR,
    PROGRAM_ARGUMENT_STATE_PTR, PROGRAM_ARGUMENT_TENSOR_PTR,
    PROGRAM_GRAPH_CAPTURE_STATIC, PROGRAM_GRAPH_REPLAY_UPDATE,
    ProgramArgumentDescriptorWire, ProgramFunctionWire, ProgramModuleWire,
    ProgramPackWire, imparo_cuda_program_pack_install,
};
use imparo_program_pack::AdmittedPack;
use imparo_program_pack::manifest::{
    FeatureReference, GraphCapture, TypedReference, Variant, WireType,
};
use sha2::{Digest as _, Sha256};
use std::collections::{BTreeMap, BTreeSet};

use super::constraints::{
    AdapterAuthority, GraphMutability, argument_graph_metadata, decode_hex_32,
    validate_variant_authority,
};

pub(crate) struct EligibleVariant<'a> {
    pub variant: &'a Variant,
    pub authority: &'a AdapterAuthority,
}

pub(crate) fn choice_group_identity(value: &str) -> [u8; 32] {
    domain_identity(b"imparo-program-choice-group-v1", value)
}

fn module_identity(value: &str) -> [u8; 32] {
    domain_identity(b"imparo-program-module-v1", value)
}

fn domain_identity(domain: &[u8], value: &str) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(domain);
    hash.update([0]);
    hash.update((value.len() as u64).to_le_bytes());
    hash.update(value.as_bytes());
    hash.finalize().into()
}

pub(crate) fn eligible_variants<'a>(
    pack: &'a AdmittedPack,
    authorities: &'a [AdapterAuthority],
) -> Result<Vec<EligibleVariant<'a>>, String> {
    let manifest = pack.manifest();
    let groups = manifest
        .choice_groups()
        .iter()
        .map(|group| (group.choice_group_id.as_str(), group))
        .collect::<BTreeMap<_, _>>();
    let mut output = Vec::new();
    for variant in manifest.variants() {
        if pack
            .admission()
            .disabled_variants()
            .contains(&variant.variant_id)
        {
            continue;
        }
        let Some(group) = groups.get(variant.choice_group_id.as_str()) else {
            return Err("validated variant references a missing choice group".into());
        };
        let Some(authority) = authorities.iter().find(|authority| {
            authority.choice_group_id == variant.choice_group_id
                && authority.contract_id == variant.contract.id
                && authority.contract_revision == variant.contract.revision
        }) else {
            continue;
        };
        if validate_variant_authority(variant, &group.workload, authority).is_ok() {
            output.push(EligibleVariant { variant, authority });
        }
    }
    // Filtering a dependency must also filter every candidate that requires it. Iterate
    // to a fixed point so unknown contracts, disabled variants and invalid adapters can
    // never leave a dependent kernel schedulable.
    dependency_closure(&mut output);
    Ok(output)
}

fn dependency_closure(output: &mut Vec<EligibleVariant<'_>>) {
    loop {
        let variant_ids = output
            .iter()
            .map(|item| item.variant.variant_id.clone())
            .collect::<BTreeSet<_>>();
        let group_ids = output
            .iter()
            .map(|item| item.variant.choice_group_id.clone())
            .collect::<BTreeSet<_>>();
        let features = output
            .iter()
            .flat_map(|item| item.variant.provides.iter())
            .map(|provided| match provided {
                FeatureReference::Feature { id } => id.clone(),
            })
            .collect::<BTreeSet<_>>();
        let prior = output.len();
        output.retain(|item| {
            item.variant.requires.iter().all(|required| match required {
                TypedReference::Variant { id } => variant_ids.contains(id.as_str()),
                TypedReference::ChoiceGroup { id } => group_ids.contains(id.as_str()),
                TypedReference::Feature { id } => features.contains(id.as_str()),
            }) && item
                .variant
                .joint_with
                .iter()
                .all(|group| group_ids.contains(group.as_str()))
        });
        if output.len() == prior {
            break;
        }
    }
}

pub(crate) fn install_native(
    pack: &AdmittedPack,
    eligible: &[EligibleVariant<'_>],
) -> Result<(), String> {
    if eligible.is_empty() {
        return Ok(());
    }
    let used_modules = eligible
        .iter()
        .map(|item| item.variant.module_id.as_str())
        .collect::<BTreeSet<_>>();
    let admitted_modules = pack
        .modules()
        .iter()
        .map(|module| (module.id(), module))
        .collect::<BTreeMap<_, _>>();
    let mut module_wires = Vec::with_capacity(used_modules.len());
    for id in used_modules {
        let module = admitted_modules
            .get(id)
            .ok_or_else(|| format!("eligible variant module {id} is absent"))?;
        module_wires.push(ProgramModuleWire {
            struct_size: size_u32::<ProgramModuleWire>()?,
            abi_version: CUDA_PROGRAM_ABI_V1,
            module_id: module_identity(id),
            image: module.bytes().as_ptr(),
            image_bytes: u64::try_from(module.bytes().len())
                .map_err(|_| "module size does not fit the Program ABI")?,
        });
    }
    // Each nested vector owns the exact schema bytes until the synchronous native
    // install returns. Native ABI26 must copy them and must never retain these pointers.
    let mut argument_schemas = Vec::with_capacity(eligible.len());
    for item in eligible {
        let mut schema = Vec::with_capacity(item.authority.arguments.len());
        for argument in &item.authority.arguments {
            let (wire_type, mutability, update_source) =
                argument_graph_metadata(item.authority, argument)?;
            schema.push(ProgramArgumentDescriptorWire {
                kind: argument_kind(wire_type),
                graph_mutability: match mutability {
                    GraphMutability::CaptureStatic => PROGRAM_GRAPH_CAPTURE_STATIC,
                    GraphMutability::ReplayUpdate => PROGRAM_GRAPH_REPLAY_UPDATE,
                },
                update_source,
                reserved: 0,
            });
        }
        argument_schemas.push(schema);
    }
    let mut function_wires = Vec::with_capacity(eligible.len());
    for (item, argument_schema) in eligible.iter().zip(&argument_schemas) {
        let variant = item.variant;
        function_wires.push(ProgramFunctionWire {
            struct_size: size_u32::<ProgramFunctionWire>()?,
            abi_version: CUDA_PROGRAM_ABI_V1,
            choice_group_id: choice_group_identity(&variant.choice_group_id),
            variant_id: decode_hex_32(&variant.variant_id)?,
            module_id: module_identity(&variant.module_id),
            symbol: variant.symbol.as_ptr(),
            symbol_bytes: u32::try_from(variant.symbol.len())
                .map_err(|_| "Program symbol exceeds the wire length")?,
            argument_count: u32::try_from(variant.launch.arguments.len())
                .map_err(|_| "Program argument count exceeds the wire length")?,
            argument_schema: argument_schema.as_ptr(),
            block_x: variant.launch.block.x,
            block_y: variant.launch.block.y,
            block_z: variant.launch.block.z,
            dynamic_shared_bytes_max: variant.resources.dynamic_shared_bytes_max,
            registers_per_thread_max: u32::from(
                variant.resources.registers_per_thread_max,
            ),
            static_shared_bytes_max: variant.resources.static_shared_bytes_max,
            local_memory_bytes_max: variant.resources.local_memory_bytes_max,
            threads_per_block_max: variant.resources.threads_per_block_max,
            graph_capture: match variant.graph_capture {
                GraphCapture::Forbidden => 0,
                GraphCapture::CaptureOnly => 1,
                GraphCapture::ReplayUpdateSafe => 2,
            },
            builtin_variant_id: item.authority.builtin_variant_id,
        });
    }
    let manifest = pack.manifest();
    let wire = ProgramPackWire {
        struct_size: size_u32::<ProgramPackWire>()?,
        abi_version: CUDA_PROGRAM_ABI_V1,
        program_pack_abi: manifest.program_pack_abi(),
        kernel_contract_abi: 1,
        backend_abi_min: manifest.backend_abi().min,
        backend_abi_max_exclusive: manifest.backend_abi().max_exclusive,
        target_sm: u32::from(manifest.target().sm),
        driver_min: manifest.target().driver_min,
        modules: module_wires.as_ptr(),
        module_count: u32::try_from(module_wires.len())
            .map_err(|_| "Program module count exceeds the wire length")?,
        functions: function_wires.as_ptr(),
        function_count: u32::try_from(function_wires.len())
            .map_err(|_| "Program function count exceeds the wire length")?,
        identity_ready: 0,
        pack_set_sha256: [0; 32],
        candidate_catalog_sha256: [0; 32],
        eligible_candidate_set_sha256: [0; 32],
    };
    // SAFETY: all pointers reference owned immutable bytes/vectors kept alive until the
    // synchronous native call returns; every nested wire is repr(C) and size-versioned.
    let rc =
        unsafe { imparo_cuda_program_pack_install(&raw const wire, wire.struct_size) };
    if rc == 0 {
        Ok(())
    } else {
        Err(format!(
            "native Program Pack install rejected with code {rc}"
        ))
    }
}

fn argument_kind(value: WireType) -> u32 {
    match value {
        WireType::TensorPtr => PROGRAM_ARGUMENT_TENSOR_PTR,
        WireType::StatePtr => PROGRAM_ARGUMENT_STATE_PTR,
        WireType::ScratchPtr => PROGRAM_ARGUMENT_SCRATCH_PTR,
        WireType::ScalarI32 => PROGRAM_ARGUMENT_SCALAR_I32,
        WireType::ScalarU32 => PROGRAM_ARGUMENT_SCALAR_U32,
        WireType::ScalarU64 => PROGRAM_ARGUMENT_SCALAR_U64,
        WireType::ScalarF32 => PROGRAM_ARGUMENT_SCALAR_F32,
        WireType::ManifestU32 => PROGRAM_ARGUMENT_MANIFEST_U32,
    }
}

fn size_u32<T>() -> Result<u32, String> {
    u32::try_from(core::mem::size_of::<T>())
        .map_err(|_| "Program ABI type is too large".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use imparo_program_pack::manifest::TypedReference;

    #[test]
    fn unknown_or_filtered_dependency_removes_dependents_to_fixed_point() {
        let manifest = super::super::constraints::tests::manifest();
        let authority = super::super::constraints::tests::authority(&manifest);
        let mut leaf = manifest.variants()[0].clone();
        leaf.variant_id =
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".into();
        leaf.requires = vec![TypedReference::Variant {
            id: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                .into(),
        }];
        let mut root = manifest.variants()[0].clone();
        root.variant_id =
            "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc".into();
        root.requires = vec![TypedReference::Variant {
            id: leaf.variant_id.clone(),
        }];
        let variants = [leaf, root];
        let mut eligible = variants
            .iter()
            .map(|variant| EligibleVariant {
                variant,
                authority: &authority,
            })
            .collect();
        dependency_closure(&mut eligible);
        assert!(eligible.is_empty(), "dependency removal must propagate");
    }
}
