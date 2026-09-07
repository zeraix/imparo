//! Engine-owned eligibility checks. Signed manifest data can only narrow a descriptor
//! compiled into Imparo; it can never construct launch authority.

use imparo_backend::{DeviceProfile, ModelFacts};
use imparo_program_pack::manifest::{
    Access, AliasMode, Determinism, Dtype, GraphCapture, LaunchArgument, Layout,
    Manifest, MathMode, NumericalClass, Quantization, Variant, WireType, Workload,
};
use std::collections::{BTreeMap, BTreeSet};

use crate::ffi::{
    PROGRAM_UPDATE_DECODE_START_POS_U32 as UPDATE_DECODE_START_POS_U32,
    PROGRAM_UPDATE_NONE as UPDATE_NONE,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RuntimeEligibility {
    pub device_sm: u32,
    pub driver_version: u32,
    pub backend_abi: u32,
    pub math_mode: MathMode,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)] // Closed registry variants land with their tracked adapters.
pub(crate) enum SlotRole {
    Input,
    Output,
    State,
    Scratch,
    Shape,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)] // Closed registry variants land with their tracked adapters.
pub(crate) enum SlotLifetime {
    Invocation,
    Graph,
    Conversation,
    Model,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GraphMutability {
    CaptureStatic,
    ReplayUpdate,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)] // Closed registry variants land with their tracked adapters.
pub(crate) enum StateInitialization {
    None,
    Zero,
    EngineInitialized,
    ContractDefined,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)] // Closed registry descriptors land with tracked adapters.
pub(crate) enum DescriptorAuthority {
    EngineAdapter,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ShapeAuthority {
    pub min: u64,
    pub max: u64,
    pub multiple_of: u64,
    pub one_of: Option<Vec<u64>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NumericalClassKind {
    BitExact,
    GateBounded,
    DiagnosticOnly,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct WorkloadAuthority {
    pub workload_id: imparo_program_pack::manifest::WorkloadId,
    pub revision: u32,
    pub fixture_sha256: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SlotAuthority {
    pub slot: String,
    pub abi_index: u32,
    pub wire_type: WireType,
    pub role: SlotRole,
    pub access: Access,
    pub lifetime: SlotLifetime,
    pub graph_mutability: GraphMutability,
    /// Adapter-owned stable source ID used by the native graph updater. This is never
    /// inferred from ABI order: capture-static is zero, replay-update is nonzero.
    pub update_source: u32,
    pub rank: u8,
    pub extent_authority: DescriptorAuthority,
    pub stride_authority: DescriptorAuthority,
    pub shapes: Vec<ShapeAuthority>,
    pub strides: Vec<ShapeAuthority>,
    pub alignment: u32,
    pub state_initialization: StateInitialization,
    pub allowed_dtypes: Vec<Dtype>,
    pub allowed_quantizations: Vec<Quantization>,
    pub allowed_layouts: Vec<Layout>,
    pub aliases: BTreeMap<String, AliasMode>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[allow(dead_code)] // Closed registry arguments land with tracked adapters.
pub(crate) enum ArgumentAuthority {
    Slot { slot: String, wire_type: WireType },
    ManifestU32 { name: String, min: u32, max: u32 },
}

#[derive(Clone)]
pub(crate) struct AdapterAuthority {
    pub choice_group_id: String,
    pub contract_id: String,
    pub contract_revision: u32,
    pub contract_sha256: [u8; 32],
    pub adapter_id: String,
    pub adapter_revision: u32,
    pub adapter_sha256: [u8; 32],
    pub state_reset_revision: u32,
    pub workloads: Vec<WorkloadAuthority>,
    pub slots: Vec<SlotAuthority>,
    pub arguments: Vec<ArgumentAuthority>,
    pub scratch_max_bytes: u64,
    pub scratch_alignment: u32,
    pub allow_zero_initialized_scratch: bool,
    pub registers_per_thread_max: u16,
    pub static_shared_bytes_max: u32,
    pub dynamic_shared_bytes_max: u32,
    pub local_memory_bytes_max: u64,
    pub threads_per_block_max: u32,
    pub allowed_numerical_classes: Vec<NumericalClassKind>,
    pub required_determinism: Option<Determinism>,
    pub allowed_graph_capture: Vec<GraphCapture>,
    pub builtin_variant_id: [u8; 32],
    pub builtin_config_id: [u8; 32],
    /// Compiled model/device applicability. Unknown or incomplete facts return false.
    pub applies: fn(&Variant, &ModelFacts, &DeviceProfile) -> bool,
}

/// The production registry is intentionally closed. Step 4 adds tracked descriptors
/// here together with their concrete CUDA workflow adapters. An empty registry means
/// every external candidate fails closed while the built-in path remains unchanged.
pub(crate) fn compiled_adapters() -> &'static [AdapterAuthority] {
    &[]
}

pub fn validate_pack_runtime(
    manifest: &Manifest,
    runtime: &RuntimeEligibility,
) -> Result<(), String> {
    if manifest.program_pack_abi() != imparo_program_pack::PROGRAM_PACK_ABI {
        return Err("unsupported Program Pack ABI".into());
    }
    if manifest.backend() != "cuda" {
        return Err("Program Pack backend is not CUDA".into());
    }
    let abi = manifest.backend_abi();
    if runtime.backend_abi < abi.min || runtime.backend_abi >= abi.max_exclusive {
        return Err("Program Pack does not admit this native CUDA ABI".into());
    }
    let target = manifest.target();
    if u32::from(target.sm) != runtime.device_sm {
        return Err("Program Pack target is not an exact SM match".into());
    }
    if target.warp_size != 32 {
        return Err("Program Pack target has an unsupported warp size".into());
    }
    if target.driver_min > runtime.driver_version {
        return Err("Program Pack requires a newer CUDA driver".into());
    }
    if target.math_mode != runtime.math_mode {
        return Err("Program Pack math mode differs from the native backend".into());
    }
    Ok(())
}

pub(crate) fn validate_variant_authority(
    variant: &Variant,
    workload: &Workload,
    authority: &AdapterAuthority,
) -> Result<(), String> {
    validate_adapter_descriptor(authority)?;
    let contract_digest = decode_hex_32(&variant.contract.sha256)?;
    if variant.choice_group_id != authority.choice_group_id
        || variant.contract.id != authority.contract_id
        || variant.contract.revision != authority.contract_revision
        || contract_digest != authority.contract_sha256
    {
        return Err("variant does not match a compiled contract adapter".into());
    }
    let fixture = decode_hex_32(&workload.fixture_sha256)?;
    if !authority.workloads.iter().any(|allowed| {
        allowed.workload_id == workload.workload_id
            && allowed.revision == workload.revision
            && allowed.fixture_sha256 == fixture
    }) {
        return Err("workload fixture is not owned by the compiled adapter".into());
    }

    if variant.launch.arguments.len() != authority.arguments.len() {
        return Err("variant argument list differs from the adapter ABI".into());
    }
    for (actual, expected) in variant.launch.arguments.iter().zip(&authority.arguments)
    {
        match (actual, expected) {
            (
                LaunchArgument::Slot { slot, wire_type },
                ArgumentAuthority::Slot {
                    slot: expected_slot,
                    wire_type: expected_wire,
                },
            ) if slot == expected_slot && wire_type == expected_wire => {}
            (
                LaunchArgument::ManifestU32 { name, value },
                ArgumentAuthority::ManifestU32 {
                    name: expected_name,
                    min,
                    max,
                },
            ) if name == expected_name && min <= value && value <= max => {}
            _ => {
                return Err(
                    "variant argument order/type/constants differ from adapter ABI"
                        .into(),
                );
            }
        }
    }

    let slots = authority
        .slots
        .iter()
        .map(|slot| (slot.slot.as_str(), slot))
        .collect::<BTreeMap<_, _>>();
    for shape in &variant.constraints.shapes {
        let slot = slots
            .get(shape.slot.as_str())
            .ok_or("shape constraint references a non-contract slot")?;
        if shape.axis >= slot.rank {
            return Err("shape constraint axis exceeds adapter-owned rank".into());
        }
        let allowed = &slot.shapes[usize::from(shape.axis)];
        if shape.min < allowed.min
            || shape.max > allowed.max
            || shape.multiple_of % allowed.multiple_of != 0
            || !shape_domain_is_subset(shape, allowed)
        {
            return Err(
                "shape constraint widens the adapter-owned extent domain".into()
            );
        }
    }
    for constraint in &variant.constraints.dtypes {
        let slot = slots
            .get(constraint.slot.as_str())
            .ok_or("dtype constraint references a non-contract slot")?;
        if constraint
            .allowed
            .iter()
            .any(|value| !slot.allowed_dtypes.contains(value))
        {
            return Err("dtype constraint widens the adapter contract".into());
        }
    }
    for constraint in &variant.constraints.quantizations {
        let slot = slots
            .get(constraint.slot.as_str())
            .ok_or("quantization constraint references a non-contract slot")?;
        if constraint
            .allowed
            .iter()
            .any(|value| !slot.allowed_quantizations.contains(value))
        {
            return Err("quantization constraint widens the adapter contract".into());
        }
    }
    for constraint in &variant.constraints.layouts {
        let slot = slots
            .get(constraint.slot.as_str())
            .ok_or("layout constraint references a non-contract slot")?;
        if constraint
            .allowed
            .iter()
            .any(|value| !slot.allowed_layouts.contains(value))
        {
            return Err("layout constraint widens the adapter contract".into());
        }
    }
    for alignment in &variant.constraints.alignments {
        let slot = slots
            .get(alignment.slot.as_str())
            .ok_or("alignment constraint references a non-contract slot")?;
        if !is_pointer(slot.wire_type)
            || alignment.bytes < slot.alignment
            || alignment.bytes % slot.alignment != 0
        {
            return Err("alignment constraint weakens the adapter contract".into());
        }
    }

    let mut effects = BTreeMap::new();
    for effect in &variant.effects {
        let aliases = effect
            .aliasing
            .iter()
            .map(|alias| (alias.target_slot.clone(), alias.mode))
            .collect::<BTreeMap<_, _>>();
        effects.insert(effect.slot.as_str(), (effect.access, aliases));
    }
    if authority.slots.iter().any(|slot| {
        effects.get(slot.slot.as_str()) != Some(&(slot.access, slot.aliases.clone()))
    }) || effects.len() != authority.slots.len()
    {
        return Err(
            "variant effects or aliases differ from the adapter contract".into(),
        );
    }

    let scratch_slots = authority
        .slots
        .iter()
        .filter(|slot| slot.wire_type == WireType::ScratchPtr)
        .count();
    if variant.scratch.max_bytes > authority.scratch_max_bytes
        || (variant.scratch.max_bytes == 0 && scratch_slots != 0)
        || (variant.scratch.max_bytes > 0
            && (scratch_slots != 1
                || variant.scratch.alignment < authority.scratch_alignment
                || !variant.scratch.alignment.is_power_of_two()))
        || (variant.scratch.zero_initialized
            && !authority.allow_zero_initialized_scratch)
    {
        return Err("variant scratch requirements exceed adapter authority".into());
    }
    let resources = &variant.resources;
    let block_threads = u64::from(variant.launch.block.x)
        * u64::from(variant.launch.block.y)
        * u64::from(variant.launch.block.z);
    if resources.registers_per_thread_max > authority.registers_per_thread_max
        || resources.static_shared_bytes_max > authority.static_shared_bytes_max
        || resources.dynamic_shared_bytes_max > authority.dynamic_shared_bytes_max
        || resources.local_memory_bytes_max > authority.local_memory_bytes_max
        || resources.threads_per_block_max > authority.threads_per_block_max
        || block_threads > u64::from(authority.threads_per_block_max)
    {
        return Err("variant resource ceilings exceed adapter authority".into());
    }
    let numerical = numerical_kind(&variant.numerical_class);
    if !authority.allowed_numerical_classes.contains(&numerical)
        || authority
            .required_determinism
            .map(|required| required != variant.determinism)
            .unwrap_or(false)
    {
        return Err(
            "variant numerical/determinism class is not adapter-authorized".into(),
        );
    }
    let expected_update_slots = authority
        .slots
        .iter()
        .filter(|slot| slot.graph_mutability == GraphMutability::ReplayUpdate)
        .map(|slot| slot.slot.clone())
        .collect::<BTreeSet<_>>();
    if !authority
        .allowed_graph_capture
        .contains(&variant.graph_capture)
        || variant
            .graph_update_slots
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>()
            != expected_update_slots
    {
        return Err("variant Graph policy differs from the adapter contract".into());
    }
    Ok(())
}

fn validate_adapter_descriptor(authority: &AdapterAuthority) -> Result<(), String> {
    if authority.choice_group_id.trim().is_empty()
        || authority.contract_id.trim().is_empty()
        || authority.contract_revision == 0
        || authority.contract_sha256 == [0; 32]
        || authority.adapter_id.trim().is_empty()
        || authority.adapter_revision == 0
        || authority.adapter_sha256 == [0; 32]
        || authority.state_reset_revision != authority.adapter_revision
        || authority.builtin_variant_id == [0; 32]
        || authority.builtin_config_id == [0; 32]
        || authority.allowed_numerical_classes.is_empty()
        || authority.allowed_graph_capture.is_empty()
        || authority.arguments.is_empty()
    {
        return Err("compiled adapter descriptor is incomplete".into());
    }
    let mut slot_names = BTreeSet::new();
    for (index, slot) in authority.slots.iter().enumerate() {
        if slot.abi_index != index as u32
            || slot.slot.trim().is_empty()
            || !slot_names.insert(slot.slot.as_str())
            || slot.alignment == 0
            || !slot.alignment.is_power_of_two()
            || slot.wire_type == WireType::ManifestU32
        {
            return Err("compiled adapter slot matrix is invalid".into());
        }
        let role_matrix_valid = match slot.wire_type {
            WireType::TensorPtr => match slot.role {
                SlotRole::Input => {
                    slot.access == Access::Read
                        && matches!(
                            slot.lifetime,
                            SlotLifetime::Invocation
                                | SlotLifetime::Graph
                                | SlotLifetime::Model
                        )
                }
                SlotRole::Output => {
                    matches!(slot.access, Access::Write | Access::ReadWrite)
                        && matches!(
                            slot.lifetime,
                            SlotLifetime::Invocation | SlotLifetime::Graph
                        )
                }
                _ => false,
            },
            WireType::StatePtr => {
                slot.role == SlotRole::State
                    && slot.access == Access::ReadWrite
                    && slot.lifetime == SlotLifetime::Conversation
                    && slot.state_initialization != StateInitialization::None
            }
            WireType::ScratchPtr => {
                slot.role == SlotRole::Scratch
                    && slot.access == Access::ReadWrite
                    && slot.lifetime == SlotLifetime::Graph
                    && slot.graph_mutability == GraphMutability::CaptureStatic
            }
            WireType::ScalarI32
            | WireType::ScalarU32
            | WireType::ScalarU64
            | WireType::ScalarF32 => {
                matches!(slot.role, SlotRole::Input | SlotRole::Shape)
                    && slot.access == Access::Read
                    && matches!(
                        slot.lifetime,
                        SlotLifetime::Invocation | SlotLifetime::Graph
                    )
                    && slot.state_initialization == StateInitialization::None
            }
            WireType::ManifestU32 => false,
        };
        if !role_matrix_valid
            || (slot.lifetime == SlotLifetime::Model
                && slot.graph_mutability != GraphMutability::CaptureStatic)
        {
            return Err(
                "compiled adapter role/access/lifetime/state matrix is invalid".into(),
            );
        }
        if (slot.graph_mutability == GraphMutability::CaptureStatic
            && slot.update_source != UPDATE_NONE)
            || (slot.graph_mutability == GraphMutability::ReplayUpdate
                && (slot.update_source != UPDATE_DECODE_START_POS_U32
                    || slot.wire_type != WireType::ScalarU32))
        {
            return Err("compiled adapter Graph update source is invalid".into());
        }
        let tensor_or_state =
            matches!(slot.wire_type, WireType::TensorPtr | WireType::StatePtr);
        if tensor_or_state
            && (slot.rank == 0
                || slot.shapes.len() != usize::from(slot.rank)
                || slot.strides.len() != usize::from(slot.rank)
                || slot.allowed_dtypes.is_empty()
                || slot.allowed_quantizations.is_empty()
                || slot.allowed_layouts.is_empty())
        {
            return Err("compiled pointer slot has no tensor domain".into());
        }
        if !tensor_or_state
            && (slot.rank != 0
                || !slot.shapes.is_empty()
                || !slot.strides.is_empty()
                || !slot.allowed_dtypes.is_empty()
                || !slot.allowed_quantizations.is_empty()
                || !slot.allowed_layouts.is_empty())
        {
            return Err("compiled scalar slot has a tensor domain".into());
        }
        for shape in slot.shapes.iter().chain(&slot.strides) {
            if shape.min == 0
                || shape.min > shape.max
                || shape.multiple_of == 0
                || shape.min % shape.multiple_of != 0
                || shape
                    .one_of
                    .as_ref()
                    .map(|values| {
                        values.is_empty()
                            || values.iter().any(|value| {
                                *value < shape.min
                                    || *value > shape.max
                                    || *value % shape.multiple_of != 0
                            })
                    })
                    .unwrap_or(false)
            {
                return Err("compiled adapter extent domain is invalid".into());
            }
        }
    }
    let slots_by_name = authority
        .slots
        .iter()
        .map(|slot| (slot.slot.as_str(), slot))
        .collect::<BTreeMap<_, _>>();
    for slot in &authority.slots {
        for (target, mode) in &slot.aliases {
            let Some(other) = slots_by_name.get(target.as_str()) else {
                return Err("compiled adapter alias targets an unknown slot".into());
            };
            if target == &slot.slot
                || matches!(slot.role, SlotRole::Scratch)
                || matches!(other.role, SlotRole::Scratch)
                || other.aliases.get(&slot.slot) != Some(mode)
            {
                return Err(
                    "compiled adapter alias relation is not exact and symmetric".into(),
                );
            }
        }
    }
    let scratch_slots = authority
        .slots
        .iter()
        .filter(|slot| slot.role == SlotRole::Scratch)
        .count();
    if (authority.scratch_max_bytes == 0 && scratch_slots != 0)
        || (authority.scratch_max_bytes > 0
            && (scratch_slots != 1
                || authority.scratch_alignment == 0
                || !authority.scratch_alignment.is_power_of_two()))
    {
        return Err("compiled adapter scratch ownership is inconsistent".into());
    }
    if authority.arguments.len() < authority.slots.len() {
        return Err("compiled adapter omits required slot arguments".into());
    }
    for (index, slot) in authority.slots.iter().enumerate() {
        match &authority.arguments[index] {
            ArgumentAuthority::Slot {
                slot: name,
                wire_type,
            } if name == &slot.slot && wire_type == &slot.wire_type => {}
            _ => {
                return Err(
                    "compiled adapter slot arguments are not dense and ordered".into(),
                );
            }
        }
    }
    let mut constant_names = BTreeSet::new();
    for argument in authority.arguments.iter().skip(authority.slots.len()) {
        match argument {
            ArgumentAuthority::ManifestU32 { name, min, max }
                if !name.trim().is_empty()
                    && min <= max
                    && constant_names.insert(name) => {}
            _ => return Err("compiled adapter manifest constants are invalid".into()),
        }
    }
    Ok(())
}

pub(crate) fn argument_graph_metadata(
    authority: &AdapterAuthority,
    argument: &ArgumentAuthority,
) -> Result<(WireType, GraphMutability, u32), String> {
    match argument {
        ArgumentAuthority::Slot { slot, wire_type } => {
            let descriptor = authority
                .slots
                .iter()
                .find(|candidate| candidate.slot == *slot)
                .ok_or("compiled argument references an unknown slot")?;
            if descriptor.wire_type != *wire_type {
                return Err("compiled argument wire differs from its slot".into());
            }
            Ok((
                *wire_type,
                descriptor.graph_mutability,
                descriptor.update_source,
            ))
        }
        ArgumentAuthority::ManifestU32 { .. } => {
            Ok((WireType::ManifestU32, GraphMutability::CaptureStatic, 0))
        }
    }
}

pub(crate) fn is_pointer(value: WireType) -> bool {
    matches!(
        value,
        WireType::TensorPtr | WireType::StatePtr | WireType::ScratchPtr
    )
}

fn numerical_kind(value: &NumericalClass) -> NumericalClassKind {
    match value {
        NumericalClass::BitExact { .. } => NumericalClassKind::BitExact,
        NumericalClass::GateBounded { .. } => NumericalClassKind::GateBounded,
        NumericalClass::DiagnosticOnly => NumericalClassKind::DiagnosticOnly,
    }
}

fn shape_domain_is_subset(
    candidate: &imparo_program_pack::manifest::ShapeConstraint,
    allowed: &ShapeAuthority,
) -> bool {
    match (&candidate.one_of, &allowed.one_of) {
        (Some(values), Some(allowed_values)) => {
            values.iter().all(|value| allowed_values.contains(value))
        }
        (Some(values), None) => values.iter().all(|value| {
            *value >= allowed.min
                && *value <= allowed.max
                && *value % allowed.multiple_of == 0
        }),
        (None, Some(_)) => false,
        (None, None) => true,
    }
}

pub fn decode_hex_32(value: &str) -> Result<[u8; 32], String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err("expected a lowercase 32-byte hexadecimal identity".into());
    }
    let mut output = [0_u8; 32];
    for (index, byte) in output.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)
            .map_err(|error| error.to_string())?;
    }
    Ok(output)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use imparo_program_pack::manifest::ShapeConstraint;

    pub(crate) fn manifest() -> Manifest {
        Manifest::parse(include_bytes!(
            "../../../../schemas/examples/program-pack-v1.minimal.json"
        ))
        .expect("repository Program Pack example")
    }

    pub(crate) fn runtime(manifest: &Manifest) -> RuntimeEligibility {
        RuntimeEligibility {
            device_sm: u32::from(manifest.target().sm),
            driver_version: manifest.target().driver_min,
            backend_abi: manifest.backend_abi().min,
            math_mode: manifest.target().math_mode,
        }
    }

    fn always(_: &Variant, _: &ModelFacts, _: &DeviceProfile) -> bool {
        true
    }

    pub(crate) fn authority(manifest: &Manifest) -> AdapterAuthority {
        let variant = &manifest.variants()[0];
        let group = &manifest.choice_groups()[0];
        let slots = variant
            .launch
            .arguments
            .iter()
            .enumerate()
            .filter_map(|(index, argument)| match argument {
                LaunchArgument::Slot { slot, wire_type } => {
                    let effect =
                        variant.effects.iter().find(|effect| effect.slot == *slot)?;
                    Some(SlotAuthority {
                        slot: slot.clone(),
                        abi_index: index as u32,
                        wire_type: *wire_type,
                        role: SlotRole::Input,
                        access: effect.access,
                        lifetime: SlotLifetime::Invocation,
                        graph_mutability: if variant.graph_update_slots.contains(slot) {
                            GraphMutability::ReplayUpdate
                        } else {
                            GraphMutability::CaptureStatic
                        },
                        update_source: if variant.graph_update_slots.contains(slot) {
                            UPDATE_DECODE_START_POS_U32
                        } else {
                            0
                        },
                        rank: if is_pointer(*wire_type) { 1 } else { 0 },
                        extent_authority: DescriptorAuthority::EngineAdapter,
                        stride_authority: DescriptorAuthority::EngineAdapter,
                        shapes: if matches!(
                            wire_type,
                            WireType::TensorPtr | WireType::StatePtr
                        ) {
                            vec![ShapeAuthority {
                                min: 1,
                                max: u32::MAX.into(),
                                multiple_of: 1,
                                one_of: None,
                            }]
                        } else {
                            vec![]
                        },
                        strides: if matches!(
                            wire_type,
                            WireType::TensorPtr | WireType::StatePtr
                        ) {
                            vec![ShapeAuthority {
                                min: 1,
                                max: u32::MAX.into(),
                                multiple_of: 1,
                                one_of: None,
                            }]
                        } else {
                            vec![]
                        },
                        alignment: variant
                            .constraints
                            .alignments
                            .iter()
                            .find(|value| value.slot == *slot)
                            .map(|value| value.bytes)
                            .unwrap_or(1),
                        state_initialization: StateInitialization::None,
                        allowed_dtypes: if is_pointer(*wire_type) {
                            vec![Dtype::F16]
                        } else {
                            vec![]
                        },
                        allowed_quantizations: if is_pointer(*wire_type) {
                            vec![Quantization::None]
                        } else {
                            vec![]
                        },
                        allowed_layouts: if is_pointer(*wire_type) {
                            vec![Layout::Contiguous]
                        } else {
                            vec![]
                        },
                        aliases: effect
                            .aliasing
                            .iter()
                            .map(|alias| (alias.target_slot.clone(), alias.mode))
                            .collect(),
                    })
                }
                LaunchArgument::ManifestU32 { .. } => None,
            })
            .collect::<Vec<_>>();
        let arguments = variant
            .launch
            .arguments
            .iter()
            .map(|argument| match argument {
                LaunchArgument::Slot { slot, wire_type } => ArgumentAuthority::Slot {
                    slot: slot.clone(),
                    wire_type: *wire_type,
                },
                LaunchArgument::ManifestU32 { name, value } => {
                    ArgumentAuthority::ManifestU32 {
                        name: name.clone(),
                        min: *value,
                        max: *value,
                    }
                }
            })
            .collect();
        AdapterAuthority {
            choice_group_id: group.choice_group_id.clone(),
            contract_id: variant.contract.id.clone(),
            contract_revision: variant.contract.revision,
            contract_sha256: decode_hex_32(&variant.contract.sha256).unwrap(),
            adapter_id: "imparo.test.adapter".into(),
            adapter_revision: 1,
            adapter_sha256: [7; 32],
            state_reset_revision: 1,
            workloads: vec![WorkloadAuthority {
                workload_id: group.workload.workload_id,
                revision: group.workload.revision,
                fixture_sha256: decode_hex_32(&group.workload.fixture_sha256).unwrap(),
            }],
            slots,
            arguments,
            scratch_max_bytes: variant.scratch.max_bytes,
            scratch_alignment: variant.scratch.alignment,
            allow_zero_initialized_scratch: variant.scratch.zero_initialized,
            registers_per_thread_max: variant.resources.registers_per_thread_max,
            static_shared_bytes_max: variant.resources.static_shared_bytes_max,
            dynamic_shared_bytes_max: variant.resources.dynamic_shared_bytes_max,
            local_memory_bytes_max: variant.resources.local_memory_bytes_max,
            threads_per_block_max: variant.resources.threads_per_block_max,
            allowed_numerical_classes: vec![numerical_kind(&variant.numerical_class)],
            required_determinism: Some(variant.determinism),
            allowed_graph_capture: vec![variant.graph_capture],
            builtin_variant_id: [8; 32],
            builtin_config_id: [9; 32],
            applies: always,
        }
    }

    #[test]
    fn runtime_gate_requires_exact_sm_driver_abi_and_math_mode() {
        let manifest = manifest();
        let valid = runtime(&manifest);
        validate_pack_runtime(&manifest, &valid).unwrap();
        for invalid in [
            RuntimeEligibility {
                device_sm: valid.device_sm + 1,
                ..valid.clone()
            },
            RuntimeEligibility {
                driver_version: valid.driver_version - 1,
                ..valid.clone()
            },
            RuntimeEligibility {
                backend_abi: manifest.backend_abi().max_exclusive,
                ..valid.clone()
            },
            RuntimeEligibility {
                math_mode: match valid.math_mode {
                    MathMode::Fast => MathMode::Strict,
                    MathMode::Strict => MathMode::Fast,
                },
                ..valid
            },
        ] {
            assert!(validate_pack_runtime(&manifest, &invalid).is_err());
        }
    }

    #[test]
    fn adapter_gate_rejects_forged_matrix_constants_and_domains() {
        let manifest = manifest();
        let variant = &manifest.variants()[0];
        let workload = &manifest.choice_groups()[0].workload;
        let valid = authority(&manifest);
        validate_variant_authority(variant, workload, &valid).unwrap();

        let mut bad = valid.clone();
        bad.adapter_sha256 = [0; 32];
        assert!(validate_variant_authority(variant, workload, &bad).is_err());
        bad = valid.clone();
        bad.state_reset_revision += 1;
        assert!(validate_variant_authority(variant, workload, &bad).is_err());
        bad = valid.clone();
        bad.arguments[0] = ArgumentAuthority::ManifestU32 {
            name: "forged".into(),
            min: 0,
            max: u32::MAX,
        };
        assert!(validate_variant_authority(variant, workload, &bad).is_err());
        bad = valid.clone();
        bad.slots[0].rank = 0;
        assert!(validate_variant_authority(variant, workload, &bad).is_err());
        bad = valid.clone();
        bad.slots[0].role = SlotRole::Scratch;
        assert!(validate_variant_authority(variant, workload, &bad).is_err());
        bad = valid.clone();
        bad.slots[0].graph_mutability = GraphMutability::ReplayUpdate;
        bad.slots[0].update_source = 99;
        assert!(validate_variant_authority(variant, workload, &bad).is_err());
        bad = valid.clone();
        bad.workloads[0].fixture_sha256[0] ^= 1;
        assert!(validate_variant_authority(variant, workload, &bad).is_err());
    }

    #[test]
    fn production_compiled_registry_is_closed_and_unknown_contracts_fail_closed() {
        assert!(compiled_adapters().is_empty());
        assert!(decode_hex_32(&"A".repeat(64)).is_err());
    }

    #[test]
    fn shape_constraints_must_narrow_every_adapter_owned_axis_dimension() {
        let manifest = manifest();
        let workload = &manifest.choice_groups()[0].workload;
        let mut authority = authority(&manifest);
        authority.slots[0].shapes[0] = ShapeAuthority {
            min: 16,
            max: 128,
            multiple_of: 8,
            one_of: None,
        };
        let mut variant = manifest.variants()[0].clone();
        variant.constraints.shapes = vec![ShapeConstraint {
            slot: authority.slots[0].slot.clone(),
            axis: 0,
            min: 16,
            max: 128,
            multiple_of: 8,
            one_of: None,
        }];
        validate_variant_authority(&variant, workload, &authority).unwrap();

        for bad_shape in [
            ShapeConstraint {
                min: 8,
                ..variant.constraints.shapes[0].clone()
            },
            ShapeConstraint {
                max: 256,
                ..variant.constraints.shapes[0].clone()
            },
            ShapeConstraint {
                multiple_of: 4,
                ..variant.constraints.shapes[0].clone()
            },
        ] {
            let mut bad = variant.clone();
            bad.constraints.shapes = vec![bad_shape];
            assert!(validate_variant_authority(&bad, workload, &authority).is_err());
        }

        authority.slots[0].shapes[0].one_of = Some(vec![16, 32]);
        assert!(validate_variant_authority(&variant, workload, &authority).is_err());
        variant.constraints.shapes[0].one_of = Some(vec![16, 48]);
        assert!(validate_variant_authority(&variant, workload, &authority).is_err());
    }
}
