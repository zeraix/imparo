//! Checked evaluation of the manifest's bounded launch-expression language.

use crate::ffi::{
    CUDA_PROGRAM_ABI_V1, PROGRAM_ARGUMENT_MANIFEST_U32, PROGRAM_ARGUMENT_SCALAR_F32,
    PROGRAM_ARGUMENT_SCALAR_I32, PROGRAM_ARGUMENT_SCALAR_U32,
    PROGRAM_ARGUMENT_SCALAR_U64, PROGRAM_ARGUMENT_SCRATCH_PTR,
    PROGRAM_ARGUMENT_STATE_PTR, PROGRAM_ARGUMENT_TENSOR_PTR, ProgramArgumentWire,
    ProgramLaunchWire, imparo_cuda_program_launch,
};
use imparo_program_pack::manifest::{
    Dtype, IntegerExpression, IntegerOp, LaunchArgument, Layout, Quantization, Variant,
    WireType,
};
use std::collections::BTreeMap;

use super::constraints::{AdapterAuthority, SlotAuthority};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LaunchLimits {
    pub grid_x: u64,
    pub grid_y: u64,
    pub grid_z: u64,
    pub dynamic_shared_bytes: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ProgramTensorValue {
    pub device_ptr: u64,
    pub extents: Vec<u64>,
    pub strides: Vec<u64>,
    pub dtype: Dtype,
    pub quantization: Quantization,
    pub layout: Layout,
}

#[derive(Clone, Debug, PartialEq)]
#[allow(dead_code)] // Constructed only by compiled adapters and the gated SM86 smoke.
pub(crate) enum ProgramValue {
    Tensor(ProgramTensorValue),
    State(ProgramTensorValue),
    Scratch { device_ptr: u64, bytes: u64 },
    ScalarI32(i32),
    ScalarU32(u32),
    ScalarU64(u64),
    ScalarF32(f32),
}

/// Launch the bound external variant. `Ok(false)` selects the existing built-in
/// fallback, so callers continue through the unchanged native operation.
///
/// # Safety
/// Pointer values must name live engine-owned CUDA allocations matching the compiled
/// adapter contract and remain valid until the backend stream completes.
#[allow(dead_code)] // Called only by compiled adapters and the gated SM86 smoke today.
pub(crate) unsafe fn launch_bound(
    group: &str,
    values: &BTreeMap<String, ProgramValue>,
) -> Result<bool, String> {
    let Some(selected) = super::registry::selected_variant(group)? else {
        return Ok(false);
    };
    let variant = &selected.manifest;
    validate_invocation(variant, &selected.authority, values)?;
    if variant.scratch.max_bytes != 0 {
        return Err("Program scratch allocation is not activated in PR-E".into());
    }
    let mut expression_slots = BTreeMap::new();
    let mut arguments = Vec::with_capacity(variant.launch.arguments.len());
    for argument in &variant.launch.arguments {
        let (kind, value) = match argument {
            LaunchArgument::ManifestU32 { value, .. } => {
                (PROGRAM_ARGUMENT_MANIFEST_U32, u64::from(*value))
            }
            LaunchArgument::Slot { slot, wire_type } => {
                let supplied = values
                    .get(slot)
                    .ok_or_else(|| format!("missing Program argument {slot}"))?;
                let (kind, bits, unsigned) = encode_value(*wire_type, supplied)?;
                if let Some(unsigned) = unsigned {
                    expression_slots.insert(slot.clone(), unsigned);
                }
                (kind, bits)
            }
        };
        arguments.push(ProgramArgumentWire {
            kind,
            reserved: 0,
            value,
        });
    }
    for alignment in &variant.constraints.alignments {
        let value = values.get(&alignment.slot).ok_or_else(|| {
            format!("missing aligned Program slot {}", alignment.slot)
        })?;
        let address = match value {
            ProgramValue::Tensor(value) | ProgramValue::State(value) => {
                value.device_ptr
            }
            ProgramValue::Scratch { device_ptr, .. } => *device_ptr,
            _ => return Err("alignment constraint targets a non-pointer slot".into()),
        };
        if address == 0 || address % u64::from(alignment.bytes) != 0 {
            return Err(format!("Program slot {} is misaligned", alignment.slot));
        }
    }
    let (grid, dynamic_shared_bytes) = evaluate_launch(
        &variant.launch,
        &expression_slots,
        LaunchLimits {
            grid_x: 2_147_483_647,
            grid_y: 65_535,
            grid_z: 65_535,
            dynamic_shared_bytes: u64::from(variant.resources.dynamic_shared_bytes_max),
        },
    )?;
    let wire = ProgramLaunchWire {
        struct_size: u32::try_from(core::mem::size_of::<ProgramLaunchWire>())
            .map_err(|_| "Program launch wire is too large")?,
        abi_version: CUDA_PROGRAM_ABI_V1,
        variant_id: super::constraints::decode_hex_32(&variant.variant_id)?,
        grid_x: grid[0],
        grid_y: grid[1],
        grid_z: grid[2],
        dynamic_shared_bytes,
        arguments: arguments.as_ptr(),
        argument_count: u32::try_from(arguments.len())
            .map_err(|_| "Program argument count does not fit u32")?,
        reserved: 0,
    };
    // SAFETY: the caller upholds pointer lifetime; local POD storage remains alive
    // through the synchronous enqueue call.
    let rc = unsafe { imparo_cuda_program_launch(&raw const wire, wire.struct_size) };
    if rc == 0 {
        Ok(true)
    } else {
        Err(format!("native Program launch failed with code {rc}"))
    }
}

#[allow(dead_code)]
fn encode_value(
    wire_type: WireType,
    value: &ProgramValue,
) -> Result<(u32, u64, Option<u64>), String> {
    let result = match (wire_type, value) {
        (WireType::TensorPtr, ProgramValue::Tensor(value)) if value.device_ptr != 0 => {
            (PROGRAM_ARGUMENT_TENSOR_PTR, value.device_ptr, None)
        }
        (WireType::StatePtr, ProgramValue::State(value)) if value.device_ptr != 0 => {
            (PROGRAM_ARGUMENT_STATE_PTR, value.device_ptr, None)
        }
        (WireType::ScratchPtr, ProgramValue::Scratch { device_ptr, .. })
            if *device_ptr != 0 =>
        {
            (PROGRAM_ARGUMENT_SCRATCH_PTR, *device_ptr, None)
        }
        (WireType::ScalarI32, ProgramValue::ScalarI32(value)) => {
            (PROGRAM_ARGUMENT_SCALAR_I32, u64::from(*value as u32), None)
        }
        (WireType::ScalarU32, ProgramValue::ScalarU32(value)) => (
            PROGRAM_ARGUMENT_SCALAR_U32,
            u64::from(*value),
            Some(u64::from(*value)),
        ),
        (WireType::ScalarU64, ProgramValue::ScalarU64(value)) => {
            (PROGRAM_ARGUMENT_SCALAR_U64, *value, Some(*value))
        }
        (WireType::ScalarF32, ProgramValue::ScalarF32(value)) => (
            PROGRAM_ARGUMENT_SCALAR_F32,
            u64::from(value.to_bits()),
            None,
        ),
        _ => return Err("Program argument wire type mismatch".into()),
    };
    Ok(result)
}

fn validate_invocation(
    variant: &Variant,
    authority: &AdapterAuthority,
    values: &BTreeMap<String, ProgramValue>,
) -> Result<(), String> {
    if values.len() != authority.slots.len()
        || authority
            .slots
            .iter()
            .any(|slot| !values.contains_key(&slot.slot))
    {
        return Err("Program invocation has missing or extra adapter slots".into());
    }
    for slot in &authority.slots {
        let value = values
            .get(&slot.slot)
            .ok_or("Program invocation omits an adapter slot")?;
        match (slot.wire_type, value) {
            (WireType::TensorPtr, ProgramValue::Tensor(tensor))
            | (WireType::StatePtr, ProgramValue::State(tensor)) => {
                validate_tensor(slot, tensor)?;
            }
            (WireType::ScratchPtr, ProgramValue::Scratch { device_ptr, bytes }) => {
                if *device_ptr == 0
                    || *device_ptr % u64::from(slot.alignment) != 0
                    || *bytes == 0
                    || *bytes > variant.scratch.max_bytes
                    || *bytes > authority.scratch_max_bytes
                {
                    return Err("Program scratch value violates adapter bounds".into());
                }
            }
            (WireType::ScalarI32, ProgramValue::ScalarI32(_))
            | (WireType::ScalarU32, ProgramValue::ScalarU32(_))
            | (WireType::ScalarU64, ProgramValue::ScalarU64(_))
            | (WireType::ScalarF32, ProgramValue::ScalarF32(_)) => {}
            (WireType::ManifestU32, _) => {
                return Err(
                    "manifest_u32 cannot be supplied as an invocation slot".into()
                );
            }
            _ => {
                return Err(
                    "Program invocation value differs from adapter slot kind".into()
                );
            }
        }
    }
    for shape in &variant.constraints.shapes {
        let tensor = tensor_value(values.get(&shape.slot))?;
        let value = tensor.extents[usize::from(shape.axis)];
        if value < shape.min
            || value > shape.max
            || value % shape.multiple_of != 0
            || shape
                .one_of
                .as_ref()
                .map(|allowed| !allowed.contains(&value))
                .unwrap_or(false)
        {
            return Err(
                "Program invocation shape is outside variant constraints".into()
            );
        }
    }
    for constraint in &variant.constraints.dtypes {
        if !constraint
            .allowed
            .contains(&tensor_value(values.get(&constraint.slot))?.dtype)
        {
            return Err(
                "Program invocation dtype is outside variant constraints".into()
            );
        }
    }
    for constraint in &variant.constraints.quantizations {
        if !constraint
            .allowed
            .contains(&tensor_value(values.get(&constraint.slot))?.quantization)
        {
            return Err(
                "Program invocation quantization is outside variant constraints".into(),
            );
        }
    }
    for constraint in &variant.constraints.layouts {
        if !constraint
            .allowed
            .contains(&tensor_value(values.get(&constraint.slot))?.layout)
        {
            return Err(
                "Program invocation layout is outside variant constraints".into()
            );
        }
    }
    Ok(())
}

fn validate_tensor(
    slot: &SlotAuthority,
    tensor: &ProgramTensorValue,
) -> Result<(), String> {
    if tensor.device_ptr == 0
        || tensor.device_ptr % u64::from(slot.alignment) != 0
        || tensor.extents.len() != usize::from(slot.rank)
        || tensor.strides.len() != usize::from(slot.rank)
        || tensor.extents.iter().any(|extent| *extent == 0)
        || tensor.strides.iter().any(|stride| *stride == 0)
        || !slot.allowed_dtypes.contains(&tensor.dtype)
        || !slot.allowed_quantizations.contains(&tensor.quantization)
        || !slot.allowed_layouts.contains(&tensor.layout)
    {
        return Err("Program tensor value violates adapter metadata".into());
    }
    for (extent, allowed) in tensor.extents.iter().zip(&slot.shapes) {
        if *extent < allowed.min
            || *extent > allowed.max
            || *extent % allowed.multiple_of != 0
            || allowed
                .one_of
                .as_ref()
                .map(|values| !values.contains(extent))
                .unwrap_or(false)
        {
            return Err("Program tensor extent violates adapter authority".into());
        }
    }
    for (stride, allowed) in tensor.strides.iter().zip(&slot.strides) {
        if *stride < allowed.min
            || *stride > allowed.max
            || *stride % allowed.multiple_of != 0
            || allowed
                .one_of
                .as_ref()
                .map(|values| !values.contains(stride))
                .unwrap_or(false)
        {
            return Err("Program tensor stride violates adapter authority".into());
        }
    }
    Ok(())
}

fn tensor_value(value: Option<&ProgramValue>) -> Result<&ProgramTensorValue, String> {
    match value {
        Some(ProgramValue::Tensor(value)) | Some(ProgramValue::State(value)) => {
            Ok(value)
        }
        _ => Err("Program tensor constraint targets a non-tensor slot".into()),
    }
}

pub fn evaluate(
    expression: &IntegerExpression,
    slots: &BTreeMap<String, u64>,
) -> Result<u64, String> {
    match expression {
        IntegerExpression::Const { value } => Ok(*value),
        IntegerExpression::Slot { slot } => slots
            .get(slot)
            .copied()
            .ok_or_else(|| format!("launch expression has no value for slot {slot}")),
        IntegerExpression::Op { op, args } => {
            let left = evaluate(&args[0], slots)?;
            let right = evaluate(&args[1], slots)?;
            match op {
                IntegerOp::Add => left.checked_add(right),
                IntegerOp::Mul => left.checked_mul(right),
                IntegerOp::CeilDiv => {
                    if right == 0 {
                        None
                    } else {
                        left.checked_add(right - 1).map(|value| value / right)
                    }
                }
                IntegerOp::Min => Some(left.min(right)),
                IntegerOp::Max => Some(left.max(right)),
            }
            .ok_or_else(|| "launch expression overflow or division by zero".into())
        }
    }
}

pub fn evaluate_launch(
    launch: &imparo_program_pack::manifest::Launch,
    slots: &BTreeMap<String, u64>,
    limits: LaunchLimits,
) -> Result<([u32; 3], u32), String> {
    let values = [
        evaluate(&launch.grid.x, slots)?,
        evaluate(&launch.grid.y, slots)?,
        evaluate(&launch.grid.z, slots)?,
    ];
    for (axis, value, ceiling) in [
        ("x", values[0], limits.grid_x),
        ("y", values[1], limits.grid_y),
        ("z", values[2], limits.grid_z),
    ] {
        if value == 0 || value > ceiling || value > u64::from(u32::MAX) {
            return Err(format!("launch grid {axis} is outside the runtime limit"));
        }
    }
    let dynamic = evaluate(&launch.dynamic_shared_bytes, slots)?;
    if dynamic > limits.dynamic_shared_bytes || dynamic > u64::from(u32::MAX) {
        return Err("dynamic shared memory exceeds the runtime limit".into());
    }
    Ok((
        [values[0] as u32, values[1] as u32, values[2] as u32],
        dynamic as u32,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use imparo_program_pack::manifest::{Block, Grid, Launch};

    fn limits() -> LaunchLimits {
        LaunchLimits {
            grid_x: 65_535,
            grid_y: 65_535,
            grid_z: 65_535,
            dynamic_shared_bytes: 48 * 1024,
        }
    }

    fn invocation_values(
        authority: &AdapterAuthority,
    ) -> BTreeMap<String, ProgramValue> {
        authority
            .slots
            .iter()
            .map(|slot| {
                let pointer = ProgramTensorValue {
                    device_ptr: u64::from(slot.alignment.max(1)) * 1024,
                    extents: slot.shapes.iter().map(|shape| shape.min).collect(),
                    strides: vec![1; usize::from(slot.rank)],
                    dtype: slot.allowed_dtypes.first().copied().unwrap_or(Dtype::F16),
                    quantization: slot
                        .allowed_quantizations
                        .first()
                        .copied()
                        .unwrap_or(Quantization::None),
                    layout: slot
                        .allowed_layouts
                        .first()
                        .copied()
                        .unwrap_or(Layout::Contiguous),
                };
                let value = match slot.wire_type {
                    WireType::TensorPtr => ProgramValue::Tensor(pointer),
                    WireType::StatePtr => ProgramValue::State(pointer),
                    WireType::ScratchPtr => ProgramValue::Scratch {
                        device_ptr: u64::from(slot.alignment.max(1)) * 1024,
                        bytes: 1,
                    },
                    WireType::ScalarI32 => ProgramValue::ScalarI32(1),
                    WireType::ScalarU32 => ProgramValue::ScalarU32(1),
                    WireType::ScalarU64 => ProgramValue::ScalarU64(1),
                    WireType::ScalarF32 => ProgramValue::ScalarF32(1.0),
                    WireType::ManifestU32 => {
                        panic!("manifest constant cannot be a slot")
                    }
                };
                (slot.slot.clone(), value)
            })
            .collect()
    }

    #[test]
    fn checked_expression_rejects_overflow_division_by_zero_and_zero_grid() {
        let slots = BTreeMap::new();
        for expression in [
            IntegerExpression::Op {
                op: IntegerOp::Add,
                args: Box::new([
                    IntegerExpression::Const { value: u64::MAX },
                    IntegerExpression::Const { value: 1 },
                ]),
            },
            IntegerExpression::Op {
                op: IntegerOp::CeilDiv,
                args: Box::new([
                    IntegerExpression::Const { value: 1 },
                    IntegerExpression::Const { value: 0 },
                ]),
            },
        ] {
            assert!(evaluate(&expression, &slots).is_err());
        }
        let launch = Launch {
            arguments: Vec::new(),
            grid: Grid {
                x: IntegerExpression::Const { value: 0 },
                y: IntegerExpression::Const { value: 1 },
                z: IntegerExpression::Const { value: 1 },
            },
            block: Block { x: 1, y: 1, z: 1 },
            dynamic_shared_bytes: IntegerExpression::Const { value: 0 },
        };
        assert!(evaluate_launch(&launch, &slots, limits()).is_err());
    }

    #[test]
    fn invocation_rejects_missing_extra_and_wrong_kind_slots() {
        let manifest = super::super::constraints::tests::manifest();
        let authority = super::super::constraints::tests::authority(&manifest);
        let variant = &manifest.variants()[0];
        let valid = invocation_values(&authority);
        validate_invocation(variant, &authority, &valid).unwrap();

        let mut missing = valid.clone();
        let first = authority.slots[0].slot.clone();
        missing.remove(&first);
        assert!(validate_invocation(variant, &authority, &missing).is_err());

        let mut extra = valid.clone();
        extra.insert("attacker_extra".into(), ProgramValue::ScalarU32(7));
        assert!(validate_invocation(variant, &authority, &extra).is_err());

        let mut wrong = valid;
        wrong.insert(first, ProgramValue::ScalarU64(1));
        assert!(validate_invocation(variant, &authority, &wrong).is_err());

        let tensor_slot = authority
            .slots
            .iter()
            .find(|slot| {
                matches!(slot.wire_type, WireType::TensorPtr | WireType::StatePtr)
            })
            .unwrap();
        let mut bad_stride = invocation_values(&authority);
        match bad_stride.get_mut(&tensor_slot.slot).unwrap() {
            ProgramValue::Tensor(value) | ProgramValue::State(value) => {
                value.strides[0] = tensor_slot.strides[0].max + 1;
            }
            _ => unreachable!(),
        }
        assert!(validate_invocation(variant, &authority, &bad_stride).is_err());
    }
}
