//! Opt-in RTX 3060/SM86 lifecycle test. Run with:
//! `IMPARO_CUDA_ARCHS=86 IMPARO_CUDA_PROGRAM_SMOKE=1 cargo test --locked
//!  -p imparo-cuda --features cuda-static program_pack_sm86 -- --nocapture`

use super::constraints::{
    AdapterAuthority, ArgumentAuthority, GraphMutability, NumericalClassKind,
    RuntimeEligibility, SlotAuthority, SlotLifetime, SlotRole, StateInitialization,
    WorkloadAuthority, decode_hex_32,
};
use super::launch::{ProgramTensorValue, ProgramValue, launch_bound};
use super::shutdown;
use crate::ffi::{
    imparo_cuda_program_test_alloc, imparo_cuda_program_test_free,
    imparo_cuda_program_test_graph_alive, imparo_cuda_program_test_graph_begin,
    imparo_cuda_program_test_graph_end_replay, imparo_cuda_program_test_graph_replay,
    imparo_cuda_program_test_read_u32, imparo_cuda_program_test_synchronize,
    imparo_cuda_program_test_write_u32,
};
use base64::Engine as _;
use ed25519_dalek::{Signer as _, SigningKey};
use imparo_backend::BackendPrograms as _;
use imparo_program_pack::identity::{encode_hex, sha256, signature_message};
use imparo_program_pack::manifest::{
    AliasMode, Determinism, Dtype, Layout, MathMode, Quantization, WireType,
};
use imparo_program_pack::{
    DistributionScope, ExtensionRegistry, Installer, Manifest, TrustStore, TrustedKey,
};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::process::Command;

const KEY_ID: &str = "imparo.test.program-smoke";
const CHANNEL: &str = "imparo.community.stable";
const CAPTURE_GROUP: &str = "imparo.cuda.program_smoke.capture.v1";
const UPDATE_GROUP: &str = "imparo.cuda.program_smoke.update.v1";
const CAPTURE_VARIANT: &str =
    "1111111111111111111111111111111111111111111111111111111111111111";
const UPDATE_VARIANT: &str =
    "1212121212121212121212121212121212121212121212121212121212121212";
const CAPTURE_SYMBOL: &str =
    "ip_1111111111111111111111111111111111111111111111111111111111111111";
const UPDATE_SYMBOL: &str =
    "ip_1212121212121212121212121212121212121212121212121212121212121212";
const TRITON_SMOKE_CONTRACT: &str = "imparo.cuda.infrastructure.copy";
const TRITON_SMOKE_CONTRACT_SHA256: &str =
    "37d08404f20f83b2076f435208883dee937b985414629a4040df0fe896dd9d75";
const TRITON_SMOKE_SYMBOL: &str =
    "ip_ab8511ce441e25e37a94fd947a072a657077a099a3ce4b944cbcc7e002a34a2c";

fn compile_sm86_cubin(root: &Path) -> Vec<u8> {
    let source = root.join("program_smoke.cu");
    let cubin = root.join("program_smoke.sm86.cubin");
    fs::write(
        &source,
        format!(
            r#"
extern "C" __global__ void {CAPTURE_SYMBOL}(unsigned int * output,
                                                   unsigned int value) {{
    if (threadIdx.x == 0 && blockIdx.x == 0) *output = value;
}}
extern "C" __global__ void {UPDATE_SYMBOL}(unsigned int * output,
                                                  unsigned int value) {{
    if (threadIdx.x == 0 && blockIdx.x == 0) *output = value;
}}
"#
        ),
    )
    .unwrap();
    let nvcc = std::env::var("NVCC").unwrap_or_else(|_| "nvcc".into());
    let output = Command::new(nvcc)
        .args(["--cubin", "-arch=sm_86", "-std=c++17"])
        .arg(&source)
        .arg("-o")
        .arg(&cubin)
        .output()
        .expect("launch nvcc for the SM86 Program smoke cubin");
    assert!(
        output.status.success(),
        "SM86 smoke cubin compile failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    fs::read(cubin).unwrap()
}

fn effect(slot: &str, access: &str) -> Value {
    json!({"slot": slot, "access": access, "aliasing": []})
}

fn launch(symbol: &str, graph_capture: &str, update_slots: &[&str]) -> Value {
    json!({
        "variant_id": if symbol == CAPTURE_SYMBOL { CAPTURE_VARIANT } else { UPDATE_VARIANT },
        "config_id": if symbol == CAPTURE_SYMBOL { "2121212121212121212121212121212121212121212121212121212121212121" } else { "2222222222222222222222222222222222222222222222222222222222222222" },
        "choice_group_id": if symbol == CAPTURE_SYMBOL { CAPTURE_GROUP } else { UPDATE_GROUP },
        "contract": {
            "id": if symbol == CAPTURE_SYMBOL { "imparo.cuda.program_smoke.capture" } else { "imparo.cuda.program_smoke.update" },
            "revision": 1,
            "sha256": if symbol == CAPTURE_SYMBOL { "3131313131313131313131313131313131313131313131313131313131313131" } else { "3232323232323232323232323232323232323232323232323232323232323232" }
        },
        "module_id": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "symbol": symbol,
        "constraints": {
            "shapes": [], "dtypes": [], "quantizations": [], "layouts": [],
            "alignments": [{"slot": "output", "bytes": 4}]
        },
        "effects": [effect("output", "write"), effect("value", "read")],
        "scratch": {"max_bytes": 0, "alignment": 16, "zero_initialized": false},
        "launch": {
            "arguments": [
                {"kind": "slot", "slot": "output", "wire_type": "tensor_ptr"},
                {"kind": "slot", "slot": "value", "wire_type": "scalar_u32"}
            ],
            "grid": {
                "x": {"kind": "const", "value": 1},
                "y": {"kind": "const", "value": 1},
                "z": {"kind": "const", "value": 1}
            },
            "block": {"x": 32, "y": 1, "z": 1},
            "dynamic_shared_bytes": {"kind": "const", "value": 0}
        },
        "resources": {
            "registers_per_thread_max": 255,
            "static_shared_bytes_max": 0,
            "dynamic_shared_bytes_max": 0,
            "local_memory_bytes_max": 0,
            "threads_per_block_max": 32
        },
        "graph_capture": graph_capture,
        "graph_update_slots": update_slots,
        "numerical_class": {
            "kind": "gate_bounded", "gate_suite": "imparo.cuda.gates.v1",
            "contract_version": 1
        },
        "determinism": "required",
        "bit_affecting": false,
        "required_entitlement_features": [],
        "requires": [], "conflicts": [], "provides": [], "joint_with": []
    })
}

fn write_signed_pack(
    root: &Path,
    cubin: &[u8],
    key: &SigningKey,
    version: &str,
    driver: u32,
) {
    let sbom = br#"{"spdxVersion":"SPDX-2.3"}"#;
    let notices = b"SM86 Program bridge hardware test only";
    let provenance = br#"{"builder":"imparo-step3-smoke","target":"sm86"}"#;
    let module_hash = encode_hex(&sha256(cubin));
    let module_file = format!("modules/{module_hash}.cubin");
    fs::create_dir_all(root.join("modules")).unwrap();
    fs::write(root.join(&module_file), cubin).unwrap();
    fs::write(root.join("SBOM.spdx.json"), sbom).unwrap();
    fs::write(root.join("THIRD_PARTY_NOTICES"), notices).unwrap();
    fs::write(root.join("provenance.json"), provenance).unwrap();

    let group = |id: &str, contract_id: &str, contract_hash: &str, fixture: &str| {
        json!({
            "choice_group_id": id,
            "contract": {"id": contract_id, "revision": 1, "sha256": contract_hash},
            "workload": {
                "workload_id": "imparo.workload.attention_decode", "revision": 1,
                "parameters": {},
                "parameters_sha256": "1aa54ff326d23ef77b0dd6db032f335ae5d32710cff814846b28b7a2fc3ad351",
                "fixture_sha256": fixture
            },
            "screened": true, "bit_affecting": false, "joint_with": []
        })
    };
    let manifest_value = json!({
        "schema": 1,
        "program_pack_abi": 1,
        "pack_id": "com.zeraix.imparo.test.cuda.sm86",
        "pack_version": version,
        "distribution": "community",
        "release_channel": CHANNEL,
        "required_entitlement_features": [],
        "backend": "cuda",
        "engine_api": {"min": 1, "max_exclusive": 2},
        "backend_abi": {"min": crate::CUDA_BACKEND_ABI, "max_exclusive": crate::CUDA_BACKEND_ABI + 1},
        "target": {"sm": 86, "warp_size": 32, "driver_min": driver, "math_mode": "fast"},
        "required_extensions": [], "optional_extensions": [],
        "toolchain": {
            "builder_image_sha256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
            "build_recipe_sha256": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
            "producer": {
                "kind": "cuda_cpp", "compiler_id": "nvcc",
                "compiler_version": "13.0", "cuda_toolkit": "13.0"
            }
        },
        "modules": [{
            "id": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "file": module_file, "format": "cubin", "bytes": cubin.len(),
            "sha256": module_hash
        }],
        "choice_groups": [
            group(CAPTURE_GROUP, "imparo.cuda.program_smoke.capture", "3131313131313131313131313131313131313131313131313131313131313131", "4141414141414141414141414141414141414141414141414141414141414141"),
            group(UPDATE_GROUP, "imparo.cuda.program_smoke.update", "3232323232323232323232323232323232323232323232323232323232323232", "4242424242424242424242424242424242424242424242424242424242424242")
        ],
        "variants": [
            launch(CAPTURE_SYMBOL, "capture_only", &[]),
            launch(UPDATE_SYMBOL, "replay_update_safe", &["value"])
        ],
        "notices_sha256": encode_hex(&sha256(notices)),
        "sbom_sha256": encode_hex(&sha256(sbom)),
        "provenance_sha256": encode_hex(&sha256(provenance))
    });
    let raw = serde_json::to_vec(&manifest_value).unwrap();
    Manifest::parse(&raw).expect("generated SM86 Program smoke manifest");
    fs::write(root.join("manifest.json"), &raw).unwrap();
    let signature = key.sign(&signature_message(&raw));
    let envelope = format!(
        "{{\"schema\":1,\"domain\":\"imparo-program-pack-v1\",\"algorithm\":\"ed25519\",\"key_id\":\"{KEY_ID}\",\"manifest_bytes\":{},\"manifest_sha256\":\"{}\",\"signature\":\"{}\"}}",
        raw.len(),
        encode_hex(&sha256(&raw)),
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(signature.to_bytes())
    );
    fs::write(root.join("manifest.sig"), envelope).unwrap();
}

#[derive(Clone, Copy)]
enum NegativePack {
    WrongSm,
    WrongBackendAbi,
    NewerDriver,
    ResourceUnderclaim,
}

fn write_negative_pack(
    root: &Path,
    cubin: &[u8],
    key: &SigningKey,
    version: &str,
    driver: u32,
    kind: NegativePack,
) {
    write_signed_pack(root, cubin, key, version, driver);
    let mut value: Value =
        serde_json::from_slice(&fs::read(root.join("manifest.json")).unwrap()).unwrap();
    match kind {
        NegativePack::WrongSm => value["target"]["sm"] = json!(89),
        NegativePack::WrongBackendAbi => {
            value["backend_abi"]["min"] = json!(crate::CUDA_BACKEND_ABI + 1);
            value["backend_abi"]["max_exclusive"] = json!(crate::CUDA_BACKEND_ABI + 2);
        }
        NegativePack::NewerDriver => {
            value["target"]["driver_min"] = json!(driver + 1);
        }
        NegativePack::ResourceUnderclaim => {
            for variant in value["variants"].as_array_mut().unwrap() {
                variant["resources"]["registers_per_thread_max"] = json!(0);
            }
        }
    }
    let raw = serde_json::to_vec(&value).unwrap();
    Manifest::parse(&raw).expect("negative smoke pack remains schema-valid");
    fs::write(root.join("manifest.json"), &raw).unwrap();
    let signature = key.sign(&signature_message(&raw));
    let envelope = format!(
        "{{\"schema\":1,\"domain\":\"imparo-program-pack-v1\",\"algorithm\":\"ed25519\",\"key_id\":\"{KEY_ID}\",\"manifest_bytes\":{},\"manifest_sha256\":\"{}\",\"signature\":\"{}\"}}",
        raw.len(),
        encode_hex(&sha256(&raw)),
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(signature.to_bytes())
    );
    fs::write(root.join("manifest.sig"), envelope).unwrap();
}

fn authorities(manifest: &Manifest) -> Vec<AdapterAuthority> {
    manifest
        .variants()
        .iter()
        .map(|variant| {
            let group = manifest
                .choice_groups()
                .iter()
                .find(|group| group.choice_group_id == variant.choice_group_id)
                .unwrap();
            let effects = variant
                .effects
                .iter()
                .map(|effect| {
                    (
                        effect.slot.as_str(),
                        (
                            effect.access,
                            effect
                                .aliasing
                                .iter()
                                .map(|item| (item.target_slot.clone(), item.mode))
                                .collect::<BTreeMap<String, AliasMode>>(),
                        ),
                    )
                })
                .collect::<BTreeMap<_, _>>();
            let slots = variant
                .launch
                .arguments
                .iter()
                .enumerate()
                .filter_map(|(index, argument)| match argument {
                    imparo_program_pack::manifest::LaunchArgument::Slot {
                        slot,
                        wire_type,
                    } => {
                        let (access, aliases) = effects.get(slot.as_str())?.clone();
                        Some(SlotAuthority {
                            slot: slot.clone(),
                            abi_index: index as u32,
                            wire_type: *wire_type,
                            role: if *wire_type == WireType::TensorPtr {
                                SlotRole::Output
                            } else {
                                SlotRole::Shape
                            },
                            access,
                            lifetime: if variant.graph_update_slots.contains(slot) {
                                SlotLifetime::Graph
                            } else {
                                SlotLifetime::Invocation
                            },
                            graph_mutability: if variant
                                .graph_update_slots
                                .contains(slot)
                            {
                                GraphMutability::ReplayUpdate
                            } else {
                                GraphMutability::CaptureStatic
                            },
                            update_source: if variant.graph_update_slots.contains(slot)
                            {
                                1
                            } else {
                                0
                            },
                            rank: u8::from(*wire_type == WireType::TensorPtr),
                            extent_authority:
                                super::constraints::DescriptorAuthority::EngineAdapter,
                            stride_authority:
                                super::constraints::DescriptorAuthority::EngineAdapter,
                            shapes: if *wire_type == WireType::TensorPtr {
                                vec![super::constraints::ShapeAuthority {
                                    min: 1,
                                    max: 1,
                                    multiple_of: 1,
                                    one_of: Some(vec![1]),
                                }]
                            } else {
                                Vec::new()
                            },
                            strides: if *wire_type == WireType::TensorPtr {
                                vec![super::constraints::ShapeAuthority {
                                    min: 1,
                                    max: 1,
                                    multiple_of: 1,
                                    one_of: Some(vec![1]),
                                }]
                            } else {
                                Vec::new()
                            },
                            alignment: if *wire_type == WireType::TensorPtr {
                                4
                            } else {
                                1
                            },
                            state_initialization: StateInitialization::None,
                            allowed_dtypes: if *wire_type == WireType::TensorPtr {
                                vec![Dtype::F16]
                            } else {
                                Vec::new()
                            },
                            allowed_quantizations: if *wire_type == WireType::TensorPtr
                            {
                                vec![Quantization::None]
                            } else {
                                Vec::new()
                            },
                            allowed_layouts: if *wire_type == WireType::TensorPtr {
                                vec![Layout::Contiguous]
                            } else {
                                Vec::new()
                            },
                            aliases,
                        })
                    }
                    imparo_program_pack::manifest::LaunchArgument::ManifestU32 {
                        ..
                    } => None,
                })
                .collect();
            let arguments = variant
                .launch
                .arguments
                .iter()
                .map(|argument| match argument {
                    imparo_program_pack::manifest::LaunchArgument::Slot {
                        slot,
                        wire_type,
                    } => ArgumentAuthority::Slot {
                        slot: slot.clone(),
                        wire_type: *wire_type,
                    },
                    imparo_program_pack::manifest::LaunchArgument::ManifestU32 {
                        name,
                        value,
                    } => ArgumentAuthority::ManifestU32 {
                        name: name.clone(),
                        min: *value,
                        max: *value,
                    },
                })
                .collect();
            AdapterAuthority {
                choice_group_id: variant.choice_group_id.clone(),
                contract_id: variant.contract.id.clone(),
                contract_revision: variant.contract.revision,
                contract_sha256: decode_hex_32(&variant.contract.sha256).unwrap(),
                adapter_id: format!("{}.adapter", variant.contract.id),
                adapter_revision: 1,
                adapter_sha256: [0xa5; 32],
                state_reset_revision: 1,
                workloads: vec![WorkloadAuthority {
                    workload_id: group.workload.workload_id,
                    revision: group.workload.revision,
                    fixture_sha256: decode_hex_32(&group.workload.fixture_sha256)
                        .unwrap(),
                }],
                slots,
                arguments,
                scratch_max_bytes: 0,
                scratch_alignment: 16,
                allow_zero_initialized_scratch: false,
                registers_per_thread_max: 255,
                static_shared_bytes_max: 0,
                dynamic_shared_bytes_max: 0,
                local_memory_bytes_max: 0,
                threads_per_block_max: 32,
                allowed_numerical_classes: vec![NumericalClassKind::GateBounded],
                required_determinism: Some(Determinism::Required),
                allowed_graph_capture: vec![variant.graph_capture],
                builtin_variant_id: if variant.choice_group_id == CAPTURE_GROUP {
                    [0xb1; 32]
                } else {
                    [0xb3; 32]
                },
                builtin_config_id: if variant.choice_group_id == CAPTURE_GROUP {
                    [0xb2; 32]
                } else {
                    [0xb4; 32]
                },
                applies: |_variant, _facts, _profile| true,
            }
        })
        .collect()
}

fn values(pointer: u64, value: u32) -> BTreeMap<String, ProgramValue> {
    BTreeMap::from([
        (
            "output".into(),
            ProgramValue::Tensor(ProgramTensorValue {
                device_ptr: pointer,
                extents: vec![1],
                strides: vec![1],
                dtype: Dtype::F16,
                quantization: Quantization::None,
                layout: Layout::Contiguous,
            }),
        ),
        ("value".into(), ProgramValue::ScalarU32(value)),
    ])
}

fn read(pointer: u64) -> u32 {
    let mut output = 0;
    assert_eq!(
        unsafe { imparo_cuda_program_test_read_u32(pointer, &raw mut output) },
        0
    );
    output
}

fn triton_smoke_authority(manifest: &Manifest) -> AdapterAuthority {
    let variant = manifest
        .variants()
        .first()
        .expect("Triton smoke pack has one variant");
    let group = manifest
        .choice_groups()
        .first()
        .expect("Triton smoke pack has one choice group");
    assert_eq!(variant.contract.id, TRITON_SMOKE_CONTRACT);
    assert_eq!(variant.contract.sha256, TRITON_SMOKE_CONTRACT_SHA256);
    assert_eq!(variant.symbol, TRITON_SMOKE_SYMBOL);
    let tensor_shape = super::constraints::ShapeAuthority {
        min: 0,
        max: 256,
        multiple_of: 1,
        one_of: None,
    };
    let tensor_stride = super::constraints::ShapeAuthority {
        min: 1,
        max: 1,
        multiple_of: 1,
        one_of: Some(vec![1]),
    };
    let slot = |name: &str, index: u32, role: SlotRole, access| SlotAuthority {
        slot: name.into(),
        abi_index: index,
        wire_type: if name == "n_elements" {
            WireType::ScalarU32
        } else {
            WireType::TensorPtr
        },
        role,
        access,
        lifetime: SlotLifetime::Invocation,
        graph_mutability: GraphMutability::CaptureStatic,
        update_source: 0,
        rank: u8::from(name != "n_elements"),
        extent_authority: super::constraints::DescriptorAuthority::EngineAdapter,
        stride_authority: super::constraints::DescriptorAuthority::EngineAdapter,
        shapes: if name == "n_elements" {
            Vec::new()
        } else {
            vec![tensor_shape.clone()]
        },
        strides: if name == "n_elements" {
            Vec::new()
        } else {
            vec![tensor_stride.clone()]
        },
        alignment: if name == "n_elements" { 1 } else { 16 },
        state_initialization: StateInitialization::None,
        allowed_dtypes: if name == "n_elements" {
            Vec::new()
        } else {
            vec![Dtype::F32]
        },
        allowed_quantizations: if name == "n_elements" {
            Vec::new()
        } else {
            vec![Quantization::None]
        },
        allowed_layouts: if name == "n_elements" {
            Vec::new()
        } else {
            vec![Layout::Contiguous]
        },
        aliases: BTreeMap::new(),
    };
    AdapterAuthority {
        choice_group_id: group.choice_group_id.clone(),
        contract_id: TRITON_SMOKE_CONTRACT.into(),
        contract_revision: 1,
        contract_sha256: decode_hex_32(TRITON_SMOKE_CONTRACT_SHA256).unwrap(),
        adapter_id: "imparo.cuda.infrastructure.copy.test-adapter".into(),
        adapter_revision: 1,
        adapter_sha256: [0xc0; 32],
        state_reset_revision: 1,
        workloads: vec![WorkloadAuthority {
            workload_id: group.workload.workload_id,
            revision: group.workload.revision,
            fixture_sha256: decode_hex_32(&group.workload.fixture_sha256).unwrap(),
        }],
        slots: vec![
            slot(
                "output",
                0,
                SlotRole::Output,
                imparo_program_pack::manifest::Access::Write,
            ),
            slot(
                "input",
                1,
                SlotRole::Input,
                imparo_program_pack::manifest::Access::Read,
            ),
            slot(
                "n_elements",
                2,
                SlotRole::Shape,
                imparo_program_pack::manifest::Access::Read,
            ),
        ],
        arguments: vec![
            ArgumentAuthority::Slot {
                slot: "output".into(),
                wire_type: WireType::TensorPtr,
            },
            ArgumentAuthority::Slot {
                slot: "input".into(),
                wire_type: WireType::TensorPtr,
            },
            ArgumentAuthority::Slot {
                slot: "n_elements".into(),
                wire_type: WireType::ScalarU32,
            },
        ],
        scratch_max_bytes: 0,
        scratch_alignment: 16,
        allow_zero_initialized_scratch: false,
        registers_per_thread_max: 255,
        static_shared_bytes_max: 0,
        dynamic_shared_bytes_max: 0,
        local_memory_bytes_max: 0,
        threads_per_block_max: 32,
        allowed_numerical_classes: vec![NumericalClassKind::DiagnosticOnly],
        required_determinism: Some(Determinism::Required),
        allowed_graph_capture: vec![
            imparo_program_pack::manifest::GraphCapture::Forbidden,
        ],
        builtin_variant_id: [0xc1; 32],
        builtin_config_id: [0xc2; 32],
        applies: |_variant, _facts, _profile| true,
    }
}

fn triton_smoke_values(output: u64, input: u64) -> BTreeMap<String, ProgramValue> {
    let tensor = |device_ptr| {
        ProgramValue::Tensor(ProgramTensorValue {
            device_ptr,
            extents: vec![1],
            strides: vec![1],
            dtype: Dtype::F32,
            quantization: Quantization::None,
            layout: Layout::Contiguous,
        })
    };
    BTreeMap::from([
        ("output".into(), tensor(output)),
        ("input".into(), tensor(input)),
        ("n_elements".into(), ProgramValue::ScalarU32(1)),
    ])
}

fn external_triton_installer()
-> Option<(std::path::PathBuf, tempfile::TempDir, Installer)> {
    let required = match std::env::var("IMPARO_REQUIRE_TRITON_SMOKE") {
        Ok(value) if value == "1" => true,
        Ok(value) if value == "0" => false,
        Ok(_) => panic!("IMPARO_REQUIRE_TRITON_SMOKE must be exactly 0 or 1"),
        Err(std::env::VarError::NotPresent) => false,
        Err(std::env::VarError::NotUnicode(_)) => {
            panic!("IMPARO_REQUIRE_TRITON_SMOKE is not Unicode")
        }
    };
    let Ok(source) = std::env::var("IMPARO_TRITON_SMOKE_PACK") else {
        assert!(
            !required,
            "formal Triton smoke requires IMPARO_TRITON_SMOKE_PACK"
        );
        eprintln!("skipping external Triton pack: IMPARO_TRITON_SMOKE_PACK is unset");
        return None;
    };
    let public_key = std::env::var("IMPARO_TRITON_SMOKE_PUBLIC_KEY_HEX")
        .expect("external Triton smoke requires its public key");
    let key_id = std::env::var("IMPARO_TRITON_SMOKE_KEY_ID")
        .expect("external Triton smoke requires its signing key id");
    let source = std::path::PathBuf::from(source);
    let source_manifest =
        Manifest::parse(&fs::read(source.join("manifest.json")).unwrap()).unwrap();
    let mut trust = TrustStore::new();
    trust
        .add(TrustedKey {
            key_id,
            public_key: decode_hex_32(&public_key).unwrap(),
            scope: DistributionScope::Community,
            release_channel: source_manifest.release_channel().into(),
        })
        .unwrap();
    let cache = tempfile::tempdir().unwrap();
    let installer =
        Installer::new(cache.path(), trust, ExtensionRegistry::new()).unwrap();
    Some((source, cache, installer))
}

/// Cross-language contract: Python/OpenSSL exact bytes must pass Rust's canonical
/// Ed25519 envelope and immutable Installer admission. No GPU or Python is used here.
#[test]
fn triton_pack_python_signature_is_accepted_by_rust_installer() {
    let Some((source, _cache, installer)) = external_triton_installer() else {
        return;
    };
    let admission = installer.install_from_staged_dir(&source).unwrap();
    let admitted = installer
        .load_admitted_pack(admission.install_digest())
        .unwrap();
    assert_eq!(
        admitted.manifest().pack_id(),
        "com.zeraix.imparo.community.cuda.smoke"
    );
    assert_eq!(admitted.manifest().target().sm, 86);
    assert_eq!(admitted.modules().len(), 1);
}

/// Cross-platform gate for a signed SM86 pack produced in pinned Linux CI. It may
/// skip in ordinary host tests, but the formal lane sets `IMPARO_REQUIRE_TRITON_SMOKE=1`
/// so a missing artifact cannot be reported as a pass.
/// The test deliberately never starts Python or a compiler. The adapter authority is
/// test-only, so this infrastructure smoke cannot become a production candidate.
#[test]
fn triton_pack_linux_aot_loads_and_runs_on_clean_windows_sm86() {
    let Some((source, _cache, installer)) = external_triton_installer() else {
        return;
    };
    let raw_manifest = fs::read(source.join("manifest.json")).unwrap();
    let source_manifest = Manifest::parse(&raw_manifest).unwrap();
    assert_eq!(source_manifest.target().sm, 86);
    assert_eq!(source_manifest.variants().len(), 1);
    assert_eq!(source_manifest.choice_groups().len(), 1);

    let identity = crate::runtime_identity().expect("live CUDA runtime identity");
    assert_eq!(identity.device_sm, 86, "this evidence is exact-SM86 only");
    let admission = installer.install_from_staged_dir(&source).unwrap();
    let admitted = installer
        .load_admitted_pack(admission.install_digest())
        .unwrap();
    let authority = triton_smoke_authority(admitted.manifest());
    let runtime = RuntimeEligibility {
        device_sm: identity.device_sm,
        driver_version: identity.driver_version,
        backend_abi: identity.backend_abi,
        math_mode: admitted.manifest().target().math_mode,
    };
    super::registry::install_for_test(
        &installer,
        admission.install_digest(),
        &runtime,
        &[authority],
    )
    .unwrap();

    let backend = crate::CudaBackend;
    let group = &admitted.manifest().choice_groups()[0].choice_group_id;
    let variant = decode_hex_32(&admitted.manifest().variants()[0].variant_id).unwrap();
    backend.freeze_program_catalog().unwrap();
    backend.bind_program_choice(group, &variant).unwrap();
    let mut output = 0_u64;
    let mut input = 0_u64;
    assert_eq!(
        unsafe { imparo_cuda_program_test_alloc(4, &raw mut output) },
        0
    );
    assert_eq!(
        unsafe { imparo_cuda_program_test_alloc(4, &raw mut input) },
        0
    );
    let expected = 3.5_f32.to_bits();
    assert_eq!(
        unsafe { imparo_cuda_program_test_write_u32(input, expected) },
        0
    );
    assert!(
        unsafe { launch_bound(group, &triton_smoke_values(output, input)) }.unwrap()
    );
    assert_eq!(unsafe { imparo_cuda_program_test_synchronize() }, 0);
    assert_eq!(read(output), expected);
    shutdown().unwrap();
    assert_eq!(unsafe { imparo_cuda_program_test_free(output) }, 0);
    assert_eq!(unsafe { imparo_cuda_program_test_free(input) }, 0);
}

fn query_candidate(backend: &crate::CudaBackend, group: &str, variant: &[u8; 32]) {
    let facts = imparo_backend::ModelFacts {
        n_embd: 1024,
        n_ff: 4096,
        n_head: 8,
        n_kv: 4,
        head_dim: 128,
        deep_head_dim: 128,
        n_experts: 0,
        n_layers: 16,
        layer_dispatches: 8,
        weight_kinds: 1,
    };
    let profile = imparo_backend::DeviceProfile {
        threadgroup_bytes: 64 * 1024,
        max_threads: 1024,
        ..imparo_backend::DeviceProfile::default()
    };
    let choices = backend.program_choices(&facts, &profile);
    assert!(choices.iter().any(|choice| {
        choice.choice_group_id == group
            && choice
                .candidates
                .iter()
                .any(|candidate| &candidate.variant_id == variant)
    }));
}

#[test]
fn program_pack_sm86_signed_lifecycle_and_graph_smoke() {
    let identity = crate::runtime_identity().expect("live CUDA runtime identity");
    assert_eq!(identity.device_sm, 86, "this evidence is exact-SM86 only");
    assert_eq!(identity.backend_abi, crate::CUDA_BACKEND_ABI);

    let temp = tempfile::tempdir().unwrap();
    let cubin = compile_sm86_cubin(temp.path());
    let key = SigningKey::from_bytes(&[0x5a; 32]);
    let valid_source = temp.path().join("valid");
    let corrupt_source = temp.path().join("corrupt");
    fs::create_dir(&valid_source).unwrap();
    fs::create_dir(&corrupt_source).unwrap();
    write_signed_pack(
        &valid_source,
        &cubin,
        &key,
        "1.0.0",
        identity.driver_version,
    );
    write_signed_pack(
        &corrupt_source,
        &cubin,
        &key,
        "1.0.1",
        identity.driver_version,
    );
    let negative_specs = [
        ("wrong-sm", "1.1.0", NegativePack::WrongSm),
        ("wrong-abi", "1.2.0", NegativePack::WrongBackendAbi),
        ("newer-driver", "1.3.0", NegativePack::NewerDriver),
        (
            "resource-underclaim",
            "1.4.0",
            NegativePack::ResourceUnderclaim,
        ),
    ];
    let mut negative_sources = Vec::new();
    for (name, version, kind) in negative_specs {
        let source = temp.path().join(name);
        fs::create_dir(&source).unwrap();
        write_negative_pack(
            &source,
            &cubin,
            &key,
            version,
            identity.driver_version,
            kind,
        );
        negative_sources.push((name, kind, source));
    }
    let cache = temp.path().join("cache");
    let mut trust = TrustStore::new();
    trust
        .add(TrustedKey {
            key_id: KEY_ID.into(),
            public_key: key.verifying_key().to_bytes(),
            scope: DistributionScope::Community,
            release_channel: CHANNEL.into(),
        })
        .unwrap();
    let installer = Installer::new(&cache, trust, ExtensionRegistry::new()).unwrap();
    let valid = installer.install_from_staged_dir(&valid_source).unwrap();
    let corrupt = installer.install_from_staged_dir(&corrupt_source).unwrap();
    let negative = negative_sources
        .into_iter()
        .map(|(name, kind, source)| {
            (
                name,
                kind,
                installer.install_from_staged_dir(&source).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    let manifest = installer
        .load_admitted_pack(valid.install_digest())
        .unwrap()
        .manifest()
        .clone();
    let authorities = authorities(&manifest);
    let runtime = RuntimeEligibility {
        device_sm: identity.device_sm,
        driver_version: identity.driver_version,
        backend_abi: identity.backend_abi,
        math_mode: MathMode::Fast,
    };
    let capture_variant = decode_hex_32(CAPTURE_VARIANT).unwrap();
    let update_variant = decode_hex_32(UPDATE_VARIANT).unwrap();
    let backend = crate::CudaBackend;
    let mut output_pointer = 0_u64;
    assert_eq!(
        unsafe { imparo_cuda_program_test_alloc(4, &raw mut output_pointer) },
        0
    );

    // Each signed-but-ineligible pack is rejected at its authoritative layer. A
    // subsequent known-good install still exposes the built-in route, proving that
    // neither Rust nor native transactional state was damaged.
    for (name, kind, admission) in &negative {
        let error = match super::registry::install_for_test(
            &installer,
            admission.install_digest(),
            &runtime,
            &authorities,
        ) {
            Ok(_) => panic!("{name} unexpectedly installed"),
            Err(error) => error,
        };
        match kind {
            NegativePack::WrongSm => assert!(error.contains("exact SM")),
            NegativePack::WrongBackendAbi => {
                assert!(error.contains("native CUDA ABI"));
            }
            NegativePack::NewerDriver => assert!(error.contains("newer CUDA driver")),
            NegativePack::ResourceUnderclaim => {
                assert!(error.contains("native Program Pack install rejected"));
            }
        }
        super::registry::install_for_test(
            &installer,
            valid.install_digest(),
            &runtime,
            &authorities,
        )
        .unwrap();
        backend.freeze_program_catalog().unwrap();
        assert!(
            !unsafe { launch_bound(CAPTURE_GROUP, &values(output_pointer, 9)) }
                .unwrap()
        );
        shutdown().unwrap();
    }

    // Exercise the complete synchronous module lifecycle, not just repeated kernel
    // launches against one pinned module.
    for iteration in 0..1_000_u32 {
        let report = super::registry::install_for_test(
            &installer,
            valid.install_digest(),
            &runtime,
            &authorities,
        )
        .unwrap();
        assert_eq!(report.eligible_variants, 2);
        query_candidate(&backend, CAPTURE_GROUP, &capture_variant);
        backend.freeze_program_catalog().unwrap();
        backend
            .bind_program_choice(CAPTURE_GROUP, &capture_variant)
            .unwrap();
        assert_eq!(
            unsafe { imparo_cuda_program_test_write_u32(output_pointer, 0) },
            0
        );
        assert!(
            unsafe {
                launch_bound(CAPTURE_GROUP, &values(output_pointer, iteration + 1))
            }
            .unwrap()
        );
        assert_eq!(unsafe { imparo_cuda_program_test_synchronize() }, 0);
        assert_eq!(read(output_pointer), iteration + 1);
        shutdown().unwrap();
    }

    super::registry::install_for_test(
        &installer,
        valid.install_digest(),
        &runtime,
        &authorities,
    )
    .unwrap();
    query_candidate(&backend, CAPTURE_GROUP, &capture_variant);
    query_candidate(&backend, UPDATE_GROUP, &update_variant);
    backend.freeze_program_catalog().unwrap();

    // A damaged, separately admitted object is rejected before touching the frozen
    // catalog. The existing built-in route remains selected and launchable as fallback.
    let corrupt_module = cache
        .join("v1")
        .join("objects")
        .join(corrupt.content_digest())
        .join(&manifest.modules()[0].file);
    let mut damaged = fs::read(&corrupt_module).unwrap();
    damaged[16] ^= 0xff;
    fs::write(&corrupt_module, damaged).unwrap();
    assert!(
        super::registry::install_for_test(
            &installer,
            corrupt.install_digest(),
            &runtime,
            &authorities,
        )
        .is_err()
    );
    assert!(
        !unsafe { launch_bound(CAPTURE_GROUP, &values(output_pointer, 7)) }.unwrap()
    );

    backend
        .bind_program_choice(CAPTURE_GROUP, &capture_variant)
        .unwrap();
    assert_eq!(
        unsafe { imparo_cuda_program_test_write_u32(output_pointer, 0) },
        0
    );
    assert_eq!(unsafe { imparo_cuda_program_test_graph_begin() }, 0);
    assert!(
        unsafe { launch_bound(CAPTURE_GROUP, &values(output_pointer, 0x1234_5678)) }
            .unwrap()
    );
    assert_eq!(unsafe { imparo_cuda_program_test_graph_end_replay(1) }, 0);
    assert_eq!(unsafe { imparo_cuda_program_test_graph_alive() }, 1);
    assert_eq!(read(output_pointer), 0x1234_5678);

    // Route mutation invalidates the capture-only graph before the next provider bind.
    backend
        .bind_program_choice(UPDATE_GROUP, &update_variant)
        .unwrap();
    assert_eq!(unsafe { imparo_cuda_program_test_graph_alive() }, 0);

    // Output is capture-static. Source 1 is the adapter-owned decode_start_pos_u32;
    // the second replay must patch the scalar through the production node scanner.
    assert_eq!(
        unsafe { imparo_cuda_program_test_write_u32(output_pointer, 0) },
        0
    );
    assert_eq!(unsafe { imparo_cuda_program_test_graph_begin() }, 0);
    assert!(
        unsafe { launch_bound(UPDATE_GROUP, &values(output_pointer, 0x1111)) }.unwrap()
    );
    assert_eq!(unsafe { imparo_cuda_program_test_graph_end_replay(1) }, 0);
    assert_eq!(unsafe { imparo_cuda_program_test_graph_alive() }, 1);
    assert_eq!(read(output_pointer), 0x1111);
    assert_eq!(unsafe { imparo_cuda_program_test_graph_replay(0x2222) }, 0);
    assert_eq!(read(output_pointer), 0x2222);

    // Leave the updated graph alive: reset must destroy it before unloading cubin.
    shutdown().unwrap();
    assert_eq!(unsafe { imparo_cuda_program_test_graph_alive() }, 0);
    assert_eq!(unsafe { imparo_cuda_program_test_free(output_pointer) }, 0);
}
