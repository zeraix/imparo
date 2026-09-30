//! CUDA's versioned correctness-receipt policy.
//!
//! This module does no GPU work and never trusts a stored tuning merely because it
//! parsed. It derives one complete `model.forward` route from the loader's single-read
//! candidate, the backend-owned knob registry, and a caller-supplied live runtime/model
//! identity. Any unproved value fails closed before a receipt can be created.

use std::collections::BTreeSet;
use std::ffi::OsString;
use std::fmt::Write as _;

use imparo_backend::numerical::{NumericalClass, RouteKey};
use imparo_backend::{BackendKnobs as _, SweepKind};
use imparo_host::Stored;
use imparo_host::correctness::{
    CorrectnessFingerprint, CorrectnessReceipt, ExpectedCorrectness, GateEvidence,
    GateRequirement, OracleFingerprint, ReceiptRoute,
};
use imparo_host::receipted_config::{UntrustedStoredConfig, config_sha256};

use crate::knobs::CUDA_KNOBS;
use crate::{CUDA_BACKEND_ABI, CudaBackend, CudaRuntimeIdentity};

/// Read current-owner F32 values for a fixed correctness probe, observing both
/// the transfer and synchronization result. This static-only surface does not
/// change the dynamic backend ABI or the ordinary inference readback path.
#[cfg(feature = "cuda-speculative")]
pub fn read_f32_checked(
    id: imparo_backend::BufId,
    offset: u64,
    output: &mut [f32],
) -> Result<(), i32> {
    let rc = unsafe {
        crate::ffi::imparo_cuda_gate_transfer_f32(
            id as u32,
            offset,
            output.as_mut_ptr(),
            output.len() as u64,
            0,
        )
    };
    if rc == 0 { Ok(()) } else { Err(rc) }
}

/// Upload a fixed correctness input through the same owner and stream. The
/// native upload arm reads these bytes and synchronizes before returning.
#[cfg(feature = "cuda-speculative")]
pub fn write_f32_checked(
    id: imparo_backend::BufId,
    offset: u64,
    input: &[f32],
) -> Result<(), i32> {
    let rc = unsafe {
        crate::ffi::imparo_cuda_gate_transfer_f32(
            id as u32,
            offset,
            input.as_ptr().cast_mut(),
            input.len() as u64,
            1,
        )
    };
    if rc == 0 { Ok(()) } else { Err(rc) }
}

/// Numerical selector policy, deliberately independent of CUDA's tuning-space version.
pub const CUDA_SELECTOR_VERSION: u32 = 5;
pub const CUDA_GATE_SUITE: &str = "cuda-llama-fa-q4_0";
pub const CUDA_Q8_GATE_SUITE: &str = "cuda-llama-fa-q8_0";
pub const CUDA_GATE_SUITE_VERSION: u32 = 8;
pub const CUDA_Q8_GATE_SUITE_VERSION: u32 = 2;
pub const CUDA_ROUTE_IMPLEMENTATION_VERSION: u32 = 5;

// The owner policy binds logical M1/M3 to one frozen physical-M3 FFN family.
// These are additional requirements, not evidence manufactured by the template.
const W4A16_FFN_GATES: &[&str] = &[
    "w4a16_cross_engine_difference_report",
    "w4a16_full_ffn_fp32_reference",
    "w4a16_ground_truth_contexts",
    "w4a16_mtp_on_off_ids_n512_s256",
    "w4a16_mtp_on_off_ids_n6144_s256",
    "w4a16_mtp_on_off_ids_n16384_s256",
    "w4a16_repeated_request_layout_restore",
    "w4a16_loaded_module_identity",
];
const LFM_RETAINED_GATES: &[&str] = &[
    "lfm_retained_dspark_on_off_n512_s256",
    "lfm_retained_dspark_on_off_n6144_s256",
    "lfm_retained_dspark_on_off_n16384_s256",
    "lfm_retained_owner_reuse_isolation",
    "lfm_retained_domain_routes_and_graph",
];
const LFM_RETAINED_KNOBS: &[(&str, u32)] = &[
    ("mmq_q8_canonical_gate_up_pair", 3),
    ("mmq_q8_canonical_load_lanes", 2),
    ("prefill_projection_q8_d4", 3),
    ("rms_norm_add", 1),
    ("mmq_q8_aligned_whole_k", 2),
    ("mmvq_q8_tm_decode_silu_pair", 1),
    ("shortconv_decode_fused", 1),
    ("decode_graph_q8_producer_reuse", 1),
    ("prefill_bcx_shortconv_ready", 0),
    ("attn_d64_mma_prefill", 2),
    ("finite_history_prefill_tail_rows", 64),
    ("e4b_ffn_w4a16", 0),
    ("e4b_retained_decode_policy", 0),
];
fn validate_lfm_retained(
    stored: &Stored,
    identity: &CudaCorrectnessIdentity<'_>,
) -> Result<bool, String> {
    let selected = stored
        .knobs
        .iter()
        .any(|(n, v)| n == crate::knobs::LFM_RETAINED_EXECUTION_KNOB && *v == 1);
    if !selected {
        return Ok(false);
    }
    if identity.runtime.device_sm != 86
        || identity.kv_k != "q8_0"
        || identity.kv_v != "q8_0"
        || !matches!(stored.batch, Some(512 | 1920))
        || LFM_RETAINED_KNOBS.iter().any(|(name, value)| {
            !stored.knobs.iter().any(|(n, v)| n == name && v == value)
        })
    {
        return Err("retained LFM policy requires its complete SM86/Q8 registered configuration".into());
    }
    Ok(true)
}

const W4A16_CUBIN_SHA256: &str =
    "9521406b46b918c23450e6199ed973149c88fcb2e33e5a7d8348212f7694cf37";

pub const CUDA_PTQ_F16_GATE_SUITE: &str = "cuda-ptq-f16-independent";
pub const CUDA_PTQ_F16_REQUIRED_GATES: &[(&str, u32)] = &[
    ("ptq_f16_independent_math", 2),
    ("ptq_f16_model_nonregression", 2),
    ("ptq_f16_original_failures_report", 2),
    ("ptq_f16_reset_split_state", 2),
    ("ptq_f16_three_length_routes", 2),
    ("ptq_f16_host_config_application", 2),
];
// The independent FP64 oracle is unchanged; execution evidence is versioned
// separately for bounded GEMM plus registered-host transfer.
pub const CUDA_PTQ_F16_V3_REQUIRED_GATES: &[(&str, u32)] = &[
    ("ptq_f16_independent_math", 3),
    ("ptq_f16_model_nonregression", 3),
    ("ptq_f16_original_failures_report", 3),
    ("ptq_f16_reset_split_state", 3),
    ("ptq_f16_three_length_routes", 3),
    ("ptq_f16_host_config_application", 3),
    ("ptq_f16_transfer_state", 3),
    ("ptq_f16_bounded_provider", 3),
];
const PTQ_F16_ORACLE_ARGUMENTS: &[&str] = &[
    "--contract",
    "ptq-f16-v2",
    "--quality",
    "independent-numerics-plus-nonregression",
];
const PTQ_F16_ORACLE_MANIFEST_SHA256: &str =
    "6af8b54178a50772c8084170dfeb1444fdb10e4ef810b5cd15a1a8da5935d1e0";

// Admission is for the complete tested configuration. The model/plan/content
// identity is still bound by the existing receipt; a shape or flag is not proof.
fn validate_ptq_f16(
    stored: &Stored,
    identity: &CudaCorrectnessIdentity<'_>,
) -> Result<(), String> {
    let selected = stored
        .knobs
        .iter()
        .find(|(n, _)| n == crate::knobs::PTQ_PREFILL_TENSORCORE_KNOB)
        .map_or(0, |(_, v)| *v);
    let f16 = identity.kv_k == "f16" && identity.kv_v == "f16";
    if !f16 {
        if selected != 0 {
            return Err("PTQ Tensor Core cannot borrow Q4/Q8 numerical receipts".into());
        }
        return Ok(());
    }
    let value = |name: &str| {
        stored
            .knobs
            .iter()
            .find(|(n, _)| n == name)
            .map_or(u32::MAX, |(_, v)| *v)
    };
    if identity.runtime.device_sm != 86
        || stored.batch != Some(128)
        || !matches!(
            (selected, value(crate::knobs::WEIGHT_TRANSFER_POLICY_KNOB)),
            (2, 0) | (3, 1)
        )
        || value("attn_decode_specialized") != 2
        || value("attn_d256_tiled") != 2
        || value(crate::knobs::FINITE_HISTORY_KNOB) != 0
        || value(crate::knobs::E4B_FFN_W4A16_KNOB) != 0
        || value(crate::knobs::E4B_RETAINED_DECODE_POLICY_KNOB) != 0
        || value(crate::knobs::LFM_RETAINED_EXECUTION_KNOB) != 0
    {
        return Err("PTQ/F16 requires SM86 batch128 GQA6-2 and an admitted (TC,transfer) pair: (2,0) or (3,1)".into());
    }
    Ok(())
}

const ROUTE_DOMAIN_VERSION: u32 = 2;
const ROUTE_PARAMETERS_VERSION: u32 = 1;
const ORACLE_OPTIONS_VERSION: u32 = 2;
const ORACLE_IMPLEMENTATION: &str = "zeraix/llama-cpp";
const ORACLE_REVISION: &str = "4695f001fece1660d8bb1b3748f50726ddcc100b";
const ORACLE_BUNDLE_MANIFEST_SHA256: &str =
    "69388d9d5f910d26b8307b5f510449ea7ec717b8891decd81b918300180eaab0";
const Q8_ORACLE_BUNDLE_MANIFEST_SHA256: &str =
    "6f41c93b94995b0d0e33d7dc25a9cad192a2438d0f0c42cd921ed8c5d42bd789";
const ORACLE_ARGUMENTS: &[&str] =
    &["-fa", "on", "-ctxcp", "0", "-ctk", "q4_0", "-ctv", "q4_0"];
const Q8_ORACLE_ARGUMENTS: &[&str] =
    &["-fa", "on", "-ctxcp", "0", "-ctk", "q8_0", "-ctv", "q8_0"];

/// Order is part of the versioned gate contract and must match `seal_receipt.py`.
pub const CUDA_REQUIRED_GATES: &[(&str, u32)] = &[
    ("logit_agree_n128_q4_0", 1),
    ("logit_agree_n449_q4_0", 1),
    ("logit_agree_n512_q4_0", 1),
    ("decode_agree_n128_s8_q4_0", 2),
    ("decode_agree_n512_s8_q4_0", 2),
    ("decode_agree_n2000_s8_q4_0", 2),
    ("prefill_graph_replay_agree_n128", 2),
];

pub const CUDA_Q8_REQUIRED_GATES: &[(&str, u32)] = &[
    ("logit_agree_n128_q8_0", 1),
    ("logit_agree_n449_q8_0", 1),
    ("logit_agree_n512_q8_0", 1),
    ("decode_agree_n128_s8_q8_0", 2),
    ("decode_agree_n512_s8_q8_0", 2),
    ("decode_agree_n2000_s8_q8_0", 2),
];

#[derive(Clone, Copy)]
struct GatePolicy {
    suite: &'static str,
    version: u32,
    gates: &'static [(&'static str, u32)],
    oracle_arguments: &'static [&'static str],
    oracle_implementation: &'static str,
    oracle_revision: &'static str,
    oracle_options_domain: &'static str,
    oracle_bundle_manifest_sha256: &'static str,
}

const ALLOWED_CUDA_ENV: &[&str] = &[
    "IMPARO_CUDA_BACKEND",
    "IMPARO_CUDA_DEVICE",
    "IMPARO_CUDA_SM",
    "IMPARO_CUDA_ARCHS",
    "IMPARO_CUDA_RESERVE_MIB",
    "IMPARO_CUDA_WEIGHT_CACHE_MIB",
    // Laboratory Graph capture is a scheduling wrapper around the already
    // receipted exact-key DAG: it may neither select a different kernel nor
    // widen the token/start/output key. Its capture-vs-ordinary byte-equality
    // gate is therefore separate from numerical-route tuning. All CUDA kernel
    // override environments remain rejected below.
    "IMPARO_CUDA_PREFILL_GRAPH_LAB",
    // Observability only; it prints capture/replay decisions and selects no work.
    "IMPARO_CUDA_TRACE_GRAPH",
    // Observability only; it prints the selected D64 vector schedule.
    "IMPARO_CUDA_ATTN_D64_VEC_TRACE",
    "IMPARO_CUDA_ATTN_D64_MMA_TRACE",
    // CUDA-event profiling changes synchronization and diagnostics only. It must be
    // able to observe the exact receipted route; rejecting the config here silently
    // profiles safe defaults instead of the selected kernels.
    "IMPARO_CUDA_PROFILE_FORWARD",
    "IMPARO_CUDA_PROFILE_MATMUL",
    "IMPARO_CUDA_PROFILE_OPS",
    "IMPARO_CUDA_PROFILE_Q8",
    "IMPARO_CUDA_PHASE_A1_PREFILL_WALL",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CudaMathMode {
    Fast,
    Precise,
}

impl CudaMathMode {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Precise => "precise",
        }
    }
}

/// Runtime/model facts which cannot be inferred from a host-config file.
#[derive(Clone, Copy, Debug)]
pub struct CudaCorrectnessIdentity<'a> {
    pub runtime: &'a CudaRuntimeIdentity,
    pub model_sha256: [u8; 32],
    pub model_plan_sha256: [u8; 32],
    pub kv_layout_sha256: [u8; 32],
    pub kv_k: &'a str,
    pub kv_v: &'a str,
    pub math_mode: CudaMathMode,
}

/// Construct the exact CUDA correctness contract for the loader's one-read candidate.
///
/// The process environment is checked here, before any candidate can receive authority.
/// CUDA diagnostic route overrides are intentionally not representable in a receipt.
pub fn expected_correctness(
    candidate: &UntrustedStoredConfig<'_>,
    identity: &CudaCorrectnessIdentity<'_>,
) -> Result<ExpectedCorrectness, String> {
    expected_correctness_with_env(
        candidate.exact_bytes(),
        candidate.stored(),
        identity,
        std::env::vars_os(),
    )
}

/// Produce the unsigned receipt emitted by `--correctness-template`.
///
/// Only the external fixed-gate sealer may turn the placeholders into passing evidence
/// and set producer/timestamp metadata.
#[must_use]
pub fn receipt_skeleton(expected: &ExpectedCorrectness) -> CorrectnessReceipt {
    CorrectnessReceipt {
        schema_version: imparo_host::correctness::CORRECTNESS_RECEIPT_SCHEMA_VERSION,
        producer: String::new(),
        producer_version: 0,
        issued_unix_seconds: 0,
        fingerprint: expected.fingerprint.clone(),
        gate_suite: expected.gate_suite.clone(),
        gate_suite_version: expected.gate_suite_version,
        routes: expected.routes.iter().map(ReceiptRoute::from).collect(),
        gates: expected
            .required_gates
            .iter()
            .map(|gate| GateEvidence {
                gate_id: gate.gate_id.clone(),
                gate_version: gate.gate_version,
                passed: false,
                command_sha256: String::new(),
                output_sha256: String::new(),
            })
            .collect(),
        oracle: expected.oracle.clone(),
    }
}

fn expected_correctness_with_env(
    exact_bytes: &[u8],
    stored: &Stored,
    identity: &CudaCorrectnessIdentity<'_>,
    environment: impl IntoIterator<Item = (OsString, OsString)>,
) -> Result<ExpectedCorrectness, String> {
    validate_environment(environment)?;
    validate_identity(identity)?;
    let mut policy = gate_policy(identity)?;
    let values = validate_candidate(stored)?;
    validate_ptq_f16(stored, identity)?;
    if identity.kv_k == "f16"
        && stored
            .knobs
            .iter()
            .any(|(n, v)| n == crate::knobs::PTQ_PREFILL_TENSORCORE_KNOB && *v == 3)
    {
        policy.version = 3;
        policy.gates = CUDA_PTQ_F16_V3_REQUIRED_GATES;
    }
    let lfm_retained = validate_lfm_retained(stored, identity)?;
    let lfm_short_domain = lfm_retained && stored.batch == Some(512);
    let fa2 = stored.knobs.iter().any(|(name, value)| {
        name == crate::knobs::D64_ATTENTION_KNOB && *value == crate::knobs::D64_FA2_MODE
    });
    if fa2
        && (identity.runtime.device_sm != 86
            || identity.kv_k != "q8_0"
            || identity.kv_v != "q8_0")
    {
        return Err("FA2 mode requires the qualified SM86 Q8 target route".into());
    }
    let finite_history = stored
        .knobs
        .iter()
        .any(|(name, value)| name == crate::knobs::FINITE_HISTORY_KNOB && *value != 0);
    let w4a16 = stored
        .knobs
        .iter()
        .any(|(name, value)| name == "e4b_ffn_w4a16" && matches!(*value, 1 | 2));
    if w4a16
        && (identity.runtime.device_sm != 86
            || identity.kv_k != "q4_0"
            || identity.kv_v != "q4_0")
    {
        return Err("W4A16 FFN requires the qualified SM86 Q4 target route".into());
    }
    let retained = stored.knobs.iter().any(|(name, value)| {
        name == crate::knobs::E4B_RETAINED_DECODE_POLICY_KNOB && *value == 1
    });
    if retained
        && (!w4a16
            || !stored
                .knobs
                .iter()
                .any(|(name, value)| name == "attn_d512_mma" && *value == 3))
    {
        return Err("retained E4B policy requires the complete W4A16/attn_d512_mma=3 combination".into());
    }
    // A state-changing workflow needs a dedicated route-hit/lifecycle witness.
    // Existing six numerical logs cannot seal this new route by themselves.
    let mut suite = if finite_history {
        format!("{}-finite-history", policy.suite)
    } else {
        policy.suite.to_string()
    };
    let suite_version = if lfm_short_domain {
        // Runtime admission binds this batch/layout to capacity1024. Its natural
        // near-capacity gate and explicit rejection proof require a new contract.
        3
    } else if lfm_retained {
        // Retained owners keep finite64 frozen. Their state proof compares the
        // same prepared policy with an explicit full-history diagnostic control;
        // it is distinct from legacy finite64/finite0 configuration evidence.
        2
    } else if w4a16 {
        // Independent numerical/ground-truth quality; cross-engine deltas are reports.
        3
    } else if finite_history || fa2 {
        1
    } else {
        policy.version
    };
    if fa2 {
        suite.push_str("-d64-fa2");
    }
    if w4a16 {
        suite.push_str("-w4a16-unified-ffn");
    }
    if lfm_retained {
        suite.push_str("-lfm-retained-v1");
    }
    let mut required_gates: Vec<GateRequirement> = policy
        .gates
        .iter()
        .map(|(id, version)| {
            let id = if lfm_short_domain && *id == "decode_agree_n2000_s8_q8_0" {
                "decode_agree_n1000_s8_q8_0"
            } else {
                *id
            };
            GateRequirement {
                gate_id: id.into(),
                gate_version: *version,
            }
        })
        .collect();
    if w4a16 {
        // Keep Graph identity as a hard invariant. Cross-engine agreement is
        // replaced by an explicit report plus independent quality evidence below.
        required_gates = vec![GateRequirement {
            gate_id: "prefill_graph_replay_agree_n128".into(),
            gate_version: 2,
        }];
    }
    if finite_history {
        required_gates.push(GateRequirement {
            gate_id: if lfm_retained {
                "lfm_retained_finite_history_state_reuse"
            } else {
                "finite_history_state_reuse"
            }
            .into(),
            gate_version: 1,
        });
    }
    if fa2 {
        // Existing short fixed gates do not cover the selected long-KV routes.
        // The sealer must gain real route/state evidence before admitting this suite.
        for id in ["fa2_d64_prefill_n512_k4096", "fa2_d64_dspark_m9_n4096"] {
            required_gates.push(GateRequirement {
                gate_id: id.into(),
                gate_version: 1,
            });
        }
    }
    if w4a16 {
        required_gates.extend(W4A16_FFN_GATES.iter().map(|id| GateRequirement {
            gate_id: (*id).into(),
            gate_version: if id.starts_with("w4a16_mtp_")
                || *id == "w4a16_repeated_request_layout_restore"
                || *id == "w4a16_loaded_module_identity"
            {
                2
            } else {
                1
            },
        }));
    }
    if lfm_retained {
        required_gates.extend(LFM_RETAINED_GATES.iter().map(|id| GateRequirement {
            gate_id: (*id).into(),
            gate_version: 1,
        }));
    }
    if lfm_short_domain {
        required_gates.push(GateRequirement {
            gate_id: "lfm_retained_short_domain_bounds".into(),
            gate_version: 1,
        });
    }
    let space_version = CudaBackend.space_version();
    if CUDA_SELECTOR_VERSION == space_version {
        return Err("CUDA selector version must be independent of tuning space".into());
    }

    let platform = format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH);
    let backend_fingerprint = identity.runtime.backend_fingerprint_sha256();
    let model_sha256 = encode_hex(&identity.model_sha256);
    let model_plan_sha256 = encode_hex(&identity.model_plan_sha256);
    let kv_layout_sha256 = encode_hex(&identity.kv_layout_sha256);
    let math_mode = identity.math_mode.as_str();
    let fingerprint = CorrectnessFingerprint {
        config_sha256: config_sha256(exact_bytes),
        model_sha256,
        model_plan_sha256,
        kv_layout_sha256,
        platform: platform.clone(),
        backend: "cuda".into(),
        device_uuid: identity.runtime.device_uuid_string(),
        device_sm: identity.runtime.device_sm,
        driver_version: identity.runtime.driver_version,
        runtime_version: identity.runtime.runtime_version,
        backend_abi: identity.runtime.backend_abi,
        backend_fingerprint_sha256: encode_hex(&backend_fingerprint),
        kv_k: identity.kv_k.into(),
        kv_v: identity.kv_v.into(),
        numerical_space_version: space_version,
        selector_version: CUDA_SELECTOR_VERSION,
        math_mode: math_mode.into(),
    };
    let route = RouteKey {
        backend: "cuda".into(),
        operation: "model.forward".into(),
        implementation: "imparo-cuda.model-forward".into(),
        implementation_version: CUDA_ROUTE_IMPLEMENTATION_VERSION,
        selector_version: CUDA_SELECTOR_VERSION,
        domain_sha256: route_domain_sha256(
            &platform,
            identity,
            space_version,
            &backend_fingerprint,
        ),
        parameters_sha256: route_parameters_sha256(exact_bytes, stored, &values),
        numerical_class: NumericalClass::GateBounded {
            gate_suite: suite.clone(),
            contract_version: suite_version,
        },
    };
    route
        .validate()
        .map_err(|error| format!("invalid CUDA numerical route: {error:?}"))?;
    Ok(ExpectedCorrectness {
        fingerprint,
        gate_suite: suite,
        gate_suite_version: suite_version,
        routes: vec![route],
        required_gates,
        oracle: OracleFingerprint {
            implementation: policy.oracle_implementation.into(),
            revision: policy.oracle_revision.into(),
            options_sha256: oracle_options_sha256_in_domain(
                policy.oracle_options_domain,
                policy.oracle_arguments,
            ),
            bundle_manifest_sha256: policy.oracle_bundle_manifest_sha256.into(),
        },
    })
}

pub(crate) fn validate_environment(
    environment: impl IntoIterator<Item = (OsString, OsString)>,
) -> Result<(), String> {
    for (key, _) in environment {
        let key = key.to_string_lossy();
        if (key.starts_with("IMPARO_CUDA_")
            && !ALLOWED_CUDA_ENV.iter().any(|allowed| key == *allowed))
            || matches!(
                key.as_ref(),
                "IMPARO_LAB_D64_FA2" | "IMPARO_LAB_D64_FA2_PREFILL"
                    // The old receipt has no owner-policy identity or on/off ID gate.
                    | "IMPARO_LAB_BATCH_INVARIANT_Q8_V1"
                    // Retained LFM choices still live outside StoredConfig.
                    // Reject their presence until a registered policy binds both
                    // target/draft execution and the complete lifecycle gates.
                    | "IMPARO_LAB_CAPTURE_AWARE_PREFILL_TAIL"
                    | "IMPARO_LAB_D64_DIRECT_MMA"
                    | "IMPARO_LAB_D64_FA2_PREFILL_1024"
                    | "IMPARO_LAB_D64_FIXED_PARTITION"
                    | "IMPARO_LAB_D64_KEYTILE16"
                    | "IMPARO_LAB_D64_PARALLEL_QK"
                    | "IMPARO_LAB_D64_PARALLEL_QK_GRAPH"
                    | "IMPARO_LAB_D64_PREFIX_STATE"
                    | "IMPARO_LAB_D64_PV_QUERYWARP"
                    | "IMPARO_LAB_DRAFT_BOUNDARY_RESUME"
                    | "IMPARO_LAB_DRAFT_BOUNDED_TAIL"
                    | "IMPARO_LAB_DRAFT_CACHE_RESUME"
                    | "IMPARO_LAB_DSPARK_COLD_OWNER_REUSE"
                    | "IMPARO_LAB_DSPARK_FRONTIER16"
                    | "IMPARO_LAB_DSPARK_TREE16"
                    | "IMPARO_LAB_DSPARK_CHAIN_VERIFY"
                    | "IMPARO_LAB_VERIFY_M1_QUANT"
                    | "IMPARO_LAB_LFM_TREE_GRAPH"
                    | "IMPARO_LAB_Q8_CANONICAL_LIVE2"
                    | "IMPARO_LAB_Q8_DOWN_TM_ASYNC"
                    | "IMPARO_LAB_Q8_M16_ROW_OWNER"
                    | "IMPARO_LAB_Q8_M9_K3"
                    | "IMPARO_LAB_Q8_M9_ROW_OWNER"
                    | "IMPARO_LAB_Q8_TM_GATE_UP_CANONICAL_DOWN"
                    | "IMPARO_LAB_RETAIN_ACTIVATION_CAPACITY"
                    | "IMPARO_LAB_TREE_CANONICAL_PARTITION"
                    | "IMPARO_LAB_TREE_TAIL_REPAIR"
                    | "IMPARO_LAB_TREE_TAIL_SHARED_KV"
                    | "IMPARO_LAB_E4B_W4A16_FFN"
                    | "IMPARO_LAB_E4B_W4A16_CUBIN"
                    | "IMPARO_STATE_DEMAND_LAB"
                    | "IMPARO_OUTPUT_REFERENCE_LAB"
                    | "IMPARO_KV_HISTORY_LAB"
                    | "IMPARO_LAB_Q8_SHORT_K_ROWS"
                    | "IMPARO_LAB_DEVICE_GREEDY_VERIFY"
                    | "IMPARO_LAB_MTP_MIN_PROB"
                    | "IMPARO_LAB_SPEC_ACTIVATION_CAP"
                    | "IMPARO_LAB_MTP_M1_GRAPH"
                    | "IMPARO_LAB_KNOBS"
                    | "IMPARO_DSPARK_KNOBS"
                    | "IMPARO_LAB_VERIFY_GRAPH_UPDATE"
                    | "IMPARO_LAB_DEFER_KV_MATERIALIZATION"
                    | "IMPARO_LAB_MTP_CACHED_GRAPH"
                    | "IMPARO_LAB_Q4_FULL_TILE_ROWS"
                    | "IMPARO_LAB_VERIFY_GRAPH_GROWING_SPAN"
                    | "IMPARO_LAB_VERIFY_COMMON_PARTITION"
                    | "IMPARO_LAB_MTP_CLUSTER_HEAD_DIR"
                    | "IMPARO_LAB_D512_M3_SHARED_DEPACK"
                    | "IMPARO_LAB_MTP_FUSED_D512"
                    | "IMPARO_LAB_D512_REGISTER_FRAGMENTS"
                    | "IMPARO_LAB_D256_PARALLEL_COMBINE"
                    | "IMPARO_LAB_SHORT_PV_REGISTER_FRAGMENTS"
            )
        {
            return Err(format!(
                "CUDA diagnostic environment override is not receiptable: {key}"
            ));
        }
    }
    Ok(())
}

fn validate_identity(identity: &CudaCorrectnessIdentity<'_>) -> Result<(), String> {
    let runtime = identity.runtime;
    if runtime.device_uuid == [0; 16]
        || runtime.device_sm < 50
        || runtime.driver_version == 0
        || runtime.runtime_version == 0
        || runtime.backend_abi != CUDA_BACKEND_ABI
        || runtime.backend_build_sha256 == [0; 32]
        || runtime.backend_artifact_sha256 == Some([0; 32])
    {
        return Err("CUDA runtime correctness identity is incomplete".into());
    }
    if identity.model_sha256 == [0; 32]
        || identity.model_plan_sha256 == [0; 32]
        || identity.kv_layout_sha256 == [0; 32]
    {
        return Err(
            "model, model-plan, or KV-layout correctness identity is zero".into(),
        );
    }
    gate_policy(identity)?;
    Ok(())
}

fn gate_policy(identity: &CudaCorrectnessIdentity<'_>) -> Result<GatePolicy, String> {
    match (identity.kv_k, identity.kv_v) {
        ("q4_0", "q4_0") => Ok(GatePolicy {
            suite: CUDA_GATE_SUITE,
            version: CUDA_GATE_SUITE_VERSION,
            gates: CUDA_REQUIRED_GATES,
            oracle_arguments: ORACLE_ARGUMENTS,
            oracle_implementation: ORACLE_IMPLEMENTATION,
            oracle_revision: ORACLE_REVISION,
            oracle_options_domain: "imparo-cuda-llama-oracle-options",
            oracle_bundle_manifest_sha256: ORACLE_BUNDLE_MANIFEST_SHA256,
        }),
        ("q8_0", "q8_0") => Ok(GatePolicy {
            suite: CUDA_Q8_GATE_SUITE,
            version: CUDA_Q8_GATE_SUITE_VERSION,
            gates: CUDA_Q8_REQUIRED_GATES,
            oracle_arguments: Q8_ORACLE_ARGUMENTS,
            oracle_implementation: ORACLE_IMPLEMENTATION,
            oracle_revision: ORACLE_REVISION,
            oracle_options_domain: "imparo-cuda-llama-oracle-options",
            oracle_bundle_manifest_sha256: Q8_ORACLE_BUNDLE_MANIFEST_SHA256,
        }),
        ("f16", "f16") => Ok(GatePolicy {
            suite: CUDA_PTQ_F16_GATE_SUITE,
            version: 2,
            gates: CUDA_PTQ_F16_REQUIRED_GATES,
            oracle_arguments: PTQ_F16_ORACLE_ARGUMENTS,
            oracle_implementation: "imparo/ptq-f16-independent",
            oracle_revision: "v2",
            oracle_options_domain: "imparo-cuda-independent-oracle-options",
            oracle_bundle_manifest_sha256: PTQ_F16_ORACLE_MANIFEST_SHA256,
        }),
        (kv_k, kv_v) => Err(format!(
            "CUDA correctness has no gate suite for K/V {kv_k}/{kv_v}"
        )),
    }
}

/// Return registry-order values only after proving the persisted list is an exact
/// permutation-free image of the current registry.
fn validate_candidate(stored: &Stored) -> Result<Vec<u32>, String> {
    let batch = stored.batch.filter(|batch| *batch > 0).ok_or_else(|| {
        "CUDA correctness candidate has no positive batch".to_string()
    })?;
    u64::try_from(batch).map_err(|_| "CUDA batch does not fit u64".to_string())?;
    if stored.knobs.len() != CUDA_KNOBS.len() {
        return Err(format!(
            "CUDA correctness candidate has {} knobs; registry requires {}",
            stored.knobs.len(),
            CUDA_KNOBS.len()
        ));
    }
    let mut names = BTreeSet::new();
    let mut values = Vec::with_capacity(CUDA_KNOBS.len());
    for ((actual_name, value), declaration) in stored.knobs.iter().zip(CUDA_KNOBS) {
        if !names.insert(actual_name.as_str()) {
            return Err(format!("duplicate CUDA knob: {actual_name}"));
        }
        if actual_name != declaration.name {
            return Err(format!(
                "CUDA knob order/name mismatch: expected {}, got {actual_name}",
                declaration.name
            ));
        }
        if declaration.legal.is_some()
            || declaration.derive.is_some()
            || declaration.candidates.is_some()
        {
            return Err(format!(
                "CUDA knob {} needs an explicit correctness-policy validator",
                declaration.name
            ));
        }
        if !declared_value(declaration.sweep, declaration.values, *value) {
            return Err(format!(
                "illegal CUDA knob value: {}={value}",
                declaration.name
            ));
        }
        values.push(*value);
    }
    Ok(values)
}

fn declared_value(sweep: SweepKind, values: &[u32], value: u32) -> bool {
    match sweep {
        SweepKind::Values | SweepKind::External => values.contains(&value),
        SweepKind::Crossing { ladder, hi, lo }
        | SweepKind::TokenMinCrossing { ladder, hi, lo }
        | SweepKind::TokenMaxCrossing { ladder, hi, lo }
        | SweepKind::SpanCrossing { ladder, hi, lo }
        | SweepKind::RowsCrossing { ladder, hi, lo } => {
            value == hi || value == lo || ladder.contains(&value)
        }
        SweepKind::Derived => false,
    }
}

fn route_domain_sha256(
    platform: &str,
    identity: &CudaCorrectnessIdentity<'_>,
    space_version: u32,
    backend_fingerprint: &[u8; 32],
) -> String {
    let mut canonical = Canonical::new("imparo-cuda-model-forward-domain");
    canonical.u32("version", ROUTE_DOMAIN_VERSION);
    canonical.string("platform", platform);
    canonical.bytes("model_sha256", &identity.model_sha256);
    canonical.bytes("model_plan_sha256", &identity.model_plan_sha256);
    canonical.bytes("kv_layout_sha256", &identity.kv_layout_sha256);
    canonical.bytes("device_uuid", &identity.runtime.device_uuid);
    canonical.u32("device_sm", identity.runtime.device_sm);
    canonical.u32("driver_version", identity.runtime.driver_version);
    canonical.u32("runtime_version", identity.runtime.runtime_version);
    canonical.u32("backend_abi", identity.runtime.backend_abi);
    canonical.bytes(
        "backend_build_sha256",
        &identity.runtime.backend_build_sha256,
    );
    match identity.runtime.backend_artifact_sha256 {
        Some(value) => {
            canonical.u32("has_backend_artifact", 1);
            canonical.bytes("backend_artifact_sha256", &value);
        }
        None => canonical.u32("has_backend_artifact", 0),
    }
    canonical.bytes("backend_fingerprint_sha256", backend_fingerprint);
    canonical.string("kv_k", identity.kv_k);
    canonical.string("kv_v", identity.kv_v);
    canonical.string("math_mode", identity.math_mode.as_str());
    canonical.u32("numerical_space_version", space_version);
    canonical.u32("selector_version", CUDA_SELECTOR_VERSION);
    canonical.u64("registry_len", CUDA_KNOBS.len() as u64);
    for declaration in CUDA_KNOBS {
        canonical.string("registry_knob", declaration.name);
    }
    canonical.finish()
}

fn route_parameters_sha256(
    exact_bytes: &[u8],
    stored: &Stored,
    values: &[u32],
) -> String {
    let mut canonical = Canonical::new("imparo-cuda-model-forward-parameters");
    canonical.u32("version", ROUTE_PARAMETERS_VERSION);
    canonical.bytes("exact_config", exact_bytes);
    canonical.u64("batch", stored.batch.expect("candidate validated") as u64);
    canonical.u64("registry_len", CUDA_KNOBS.len() as u64);
    for (declaration, value) in CUDA_KNOBS.iter().zip(values) {
        canonical.string("knob_name", declaration.name);
        canonical.u32("knob_value", *value);
    }
    if stored
        .knobs
        .iter()
        .any(|(name, value)| name == "e4b_ffn_w4a16" && matches!(*value, 1 | 2))
    {
        canonical.string("w4a16_module_sha256", W4A16_CUBIN_SHA256);
        canonical.string(
            "w4a16_weight_layout",
            "q4_0-to-marlin-g32-gu-gap-transpose8-v1",
        );
        canonical.string("w4a16_arithmetic", "fp16-u4b8-fp32-reduce-m3-v1");
        canonical.string(
            "w4a16_owner_policy",
            "single-resident-prepare-frozen-m1-m3-v1",
        );
        canonical.string("w4a16_domain", "sm86-sm30-h2560-f10240-l42-q4kv");
    }
    if stored
        .knobs
        .iter()
        .any(|(name, value)| name == "e4b_ffn_w4a16" && *value == 2)
    {
        canonical.string(
            "w4a16_prefill_lifecycle",
            "mode2:domain1=cold-canonical-first-decode-pack-warm-packed;domain2+domain3+unbound=canonical-packed-v1",
        );
        canonical.string(
            "w4a16_packed_prefill_reader",
            "marlin-g32-u4b8-direct-stage-original-q4xq8-dp4a-fp32-v1",
        );
    }
    if stored.knobs.iter().any(|(name, value)| {
        name == crate::knobs::E4B_RETAINED_DECODE_POLICY_KNOB && *value == 1
    }) {
        canonical.string("e4b_retained_policy", "model-owner-frozen-v1");
        canonical.string(
            "e4b_retained_domain",
            "sm86-sm30-q4kv-h2560-f10240-l42-h8-kv2-d256-d512",
        );
        canonical.string("e4b_retained_capacity_domains", "1024:common+growing+cluster+d256-mode1;6656:shared+fused+cluster+d256-mode2-shared+parallel-combine;16896:shared+fused+register-fragments+d256-mode2-shared+parallel-combine");
        canonical.string("e4b_retained_common", "output-reference;state-demand;kv-history;device-greedy;capacity3;graph3;defer-kv;cached-mtp;q4-fulltiles;q8-shortk;no-grouped;query-parallel;gqa2-d512;d256-vec;d512-mma+pair;mtp-prob0;m1-graph0");
        canonical.string(
            "e4b_cluster_centroids_sha256",
            "d293fc2fc2b68dea9716cc6cad81c4847084640d393962c637ef593415aa68c7",
        );
        canonical.string(
            "e4b_cluster_ordering_sha256",
            "2d4a619b6fdf687972daaf298bf5ce341f4e2f1d05c20de10c77684e20481b07",
        );
    }
    if stored
        .knobs
        .iter()
        .any(|(n, v)| n == crate::knobs::LFM_RETAINED_EXECUTION_KNOB && *v == 1)
    {
        canonical.string("lfm_retained_policy", "model-owner-frozen-v1");
        canonical.string(
            "lfm_retained_model",
            "sm86-sm30-q8kv-h2048-f10752-l30-h32-kv8-d64",
        );
        canonical.string("lfm_retained_domains", "1024:batch512-gu3-down3-async-owner-reuse-activation-reuse;6656:batch1920-gu3-down2-bounded-tail;16896:batch1920-gu3-down2-bounded-tail-treegraph-sharedkv-workspace96mib");
        canonical.string("lfm_retained_common", "target-v1;draft-legacy;frontier-tree16;postlayer-tail64;boundary-resume;draft-cache-resume;device-greedy;canonical-live2;m9-k3;m9-row-owner;q8-shortk;parallel-qk-graph;pv-querywarp");
    }
    canonical.finish()
}

#[must_use]
pub fn oracle_options_sha256() -> String {
    oracle_options_sha256_for(ORACLE_ARGUMENTS)
}

fn oracle_options_sha256_for(arguments: &[&str]) -> String {
    oracle_options_sha256_in_domain("imparo-cuda-llama-oracle-options", arguments)
}

fn oracle_options_sha256_in_domain(domain: &str, arguments: &[&str]) -> String {
    let mut canonical = Canonical::new(domain);
    canonical.u32("version", ORACLE_OPTIONS_VERSION);
    canonical.u64("argument_count", arguments.len() as u64);
    for argument in arguments {
        canonical.string("argument", argument);
    }
    canonical.finish()
}

fn encode_hex(bytes: &[u8]) -> String {
    bytes.iter().fold(
        String::with_capacity(bytes.len() * 2),
        |mut output, byte| {
            write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
            output
        },
    )
}

struct Canonical(Vec<u8>);

impl Canonical {
    fn new(domain: &str) -> Self {
        let mut output = Self(Vec::new());
        output.string("domain", domain);
        output
    }

    fn bytes(&mut self, name: &str, value: &[u8]) {
        self.0.extend_from_slice(&(name.len() as u64).to_le_bytes());
        self.0.extend_from_slice(name.as_bytes());
        self.0
            .extend_from_slice(&(value.len() as u64).to_le_bytes());
        self.0.extend_from_slice(value);
    }

    fn string(&mut self, name: &str, value: &str) {
        self.bytes(name, value.as_bytes());
    }

    fn u32(&mut self, name: &str, value: u32) {
        self.bytes(name, &value.to_le_bytes());
    }

    fn u64(&mut self, name: &str, value: u64) {
        self.bytes(name, &value.to_le_bytes());
    }

    fn finish(self) -> String {
        config_sha256(&self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runtime_fixture() -> CudaRuntimeIdentity {
        CudaRuntimeIdentity {
            device_uuid: [1; 16],
            device_sm: 86,
            driver_version: 12_080,
            runtime_version: 12_080,
            backend_abi: CUDA_BACKEND_ABI,
            backend_build_sha256: [2; 32],
            backend_artifact_sha256: Some([3; 32]),
        }
    }

    fn identity(runtime: &CudaRuntimeIdentity) -> CudaCorrectnessIdentity<'_> {
        CudaCorrectnessIdentity {
            runtime,
            model_sha256: [4; 32],
            model_plan_sha256: [5; 32],
            kv_layout_sha256: [6; 32],
            kv_k: "q4_0",
            kv_v: "q4_0",
            math_mode: CudaMathMode::Fast,
        }
    }

    fn alternate_value(declaration: &imparo_backend::KnobDecl, current: u32) -> u32 {
        match declaration.sweep {
            SweepKind::Values | SweepKind::External => declaration
                .values
                .iter()
                .copied()
                .find(|value| *value != current)
                .unwrap(),
            SweepKind::Crossing { ladder, hi, lo }
            | SweepKind::TokenMinCrossing { ladder, hi, lo }
            | SweepKind::TokenMaxCrossing { ladder, hi, lo }
            | SweepKind::SpanCrossing { ladder, hi, lo }
            | SweepKind::RowsCrossing { ladder, hi, lo } => ladder
                .iter()
                .copied()
                .chain([hi, lo])
                .find(|value| *value != current)
                .unwrap(),
            SweepKind::Derived => panic!("fixture registry has no derived knobs"),
        }
    }

    fn stored() -> Stored {
        Stored {
            knobs: CUDA_KNOBS
                .iter()
                .map(|declaration| {
                    let value = match declaration.sweep {
                        SweepKind::Values | SweepKind::External => {
                            declaration.values[0]
                        }
                        SweepKind::Crossing { ladder, .. }
                        | SweepKind::TokenMinCrossing { ladder, .. }
                        | SweepKind::TokenMaxCrossing { ladder, .. }
                        | SweepKind::SpanCrossing { ladder, .. }
                        | SweepKind::RowsCrossing { ladder, .. } => ladder[0],
                        SweepKind::Derived => {
                            panic!("fixture registry has no derived knobs")
                        }
                    };
                    (declaration.name.into(), value)
                })
                .collect(),
            batch: Some(512),
            path: std::path::PathBuf::default(),
        }
    }

    fn build(
        bytes: &[u8],
        stored: &Stored,
        identity: &CudaCorrectnessIdentity<'_>,
    ) -> Result<ExpectedCorrectness, String> {
        expected_correctness_with_env(bytes, stored, identity, Vec::new())
    }

    fn ptq_stored() -> Stored {
        let mut c = stored();
        c.batch = Some(128);
        for (name, value) in &mut c.knobs {
            match name.as_str() {
                "ptq_prefill_tensorcore"
                | "attn_decode_specialized"
                | "attn_d256_tiled" => *value = 2,
                "finite_history_prefill_tail_rows"
                | "e4b_ffn_w4a16"
                | "e4b_retained_decode_policy"
                | "lfm2_retained_execution" => *value = 0,
                _ => {}
            }
        }
        c
    }

    #[test]
    fn ptq_f16_contract_binds_independent_oracle_and_requires_every_gate() {
        use imparo_host::correctness::validate_receipt;
        let runtime = runtime_fixture();
        let mut id = identity(&runtime);
        id.kv_k = "f16";
        id.kv_v = "f16";
        let c = ptq_stored();
        let expected = build(b"ptq", &c, &id).unwrap();
        assert_eq!(expected.gate_suite, CUDA_PTQ_F16_GATE_SUITE);
        assert_eq!(expected.oracle.implementation, "imparo/ptq-f16-independent");
        assert_eq!(
            expected.oracle.options_sha256,
            "f68c509d50b80df77547a8a55681296cd34c8e6a9cfd533e7f20d514653dedfe"
        );
        assert_eq!(
            expected.required_gates.len(),
            CUDA_PTQ_F16_REQUIRED_GATES.len()
        );
        let unsigned = receipt_skeleton(&expected);
        assert!(unsigned.gates.iter().all(|g| !g.passed));
        assert!(
            !validate_receipt(Some(&unsigned), &expected).allows(&expected.routes[0])
        );
        for (name, _) in CUDA_PTQ_F16_REQUIRED_GATES {
            let mut absent = passing(&expected);
            absent.gates.retain(|g| g.gate_id != *name);
            assert!(
                !validate_receipt(Some(&absent), &expected).allows(&expected.routes[0])
            );
            let mut failed = passing(&expected);
            failed
                .gates
                .iter_mut()
                .find(|g| g.gate_id == *name)
                .unwrap()
                .passed = false;
            assert!(
                !validate_receipt(Some(&failed), &expected).allows(&expected.routes[0])
            );
        }
        let old = build(b"old", &stored(), &identity(&runtime)).unwrap();
        assert!(
            !validate_receipt(Some(&passing(&old)), &expected)
                .allows(&expected.routes[0])
        );
        let mut stale = passing(&expected);
        stale.oracle.implementation = ORACLE_IMPLEMENTATION.into();
        assert!(!validate_receipt(Some(&stale), &expected).allows(&expected.routes[0]));
        let mut different = c.clone();
        let i = different
            .knobs
            .iter()
            .position(|(n, _)| n == "rms_threads")
            .unwrap();
        different.knobs[i].1 = alternate_value(&CUDA_KNOBS[i], different.knobs[i].1);
        let changed = build(b"ptq", &different, &id).unwrap();
        assert_ne!(
            expected.routes[0].parameters_sha256,
            changed.routes[0].parameters_sha256
        );
    }

    #[test]
    fn ptq_bounded_combination_requires_new_gates_and_cannot_borrow_v2() {
        use imparo_host::correctness::validate_receipt;
        let runtime = runtime_fixture();
        let mut id = identity(&runtime);
        id.kv_k = "f16";
        id.kv_v = "f16";
        let old = ptq_stored();
        let old_expected = build(b"old", &old, &id).unwrap();
        let mut c = old.clone();
        c.knobs
            .iter_mut()
            .find(|(n, _)| n == crate::knobs::PTQ_PREFILL_TENSORCORE_KNOB)
            .unwrap()
            .1 = 3;
        assert!(build(b"partial", &c, &id).is_err());
        c.knobs
            .iter_mut()
            .find(|(n, _)| n == crate::knobs::WEIGHT_TRANSFER_POLICY_KNOB)
            .unwrap()
            .1 = 1;
        let expected = build(b"new", &c, &id).unwrap();
        assert_eq!(expected.gate_suite_version, 3);
        assert_eq!(
            expected.required_gates.len(),
            CUDA_PTQ_F16_V3_REQUIRED_GATES.len()
        );
        assert!(
            !validate_receipt(Some(&passing(&old_expected)), &expected)
                .allows(&expected.routes[0])
        );
        for (name, _) in CUDA_PTQ_F16_V3_REQUIRED_GATES {
            let mut incomplete = passing(&expected);
            incomplete.gates.retain(|g| g.gate_id != *name);
            assert!(
                !validate_receipt(Some(&incomplete), &expected)
                    .allows(&expected.routes[0])
            );
        }
        c.knobs
            .iter_mut()
            .find(|(n, _)| n == crate::knobs::PTQ_PREFILL_TENSORCORE_KNOB)
            .unwrap()
            .1 = 2;
        assert!(build(b"unadmitted-mix", &c, &id).is_err());
    }

    #[test]
    fn ptq_f16_rejects_old_kv_wrong_hardware_partial_config_and_other_workflows() {
        let runtime = runtime_fixture();
        let candidate = ptq_stored();
        for kv in ["q4_0", "q8_0"] {
            let mut id = identity(&runtime);
            id.kv_k = kv;
            id.kv_v = kv;
            assert!(
                build(b"ptq", &candidate, &id)
                    .unwrap_err()
                    .contains("cannot borrow")
            );
        }
        let mut id = identity(&runtime);
        id.kv_k = "f16";
        id.kv_v = "f16";
        for (name, value) in [
            ("ptq_prefill_tensorcore", 0),
            ("ptq_prefill_tensorcore", 1),
            ("attn_decode_specialized", 1),
            ("attn_d256_tiled", 0),
            ("finite_history_prefill_tail_rows", 64),
            ("e4b_ffn_w4a16", 1),
            ("e4b_retained_decode_policy", 1),
            ("lfm2_retained_execution", 1),
        ] {
            let mut changed = candidate.clone();
            changed.knobs.iter_mut().find(|(n, _)| n == name).unwrap().1 = value;
            assert!(build(b"ptq", &changed, &id).is_err(), "{name}={value}");
        }
        let mut changed = candidate.clone();
        changed.batch = Some(512);
        assert!(build(b"ptq", &changed, &id).is_err());
        let mut gpu = runtime_fixture();
        gpu.device_sm = 89;
        id.runtime = &gpu;
        assert!(build(b"ptq", &candidate, &id).is_err());
    }

    #[test]
    fn finite_history_candidate_requires_its_own_state_reuse_gate() {
        let runtime = runtime_fixture();
        let mut candidate = stored();
        let knob = candidate
            .knobs
            .iter_mut()
            .find(|(name, _)| name == crate::knobs::FINITE_HISTORY_KNOB)
            .unwrap();
        knob.1 = 0;
        let plain = build(b"plain", &candidate, &identity(&runtime)).unwrap();
        candidate
            .knobs
            .iter_mut()
            .find(|(name, _)| name == crate::knobs::FINITE_HISTORY_KNOB)
            .unwrap()
            .1 = 64;
        let changed = build(b"finite", &candidate, &identity(&runtime)).unwrap();
        assert_eq!(changed.required_gates.len(), plain.required_gates.len() + 1);
        assert_eq!(
            changed.required_gates.last().unwrap().gate_id,
            "finite_history_state_reuse"
        );
        assert!(changed.gate_suite.ends_with("-finite-history"));
        assert_eq!(changed.gate_suite_version, 1);
        assert_ne!(
            plain.routes[0].parameters_sha256,
            changed.routes[0].parameters_sha256
        );
    }

    // Synthetic validator fixtures only; never written as real evidence.
    fn passing(expected: &ExpectedCorrectness) -> CorrectnessReceipt {
        let mut receipt = receipt_skeleton(expected);
        receipt.producer = "unit-test".into();
        receipt.producer_version = 1;
        receipt.issued_unix_seconds = 1;
        for gate in &mut receipt.gates {
            gate.passed = true;
            gate.command_sha256 = "a".repeat(64);
            gate.output_sha256 = "b".repeat(64);
        }
        receipt
    }

    #[test]
    fn w4a16_policy_requires_all_length_ids_lifecycle_and_module_evidence() {
        use imparo_host::correctness::{
            ReceiptDecision, ReceiptRejection, validate_receipt,
        };
        let runtime = runtime_fixture();
        let mut candidate = stored();
        let _plain = build(b"plain", &candidate, &identity(&runtime)).unwrap();
        candidate
            .knobs
            .iter_mut()
            .find(|(name, _)| name == "e4b_ffn_w4a16")
            .unwrap()
            .1 = 1;
        let enabled = build(b"w4a16", &candidate, &identity(&runtime)).unwrap();
        assert!(enabled.gate_suite.ends_with("-w4a16-unified-ffn"));
        assert_eq!(enabled.gate_suite_version, 3);
        assert_eq!(enabled.routes.len(), 1); // one policy, not separate M1/M3 tuning
        assert_eq!(enabled.required_gates.len(), 1 + W4A16_FFN_GATES.len());
        for id in W4A16_FFN_GATES {
            let mut incomplete = passing(&enabled);
            incomplete.gates.retain(|gate| gate.gate_id != *id);
            assert!(matches!(
                validate_receipt(Some(&incomplete), &enabled),
                ReceiptDecision::SafeFallback(ReceiptRejection::MissingGate(_))
            ));
        }
        let mut stale = passing(&enabled);
        stale.gate_suite_version = 2;
        assert!(!validate_receipt(Some(&stale), &enabled).allows(&enabled.routes[0]));
        for requirement in enabled
            .required_gates
            .iter()
            .filter(|g| g.gate_version == 2)
        {
            let mut stale = passing(&enabled);
            stale
                .gates
                .iter_mut()
                .find(|g| g.gate_id == requirement.gate_id)
                .unwrap()
                .gate_version = 1;
            assert!(
                !validate_receipt(Some(&stale), &enabled).allows(&enabled.routes[0])
            );
        }
        let mut wrong_runtime = runtime_fixture();
        wrong_runtime.device_sm = 89;
        assert!(build(b"w4a16", &candidate, &identity(&wrong_runtime)).is_err());
        for key in ["IMPARO_LAB_E4B_W4A16_FFN", "IMPARO_LAB_E4B_W4A16_CUBIN"] {
            assert!(
                validate_environment([(OsString::from(key), OsString::from("1"))])
                    .is_err()
            );
        }
    }

    #[test]
    fn packed_prefill_mode_is_value_bound_and_keeps_all_quality_gates() {
        use imparo_host::correctness::{
            ReceiptDecision, ReceiptRejection, validate_receipt,
        };
        let runtime = runtime_fixture();
        let mut candidate = stored();
        let index = candidate
            .knobs
            .iter()
            .position(|(name, _)| name == "e4b_ffn_w4a16")
            .unwrap();
        candidate.knobs[index].1 = 1;
        let mode1 =
            build(b"same bytes isolate mode", &candidate, &identity(&runtime)).unwrap();
        candidate.knobs[index].1 = 2;
        // No retained context is needed for the pure FFN quality witness.
        let mode2 =
            build(b"same bytes isolate mode", &candidate, &identity(&runtime)).unwrap();
        assert_eq!(mode2.gate_suite, mode1.gate_suite);
        assert_eq!(mode2.gate_suite_version, mode1.gate_suite_version);
        assert_eq!(mode2.required_gates, mode1.required_gates);
        assert_eq!(mode2.required_gates.len(), 1 + W4A16_FFN_GATES.len());
        assert_ne!(
            mode2.routes[0].parameters_sha256,
            mode1.routes[0].parameters_sha256
        );
        assert!(
            validate_receipt(Some(&passing(&mode2)), &mode2).allows(&mode2.routes[0])
        );
        assert!(matches!(
            validate_receipt(Some(&passing(&mode1)), &mode2),
            ReceiptDecision::SafeFallback(ReceiptRejection::RouteSetMismatch)
        ));
        candidate.knobs[index].1 = 3;
        assert!(build(b"invalid mode", &candidate, &identity(&runtime)).is_err());
    }

    #[cfg(all(feature = "cuda-speculative", target_os = "windows"))]
    #[test]
    fn retained_policy_binds_full_combination_and_forbids_lab_overrides() {
        let runtime = runtime_fixture();
        let mut candidate = stored();
        candidate
            .knobs
            .iter_mut()
            .find(|(n, _)| n == crate::knobs::E4B_RETAINED_DECODE_POLICY_KNOB)
            .unwrap()
            .1 = 1;
        assert!(build(b"incomplete", &candidate, &identity(&runtime)).is_err());
        candidate
            .knobs
            .iter_mut()
            .find(|(n, _)| n == "e4b_ffn_w4a16")
            .unwrap()
            .1 = 1;
        assert!(build(b"incomplete", &candidate, &identity(&runtime)).is_err());
        candidate
            .knobs
            .iter_mut()
            .find(|(n, _)| n == "attn_d512_mma")
            .unwrap()
            .1 = 3;
        let admitted = build(b"fixed", &candidate, &identity(&runtime)).unwrap();
        assert_eq!(admitted.required_gates.len(), 1 + W4A16_FFN_GATES.len());
        let with_policy = route_parameters_sha256(
            b"fixed",
            &candidate,
            &validate_candidate(&candidate).unwrap(),
        );
        candidate
            .knobs
            .iter_mut()
            .find(|(n, _)| n == crate::knobs::E4B_RETAINED_DECODE_POLICY_KNOB)
            .unwrap()
            .1 = 0;
        assert_ne!(
            with_policy,
            route_parameters_sha256(
                b"fixed",
                &candidate,
                &validate_candidate(&candidate).unwrap()
            )
        );
        for name in [
            "IMPARO_OUTPUT_REFERENCE_LAB",
            "IMPARO_LAB_MTP_CLUSTER_HEAD_DIR",
            "IMPARO_LAB_VERIFY_GRAPH_UPDATE",
            "IMPARO_LAB_D512_REGISTER_FRAGMENTS",
            "IMPARO_LAB_D256_PARALLEL_COMBINE",
        ] {
            assert!(
                validate_environment([(OsString::from(name), OsString::from("0"))])
                    .is_err()
            );
        }
    }

    #[test]
    fn complete_expected_and_unsigned_skeleton_match_fixed_contract() {
        let runtime = runtime_fixture();
        let expected = build(b"exact config", &stored(), &identity(&runtime)).unwrap();
        assert_eq!(expected.fingerprint.backend, "cuda");
        assert_eq!(expected.fingerprint.device_sm, 86);
        // The fingerprint carries the registry's space version. Its one literal pin is
        // registry_bump_invalidates_pre_boundary_cuda_configs in knobs.rs.
        assert_eq!(
            expected.fingerprint.numerical_space_version,
            CudaBackend.space_version()
        );
        assert_eq!(expected.fingerprint.selector_version, CUDA_SELECTOR_VERSION);
        assert_ne!(
            expected.fingerprint.selector_version,
            expected.fingerprint.numerical_space_version
        );
        assert_eq!(expected.routes.len(), 1);
        assert_eq!(
            expected.routes[0].implementation_version,
            CUDA_ROUTE_IMPLEMENTATION_VERSION
        );
        assert_eq!(expected.routes[0].operation, "model.forward");
        assert_eq!(expected.routes[0].domain_sha256.len(), 64);
        assert_eq!(expected.routes[0].parameters_sha256.len(), 64);
        assert_eq!(expected.oracle.implementation, ORACLE_IMPLEMENTATION);
        assert_eq!(expected.oracle.revision, ORACLE_REVISION);
        assert_eq!(
            expected
                .required_gates
                .iter()
                .map(|gate| (gate.gate_id.as_str(), gate.gate_version))
                .collect::<Vec<_>>(),
            CUDA_REQUIRED_GATES
        );

        let skeleton = receipt_skeleton(&expected);
        assert_eq!(skeleton.fingerprint, expected.fingerprint);
        assert_eq!(skeleton.routes.len(), 1);
        assert_eq!(skeleton.gates.len(), CUDA_REQUIRED_GATES.len());
        assert!(skeleton.gates.iter().all(|gate| {
            !gate.passed
                && gate.command_sha256.is_empty()
                && gate.output_sha256.is_empty()
        }));
        assert_eq!(skeleton.producer, "");
        assert_eq!(skeleton.producer_version, 0);
        assert_eq!(skeleton.issued_unix_seconds, 0);
    }

    #[test]
    fn natural_decode_contract_rejects_each_legacy_version_one_gate() {
        use imparo_host::correctness::{
            ReceiptDecision, ReceiptRejection, validate_receipt,
        };
        let runtime = runtime_fixture();
        for (kv, w4a16, suite_version) in
            [("q4_0", false, 8), ("q8_0", false, 2), ("q4_0", true, 3)]
        {
            let mut input = identity(&runtime);
            input.kv_k = kv;
            input.kv_v = kv;
            let mut candidate = stored();
            if w4a16 {
                candidate
                    .knobs
                    .iter_mut()
                    .find(|(name, _)| name == "e4b_ffn_w4a16")
                    .unwrap()
                    .1 = 1;
            }
            let expected =
                build(b"natural decode contract", &candidate, &input).unwrap();
            assert_eq!(expected.gate_suite_version, suite_version);
            let current = passing(&expected);
            assert!(
                validate_receipt(Some(&current), &expected).allows(&expected.routes[0])
            );
            let decode_gates: Vec<_> = expected
                .required_gates
                .iter()
                .filter(|gate| gate.gate_id.starts_with("decode_agree_"))
                .collect();
            assert_eq!(decode_gates.len(), if w4a16 { 0 } else { 3 });
            for requirement in decode_gates {
                assert_eq!(requirement.gate_version, 2);
                // Even relabelling the suite cannot upgrade old decode evidence.
                let mut stale = passing(&expected);
                stale
                    .gates
                    .iter_mut()
                    .find(|gate| gate.gate_id == requirement.gate_id)
                    .unwrap()
                    .gate_version = 1;
                assert!(matches!(
                    validate_receipt(Some(&stale), &expected),
                    ReceiptDecision::SafeFallback(ReceiptRejection::MissingGate(ref missing))
                        if missing == requirement
                ));
            }
        }
    }

    #[test]
    fn candidate_requires_exact_registry_order_names_values_and_batch() {
        let runtime = runtime_fixture();
        let identity = identity(&runtime);
        let mut candidate = stored();
        candidate.knobs.pop();
        assert!(
            build(b"x", &candidate, &identity)
                .unwrap_err()
                .contains("requires")
        );

        let mut candidate = stored();
        candidate.knobs.swap(0, 1);
        assert!(
            build(b"x", &candidate, &identity)
                .unwrap_err()
                .contains("order/name")
        );

        let mut candidate = stored();
        candidate.knobs[0].0 = "unknown_knob".into();
        assert!(
            build(b"x", &candidate, &identity)
                .unwrap_err()
                .contains("order/name")
        );

        let mut candidate = stored();
        candidate.knobs[0].1 = 3;
        assert!(
            build(b"x", &candidate, &identity)
                .unwrap_err()
                .contains("illegal")
        );

        let mut candidate = stored();
        candidate.batch = None;
        assert!(
            build(b"x", &candidate, &identity)
                .unwrap_err()
                .contains("batch")
        );
    }

    #[test]
    fn current_suite_rejects_non_q4_and_incomplete_runtime_identity() {
        let runtime = runtime_fixture();
        let mut input = identity(&runtime);
        input.kv_v = "f16";
        assert!(build(b"x", &stored(), &input).unwrap_err().contains("q4_0"));

        let mut bad_runtime = runtime_fixture();
        bad_runtime.backend_build_sha256 = [0; 32];
        assert!(
            build(b"x", &stored(), &identity(&bad_runtime))
                .unwrap_err()
                .contains("incomplete")
        );
    }

    #[test]
    fn sidecar_safe_off_is_a_receiptable_declared_value() {
        let runtime = runtime_fixture();
        let mut candidate = stored();
        let index = candidate
            .knobs
            .iter()
            .position(|(name, _)| name == "ffn_sidecar_min_tokens")
            .unwrap();
        candidate.knobs[index].1 = 0;
        assert!(build(b"safe-off", &candidate, &identity(&runtime)).is_ok());
    }

    #[test]
    fn exact128_atomic_route_is_value_bound_and_safe_off() {
        let runtime = runtime_fixture();
        let input = identity(&runtime);
        let mut off = stored();
        let index = off
            .knobs
            .iter()
            .position(|(name, _)| name == "prefill_exact128_sm86_route")
            .unwrap();
        off.knobs[index].1 = 0;
        let off_expected = build(b"exact128-off", &off, &input).unwrap();
        let mut on = off.clone();
        on.knobs[index].1 = 1;
        let on_expected = build(b"exact128-on", &on, &input).unwrap();
        assert_eq!(
            off_expected.routes[0].domain_sha256,
            on_expected.routes[0].domain_sha256
        );
        assert_ne!(
            off_expected.routes[0].parameters_sha256,
            on_expected.routes[0].parameters_sha256
        );
        on.knobs[index].1 = 2;
        let token64_expected = build(b"exact128-token64", &on, &input).unwrap();
        assert_ne!(
            on_expected.routes[0].parameters_sha256,
            token64_expected.routes[0].parameters_sha256
        );
        let mut declared_hashes = std::collections::BTreeSet::from([
            off_expected.routes[0].parameters_sha256.clone(),
            on_expected.routes[0].parameters_sha256.clone(),
            token64_expected.routes[0].parameters_sha256.clone(),
        ]);
        for value in 3..=5 {
            on.knobs[index].1 = value;
            let expected =
                build(format!("exact128-route-{value}").as_bytes(), &on, &input)
                    .unwrap();
            assert!(
                declared_hashes.insert(expected.routes[0].parameters_sha256.clone())
            );
        }
        on.knobs[index].1 = 6;
        assert!(
            build(b"exact128-illegal", &on, &input)
                .unwrap_err()
                .contains("illegal")
        );
    }

    #[test]
    fn exact128_graph_is_value_bound_safe_off_and_not_a_numerical_route() {
        let runtime = runtime_fixture();
        let input = identity(&runtime);
        let mut off = stored();
        let index = off
            .knobs
            .iter()
            .position(|(name, _)| name == "prefill_exact128_graph")
            .unwrap();
        off.knobs[index].1 = 0;
        let off_expected = build(b"graph-off", &off, &input).unwrap();
        let mut on = off.clone();
        on.knobs[index].1 = 1;
        let on_expected = build(b"graph-on", &on, &input).unwrap();
        assert_eq!(
            off_expected.routes[0].domain_sha256,
            on_expected.routes[0].domain_sha256
        );
        assert_ne!(
            off_expected.routes[0].parameters_sha256,
            on_expected.routes[0].parameters_sha256
        );
        assert_eq!(
            on_expected.routes[0].numerical_class,
            off_expected.routes[0].numerical_class
        );
        on.knobs[index].1 = 2;
        assert!(
            build(b"graph-illegal", &on, &input)
                .unwrap_err()
                .contains("illegal")
        );
    }

    #[test]
    fn unregistered_lfm_execution_overrides_cannot_reuse_a_receipt() {
        let runtime = runtime_fixture();
        let mut input = identity(&runtime);
        // Check the public expectation builder, not only the private filter.
        // Even "0" is an override: several native selectors test presence.
        for (key, kv) in [
            "IMPARO_LAB_CAPTURE_AWARE_PREFILL_TAIL",
            "IMPARO_LAB_D64_DIRECT_MMA",
            "IMPARO_LAB_D64_FA2_PREFILL_1024",
            "IMPARO_LAB_D64_FIXED_PARTITION",
            "IMPARO_LAB_D64_KEYTILE16",
            "IMPARO_LAB_D64_PARALLEL_QK",
            "IMPARO_LAB_D64_PARALLEL_QK_GRAPH",
            "IMPARO_LAB_D64_PREFIX_STATE",
            "IMPARO_LAB_D64_PV_QUERYWARP",
            "IMPARO_LAB_DRAFT_BOUNDARY_RESUME",
            "IMPARO_LAB_DRAFT_BOUNDED_TAIL",
            "IMPARO_LAB_DRAFT_CACHE_RESUME",
            "IMPARO_LAB_DSPARK_COLD_OWNER_REUSE",
            "IMPARO_LAB_DSPARK_FRONTIER16",
            "IMPARO_LAB_DSPARK_TREE16",
            "IMPARO_LAB_LFM_TREE_GRAPH",
            "IMPARO_LAB_Q8_CANONICAL_LIVE2",
            "IMPARO_LAB_Q8_DOWN_TM_ASYNC",
            "IMPARO_LAB_Q8_M16_ROW_OWNER",
            "IMPARO_LAB_Q8_M9_K3",
            "IMPARO_LAB_Q8_M9_ROW_OWNER",
            "IMPARO_LAB_Q8_TM_GATE_UP_CANONICAL_DOWN",
            "IMPARO_LAB_RETAIN_ACTIVATION_CAPACITY",
            "IMPARO_LAB_TREE_CANONICAL_PARTITION",
            "IMPARO_LAB_TREE_TAIL_REPAIR",
            "IMPARO_LAB_TREE_TAIL_SHARED_KV",
        ]
        .into_iter()
        .flat_map(|key| ["q4_0", "q8_0"].map(|kv| (key, kv)))
        {
            input.kv_k = kv;
            input.kv_v = kv;
            for value in ["0", "1"] {
                let error = expected_correctness_with_env(
                    b"same persisted config",
                    &stored(),
                    &input,
                    [(OsString::from(key), OsString::from(value))],
                )
                .unwrap_err();
                assert!(
                    error.contains("not receiptable") && error.contains(key),
                    "unregistered execution selector {key}={value}: {error}"
                );
            }
        }
        // Observing a route must not itself select a different route.
        let observers = [
            "IMPARO_LAB_GENERATED_TOKEN_IDS",
            "IMPARO_LAB_DRAFT_ACCEPTANCE_TRACE",
            "IMPARO_DEVICE_GREEDY_VERIFY_TRACE",
        ]
        .map(|key| (OsString::from(key), OsString::from("1")));
        assert!(
            expected_correctness_with_env(
                b"same persisted config",
                &stored(),
                &input,
                observers,
            )
            .is_ok()
        );
    }

    #[test]
    fn fa2_environment_overrides_cannot_reuse_a_receipt() {
        for key in ["IMPARO_LAB_D64_FA2", "IMPARO_LAB_D64_FA2_PREFILL"] {
            for value in ["0", "1"] {
                assert!(
                    validate_environment([(
                        OsString::from(key),
                        OsString::from(value)
                    )])
                    .unwrap_err()
                    .contains("not receiptable")
                );
            }
        }
    }

    #[cfg(feature = "cuda-speculative")]
    #[test]
    fn fa2_mode_requires_new_coverage_and_rejects_old_or_incomplete_receipts() {
        use imparo_host::correctness::{
            ReceiptDecision, ReceiptRejection, validate_receipt,
        };
        let runtime = runtime_fixture();
        let mut input = identity(&runtime);
        input.kv_k = "q8_0";
        input.kv_v = "q8_0";
        let mut candidate = stored();
        let index = candidate
            .knobs
            .iter()
            .position(|(name, _)| name == crate::knobs::D64_ATTENTION_KNOB)
            .unwrap();
        candidate.knobs[index].1 = 1;
        let old = build(b"mode1", &candidate, &input).unwrap();
        candidate.knobs[index].1 = 2;
        let new = build(b"mode2", &candidate, &input).unwrap();
        assert_eq!(new.required_gates.len(), old.required_gates.len() + 2);
        assert!(new.gate_suite.ends_with("-d64-fa2"));
        assert_eq!(new.gate_suite_version, 1);
        assert_ne!(
            new.routes[0].parameters_sha256,
            old.routes[0].parameters_sha256
        );
        assert!(receipt_skeleton(&new).gates.iter().all(|gate| !gate.passed));
        // Synthetic validator fixtures only: these receipts are never written or sealed.

        assert!(!validate_receipt(Some(&passing(&old)), &new).allows(&new.routes[0]));
        let mut missing = passing(&new);
        missing
            .gates
            .retain(|gate| !gate.gate_id.starts_with("fa2_d64_"));
        assert!(matches!(
            validate_receipt(Some(&missing), &new),
            ReceiptDecision::SafeFallback(ReceiptRejection::MissingGate(_))
        ));
        let mut failed = passing(&new);
        failed
            .gates
            .iter_mut()
            .find(|gate| gate.gate_id.starts_with("fa2_d64_"))
            .unwrap()
            .passed = false;
        assert!(matches!(
            validate_receipt(Some(&failed), &new),
            ReceiptDecision::SafeFallback(ReceiptRejection::GateFailed(_))
        ));
        assert!(build(b"mode2-q4", &candidate, &identity(&runtime)).is_err());
    }

    #[test]
    fn diagnostic_cuda_environment_fails_closed_and_identity_selectors_are_allowed() {
        let runtime = runtime_fixture();
        let input = identity(&runtime);
        let rejected = vec![(
            OsString::from("IMPARO_CUDA_STREAMK_NUMERIC"),
            OsString::from("1"),
        )];
        assert!(
            expected_correctness_with_env(b"x", &stored(), &input, rejected)
                .unwrap_err()
                .contains("not receiptable")
        );
        assert!(ALLOWED_CUDA_ENV.contains(&"IMPARO_CUDA_PREFILL_GRAPH_LAB"));
        assert!(ALLOWED_CUDA_ENV.contains(&"IMPARO_CUDA_TRACE_GRAPH"));
        assert!(ALLOWED_CUDA_ENV.contains(&"IMPARO_CUDA_ATTN_D64_VEC_TRACE"));
        assert!(ALLOWED_CUDA_ENV.contains(&"IMPARO_CUDA_ATTN_D64_MMA_TRACE"));
        assert!(ALLOWED_CUDA_ENV.contains(&"IMPARO_CUDA_PROFILE_FORWARD"));
        assert!(ALLOWED_CUDA_ENV.contains(&"IMPARO_CUDA_PROFILE_MATMUL"));
        assert!(ALLOWED_CUDA_ENV.contains(&"IMPARO_CUDA_PROFILE_OPS"));
        assert!(ALLOWED_CUDA_ENV.contains(&"IMPARO_CUDA_PROFILE_Q8"));
        assert!(ALLOWED_CUDA_ENV.contains(&"IMPARO_CUDA_PHASE_A1_PREFILL_WALL"));
        let allowed = ALLOWED_CUDA_ENV
            .iter()
            .map(|key| (OsString::from(key), OsString::from("fixture")));
        assert!(
            expected_correctness_with_env(b"x", &stored(), &input, allowed).is_ok()
        );
        assert!(
            validate_environment([
                (
                    OsString::from("IMPARO_BACKEND_CACHE"),
                    OsString::from("cache")
                ),
                (OsString::from("PATH"), OsString::from("path")),
            ])
            .is_ok()
        );
    }

    #[test]
    fn lfm_retained_contract_rejects_partial_migration_and_binds_identity() {
        let runtime = runtime_fixture();
        let mut input = identity(&runtime);
        input.kv_k = "q8_0";
        input.kv_v = "q8_0";
        let mut candidate = stored();
        for (name, value) in &mut candidate.knobs {
            if name == crate::knobs::LFM_RETAINED_EXECUTION_KNOB {
                *value = 1;
            }
            if let Some((_, required)) =
                LFM_RETAINED_KNOBS.iter().find(|(n, _)| name == n)
            {
                *value = *required;
            }
        }
        assert_eq!(validate_lfm_retained(&candidate, &input), Ok(true));
        for (name, _) in LFM_RETAINED_KNOBS {
            let mut incomplete = candidate.clone();
            incomplete
                .knobs
                .iter_mut()
                .find(|(n, _)| n == name)
                .unwrap()
                .1 += 1;
            assert!(
                validate_lfm_retained(&incomplete, &input).is_err(),
                "missing {name}"
            );
        }
        for batch in [0, 511, 1024, 1919, 2048] {
            let mut bad = candidate.clone();
            bad.batch = Some(batch);
            assert!(validate_lfm_retained(&bad, &input).is_err());
        }
        assert!(validate_lfm_retained(&candidate, &identity(&runtime)).is_err());
        let values: Vec<_> = candidate.knobs.iter().map(|(_, v)| *v).collect();
        let selected_hash = route_parameters_sha256(b"fixture", &candidate, &values);
        candidate
            .knobs
            .iter_mut()
            .find(|(n, _)| n == crate::knobs::LFM_RETAINED_EXECUTION_KNOB)
            .unwrap()
            .1 = 0;
        let values: Vec<_> = candidate.knobs.iter().map(|(_, v)| *v).collect();
        assert_ne!(
            selected_hash,
            route_parameters_sha256(b"fixture", &candidate, &values)
        );
    }

    #[cfg(feature = "cuda-speculative")]
    fn lfm_retained_fixture(batch: usize) -> Stored {
        let mut candidate = stored();
        candidate.batch = Some(batch);
        for (name, value) in &mut candidate.knobs {
            if name == crate::knobs::LFM_RETAINED_EXECUTION_KNOB {
                *value = 1;
            }
            if let Some((_, required)) =
                LFM_RETAINED_KNOBS.iter().find(|(n, _)| name == n)
            {
                *value = *required;
            }
        }
        candidate
    }

    #[cfg(feature = "cuda-speculative")]
    #[test]
    fn lfm_retained_domain_contract_preserves_exact_short_and_wide_requirements() {
        use imparo_host::correctness::{
            ReceiptDecision, ReceiptRejection, validate_receipt,
        };
        let runtime = runtime_fixture();
        let mut input = identity(&runtime);
        input.kv_k = "q8_0";
        input.kv_v = "q8_0";
        let wide_gates = [
            ("logit_agree_n128_q8_0", 1),
            ("logit_agree_n449_q8_0", 1),
            ("logit_agree_n512_q8_0", 1),
            ("decode_agree_n128_s8_q8_0", 2),
            ("decode_agree_n512_s8_q8_0", 2),
            ("decode_agree_n2000_s8_q8_0", 2),
            ("lfm_retained_finite_history_state_reuse", 1),
            ("fa2_d64_prefill_n512_k4096", 1),
            ("fa2_d64_dspark_m9_n4096", 1),
            ("lfm_retained_dspark_on_off_n512_s256", 1),
            ("lfm_retained_dspark_on_off_n6144_s256", 1),
            ("lfm_retained_dspark_on_off_n16384_s256", 1),
            ("lfm_retained_owner_reuse_isolation", 1),
            ("lfm_retained_domain_routes_and_graph", 1),
        ];
        assert_eq!(CUDA_Q8_REQUIRED_GATES, &wide_gates[..6]);
        for (batch, version) in [(512, 3), (1920, 2)] {
            let mut gates = wide_gates.to_vec();
            if batch == 512 {
                gates[5] = ("decode_agree_n1000_s8_q8_0", 2);
                gates.push(("lfm_retained_short_domain_bounds", 1));
            }
            let expected = build(
                b"retained state contract fixture",
                &lfm_retained_fixture(batch),
                &input,
            )
            .unwrap();
            assert_eq!(
                expected.gate_suite,
                "cuda-llama-fa-q8_0-finite-history-d64-fa2-lfm-retained-v1"
            );
            assert_eq!(expected.gate_suite_version, version);
            assert_eq!(
                expected.required_gates.len(),
                if batch == 512 { 15 } else { 14 }
            );
            assert_eq!(
                expected.routes[0].numerical_class,
                NumericalClass::GateBounded {
                    gate_suite: expected.gate_suite.clone(),
                    contract_version: version,
                }
            );
            assert_eq!(
                expected
                    .required_gates
                    .iter()
                    .map(|gate| (gate.gate_id.as_str(), gate.gate_version))
                    .collect::<Vec<_>>(),
                gates
            );
            assert!(
                validate_receipt(Some(&passing(&expected)), &expected)
                    .allows(&expected.routes[0])
            );

            let mut stale = passing(&expected);
            stale.gate_suite_version = 1;
            stale.gates[6].gate_id = "finite_history_state_reuse".into();
            assert!(matches!(
                validate_receipt(Some(&stale), &expected),
                ReceiptDecision::SafeFallback(ReceiptRejection::GateSuiteMismatch)
            ));
            // Changing the suite label cannot upgrade legacy state evidence.
            stale.gate_suite_version = version;
            assert!(matches!(
                validate_receipt(Some(&stale), &expected),
                ReceiptDecision::SafeFallback(ReceiptRejection::MissingGate(ref gate))
                    if gate.gate_id == "lfm_retained_finite_history_state_reuse"
                        && gate.gate_version == 1
            ));
        }
    }

    #[cfg(feature = "cuda-speculative")]
    #[test]
    fn lfm_short_domain_requires_new_natural_and_bounds_evidence() {
        use imparo_host::correctness::{
            ReceiptDecision, ReceiptRejection, validate_receipt,
        };
        let runtime = runtime_fixture();
        let mut input = identity(&runtime);
        input.kv_k = "q8_0";
        input.kv_v = "q8_0";
        let expected = build(
            b"short domain contract fixture",
            &lfm_retained_fixture(512),
            &input,
        )
        .unwrap();

        let mut old = passing(&expected);
        old.gate_suite_version = 2;
        old.gates[5].gate_id = "decode_agree_n2000_s8_q8_0".into();
        old.gates
            .retain(|gate| gate.gate_id != "lfm_retained_short_domain_bounds");
        assert!(matches!(
            validate_receipt(Some(&old), &expected),
            ReceiptDecision::SafeFallback(ReceiptRejection::GateSuiteMismatch)
        ));
        // Relabelling a prior receipt cannot prove the new natural prompt.
        old.gate_suite_version = 3;
        assert!(matches!(
            validate_receipt(Some(&old), &expected),
            ReceiptDecision::SafeFallback(ReceiptRejection::MissingGate(ref gate))
                if gate.gate_id == "decode_agree_n1000_s8_q8_0"
                    && gate.gate_version == 2
        ));
        for (id, version) in [
            ("decode_agree_n1000_s8_q8_0", 2),
            ("lfm_retained_short_domain_bounds", 1),
        ] {
            let mut incomplete = passing(&expected);
            incomplete.gates.retain(|gate| gate.gate_id != id);
            assert!(matches!(
                validate_receipt(Some(&incomplete), &expected),
                ReceiptDecision::SafeFallback(ReceiptRejection::MissingGate(ref gate))
                    if gate.gate_id == id && gate.gate_version == version
            ));
        }
        let mut wrong_natural_version = passing(&expected);
        wrong_natural_version.gates[5].gate_version = 1;
        assert!(matches!(
            validate_receipt(Some(&wrong_natural_version), &expected),
            ReceiptDecision::SafeFallback(ReceiptRejection::MissingGate(ref gate))
                if gate.gate_id == "decode_agree_n1000_s8_q8_0"
                    && gate.gate_version == 2
        ));
    }

    #[cfg(feature = "cuda-speculative")]
    #[test]
    fn lfm_retained_state_contract_keeps_legacy_finite_history_v1() {
        let runtime = runtime_fixture();
        let mut input = identity(&runtime);
        input.kv_k = "q8_0";
        input.kv_v = "q8_0";
        let mut candidate = lfm_retained_fixture(512);
        candidate
            .knobs
            .iter_mut()
            .find(|(name, _)| name == crate::knobs::LFM_RETAINED_EXECUTION_KNOB)
            .unwrap()
            .1 = 0;
        let expected =
            build(b"legacy finite history fixture", &candidate, &input).unwrap();
        assert_eq!(
            expected.gate_suite,
            "cuda-llama-fa-q8_0-finite-history-d64-fa2"
        );
        assert_eq!(expected.gate_suite_version, 1);
        let mut gates = CUDA_Q8_REQUIRED_GATES.to_vec();
        gates.extend([
            ("finite_history_state_reuse", 1),
            ("fa2_d64_prefill_n512_k4096", 1),
            ("fa2_d64_dspark_m9_n4096", 1),
        ]);
        assert_eq!(
            expected
                .required_gates
                .iter()
                .map(|gate| (gate.gate_id.as_str(), gate.gate_version))
                .collect::<Vec<_>>(),
            gates
        );
        assert!(
            imparo_host::correctness::validate_receipt(
                Some(&passing(&expected)),
                &expected,
            )
            .allows(&expected.routes[0])
        );
    }

    #[test]
    fn versioned_hashes_cover_candidate_registry_and_execution_identity() {
        let runtime = runtime_fixture();
        let base_stored = stored();
        let base_identity = identity(&runtime);
        let base = build(b"exact A", &base_stored, &base_identity).unwrap();
        let base_domain = &base.routes[0].domain_sha256;
        let base_parameters = &base.routes[0].parameters_sha256;

        let changed_bytes = build(b"exact B", &base_stored, &base_identity).unwrap();
        assert_ne!(base_parameters, &changed_bytes.routes[0].parameters_sha256);
        let mut changed_batch = base_stored.clone();
        changed_batch.batch = Some(256);
        let changed_batch = build(b"exact A", &changed_batch, &base_identity).unwrap();
        assert_ne!(base_parameters, &changed_batch.routes[0].parameters_sha256);

        for (index, declaration) in CUDA_KNOBS.iter().enumerate() {
            // The Q8-only compound policy has a dedicated complete-contract test.
            if matches!(
                declaration.name,
                crate::knobs::LFM_RETAINED_EXECUTION_KNOB
                    | crate::knobs::PTQ_PREFILL_TENSORCORE_KNOB
            ) {
                continue;
            }
            let mut comparison = base_stored.clone();
            let mut comparison_parameters = base_parameters.to_owned();
            if declaration.name == crate::knobs::E4B_RETAINED_DECODE_POLICY_KNOB {
                // This policy is executable only with the owner provider and mode3.
                if !cfg!(all(feature = "cuda-speculative", target_os = "windows")) {
                    continue;
                }
                for (name, value) in &mut comparison.knobs {
                    match name.as_str() {
                        "e4b_ffn_w4a16" => *value = 1,
                        "attn_d512_mma" => *value = 3,
                        _ => {}
                    }
                }
                comparison_parameters = build(b"exact A", &comparison, &base_identity)
                    .unwrap()
                    .routes[0]
                    .parameters_sha256
                    .clone();
            }
            let mut changed = comparison;
            changed.knobs[index].1 =
                alternate_value(declaration, changed.knobs[index].1);
            let changed = build(b"exact A", &changed, &base_identity).unwrap();
            assert_ne!(
                &comparison_parameters, &changed.routes[0].parameters_sha256,
                "{} missing from parameters identity",
                declaration.name
            );
        }

        let mut changed = base_identity;
        changed.model_sha256 = [6; 32];
        assert_ne!(
            base_domain,
            &build(b"exact A", &base_stored, &changed).unwrap().routes[0].domain_sha256
        );
        let mut changed = base_identity;
        changed.model_plan_sha256 = [7; 32];
        assert_ne!(
            base_domain,
            &build(b"exact A", &base_stored, &changed).unwrap().routes[0].domain_sha256
        );
        let mut changed = base_identity;
        changed.kv_layout_sha256 = [8; 32];
        assert_ne!(
            base_domain,
            &build(b"exact A", &base_stored, &changed).unwrap().routes[0].domain_sha256
        );
        let mut changed_runtime = runtime_fixture();
        changed_runtime.device_sm = 89;
        assert_ne!(
            base_domain,
            &build(b"exact A", &base_stored, &identity(&changed_runtime))
                .unwrap()
                .routes[0]
                .domain_sha256
        );
        let mut changed_runtime = runtime_fixture();
        changed_runtime.backend_build_sha256 = [8; 32];
        assert_ne!(
            base_domain,
            &build(b"exact A", &base_stored, &identity(&changed_runtime))
                .unwrap()
                .routes[0]
                .domain_sha256
        );
        let mut changed = base_identity;
        changed.math_mode = CudaMathMode::Precise;
        assert_ne!(
            base_domain,
            &build(b"exact A", &base_stored, &changed).unwrap().routes[0].domain_sha256
        );

        let mut q8 = base_identity;
        q8.kv_k = "q8_0";
        q8.kv_v = "q8_0";
        assert_ne!(
            route_domain_sha256(
                &base.fingerprint.platform,
                &base_identity,
                base.fingerprint.numerical_space_version,
                &runtime.backend_fingerprint_sha256(),
            ),
            route_domain_sha256(
                &base.fingerprint.platform,
                &q8,
                base.fingerprint.numerical_space_version,
                &runtime.backend_fingerprint_sha256(),
            )
        );
    }

    #[test]
    fn oracle_options_pin_fa_q4_both_sides_and_context_copy() {
        assert_eq!(CUDA_GATE_SUITE_VERSION, 8);
        assert_eq!(ORACLE_OPTIONS_VERSION, 2);
        assert_eq!(
            ORACLE_ARGUMENTS,
            ["-fa", "on", "-ctxcp", "0", "-ctk", "q4_0", "-ctv", "q4_0"]
        );
        assert_eq!(oracle_options_sha256().len(), 64);
        assert_eq!(
            ORACLE_BUNDLE_MANIFEST_SHA256,
            "69388d9d5f910d26b8307b5f510449ea7ec717b8891decd81b918300180eaab0"
        );
    }

    #[test]
    fn q8_identity_selects_the_q8_gate_contract() {
        let runtime = runtime_fixture();
        let mut q8 = identity(&runtime);
        q8.kv_k = "q8_0";
        q8.kv_v = "q8_0";
        let expected = build(b"q8 candidate", &stored(), &q8).unwrap();

        assert_eq!(CUDA_Q8_GATE_SUITE_VERSION, 2);
        assert_eq!(expected.gate_suite, CUDA_Q8_GATE_SUITE);
        assert_eq!(expected.gate_suite_version, CUDA_Q8_GATE_SUITE_VERSION);
        assert_eq!(
            expected
                .required_gates
                .iter()
                .map(|gate| (gate.gate_id.as_str(), gate.gate_version))
                .collect::<Vec<_>>(),
            CUDA_Q8_REQUIRED_GATES
        );
        assert_eq!(
            expected.oracle.options_sha256,
            oracle_options_sha256_for(Q8_ORACLE_ARGUMENTS)
        );
        assert_eq!(
            expected.oracle.bundle_manifest_sha256,
            Q8_ORACLE_BUNDLE_MANIFEST_SHA256
        );
    }

    #[test]
    fn tracked_oracle_manifest_matches_the_compiled_contract() {
        use sha2::{Digest, Sha256};

        // Read repository-only audit fixtures at test runtime rather than embedding
        // them at compile time. The detached public tree deliberately excludes these
        // internal receipts, but its `cargo check --all-targets` must remain complete.
        let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let manifest_bytes = std::fs::read(
            repo_root.join("dev_harness/refs/oracles/llama-4695f001-windows-sm86.json"),
        )
        .unwrap();
        let manifest: serde_json::Value =
            serde_json::from_slice(&manifest_bytes).unwrap();
        let canonical = serde_json::to_vec(&manifest).unwrap();
        let digest = encode_hex(&Sha256::digest(canonical));
        assert_eq!(digest, ORACLE_BUNDLE_MANIFEST_SHA256);

        let receipt_bytes = std::fs::read(repo_root.join(
            "docs/evidence/cuda-onto-v2/sm86-step9/step9-lfm2-sm86-q4-v24.txt.receipt.json",
        ))
        .unwrap();
        let receipt: serde_json::Value =
            serde_json::from_slice(&receipt_bytes).unwrap();
        assert_eq!(
            receipt["oracle"]["bundle_manifest_sha256"],
            ORACLE_BUNDLE_MANIFEST_SHA256
        );
        let config =
            std::fs::read(repo_root.join(
                "docs/evidence/cuda-onto-v2/sm86-step9/step9-lfm2-sm86-q4-v24.txt",
            ))
            .unwrap();
        assert_eq!(
            receipt["fingerprint"]["config_sha256"],
            encode_hex(&Sha256::digest(&config))
        );
    }
}
