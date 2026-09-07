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

/// Numerical selector policy, deliberately independent of CUDA's tuning-space version.
pub const CUDA_SELECTOR_VERSION: u32 = 4;
pub const CUDA_GATE_SUITE: &str = "cuda-llama-fa-q4_0";
pub const CUDA_Q8_GATE_SUITE: &str = "cuda-llama-fa-q8_0";
pub const CUDA_GATE_SUITE_VERSION: u32 = 7;
pub const CUDA_Q8_GATE_SUITE_VERSION: u32 = 1;
pub const CUDA_ROUTE_IMPLEMENTATION_VERSION: u32 = 4;

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
    ("decode_agree_n128_s8_q4_0", 1),
    ("decode_agree_n512_s8_q4_0", 1),
    ("decode_agree_n2000_s8_q4_0", 1),
    ("prefill_graph_replay_agree_n128", 2),
];

pub const CUDA_Q8_REQUIRED_GATES: &[(&str, u32)] = &[
    ("logit_agree_n128_q8_0", 1),
    ("logit_agree_n449_q8_0", 1),
    ("logit_agree_n512_q8_0", 1),
    ("decode_agree_n128_s8_q8_0", 1),
    ("decode_agree_n512_s8_q8_0", 1),
    ("decode_agree_n2000_s8_q8_0", 1),
];

#[derive(Clone, Copy)]
struct GatePolicy {
    suite: &'static str,
    version: u32,
    gates: &'static [(&'static str, u32)],
    oracle_arguments: &'static [&'static str],
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
    let policy = gate_policy(identity)?;
    let values = validate_candidate(stored)?;
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
            gate_suite: policy.suite.into(),
            contract_version: policy.version,
        },
    };
    route
        .validate()
        .map_err(|error| format!("invalid CUDA numerical route: {error:?}"))?;
    Ok(ExpectedCorrectness {
        fingerprint,
        gate_suite: policy.suite.into(),
        gate_suite_version: policy.version,
        routes: vec![route],
        required_gates: policy
            .gates
            .iter()
            .map(|(gate_id, gate_version)| GateRequirement {
                gate_id: (*gate_id).into(),
                gate_version: *gate_version,
            })
            .collect(),
        oracle: OracleFingerprint {
            implementation: ORACLE_IMPLEMENTATION.into(),
            revision: ORACLE_REVISION.into(),
            options_sha256: oracle_options_sha256_for(policy.oracle_arguments),
            bundle_manifest_sha256: policy.oracle_bundle_manifest_sha256.into(),
        },
    })
}

fn validate_environment(
    environment: impl IntoIterator<Item = (OsString, OsString)>,
) -> Result<(), String> {
    for (key, _) in environment {
        let key = key.to_string_lossy();
        if key.starts_with("IMPARO_CUDA_")
            && !ALLOWED_CUDA_ENV.iter().any(|allowed| key == *allowed)
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
            oracle_bundle_manifest_sha256: ORACLE_BUNDLE_MANIFEST_SHA256,
        }),
        ("q8_0", "q8_0") => Ok(GatePolicy {
            suite: CUDA_Q8_GATE_SUITE,
            version: CUDA_Q8_GATE_SUITE_VERSION,
            gates: CUDA_Q8_REQUIRED_GATES,
            oracle_arguments: Q8_ORACLE_ARGUMENTS,
            oracle_bundle_manifest_sha256: Q8_ORACLE_BUNDLE_MANIFEST_SHA256,
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
        | SweepKind::SpanCrossing { ladder, hi, lo } => {
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
    canonical.finish()
}

#[must_use]
pub fn oracle_options_sha256() -> String {
    oracle_options_sha256_for(ORACLE_ARGUMENTS)
}

fn oracle_options_sha256_for(arguments: &[&str]) -> String {
    let mut canonical = Canonical::new("imparo-cuda-llama-oracle-options");
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
            | SweepKind::SpanCrossing { ladder, hi, lo } => ladder
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
                        | SweepKind::SpanCrossing { ladder, .. } => ladder[0],
                        SweepKind::Derived => {
                            panic!("fixture registry has no derived knobs")
                        }
                    };
                    (declaration.name.into(), value)
                })
                .collect(),
            batch: Some(512),
            path: Default::default(),
        }
    }

    fn build(
        bytes: &[u8],
        stored: &Stored,
        identity: &CudaCorrectnessIdentity<'_>,
    ) -> Result<ExpectedCorrectness, String> {
        expected_correctness_with_env(bytes, stored, identity, Vec::new())
    }

    #[test]
    fn complete_expected_and_unsigned_skeleton_match_fixed_contract() {
        let runtime = runtime_fixture();
        let expected = build(b"exact config", &stored(), &identity(&runtime)).unwrap();
        assert_eq!(expected.fingerprint.backend, "cuda");
        assert_eq!(expected.fingerprint.device_sm, 86);
        assert_eq!(expected.fingerprint.numerical_space_version, 43);
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
            let mut changed = base_stored.clone();
            changed.knobs[index].1 =
                alternate_value(declaration, changed.knobs[index].1);
            let changed = build(b"exact A", &changed, &base_identity).unwrap();
            assert_ne!(
                base_parameters, &changed.routes[0].parameters_sha256,
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
        assert_eq!(CUDA_GATE_SUITE_VERSION, 7);
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
