//! Strict, backend-agnostic correctness-receipt contract.
//!
//! This module deliberately performs no filesystem access and never manufactures a
//! receipt. Callers supply the bytes they loaded and the complete identity they expect;
//! every parse, identity or gate failure produces an empty safe-fallback admission.

use std::collections::{BTreeMap, BTreeSet};

use imparo_backend::numerical::{NumericalClass, RouteKey};
use serde::{Deserialize, Serialize};

/// Current on-disk correctness-receipt schema.
pub const CORRECTNESS_RECEIPT_SCHEMA_VERSION: u32 = 3;

/// Exact execution identity to which a receipt is bound.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CorrectnessFingerprint {
    /// SHA-256 of the exact tuned-config bytes admitted by this receipt.
    pub config_sha256: String,
    pub model_sha256: String,
    pub model_plan_sha256: String,
    pub kv_layout_sha256: String,
    pub platform: String,
    pub backend: String,
    pub device_uuid: String,
    pub device_sm: u32,
    pub driver_version: u32,
    pub runtime_version: u32,
    pub backend_abi: u32,
    pub backend_fingerprint_sha256: String,
    pub kv_k: String,
    pub kv_v: String,
    pub numerical_space_version: u32,
    pub selector_version: u32,
    pub math_mode: String,
}

/// Serialized numerical class. Kept local so the low-level backend contract remains
/// dependency-free; conversion to [`NumericalClass`] is exact.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ReceiptNumericalClass {
    BitExact {
        reference: String,
        contract_version: u32,
    },
    GateBounded {
        gate_suite: String,
        contract_version: u32,
    },
    DiagnosticOnly,
}

impl From<&NumericalClass> for ReceiptNumericalClass {
    fn from(value: &NumericalClass) -> Self {
        match value {
            NumericalClass::BitExact {
                reference,
                contract_version,
            } => Self::BitExact {
                reference: reference.clone(),
                contract_version: *contract_version,
            },
            NumericalClass::GateBounded {
                gate_suite,
                contract_version,
            } => Self::GateBounded {
                gate_suite: gate_suite.clone(),
                contract_version: *contract_version,
            },
            NumericalClass::DiagnosticOnly => Self::DiagnosticOnly,
        }
    }
}

impl From<&ReceiptNumericalClass> for NumericalClass {
    fn from(value: &ReceiptNumericalClass) -> Self {
        match value {
            ReceiptNumericalClass::BitExact {
                reference,
                contract_version,
            } => Self::BitExact {
                reference: reference.clone(),
                contract_version: *contract_version,
            },
            ReceiptNumericalClass::GateBounded {
                gate_suite,
                contract_version,
            } => Self::GateBounded {
                gate_suite: gate_suite.clone(),
                contract_version: *contract_version,
            },
            ReceiptNumericalClass::DiagnosticOnly => Self::DiagnosticOnly,
        }
    }
}

/// Serialized route identity covered by a receipt.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReceiptRoute {
    pub backend: String,
    pub operation: String,
    pub implementation: String,
    pub implementation_version: u32,
    pub selector_version: u32,
    pub domain_sha256: String,
    pub parameters_sha256: String,
    pub numerical_class: ReceiptNumericalClass,
}

impl From<&RouteKey> for ReceiptRoute {
    fn from(value: &RouteKey) -> Self {
        Self {
            backend: value.backend.clone(),
            operation: value.operation.clone(),
            implementation: value.implementation.clone(),
            implementation_version: value.implementation_version,
            selector_version: value.selector_version,
            domain_sha256: value.domain_sha256.clone(),
            parameters_sha256: value.parameters_sha256.clone(),
            numerical_class: (&value.numerical_class).into(),
        }
    }
}

impl From<&ReceiptRoute> for RouteKey {
    fn from(value: &ReceiptRoute) -> Self {
        Self {
            backend: value.backend.clone(),
            operation: value.operation.clone(),
            implementation: value.implementation.clone(),
            implementation_version: value.implementation_version,
            selector_version: value.selector_version,
            domain_sha256: value.domain_sha256.clone(),
            parameters_sha256: value.parameters_sha256.clone(),
            numerical_class: (&value.numerical_class).into(),
        }
    }
}

/// One mechanically captured gate result.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GateEvidence {
    pub gate_id: String,
    pub gate_version: u32,
    pub passed: bool,
    pub command_sha256: String,
    pub output_sha256: String,
}

/// Exact oracle build and invocation contract used by every gate in the receipt.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OracleFingerprint {
    pub implementation: String,
    pub revision: String,
    /// Hash of canonical options, including explicitly enabled fast attention.
    pub options_sha256: String,
    /// Hash of the tracked manifest authenticating the complete oracle bundle.
    pub bundle_manifest_sha256: String,
}

/// Strict receipt envelope. Unknown fields are rejected at every nesting level.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CorrectnessReceipt {
    pub schema_version: u32,
    pub producer: String,
    pub producer_version: u32,
    pub issued_unix_seconds: u64,
    pub fingerprint: CorrectnessFingerprint,
    pub gate_suite: String,
    pub gate_suite_version: u32,
    pub routes: Vec<ReceiptRoute>,
    pub gates: Vec<GateEvidence>,
    pub oracle: OracleFingerprint,
}

/// A gate required by the running engine's versioned contract.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct GateRequirement {
    pub gate_id: String,
    pub gate_version: u32,
}

/// Complete contract expected by the running engine.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpectedCorrectness {
    pub fingerprint: CorrectnessFingerprint,
    pub gate_suite: String,
    pub gate_suite_version: u32,
    pub routes: Vec<RouteKey>,
    pub required_gates: Vec<GateRequirement>,
    pub oracle: OracleFingerprint,
}

/// Fingerprint component which failed an exact comparison.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FingerprintField {
    ConfigSha256,
    ModelSha256,
    ModelPlanSha256,
    KvLayoutSha256,
    Platform,
    Backend,
    DeviceUuid,
    DeviceSm,
    DriverVersion,
    RuntimeVersion,
    BackendAbi,
    BackendFingerprintSha256,
    KvK,
    KvV,
    NumericalSpaceVersion,
    SelectorVersion,
    MathMode,
}

/// Why no optimized numerical route was admitted.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReceiptRejection {
    MissingReceipt,
    MalformedReceipt(String),
    SchemaVersion { expected: u32, actual: u32 },
    InvalidExpectedContract(&'static str),
    InvalidReceiptField(&'static str),
    FingerprintMismatch(FingerprintField),
    GateSuiteMismatch,
    OracleMismatch,
    RouteSetMismatch,
    DuplicateRoute,
    DuplicateGate,
    MissingGate(GateRequirement),
    GateFailed(GateRequirement),
}

/// Routes proven by one validated receipt. Construction is private so a caller cannot
/// turn an unchecked receipt into positive execution authority.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValidatedRoutes {
    routes: BTreeSet<RouteKey>,
}

impl ValidatedRoutes {
    #[must_use]
    pub fn allows(&self, route: &RouteKey) -> bool {
        self.routes.contains(route)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.routes.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.routes.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &RouteKey> {
        self.routes.iter()
    }
}

/// Fail-closed validation result. Safe fallback never carries a route set.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReceiptDecision {
    Admitted(ValidatedRoutes),
    SafeFallback(ReceiptRejection),
}

impl ReceiptDecision {
    /// False for every safe-fallback outcome.
    #[must_use]
    pub fn allows(&self, route: &RouteKey) -> bool {
        match self {
            Self::Admitted(routes) => routes.allows(route),
            Self::SafeFallback(_) => false,
        }
    }

    #[must_use]
    pub fn rejection(&self) -> Option<&ReceiptRejection> {
        match self {
            Self::Admitted(_) => None,
            Self::SafeFallback(reason) => Some(reason),
        }
    }
}

/// Parse optional receipt bytes and validate them. Missing, malformed and unknown-field
/// inputs all return a safe fallback with no admitted routes.
#[must_use]
pub fn parse_and_validate(
    body: Option<&str>,
    expected: &ExpectedCorrectness,
) -> ReceiptDecision {
    let Some(body) = body else {
        return ReceiptDecision::SafeFallback(ReceiptRejection::MissingReceipt);
    };
    let receipt: CorrectnessReceipt = match serde_json::from_str(body) {
        Ok(receipt) => receipt,
        Err(error) => {
            return ReceiptDecision::SafeFallback(ReceiptRejection::MalformedReceipt(
                error.to_string(),
            ));
        }
    };
    validate_receipt(Some(&receipt), expected)
}

/// Validate an already parsed receipt without filesystem, device or environment access.
#[must_use]
pub fn validate_receipt(
    receipt: Option<&CorrectnessReceipt>,
    expected: &ExpectedCorrectness,
) -> ReceiptDecision {
    let Some(receipt) = receipt else {
        return ReceiptDecision::SafeFallback(ReceiptRejection::MissingReceipt);
    };
    if let Err(field) = validate_expected(expected) {
        return ReceiptDecision::SafeFallback(
            ReceiptRejection::InvalidExpectedContract(field),
        );
    }
    if receipt.schema_version != CORRECTNESS_RECEIPT_SCHEMA_VERSION {
        return ReceiptDecision::SafeFallback(ReceiptRejection::SchemaVersion {
            expected: CORRECTNESS_RECEIPT_SCHEMA_VERSION,
            actual: receipt.schema_version,
        });
    }
    if receipt.producer.trim().is_empty() || receipt.producer_version == 0 {
        return ReceiptDecision::SafeFallback(ReceiptRejection::InvalidReceiptField(
            "producer",
        ));
    }
    if receipt.issued_unix_seconds == 0 {
        return ReceiptDecision::SafeFallback(ReceiptRejection::InvalidReceiptField(
            "issued_unix_seconds",
        ));
    }
    if let Err(field) = validate_fingerprint(&receipt.fingerprint) {
        return ReceiptDecision::SafeFallback(ReceiptRejection::InvalidReceiptField(
            field,
        ));
    }
    if let Some(field) =
        fingerprint_mismatch(&receipt.fingerprint, &expected.fingerprint)
    {
        return ReceiptDecision::SafeFallback(ReceiptRejection::FingerprintMismatch(
            field,
        ));
    }
    if receipt.gate_suite != expected.gate_suite
        || receipt.gate_suite_version != expected.gate_suite_version
    {
        return ReceiptDecision::SafeFallback(ReceiptRejection::GateSuiteMismatch);
    }
    if validate_oracle(&receipt.oracle).is_err() {
        return ReceiptDecision::SafeFallback(ReceiptRejection::InvalidReceiptField(
            "oracle",
        ));
    }
    if receipt.oracle != expected.oracle {
        return ReceiptDecision::SafeFallback(ReceiptRejection::OracleMismatch);
    }

    let mut routes = BTreeSet::new();
    for stored in &receipt.routes {
        let route: RouteKey = stored.into();
        if route.validate().is_err() {
            return ReceiptDecision::SafeFallback(
                ReceiptRejection::InvalidReceiptField("routes"),
            );
        }
        if !routes.insert(route) {
            return ReceiptDecision::SafeFallback(ReceiptRejection::DuplicateRoute);
        }
    }
    let expected_routes: BTreeSet<_> = expected.routes.iter().cloned().collect();
    if routes != expected_routes {
        return ReceiptDecision::SafeFallback(ReceiptRejection::RouteSetMismatch);
    }

    let mut gates = BTreeMap::new();
    for evidence in &receipt.gates {
        let requirement = GateRequirement {
            gate_id: evidence.gate_id.clone(),
            gate_version: evidence.gate_version,
        };
        if requirement.gate_id.trim().is_empty()
            || requirement.gate_version == 0
            || !is_sha256_hex(&evidence.command_sha256)
            || !is_sha256_hex(&evidence.output_sha256)
        {
            return ReceiptDecision::SafeFallback(
                ReceiptRejection::InvalidReceiptField("gates"),
            );
        }
        if gates.insert(requirement.clone(), evidence.passed).is_some() {
            return ReceiptDecision::SafeFallback(ReceiptRejection::DuplicateGate);
        }
        if !evidence.passed {
            return ReceiptDecision::SafeFallback(ReceiptRejection::GateFailed(
                requirement,
            ));
        }
    }
    for requirement in &expected.required_gates {
        if !gates.contains_key(requirement) {
            return ReceiptDecision::SafeFallback(ReceiptRejection::MissingGate(
                requirement.clone(),
            ));
        }
    }

    ReceiptDecision::Admitted(ValidatedRoutes { routes })
}

fn validate_expected(expected: &ExpectedCorrectness) -> Result<(), &'static str> {
    validate_fingerprint(&expected.fingerprint)?;
    if expected.gate_suite.trim().is_empty() || expected.gate_suite_version == 0 {
        return Err("gate_suite");
    }
    validate_oracle(&expected.oracle)?;
    if expected.routes.is_empty() {
        return Err("routes");
    }
    let mut routes = BTreeSet::new();
    for route in &expected.routes {
        route.validate().map_err(|_| "routes")?;
        if route.backend != expected.fingerprint.backend
            || route.selector_version != expected.fingerprint.selector_version
        {
            return Err("routes");
        }
        if let NumericalClass::GateBounded {
            gate_suite,
            contract_version,
        } = &route.numerical_class
        {
            if gate_suite != &expected.gate_suite
                || *contract_version != expected.gate_suite_version
            {
                return Err("routes");
            }
        }
        if !routes.insert(route) {
            return Err("routes");
        }
    }
    if expected.required_gates.is_empty() {
        return Err("required_gates");
    }
    let mut gates = BTreeSet::new();
    for gate in &expected.required_gates {
        if gate.gate_id.trim().is_empty()
            || gate.gate_version == 0
            || !gates.insert(gate)
        {
            return Err("required_gates");
        }
    }
    Ok(())
}

fn validate_fingerprint(value: &CorrectnessFingerprint) -> Result<(), &'static str> {
    if !is_sha256_hex(&value.config_sha256) {
        return Err("fingerprint.config_sha256");
    }
    if !is_sha256_hex(&value.model_sha256) {
        return Err("fingerprint.model_sha256");
    }
    if !is_sha256_hex(&value.model_plan_sha256) {
        return Err("fingerprint.model_plan_sha256");
    }
    if !is_sha256_hex(&value.kv_layout_sha256) {
        return Err("fingerprint.kv_layout_sha256");
    }
    if value.platform.trim().is_empty() {
        return Err("fingerprint.platform");
    }
    if value.backend.trim().is_empty() {
        return Err("fingerprint.backend");
    }
    if value.device_uuid.trim().is_empty() {
        return Err("fingerprint.device_uuid");
    }
    if value.device_sm == 0 {
        return Err("fingerprint.device_sm");
    }
    if value.driver_version == 0 {
        return Err("fingerprint.driver_version");
    }
    if value.runtime_version == 0 {
        return Err("fingerprint.runtime_version");
    }
    if value.backend_abi == 0 {
        return Err("fingerprint.backend_abi");
    }
    if !is_sha256_hex(&value.backend_fingerprint_sha256) {
        return Err("fingerprint.backend_fingerprint_sha256");
    }
    if value.kv_k.trim().is_empty() || value.kv_v.trim().is_empty() {
        return Err("fingerprint.kv_types");
    }
    if value.numerical_space_version == 0 {
        return Err("fingerprint.numerical_space_version");
    }
    if value.selector_version == 0 {
        return Err("fingerprint.selector_version");
    }
    if value.math_mode.trim().is_empty() {
        return Err("fingerprint.math_mode");
    }
    Ok(())
}

fn validate_oracle(value: &OracleFingerprint) -> Result<(), &'static str> {
    if value.implementation.trim().is_empty() || value.revision.trim().is_empty() {
        return Err("oracle");
    }
    if !is_sha256_hex(&value.options_sha256) {
        return Err("oracle");
    }
    if !is_sha256_hex(&value.bundle_manifest_sha256) {
        return Err("oracle");
    }
    Ok(())
}

fn fingerprint_mismatch(
    actual: &CorrectnessFingerprint,
    expected: &CorrectnessFingerprint,
) -> Option<FingerprintField> {
    let checks = [
        (
            actual.config_sha256 == expected.config_sha256,
            FingerprintField::ConfigSha256,
        ),
        (
            actual.model_sha256 == expected.model_sha256,
            FingerprintField::ModelSha256,
        ),
        (
            actual.model_plan_sha256 == expected.model_plan_sha256,
            FingerprintField::ModelPlanSha256,
        ),
        (
            actual.kv_layout_sha256 == expected.kv_layout_sha256,
            FingerprintField::KvLayoutSha256,
        ),
        (
            actual.platform == expected.platform,
            FingerprintField::Platform,
        ),
        (
            actual.backend == expected.backend,
            FingerprintField::Backend,
        ),
        (
            actual.device_uuid == expected.device_uuid,
            FingerprintField::DeviceUuid,
        ),
        (
            actual.device_sm == expected.device_sm,
            FingerprintField::DeviceSm,
        ),
        (
            actual.driver_version == expected.driver_version,
            FingerprintField::DriverVersion,
        ),
        (
            actual.runtime_version == expected.runtime_version,
            FingerprintField::RuntimeVersion,
        ),
        (
            actual.backend_abi == expected.backend_abi,
            FingerprintField::BackendAbi,
        ),
        (
            actual.backend_fingerprint_sha256 == expected.backend_fingerprint_sha256,
            FingerprintField::BackendFingerprintSha256,
        ),
        (actual.kv_k == expected.kv_k, FingerprintField::KvK),
        (actual.kv_v == expected.kv_v, FingerprintField::KvV),
        (
            actual.numerical_space_version == expected.numerical_space_version,
            FingerprintField::NumericalSpaceVersion,
        ),
        (
            actual.selector_version == expected.selector_version,
            FingerprintField::SelectorVersion,
        ),
        (
            actual.math_mode == expected.math_mode,
            FingerprintField::MathMode,
        ),
    ];
    checks
        .into_iter()
        .find_map(|(matches, field)| (!matches).then_some(field))
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn route() -> RouteKey {
        RouteKey {
            backend: "cuda".into(),
            operation: "attention.decode".into(),
            implementation: "d256.vec.q4".into(),
            implementation_version: 1,
            selector_version: 7,
            domain_sha256: hash('a'),
            parameters_sha256: hash('b'),
            numerical_class: NumericalClass::GateBounded {
                gate_suite: "cuda-step6".into(),
                contract_version: 3,
            },
        }
    }

    fn fingerprint() -> CorrectnessFingerprint {
        CorrectnessFingerprint {
            config_sha256: hash('0'),
            model_sha256: hash('1'),
            model_plan_sha256: hash('2'),
            kv_layout_sha256: hash('4'),
            platform: "windows-x86_64".into(),
            backend: "cuda".into(),
            device_uuid: "GPU-01234567-89ab-cdef-0123-456789abcdef".into(),
            device_sm: 86,
            driver_version: 13010,
            runtime_version: 13000,
            backend_abi: 4,
            backend_fingerprint_sha256: hash('3'),
            kv_k: "q4_0".into(),
            kv_v: "q4_0".into(),
            numerical_space_version: 1,
            selector_version: 7,
            math_mode: "cuda-no-fast-math".into(),
        }
    }

    fn oracle() -> OracleFingerprint {
        OracleFingerprint {
            implementation: "zeraix/llama-cpp".into(),
            revision: "4695f001fece1660d8bb1b3748f50726ddcc100b".into(),
            options_sha256: hash('8'),
            bundle_manifest_sha256: hash('9'),
        }
    }

    fn requirement(id: &str) -> GateRequirement {
        GateRequirement {
            gate_id: id.into(),
            gate_version: 1,
        }
    }

    fn expected() -> ExpectedCorrectness {
        ExpectedCorrectness {
            fingerprint: fingerprint(),
            gate_suite: "cuda-step6".into(),
            gate_suite_version: 3,
            routes: vec![route()],
            required_gates: vec![
                requirement("logit_agree"),
                requirement("decode_agree"),
            ],
            oracle: oracle(),
        }
    }

    fn receipt() -> CorrectnessReceipt {
        CorrectnessReceipt {
            schema_version: CORRECTNESS_RECEIPT_SCHEMA_VERSION,
            producer: "dev_harness.receipt_gate".into(),
            producer_version: 1,
            issued_unix_seconds: 1,
            fingerprint: fingerprint(),
            gate_suite: "cuda-step6".into(),
            gate_suite_version: 3,
            routes: vec![ReceiptRoute::from(&route())],
            gates: vec![
                GateEvidence {
                    gate_id: "logit_agree".into(),
                    gate_version: 1,
                    passed: true,
                    command_sha256: hash('4'),
                    output_sha256: hash('5'),
                },
                GateEvidence {
                    gate_id: "decode_agree".into(),
                    gate_version: 1,
                    passed: true,
                    command_sha256: hash('6'),
                    output_sha256: hash('7'),
                },
            ],
            oracle: oracle(),
        }
    }

    fn fallback(reason: ReceiptRejection) -> ReceiptDecision {
        ReceiptDecision::SafeFallback(reason)
    }

    #[test]
    fn valid_receipt_admits_only_its_exact_route() {
        let expected = expected();
        let decision = validate_receipt(Some(&receipt()), &expected);
        assert!(decision.allows(&expected.routes[0]));
        let mut other = expected.routes[0].clone();
        other.parameters_sha256 = hash('c');
        assert!(!decision.allows(&other));
        let ReceiptDecision::Admitted(routes) = decision else {
            panic!("valid receipt fell back");
        };
        assert_eq!(routes.len(), 1);
    }

    #[test]
    fn missing_and_malformed_receipts_fail_closed() {
        let expected = expected();
        assert_eq!(
            parse_and_validate(None, &expected),
            fallback(ReceiptRejection::MissingReceipt)
        );
        let decision = parse_and_validate(Some("{not-json"), &expected);
        assert!(matches!(
            decision,
            ReceiptDecision::SafeFallback(ReceiptRejection::MalformedReceipt(_))
        ));
        assert!(!decision.allows(&expected.routes[0]));
    }

    #[test]
    fn unknown_fields_are_rejected_at_every_schema_level() {
        let expected = expected();
        for path in ["top", "fingerprint", "route", "class", "gate", "oracle"] {
            let mut value = serde_json::to_value(receipt()).unwrap();
            match path {
                "top" => value["unknown"] = true.into(),
                "fingerprint" => value["fingerprint"]["unknown"] = true.into(),
                "route" => value["routes"][0]["unknown"] = true.into(),
                "class" => {
                    value["routes"][0]["numerical_class"]["unknown"] = true.into();
                }
                "gate" => value["gates"][0]["unknown"] = true.into(),
                "oracle" => value["oracle"]["unknown"] = true.into(),
                _ => unreachable!(),
            }
            let decision = parse_and_validate(Some(&value.to_string()), &expected);
            assert!(
                matches!(
                    decision,
                    ReceiptDecision::SafeFallback(ReceiptRejection::MalformedReceipt(
                        _
                    ))
                ),
                "unknown field at {path} was accepted"
            );
        }
    }

    #[test]
    fn schema_and_gate_suite_mismatches_fail_closed() {
        let expected = expected();
        let mut value = receipt();
        value.schema_version += 1;
        assert_eq!(
            validate_receipt(Some(&value), &expected),
            fallback(ReceiptRejection::SchemaVersion {
                expected: CORRECTNESS_RECEIPT_SCHEMA_VERSION,
                actual: value.schema_version,
            })
        );
        value = receipt();
        value.gate_suite_version += 1;
        assert_eq!(
            validate_receipt(Some(&value), &expected),
            fallback(ReceiptRejection::GateSuiteMismatch)
        );
    }

    #[test]
    fn every_fingerprint_component_is_exact() {
        let expected = expected();
        let cases: Vec<(FingerprintField, Box<dyn Fn(&mut CorrectnessFingerprint)>)> = vec![
            (
                FingerprintField::ConfigSha256,
                Box::new(|v| v.config_sha256 = hash('f')),
            ),
            (
                FingerprintField::ModelSha256,
                Box::new(|v| v.model_sha256 = hash('8')),
            ),
            (
                FingerprintField::ModelPlanSha256,
                Box::new(|v| v.model_plan_sha256 = hash('9')),
            ),
            (
                FingerprintField::KvLayoutSha256,
                Box::new(|v| v.kv_layout_sha256 = hash('b')),
            ),
            (
                FingerprintField::Platform,
                Box::new(|v| v.platform.push_str("-other")),
            ),
            (
                FingerprintField::Backend,
                Box::new(|v| v.backend.push_str("-other")),
            ),
            (
                FingerprintField::DeviceUuid,
                Box::new(|v| v.device_uuid.push_str("-other")),
            ),
            (FingerprintField::DeviceSm, Box::new(|v| v.device_sm = 89)),
            (
                FingerprintField::DriverVersion,
                Box::new(|v| v.driver_version += 1),
            ),
            (
                FingerprintField::RuntimeVersion,
                Box::new(|v| v.runtime_version += 1),
            ),
            (
                FingerprintField::BackendAbi,
                Box::new(|v| v.backend_abi += 1),
            ),
            (
                FingerprintField::BackendFingerprintSha256,
                Box::new(|v| v.backend_fingerprint_sha256 = hash('a')),
            ),
            (FingerprintField::KvK, Box::new(|v| v.kv_k = "f16".into())),
            (FingerprintField::KvV, Box::new(|v| v.kv_v = "f16".into())),
            (
                FingerprintField::NumericalSpaceVersion,
                Box::new(|v| v.numerical_space_version += 1),
            ),
            (
                FingerprintField::SelectorVersion,
                Box::new(|v| v.selector_version += 1),
            ),
            (
                FingerprintField::MathMode,
                Box::new(|v| v.math_mode.push_str("-other")),
            ),
        ];
        for (field, mutate) in cases {
            let mut value = receipt();
            mutate(&mut value.fingerprint);
            assert_eq!(
                validate_receipt(Some(&value), &expected),
                fallback(ReceiptRejection::FingerprintMismatch(field))
            );
        }
    }

    #[test]
    fn oracle_revision_options_and_bundle_manifest_are_exact() {
        let expected = expected();
        let mut value = receipt();
        value.oracle.revision.push_str("-other");
        assert_eq!(
            validate_receipt(Some(&value), &expected),
            fallback(ReceiptRejection::OracleMismatch)
        );
        value = receipt();
        value.oracle.options_sha256 = "bad".into();
        assert_eq!(
            validate_receipt(Some(&value), &expected),
            fallback(ReceiptRejection::InvalidReceiptField("oracle"))
        );
        value = receipt();
        value.oracle.bundle_manifest_sha256 = hash('7');
        assert_eq!(
            validate_receipt(Some(&value), &expected),
            fallback(ReceiptRejection::OracleMismatch)
        );
        value = receipt();
        value.oracle.bundle_manifest_sha256 = "bad".into();
        assert_eq!(
            validate_receipt(Some(&value), &expected),
            fallback(ReceiptRejection::InvalidReceiptField("oracle"))
        );
    }

    #[test]
    fn route_or_numerical_class_mismatch_rejects_the_whole_receipt() {
        let expected = expected();
        let mut value = receipt();
        value.routes[0].implementation.push_str("-other");
        assert_eq!(
            validate_receipt(Some(&value), &expected),
            fallback(ReceiptRejection::RouteSetMismatch)
        );
        value = receipt();
        value.routes[0].numerical_class = ReceiptNumericalClass::BitExact {
            reference: "cuda-safe-default".into(),
            contract_version: 1,
        };
        assert_eq!(
            validate_receipt(Some(&value), &expected),
            fallback(ReceiptRejection::RouteSetMismatch)
        );
    }

    #[test]
    fn diagnostic_invalid_and_duplicate_routes_fail_closed() {
        let expected = expected();
        let mut value = receipt();
        value.routes[0].numerical_class = ReceiptNumericalClass::DiagnosticOnly;
        assert_eq!(
            validate_receipt(Some(&value), &expected),
            fallback(ReceiptRejection::InvalidReceiptField("routes"))
        );
        value = receipt();
        value.routes[0].parameters_sha256 = "not-a-hash".into();
        assert_eq!(
            validate_receipt(Some(&value), &expected),
            fallback(ReceiptRejection::InvalidReceiptField("routes"))
        );
        value = receipt();
        value.routes.push(value.routes[0].clone());
        assert_eq!(
            validate_receipt(Some(&value), &expected),
            fallback(ReceiptRejection::DuplicateRoute)
        );
    }

    #[test]
    fn missing_failed_invalid_and_duplicate_gates_fail_closed() {
        let expected = expected();
        let mut value = receipt();
        value.gates.pop();
        assert_eq!(
            validate_receipt(Some(&value), &expected),
            fallback(ReceiptRejection::MissingGate(requirement("decode_agree")))
        );
        value = receipt();
        value.gates[1].passed = false;
        assert_eq!(
            validate_receipt(Some(&value), &expected),
            fallback(ReceiptRejection::GateFailed(requirement("decode_agree")))
        );
        value = receipt();
        value.gates[0].output_sha256 = "bad".into();
        assert_eq!(
            validate_receipt(Some(&value), &expected),
            fallback(ReceiptRejection::InvalidReceiptField("gates"))
        );
        value = receipt();
        value.gates.push(value.gates[0].clone());
        assert_eq!(
            validate_receipt(Some(&value), &expected),
            fallback(ReceiptRejection::DuplicateGate)
        );
    }

    #[test]
    fn invalid_expected_contract_cannot_gain_authority() {
        let mut expected = expected();
        expected.required_gates.clear();
        let decision = validate_receipt(Some(&receipt()), &expected);
        assert_eq!(
            decision,
            fallback(ReceiptRejection::InvalidExpectedContract("required_gates"))
        );
        assert!(!decision.allows(&route()));
    }

    #[test]
    fn valid_json_round_trip_keeps_the_exact_contract() {
        let receipt = receipt();
        let body = serde_json::to_string(&receipt).unwrap();
        assert_eq!(
            parse_and_validate(Some(&body), &expected()),
            validate_receipt(Some(&receipt), &expected())
        );
    }
}
