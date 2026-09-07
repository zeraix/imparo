//! CUDA Gate0 tensor capture for offline kernel-lab experiments.
//!
//! Compiled only by the non-default cuda-gate0-capture feature. Capture inserts
//! synchronization/readback boundaries, so every artifact is timing-inadmissible.
//!
//! The Backend read surface is infallible. This producer is therefore provisional:
//! it uses two sentinel-prefilled reads plus bitwise agreement, but that is not a
//! substitute for a production fallible read API.

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use imparo_backend::{Backend, BufId};
use imparo_gguf::weights::Weights;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::ModelPlan;

const SCHEMA_VERSION: u32 = 1;
const TENSOR_VERSION: u32 = 1;
const TENSOR_MAGIC: &[u8; 16] = b"IMPARO_GATE0\0\0\0\0";
const HEADER_BYTES: u32 = 48;
const DEFAULT_MAX_RECORDS: usize = 8;
const HARD_MAX_RECORDS: usize = 256;
const DEFAULT_MAX_BYTES: u64 = 64 * 1024 * 1024;
const HARD_MAX_BYTES: u64 = 1024 * 1024 * 1024;
const READ_SENTINEL_A_BITS: u32 = 0x7fc0_a501;
const READ_SENTINEL_B_BITS: u32 = 0x7fc0_b602;

const ENV_DIR: &str = "IMPARO_GATE0_CAPTURE_DIR";
const ENV_LAYERS: &str = "IMPARO_GATE0_CAPTURE_LAYERS";
const ENV_OPS: &str = "IMPARO_GATE0_CAPTURE_OPS";
const ENV_MAX_RECORDS: &str = "IMPARO_GATE0_CAPTURE_MAX_RECORDS";
const ENV_MAX_BYTES: &str = "IMPARO_GATE0_CAPTURE_MAX_BYTES";
const ENV_GIT_COMMIT: &str = "IMPARO_GATE0_CAPTURE_GIT_COMMIT";
const ENV_GIT_STATE: &str = "IMPARO_GATE0_CAPTURE_GIT_STATE_SHA256";

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum CaptureOp {
    FfnNormInput,
    FfnDownInput,
}

impl CaptureOp {
    const fn name(self) -> &'static str {
        match self {
            Self::FfnNormInput => "ffn_norm_input",
            Self::FfnDownInput => "ffn_down_input",
        }
    }

    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "ffn_norm_input" => Ok(Self::FfnNormInput),
            "ffn_down_input" => Ok(Self::FfnDownInput),
            _ => Err(format!(
                "unknown {ENV_OPS} entry {value:?}; expected \
                 ffn_norm_input or ffn_down_input"
            )),
        }
    }
}

#[derive(Debug)]
struct Config {
    root: PathBuf,
    layers: BTreeSet<usize>,
    ops: BTreeSet<CaptureOp>,
    max_records: usize,
    max_bytes: u64,
}

impl Config {
    fn from_environment() -> Result<Option<Self>, String> {
        Self::from_lookup(|name| env::var(name).ok())
    }

    fn from_lookup(
        mut get: impl FnMut(&str) -> Option<String>,
    ) -> Result<Option<Self>, String> {
        let Some(directory) = get(ENV_DIR) else {
            return Ok(None);
        };
        if directory.trim().is_empty() {
            return Err(format!("{ENV_DIR} must not be empty"));
        }
        let layers = parse_layers(&get(ENV_LAYERS).ok_or_else(|| {
            format!("{ENV_LAYERS} is required when {ENV_DIR} is set")
        })?)?;
        let ops =
            parse_ops(&get(ENV_OPS).ok_or_else(|| {
                format!("{ENV_OPS} is required when {ENV_DIR} is set")
            })?)?;
        let max_records = parse_limit(
            get(ENV_MAX_RECORDS).as_deref(),
            DEFAULT_MAX_RECORDS as u64,
            HARD_MAX_RECORDS as u64,
            ENV_MAX_RECORDS,
        )? as usize;
        let max_bytes = parse_limit(
            get(ENV_MAX_BYTES).as_deref(),
            DEFAULT_MAX_BYTES,
            HARD_MAX_BYTES,
            ENV_MAX_BYTES,
        )?;
        Ok(Some(Self {
            root: PathBuf::from(directory),
            layers,
            ops,
            max_records,
            max_bytes,
        }))
    }
}

#[derive(Debug)]
struct Active {
    directory: PathBuf,
    run_identity_sha256: String,
    run_manifest_sha256: String,
    config: Config,
    expected: BTreeSet<RecordKey>,
    records_in_order: Vec<RecordIdentity>,
    records: usize,
    bytes: u64,
    sealed: bool,
}

pub(crate) struct CaptureRun {
    active: Option<Active>,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct RecordKey {
    layer: usize,
    op: CaptureOp,
}

#[derive(Clone, Debug)]
struct RecordIdentity {
    sequence: usize,
    key: RecordKey,
    tensor_file: String,
    tensor_sha256: String,
    record_manifest_file: String,
    record_manifest_sha256: String,
    payload_bytes: u64,
}

impl CaptureRun {
    pub(crate) fn from_environment(
        plan: &ModelPlan,
        weights: &Weights,
        tokens: &[u32],
        start_pos: usize,
        effective_kv_route: &str,
        backend: &dyn Backend,
    ) -> Result<Self, String> {
        let Some(config) = Config::from_environment()? else {
            return Ok(Self { active: None });
        };
        let device_tag = backend.device_tag();
        validate_capture_request(start_pos, tokens.len(), &device_tag)?;
        require_environment_one("IMPARO_NO_HOSTCONFIG")?;
        require_environment_one("IMPARO_CORRECTNESS_GATE")?;
        validate_requested_layers(&config.layers, plan.layers.len())?;
        let expected = config
            .layers
            .iter()
            .flat_map(|&layer| {
                config
                    .ops
                    .iter()
                    .copied()
                    .map(move |op| RecordKey { layer, op })
            })
            .collect::<BTreeSet<_>>();
        let runtime = imparo_cuda::runtime_identity()
            .map_err(|message| format!("Gate0 CUDA identity unavailable: {message}"))?;
        let identity = json!({
            "schema_version": SCHEMA_VERSION,
            "git": git_identity()?,
            "executable": current_executable_identity()?,
            "model": {
                "plan_sha256": plan.sha256_identity().to_hex(),
                "file_sha256": weights.full_file_sha256_hex(),
            },
            "device": {
                "tag": device_tag,
                "uuid": runtime.device_uuid_string(),
                "sm": runtime.device_sm,
                "driver_version": runtime.driver_version,
                "runtime_version": runtime.runtime_version,
                "backend_abi": runtime.backend_abi,
                "backend_fingerprint_sha256":
                    hex(&runtime.backend_fingerprint_sha256()),
            },
            "route": {
                "effective_kv_route": effective_kv_route,
                "backend_kv_route": format!("{:?}", backend.kv_quantization_route()),
                "backend_kv_route_override":
                    format!("{:?}", backend.kv_quantization_route_override()),
                "backend_kv_codec_route": format!("{:?}", backend.kv_byte_codec_route()),
                "kv_tag": backend.kv_tag(),
                "environment": route_environment(),
            },
            "request": {
                "start_pos": start_pos,
                "token_count": tokens.len(),
                "tokens_sha256": tokens_sha256(tokens),
            },
        });
        let identity_bytes = serde_json::to_vec(&identity)
            .map_err(|error| format!("serialize Gate0 run identity: {error}"))?;
        let run_identity_sha256 = sha256(&identity_bytes);

        fs::create_dir_all(&config.root)
            .map_err(|error| format!("create Gate0 capture root: {error}"))?;
        let root = config
            .root
            .canonicalize()
            .map_err(|error| format!("canonicalize Gate0 capture root: {error}"))?;
        if !root
            .metadata()
            .map_err(|error| format!("stat Gate0 capture root: {error}"))?
            .is_dir()
        {
            return Err(format!("Gate0 root {} is not a directory", root.display()));
        }
        let directory = root.join(format!(
            "gate0-{}-pid{}-{}",
            now_nanos()?,
            std::process::id(),
            &run_identity_sha256[..16]
        ));
        fs::create_dir(&directory).map_err(|error| {
            format!(
                "create unique Gate0 run {} (overwrite refused): {error}",
                directory.display()
            )
        })?;
        let directory = directory
            .canonicalize()
            .map_err(|error| format!("canonicalize Gate0 run: {error}"))?;
        if directory.parent() != Some(root.as_path()) {
            return Err("Gate0 run directory escaped capture root".to_string());
        }

        let manifest = json!({
            "format": "imparo-gate0-capture",
            "schema_version": SCHEMA_VERSION,
            "timing_admissible": false,
            "run_identity_sha256": run_identity_sha256,
            "identity": identity,
            "capture_contract": {
                "layers": config.layers.iter().copied().collect::<Vec<_>>(),
                "ops": config.ops.iter().map(|op| op.name()).collect::<Vec<_>>(),
                "expected_records": expected.iter().map(record_key_json)
                    .collect::<Vec<_>>(),
                "max_records": config.max_records,
                "max_payload_bytes": config.max_bytes,
                "readback": {
                    "status": "provisional",
                    "production_fallible_read": false,
                    "verification": "two distinct NaN sentinels plus bitwise double-read",
                },
                "routes": {
                    "ffn_norm_input": "materialized_workflow_boundary",
                    "ffn_down_input": "materialized_fallback_not_complete_sidecar",
                },
                "consumer_compatibility": {
                    "w4a4_capture_envelope": false,
                    "reason": "versioned producer tensor is not the W4A4 opaque envelope",
                },
                "promotion_requirement": {
                    "capture_on_off_final_logits_equivalence": "required_after_first_capture",
                },
            },
        });
        let run_manifest_sha256 = write_json_new(
            &directory,
            &directory.join("run.manifest.json"),
            &manifest,
        )?;
        Ok(Self {
            active: Some(Active {
                directory,
                run_identity_sha256,
                run_manifest_sha256,
                config,
                expected,
                records_in_order: Vec::new(),
                records: 0,
                bytes: 0,
                sealed: false,
            }),
        })
    }

    pub(crate) fn is_active(&self) -> bool {
        self.active.is_some()
    }

    pub(crate) fn wants(&self, layer: usize, op: CaptureOp) -> bool {
        self.active.as_ref().is_some_and(|active| {
            active.config.layers.contains(&layer) && active.config.ops.contains(&op)
        })
    }

    /// The limit checks happen before synchronization. Once end succeeds, any write
    /// failure aborts the forward instead of resuming with an incomplete receipt.
    pub(crate) fn capture_tensor(
        &mut self,
        backend: &dyn Backend,
        decode: bool,
        layer: usize,
        op: CaptureOp,
        buffer: BufId,
        element_offset: u64,
        rows: usize,
        cols: usize,
    ) -> Result<(), String> {
        let Some(active) = self.active.as_mut() else {
            return Ok(());
        };
        if !active.config.layers.contains(&layer) || !active.config.ops.contains(&op) {
            return Ok(());
        }
        let elements = rows
            .checked_mul(cols)
            .ok_or_else(|| "Gate0 capture element count overflow".to_string())?;
        if elements == 0 {
            return Err(format!("{} has an empty shape", op.name()));
        }
        let payload_bytes = u64::try_from(elements)
            .ok()
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| "Gate0 capture byte count overflow".to_string())?;
        let double_read_bytes = payload_bytes
            .checked_mul(2)
            .ok_or_else(|| "Gate0 double-read byte count overflow".to_string())?;
        if active.records >= active.config.max_records {
            return Err(format!(
                "Gate0 record limit {} exceeded",
                active.config.max_records
            ));
        }
        if active
            .bytes
            .checked_add(payload_bytes)
            .is_none_or(|total| total > active.config.max_bytes)
            || double_read_bytes > active.config.max_bytes
        {
            return Err(format!(
                "Gate0 byte limit {} exceeded",
                active.config.max_bytes
            ));
        }
        let key = RecordKey { layer, op };
        if active
            .records_in_order
            .iter()
            .any(|record| record.key == key)
        {
            return Err(format!(
                "Gate0 refuses duplicate record layer={layer} op={}",
                op.name()
            ));
        }

        backend
            .end()
            .map_err(|rc| format!("Gate0 synchronization failed rc={rc}"))?;
        let values = verified_double_read(backend, buffer, element_offset, elements)?;
        let tensor = encode_tensor(rows, cols, &values)?;
        let tensor_sha256 = sha256(&tensor);
        let sequence = active.records;
        let stem = format!("record-{sequence:04}-layer{layer:03}-{}", op.name());
        let tensor_name = format!("{stem}.tensor.f32le");
        let record_manifest_file = format!("{stem}.manifest.json");
        atomic_write_new(
            &active.directory,
            &active.directory.join(&tensor_name),
            &tensor,
        )?;
        let record = json!({
            "format": "imparo-gate0-capture-record",
            "schema_version": SCHEMA_VERSION,
            "timing_admissible": false,
            "run_identity_sha256": active.run_identity_sha256,
            "run_manifest": "run.manifest.json",
            "record": {
                "sequence": sequence,
                "layer": layer,
                "op": op.name(),
                "buffer": format!("{buffer:?}"),
                "element_offset": element_offset,
            },
            "tensor": {
                "file": tensor_name.clone(),
                "file_sha256": tensor_sha256.clone(),
                "format_version": TENSOR_VERSION,
                "dtype": "f32",
                "endianness": "little",
                "shape": [rows, cols],
                "payload_bytes": payload_bytes,
            },
        });
        let record_manifest_sha256 = write_json_new(
            &active.directory,
            &active.directory.join(&record_manifest_file),
            &record,
        )?;
        active.records_in_order.push(RecordIdentity {
            sequence,
            key,
            tensor_file: tensor_name,
            tensor_sha256,
            record_manifest_file,
            record_manifest_sha256,
            payload_bytes,
        });
        active.records += 1;
        active.bytes += payload_bytes;
        backend.begin_forward(decode);
        Ok(())
    }

    /// Publish the only artifact that makes a run consumable.
    ///
    /// The caller invokes this after the final device end and logits read. Any earlier
    /// failure leaves the run unsealed, which every consumer must reject.
    pub(crate) fn finalize(&mut self) -> Result<(), String> {
        let Some(active) = self.active.as_mut() else {
            return Ok(());
        };
        if active.sealed {
            return Err("Gate0 run is already sealed".to_string());
        }
        let actual = active
            .records_in_order
            .iter()
            .map(|record| record.key)
            .collect::<BTreeSet<_>>();
        if actual != active.expected
            || active.records != active.records_in_order.len()
            || active.records != active.expected.len()
        {
            return Err(format!(
                "Gate0 expected/actual record set is incomplete: expected={} actual={}",
                active.expected.len(),
                actual.len()
            ));
        }
        let summed_bytes =
            active
                .records_in_order
                .iter()
                .try_fold(0_u64, |total, record| {
                    total
                        .checked_add(record.payload_bytes)
                        .ok_or_else(|| "Gate0 sealed byte total overflow".to_string())
                })?;
        if summed_bytes != active.bytes {
            return Err(format!(
                "Gate0 record byte total mismatch: tracked={} records={summed_bytes}",
                active.bytes
            ));
        }
        let completion = json!({
            "format": "imparo-gate0-capture-complete",
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "timing_admissible": false,
            "promotion_admissible": false,
            "final_logits_equivalence": "pending_external_capture_on_off_check",
            "run_identity_sha256": active.run_identity_sha256,
            "run_manifest": {
                "file": "run.manifest.json",
                "sha256": active.run_manifest_sha256,
            },
            "capture_contract": {
                "expected": active.expected.iter().map(record_key_json)
                    .collect::<Vec<_>>(),
                "actual": actual.iter().map(record_key_json).collect::<Vec<_>>(),
                "expected_count": active.expected.len(),
                "actual_count": active.records_in_order.len(),
                "payload_bytes": active.bytes,
                "complete": true,
            },
            "records": active.records_in_order.iter().map(record_identity_json)
                .collect::<Vec<_>>(),
        });
        write_json_new(
            &active.directory,
            &active.directory.join("run.complete.json"),
            &completion,
        )?;
        validate_complete_run(&active.directory)?;
        active.sealed = true;
        Ok(())
    }
}

fn record_key_json(key: &RecordKey) -> Value {
    json!({
        "layer": key.layer,
        "op": key.op.name(),
    })
}

fn record_identity_json(record: &RecordIdentity) -> Value {
    json!({
        "sequence": record.sequence,
        "layer": record.key.layer,
        "op": record.key.op.name(),
        "tensor_file": record.tensor_file,
        "tensor_sha256": record.tensor_sha256,
        "record_manifest_file": record.record_manifest_file,
        "record_manifest_sha256": record.record_manifest_sha256,
        "payload_bytes": record.payload_bytes,
    })
}

fn validate_complete_run(directory: &Path) -> Result<Value, String> {
    let run_bytes = read_named_file(directory, "run.manifest.json")?;
    let complete_bytes = read_named_file(directory, "run.complete.json")
        .map_err(|error| format!("Gate0 run has no valid completion seal: {error}"))?;
    let run: Value = serde_json::from_slice(&run_bytes)
        .map_err(|error| format!("parse Gate0 run manifest: {error}"))?;
    let complete: Value = serde_json::from_slice(&complete_bytes)
        .map_err(|error| format!("parse Gate0 completion seal: {error}"))?;
    if run.get("timing_admissible").and_then(Value::as_bool) != Some(false)
        || complete.get("timing_admissible").and_then(Value::as_bool) != Some(false)
        || complete.get("status").and_then(Value::as_str) != Some("complete")
        || complete
            .get("promotion_admissible")
            .and_then(Value::as_bool)
            != Some(false)
        || complete
            .get("final_logits_equivalence")
            .and_then(Value::as_str)
            != Some("pending_external_capture_on_off_check")
        || complete
            .pointer("/capture_contract/complete")
            .and_then(Value::as_bool)
            != Some(true)
    {
        return Err(
            "Gate0 run/seal timing or completion contract is invalid".to_string()
        );
    }
    let sealed_run_hash = complete
        .pointer("/run_manifest/sha256")
        .and_then(Value::as_str)
        .ok_or_else(|| "Gate0 seal lacks run manifest SHA256".to_string())?;
    if sha256(&run_bytes) != sealed_run_hash {
        return Err(
            "Gate0 run manifest SHA256 does not match completion seal".to_string()
        );
    }
    let run_identity = string_value(
        run.get("run_identity_sha256"),
        "run manifest identity SHA256",
    )?;
    if Some(run_identity) != complete.get("run_identity_sha256").and_then(Value::as_str)
    {
        return Err("Gate0 run identity differs between manifest and seal".to_string());
    }
    let identity = run
        .get("identity")
        .ok_or_else(|| "Gate0 run manifest lacks identity".to_string())?;
    let identity_bytes = serde_json::to_vec(identity).map_err(|error| {
        format!("serialize Gate0 run identity for validation: {error}")
    })?;
    if sha256(&identity_bytes) != run_identity {
        return Err(
            "Gate0 run identity SHA256 does not bind identity fields".to_string()
        );
    }

    let manifest_expected = parse_record_set(
        run.pointer("/capture_contract/expected_records"),
        "run manifest expected_records",
    )?;
    let sealed_expected = parse_record_set(
        complete.pointer("/capture_contract/expected"),
        "seal expected",
    )?;
    let sealed_actual =
        parse_record_set(complete.pointer("/capture_contract/actual"), "seal actual")?;
    if manifest_expected != sealed_expected || sealed_expected != sealed_actual {
        return Err(format!(
            "Gate0 completion set is incomplete: manifest={} expected={} actual={}",
            manifest_expected.len(),
            sealed_expected.len(),
            sealed_actual.len()
        ));
    }

    let records = complete
        .get("records")
        .and_then(Value::as_array)
        .ok_or_else(|| "Gate0 seal lacks ordered records".to_string())?;
    let expected_count = usize_value(
        complete.pointer("/capture_contract/expected_count"),
        "expected_count",
    )?;
    let actual_count = usize_value(
        complete.pointer("/capture_contract/actual_count"),
        "actual_count",
    )?;
    if expected_count != manifest_expected.len()
        || actual_count != records.len()
        || expected_count != actual_count
    {
        return Err("Gate0 completion record counts are inconsistent".to_string());
    }

    let mut observed = BTreeSet::new();
    let mut payload_total = 0_u64;
    for (index, record) in records.iter().enumerate() {
        if usize_value(record.get("sequence"), "record sequence")? != index {
            return Err(format!(
                "Gate0 record sequence is not contiguous at {index}"
            ));
        }
        let key = parse_record_key(record, "ordered record")?;
        if !observed.insert(key) {
            return Err(format!(
                "Gate0 seal contains duplicate layer={} op={}",
                key.layer,
                key.op.name()
            ));
        }
        let tensor_name = string_value(record.get("tensor_file"), "tensor_file")?;
        let tensor_bytes = read_named_file(directory, tensor_name)?;
        let tensor_sha = string_value(record.get("tensor_sha256"), "tensor_sha256")?;
        if sha256(&tensor_bytes) != tensor_sha {
            return Err(format!("Gate0 tensor hash mismatch for {tensor_name}"));
        }
        let record_name =
            string_value(record.get("record_manifest_file"), "record_manifest_file")?;
        let record_bytes = read_named_file(directory, record_name)?;
        let record_sha = string_value(
            record.get("record_manifest_sha256"),
            "record_manifest_sha256",
        )?;
        if sha256(&record_bytes) != record_sha {
            return Err(format!(
                "Gate0 record manifest hash mismatch for {record_name}"
            ));
        }
        let record_manifest: Value =
            serde_json::from_slice(&record_bytes).map_err(|error| {
                format!("parse Gate0 record manifest {record_name}: {error}")
            })?;
        if record_manifest
            .pointer("/tensor/file_sha256")
            .and_then(Value::as_str)
            != Some(tensor_sha)
            || parse_record_key(
                record_manifest.get("record").unwrap_or(&Value::Null),
                "record manifest",
            )? != key
        {
            return Err(format!(
                "Gate0 record manifest {record_name} does not bind its tensor/key"
            ));
        }
        let payload = u64_value(record.get("payload_bytes"), "payload_bytes")?;
        let envelope_payload = validate_tensor_envelope(&tensor_bytes)?;
        if payload != envelope_payload {
            return Err(format!(
                "Gate0 tensor payload size differs from seal for {tensor_name}"
            ));
        }
        payload_total = payload_total
            .checked_add(payload)
            .ok_or_else(|| "Gate0 consumer payload total overflow".to_string())?;
        let expected_file_bytes = u64::from(HEADER_BYTES)
            .checked_add(payload)
            .ok_or_else(|| "Gate0 tensor file size overflow".to_string())?;
        if tensor_bytes.len() as u64 != expected_file_bytes {
            return Err(format!("Gate0 tensor file size mismatch for {tensor_name}"));
        }
    }
    if observed != manifest_expected {
        return Err("Gate0 ordered records do not cover the expected set".to_string());
    }
    let sealed_bytes = u64_value(
        complete.pointer("/capture_contract/payload_bytes"),
        "sealed payload_bytes",
    )?;
    if sealed_bytes != payload_total {
        return Err(format!(
            "Gate0 payload byte total mismatch: seal={sealed_bytes} records={payload_total}"
        ));
    }
    Ok(complete)
}

fn parse_record_set(
    value: Option<&Value>,
    label: &str,
) -> Result<BTreeSet<RecordKey>, String> {
    let values = value
        .and_then(Value::as_array)
        .ok_or_else(|| format!("Gate0 {label} is not an array"))?;
    let mut set = BTreeSet::new();
    for value in values {
        let key = parse_record_key(value, label)?;
        if !set.insert(key) {
            return Err(format!("Gate0 {label} contains a duplicate"));
        }
    }
    if set.is_empty() {
        return Err(format!("Gate0 {label} is empty"));
    }
    Ok(set)
}

fn parse_record_key(value: &Value, label: &str) -> Result<RecordKey, String> {
    let layer = usize_value(value.get("layer"), &format!("{label} layer"))?;
    let op = CaptureOp::parse(string_value(value.get("op"), &format!("{label} op"))?)?;
    Ok(RecordKey { layer, op })
}

fn string_value<'a>(value: Option<&'a Value>, label: &str) -> Result<&'a str, String> {
    value
        .and_then(Value::as_str)
        .ok_or_else(|| format!("Gate0 {label} is missing or not a string"))
}

fn u64_value(value: Option<&Value>, label: &str) -> Result<u64, String> {
    value
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("Gate0 {label} is missing or not an unsigned integer"))
}

fn usize_value(value: Option<&Value>, label: &str) -> Result<usize, String> {
    usize::try_from(u64_value(value, label)?)
        .map_err(|_| format!("Gate0 {label} does not fit usize"))
}

fn read_named_file(directory: &Path, name: &str) -> Result<Vec<u8>, String> {
    let relative = Path::new(name);
    if relative.components().count() != 1
        || relative.file_name().and_then(|value| value.to_str()) != Some(name)
    {
        return Err(format!("Gate0 consumer rejects unsafe filename {name:?}"));
    }
    fs::read(directory.join(relative))
        .map_err(|error| format!("read Gate0 artifact {name}: {error}"))
}

fn verified_double_read(
    backend: &dyn Backend,
    buffer: BufId,
    element_offset: u64,
    elements: usize,
) -> Result<Vec<f32>, String> {
    let mut first = sentinel_buffer(elements, READ_SENTINEL_A_BITS)?;
    let mut second = sentinel_buffer(elements, READ_SENTINEL_B_BITS)?;
    backend.read(buffer, element_offset, &mut first);
    backend.read(buffer, element_offset, &mut second);
    validate_double_read(&first, &second, READ_SENTINEL_A_BITS, READ_SENTINEL_B_BITS)?;
    Ok(first)
}

fn sentinel_buffer(elements: usize, bits: u32) -> Result<Vec<f32>, String> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|error| format!("allocate Gate0 readback buffer: {error}"))?;
    values.resize(elements, f32::from_bits(bits));
    Ok(values)
}

fn validate_double_read(
    first: &[f32],
    second: &[f32],
    first_sentinel: u32,
    second_sentinel: u32,
) -> Result<(), String> {
    if first.len() != second.len() {
        return Err("Gate0 double-read lengths differ".to_string());
    }
    for (index, (&left, &right)) in first.iter().zip(second).enumerate() {
        let left_bits = left.to_bits();
        let right_bits = right.to_bits();
        if left_bits == first_sentinel
            || left_bits == second_sentinel
            || right_bits == first_sentinel
            || right_bits == second_sentinel
        {
            return Err(format!(
                "Gate0 readback retained a sentinel at element {index}"
            ));
        }
        if !left.is_finite() || !right.is_finite() {
            return Err(format!(
                "Gate0 readback produced NaN/Inf at element {index}"
            ));
        }
        if left_bits != right_bits {
            return Err(format!(
                "Gate0 double-read bit mismatch at element {index}: \
                 {left_bits:08x} != {right_bits:08x}"
            ));
        }
    }
    Ok(())
}

fn validate_tensor_envelope(bytes: &[u8]) -> Result<u64, String> {
    if bytes.len() < HEADER_BYTES as usize || &bytes[..16] != TENSOR_MAGIC {
        return Err("Gate0 tensor has an invalid or truncated header".to_string());
    }
    let word = |start: usize| -> Result<[u8; 8], String> {
        bytes
            .get(start..start + 8)
            .ok_or_else(|| "Gate0 tensor header is truncated".to_string())?
            .try_into()
            .map_err(|_| "Gate0 tensor header word has an invalid width".to_string())
    };
    let version = u32::from_le_bytes(
        bytes[16..20]
            .try_into()
            .map_err(|_| "Gate0 tensor version is truncated".to_string())?,
    );
    let header_bytes = u32::from_le_bytes(
        bytes[20..24]
            .try_into()
            .map_err(|_| "Gate0 tensor header size is truncated".to_string())?,
    );
    if version != TENSOR_VERSION || header_bytes != HEADER_BYTES {
        return Err(format!(
            "Gate0 tensor version/header mismatch: version={version} header={header_bytes}"
        ));
    }
    let rows = u64::from_le_bytes(word(24)?);
    let cols = u64::from_le_bytes(word(32)?);
    let elements = u64::from_le_bytes(word(40)?);
    if rows == 0
        || cols == 0
        || rows
            .checked_mul(cols)
            .is_none_or(|product| product != elements)
    {
        return Err("Gate0 tensor dimensions/elements are inconsistent".to_string());
    }
    let payload = elements
        .checked_mul(4)
        .ok_or_else(|| "Gate0 tensor payload size overflow".to_string())?;
    let expected = u64::from(HEADER_BYTES)
        .checked_add(payload)
        .ok_or_else(|| "Gate0 tensor file size overflow".to_string())?;
    if u64::try_from(bytes.len()).ok() != Some(expected) {
        return Err(
            "Gate0 tensor payload is truncated or has trailing bytes".to_string()
        );
    }
    Ok(payload)
}

fn encode_tensor(rows: usize, cols: usize, values: &[f32]) -> Result<Vec<u8>, String> {
    if values.len() != rows.checked_mul(cols).unwrap_or(usize::MAX) {
        return Err("Gate0 tensor value count does not match shape".to_string());
    }
    let payload = values
        .len()
        .checked_mul(4)
        .ok_or_else(|| "Gate0 tensor serialization overflow".to_string())?;
    let capacity = (HEADER_BYTES as usize)
        .checked_add(payload)
        .ok_or_else(|| "Gate0 tensor total size overflow".to_string())?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .map_err(|error| format!("allocate Gate0 tensor serialization: {error}"))?;
    bytes.extend_from_slice(TENSOR_MAGIC);
    bytes.extend_from_slice(&TENSOR_VERSION.to_le_bytes());
    bytes.extend_from_slice(&HEADER_BYTES.to_le_bytes());
    for value in [rows, cols, values.len()] {
        bytes.extend_from_slice(
            &u64::try_from(value)
                .map_err(|_| "Gate0 tensor dimension does not fit u64".to_string())?
                .to_le_bytes(),
        );
    }
    for value in values {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    Ok(bytes)
}

fn write_json_new(root: &Path, target: &Path, value: &Value) -> Result<String, String> {
    let mut bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| format!("serialize {}: {error}", target.display()))?;
    bytes.push(b'\n');
    let digest = sha256(&bytes);
    atomic_write_new(root, target, &bytes)?;
    Ok(digest)
}

fn atomic_write_new(root: &Path, target: &Path, bytes: &[u8]) -> Result<(), String> {
    if target.parent() != Some(root) || target.file_name().is_none() {
        return Err(format!(
            "Gate0 target {} escaped run root",
            target.display()
        ));
    }
    if target.exists() {
        return Err(format!("Gate0 refuses to overwrite {}", target.display()));
    }
    let name = target
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "Gate0 target filename is not UTF-8".to_string())?;
    let temporary = root.join(format!(
        ".{name}.tmp-pid{}-{}",
        std::process::id(),
        now_nanos()?
    ));
    let result = (|| {
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)
            .map_err(|error| format!("create {}: {error}", temporary.display()))?;
        file.write_all(bytes)
            .map_err(|error| format!("write {}: {error}", temporary.display()))?;
        file.sync_all()
            .map_err(|error| format!("sync {}: {error}", temporary.display()))?;
        if target.exists() {
            return Err(format!("Gate0 refuses to overwrite {}", target.display()));
        }
        fs::rename(&temporary, target).map_err(|error| {
            format!(
                "atomically publish {} as {}: {error}",
                temporary.display(),
                target.display()
            )
        })
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn parse_layers(value: &str) -> Result<BTreeSet<usize>, String> {
    let mut layers = BTreeSet::new();
    for item in value
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
    {
        if let Some((first, last)) = item.split_once('-') {
            let first = first
                .parse::<usize>()
                .map_err(|_| format!("invalid {ENV_LAYERS} range {item:?}"))?;
            let last = last
                .parse::<usize>()
                .map_err(|_| format!("invalid {ENV_LAYERS} range {item:?}"))?;
            if first > last || last > 4095 {
                return Err(format!("invalid {ENV_LAYERS} range {item:?}"));
            }
            layers.extend(first..=last);
        } else {
            let layer = item
                .parse::<usize>()
                .map_err(|_| format!("invalid {ENV_LAYERS} entry {item:?}"))?;
            if layer > 4095 {
                return Err(format!("{ENV_LAYERS} entry exceeds hard limit 4095"));
            }
            layers.insert(layer);
        }
    }
    if layers.is_empty() {
        return Err(format!("{ENV_LAYERS} must contain at least one layer"));
    }
    Ok(layers)
}

fn parse_ops(value: &str) -> Result<BTreeSet<CaptureOp>, String> {
    let mut ops = BTreeSet::new();
    for item in value
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
    {
        ops.insert(CaptureOp::parse(item)?);
    }
    if ops.is_empty() {
        return Err(format!("{ENV_OPS} must contain at least one operation"));
    }
    Ok(ops)
}

fn parse_limit(
    value: Option<&str>,
    default: u64,
    maximum: u64,
    name: &str,
) -> Result<u64, String> {
    let value = match value {
        Some(value) => value
            .parse::<u64>()
            .map_err(|_| format!("{name} must be an unsigned integer"))?,
        None => default,
    };
    if value == 0 || value > maximum {
        return Err(format!("{name} must be between 1 and {maximum}"));
    }
    Ok(value)
}

fn route_environment() -> BTreeMap<String, String> {
    env::vars()
        .filter(|(name, _)| {
            name == "CUDA_VISIBLE_DEVICES"
                || matches!(
                    name.as_str(),
                    "IMPARO_BACKEND"
                        | "IMPARO_CORRECTNESS_GATE"
                        | "IMPARO_CTK"
                        | "IMPARO_CTV"
                        | "IMPARO_GPU"
                        | "IMPARO_KVQ_MASK"
                        | "IMPARO_MAX_SCORES"
                        | "IMPARO_NO_HOSTCONFIG"
                )
                || name.starts_with("IMPARO_CUDA_")
                || name.starts_with("IMPARO_SKIP_")
        })
        .collect()
}

fn validate_capture_request(
    start_pos: usize,
    token_count: usize,
    device_tag: &str,
) -> Result<(), String> {
    if start_pos != 0 {
        return Err(format!(
            "Gate0 capture requires a cold single-batch prefill at start_pos 0, got {start_pos}"
        ));
    }
    if token_count < 2 {
        return Err(
            "Gate0 capture requires a non-empty multi-token prefill batch".to_string(),
        );
    }
    if device_tag != "cuda" && !device_tag.starts_with("cuda:") {
        return Err(format!(
            "Gate0 capture requires the CUDA backend, got device tag {device_tag:?}"
        ));
    }
    Ok(())
}

fn validate_requested_layers(
    layers: &BTreeSet<usize>,
    layer_count: usize,
) -> Result<(), String> {
    if let Some(layer) = layers.iter().copied().find(|&layer| layer >= layer_count) {
        return Err(format!(
            "Gate0 requested layer {layer}, but model has {layer_count} layers"
        ));
    }
    Ok(())
}

fn require_environment_one(name: &str) -> Result<(), String> {
    match env::var(name).as_deref() {
        Ok("1") => Ok(()),
        _ => Err(format!("Gate0 capture requires {name}=1")),
    }
}

fn current_executable_identity() -> Result<Value, String> {
    let path = env::current_exe()
        .map_err(|error| format!("resolve current Gate0 executable: {error}"))?;
    executable_identity_for(&path)
}

fn executable_identity_for(path: &Path) -> Result<Value, String> {
    let canonical = path.canonicalize().map_err(|error| {
        format!("canonicalize Gate0 executable {}: {error}", path.display())
    })?;
    let file = File::open(&canonical).map_err(|error| {
        format!("open Gate0 executable {}: {error}", canonical.display())
    })?;
    let metadata = file.metadata().map_err(|error| {
        format!("stat Gate0 executable {}: {error}", canonical.display())
    })?;
    if !metadata.is_file() {
        return Err(format!(
            "Gate0 executable identity target {} is not a file",
            canonical.display()
        ));
    }
    let mut reader = BufReader::new(file);
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    let mut length = 0_u64;
    loop {
        let count = reader.read(&mut buffer).map_err(|error| {
            format!("hash Gate0 executable {}: {error}", canonical.display())
        })?;
        if count == 0 {
            break;
        }
        length = length
            .checked_add(count as u64)
            .ok_or_else(|| "Gate0 executable length overflow".to_string())?;
        digest.update(&buffer[..count]);
    }
    if length != metadata.len() {
        return Err(format!(
            "Gate0 executable changed while hashing: metadata={} streamed={length}",
            metadata.len()
        ));
    }
    Ok(json!({
        "path": canonical,
        "bytes": length,
        "sha256": hex(&digest.finalize()),
    }))
}

fn now_nanos() -> Result<u128, String> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .map_err(|error| format!("Gate0 system time is invalid: {error}"))
}

fn sha256(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().fold(
        String::with_capacity(bytes.len() * 2),
        |mut output, byte| {
            use std::fmt::Write as _;
            write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
            output
        },
    )
}

fn tokens_sha256(tokens: &[u32]) -> String {
    let mut hash = Sha256::new();
    hash.update(b"imparo-gate0-tokens-v1\0");
    hash.update((tokens.len() as u64).to_le_bytes());
    for token in tokens {
        hash.update(token.to_le_bytes());
    }
    hex(&hash.finalize())
}

fn git_identity() -> Result<Value, String> {
    let commit_override = env::var(ENV_GIT_COMMIT).ok();
    let state_override = env::var(ENV_GIT_STATE).ok();
    if commit_override.is_some() || state_override.is_some() {
        let commit = validate_hex(
            &commit_override.ok_or_else(|| {
                format!("{ENV_GIT_COMMIT} is required with {ENV_GIT_STATE}")
            })?,
            &[40, 64],
            ENV_GIT_COMMIT,
        )?;
        let state = validate_hex(
            &state_override.ok_or_else(|| {
                format!("{ENV_GIT_STATE} is required with {ENV_GIT_COMMIT}")
            })?,
            &[64],
            ENV_GIT_STATE,
        )?;
        return Ok(json!({
            "source": "environment",
            "advisory": true,
            "identity_role": "advisory_only_not_binary_identity",
            "commit": commit,
            "worktree_state_sha256": state,
        }));
    }
    let repository = Path::new(env!("CARGO_MANIFEST_DIR"));
    let commit = git_output(repository, &["rev-parse", "HEAD"])?;
    let commit = String::from_utf8(commit)
        .map_err(|_| "git commit output is not UTF-8".to_string())?;
    let commit = validate_hex(commit.trim(), &[40, 64], "git commit")?;
    let status = git_output(
        repository,
        &["status", "--porcelain=v1", "--untracked-files=all"],
    )?;
    let diff = git_output(repository, &["diff", "--binary", "HEAD"])?;
    let mut state = Sha256::new();
    state.update(b"imparo-gate0-git-state-v1\0");
    state.update((status.len() as u64).to_le_bytes());
    state.update(&status);
    state.update((diff.len() as u64).to_le_bytes());
    state.update(&diff);
    Ok(json!({
        "source": "worktree",
        "advisory": true,
        "identity_role": "advisory_only_not_binary_identity",
        "commit": commit,
        "dirty": !status.is_empty(),
        "status_sha256": sha256(&status),
        "tracked_diff_sha256": sha256(&diff),
        "worktree_state_sha256": hex(&state.finalize()),
    }))
}

fn git_output(repository: &Path, arguments: &[&str]) -> Result<Vec<u8>, String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(repository)
        .args(arguments)
        .output()
        .map_err(|error| format!("run git {}: {error}", arguments.join(" ")))?;
    if !output.status.success() {
        return Err(format!(
            "git {} failed with status {}: {}",
            arguments.join(" "),
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(output.stdout)
}

fn validate_hex(value: &str, lengths: &[usize], name: &str) -> Result<String, String> {
    if !lengths.contains(&value.len())
        || !value.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(format!("{name} is not a valid hexadecimal identity"));
    }
    Ok(value.to_ascii_lowercase())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(values: &[(&str, &str)]) -> Result<Option<Config>, String> {
        let values = values
            .iter()
            .map(|(name, value)| ((*name).to_string(), (*value).to_string()))
            .collect::<BTreeMap<_, _>>();
        Config::from_lookup(|name| values.get(name).cloned())
    }

    #[test]
    fn directory_is_the_only_activation_switch() {
        assert!(config(&[]).unwrap().is_none());
    }

    #[test]
    fn explicit_allowlists_are_required_and_closed() {
        let value = config(&[
            (ENV_DIR, "capture"),
            (ENV_LAYERS, "1,3-5"),
            (ENV_OPS, "ffn_norm_input,ffn_down_input"),
        ])
        .unwrap()
        .unwrap();
        assert_eq!(
            value.layers.into_iter().collect::<Vec<_>>(),
            vec![1, 3, 4, 5]
        );
        assert!(value.ops.contains(&CaptureOp::FfnNormInput));
        assert!(config(&[(ENV_DIR, "capture"), (ENV_OPS, "ffn_norm_input")]).is_err());
        assert!(
            config(&[
                (ENV_DIR, "capture"),
                (ENV_LAYERS, "1"),
                (ENV_OPS, "../escape"),
            ])
            .is_err()
        );
    }

    #[test]
    fn limits_reject_zero_and_excess() {
        assert!(parse_limit(Some("0"), 1, 8, "limit").is_err());
        assert!(parse_limit(Some("9"), 1, 8, "limit").is_err());
        assert_eq!(parse_limit(Some("8"), 1, 8, "limit").unwrap(), 8);
    }

    #[test]
    fn activation_contract_rejects_decode_empty_and_non_cuda_routes() {
        assert!(validate_capture_request(0, 2, "cuda").is_ok());
        assert!(validate_capture_request(0, 2, "cuda:0").is_ok());
        assert!(validate_capture_request(1, 2, "cuda:0").is_err());
        assert!(validate_capture_request(0, 0, "cuda:0").is_err());
        assert!(validate_capture_request(0, 1, "cuda:0").is_err());
        assert!(validate_capture_request(0, 2, "cpu").is_err());
        assert!(validate_capture_request(0, 2, "metal:0").is_err());
    }

    #[test]
    fn requested_layers_must_exist_before_a_run_can_start() {
        let layers = [0, 2].into_iter().collect::<BTreeSet<_>>();
        assert!(validate_requested_layers(&layers, 3).is_ok());
        assert!(validate_requested_layers(&layers, 2).is_err());
        assert!(validate_requested_layers(&layers, 0).is_err());
    }

    #[test]
    fn tensor_encoding_is_versioned_and_little_endian() {
        let bytes = encode_tensor(1, 2, &[1.0, -2.0]).unwrap();
        assert_eq!(validate_tensor_envelope(&bytes).unwrap(), 8);
        assert_eq!(&bytes[..16], TENSOR_MAGIC);
        assert_eq!(
            u32::from_le_bytes(bytes[16..20].try_into().unwrap()),
            TENSOR_VERSION
        );
        assert_eq!(
            &bytes[HEADER_BYTES as usize..HEADER_BYTES as usize + 4],
            &1.0_f32.to_bits().to_le_bytes()
        );
        let mut wrong_version = bytes.clone();
        wrong_version[16..20].copy_from_slice(&2_u32.to_le_bytes());
        assert!(validate_tensor_envelope(&wrong_version).is_err());
        let mut wrong_shape = bytes;
        wrong_shape[40..48].copy_from_slice(&3_u64.to_le_bytes());
        assert!(validate_tensor_envelope(&wrong_shape).is_err());
    }

    #[test]
    fn atomic_publish_refuses_overwrite_and_parent_escape() {
        let root = env::temp_dir().join(format!(
            "imparo-gate0-test-{}-{}",
            std::process::id(),
            now_nanos().unwrap()
        ));
        fs::create_dir(&root).unwrap();
        let target = root.join("record.tensor");
        atomic_write_new(&root, &target, b"first").unwrap();
        assert!(atomic_write_new(&root, &target, b"second").is_err());
        assert_eq!(fs::read(&target).unwrap(), b"first");
        assert!(
            atomic_write_new(&root, &root.join("nested").join("escape"), b"x").is_err()
        );
        fs::remove_file(target).unwrap();
        fs::remove_dir(root).unwrap();
    }

    #[test]
    fn token_hash_binds_order_and_length() {
        assert_eq!(tokens_sha256(&[1, 2]), tokens_sha256(&[1, 2]));
        assert_ne!(tokens_sha256(&[1, 2]), tokens_sha256(&[2, 1]));
        assert_ne!(tokens_sha256(&[1, 2]), tokens_sha256(&[1, 2, 0]));
    }

    #[test]
    fn override_identity_requires_both_valid_hashes() {
        assert!(validate_hex(&"a".repeat(40), &[40], "commit").is_ok());
        assert!(validate_hex("not-hex", &[40], "commit").is_err());
    }

    #[test]
    fn double_read_rejects_sentinel_mismatch_and_nonfinite_values() {
        assert!(
            validate_double_read(
                &[1.0, -0.0],
                &[1.0, -0.0],
                READ_SENTINEL_A_BITS,
                READ_SENTINEL_B_BITS,
            )
            .is_ok()
        );
        assert!(
            validate_double_read(
                &[f32::from_bits(READ_SENTINEL_A_BITS)],
                &[1.0],
                READ_SENTINEL_A_BITS,
                READ_SENTINEL_B_BITS,
            )
            .unwrap_err()
            .contains("sentinel")
        );
        assert!(
            validate_double_read(
                &[1.0],
                &[f32::from_bits(READ_SENTINEL_B_BITS)],
                READ_SENTINEL_A_BITS,
                READ_SENTINEL_B_BITS,
            )
            .unwrap_err()
            .contains("sentinel")
        );
        assert!(
            validate_double_read(
                &[1.0],
                &[f32::from_bits(1.0_f32.to_bits() + 1)],
                READ_SENTINEL_A_BITS,
                READ_SENTINEL_B_BITS,
            )
            .unwrap_err()
            .contains("mismatch")
        );
        for nonfinite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(
                validate_double_read(
                    &[nonfinite],
                    &[nonfinite],
                    READ_SENTINEL_A_BITS,
                    READ_SENTINEL_B_BITS,
                )
                .unwrap_err()
                .contains("NaN/Inf")
            );
        }
    }

    #[test]
    fn consumer_accepts_a_complete_hash_bound_run_and_rejects_tensor_tamper() {
        let root = test_directory("complete");
        let key = RecordKey {
            layer: 1,
            op: CaptureOp::FfnNormInput,
        };
        let identity = json!({"test_fixture": "complete-run"});
        let run_identity = sha256(&serde_json::to_vec(&identity).unwrap());
        let run = json!({
            "timing_admissible": false,
            "run_identity_sha256": run_identity,
            "identity": identity,
            "capture_contract": {
                "expected_records": [record_key_json(&key)],
            },
        });
        let run_sha =
            write_json_new(&root, &root.join("run.manifest.json"), &run).unwrap();

        let tensor_name = "record-0000-layer001-ffn_norm_input.tensor.f32le";
        let record_name = "record-0000-layer001-ffn_norm_input.manifest.json";
        let tensor = encode_tensor(1, 2, &[1.0, -2.0]).unwrap();
        let tensor_sha = sha256(&tensor);
        atomic_write_new(&root, &root.join(tensor_name), &tensor).unwrap();
        let record_manifest = json!({
            "timing_admissible": false,
            "run_identity_sha256": run_identity,
            "record": {
                "sequence": 0,
                "layer": key.layer,
                "op": key.op.name(),
            },
            "tensor": {
                "file": tensor_name,
                "file_sha256": tensor_sha,
                "payload_bytes": 8,
            },
        });
        let record_sha =
            write_json_new(&root, &root.join(record_name), &record_manifest).unwrap();
        let complete = json!({
            "status": "complete",
            "timing_admissible": false,
            "promotion_admissible": false,
            "final_logits_equivalence": "pending_external_capture_on_off_check",
            "run_identity_sha256": run_identity,
            "run_manifest": {"sha256": run_sha},
            "capture_contract": {
                "complete": true,
                "expected": [record_key_json(&key)],
                "actual": [record_key_json(&key)],
                "expected_count": 1,
                "actual_count": 1,
                "payload_bytes": 8,
            },
            "records": [{
                "sequence": 0,
                "layer": key.layer,
                "op": key.op.name(),
                "tensor_file": tensor_name,
                "tensor_sha256": tensor_sha,
                "record_manifest_file": record_name,
                "record_manifest_sha256": record_sha,
                "payload_bytes": 8,
            }],
        });
        write_json_new(&root, &root.join("run.complete.json"), &complete).unwrap();
        assert!(validate_complete_run(&root).is_ok());

        let mut tampered = tensor;
        tampered[HEADER_BYTES as usize] ^= 1;
        fs::write(root.join(tensor_name), tampered).unwrap();
        assert!(
            validate_complete_run(&root)
                .unwrap_err()
                .contains("tensor hash mismatch")
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn consumer_rejects_missing_completion_seal() {
        let root = test_directory("no-seal");
        fs::write(root.join("run.manifest.json"), b"{}\n").unwrap();
        assert!(
            validate_complete_run(&root)
                .unwrap_err()
                .contains("no valid completion seal")
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn consumer_rejects_incomplete_expected_set() {
        let root = test_directory("incomplete");
        let first = RecordKey {
            layer: 1,
            op: CaptureOp::FfnNormInput,
        };
        let second = RecordKey {
            layer: 1,
            op: CaptureOp::FfnDownInput,
        };
        let identity = json!({"test_fixture": "incomplete-set"});
        let run_identity = sha256(&serde_json::to_vec(&identity).unwrap());
        let run = json!({
            "timing_admissible": false,
            "run_identity_sha256": run_identity,
            "identity": identity,
            "capture_contract": {
                "expected_records": [record_key_json(&first), record_key_json(&second)],
            },
        });
        let run_sha =
            write_json_new(&root, &root.join("run.manifest.json"), &run).unwrap();
        let complete = json!({
            "status": "complete",
            "timing_admissible": false,
            "promotion_admissible": false,
            "final_logits_equivalence": "pending_external_capture_on_off_check",
            "run_identity_sha256": run_identity,
            "run_manifest": {"sha256": run_sha},
            "capture_contract": {
                "complete": true,
                "expected": [record_key_json(&first), record_key_json(&second)],
                "actual": [record_key_json(&first)],
                "expected_count": 2,
                "actual_count": 1,
                "payload_bytes": 0,
            },
            "records": [],
        });
        write_json_new(&root, &root.join("run.complete.json"), &complete).unwrap();
        assert!(
            validate_complete_run(&root)
                .unwrap_err()
                .contains("incomplete")
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn executable_identity_hashes_streamed_bytes_and_rejects_invalid_targets() {
        let root = test_directory("executable");
        let binary = root.join("capture.exe");
        fs::write(&binary, b"first-binary").unwrap();
        let first = executable_identity_for(&binary).unwrap();
        fs::write(&binary, b"second-binary").unwrap();
        let second = executable_identity_for(&binary).unwrap();
        assert_ne!(first["sha256"], second["sha256"]);
        assert_eq!(second["bytes"].as_u64(), Some(13));
        assert!(executable_identity_for(&root).is_err());
        assert!(executable_identity_for(&root.join("missing.exe")).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn workflow_seals_only_after_final_end_and_logits_read() {
        let source = include_str!("gemma4/workflow_gpu.rs");
        let end = source.rfind("be().end()").unwrap();
        let argmax_read = source.rfind("be().read(BufId::Tmp").unwrap();
        let logits_read = source.rfind("be().read(BufId::Logits").unwrap();
        let finalize = source.rfind("gate0_capture.finalize()?").unwrap();
        assert!(end < argmax_read);
        assert!(argmax_read < finalize);
        assert!(end < logits_read);
        assert!(logits_read < finalize);
    }

    fn test_directory(label: &str) -> PathBuf {
        let root = env::temp_dir().join(format!(
            "imparo-gate0-{label}-{}-{}",
            std::process::id(),
            now_nanos().unwrap()
        ));
        fs::create_dir(&root).unwrap();
        root
    }
}
