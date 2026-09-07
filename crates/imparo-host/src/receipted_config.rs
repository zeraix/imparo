//! Receipt-gated tuned-config loading and crash-safe persistence.
//!
//! A performance config may only select a numerical route after a receipt has proven
//! that exact config on the running model/backend identity. The legacy performance-only
//! reader remains separate so existing Metal behavior is unchanged.

use std::collections::BTreeSet;
use std::ffi::OsString;
use std::fmt::Write as _;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write as _};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use sha2::{Digest as _, Sha256};

use crate::Stored;
use crate::correctness::{
    ExpectedCorrectness, ReceiptDecision, ValidatedRoutes, parse_and_validate,
};

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// A strictly parsed config snapshot which has not yet passed its correctness receipt.
///
/// Its fields are private so the receipt path cannot accidentally hand an ordinary
/// [`Stored`] to an apply loop. The contract builder may only inspect the exact bytes
/// and parsed values from the loader's single filesystem read.
#[derive(Clone, Copy, Debug)]
pub struct UntrustedStoredConfig<'a> {
    exact_bytes: &'a [u8],
    stored: &'a Stored,
}

impl<'a> UntrustedStoredConfig<'a> {
    #[must_use]
    pub fn exact_bytes(&self) -> &'a [u8] {
        self.exact_bytes
    }

    #[must_use]
    pub fn stored(&self) -> &'a Stored {
        self.stored
    }
}

/// A config whose exact snapshot, numerical routes and required gates were admitted.
///
/// Construction stays private to this module. Callers can only obtain this authority
/// from the fail-closed loader and should retain the validated routes until application.
#[derive(Clone, Debug)]
pub struct ReceiptedStored {
    stored: Stored,
    validated_routes: ValidatedRoutes,
}

impl ReceiptedStored {
    #[must_use]
    pub fn stored(&self) -> &Stored {
        &self.stored
    }

    #[must_use]
    pub fn validated_routes(&self) -> &ValidatedRoutes {
        &self.validated_routes
    }

    #[must_use]
    pub fn into_parts(self) -> (Stored, ValidatedRoutes) {
        (self.stored, self.validated_routes)
    }
}

/// Sidecar path used by the correctness gate for one tuned config.
#[must_use]
pub fn receipt_path_for(config_path: &Path) -> PathBuf {
    let mut name = config_path
        .file_name()
        .map_or_else(|| OsString::from("tune"), ToOwned::to_owned);
    name.push(".receipt.json");
    config_path.with_file_name(name)
}

/// Canonical content identity used to bind a receipt to exact config bytes.
#[must_use]
pub fn config_sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .fold(String::with_capacity(64), |mut output, byte| {
            write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
            output
        })
}

/// Read a tuned config only after its sidecar correctness receipt admits the exact
/// config, complete execution fingerprint, numerical route and required gates.
///
/// `expected_for` receives the strictly parsed but explicitly untrusted candidate and
/// the exact bytes from the loader's single read. It must independently construct the
/// running model/backend contract and derive routes from those candidate values.
///
/// A missing config/receipt, incomplete runtime identity, any mismatch, or any failed
/// gate returns `None`; callers must retain compiled safe defaults.
#[must_use]
pub fn read_for_device_receipted(
    model_bytes: u64,
    quiet: bool,
    device: (&str, u32, &str),
    expected_for: impl FnOnce(
        &UntrustedStoredConfig<'_>,
    ) -> Result<ExpectedCorrectness, String>,
) -> Option<ReceiptedStored> {
    let fp = crate::fingerprint_for(device.0, device.1, device.2);
    let config_path = crate::selected_config_path(&fp, model_bytes);
    let receipt_path = receipt_path_for(&config_path);
    read_receipted_at(
        &config_path,
        &receipt_path,
        model_bytes,
        quiet,
        device,
        expected_for,
    )
}

/// Path-explicit form used by embedders and tests which manage their own config root.
/// The config file is read exactly once; the returned parsed values always correspond to
/// the bytes bound by the receipt, even if another process replaces the path meanwhile.
#[must_use]
pub fn read_receipted_at(
    config_path: &Path,
    receipt_path: &Path,
    model_bytes: u64,
    quiet: bool,
    device: (&str, u32, &str),
    expected_for: impl FnOnce(
        &UntrustedStoredConfig<'_>,
    ) -> Result<ExpectedCorrectness, String>,
) -> Option<ReceiptedStored> {
    let fallback = |reason: &str| {
        if !quiet {
            eprintln!(
                "[imparo] tuned config at {} rejected ({reason}); using safe defaults",
                config_path.display()
            );
        }
        None
    };

    let Ok(config_bytes) = fs::read(config_path) else {
        return fallback("config missing or unreadable");
    };
    let Ok(body) = std::str::from_utf8(&config_bytes) else {
        return fallback("config is not UTF-8");
    };
    let fp = crate::fingerprint_for(device.0, device.1, device.2);
    let Some(stored) = parse_strict(body, model_bytes, &fp) else {
        return fallback("malformed or mismatched config");
    };
    let candidate = UntrustedStoredConfig {
        exact_bytes: &config_bytes,
        stored: &stored,
    };
    let mut bound = match expected_for(&candidate) {
        Ok(expected) => expected,
        Err(reason) => return fallback(&format!("correctness contract: {reason}")),
    };
    if bound.fingerprint.backend != device.0 {
        return fallback("backend fingerprint mismatch");
    }
    if bound.fingerprint.numerical_space_version != device.1 {
        return fallback("numerical space-version mismatch");
    }
    if !kv_tag_matches(device.2, &bound.fingerprint.kv_k, &bound.fingerprint.kv_v) {
        return fallback("KV fingerprint mismatch");
    }

    let Ok(receipt) = fs::read_to_string(receipt_path) else {
        return fallback("correctness receipt missing or unreadable");
    };
    // Config content is filesystem state, not caller input. Bind the expected contract
    // to the bytes just parsed so the receipt must name this exact file.
    bound.fingerprint.config_sha256 = config_sha256(&config_bytes);
    match parse_and_validate(Some(&receipt), &bound) {
        ReceiptDecision::Admitted(validated_routes) => Some(ReceiptedStored {
            stored,
            validated_routes,
        }),
        ReceiptDecision::SafeFallback(reason) => {
            fallback(&format!("correctness receipt: {reason:?}"))
        }
    }
}

fn kv_tag_matches(tag: &str, k: &str, v: &str) -> bool {
    tag == crate::canonical_kv_tag(k, v)
}

/// Strictly read one explicit candidate only for an isolated correctness-gate process.
/// This does not validate a receipt and must never be used by a production apply path.
/// The CUDA caller additionally requires `IMPARO_CORRECTNESS_GATE=1` before calling it.
#[must_use]
pub fn read_unreceipted_for_gate_at(
    config_path: &Path,
    model_bytes: u64,
    device: (&str, u32, &str),
) -> Option<Stored> {
    let fp = crate::fingerprint_for(device.0, device.1, device.2);
    let Ok(config_bytes) = fs::read(config_path) else {
        return None;
    };
    let Ok(body) = std::str::from_utf8(&config_bytes) else {
        return None;
    };
    parse_strict(body, model_bytes, &fp)
}

/// Strictly inspect one explicit unreceipted candidate in an isolated correctness-gate
/// process. The bytes are read once and the policy receives the exact parsed snapshot.
/// Production loaders must use `read_for_device_receipted` instead.
pub fn inspect_unreceipted_for_gate_at<T>(
    config_path: &Path,
    model_bytes: u64,
    device: (&str, u32, &str),
    inspect: impl FnOnce(&UntrustedStoredConfig<'_>) -> Result<T, String>,
) -> Result<(Stored, T), String> {
    let fp = crate::fingerprint_for(device.0, device.1, device.2);
    let config_bytes = fs::read(config_path).map_err(|error| {
        format!("read gate candidate {}: {error}", config_path.display())
    })?;
    let body = std::str::from_utf8(&config_bytes)
        .map_err(|_| "gate candidate is not UTF-8".to_string())?;
    let stored = parse_strict(body, model_bytes, &fp)
        .ok_or_else(|| "gate candidate is malformed or mismatched".to_string())?;
    let candidate = UntrustedStoredConfig {
        exact_bytes: &config_bytes,
        stored: &stored,
    };
    let inspected = inspect(&candidate)?;
    Ok((stored, inspected))
}

fn parse_strict(body: &str, model_bytes: u64, fingerprint: &str) -> Option<Stored> {
    let mut stored = Stored::default();
    let mut stored_fp = None;
    let mut stored_model = None;
    let mut seen = BTreeSet::new();
    for raw in body.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (key, value) = line.split_once('=')?;
        let (key, value) = (key.trim(), value.trim());
        if key.is_empty() || value.is_empty() || !seen.insert(key.to_owned()) {
            return None;
        }
        match key {
            "fingerprint" => stored_fp = Some(value.to_owned()),
            // Descriptive fields, not knobs. Without an arm here the catch-all below runs
            // `value.parse::<u32>().ok()?` on a version string and REJECTS THE WHOLE FILE
            // -- a sealed receipt would stop loading the moment `toolchain` appeared.
            "model" | "toolchain" => {}
            "model_bytes" => stored_model = value.parse::<u64>().ok(),
            "batch" => stored.batch = value.parse::<usize>().ok(),
            _ if key.starts_with("measured_") => {}
            // Legacy, ignored: device measurements moved to their own file, keyed by
            // host and backend rather than by the knob space, cache type and model that
            // this file is keyed by. See `read_device_profile`.
            _ if key.starts_with("device_") => {}
            _ => stored.knobs.push((key.to_owned(), value.parse().ok()?)),
        }
    }
    if stored_fp.as_deref() != Some(fingerprint)
        || stored_model != Some(model_bytes)
        || stored.batch.is_none()
    {
        return None;
    }
    Some(stored)
}

/// Persist bytes by syncing a same-directory temporary and atomically replacing the
/// destination. A torn config/receipt pair is still safe because loading is fail-closed.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let mut name = path
        .file_name()
        .map_or_else(|| OsString::from("imparo"), ToOwned::to_owned);
    name.push(format!(".tmp-{}-{sequence}", std::process::id()));
    let temporary = parent.join(name);
    let result = (|| {
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        replace_file(&temporary, path)?;
        if let Ok(directory) = File::open(parent) {
            let _ = directory.sync_all();
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

#[cfg(windows)]
fn replace_file(source: &Path, destination: &Path) -> io::Result<()> {
    use std::os::windows::ffi::OsStrExt as _;
    unsafe extern "system" {
        fn MoveFileExW(existing: *const u16, target: *const u16, flags: u32) -> i32;
    }
    const MOVEFILE_REPLACE_EXISTING: u32 = 0x1;
    const MOVEFILE_WRITE_THROUGH: u32 = 0x8;
    let source: Vec<u16> = source.as_os_str().encode_wide().chain([0]).collect();
    let destination: Vec<u16> =
        destination.as_os_str().encode_wide().chain([0]).collect();
    // SAFETY: both are NUL-terminated UTF-16 paths that remain alive for the call.
    if unsafe {
        MoveFileExW(
            source.as_ptr(),
            destination.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    } == 0
    {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

#[cfg(not(windows))]
fn replace_file(source: &Path, destination: &Path) -> io::Result<()> {
    fs::rename(source, destination)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::correctness::*;
    use imparo_backend::numerical::{NumericalClass, RouteKey};

    fn hash(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn route() -> RouteKey {
        RouteKey {
            backend: "cuda".into(),
            operation: "attention.decode".into(),
            implementation: "d512.mma.stream-k".into(),
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

    fn expected(config: &[u8]) -> ExpectedCorrectness {
        ExpectedCorrectness {
            fingerprint: CorrectnessFingerprint {
                config_sha256: config_sha256(config),
                model_sha256: hash('1'),
                model_plan_sha256: hash('2'),
                kv_layout_sha256: hash('4'),
                platform: "windows-x86_64".into(),
                backend: "cuda".into(),
                device_uuid: "GPU-test".into(),
                device_sm: 86,
                driver_version: 13010,
                runtime_version: 13000,
                backend_abi: 19,
                backend_fingerprint_sha256: hash('3'),
                kv_k: "q4_0".into(),
                kv_v: "q4_0".into(),
                numerical_space_version: 19,
                selector_version: 7,
                math_mode: "cuda-no-fast-math".into(),
            },
            gate_suite: "cuda-step6".into(),
            gate_suite_version: 3,
            routes: vec![route()],
            required_gates: vec![GateRequirement {
                gate_id: "decode_agree".into(),
                gate_version: 1,
            }],
            oracle: OracleFingerprint {
                implementation: "zeraix/llama-cpp".into(),
                revision: "4695f001fece1660d8bb1b3748f50726ddcc100b".into(),
                options_sha256: hash('8'),
                bundle_manifest_sha256: hash('9'),
            },
        }
    }

    fn receipt(expected: &ExpectedCorrectness) -> CorrectnessReceipt {
        CorrectnessReceipt {
            schema_version: CORRECTNESS_RECEIPT_SCHEMA_VERSION,
            producer: "receipt_gate".into(),
            producer_version: 1,
            issued_unix_seconds: 1,
            fingerprint: expected.fingerprint.clone(),
            gate_suite: expected.gate_suite.clone(),
            gate_suite_version: expected.gate_suite_version,
            routes: expected.routes.iter().map(ReceiptRoute::from).collect(),
            gates: vec![GateEvidence {
                gate_id: "decode_agree".into(),
                gate_version: 1,
                passed: true,
                command_sha256: hash('4'),
                output_sha256: hash('5'),
            }],
            oracle: expected.oracle.clone(),
        }
    }

    fn fixture() -> (PathBuf, PathBuf, Vec<u8>, ExpectedCorrectness) {
        let root = std::env::temp_dir().join(format!(
            "imparo-receipt-test-{}-{}",
            std::process::id(),
            TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).unwrap();
        let config_path = root.join("tune.txt");
        let receipt_path = receipt_path_for(&config_path);
        let fp = crate::fingerprint_for("cuda", 19, "q4_0");
        let config = format!(
            "fingerprint={fp}\nmodel=E4B\nmodel_bytes=4096\nstreamk=1\nbatch=512\n"
        )
        .into_bytes();
        fs::write(&config_path, &config).unwrap();
        let expected = expected(&config);
        fs::write(
            &receipt_path,
            serde_json::to_vec(&receipt(&expected)).unwrap(),
        )
        .unwrap();
        (config_path, receipt_path, config, expected)
    }

    fn load(
        config: &Path,
        receipt: &Path,
        expected: &ExpectedCorrectness,
    ) -> Option<ReceiptedStored> {
        let expected = expected.clone();
        read_receipted_at(
            config,
            receipt,
            4096,
            true,
            ("cuda", 19, "q4_0"),
            move |_| Ok(expected),
        )
    }

    #[test]
    fn valid_receipt_loads_exact_config() {
        let (config, sidecar, _, expected) = fixture();
        let admitted =
            load(&config, &sidecar, &expected).expect("valid config rejected");
        let stored = admitted.stored();
        assert_eq!(stored.batch, Some(512));
        assert_eq!(stored.knobs, vec![("streamk".into(), 1)]);
        assert!(admitted.validated_routes().allows(&route()));
    }

    #[test]
    fn mixed_kv_tag_has_one_canonical_separator() {
        assert!(kv_tag_matches(
            &crate::canonical_kv_tag("q4_0", "f16"),
            "q4_0",
            "f16"
        ));
        assert!(!kv_tag_matches("q4_0-f16", "q4_0", "f16"));
    }

    #[test]
    fn missing_receipt_and_config_tamper_fall_back() {
        let (config, sidecar, _, expected) = fixture();
        fs::remove_file(&sidecar).unwrap();
        assert!(load(&config, &sidecar, &expected).is_none());
        fs::write(&sidecar, serde_json::to_vec(&receipt(&expected)).unwrap()).unwrap();
        fs::write(&config, b"malformed=true\n").unwrap();
        assert!(load(&config, &sidecar, &expected).is_none());
    }

    #[test]
    fn model_device_driver_abi_space_and_kv_mismatches_fall_back() {
        type Mutation = fn(&mut ExpectedCorrectness);
        let cases: [Mutation; 7] = [
            |v| v.fingerprint.model_sha256 = hash('9'),
            |v| v.fingerprint.device_uuid.push_str("-other"),
            |v| v.fingerprint.driver_version += 1,
            |v| v.fingerprint.backend_abi += 1,
            |v| v.fingerprint.numerical_space_version += 1,
            |v| v.fingerprint.kv_k = "f16".into(),
            |v| v.fingerprint.kv_v = "f16".into(),
        ];
        for mutate in cases {
            let (config, sidecar, _, mut expected) = fixture();
            mutate(&mut expected);
            assert!(load(&config, &sidecar, &expected).is_none());
        }
    }

    #[test]
    fn numeric_route_mismatch_and_failed_gate_fall_back() {
        let (config, sidecar, _, mut expected) = fixture();
        expected.routes[0].parameters_sha256 = hash('c');
        assert!(load(&config, &sidecar, &expected).is_none());

        let (config, sidecar, _, expected) = fixture();
        let mut rejected = receipt(&expected);
        rejected.gates[0].passed = false;
        fs::write(&sidecar, serde_json::to_vec(&rejected).unwrap()).unwrap();
        assert!(load(&config, &sidecar, &expected).is_none());
    }

    #[test]
    fn builder_sees_one_exact_strict_snapshot() {
        let (config, sidecar, original, original_expected) = fixture();
        let replacement = String::from_utf8(original.clone())
            .unwrap()
            .replace("streamk=1", "streamk=9")
            .into_bytes();

        let admitted = read_receipted_at(
            &config,
            &sidecar,
            4096,
            true,
            ("cuda", 19, "q4_0"),
            |candidate| {
                assert_eq!(candidate.exact_bytes(), original);
                assert_eq!(candidate.stored().knobs, vec![("streamk".into(), 1)]);
                // Simulate an atomic tuner replacement after the loader's one read.
                write_atomic(&config, &replacement).unwrap();
                Ok(original_expected.clone())
            },
        )
        .expect("the exact original snapshot and receipt should remain admissible");
        assert_eq!(admitted.stored().knobs, vec![("streamk".into(), 1)]);
        assert_eq!(fs::read(&config).unwrap(), replacement);

        // A later load sees the replacement bytes and rejects the old receipt.
        let replacement_expected = expected(&replacement);
        assert!(load(&config, &sidecar, &replacement_expected).is_none());
    }

    #[test]
    fn builder_is_not_called_for_a_malformed_candidate() {
        use std::cell::Cell;

        let (config, sidecar, _, _) = fixture();
        fs::write(&config, b"malformed=true\n").unwrap();
        let called = Cell::new(false);
        let admitted = read_receipted_at(
            &config,
            &sidecar,
            4096,
            true,
            ("cuda", 19, "q4_0"),
            |_| {
                called.set(true);
                Err("must not run".into())
            },
        );
        assert!(admitted.is_none());
        assert!(!called.get());
    }

    #[test]
    fn contract_builder_failure_is_fail_closed() {
        let (config, sidecar, _, _) = fixture();
        assert!(
            read_receipted_at(
                &config,
                &sidecar,
                4096,
                true,
                ("cuda", 19, "q4_0"),
                |_| Err("runtime identity unavailable".into()),
            )
            .is_none()
        );
    }

    #[test]
    fn atomic_writer_replaces_existing_file_without_temp_debris() {
        let (config, _, _, _) = fixture();
        write_atomic(&config, b"new-body").unwrap();
        assert_eq!(fs::read(&config).unwrap(), b"new-body");
        let parent = config.parent().unwrap();
        assert!(fs::read_dir(parent).unwrap().all(|entry| {
            !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .contains(".tmp-")
        }));
    }
}
