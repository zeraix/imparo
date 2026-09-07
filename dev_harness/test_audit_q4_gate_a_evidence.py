from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from dev_harness.audit_q4_gate_a_evidence import (
    Q4StrictAuditError,
    _resources,
    _timing,
    _wrong_grid,
    audit,
    validate_device,
)
from dev_harness.finalize_q4_sanitizer_evidence import load_json, sha256_file
from dev_harness.finalize_q4_sanitizer_evidence import validate_matrix


REPO = Path(__file__).resolve().parents[1]
POLICY = REPO / "config/kernel-lab-policy.toml"
FORMAL = REPO / "artifacts/kernel-lab/q4-variants-perf/variants-b/formal-matrix-evidence.json"
SANITIZER = REPO / "artifacts/kernel-lab/q4-variants-sanitizer/variants-c/sanitizer-evidence-v2.json"
LEGACY_SANITIZER = REPO / "artifacts/kernel-lab/q4-variants-sanitizer/variants-b/sanitizer-evidence.json"
METADATA = REPO / "program-packs/lab/q4-q8-mmq-sm86/variants-b/lab-metadata.json"
SOURCE = REPO / "crates/imparo-cuda/native/tests/kernel_lab_q4_q8.cu"
WITNESS = REPO / "artifacts/kernel-lab/q4-variants-perf/variants-b/environment-witness.json"


@unittest.skipUnless(
    all(path.is_file() for path in (
        POLICY, FORMAL, SANITIZER, LEGACY_SANITIZER, METADATA, SOURCE, WITNESS
    )),
    "local Q4 evidence unavailable",
)
class Q4StrictAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = __import__("tomllib").loads(POLICY.read_text(encoding="utf-8"))
        metadata = load_json(METADATA)
        cls.expected = {}
        for item in metadata["survivors"]:
            tile, resources = item["tile"], item["resources"]
            cls.expected[(item["lab_label"], item["shape_id"])] = {
                **item,
                "block_m": tile["rows"], "block_n": tile["tokens"],
                "threads": item["num_warps"] * 32,
                "dynamic_shared_bytes": resources["dynamic_shared_bytes"],
                "registers_per_thread": resources["registers_per_thread"],
            }
        cls.result = load_json(
            REPO / "artifacts/kernel-lab/q4-variants-perf/variants-b/bm64-bn64-w8-s1/stdout.json"
        )
        cls.contraction = next(
            case for case in cls.result["cases"]
            if case["n_tok"] == 512 and case["shape_id"] == "k10240-m2560"
        )

    def _audit(self, *, sanitizer: Path = SANITIZER, witness: Path = WITNESS):
        return audit(
            policy_path=POLICY, expected_policy_sha256=sha256_file(POLICY),
            formal_matrix_path=FORMAL, expected_formal_matrix_sha256=sha256_file(FORMAL),
            sanitizer_manifest_path=sanitizer,
            expected_sanitizer_manifest_sha256=sha256_file(sanitizer),
            metadata_path=METADATA, expected_metadata_sha256=sha256_file(METADATA),
            harness_source_path=SOURCE, expected_harness_source_sha256=sha256_file(SOURCE),
            environment_witness_path=witness,
            expected_environment_witness_sha256=sha256_file(witness),
            expected_min_kernel_speedup="1.10",
            expected_min_projected_e2e="0.03",
        )

    def test_current_raw_evidence_passes_strict_audit(self) -> None:
        result = self._audit()
        self.assertEqual(result["device"]["sm"], 86)
        self.assertEqual(result["device"]["sm_count"], 30)
        self.assertEqual(result["sanitizer"]["schema"], 2)
        self.assertEqual(result["sanitizer"]["run_count"], 16)
        self.assertEqual(result["sanitizer"]["preflight_count"], 8)

    def test_incomplete_device_rejected(self) -> None:
        device = copy.deepcopy(self.result["device"])
        device["driver_version"] = 0
        with self.assertRaisesRegex(Q4StrictAuditError, "device identity is invalid"):
            validate_device(device, "fixture")

    def test_sample_summary_tamper_rejected(self) -> None:
        case = copy.deepcopy(self.contraction)
        case["timing"]["native_mean_us"] += 0.01
        with self.assertRaisesRegex(Q4StrictAuditError, "differs from samples"):
            _timing(case, self.policy["noise"], "fixture")

    def test_paired_mad_tamper_rejected(self) -> None:
        case = copy.deepcopy(self.contraction)
        case["timing"]["paired_mad_fraction"] += 0.001
        with self.assertRaisesRegex(Q4StrictAuditError, "differs from samples"):
            _timing(case, self.policy["noise"], "fixture")

    def test_vram_formula_tamper_rejected(self) -> None:
        case = copy.deepcopy(self.contraction)
        case["artifact_resources"]["observed_peak_delta_bytes"] += 1
        expected = self.expected[("bm64-bn64-w8-s1", "k10240-m2560")]
        with self.assertRaisesRegex(Q4StrictAuditError, "VRAM delta formula mismatch"):
            _resources(case, expected, self.policy, "fixture")

    def test_resource_limit_tamper_rejected(self) -> None:
        case = copy.deepcopy(self.contraction)
        case["artifact_resources"]["triton_function"]["local_bytes"] = 1
        expected = self.expected[("bm64-bn64-w8-s1", "k10240-m2560")]
        with self.assertRaisesRegex(Q4StrictAuditError, "function resource limit failed"):
            _resources(case, expected, self.policy, "fixture")

    def test_wrong_grid_actual_gpu_difference_required(self) -> None:
        case = copy.deepcopy(self.contraction)
        case["cpu_oracle"]["wrong_grid_gpu_vs_correct"]["max_abs"] = 0
        expected = self.expected[("bm64-bn64-w8-s1", "k10240-m2560")]
        with self.assertRaisesRegex(Q4StrictAuditError, "actual GPU wrong-grid"):
            _wrong_grid(case, expected, "fixture")

    def test_wrong_grid_nan_rejected(self) -> None:
        case = copy.deepcopy(self.contraction)
        case["cpu_oracle"]["wrong_grid_gpu_vs_correct"]["max_abs"] = float("nan")
        expected = self.expected[("bm64-bn64-w8-s1", "k10240-m2560")]
        with self.assertRaisesRegex(Q4StrictAuditError, "actual GPU wrong-grid"):
            _wrong_grid(case, expected, "fixture")

    def test_wrong_grid_vs_native_zero_rejected(self) -> None:
        case = copy.deepcopy(self.contraction)
        case["cpu_oracle"]["wrong_grid_gpu_vs_native"].update(
            max_abs=0, bitwise_different=0
        )
        expected = self.expected[("bm64-bn64-w8-s1", "k10240-m2560")]
        with self.assertRaisesRegex(Q4StrictAuditError, "actual GPU wrong-grid"):
            _wrong_grid(case, expected, "fixture")

    def test_legacy_preflight_false_check_rejected(self) -> None:
        # Preserve the legacy parser's all-checks-true negative coverage without
        # treating schema 1 as the current Gate-A authority.
        manifest = load_json(LEGACY_SANITIZER)
        manifest["preflight"][0]["checks"]["module_sha256"] = False
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                Exception, "preflight differs from variant metadata/resources"
            ):
                validate_matrix(path, POLICY)

    def test_legacy_harness_hash_tamper_rejected(self) -> None:
        # Schema 1 remains testable as historical input, but never supplies
        # the current witness authority used by _audit().
        manifest = load_json(LEGACY_SANITIZER)
        first, second = manifest["runs"][0], manifest["runs"][1]
        second["harness_output_sha256"] = first["harness_output_sha256"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(Exception, "run.harness SHA-256 mismatch"):
                validate_matrix(path, POLICY)

    def test_legacy_schema1_cannot_satisfy_current_authority(self) -> None:
        with self.assertRaisesRegex(
            Q4StrictAuditError, "current environment witness authority differs"
        ):
            self._audit(sanitizer=LEGACY_SANITIZER)

    def test_current_witness_device_tamper_rejected(self) -> None:
        witness = load_json(WITNESS)
        witness["device"]["runtime_version"] = witness["device"].pop("cuda_runtime_version")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "witness.json"
            path.write_text(json.dumps(witness), encoding="utf-8")
            with self.assertRaisesRegex(Q4StrictAuditError, "device key set"):
                self._audit(witness=path)

    def test_arbitrary_tool_version_string_rejected(self) -> None:
        witness = load_json(WITNESS)
        witness["local_toolchain"]["nvcc"] = "arbitrary non-empty"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "witness.json"
            path.write_text(json.dumps(witness), encoding="utf-8")
            with self.assertRaisesRegex(Q4StrictAuditError, "tool binary/version differs"):
                self._audit(witness=path)

    def test_external_policy_root_tamper_rejected(self) -> None:
        with self.assertRaisesRegex(Q4StrictAuditError, "policy hash mismatch"):
            audit(
                policy_path=POLICY, expected_policy_sha256="0" * 64,
                formal_matrix_path=FORMAL, expected_formal_matrix_sha256=sha256_file(FORMAL),
                sanitizer_manifest_path=SANITIZER,
                expected_sanitizer_manifest_sha256=sha256_file(SANITIZER),
                metadata_path=METADATA, expected_metadata_sha256=sha256_file(METADATA),
                harness_source_path=SOURCE, expected_harness_source_sha256=sha256_file(SOURCE),
                environment_witness_path=WITNESS,
                expected_environment_witness_sha256=sha256_file(WITNESS),
                expected_min_kernel_speedup="1.10",
                expected_min_projected_e2e="0.03",
            )

    def test_reviewed_floor_mismatch_rejected(self) -> None:
        with self.assertRaisesRegex(Q4StrictAuditError, "reviewed Decision-A profile"):
            audit(
                policy_path=POLICY, expected_policy_sha256=sha256_file(POLICY),
                formal_matrix_path=FORMAL,
                expected_formal_matrix_sha256=sha256_file(FORMAL),
                sanitizer_manifest_path=SANITIZER,
                expected_sanitizer_manifest_sha256=sha256_file(SANITIZER),
                metadata_path=METADATA, expected_metadata_sha256=sha256_file(METADATA),
                harness_source_path=SOURCE,
                expected_harness_source_sha256=sha256_file(SOURCE),
                environment_witness_path=WITNESS,
                expected_environment_witness_sha256=sha256_file(WITNESS),
                expected_min_kernel_speedup="1.09",
                expected_min_projected_e2e="0.03",
            )


if __name__ == "__main__":
    unittest.main()
