from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dev_harness.aggregate_triton_gate_a import sha256_file
from dev_harness.finalize_triton_gate_a_decision import (
    DecisionAError,
    decide_no_go_from_candidates,
    finalize,
)
from dev_harness.finalize_q4_gate_a_evidence import finalize as finalize_q4
from dev_harness.test_validate_q4_sanitizer_schema2 import build_schema2_fixture


REPO = Path(__file__).resolve().parents[1]
POLICY = REPO / "config/kernel-lab-policy.toml"
PHASE_A1 = REPO / "docs/evidence/triton/lab-a-phase-a1.md"
RMS = REPO / "artifacts/kernel-lab/rms-q8-sm86/run-g-batch16/gate-a-decision-finalized.json"
FORMAL = REPO / "artifacts/kernel-lab/q4-variants-perf/variants-b/formal-matrix-evidence.json"
SANITIZER = REPO / "artifacts/kernel-lab/q4-variants-sanitizer/variants-c/sanitizer-evidence-v2.json"
LEGACY_SANITIZER = REPO / "artifacts/kernel-lab/q4-variants-sanitizer/variants-b/sanitizer-evidence.json"
METADATA = REPO / "program-packs/lab/q4-q8-mmq-sm86/variants-b/lab-metadata.json"
SOURCE = REPO / "crates/imparo-cuda/native/tests/kernel_lab_q4_q8.cu"
WITNESS = REPO / "artifacts/kernel-lab/q4-variants-perf/variants-b/environment-witness.json"
INPUTS = (
    POLICY, PHASE_A1, RMS, FORMAL, SANITIZER, LEGACY_SANITIZER,
    METADATA, SOURCE, WITNESS,
)


@unittest.skipUnless(all(path.is_file() for path in INPUTS), "local Gate-A evidence unavailable")
class DecisionAFinalizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = tempfile.TemporaryDirectory()
        cls.q4 = Path(cls.directory.name) / "q4-decision.json"
        record = finalize_q4(
            formal_matrix_path=FORMAL, sanitizer_manifest_path=SANITIZER,
            policy_path=POLICY, harness_source_path=SOURCE,
            environment_witness_path=WITNESS,
            expected_policy_sha256=sha256_file(POLICY),
            expected_formal_matrix_sha256=sha256_file(FORMAL),
            expected_sanitizer_manifest_sha256=sha256_file(SANITIZER),
            expected_metadata_sha256=sha256_file(METADATA),
            expected_harness_source_sha256=sha256_file(SOURCE),
           expected_environment_witness_sha256=sha256_file(WITNESS),
            expected_min_kernel_speedup="1.10",
            expected_min_projected_e2e="0.03",
        )
        cls.q4.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()

    def _finalize(self, q4_path: Path | None = None, q4_sha: str | None = None):
        q4_path = q4_path or self.q4
        return finalize(
            policy_path=POLICY, policy_sha256=sha256_file(POLICY),
            phase_a1_path=PHASE_A1, phase_a1_sha256=sha256_file(PHASE_A1),
            rms_decision_path=RMS, rms_decision_sha256=sha256_file(RMS),
            q4_decision_path=q4_path, q4_decision_sha256=q4_sha or sha256_file(q4_path),
            q4_formal_matrix_path=FORMAL,
            q4_sanitizer_manifest_path=SANITIZER,
            q4_metadata_path=METADATA,
            q4_harness_source_path=SOURCE,
            q4_environment_witness_path=WITNESS,
            q4_formal_matrix_sha256=sha256_file(FORMAL),
            q4_sanitizer_manifest_sha256=sha256_file(SANITIZER),
            q4_metadata_sha256=sha256_file(METADATA),
            q4_harness_source_sha256=sha256_file(SOURCE),
            q4_environment_witness_sha256=sha256_file(WITNESS),
            expected_min_kernel_speedup="1.10",
            expected_min_projected_e2e="0.03",
        )

    def test_q4_floor_failure_issues_no_go_and_rms_win_cannot_override(self) -> None:
        record = {
            "correctness_admissible": True,
            "measurement_admissible": True,
            "sanitizer_admissible": True,
            "provenance_admissible": True,
            "kernel_floor_met": False,
            "candidate_admissible": False,
        }
        self.assertEqual(
            decide_no_go_from_candidates(
                rms_candidate_admissible=True, q4_record=record
            ),
            "mandatory-high-share-q4-kernel-floor-failed",
        )

    def test_current_schema2_authority_issues_decision_a_no_go(self) -> None:
        decision = self._finalize()
        self.assertEqual(decision["decision"], "gate-a-no-go")
        self.assertFalse(decision["gate_a_go"])
        self.assertTrue(decision["stop_phase_b_c_d"])

        record = json.loads(self.q4.read_text(encoding="utf-8"))
        self.assertTrue(record["sanitizer_admissible"])
        self.assertTrue(record["provenance_admissible"])
        self.assertEqual(record["sanitizer_authority_schema"], 2)
        self.assertFalse(record["kernel_floor_met"])
        self.assertFalse(record["candidate_admissible"])
        self.assertEqual(len(record["violations"]), 1)

    def test_legacy_schema1_cannot_issue_current_q4_authority(self) -> None:
        with self.assertRaisesRegex(
            Exception, "current environment witness authority differs"
        ):
            finalize_q4(
                formal_matrix_path=FORMAL,
                sanitizer_manifest_path=LEGACY_SANITIZER,
                policy_path=POLICY,
                harness_source_path=SOURCE,
                environment_witness_path=WITNESS,
                expected_policy_sha256=sha256_file(POLICY),
                expected_formal_matrix_sha256=sha256_file(FORMAL),
                expected_sanitizer_manifest_sha256=sha256_file(LEGACY_SANITIZER),
                expected_metadata_sha256=sha256_file(METADATA),
                expected_harness_source_sha256=sha256_file(SOURCE),
                expected_environment_witness_sha256=sha256_file(WITNESS),
                expected_min_kernel_speedup="1.10",
                expected_min_projected_e2e="0.03",
            )

    def test_q4_decision_hash_tamper_is_rejected(self) -> None:
        with self.assertRaisesRegex(Exception, "hash mismatch"):
            self._finalize(q4_sha="0" * 64)

    def test_schema2_q4_authority_issues_decision_a_no_go(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sanitizer, witness = build_schema2_fixture(root)
            q4_path = root / "q4-decision.json"
            q4_record = finalize_q4(
                formal_matrix_path=FORMAL, sanitizer_manifest_path=sanitizer,
                policy_path=POLICY, harness_source_path=SOURCE,
                environment_witness_path=witness,
                expected_policy_sha256=sha256_file(POLICY),
                expected_formal_matrix_sha256=sha256_file(FORMAL),
                expected_sanitizer_manifest_sha256=sha256_file(sanitizer),
                expected_metadata_sha256=sha256_file(METADATA),
                expected_harness_source_sha256=sha256_file(SOURCE),
                expected_environment_witness_sha256=sha256_file(witness),
                expected_min_kernel_speedup="1.10",
                expected_min_projected_e2e="0.03",
            )
            q4_path.write_text(
                json.dumps(q4_record, sort_keys=True) + "\n", encoding="utf-8"
            )
            decision = finalize(
                policy_path=POLICY, policy_sha256=sha256_file(POLICY),
                phase_a1_path=PHASE_A1, phase_a1_sha256=sha256_file(PHASE_A1),
                rms_decision_path=RMS, rms_decision_sha256=sha256_file(RMS),
                q4_decision_path=q4_path, q4_decision_sha256=sha256_file(q4_path),
                q4_formal_matrix_path=FORMAL,
                q4_sanitizer_manifest_path=sanitizer,
                q4_metadata_path=METADATA, q4_harness_source_path=SOURCE,
                q4_environment_witness_path=witness,
                q4_formal_matrix_sha256=sha256_file(FORMAL),
                q4_sanitizer_manifest_sha256=sha256_file(sanitizer),
                q4_metadata_sha256=sha256_file(METADATA),
                q4_harness_source_sha256=sha256_file(SOURCE),
                q4_environment_witness_sha256=sha256_file(witness),
                expected_min_kernel_speedup="1.10",
                expected_min_projected_e2e="0.03",
            )
            self.assertEqual(decision["decision"], "gate-a-no-go")
            self.assertFalse(decision["gate_a_go"])
            self.assertTrue(decision["stop_phase_b_c_d"])

    def test_missing_mandatory_failure_cannot_be_relabelled(self) -> None:
        record = json.loads(self.q4.read_text(encoding="utf-8"))
        record["violations"] = record["violations"][:-1]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "q4-relabelled.json"
            path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(DecisionAError, "differs from recomputed"):
                self._finalize(path)

    def test_performance_failure_cannot_be_flipped_to_pass(self) -> None:
        record = json.loads(self.q4.read_text(encoding="utf-8"))
        record["kernel_floor_met"] = True
        record["candidate_admissible"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "q4-flipped.json"
            path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(DecisionAError, "differs from recomputed"):
                self._finalize(path)


if __name__ == "__main__":
    unittest.main()
