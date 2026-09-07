from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dev_harness.finalize_q4_gate_a_evidence import (
    Q4GateEvidenceError,
    finalize,
    select_best_pair,
)
from dev_harness.finalize_q4_sanitizer_evidence import (
    Q4SanitizerEvidenceError,
    sha256_file,
    validate_matrix,
)


REPO = Path(__file__).resolve().parents[1]
POLICY = REPO / "config/kernel-lab-policy.toml"
SANITIZER = (
    REPO
    / "artifacts/kernel-lab/q4-variants-sanitizer/variants-c/sanitizer-evidence-v2.json"
)
LEGACY_SANITIZER = (
    REPO
    / "artifacts/kernel-lab/q4-variants-sanitizer/variants-b/sanitizer-evidence.json"
)
FORMAL = REPO / "artifacts/kernel-lab/q4-variants-perf/variants-b/formal-matrix-evidence.json"
METADATA = REPO / "program-packs/lab/q4-q8-mmq-sm86/variants-b/lab-metadata.json"
SOURCE = REPO / "crates/imparo-cuda/native/tests/kernel_lab_q4_q8.cu"
WITNESS = REPO / "artifacts/kernel-lab/q4-variants-perf/variants-b/environment-witness.json"


@unittest.skipUnless(
    all(path.is_file() for path in (
        POLICY, SANITIZER, LEGACY_SANITIZER, FORMAL, SOURCE, WITNESS
    )),
    "local immutable Q4 Gate-A raw evidence is unavailable",
)
class Q4GateEvidenceFinalizerTests(unittest.TestCase):
    def test_pair_floor_requires_both_shapes_but_other_variants_do_not_veto(self) -> None:
        formal = [
            {"label": "winner", "timings": {
                "exp": {"native_over_triton": 1.20},
                "con": {"native_over_triton": 1.15},
            }},
            {"label": "loser", "timings": {
                "exp": {"native_over_triton": 0.20},
                "con": {"native_over_triton": 0.30},
            }},
        ]
        label, score, _, passed = select_best_pair(formal, 1.10)
        self.assertEqual(label, "winner")
        self.assertAlmostEqual(score, 1.15)
        self.assertTrue(passed)

    def test_one_fast_shape_cannot_pass_pair_floor(self) -> None:
        formal = [{"label": "split", "timings": {
            "exp": {"native_over_triton": 1.50},
            "con": {"native_over_triton": 1.00},
        }}]
        _, score, _, passed = select_best_pair(formal, 1.10)
        self.assertAlmostEqual(score, 1.00)
        self.assertFalse(passed)

    def _finalize(
        self,
        *,
        formal: Path = FORMAL,
        formal_sha: str | None = None,
        sanitizer: Path = SANITIZER,
    ):
        return finalize(
            formal_matrix_path=formal,
            sanitizer_manifest_path=sanitizer,
            policy_path=POLICY,
            harness_source_path=SOURCE,
            environment_witness_path=WITNESS,
            expected_policy_sha256=sha256_file(POLICY),
            expected_formal_matrix_sha256=formal_sha or sha256_file(formal),
            expected_sanitizer_manifest_sha256=sha256_file(sanitizer),
            expected_metadata_sha256=sha256_file(METADATA),
            expected_harness_source_sha256=sha256_file(SOURCE),
            expected_environment_witness_sha256=sha256_file(WITNESS),
            expected_min_kernel_speedup="1.10",
            expected_min_projected_e2e="0.03",
        )

    def test_complete_raw_matrix_finalizes_as_kernel_floor_no_go(self) -> None:
        raw = (POLICY, SANITIZER, FORMAL, SOURCE, WITNESS)
        before = {path: sha256_file(path) for path in raw}
        decision = self._finalize()
        self.assertEqual(before, {path: sha256_file(path) for path in raw})
        self.assertTrue(decision["correctness_admissible"])
        self.assertTrue(decision["measurement_admissible"])
        self.assertTrue(decision["sanitizer_admissible"])
        self.assertTrue(decision["provenance_admissible"])
        self.assertEqual(decision["sanitizer_authority_schema"], 2)
        self.assertFalse(decision["kernel_floor_met"])
        self.assertFalse(decision["candidate_admissible"])
        self.assertFalse(decision["final_gate_a_decision"])
        self.assertEqual(decision["sanitizer_matrix"]["run_count"], 16)
        self.assertEqual(decision["sanitizer_matrix"]["preflight_count"], 8)
        self.assertEqual(decision["best_observed_pair"]["label"], "bm64-bn64-w8-s1")
        self.assertAlmostEqual(
            decision["best_observed_pair"]["pair_score_min_native_over_triton"],
            0.34197208573254223,
        )
        self.assertTrue(any(
            item.startswith("mandatory high-share Q4 kernel speedup floor failed")
            for item in decision["violations"]
        ))
        self.assertFalse(any("schema 2 variants-c" in item for item in decision["violations"]))
        self.assertEqual(len(decision["violations"]), 1)

    def test_legacy_schema1_cannot_be_current_authority(self) -> None:
        with self.assertRaisesRegex(
            Exception, "current environment witness authority differs"
        ):
            self._finalize(sanitizer=LEGACY_SANITIZER)

    def test_missing_sanitizer_cartesian_run_is_rejected(self) -> None:
        manifest = json.loads(LEGACY_SANITIZER.read_text(encoding="utf-8"))
        manifest["runs"].pop()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sanitizer-missing-run.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                Q4SanitizerEvidenceError, "exactly 16 runs"
            ):
                validate_matrix(path, POLICY)

    def test_sanitizer_harness_hash_tamper_is_rejected(self) -> None:
        manifest = json.loads(LEGACY_SANITIZER.read_text(encoding="utf-8"))
        manifest["runs"][0]["harness_output_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sanitizer-bad-harness-hash.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                Q4SanitizerEvidenceError, "run.harness SHA-256 mismatch"
            ):
                validate_matrix(path, POLICY)

    def test_missing_sanitizer_zero_summary_is_rejected(self) -> None:
        manifest = json.loads(LEGACY_SANITIZER.read_text(encoding="utf-8"))
        manifest["runs"][0]["summary_zero"] = False
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sanitizer-missing-summary.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                Q4SanitizerEvidenceError, "sanitizer status did not pass"
            ):
                validate_matrix(path, POLICY)

    def test_formal_aggregate_median_tamper_is_rejected(self) -> None:
        matrix = json.loads(FORMAL.read_text(encoding="utf-8"))
        matrix["runs"][0]["expansion_triton_median_us"] += 1.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "formal-bad-median.json"
            path.write_text(json.dumps(matrix), encoding="utf-8")
            with self.assertRaisesRegex(Exception, "environment witness authority differs"):
                self._finalize(formal=path)


if __name__ == "__main__":
    unittest.main()
