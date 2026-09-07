from __future__ import annotations

import copy
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from dev_harness.aggregate_triton_gate_a import GateInputError, aggregate


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DecisionAAggregationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.policy = self.root / "kernel-lab-policy.toml"
        self.policy.write_text(
            "[gate]\nmin_projected_e2e_improvement = 0.03\n",
            encoding="utf-8",
        )
        self.phase_a1 = self.root / "lab-a-phase-a1.md"
        self.phase_a1.write_text(
            "# Fixture\n\n"
            "The two plain epilogue-0 directions account for the following measured logical\n"
            "matmul time. This denominator is not whole-prefill time.\n\n"
            "| Q4_0 logical shape | epilogue | calls | total ms | matmul share | median us |\n"
            "| --- | ---: | ---: | ---: | ---: | ---: |\n"
            "| 10240 -> 2560 x 512 | 0 | 42 | 46.53 | 25.71% | 1105.92 |\n"
            "| 2560 -> 10240 x 512 | 1 | 42 | 46.47 | 25.68% | 1040.38 |\n"
            "| 2560 -> 10240 x 512 | 0 | 42 | 36.98 | 20.43% | 886.78 |\n",
            encoding="utf-8",
        )
        self.device = {
            "uuid": "fixture-device",
            "name": "RTX 3060 fixture",
            "sm": 86,
            "sm_count": 30,
            "driver_version": 13020,
            "cuda_runtime_version": 12090,
        }

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _result(self, family: str, timings: dict[str, tuple[float, float]]) -> Path:
        result_path = self.root / family / "result.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        cases = [
            {
                "shape_id": shape,
                "n_tok": 512,
                "timing": {
                    "native_median_us": native,
                    "triton_median_us": triton,
                },
            }
            for shape, (native, triton) in timings.items()
        ]
        result_path.write_text(
            json.dumps({"schema": 1, "cases": cases}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return result_path

    def _decision(
        self,
        family: str,
        timings: dict[str, tuple[float, float]],
        *,
        decision_updates: dict[str, object] | None = None,
        provenance_updates: dict[str, object] | None = None,
    ) -> tuple[Path, str]:
        result_path = self._result(family, timings)
        decision_path = self.root / family / "gate-a-decision.json"
        sanitizer: dict[str, object] = {}
        for tool in ("memcheck", "initcheck", "racecheck", "synccheck"):
            log_path = self.root / family / f"compute-sanitizer-{tool}.log"
            log_path.write_text(f"{tool}: zero errors\n", encoding="utf-8")
            sanitizer[tool] = {
                "path": str(log_path),
                "sha256": file_sha256(log_path),
                "exit_code": 0,
                "pass": True,
            }
        provenance: dict[str, object] = {
            "policy_sha256": file_sha256(self.policy),
            "metadata_sha256": "1" * 64,
            "harness_source_sha256": "2" * 64,
            "harness_exe_sha256": "3" * 64,
            "result_json_sha256": file_sha256(result_path),
            "cubins": [{"shape": "fixture", "sha256": "4" * 64, "bytes": 1024}],
            "device": copy.deepcopy(self.device),
            "local_toolchain": {"nvcc": "fixture", "target_sm": 86},
            "builder": {"triton": "fixture", "semantic_recipe_sha256": "5" * 64},
        }
        if provenance_updates:
            provenance.update(provenance_updates)
        decision: dict[str, object] = {
            "schema": 1,
            "decision": "gate-a-candidate",
            "candidate": (
                "rms_norm_q8_1_mmq"
                if family == "rms_q8"
                else "q4_0_x_q8_1_mmq_epilogue0"
            ),
            "production_authority": False,
            "final_gate_a_decision": False,
            "correctness_admissible": True,
            "measurement_admissible": True,
            "candidate_admissible": True,
            "kernel_floor_met": True,
            "violations": [],
            "provenance": provenance,
            "sanitizer": sanitizer,
            "result": str(result_path),
        }
        if decision_updates:
            decision.update(decision_updates)
        decision_path.write_text(
            json.dumps(decision, sort_keys=True) + "\n", encoding="utf-8"
        )
        return decision_path, file_sha256(decision_path)

    def _aggregate(
        self,
        *,
        rms: tuple[Path, str],
        q4: tuple[Path, str],
        rms_sha256: str | None = None,
    ) -> dict[str, object]:
        return aggregate(
            policy_path=self.policy,
            policy_sha256=file_sha256(self.policy),
            phase_a1_path=self.phase_a1,
            phase_a1_sha256=file_sha256(self.phase_a1),
            rms_decision_path=rms[0],
            rms_decision_sha256=rms_sha256 or rms[1],
            q4_decision_path=q4[0],
            q4_decision_sha256=q4[1],
        )

    def test_projects_both_epilogue0_shapes_but_does_not_issue_gate(self) -> None:
        rms = self._decision("rms_q8", {})
        q4 = self._decision(
            "q4_mmq",
            {
                "k10240-m2560": (1000.0, 800.0),
                "k2560-m10240": (1200.0, 900.0),
            },
        )
        report = self._aggregate(rms=rms, q4=q4)
        prerequisites = report["candidate_prerequisites"]
        self.assertTrue(prerequisites["rms_q8_complete"])
        self.assertTrue(prerequisites["q4_mmq_complete"])
        self.assertTrue(prerequisites["same_device"])
        projection = report["projection"]
        expected = 0.2571 * 0.20 + 0.2043 * 0.25
        self.assertTrue(
            math.isclose(
                projection["combined_phase_a1_logical_matmat_share"],
                0.4614,
                abs_tol=1e-12,
            )
        )
        self.assertTrue(
            math.isclose(projection["logical_matmat_projection"], expected, abs_tol=1e-12)
        )
        self.assertTrue(projection["logical_matmat_proxy_floor_met"])
        self.assertIsNone(projection["projected_whole_prefill_improvement"])
        self.assertFalse(projection["projected_whole_prefill_floor_evaluable"])
        self.assertFalse(projection["projected_whole_prefill_floor_met"])
        self.assertEqual(report["decision"], "not-issued")
        self.assertFalse(report["final_gate_a_decision"])
        self.assertFalse(report["decision_inputs_complete"])
        self.assertIn(
            "whole-prefill denominator is absent from current Phase A1 evidence",
            report["blocking_reasons"],
        )

    def test_rms_micro_win_cannot_replace_high_share_q4_candidate(self) -> None:
        rms = self._decision("rms_q8", {})
        q4 = self._decision(
            "q4_mmq",
            {
                "k10240-m2560": (1000.0, 800.0),
                "k2560-m10240": (1200.0, 900.0),
            },
            decision_updates={"candidate_admissible": False},
        )
        report = self._aggregate(rms=rms, q4=q4)
        self.assertTrue(report["rms_micro_win_cannot_authorize_gate_a"])
        self.assertTrue(report["high_share_q4_candidate_required"])
        self.assertFalse(report["candidate_prerequisites"]["q4_mmq_complete"])
        self.assertFalse(report["decision_inputs_complete"])

    def test_missing_sanitizer_tool_fails_closed(self) -> None:
        rms = self._decision("rms_q8", {})
        q4_path, _ = self._decision(
            "q4_mmq",
            {
                "k10240-m2560": (1000.0, 800.0),
                "k2560-m10240": (1200.0, 900.0),
            },
        )
        q4_record = json.loads(q4_path.read_text(encoding="utf-8"))
        del q4_record["sanitizer"]["racecheck"]
        q4_path.write_text(json.dumps(q4_record, sort_keys=True) + "\n", encoding="utf-8")
        report = self._aggregate(rms=rms, q4=(q4_path, file_sha256(q4_path)))
        self.assertFalse(report["candidate_prerequisites"]["q4_mmq_complete"])
        self.assertIn(
            "exact four-tool sanitizer evidence is missing",
            report["candidate_prerequisites"]["q4_mmq_violations"],
        )

    def test_tampered_decision_hash_is_rejected(self) -> None:
        rms = self._decision("rms_q8", {})
        q4 = self._decision(
            "q4_mmq",
            {
                "k10240-m2560": (1000.0, 800.0),
                "k2560-m10240": (1200.0, 900.0),
            },
        )
        with self.assertRaisesRegex(GateInputError, "hash mismatch"):
            self._aggregate(rms=rms, q4=q4, rms_sha256="0" * 64)

    def test_result_hash_mismatch_fails_candidate_prerequisite(self) -> None:
        rms = self._decision("rms_q8", {})
        q4 = self._decision(
            "q4_mmq",
            {
                "k10240-m2560": (1000.0, 800.0),
                "k2560-m10240": (1200.0, 900.0),
            },
            provenance_updates={"result_json_sha256": "f" * 64},
        )
        report = self._aggregate(rms=rms, q4=q4)
        self.assertFalse(report["candidate_prerequisites"]["q4_mmq_complete"])
        self.assertTrue(
            any(
                "result_sha256 hash mismatch" in violation
                for violation in report["candidate_prerequisites"]["q4_mmq_violations"]
            )
        )

    def test_negative_q4_savings_does_not_meet_proxy_floor(self) -> None:
        rms = self._decision("rms_q8", {})
        q4 = self._decision(
            "q4_mmq",
            {
                "k10240-m2560": (1000.0, 1200.0),
                "k2560-m10240": (1000.0, 1100.0),
            },
        )
        report = self._aggregate(rms=rms, q4=q4)
        projection = report["projection"]
        self.assertLess(projection["logical_matmat_projection"], 0.0)
        self.assertFalse(projection["logical_matmat_proxy_floor_met"])
        self.assertFalse(projection["projected_whole_prefill_floor_met"])


if __name__ == "__main__":
    unittest.main()
