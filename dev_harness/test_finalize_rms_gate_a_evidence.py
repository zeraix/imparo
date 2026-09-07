from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from dev_harness.finalize_rms_gate_a_evidence import RmsEvidenceError, finalize


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RmsEvidenceFinalizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.policy = self.root / "policy.toml"
        self.policy.write_text(
            """schema = 1
phase = "A2"
decision = "gate-a"
production_authority = false
target_sm = 86

[workload]
contract = "rms_norm_q8_1_mmq"
width = 2560
n_tok = 512
native_threads = 1024
triton_warps = [4, 8]
eps = 0.000001

[measurement]
warmup = 20
abba_baab_pairs = 50
min_samples_per_route = 100
launches_per_sample = 16
interleaved = "abba-baab"

[correctness]
max_normalized_abs_vs_native = 0.0002
max_normalized_rel_vs_native = 0.002
max_q8_scale_abs_vs_native = 0.00002
max_q8_value_abs_vs_native = 2
min_q8_byte_agreement = 0.995
max_q8_dequant_abs = 0.05
max_non_finite = 0
max_canary_errors = 0

[noise]
max_native_cv = 0.05
max_candidate_cv = 0.05
max_paired_mad_fraction = 0.03

[gate]
min_kernel_speedup_ratio = 1.10
min_projected_e2e_improvement = 0.03
require_structural_ok = true
require_all_variants_correct = true

[resources]
max_registers_per_thread = 255
max_static_shared_bytes = 101376
max_dynamic_shared_bytes = 101376
max_local_memory_bytes = 0
""",
            encoding="utf-8",
        )
        self.triton_source = self.root / "rms_q8.py"
        self.triton_source.write_text("# fixture\n", encoding="utf-8")
        self.harness_source = self.root / "kernel_lab_rms_q8.cu"
        self.harness_source.write_text("// fixture\n", encoding="utf-8")
        self.harness_exe = self.root / "kernel_lab_rms_q8_sm86.exe"
        self.harness_exe.write_bytes(b"fixture-exe")
        modules_dir = self.root / "pack" / "modules"
        modules_dir.mkdir(parents=True)
        self.modules = [modules_dir / "w4.cubin", modules_dir / "w8.cubin"]
        self.modules[0].write_bytes(b"fixture-w4")
        self.modules[1].write_bytes(b"fixture-w8")
        self.metadata = self.root / "pack" / "lab-metadata.json"
        metadata = {
            "schema": 1,
            "phase": "A2",
            "candidate": "rms_norm_q8_1_mmq",
            "production_enabled": False,
            "target": "cuda:86:32",
            "shape": {"n_tok": 512, "width": 2560},
            "source": {"path": "rms_q8.py", "sha256": sha256(self.triton_source)},
            "builder": {"python": "3.12.3", "triton": "fixture", "image_id": "fixture"},
            "variants": [
                self._metadata_variant("w4-s1", "symbol_w4", 4, 2048, self.modules[0]),
                self._metadata_variant("w8-s1", "symbol_w8", 8, 4096, self.modules[1]),
            ],
        }
        self._write_json(self.metadata, metadata)
        self.result = self.root / "result.json"
        self._write_json(self.result, self._result(launches=16, sample_count=100))
        self.logs: dict[str, Path] = {}
        sanitizer_result = self._result(launches=1, sample_count=2)
        for tool in ("memcheck", "initcheck", "racecheck", "synccheck"):
            log = self.root / f"compute-sanitizer-{tool}.log"
            summary = (
                "========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)"
                if tool == "racecheck"
                else "========= ERROR SUMMARY: 0 errors"
            )
            log.write_text(
                "========= COMPUTE-SANITIZER\n"
                + json.dumps(sanitizer_result, separators=(",", ":"))
                + "\nkernel-lab-phase: complete\n"
                + summary
                + "\n",
                encoding="utf-8",
            )
            self.logs[tool] = log
        self.legacy = self.root / "gate-a-decision.json"
        self._write_legacy(self.legacy, self.result)
        self.witness = self.root / "environment-witness.json"
        self._write_json(
            self.witness,
            {
                "provenance": {
                    "device": {
                        "uuid": "fixture-uuid",
                        "name": "fixture GPU",
                        "sm": 86,
                        "sm_count": 30,
                        "driver_version": 13020,
                        "cuda_runtime_version": 12090,
                    },
                    "local_toolchain": {"target_sm": 86, "backend_abi": 26, "nvcc": "fixture"},
                }
            },
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @staticmethod
    def _write_json(path: Path, value: object) -> None:
        path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")

    @staticmethod
    def _metadata_variant(
        variant_id: str, symbol: str, warps: int, dynamic_shared: int, module: Path
    ) -> dict[str, object]:
        return {
            "variant_id": variant_id,
            "symbol": symbol,
            "num_warps": warps,
            "num_stages": 1,
            "dynamic_shared_bytes": dynamic_shared,
            "module": f"modules/{module.name}",
            "module_bytes": module.stat().st_size,
            "module_sha256": sha256(module),
            "resources": {
                "registers_per_thread": 48,
                "local_memory_bytes": 0,
                "global_scratch_bytes": 0,
                "profile_scratch_bytes": 0,
            },
        }

    @staticmethod
    def _variant(symbol: str, warps: int, dynamic_shared: int, sample_count: int) -> dict[str, object]:
        samples = [10.0] * sample_count
        return {
            "symbol": symbol,
            "warps": warps,
            "dynamic_shared_bytes": dynamic_shared,
            "cubin_bytes": 10,
            "resources": {
                "registers_per_thread": 48,
                "static_shared_bytes": 0,
                "local_bytes": 0,
            },
            "normalized_vs_native": {"max_abs": 1e-6, "max_rel": 1e-6, "non_finite": 0},
            "q8_scales_vs_native": {"max_abs": 0.0},
            "q8_value_max_abs": 1,
            "q8_byte_agreement": 0.999,
            "q8_dequant": {"max_abs": 0.01, "non_finite": 0},
            "canary_errors": 0,
            "timing": {
                "speedup_median": 1.2,
                "native_cv": 0.01,
                "triton_cv": 0.01,
                "paired_mad_fraction": 0.01,
                "native_samples_us": samples,
                "triton_samples_us": samples,
            },
        }

    def _result(self, *, launches: int, sample_count: int) -> dict[str, object]:
        variants = [
            self._variant("symbol_w4", 4, 2048, sample_count),
            self._variant("symbol_w8", 8, 4096, sample_count),
        ]
        variants[0]["cubin_bytes"] = self.modules[0].stat().st_size
        variants[1]["cubin_bytes"] = self.modules[1].stat().st_size
        return {
            "schema": 1,
            "phase": "A2",
            "production_enabled": False,
            "same_primary_context": True,
            "same_stream": True,
            "separate_outputs": True,
            "timing_clock": "cuda-events",
            "interleaved_schedule": "abba-baab",
            "launches_per_sample": launches,
            "target_sm": 86,
            "width": 2560,
            "n_tok": 512,
            "all_canary_errors": 0,
            "structural_ok": True,
            "variants": variants,
        }

    def _write_legacy(self, path: Path, result: Path) -> None:
        self._write_json(
            path,
            {
                "schema": 1,
                "decision": "gate-a-candidate",
                "production_authority": False,
                "final_gate_a_decision": False,
                "candidate_admissible": True,
                "kernel_floor_met": True,
                "violations": [],
                "result": str(result),
                "sanitizer": {
                    tool: {"path": str(log), "exit_code": 0, "pass": True}
                    for tool, log in self.logs.items()
                },
            },
        )

    def _finalize(self, **updates: Path) -> dict[str, object]:
        paths = {
            "policy_path": self.policy,
            "metadata_path": self.metadata,
            "triton_source_path": self.triton_source,
            "harness_source_path": self.harness_source,
            "harness_exe_path": self.harness_exe,
            "result_path": self.result,
            "legacy_decision_path": self.legacy,
            "environment_witness_path": self.witness,
        }
        paths.update(updates)
        return finalize(**paths)

    def test_finalizes_complete_evidence_without_mutating_raw_inputs(self) -> None:
        raw_paths = [
            self.policy,
            self.metadata,
            self.triton_source,
            self.harness_source,
            self.harness_exe,
            self.result,
            self.legacy,
            self.witness,
            *self.modules,
            *self.logs.values(),
        ]
        before = {path: sha256(path) for path in raw_paths}
        decision = self._finalize()
        after = {path: sha256(path) for path in raw_paths}
        self.assertEqual(before, after)
        self.assertTrue(decision["correctness_admissible"])
        self.assertTrue(decision["measurement_admissible"])
        self.assertTrue(decision["sanitizer_admissible"])
        self.assertTrue(decision["provenance_admissible"])
        self.assertTrue(decision["candidate_admissible"])
        self.assertFalse(decision["final_gate_a_decision"])
        self.assertIsNone(decision["projected_e2e_improvement"])
        self.assertEqual(set(decision["sanitizer"]), set(self.logs))

    def test_rejects_missing_sanitizer_success_summary(self) -> None:
        log = self.logs["memcheck"]
        log.write_text(
            log.read_text(encoding="utf-8").replace(
                "========= ERROR SUMMARY: 0 errors", "========= ERROR SUMMARY: 1 error"
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(RmsEvidenceError, "zero-error summary"):
            self._finalize()

    def test_rejects_cubin_hash_mismatch(self) -> None:
        metadata = json.loads(self.metadata.read_text(encoding="utf-8"))
        metadata["variants"][0]["module_sha256"] = "0" * 64
        self._write_json(self.metadata, metadata)
        with self.assertRaisesRegex(RmsEvidenceError, "module hash mismatch"):
            self._finalize()

    def test_rejects_incomplete_environment_witness(self) -> None:
        witness = json.loads(self.witness.read_text(encoding="utf-8"))
        del witness["provenance"]["device"]["driver_version"]
        self._write_json(self.witness, witness)
        with self.assertRaisesRegex(RmsEvidenceError, "device is incomplete"):
            self._finalize()

    def test_measurement_gate_failure_cannot_become_admissible(self) -> None:
        result = json.loads(self.result.read_text(encoding="utf-8"))
        result["launches_per_sample"] = 1
        alternate_result = self.root / "result-bad-measurement.json"
        self._write_json(alternate_result, result)
        alternate_legacy = self.root / "legacy-bad-measurement.json"
        self._write_legacy(alternate_legacy, alternate_result)
        decision = self._finalize(
            result_path=alternate_result, legacy_decision_path=alternate_legacy
        )
        self.assertTrue(decision["correctness_admissible"])
        self.assertFalse(decision["measurement_admissible"])
        self.assertFalse(decision["candidate_admissible"])
        self.assertIn("formal: launches per sample mismatch", decision["violations"])


if __name__ == "__main__":
    unittest.main()
