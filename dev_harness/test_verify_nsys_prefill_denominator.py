from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from dev_harness.verify_nsys_prefill_denominator import (
    DenominatorError,
    canonical_token_hash,
    sha256_file,
    verify,
)


class NsysPrefillDenominatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.executable = self.root / "imparo-forward.exe"
        self.dll = self.root / "imparo-cuda-sm86.dll"
        self.model = self.root / "model.gguf"
        self.policy = self.root / "kernel-lab-policy.toml"
        self.executable.write_bytes(b"exact executable")
        self.dll.write_bytes(b"exact dll")
        self.model.write_bytes(b"exact model")
        self.policy.write_text(
            "[gate]\nmin_projected_e2e_improvement = 0.03\n",
            encoding="utf-8",
        )
        self.sqlite = self.root / "capture.sqlite"
        self._write_sqlite(
            [(0, 100, 0, 7, 13, 99), (110, 210, 0, 7, 13, 99),
             (220, 320, 0, 7, 13, 99)]
        )
        self.wall = self.root / "prefill-wall.stdout"
        self._write_wall()
        self.evidence = self.root / "profile-evidence.json"
        self._write_evidence()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_sqlite(self, kernels: list[tuple[int, int, int, int, int, int]]) -> None:
        if self.sqlite.exists():
            self.sqlite.unlink()
        connection = sqlite3.connect(self.sqlite)
        connection.execute(
            "CREATE TABLE META_DATA_CAPTURE (name TEXT NOT NULL, value TEXT)"
        )
        connection.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL ("
            "start INTEGER NOT NULL, end INTEGER NOT NULL, "
            "deviceId INTEGER NOT NULL, contextId INTEGER NOT NULL, "
            "streamId INTEGER NOT NULL, globalPid INTEGER)"
        )
        metadata = [
            ("PROCESS_0:COMMAND", r".\target\release\imparo-forward.exe"),
            ("PROCESS_0:ARGUMENT_0", r".\models\model.gguf"),
            ("PROCESS_0:ARGUMENT_1", "2"),
            ("PROCESS_0:ARGUMENT_2", "1001"),
            ("PROCESS_0:ENVIRONMENT_VARIABLE", "A=1"),
            ("PROCESS_0:ENVIRONMENT_VARIABLE", "B=2"),
        ]
        connection.executemany(
            "INSERT INTO META_DATA_CAPTURE VALUES (?, ?)", metadata
        )
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?)",
            kernels,
        )
        connection.commit()
        connection.close()

    def _write_wall(self, **updates: object) -> None:
        record: dict[str, object] = {
            "schema": 1,
            "phase": "A1",
            "event": "prefill_wall",
            "lab_only": True,
            "production_authority": False,
            "rep": 0,
            "token_count": 2,
            "start_pos": 0,
            "split_state": {"mode": "single", "split_at": None, "parts": None},
            "prefill_wall_ms": 0.00032,
        }
        record.update(updates)
        self.wall.write_text(
            "load diagnostic\n"
            + json.dumps(record, sort_keys=True)
            + "\ntop10 diagnostic\n",
            encoding="utf-8",
        )

    def _write_evidence(self, **updates: object) -> None:
        record: dict[str, object] = {
            "schema": 1,
            "phase": "A1",
            "scope": "one-prefill-gpu-timeline-denominator",
            "head_commit": "1" * 40,
            "backend_abi": 26,
            "target_sm": 86,
            "inputs": {
                "sqlite_sha256": sha256_file(self.sqlite),
                "executable_sha256": sha256_file(self.executable),
                "dll_sha256": sha256_file(self.dll),
                "model_sha256": sha256_file(self.model),
                "prefill_wall_sha256": sha256_file(self.wall),
                "capture_command": r".\target\release\imparo-forward.exe",
                "capture_model_argument": r".\models\model.gguf",
                "token_count": 2,
                "token_ids_sha256": canonical_token_hash([2, 1001]),
                "repeat_count": 1,
                "prefill_token_count": 2,
                "prefill_start_pos": 0,
                "prefill_split_state": {
                    "mode": "single", "split_at": None, "parts": None
                },
            },
            "logical_matmat": {
                "scope": "complete-native-imparo_cuda_matmat-cuda-events",
                "collection_relationship": "same-run",
                "calls": 3,
                "total_ns": 160,
            },
        }
        for key, value in updates.items():
            if key == "inputs":
                record["inputs"].update(value)  # type: ignore[union-attr]
            elif key == "logical_matmat":
                record["logical_matmat"].update(value)  # type: ignore[union-attr]
            else:
                record[key] = value
        self.evidence.write_text(
            json.dumps(record, sort_keys=True) + "\n", encoding="utf-8"
        )

    def _verify(self, **overrides: object) -> dict[str, object]:
        arguments: dict[str, object] = {
            "sqlite_path": self.sqlite,
            "sqlite_sha256": sha256_file(self.sqlite),
            "executable_path": self.executable,
            "executable_sha256": sha256_file(self.executable),
            "dll_path": self.dll,
            "dll_sha256": sha256_file(self.dll),
            "model_path": self.model,
            "model_sha256": sha256_file(self.model),
            "profile_evidence_path": self.evidence,
            "profile_evidence_sha256": sha256_file(self.evidence),
            "prefill_wall_path": self.wall,
            "prefill_wall_sha256": sha256_file(self.wall),
            "policy_path": self.policy,
            "policy_sha256": sha256_file(self.policy),
        }
        arguments.update(overrides)
        return verify(**arguments)  # type: ignore[arg-type]

    def test_distinguishes_gpu_timeline_prefill_wall_and_full_request_wall(self) -> None:
        report = self._verify()
        timeline = report["kernel_timeline"]
        self.assertEqual(timeline["total_cuda_kernel_work_ns"], 300)
        self.assertEqual(timeline["serialized_kernel_envelope_ns"], 320)
        self.assertEqual(timeline["serialized_idle_gap_ns"], 20)
        self.assertAlmostEqual(
            report["logical_matmat"]["fraction_of_whole_prefill_gpu_timeline"],
            0.5,
        )
        self.assertAlmostEqual(
            report["logical_matmat"]["fraction_of_whole_prefill_host_wall"],
            0.5,
        )
        self.assertEqual(
            report["logical_matmat"]["fraction_kind"], "same-run-measured"
        )
        boundary = report["projection_boundary"]
        self.assertTrue(boundary["whole_prefill_host_wall_denominator_available"])
        self.assertFalse(
            boundary["full_request_host_wall_end_to_end_denominator_available"]
        )
        self.assertIsNone(boundary["projected_whole_prefill_host_wall_improvement"])
        self.assertFalse(boundary["projected_whole_prefill_floor_evaluable"])
        self.assertIsNone(boundary["projected_full_request_host_wall_improvement"])
        self.assertEqual(boundary["min_projected_e2e_improvement"], 0.03)
        self.assertFalse(boundary["three_percent_policy_weakened"])

    def test_rejects_multiple_streams(self) -> None:
        self._write_sqlite([(0, 100, 0, 7, 13, 99), (110, 210, 0, 7, 14, 99)])
        self._write_evidence()
        with self.assertRaisesRegex(DenominatorError, "multiple streams"):
            self._verify()

    def test_rejects_overlapping_kernels_even_on_one_stream(self) -> None:
        self._write_sqlite([(0, 100, 0, 7, 13, 99), (90, 210, 0, 7, 13, 99)])
        self._write_evidence()
        with self.assertRaisesRegex(DenominatorError, "overlap"):
            self._verify()

    def test_rejects_command_or_token_mismatch(self) -> None:
        for update in (
            {"capture_command": "wrong.exe"},
            {"token_ids_sha256": "0" * 64},
        ):
            with self.subTest(update=update):
                self._write_evidence(inputs=update)
                with self.assertRaisesRegex(
                    DenominatorError, "command/model/token identity mismatch"
                ):
                    self._verify()

    def test_rejects_artifact_hash_or_evidence_binding_mismatch(self) -> None:
        with self.assertRaisesRegex(DenominatorError, "executable_sha256 mismatch"):
            self._verify(executable_sha256="0" * 64)
        self._write_evidence(inputs={"dll_sha256": "0" * 64})
        with self.assertRaisesRegex(DenominatorError, "profile evidence dll_sha256"):
            self._verify()

    def test_rejects_wall_hash_or_record_identity_mismatch(self) -> None:
        with self.assertRaisesRegex(DenominatorError, "prefill_wall_sha256 mismatch"):
            self._verify(prefill_wall_sha256="0" * 64)
        self._write_wall(start_pos=1)
        self._write_evidence()
        with self.assertRaisesRegex(DenominatorError, "wall record identity"):
            self._verify()

    def test_rejects_logical_time_larger_than_gpu_envelope(self) -> None:
        self._write_evidence(logical_matmat={"total_ns": 321})
        with self.assertRaisesRegex(DenominatorError, "exceeds GPU kernel envelope"):
            self._verify()

    def test_labels_identity_matched_separate_run_as_estimate(self) -> None:
        self._write_evidence(
            logical_matmat={
                "collection_relationship": "identity-matched-separate-run"
            }
        )
        report = self._verify()
        self.assertEqual(
            report["logical_matmat"]["fraction_kind"],
            "identity-matched-cross-run-estimate",
        )

    def test_rejects_policy_floor_below_three_percent(self) -> None:
        self.policy.write_text(
            "[gate]\nmin_projected_e2e_improvement = 0.029\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(DenominatorError, "remain at least 0.03"):
            self._verify()


if __name__ == "__main__":
    unittest.main()
