from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from dev_harness import run_q4_variant_sanitizer_matrix as runner
from dev_harness.audit_q4_gate_a_evidence import Q4StrictAuditError, audit
from dev_harness.finalize_q4_gate_a_evidence import finalize
from dev_harness.finalize_q4_sanitizer_evidence import sha256_file
from dev_harness.validate_q4_sanitizer_schema2 import (
    CASE_KEYS,
    Q4SanitizerSchema2Error,
    _evidence_path,
    _exact_keys,
    _snapshot,
    _strict_json_text,
    _tool_identity,
    validate_schema2_matrix,
)


ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "config/kernel-lab-policy.toml"
FORMAL = ROOT / "artifacts/kernel-lab/q4-variants-perf/variants-b/formal-matrix-evidence.json"
LEGACY = ROOT / "artifacts/kernel-lab/q4-variants-sanitizer/variants-b/sanitizer-evidence.json"
METADATA = ROOT / "program-packs/lab/q4-q8-mmq-sm86/variants-b/lab-metadata.json"
SOURCE = ROOT / "crates/imparo-cuda/native/tests/kernel_lab_q4_q8.cu"
BASE_WITNESS = ROOT / "artifacts/kernel-lab/q4-variants-perf/variants-b/environment-witness.json"
REQUIRED = (POLICY, FORMAL, LEGACY, METADATA, SOURCE, BASE_WITNESS)


class Q4SanitizerSchema2PortableStrictTests(unittest.TestCase):
    def test_duplicate_and_nonfinite_json_are_rejected(self) -> None:
        with self.assertRaisesRegex(Q4SanitizerSchema2Error, "duplicate JSON key"):
            _strict_json_text('{"schema":2,"schema":2}', "fixture")
        for value in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(value=value), self.assertRaisesRegex(
                Q4SanitizerSchema2Error, "non-finite JSON constant"
            ):
                _strict_json_text('{"value":' + value + '}', "fixture")

    def test_extra_harness_case_key_is_rejected(self) -> None:
        value = {key: None for key in CASE_KEYS}
        value["wishful"] = True
        with self.assertRaisesRegex(Q4SanitizerSchema2Error, "keys differ"):
            _exact_keys(value, CASE_KEYS, "raw harness case[0]")

    def test_bound_tool_version_is_locally_replayed_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tool = Path(directory) / "compute-sanitizer.bin"
            tool.write_bytes(b"bound-tool")
            identity = {
                "path": str(tool.resolve()), "sha256": sha256_file(tool),
                "version_argv": [str(tool.resolve()), "--version"],
                "version_exit_code": 0,
                "version_stdout": "Compute Sanitizer version 999\n",
                "version_stderr": "", "version": "Compute Sanitizer version 999",
            }
            replay = SimpleNamespace(
                returncode=0, stdout="Compute Sanitizer real version\n", stderr=""
            )
            with mock.patch(
                "dev_harness.validate_q4_sanitizer_schema2.subprocess.run",
                return_value=replay,
            ), self.assertRaisesRegex(Q4SanitizerSchema2Error, "local version replay"):
                _tool_identity(identity, tool, "compute sanitizer")
            replay.stdout = identity["version_stdout"]
            with mock.patch(
                "dev_harness.validate_q4_sanitizer_schema2.subprocess.run",
                return_value=replay,
            ):
                self.assertEqual(
                    _tool_identity(identity, tool, "compute sanitizer")["sha256"],
                    sha256_file(tool),
                )

    def test_arbitrary_gpu_csv_is_rejected(self) -> None:
        value = {
            "captured_utc": "2026-08-28T00:00:00+00:00",
            "gpu_csv": "arbitrary",
            "compute_apps_csv": [],
        }
        with self.assertRaisesRegex(Q4SanitizerSchema2Error, "GPU CSV identity"):
            _snapshot(value, "fixture")

    def test_evidence_path_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            inside = root / "inside"
            inside.mkdir()
            outside = root.parent / "outside-schema2-fixture"
            outside.write_bytes(b"fixture")
            try:
                with self.assertRaisesRegex(Q4SanitizerSchema2Error, "escapes evidence root"):
                    _evidence_path(str(outside.resolve()), root, outside, "fixture")
            finally:
                outside.unlink(missing_ok=True)

    def test_final_filename_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "partial-manifest.json"
            manifest.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(Q4SanitizerSchema2Error, "final filename"):
                validate_schema2_matrix(
                    manifest, manifest,
                    expected_metadata_sha256="0" * 64,
                    expected_harness_source_sha256="0" * 64,
                )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _tool(path: Path) -> dict[str, object]:
    completed = subprocess.run(
        [str(path.resolve()), "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise unittest.SkipTest(f"tool version replay failed: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "version_argv": [str(path.resolve()), "--version"],
        "version_exit_code": completed.returncode,
        "version_stdout": completed.stdout,
        "version_stderr": completed.stderr,
        "version": (completed.stdout + completed.stderr).strip(),
    }


def _cuobjdump_text(symbol: str, threads: int, registers: int) -> str:
    parameters = []
    for ordinal, (offset, size) in enumerate(
        zip(runner.EXPECTED_KPARAM_OFFSETS, runner.EXPECTED_KPARAM_SIZES)
    ):
        parameters.append(
            "\tAttribute:\tEIATTR_KPARAM_INFO\n"
            f"\tValue: Ordinal : 0x{ordinal:x} Offset : 0x{offset:x} Size : 0x{size:x}\n"
        )
    return (
        "64bit elf: type=EXEC, abi=7, sm=86, toolkit=129, flags = 0x560534\n"
        ".nv.info\n"
        "\tAttribute: EIATTR_REGCOUNT\n"
        f"\tValue: function: {symbol}(0xa) register count: {registers}\n"
        "\tAttribute: EIATTR_FRAME_SIZE\n"
        f"\tValue: function: {symbol}(0xa) frame size: 0x0\n\n"
        f".nv.info.{symbol}\n"
        "\tAttribute: EIATTR_CBANK_PARAM_SIZE\n\tValue: 0x48\n"
        + "".join(parameters)
        + f"\tAttribute: EIATTR_REQNTID\n\tValue: 0x{threads:x} 0x1 0x1\n"
        "\n.nv.callgraph\n<0,-1>\n\n"
        f".text.{symbol}\nbar = 0 reg = {registers} lmem = 0 smem = 0\n"
    )


def build_schema2_fixture(root: Path) -> tuple[Path, Path]:
    legacy = json.loads(LEGACY.read_text(encoding="utf-8"))
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    base_witness = json.loads(BASE_WITNESS.read_text(encoding="utf-8"))
    pairs = runner.build_pairs(metadata, METADATA)
    expected_ids = runner.exact_run_ids(pairs)
    legacy_runs = {
        (item["label"], item["tool"]): item for item in legacy["runs"]
    }
    executable = Path(legacy["harness_exe"]).resolve()
    compute_path = Path(legacy["compute_sanitizer"]).resolve()
    cuobjdump_path = Path(base_witness["local_toolchain"]["cuobjdump_path"]).resolve()
    discovered = runner.discover_tool_paths()
    nvidia_smi_path = discovered["nvidia_smi"]
    if nvidia_smi_path is None:
        raise unittest.SkipTest("nvidia-smi is unavailable for schema2 fixture")
    compute = _tool(compute_path)
    cuobjdump = _tool(cuobjdump_path)
    nvidia_smi = _tool(nvidia_smi_path)
    environment = {name: None for name in runner.RELEVANT_ENVIRONMENT}
    environment["IMPARO_CUDA_KERNEL_LAB"] = "1"
    environment["PATH"] = "schema2-fixture-path"
    environment["PATH_sha256"] = hashlib.sha256(
        environment["PATH"].encode("utf-8")
    ).hexdigest()
    device = json.loads(
        Path(legacy["runs"][0]["harness_output"]).read_text(encoding="utf-8")
    )["device"]
    snapshot = {
        "captured_utc": "2026-08-28T00:00:00+00:00",
        "gpu_csv": f"2026/08/28, {device['uuid']}, {device['name']}, 580.97, 40, 1, 1, 1, P8, 1, 0, 0, 1, 1",
        "compute_apps_csv": [],
    }
    identity = {
        "policy": str(POLICY.resolve()),
        "policy_sha256": sha256_file(POLICY),
        "metadata": str(METADATA.resolve()),
        "metadata_sha256": sha256_file(METADATA),
        "harness_source": str(SOURCE.resolve()),
        "harness_source_sha256": sha256_file(SOURCE),
        "harness_executable": str(executable),
        "harness_executable_sha256": sha256_file(executable),
        "runner": str(Path(runner.__file__).resolve()),
        "runner_sha256": sha256_file(Path(runner.__file__).resolve()),
        "compute_sanitizer": compute,
        "cuobjdump": cuobjdump,
        "nvidia_smi": nvidia_smi,
        "relevant_environment": environment,
    }
    modules = []
    for pair in pairs:
        for variant in (pair["expansion"], pair["contraction"]):
            raw = root / "preflight" / f"{pair['label']}--{variant['shape']}.cuobjdump.txt"
            raw.parent.mkdir(parents=True, exist_ok=True)
            cuobjdump_run = subprocess.run(
                [str(cuobjdump_path), "-elf", variant["module"]],
                check=False,
                capture_output=True,
                text=True,
            )
            if cuobjdump_run.returncode != 0:
                raise unittest.SkipTest("cuobjdump replay failed for schema2 fixture")
            raw.write_text(
                cuobjdump_run.stdout + cuobjdump_run.stderr,
                encoding="utf-8",
            )
            modules.append({
                **variant,
                "cuobjdump_argv": [cuobjdump["path"], "-elf", variant["module"]],
                "cuobjdump_raw": str(raw.resolve()),
                "cuobjdump_raw_sha256": sha256_file(raw),
                "module_stat_bytes": Path(variant["module"]).stat().st_size,
                "cuobjdump_identity_sha256": cuobjdump["sha256"],
                "inspection": runner.parse_cuobjdump(
                    cuobjdump_run.stdout, symbol=variant["symbol"],
                    threads=variant["threads"],
                    registers=variant["registers_per_thread"],
                ),
            })
    runs = []
    pair_map = {pair["label"]: pair for pair in pairs}
    for index, run_id in enumerate(expected_ids, start=1):
        label, tool = run_id.rsplit("--", 1)
        pair = pair_map[label]
        legacy_run = legacy_runs[(label, tool)]
        run_dir = root / "runs" / f"{index:02d}--{run_id}"
        run_dir.mkdir(parents=True)
        raw_stdout = run_dir / "process-stdout.raw"
        raw_stderr = run_dir / "process-stderr.raw"
        stdout = run_dir / "stdout.json"
        stderr = run_dir / "stderr.log"
        exit_file = run_dir / "exit-code.txt"
        harness_text = Path(legacy_run["harness_output"]).read_text(encoding="utf-8")
        diagnostics = Path(legacy_run["log"]).read_text(encoding="utf-8")
        raw_stdout.write_text(harness_text, encoding="utf-8")
        raw_stderr.write_text(diagnostics, encoding="utf-8")
        normalized_stdout, normalized_stderr, normalization = runner.split_process_streams(
            harness_text, diagnostics
        )
        normalization["passed"] = True
        stdout.write_text(normalized_stdout, encoding="utf-8")
        stderr.write_text(normalized_stderr, encoding="utf-8")
        exit_file.write_text("0\n", encoding="utf-8")
        actual_argv = [
            compute["path"], "--tool", tool, "--error-exitcode", "86",
            *(("--leak-check", "full") if tool == "memcheck" else ()),
            str(executable), *runner.harness_args(pair),
        ]
        started = f"2026-08-28T00:{index:02d}:00+00:00"
        ended = f"2026-08-28T00:{index:02d}:01+00:00"
        command = {
            "schema": 1, "run_id": run_id, "actual_argv": actual_argv,
            "runner": identity["runner"], "runner_sha256": identity["runner_sha256"],
            "compute_sanitizer": compute, "relevant_environment": environment,
            "capture_normalization": normalization,
            "process_stdout_raw": str(raw_stdout.resolve()),
            "process_stdout_raw_sha256": sha256_file(raw_stdout),
            "process_stderr_raw": str(raw_stderr.resolve()),
            "process_stderr_raw_sha256": sha256_file(raw_stderr),
            "started_utc": started, "ended_utc": ended, "exit_code": 0,
        }
        command_path = run_dir / "command.json"
        _write_json(command_path, command)
        summary = runner.sanitizer_summary(tool, normalized_stderr)
        harness = runner.validate_harness_stdout(normalized_stdout, pair)
        runs.append({
            "run_id": run_id, "label": label, "tool": tool,
            "directory": str(run_dir.resolve()), "command": str(command_path.resolve()),
            "command_sha256": sha256_file(command_path), "actual_argv": actual_argv,
            "started_utc": started, "ended_utc": ended, "exit_code": 0,
            "stdout": str(stdout.resolve()), "stdout_sha256": sha256_file(stdout),
            "stderr": str(stderr.resolve()), "stderr_sha256": sha256_file(stderr),
            "process_stdout_raw": str(raw_stdout.resolve()),
            "process_stdout_raw_sha256": sha256_file(raw_stdout),
            "process_stderr_raw": str(raw_stderr.resolve()),
            "process_stderr_raw_sha256": sha256_file(raw_stderr),
            "capture_normalization": normalization,
            "exit_code_file": str(exit_file.resolve()),
            "exit_code_file_sha256": sha256_file(exit_file),
            "device_before": copy.deepcopy(snapshot),
            "device_after": copy.deepcopy(snapshot),
            "summary": summary, "harness": harness, "pass": True,
        })
    manifest = {
        "schema": 2, "phase": "A2",
        "milestone": "q4-variant-sanitizer-matrix-v2",
        "production_enabled": False, "all_pass": True, "complete": True,
        "expected_run_ids": expected_ids,
        "execution_order": {
            "labels": list(runner.LABELS), "tools": list(runner.TOOLS),
        },
        "identity": identity,
        "device_snapshots": {"before": snapshot, "after": snapshot},
        "modules": modules, "runs": runs,
    }
    manifest_path = root / "sanitizer-evidence-v2.json"
    _write_json(manifest_path, manifest)
    witness = copy.deepcopy(base_witness)
    witness["sanitizer_manifest_sha256"] = sha256_file(manifest_path)
    witness["harness_exe_sha256"] = identity["harness_executable_sha256"]
    witness["local_toolchain"]["compute_sanitizer_sha256"] = compute["sha256"]
    witness["local_toolchain"]["compute_sanitizer_version"] = compute["version"]
    witness["sanitizer_authority"] = {
        "schema": 2, "runner_sha256": identity["runner_sha256"],
        "compute_sanitizer_sha256": compute["sha256"],
        "compute_sanitizer_version": compute["version"],
        "cuobjdump_sha256": cuobjdump["sha256"],
        "cuobjdump_version": cuobjdump["version"],
        "nvidia_smi_sha256": nvidia_smi["sha256"],
        "nvidia_smi_version": nvidia_smi["version"],
    }
    witness_path = root / "environment-witness.json"
    _write_json(witness_path, witness)
    return manifest_path, witness_path


def rewrite_first_harness(manifest: Path, transform) -> None:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    run = data["runs"][0]
    raw = Path(run["process_stdout_raw"])
    stdout = Path(run["stdout"])
    updated = transform(raw.read_text(encoding="utf-8"))
    raw.write_text(updated, encoding="utf-8")
    stdout.write_text(updated, encoding="utf-8")
    run["process_stdout_raw_sha256"] = sha256_file(raw)
    run["stdout_sha256"] = sha256_file(stdout)
    command_path = Path(run["command"])
    command = json.loads(command_path.read_text(encoding="utf-8"))
    command["process_stdout_raw_sha256"] = run["process_stdout_raw_sha256"]
    _write_json(command_path, command)
    run["command_sha256"] = sha256_file(command_path)
    _write_json(manifest, data)


@unittest.skipUnless(all(path.is_file() for path in REQUIRED), "Q4 raw fixture unavailable")
class Q4SanitizerSchema2Tests(unittest.TestCase):
    def _validate(self, path: Path):
        return validate_schema2_matrix(
            path, POLICY,
            expected_metadata_sha256=sha256_file(METADATA),
            expected_harness_source_sha256=sha256_file(SOURCE),
        )

    def test_schema2_happy_fixture_is_authoritative_but_floor_still_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, witness = build_schema2_fixture(Path(directory))
            binding = self._validate(manifest)
            self.assertEqual(binding["run_count"], 16)
            self.assertEqual(binding["preflight_count"], 8)
            decision = finalize(
                formal_matrix_path=FORMAL, sanitizer_manifest_path=manifest,
                policy_path=POLICY, harness_source_path=SOURCE,
                environment_witness_path=witness,
                expected_policy_sha256=sha256_file(POLICY),
                expected_formal_matrix_sha256=sha256_file(FORMAL),
                expected_sanitizer_manifest_sha256=sha256_file(manifest),
                expected_metadata_sha256=sha256_file(METADATA),
                expected_harness_source_sha256=sha256_file(SOURCE),
                expected_environment_witness_sha256=sha256_file(witness),
                expected_min_kernel_speedup="1.10",
                expected_min_projected_e2e="0.03",
            )
            self.assertTrue(decision["sanitizer_admissible"])
            self.assertTrue(decision["provenance_admissible"])
            self.assertFalse(decision["kernel_floor_met"])
            self.assertFalse(decision["candidate_admissible"])

    def test_incomplete_or_missing_cartesian_run_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, _ = build_schema2_fixture(Path(directory))
            data = json.loads(manifest.read_text())
            data["runs"].pop()
            _write_json(manifest, data)
            with self.assertRaisesRegex(Q4SanitizerSchema2Error, "exactly sixteen"):
                self._validate(manifest)

    def test_runner_hash_tamper_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, _ = build_schema2_fixture(Path(directory))
            data = json.loads(manifest.read_text())
            data["identity"]["runner_sha256"] = "0" * 64
            _write_json(manifest, data)
            with self.assertRaisesRegex(Q4SanitizerSchema2Error, "runner SHA-256"):
                self._validate(manifest)

    def test_tool_identity_tamper_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, _ = build_schema2_fixture(Path(directory))
            data = json.loads(manifest.read_text())
            data["identity"]["compute_sanitizer"]["version_exit_code"] = 1
            _write_json(manifest, data)
            with self.assertRaisesRegex(Q4SanitizerSchema2Error, "version identity"):
                self._validate(manifest)

    def test_fabricated_tool_version_999_is_rejected_by_local_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, _ = build_schema2_fixture(Path(directory))
            data = json.loads(manifest.read_text())
            identity = data["identity"]["compute_sanitizer"]
            identity["version_stdout"] = "NVIDIA Compute Sanitizer version 999\n"
            identity["version_stderr"] = ""
            identity["version"] = identity["version_stdout"].strip()
            _write_json(manifest, data)
            with self.assertRaisesRegex(Q4SanitizerSchema2Error, "local version replay"):
                self._validate(manifest)

    def test_duplicate_manifest_key_is_rejected_before_self_consistency(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, _ = build_schema2_fixture(Path(directory))
            text = manifest.read_text(encoding="utf-8")
            manifest.write_text(
                text.replace('  "schema": 2,', '  "schema": 2,\n  "schema": 2,', 1),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(Q4SanitizerSchema2Error, "duplicate JSON key"):
                self._validate(manifest)

    def test_raw_harness_nan_is_rejected_even_when_all_hashes_are_updated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, _ = build_schema2_fixture(Path(directory))
            def mutate(text: str) -> str:
                document = json.loads(text)
                document["debug_exact_triton_native_pass"] = float("nan")
                return json.dumps(document, separators=(",", ":")) + "\n"
            rewrite_first_harness(manifest, mutate)
            with self.assertRaisesRegex(Q4SanitizerSchema2Error, "non-finite JSON constant"):
                self._validate(manifest)

    def test_raw_harness_extra_case_key_is_rejected_with_rehashed_capture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, _ = build_schema2_fixture(Path(directory))
            def mutate(text: str) -> str:
                document = json.loads(text)
                document["cases"][0]["wishful"] = True
                return json.dumps(document, separators=(",", ":")) + "\n"
            rewrite_first_harness(manifest, mutate)
            with self.assertRaisesRegex(Q4SanitizerSchema2Error, "case\[0\] keys differ"):
                self._validate(manifest)

    def test_synthetic_cuobjdump_self_consistency_is_rejected_by_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, _ = build_schema2_fixture(Path(directory))
            data = json.loads(manifest.read_text())
            module = data["modules"][0]
            raw = Path(module["cuobjdump_raw"])
            synthetic = _cuobjdump_text(
                module["symbol"], module["threads"], module["registers_per_thread"]
            )
            raw.write_text(synthetic, encoding="utf-8")
            module["cuobjdump_raw_sha256"] = sha256_file(raw)
            module["inspection"] = runner.parse_cuobjdump(
                synthetic, symbol=module["symbol"], threads=module["threads"],
                registers=module["registers_per_thread"],
            )
            _write_json(manifest, data)
            with self.assertRaisesRegex(Q4SanitizerSchema2Error, "local cuobjdump replay"):
                self._validate(manifest)

    def test_arbitrary_run_snapshot_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, _ = build_schema2_fixture(Path(directory))
            data = json.loads(manifest.read_text())
            fields = data["runs"][0]["device_before"]["gpu_csv"].split(", ")
            fields[1] = "GPU-00000000-0000-0000-0000-000000000000"
            data["runs"][0]["device_before"]["gpu_csv"] = ", ".join(fields)
            _write_json(manifest, data)
            with self.assertRaisesRegex(Q4SanitizerSchema2Error, "GPU snapshot identity"):
                self._validate(manifest)

    def test_module_raw_preflight_tamper_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, _ = build_schema2_fixture(Path(directory))
            data = json.loads(manifest.read_text())
            data["modules"][0]["inspection"]["registers_per_thread"] += 1
            _write_json(manifest, data)
            with self.assertRaisesRegex(Q4SanitizerSchema2Error, "inspection differs"):
                self._validate(manifest)

    def test_actual_argv_tamper_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, _ = build_schema2_fixture(Path(directory))
            data = json.loads(manifest.read_text())
            data["runs"][0]["actual_argv"][4] = "0"
            _write_json(manifest, data)
            with self.assertRaisesRegex(Q4SanitizerSchema2Error, "argv/exit/pass"):
                self._validate(manifest)

    def test_raw_capture_self_rehash_cannot_hide_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, _ = build_schema2_fixture(Path(directory))
            data = json.loads(manifest.read_text())
            raw = Path(data["runs"][0]["process_stderr_raw"])
            raw.write_text(raw.read_text() + "unexpected\n", encoding="utf-8")
            digest = sha256_file(raw)
            data["runs"][0]["process_stderr_raw_sha256"] = digest
            command = Path(data["runs"][0]["command"])
            command_data = json.loads(command.read_text())
            command_data["process_stderr_raw_sha256"] = digest
            _write_json(command, command_data)
            data["runs"][0]["command_sha256"] = sha256_file(command)
            _write_json(manifest, data)
            with self.assertRaisesRegex(Q4SanitizerSchema2Error, "normalized capture differs"):
                self._validate(manifest)

    def test_command_hash_tamper_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, _ = build_schema2_fixture(Path(directory))
            data = json.loads(manifest.read_text())
            data["runs"][0]["command_sha256"] = "0" * 64
            _write_json(manifest, data)
            with self.assertRaisesRegex(Q4SanitizerSchema2Error, "command SHA-256"):
                self._validate(manifest)

    def test_witness_schema2_authority_tamper_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, witness = build_schema2_fixture(Path(directory))
            data = json.loads(witness.read_text())
            data["sanitizer_authority"]["runner_sha256"] = "0" * 64
            _write_json(witness, data)
            with self.assertRaisesRegex(Q4StrictAuditError, "schema2 sanitizer authority"):
                audit(
                    policy_path=POLICY,
                    expected_policy_sha256=sha256_file(POLICY),
                    formal_matrix_path=FORMAL,
                    expected_formal_matrix_sha256=sha256_file(FORMAL),
                    sanitizer_manifest_path=manifest,
                    expected_sanitizer_manifest_sha256=sha256_file(manifest),
                    metadata_path=METADATA,
                    expected_metadata_sha256=sha256_file(METADATA),
                    harness_source_path=SOURCE,
                    expected_harness_source_sha256=sha256_file(SOURCE),
                    environment_witness_path=witness,
                    expected_environment_witness_sha256=sha256_file(witness),
                    expected_min_kernel_speedup="1.10",
                    expected_min_projected_e2e="0.03",
                )


if __name__ == "__main__":
    unittest.main()
