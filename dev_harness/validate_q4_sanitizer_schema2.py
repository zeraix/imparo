#!/usr/bin/env python3
"""Validate immutable Phase A2 Q4 sanitizer schema-2 evidence fail closed."""

from __future__ import annotations

import hashlib
import csv
import json
import math
import os
import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any

try:
    from dev_harness.finalize_q4_sanitizer_evidence import load_json, sha256_file
    from dev_harness import run_q4_variant_sanitizer_matrix as runner
except ModuleNotFoundError:
    from finalize_q4_sanitizer_evidence import load_json, sha256_file  # type: ignore[no-redef]
    import run_q4_variant_sanitizer_matrix as runner  # type: ignore[no-redef]


class Q4SanitizerSchema2Error(ValueError):
    pass


TOP_KEYS = {
    "schema", "phase", "milestone", "production_enabled", "all_pass",
    "complete", "expected_run_ids", "execution_order", "identity",
    "device_snapshots", "modules", "runs",
}
IDENTITY_KEYS = {
    "policy", "policy_sha256", "metadata", "metadata_sha256",
    "harness_source", "harness_source_sha256", "harness_executable",
    "harness_executable_sha256", "runner", "runner_sha256",
    "compute_sanitizer", "cuobjdump", "nvidia_smi", "relevant_environment",
}
TOOL_KEYS = {
    "path", "sha256", "version_argv", "version_exit_code",
    "version_stdout", "version_stderr", "version",
}
MODULE_KEYS = {
    "label", "shape", "module", "module_sha256", "module_bytes", "symbol",
    "block_m", "block_n", "threads", "dynamic_shared_bytes",
    "registers_per_thread", "local_memory_bytes", "config_id",
    "identity_descriptor_sha256", "cuobjdump_argv", "cuobjdump_raw",
    "cuobjdump_raw_sha256", "module_stat_bytes", "cuobjdump_identity_sha256",
    "inspection",
}
RUN_KEYS = {
    "run_id", "label", "tool", "directory", "command", "command_sha256",
    "actual_argv", "started_utc", "ended_utc", "exit_code", "stdout",
    "stdout_sha256", "stderr", "stderr_sha256", "process_stdout_raw",
    "process_stdout_raw_sha256", "process_stderr_raw",
    "process_stderr_raw_sha256", "capture_normalization", "exit_code_file",
    "exit_code_file_sha256", "device_before", "device_after", "summary",
    "harness", "pass",
}
COMMAND_KEYS = {
    "schema", "run_id", "actual_argv", "runner", "runner_sha256",
    "compute_sanitizer", "relevant_environment", "capture_normalization",
    "process_stdout_raw", "process_stdout_raw_sha256", "process_stderr_raw",
    "process_stderr_raw_sha256", "started_utc", "ended_utc", "exit_code",
}
SNAPSHOT_KEYS = {"captured_utc", "gpu_csv", "compute_apps_csv"}
CASE_KEYS = {
    "shape_id", "n_in", "n_out", "n_tok", "out_stride", "native_route",
    "native_tiles", "logical_tiles", "physical_tiles", "efficiency",
    "numeric_seams", "fused_epilogue", "numeric_stream_grid",
    "triton_launch_config", "workspace_bytes", "comparison",
    "native_input_mismatches", "triton_input_mismatches",
    "native_padding_errors", "triton_padding_errors", "canary_errors",
    "cpu_oracle", "timing", "artifact_resources", "post_timing_comparison",
    "post_timing_native_input_mismatches",
    "post_timing_triton_input_mismatches",
    "post_timing_native_padding_errors", "post_timing_triton_padding_errors",
    "post_timing_canary_errors",
}
COMPARISON_KEYS = {
    "max_abs", "max_rel", "max_normalized_rel", "rms", "finite",
    "non_finite", "bitwise_different",
}
F64_COMPARISON_KEYS = {
    "max_abs", "max_rel", "max_normalized_rel", "rms", "count", "non_finite",
}
ORACLE_KEYS = {
    "samples", "seam_104_samples", "seam_208_samples", "no_seam_samples",
    "native_vs_strict_float", "triton_vs_strict_float", "native_vs_f64",
    "triton_vs_f64", "wrong_grid_bitwise_different",
    "seam_minus_one_bitwise_different", "seam_plus_one_bitwise_different",
    "wrong_grid_gpu_launched", "wrong_grid_gpu_canary_errors",
    "wrong_grid_gpu_input_mismatches", "wrong_grid_gpu_padding_errors",
    "directed_seam_tiles", "adjacent_no_seam_tiles",
    "directed_block_boundaries", "wrong_grid_gpu_vs_correct",
    "wrong_grid_gpu_vs_native",
}
TIMING_KEYS = {
    "clock", "schedule", "warmup", "pairs", "launches_per_sample",
    "samples_per_route", "native_first_launch_us", "triton_first_launch_us",
    "native_samples_us", "triton_samples_us", "native_median_us",
    "triton_median_us", "native_mean_us", "triton_mean_us", "native_cv",
    "triton_cv", "paired_mad_fraction", "median_speedup_native_over_triton",
}
RESOURCE_KEYS = {
    "cubin_bytes", "module_load_wall_us", "cuda_mem_free_before_buffers",
    "cuda_mem_free_after_buffers", "cuda_mem_free_before_module",
    "cuda_mem_free_after_module", "cuda_mem_free_after_first_launches",
    "cuda_mem_free_after_timing", "cuda_mem_total_bytes",
    "observed_buffer_delta_bytes", "observed_module_delta_bytes",
    "observed_first_launch_delta_bytes", "observed_timing_delta_bytes",
    "observed_peak_delta_bytes", "weights_logical_bytes", "q8_logical_bytes",
    "output_logical_bytes_per_buffer", "output_buffer_count",
    "workspace_logical_bytes", "guarded_allocation_bytes", "native_function",
    "triton_function",
}
FUNCTION_KEYS = {
    "registers_per_thread", "static_shared_bytes", "local_bytes",
    "max_threads_per_block", "max_dynamic_shared_bytes", "binary_version",
    "launch_threads", "launch_dynamic_shared_bytes", "block_m", "block_n",
}


def _exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = set(value) if isinstance(value, dict) else set()
        raise Q4SanitizerSchema2Error(
            f"{label} keys differ: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise Q4SanitizerSchema2Error(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> Any:
    raise Q4SanitizerSchema2Error(f"non-finite JSON constant: {value}")


def _strict_json_text(text: str, label: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise Q4SanitizerSchema2Error(f"{label} strict JSON rejected: {error}") from error


def _load_json_strict(path: Path, label: str) -> dict[str, Any]:
    try:
        value = _strict_json_text(path.read_text(encoding="utf-8"), label)
    except OSError as error:
        raise Q4SanitizerSchema2Error(f"read {label}: {error}") from error
    if not isinstance(value, dict):
        raise Q4SanitizerSchema2Error(f"{label} is not an object")
    return value


def _assert_finite(value: Any, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise Q4SanitizerSchema2Error(f"{label} contains non-finite number")
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite(item, f"{label}[{index}]")


def _has_reparse_point(path: Path) -> bool:
    try:
        stat = os.lstat(path)
    except OSError as error:
        raise Q4SanitizerSchema2Error(f"cannot lstat evidence path {path}: {error}") from error
    attributes = getattr(stat, "st_file_attributes", 0)
    return path.is_symlink() or bool(attributes & 0x400)


def _evidence_path(value: Any, root: Path, expected: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise Q4SanitizerSchema2Error(f"{label} path is missing")
    lexical = Path(value)
    if not lexical.is_absolute() or value != str(lexical):
        raise Q4SanitizerSchema2Error(f"{label} path is not lexical absolute canonical form")
    resolved = lexical.resolve()
    if resolved != expected.resolve():
        raise Q4SanitizerSchema2Error(f"{label} path differs")
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise Q4SanitizerSchema2Error(f"{label} escapes evidence root") from error
    cursor = resolved
    while True:
        if _has_reparse_point(cursor):
            raise Q4SanitizerSchema2Error(f"{label} crosses symlink/junction/reparse point")
        if cursor == root.resolve():
            break
        cursor = cursor.parent
    return resolved


def _resolve(value: Any, anchor: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise Q4SanitizerSchema2Error(f"{label} path is missing")
    path = Path(value)
    return (path if path.is_absolute() else anchor.parent / path).resolve()


def _require_hash(path: Path, expected: Any, label: str) -> str:
    if not path.is_file() or not isinstance(expected, str):
        raise Q4SanitizerSchema2Error(f"{label} file/hash is missing")
    actual = sha256_file(path)
    if actual != expected:
        raise Q4SanitizerSchema2Error(f"{label} SHA-256 mismatch")
    return actual


def _tool_identity(value: Any, manifest: Path, product: str) -> dict[str, Any]:
    identity = _exact_keys(value, TOOL_KEYS, f"{product} identity")
    path = _resolve(identity["path"], manifest, f"{product} executable")
    digest = _require_hash(path, identity["sha256"], f"{product} executable")
    if (
        identity["path"] != str(path)
        or identity["version_argv"] != [str(path), "--version"]
        or identity["version_exit_code"] != 0
        or not isinstance(identity["version_stdout"], str)
        or not isinstance(identity["version_stderr"], str)
        or identity["version"]
        != (identity["version_stdout"] + identity["version_stderr"]).strip()
        or not identity["version"]
        or product not in identity["version"].lower()
    ):
        raise Q4SanitizerSchema2Error(f"{product} version identity differs")
    completed = subprocess.run(
        identity["version_argv"], check=False, capture_output=True, text=True
    )
    if (
        completed.returncode != identity["version_exit_code"]
        or completed.stdout != identity["version_stdout"]
        or completed.stderr != identity["version_stderr"]
    ):
        raise Q4SanitizerSchema2Error(f"{product} local version replay differs")
    return {**identity, "path": str(path), "sha256": digest}


def _snapshot(value: Any, label: str) -> tuple[dict[str, Any], tuple[str, str, str]]:
    snapshot = _exact_keys(value, SNAPSHOT_KEYS, label)
    if (
        not isinstance(snapshot["captured_utc"], str)
        or not snapshot["captured_utc"]
        or not isinstance(snapshot["gpu_csv"], str)
        or not snapshot["gpu_csv"].strip()
        or len(snapshot["gpu_csv"].splitlines()) != 1
        or not isinstance(snapshot["compute_apps_csv"], list)
        or any(not isinstance(line, str) for line in snapshot["compute_apps_csv"])
    ):
        raise Q4SanitizerSchema2Error(f"{label} is incomplete")
    try:
        row = next(csv.reader([snapshot["gpu_csv"]], skipinitialspace=True))
    except (csv.Error, StopIteration) as error:
        raise Q4SanitizerSchema2Error(f"{label} GPU CSV is invalid") from error
    if (
        len(row) != 14
        or not row[1]
        or not row[2]
        or re.fullmatch(r"\d+(?:\.\d+)+", row[3]) is None
    ):
        raise Q4SanitizerSchema2Error(f"{label} GPU CSV identity differs")
    uuid = row[1].removeprefix("GPU-").replace("-", "").lower()
    if re.fullmatch(r"[0-9a-f]{32}", uuid) is None:
        raise Q4SanitizerSchema2Error(f"{label} GPU UUID differs")
    return snapshot, (uuid, row[2], row[3])


def _strict_harness(stdout: str, pair: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    lines = stdout.splitlines()
    if len(lines) != 1 or not lines[0].strip():
        raise Q4SanitizerSchema2Error("raw harness must contain exactly one JSON line")
    document = _strict_json_text(lines[0], "raw harness")
    if not isinstance(document, dict):
        raise Q4SanitizerSchema2Error("raw harness is not an object")
    _assert_finite(document, "raw harness")
    _exact_keys(document, {
        "schema", "phase", "milestone", "production_enabled", "target_sm",
        "device", "same_primary_context", "same_stream", "visible_arguments",
        "hidden_arguments", "tail_tokens", "measurement_request",
        "policy_admissible", "metadata_kparam_preflight",
        "debug_exact_triton_native_required",
        "debug_exact_triton_native_observed", "debug_exact_triton_native_pass",
        "direct_debug_exit_requires_exact", "formal_pass", "cases",
        "structural_ok",
    }, "raw harness top")
    if document.get("debug_exact_triton_native_pass") is not False:
        raise Q4SanitizerSchema2Error("raw harness debug exact flag differs")
    cases = document.get("cases")
    if not isinstance(cases, list) or len(cases) != 12:
        raise Q4SanitizerSchema2Error("raw harness case count differs")
    for index, case_value in enumerate(cases):
        case = _exact_keys(case_value, CASE_KEYS, f"raw harness case[{index}]")
        _exact_keys(case.get("comparison"), COMPARISON_KEYS, f"case[{index}].comparison")
        _exact_keys(case.get("triton_launch_config"), {
            "bm", "bn", "threads", "dynamic_shared_bytes",
        }, f"case[{index}].launch")
        _exact_keys(case.get("native_tiles"), {"rows", "tokens"}, f"case[{index}].native_tiles")
        if case.get("n_tok") == 512:
            oracle = _exact_keys(case.get("cpu_oracle"), ORACLE_KEYS, f"case[{index}].oracle")
            for name in ("native_vs_f64", "triton_vs_f64"):
                _exact_keys(
                    oracle.get(name), F64_COMPARISON_KEYS,
                    f"case[{index}].oracle.{name}",
                )
            for name in (
                "native_vs_strict_float", "triton_vs_strict_float",
                "wrong_grid_gpu_vs_correct", "wrong_grid_gpu_vs_native",
            ):
                _exact_keys(
                    oracle.get(name), COMPARISON_KEYS,
                    f"case[{index}].oracle.{name}",
                )
            _exact_keys(case.get("timing"), TIMING_KEYS, f"case[{index}].timing")
            resources = _exact_keys(
                case.get("artifact_resources"), RESOURCE_KEYS,
                f"case[{index}].resources",
            )
            _exact_keys(resources.get("native_function"), FUNCTION_KEYS, f"case[{index}].native_function")
            _exact_keys(resources.get("triton_function"), FUNCTION_KEYS, f"case[{index}].triton_function")
            _exact_keys(case.get("post_timing_comparison"), COMPARISON_KEYS, f"case[{index}].post")
        elif any(case.get(name) is not None for name in (
            "cpu_oracle", "timing", "artifact_resources", "post_timing_comparison",
        )):
            raise Q4SanitizerSchema2Error(f"case[{index}] tail formal fields differ")
    try:
        summary = runner.validate_harness_stdout(stdout, pair)
    except (runner.EvidenceError, json.JSONDecodeError) as error:
        raise Q4SanitizerSchema2Error(f"raw harness contract rejected: {error}") from error
    return summary, document


def validate_schema2_matrix(
    manifest_path: Path,
    policy_path: Path,
    *,
    expected_metadata_sha256: str,
    expected_harness_source_sha256: str,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    policy_path = policy_path.resolve()
    if manifest_path.name != "sanitizer-evidence-v2.json":
        raise Q4SanitizerSchema2Error("schema2 final filename differs")
    evidence_root = manifest_path.parent.resolve()
    _evidence_path(str(manifest_path), evidence_root, manifest_path, "schema2 manifest")
    manifest = _load_json_strict(manifest_path, "schema2 manifest")
    _assert_finite(manifest, "schema2 manifest")
    _exact_keys(manifest, TOP_KEYS, "schema2 manifest")
    if (
        manifest["schema"] != 2
        or manifest["phase"] != "A2"
        or manifest["milestone"] != "q4-variant-sanitizer-matrix-v2"
        or manifest["production_enabled"] is not False
        or manifest["all_pass"] is not True
        or manifest["complete"] is not True
        or manifest["execution_order"]
        != {"labels": list(runner.LABELS), "tools": list(runner.TOOLS)}
    ):
        raise Q4SanitizerSchema2Error("schema2 identity/order/completion differs")
    try:
        policy = tomllib.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise Q4SanitizerSchema2Error(f"read policy: {error}") from error
    if policy.get("schema") != 1 or policy.get("target_sm") != 86:
        raise Q4SanitizerSchema2Error("policy identity differs")

    identity = _exact_keys(manifest["identity"], IDENTITY_KEYS, "schema2 identity")
    bound_policy = _resolve(identity["policy"], manifest_path, "policy")
    metadata_path = _resolve(identity["metadata"], manifest_path, "metadata")
    source_path = _resolve(identity["harness_source"], manifest_path, "harness source")
    executable_path = _resolve(
        identity["harness_executable"], manifest_path, "harness executable"
    )
    runner_path = _resolve(identity["runner"], manifest_path, "runner")
    if bound_policy != policy_path:
        raise Q4SanitizerSchema2Error("schema2 policy path differs")
    policy_hash = _require_hash(policy_path, identity["policy_sha256"], "policy")
    metadata_hash = _require_hash(
        metadata_path, identity["metadata_sha256"], "metadata"
    )
    source_hash = _require_hash(
        source_path, identity["harness_source_sha256"], "harness source"
    )
    executable_hash = _require_hash(
        executable_path, identity["harness_executable_sha256"], "harness executable"
    )
    current_runner = Path(runner.__file__).resolve()
    runner_hash = _require_hash(runner_path, identity["runner_sha256"], "runner")
    if (
        metadata_hash != expected_metadata_sha256
        or source_hash != expected_harness_source_sha256
        or runner_path != current_runner
        or identity["policy"] != str(policy_path)
        or identity["metadata"] != str(metadata_path)
        or identity["harness_source"] != str(source_path)
        or identity["harness_executable"] != str(executable_path)
        or identity["runner"] != str(runner_path)
    ):
        raise Q4SanitizerSchema2Error("schema2 immutable authority path/hash differs")
    compute = _tool_identity(identity["compute_sanitizer"], manifest_path, "compute sanitizer")
    cuobjdump = _tool_identity(identity["cuobjdump"], manifest_path, "cuobjdump")
    nvidia_smi = _tool_identity(identity["nvidia_smi"], manifest_path, "nvidia-smi")
    environment = _exact_keys(
        identity["relevant_environment"],
        set(runner.RELEVANT_ENVIRONMENT) | {"PATH", "PATH_sha256"},
        "relevant environment",
    )
    if (
        environment["IMPARO_CUDA_KERNEL_LAB"] != "1"
        or not isinstance(environment["PATH"], str)
        or environment["PATH_sha256"]
        != hashlib.sha256(environment["PATH"].encode("utf-8")).hexdigest()
    ):
        raise Q4SanitizerSchema2Error("relevant environment differs")

    metadata = _load_json_strict(metadata_path, "metadata")
    pairs = runner.build_pairs(metadata, metadata_path)
    expected_ids = runner.exact_run_ids(pairs)
    if manifest["expected_run_ids"] != expected_ids:
        raise Q4SanitizerSchema2Error("expected run IDs differ from exact Cartesian")
    snapshots = _exact_keys(
        manifest["device_snapshots"], {"before", "after"}, "device snapshots"
    )
    _, matrix_before_identity = _snapshot(snapshots["before"], "matrix device before")
    _, matrix_after_identity = _snapshot(snapshots["after"], "matrix device after")
    if matrix_before_identity != matrix_after_identity:
        raise Q4SanitizerSchema2Error("matrix GPU snapshot identities differ")

    expected_variants = [
        variant
        for pair in pairs
        for variant in (pair["expansion"], pair["contraction"])
    ]
    modules = manifest["modules"]
    if not isinstance(modules, list) or len(modules) != 8:
        raise Q4SanitizerSchema2Error("schema2 module set is not exactly eight")
    preflight: list[dict[str, Any]] = []
    seen_raw: set[Path] = set()
    for index, (entry_value, wanted) in enumerate(zip(modules, expected_variants)):
        entry = _exact_keys(entry_value, MODULE_KEYS, f"module[{index}]")
        base = {key: wanted[key] for key in (
            "label", "shape", "module", "module_sha256", "module_bytes", "symbol",
            "block_m", "block_n", "threads", "dynamic_shared_bytes",
            "registers_per_thread", "local_memory_bytes", "config_id",
            "identity_descriptor_sha256",
        )}
        if any(entry[key] != value for key, value in base.items()):
            raise Q4SanitizerSchema2Error(f"module[{index}] differs from metadata")
        module_path = Path(wanted["module"]).resolve()
        _require_hash(module_path, wanted["module_sha256"], f"module[{index}]")
        if entry["module_stat_bytes"] != module_path.stat().st_size:
            raise Q4SanitizerSchema2Error(f"module[{index}] stat differs")
        raw_expected = manifest_path.parent / "preflight" / (
            f"{wanted['label']}--{wanted['shape']}.cuobjdump.txt"
        )
        raw_path = _evidence_path(
            entry["cuobjdump_raw"], evidence_root, raw_expected,
            f"module[{index}] raw",
        )
        if raw_path in seen_raw:
            raise Q4SanitizerSchema2Error(f"module[{index}] raw path differs/reused")
        seen_raw.add(raw_path)
        raw_hash = _require_hash(raw_path, entry["cuobjdump_raw_sha256"], f"module[{index}] raw")
        if (
            entry["cuobjdump_argv"] != [cuobjdump["path"], "-elf", str(module_path)]
            or entry["cuobjdump_identity_sha256"] != cuobjdump["sha256"]
        ):
            raise Q4SanitizerSchema2Error(f"module[{index}] cuobjdump authority differs")
        replay = subprocess.run(
            entry["cuobjdump_argv"], check=False, capture_output=True, text=True
        )
        if replay.returncode != 0 or replay.stdout + replay.stderr != raw_path.read_text(encoding="utf-8"):
            raise Q4SanitizerSchema2Error(f"module[{index}] local cuobjdump replay differs")
        try:
            inspection = json.loads(json.dumps(runner.parse_cuobjdump(
                replay.stdout, symbol=wanted["symbol"],
                threads=wanted["threads"], registers=wanted["registers_per_thread"],
            )))
        except runner.EvidenceError as error:
            raise Q4SanitizerSchema2Error(f"module[{index}] inspection rejected: {error}") from error
        if entry["inspection"] != inspection:
            raise Q4SanitizerSchema2Error(f"module[{index}] inspection differs from raw")
        preflight.append({
            "label": wanted["label"], "shape_id": wanted["shape"],
            "config_id": wanted["config_id"], "symbol": wanted["symbol"],
            "module": str(module_path), "module_sha256": wanted["module_sha256"],
            "module_bytes": wanted["module_bytes"],
            "n_in": 2560 if wanted["shape"] == "k2560-m10240" else 10240,
            "n_out": 10240 if wanted["shape"] == "k2560-m10240" else 2560,
            "block_m": wanted["block_m"], "block_n": wanted["block_n"],
            "threads": wanted["threads"],
            "dynamic_shared_bytes": wanted["dynamic_shared_bytes"],
            "registers_per_thread": wanted["registers_per_thread"],
            "raw_cuobjdump": str(raw_path), "raw_cuobjdump_sha256": raw_hash,
            "inspection": inspection,
        })

    runs = manifest["runs"]
    if not isinstance(runs, list) or len(runs) != 16:
        raise Q4SanitizerSchema2Error("schema2 run set is not exactly sixteen")
    normalized_runs: list[dict[str, Any]] = []
    common_device: dict[str, Any] | None = None
    seen_artifacts: set[Path] = set()
    pair_map = {pair["label"]: pair for pair in pairs}
    for index, (entry_value, run_id) in enumerate(zip(runs, expected_ids), start=1):
        entry = _exact_keys(entry_value, RUN_KEYS, f"run[{index}]")
        label, tool = run_id.rsplit("--", 1)
        pair = pair_map[label]
        expected_dir = manifest_path.parent / "runs" / f"{index:02d}--{run_id}"
        run_dir = _evidence_path(
            entry["directory"], evidence_root, expected_dir,
            f"run[{index}] directory",
        )
        if entry["run_id"] != run_id or entry["label"] != label or entry["tool"] != tool:
            raise Q4SanitizerSchema2Error(f"run[{index}] Cartesian/path identity differs")
        expected_argv = [
            compute["path"], "--tool", tool, "--error-exitcode", "86",
            *(("--leak-check", "full") if tool == "memcheck" else ()),
            str(executable_path), *runner.harness_args(pair),
        ]
        if entry["actual_argv"] != expected_argv or entry["exit_code"] != 0 or entry["pass"] is not True:
            raise Q4SanitizerSchema2Error(f"run[{index}] argv/exit/pass differs")
        names = {
            "command": "command.json", "stdout": "stdout.json", "stderr": "stderr.log",
            "process_stdout_raw": "process-stdout.raw",
            "process_stderr_raw": "process-stderr.raw", "exit_code_file": "exit-code.txt",
        }
        resolved: dict[str, Path] = {}
        for field, filename in names.items():
            path = _evidence_path(
                entry[field], evidence_root, run_dir / filename,
                f"run[{index}].{field}",
            )
            if path in seen_artifacts:
                raise Q4SanitizerSchema2Error(f"run[{index}].{field} path differs/reused")
            seen_artifacts.add(path)
            _require_hash(path, entry[f"{field}_sha256"], f"run[{index}].{field}")
            resolved[field] = path
        if resolved["exit_code_file"].read_text(encoding="utf-8") != "0\n":
            raise Q4SanitizerSchema2Error(f"run[{index}] exit file differs")
        raw_stdout = resolved["process_stdout_raw"].read_text(encoding="utf-8")
        raw_stderr = resolved["process_stderr_raw"].read_text(encoding="utf-8")
        try:
            stdout_text, stderr_text, normalization = runner.split_process_streams(
                raw_stdout, raw_stderr
            )
        except runner.EvidenceError as error:
            raise Q4SanitizerSchema2Error(f"run[{index}] raw capture rejected: {error}") from error
        normalization["passed"] = True
        if (
            resolved["stdout"].read_text(encoding="utf-8") != stdout_text
            or resolved["stderr"].read_text(encoding="utf-8") != stderr_text
            or entry["capture_normalization"] != normalization
        ):
            raise Q4SanitizerSchema2Error(f"run[{index}] normalized capture differs")
        summary = runner.sanitizer_summary(tool, stderr_text)
        harness, _ = _strict_harness(stdout_text, pair)
        if entry["summary"] != summary or summary["passed"] is not True or entry["harness"] != harness or harness["passed"] is not True:
            raise Q4SanitizerSchema2Error(f"run[{index}] summary/harness differs")
        command = _load_json_strict(resolved["command"], f"run[{index}] command")
        _assert_finite(command, f"run[{index}] command")
        _exact_keys(command, COMMAND_KEYS, f"run[{index}] command")
        expected_command = {
            "schema": 1, "run_id": run_id, "actual_argv": expected_argv,
            "runner": str(runner_path), "runner_sha256": runner_hash,
            "compute_sanitizer": identity["compute_sanitizer"],
            "relevant_environment": environment,
            "capture_normalization": normalization,
            "process_stdout_raw": str(resolved["process_stdout_raw"]),
            "process_stdout_raw_sha256": entry["process_stdout_raw_sha256"],
            "process_stderr_raw": str(resolved["process_stderr_raw"]),
            "process_stderr_raw_sha256": entry["process_stderr_raw_sha256"],
            "started_utc": entry["started_utc"], "ended_utc": entry["ended_utc"],
            "exit_code": 0,
        }
        if command != expected_command:
            raise Q4SanitizerSchema2Error(f"run[{index}] command authority differs")
        _, before_identity = _snapshot(entry["device_before"], f"run[{index}] device before")
        _, after_identity = _snapshot(entry["device_after"], f"run[{index}] device after")
        device = harness["device"]
        device_uuid = device["uuid"].removeprefix("GPU-").replace("-", "").lower()
        expected_gpu_identity = (device_uuid, device["name"], matrix_before_identity[2])
        if (
            before_identity != expected_gpu_identity
            or after_identity != expected_gpu_identity
            or matrix_before_identity != expected_gpu_identity
        ):
            raise Q4SanitizerSchema2Error(f"run[{index}] GPU snapshot identity differs")
        if common_device is None:
            common_device = device
        elif device != common_device:
            raise Q4SanitizerSchema2Error("schema2 harness device identities differ")
        normalized_runs.append({
            "label": label, "tool": tool, "run_id": run_id,
            "directory": str(run_dir), "command": str(resolved["command"]),
            "command_sha256": entry["command_sha256"], "actual_argv": expected_argv,
            "log": str(resolved["stderr"]), "log_sha256": entry["stderr_sha256"],
            "harness_output": str(resolved["stdout"]),
            "harness_output_sha256": entry["stdout_sha256"],
            "process_stdout_raw": str(resolved["process_stdout_raw"]),
            "process_stdout_raw_sha256": entry["process_stdout_raw_sha256"],
            "process_stderr_raw": str(resolved["process_stderr_raw"]),
            "process_stderr_raw_sha256": entry["process_stderr_raw_sha256"],
            "exit_code_file": str(resolved["exit_code_file"]),
            "exit_code_file_sha256": entry["exit_code_file_sha256"],
            "capture_normalization": normalization, "summary": summary,
            "harness": harness, "exit_code": 0, "pass": True,
        })
    if [run["run_id"] for run in normalized_runs] != expected_ids:
        raise Q4SanitizerSchema2Error("schema2 run order differs")
    if common_device is None:
        raise Q4SanitizerSchema2Error("schema2 device evidence is missing")
    common_uuid = common_device["uuid"].removeprefix("GPU-").replace("-", "").lower()
    expected_matrix_gpu = (
        common_uuid, common_device["name"], matrix_before_identity[2]
    )
    if (
        matrix_before_identity != expected_matrix_gpu
        or matrix_after_identity != expected_matrix_gpu
    ):
        raise Q4SanitizerSchema2Error(
            "matrix GPU snapshots do not bind the normalized harness GPU identity"
        )
    return {
        "schema": 2,
        "decision": "sanitizer-matrix-evidence",
        "candidate": "q4_0_x_q8_1_mmq_epilogue0",
        "production_authority": False,
        "final_gate_a_decision": False,
        "target_sm": 86,
        "all_pass": True,
        "device": common_device,
        "pairs": {
            pair["label"]: {
                "k2560-m10240": pair["expansion"]["config_id"],
                "k10240-m2560": pair["contraction"]["config_id"],
            }
            for pair in pairs
        },
        "preflight_count": 8,
        "preflight": preflight,
        "run_count": 16,
        "runs": normalized_runs,
        "provenance": {
            "policy": str(policy_path), "policy_sha256": policy_hash,
            "metadata": str(metadata_path), "metadata_sha256": metadata_hash,
            "harness_source": str(source_path), "harness_source_sha256": source_hash,
            "harness_exe": str(executable_path), "harness_exe_sha256": executable_hash,
            "runner": str(runner_path), "runner_sha256": runner_hash,
            "compute_sanitizer": compute, "cuobjdump": cuobjdump,
            "nvidia_smi": nvidia_smi, "relevant_environment": environment,
        },
    }
