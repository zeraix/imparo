#!/usr/bin/env python3
"""Validate and hash-bind the bounded Q4 sanitizer matrix without GPU work."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tomllib
from pathlib import Path
from typing import Any


LABELS = (
    "bm128-bn64-w8-s2",
    "bm64-bn128-w8-s2",
    "bm64-bn64-w8-s2",
    "bm64-bn64-w8-s1",
)
TOOLS = ("memcheck", "initcheck", "racecheck", "synccheck")
SHAPES = {"k2560-m10240", "k10240-m2560"}
TAILS = (1, 127, 128, 129, 511, 512)


class Q4SanitizerEvidenceError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise Q4SanitizerEvidenceError(f"duplicate JSON key {key!r} in {path}")
            value[key] = item
        return value

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicates)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Q4SanitizerEvidenceError(f"read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise Q4SanitizerEvidenceError(f"JSON root is not an object: {path}")
    return value


def resolve_path(raw: object, owner: Path, field: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise Q4SanitizerEvidenceError(f"{field} path is missing")
    path = Path(raw)
    if not path.is_absolute():
        path = owner.parent / path
    return path.resolve()


def require_hash(path: Path, expected: object, field: str) -> str:
    if not isinstance(expected, str) or len(expected) != 64:
        raise Q4SanitizerEvidenceError(f"{field} SHA-256 is missing")
    actual = sha256_file(path)
    if actual != expected:
        raise Q4SanitizerEvidenceError(
            f"{field} SHA-256 mismatch: expected {expected}, observed {actual}"
        )
    return actual


def finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Q4SanitizerEvidenceError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise Q4SanitizerEvidenceError(f"{field} must be finite")
    return number


def check_difference(
    value: object, *, max_abs: float, max_rms: float, max_normalized_rel: float, field: str
) -> None:
    difference = value if isinstance(value, dict) else {}
    if (
        difference.get("non_finite") != 0
        or finite(difference.get("max_abs"), f"{field}.max_abs") > max_abs
        or finite(difference.get("rms"), f"{field}.rms") > max_rms
        or finite(difference.get("max_normalized_rel"), f"{field}.max_normalized_rel")
        > max_normalized_rel
    ):
        raise Q4SanitizerEvidenceError(f"{field} exceeds Q4 numerical policy")


def validate_harness(
    path: Path,
    *,
    label: str,
    expected_by_shape: dict[str, dict[str, Any]],
    policy: dict[str, Any],
    measurement_request: dict[str, Any] | None = None,
    require_formal_timing: bool = False,
) -> dict[str, Any]:
    data = load_json(path)
    q4 = policy["q4_mmq"]
    if (
        data.get("schema") != 1
        or data.get("phase") != "A2"
        or data.get("milestone") != "C0"
        or data.get("production_enabled") is not False
        or data.get("target_sm") != 86
        or data.get("same_primary_context") is not True
        or data.get("same_stream") is not True
        or data.get("visible_arguments") != q4["q4_visible_arguments"]
        or data.get("hidden_arguments") != q4["q4_hidden_arguments"]
        or data.get("tail_tokens") != list(TAILS)
        or data.get("policy_admissible") is not False
        or data.get("formal_pass") is not False
        or data.get("structural_ok") is not True
    ):
        raise Q4SanitizerEvidenceError(f"{label} harness top-level C0 contract mismatch")
    request = data.get("measurement_request")
    expected_request = measurement_request or {
        "warmup": 1,
        "pairs": 1,
        "launches_per_sample": 1,
        "samples_per_route": 2,
        "formal_contract": False,
    }
    if not isinstance(request, dict) or request != expected_request:
        raise Q4SanitizerEvidenceError(f"{label} sanitizer launch contract mismatch")
    device = data.get("device")
    if not isinstance(device, dict) or any(
        key not in device
        for key in (
            "uuid",
            "name",
            "sm",
            "sm_count",
            "driver_version",
            "cuda_runtime_version",
        )
    ) or device.get("sm") != 86:
        raise Q4SanitizerEvidenceError(f"{label} harness device identity is incomplete")
    cases = data.get("cases")
    if not isinstance(cases, list) or len(cases) != 12:
        raise Q4SanitizerEvidenceError(f"{label} harness must contain 12 cases")
    seen: set[tuple[str, int]] = set()
    for case_value in cases:
        if not isinstance(case_value, dict):
            raise Q4SanitizerEvidenceError(f"{label} case is not an object")
        shape = case_value.get("shape_id")
        token = case_value.get("n_tok")
        key = (shape, token)
        if shape not in SHAPES or token not in TAILS or key in seen:
            raise Q4SanitizerEvidenceError(f"{label} harness case set is duplicate/unexpected")
        seen.add(key)
        expected = expected_by_shape[shape]
        launch = case_value.get("triton_launch_config")
        expected_launch = {
            "bm": expected["block_m"],
            "bn": expected["block_n"],
            "threads": expected["threads"],
            "dynamic_shared_bytes": expected["dynamic_shared_bytes"],
        }
        if launch != expected_launch:
            raise Q4SanitizerEvidenceError(f"{label}/{shape}/{token} launch geometry mismatch")
        expected_finite = expected["n_out"] * token
        comparison = case_value.get("comparison")
        check_difference(
            comparison,
            max_abs=q4["q4_max_abs_vs_native"],
            max_rms=q4["q4_max_rms_vs_native"],
            max_normalized_rel=q4["q4_max_normalized_rel_vs_native"],
            field=f"{label}/{shape}/{token}.comparison",
        )
        if not isinstance(comparison, dict) or comparison.get("finite") != expected_finite:
            raise Q4SanitizerEvidenceError(f"{label}/{shape}/{token} finite count mismatch")
        for field in (
            "native_input_mismatches",
            "triton_input_mismatches",
            "native_padding_errors",
            "triton_padding_errors",
            "canary_errors",
        ):
            if case_value.get(field) != 0:
                raise Q4SanitizerEvidenceError(f"{label}/{shape}/{token} {field} is non-zero")
        if token != 512:
            if case_value.get("timing") is not None or case_value.get("cpu_oracle") is not None:
                raise Q4SanitizerEvidenceError(f"{label}/{shape}/{token} carries unexpected timing/oracle")
            continue
        timing = case_value.get("timing")
        if require_formal_timing:
            if not isinstance(timing, dict):
                raise Q4SanitizerEvidenceError(f"{label}/{shape}/512 formal timing is missing")
            if (
                timing.get("clock") != "cuda_event"
                or timing.get("schedule") != "ABBA_BAAB"
                or timing.get("warmup") != expected_request["warmup"]
                or timing.get("pairs") != expected_request["pairs"]
                or timing.get("launches_per_sample") != expected_request["launches_per_sample"]
                or timing.get("samples_per_route") != expected_request["samples_per_route"]
                or not isinstance(timing.get("native_samples_us"), list)
                or len(timing["native_samples_us"]) != expected_request["samples_per_route"]
                or not isinstance(timing.get("triton_samples_us"), list)
                or len(timing["triton_samples_us"]) != expected_request["samples_per_route"]
                or finite(timing.get("native_median_us"), "native median") <= 0
                or finite(timing.get("triton_median_us"), "Triton median") <= 0
                or finite(timing.get("native_cv"), "native CV") > policy["noise"]["max_native_cv"]
                or finite(timing.get("triton_cv"), "Triton CV") > policy["noise"]["max_candidate_cv"]
                or finite(timing.get("paired_mad_fraction"), "paired MAD")
                > policy["noise"]["max_paired_mad_fraction"]
            ):
                raise Q4SanitizerEvidenceError(f"{label}/{shape}/512 formal timing/noise gate failed")
            if any(
                finite(sample, "timing sample") <= 0
                for sample in timing["native_samples_us"] + timing["triton_samples_us"]
            ):
                raise Q4SanitizerEvidenceError(f"{label}/{shape}/512 timing sample is invalid")
            resource = case_value.get("artifact_resources")
            triton_function = resource.get("triton_function") if isinstance(resource, dict) else None
            native_function = resource.get("native_function") if isinstance(resource, dict) else None
            if (
                not isinstance(resource, dict)
                or resource.get("cubin_bytes") != expected["module_bytes"]
                or finite(resource.get("module_load_wall_us"), "module load") <= 0
                or not isinstance(triton_function, dict)
                or triton_function.get("registers_per_thread") != expected["registers_per_thread"]
                or triton_function.get("local_bytes") != 0
                or triton_function.get("static_shared_bytes") != 0
                or triton_function.get("launch_threads") != expected["threads"]
                or triton_function.get("launch_dynamic_shared_bytes")
                != expected["dynamic_shared_bytes"]
                or triton_function.get("block_m") != expected["block_m"]
                or triton_function.get("block_n") != expected["block_n"]
                or not isinstance(native_function, dict)
                or native_function.get("local_bytes") != 0
                or native_function.get("binary_version") != 86
            ):
                raise Q4SanitizerEvidenceError(f"{label}/{shape}/512 resource gate failed")
        post = case_value.get("post_timing_comparison")
        check_difference(
            post,
            max_abs=q4["q4_max_abs_vs_native"],
            max_rms=q4["q4_max_rms_vs_native"],
            max_normalized_rel=q4["q4_max_normalized_rel_vs_native"],
            field=f"{label}/{shape}/512.post_timing",
        )
        if not isinstance(post, dict) or post.get("finite") != expected_finite:
            raise Q4SanitizerEvidenceError(f"{label}/{shape}/512 post-timing finite mismatch")
        for field in (
            "post_timing_native_input_mismatches",
            "post_timing_triton_input_mismatches",
            "post_timing_native_padding_errors",
            "post_timing_triton_padding_errors",
            "post_timing_canary_errors",
        ):
            if case_value.get(field) != 0:
                raise Q4SanitizerEvidenceError(f"{label}/{shape}/512 {field} is non-zero")
        oracle = case_value.get("cpu_oracle")
        if not isinstance(oracle, dict):
            raise Q4SanitizerEvidenceError(f"{label}/{shape}/512 CPU oracle is missing")
        native_strict = oracle.get("native_vs_strict_float")
        triton_strict = oracle.get("triton_vs_strict_float")
        native_f64 = oracle.get("native_vs_f64")
        triton_f64 = oracle.get("triton_vs_f64")
        if (
            not isinstance(native_strict, dict)
            or native_strict.get("non_finite") != 0
            or finite(native_strict.get("max_abs"), "native strict max_abs")
            > q4["q4_oracle_strict_native_max_abs"]
            or not isinstance(triton_strict, dict)
            or triton_strict.get("non_finite") != 0
            or finite(triton_strict.get("max_abs"), "Triton strict max_abs")
            > q4["q4_oracle_strict_triton_max_abs"]
            or not isinstance(native_f64, dict)
            or native_f64.get("non_finite") != 0
            or finite(native_f64.get("max_abs"), "native f64 max_abs")
            > q4["q4_oracle_f64_max_abs"]
            or not isinstance(triton_f64, dict)
            or triton_f64.get("non_finite") != 0
            or finite(triton_f64.get("max_abs"), "Triton f64 max_abs")
            > q4["q4_oracle_f64_max_abs"]
        ):
            raise Q4SanitizerEvidenceError(f"{label}/{shape}/512 CPU oracle gate failed")
        if shape == "k10240-m2560" and (
            oracle.get("seam_minus_one_bitwise_different", 0)
            < q4["q4_mutation_min_bitwise_different"]
            or oracle.get("seam_plus_one_bitwise_different", 0)
            < q4["q4_mutation_min_bitwise_different"]
            or oracle.get("wrong_grid_gpu_launched") is not True
            or oracle.get("wrong_grid_bitwise_different", 0)
            < q4["q4_mutation_min_bitwise_different"]
            or oracle.get("wrong_grid_gpu_canary_errors") != 0
            or oracle.get("wrong_grid_gpu_input_mismatches") != 0
            or oracle.get("wrong_grid_gpu_padding_errors") != 0
        ):
            raise Q4SanitizerEvidenceError(f"{label}/{shape}/512 mutation gate failed")
    if seen != {(shape, token) for shape in SHAPES for token in TAILS}:
        raise Q4SanitizerEvidenceError(f"{label} harness case coverage is incomplete")
    return device


def validate_matrix(manifest_path: Path, policy_path: Path) -> dict[str, Any]:
    try:
        policy = tomllib.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise Q4SanitizerEvidenceError(f"read policy: {error}") from error
    q4 = policy.get("q4_mmq")
    if policy.get("schema") != 1 or policy.get("target_sm") != 86 or not isinstance(q4, dict):
        raise Q4SanitizerEvidenceError("Q4 policy identity is invalid")
    manifest = load_json(manifest_path)
    if (
        manifest.get("schema") != 1
        or manifest.get("phase") != "A2"
        or manifest.get("milestone") != "post-c0-sanitizer-matrix"
        or manifest.get("production_enabled") is not False
        or manifest.get("target_sm") != 86
        or manifest.get("execution_order") != list(LABELS)
        or manifest.get("tool_order") != list(TOOLS)
        or manifest.get("all_pass") is not True
    ):
        raise Q4SanitizerEvidenceError("sanitizer matrix identity/order/status mismatch")
    launch = manifest.get("launch_contract")
    if launch != {
        "warmup": 1,
        "pairs": 1,
        "launches_per_sample": 1,
        "separate_process_per_tool": True,
        "fresh_primary_context_per_tool": True,
    }:
        raise Q4SanitizerEvidenceError("sanitizer matrix launch contract mismatch")
    metadata_path = resolve_path(manifest.get("metadata"), manifest_path, "metadata")
    require_hash(metadata_path, manifest.get("metadata_sha256"), "metadata")
    if manifest.get("metadata_sha256") != q4["q4_variants_metadata_sha256"]:
        raise Q4SanitizerEvidenceError("sanitizer matrix metadata differs from policy")
    metadata = load_json(metadata_path)
    survivors = metadata.get("survivors")
    if not isinstance(survivors, list) or len(survivors) != 8:
        raise Q4SanitizerEvidenceError("variant metadata must contain eight survivors")
    expected: dict[tuple[str, str], dict[str, Any]] = {}
    pairs: dict[str, dict[str, str]] = {label: {} for label in LABELS}
    for survivor_value in survivors:
        if not isinstance(survivor_value, dict):
            raise Q4SanitizerEvidenceError("survivor entry is not an object")
        label = survivor_value.get("lab_label")
        shape = survivor_value.get("shape_id")
        if label not in LABELS or shape not in SHAPES or (label, shape) in expected:
            raise Q4SanitizerEvidenceError("survivor label/shape set is duplicate or unexpected")
        tile = survivor_value.get("tile")
        resources = survivor_value.get("resources")
        if not isinstance(tile, dict) or not isinstance(resources, dict):
            raise Q4SanitizerEvidenceError("survivor tile/resources are missing")
        module_path = resolve_path(survivor_value.get("module"), metadata_path, "module")
        require_hash(module_path, survivor_value.get("module_sha256"), "module")
        record = {
            **survivor_value,
            "block_m": tile.get("rows"),
            "block_n": tile.get("tokens"),
            "threads": survivor_value.get("num_warps") * 32,
            "dynamic_shared_bytes": resources.get("dynamic_shared_bytes"),
            "registers_per_thread": resources.get("registers_per_thread"),
            "module_path": module_path,
        }
        expected[(label, shape)] = record
        pairs[label][shape] = survivor_value.get("config_id")
    if set(expected) != {(label, shape) for label in LABELS for shape in SHAPES}:
        raise Q4SanitizerEvidenceError("variant metadata survivor Cartesian set is incomplete")
    preflight = manifest.get("preflight")
    if not isinstance(preflight, list) or len(preflight) != 8:
        raise Q4SanitizerEvidenceError("sanitizer matrix must bind eight preflight entries")
    seen_preflight: set[tuple[str, str]] = set()
    preflight_output: list[dict[str, Any]] = []
    for entry_value in preflight:
        if not isinstance(entry_value, dict):
            raise Q4SanitizerEvidenceError("preflight entry is not an object")
        key = (entry_value.get("label"), entry_value.get("shape"))
        if key not in expected or key in seen_preflight:
            raise Q4SanitizerEvidenceError("preflight label/shape is duplicate or unexpected")
        seen_preflight.add(key)
        wanted = expected[key]
        module_path = resolve_path(entry_value.get("module"), manifest_path, "preflight.module")
        require_hash(module_path, entry_value.get("module_sha256"), "preflight.module")
        checks = entry_value.get("checks")
        if (
            entry_value.get("symbol") != wanted.get("symbol")
            or module_path != wanted["module_path"]
            or entry_value.get("module_sha256") != wanted.get("module_sha256")
            or entry_value.get("block_m") != wanted["block_m"]
            or entry_value.get("block_n") != wanted["block_n"]
            or entry_value.get("threads") != wanted["threads"]
            or entry_value.get("dynamic_shared_bytes") != wanted["dynamic_shared_bytes"]
            or entry_value.get("registers_per_thread") != wanted["registers_per_thread"]
            or not isinstance(checks, dict)
            or set(checks.values()) != {True}
        ):
            raise Q4SanitizerEvidenceError("preflight differs from variant metadata/resources")
        preflight_output.append(
            {
                "label": key[0],
                "shape_id": key[1],
                "config_id": wanted["config_id"],
                "symbol": wanted["symbol"],
                "module": str(module_path),
                "module_sha256": wanted["module_sha256"],
                "module_bytes": wanted["module_bytes"],
                "n_in": wanted["n_in"],
                "n_out": wanted["n_out"],
                "block_m": wanted["block_m"],
                "block_n": wanted["block_n"],
                "threads": wanted["threads"],
                "dynamic_shared_bytes": wanted["dynamic_shared_bytes"],
                "registers_per_thread": wanted["registers_per_thread"],
            }
        )
    runs = manifest.get("runs")
    if not isinstance(runs, list) or len(runs) != 16:
        raise Q4SanitizerEvidenceError("sanitizer matrix must contain exactly 16 runs")
    seen_runs: set[tuple[str, str]] = set()
    run_output: list[dict[str, Any]] = []
    common_device: dict[str, Any] | None = None
    for run_value in runs:
        if not isinstance(run_value, dict):
            raise Q4SanitizerEvidenceError("sanitizer run is not an object")
        label = run_value.get("label")
        tool = run_value.get("tool")
        key = (label, tool)
        if label not in LABELS or tool not in TOOLS or key in seen_runs:
            raise Q4SanitizerEvidenceError("sanitizer run Cartesian key is duplicate/unexpected")
        seen_runs.add(key)
        if any(
            run_value.get(field) is not expected_value
            for field, expected_value in (
                ("separate_process", True),
                ("summary_zero", True),
                ("harness_json_valid", True),
                ("structural_ok", True),
                ("pass", True),
            )
        ) or run_value.get("exit_code") != 0 or run_value.get("error_exitcode") != 86:
            raise Q4SanitizerEvidenceError(f"{label}/{tool} sanitizer status did not pass")
        log_path = resolve_path(run_value.get("log"), manifest_path, "run.log")
        log_hash = require_hash(log_path, run_value.get("log_sha256"), "run.log")
        log_text = log_path.read_text(encoding="utf-8")
        summary = (
            "========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)"
            if tool == "racecheck"
            else "========= ERROR SUMMARY: 0 errors"
        )
        if log_text.count("========= COMPUTE-SANITIZER") != 1 or log_text.count(summary) != 1:
            raise Q4SanitizerEvidenceError(f"{label}/{tool} zero-error summary mismatch")
        harness_path = resolve_path(run_value.get("harness_output"), manifest_path, "run.harness")
        harness_hash = require_hash(
            harness_path, run_value.get("harness_output_sha256"), "run.harness"
        )
        expected_by_shape = {shape: expected[(label, shape)] for shape in SHAPES}
        device = validate_harness(
            harness_path, label=label, expected_by_shape=expected_by_shape, policy=policy
        )
        if common_device is None:
            common_device = device
        elif device != common_device:
            raise Q4SanitizerEvidenceError("sanitizer harness device identities differ")
        run_output.append(
            {
                "label": label,
                "tool": tool,
                "log": str(log_path),
                "log_sha256": log_hash,
                "harness_output": str(harness_path),
                "harness_output_sha256": harness_hash,
                "exit_code": 0,
                "pass": True,
            }
        )
    if seen_runs != {(label, tool) for label in LABELS for tool in TOOLS}:
        raise Q4SanitizerEvidenceError("sanitizer run Cartesian coverage is incomplete")
    harness_exe = resolve_path(manifest.get("harness_exe"), manifest_path, "harness_exe")
    harness_exe_hash = require_hash(
        harness_exe, manifest.get("harness_exe_sha256"), "harness_exe"
    )
    sanitizer_exe = resolve_path(
        manifest.get("compute_sanitizer"), manifest_path, "compute_sanitizer"
    )
    sanitizer_exe_hash = require_hash(
        sanitizer_exe, manifest.get("compute_sanitizer_sha256"), "compute_sanitizer"
    )
    return {
        "schema": 1,
        "decision": "sanitizer-matrix-evidence",
        "candidate": "q4_0_x_q8_1_mmq_epilogue0",
        "production_authority": False,
        "final_gate_a_decision": False,
        "target_sm": 86,
        "all_pass": True,
        "labels": list(LABELS),
        "tools": list(TOOLS),
        "pair_count": 4,
        "preflight_count": 8,
        "run_count": 16,
        "pairs": pairs,
        "device": common_device,
        "preflight": sorted(preflight_output, key=lambda item: (item["label"], item["shape_id"])),
        "runs": sorted(run_output, key=lambda item: (item["label"], item["tool"])),
        "provenance": {
            "policy": str(policy_path.resolve()),
            "policy_sha256": sha256_file(policy_path),
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": sha256_file(manifest_path),
            "metadata": str(metadata_path),
            "metadata_sha256": manifest["metadata_sha256"],
            "harness_exe": str(harness_exe),
            "harness_exe_sha256": harness_exe_hash,
            "compute_sanitizer": str(sanitizer_exe),
            "compute_sanitizer_sha256": sanitizer_exe_hash,
            "compute_sanitizer_version": manifest.get("compute_sanitizer_version"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output in {args.manifest.resolve(), args.policy.resolve()}:
        print("refusing to overwrite raw evidence", file=sys.stderr)
        return 2
    try:
        binding = validate_matrix(args.manifest.resolve(), args.policy.resolve())
    except (Q4SanitizerEvidenceError, OSError, KeyError, TypeError) as error:
        print(f"Q4 sanitizer finalization rejected: {error}", file=sys.stderr)
        return 2
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(binding, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(binding, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
