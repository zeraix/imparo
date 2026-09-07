#!/usr/bin/env python3
"""Independent fail-closed audit of Q4 Gate-A raw evidence (no GPU work)."""

from __future__ import annotations

import math
import re
import statistics
import tomllib
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

try:
    from dev_harness.finalize_q4_sanitizer_evidence import (
        LABELS, TOOLS, SHAPES, load_json, resolve_path, sha256_file,
    )
    from dev_harness.validate_q4_sanitizer_schema2 import (
        Q4SanitizerSchema2Error, validate_schema2_matrix,
    )
except ModuleNotFoundError:
    from finalize_q4_sanitizer_evidence import (  # type: ignore[no-redef]
        LABELS, TOOLS, SHAPES, load_json, resolve_path, sha256_file,
    )
    from validate_q4_sanitizer_schema2 import (  # type: ignore[no-redef]
        Q4SanitizerSchema2Error, validate_schema2_matrix,
    )


PREFLIGHT_CHECKS = {
    "sm86", "symbol", "kparam_bytes", "kparam_count", "reqntid",
    "registers", "frame_zero",
}
DEVICE_KEYS = {
    "uuid", "name", "sm", "sm_count", "driver_version", "cuda_runtime_version",
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


class Q4StrictAuditError(ValueError):
    pass


def validate_reviewed_floors(
    policy_path: Path, expected_min_kernel_speedup: str,
    expected_min_projected_e2e: str,
) -> dict[str, str]:
    text = policy_path.read_text(encoding="utf-8")
    result: dict[str, str] = {}
    for key, expected in (
        ("min_kernel_speedup_ratio", expected_min_kernel_speedup),
        ("min_projected_e2e_improvement", expected_min_projected_e2e),
    ):
        match = re.search(rf"(?m)^{key}\s*=\s*([0-9]+\.[0-9]+)\s*$", text)
        try:
            valid_decimal = Decimal(expected) > 0
        except (InvalidOperation, TypeError):
            valid_decimal = False
        if match is None or not valid_decimal or match.group(1) != expected:
            raise Q4StrictAuditError(
                f"policy {key} differs from reviewed Decision-A profile"
            )
        result[key] = expected
    return result


def _expect_hash(path: Path, expected: str, field: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise Q4StrictAuditError(f"{field} hash mismatch: expected {expected}, observed {actual}")
    return actual


def validate_device(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != DEVICE_KEYS:
        raise Q4StrictAuditError(f"{field} device key set is incomplete/unexpected")
    if (
        not isinstance(value["uuid"], str) or not value["uuid"]
        or not isinstance(value["name"], str) or not value["name"]
        or value["sm"] != 86 or value["sm_count"] != 30
        or isinstance(value["driver_version"], bool)
        or not isinstance(value["driver_version"], int) or value["driver_version"] <= 0
        or isinstance(value["cuda_runtime_version"], bool)
        or not isinstance(value["cuda_runtime_version"], int)
        or value["cuda_runtime_version"] <= 0
    ):
        raise Q4StrictAuditError(f"{field} device identity is invalid")
    return value


def _close(observed: object, recomputed: float, field: str, *, atol: float) -> None:
    if isinstance(observed, bool) or not isinstance(observed, (int, float)):
        raise Q4StrictAuditError(f"{field} is not numeric")
    if not math.isfinite(float(observed)) or not math.isclose(
        float(observed), recomputed, rel_tol=1e-9, abs_tol=atol
    ):
        raise Q4StrictAuditError(
            f"{field} differs from samples: reported {observed}, recomputed {recomputed}"
        )


def _timing(case: dict[str, Any], noise: dict[str, Any], field: str) -> None:
    timing = case.get("timing")
    if not isinstance(timing, dict):
        raise Q4StrictAuditError(f"{field} timing is missing")
    native = timing.get("native_samples_us")
    triton = timing.get("triton_samples_us")
    if not isinstance(native, list) or not isinstance(triton, list) or len(native) != 100 or len(triton) != 100:
        raise Q4StrictAuditError(f"{field} must bind exactly 100 samples per route")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or float(v) <= 0 for v in native + triton):
        raise Q4StrictAuditError(f"{field} timing samples are invalid")
    native_f = [float(v) for v in native]
    triton_f = [float(v) for v in triton]
    nmean = sum(native_f) / len(native_f)
    tmean = sum(triton_f) / len(triton_f)
    ncv = statistics.stdev(native_f) / nmean
    tcv = statistics.stdev(triton_f) / tmean
    ratios = [n / t for n, t in zip(native_f, triton_f)]
    center = statistics.median(ratios)
    mad = statistics.median([abs(v - center) for v in ratios]) / center
    _close(timing.get("native_median_us"), statistics.median(native_f), f"{field}.native_median", atol=2e-4)
    _close(timing.get("triton_median_us"), statistics.median(triton_f), f"{field}.triton_median", atol=2e-4)
    _close(timing.get("native_mean_us"), nmean, f"{field}.native_mean", atol=2e-4)
    _close(timing.get("triton_mean_us"), tmean, f"{field}.triton_mean", atol=2e-4)
    _close(timing.get("native_cv"), ncv, f"{field}.native_cv", atol=1e-7)
    _close(timing.get("triton_cv"), tcv, f"{field}.triton_cv", atol=1e-7)
    _close(timing.get("paired_mad_fraction"), mad, f"{field}.paired_mad_fraction", atol=1e-7)
    nmedian = statistics.median(native_f)
    tmedian = statistics.median(triton_f)
    # The harness divides the pre-serialization float medians; JSON samples are
    # rounded to 10 significant digits, so retain a tight 5e-8 serialization band.
    _close(timing.get("median_speedup_native_over_triton"), nmedian / tmedian, f"{field}.speedup", atol=5e-8)
    if ncv > float(noise["max_native_cv"]) or tcv > float(noise["max_candidate_cv"]) or mad > float(noise["max_paired_mad_fraction"]):
        raise Q4StrictAuditError(f"{field} recomputed noise gate failed")
    if float(timing.get("native_first_launch_us", 0)) <= 0 or float(timing.get("triton_first_launch_us", 0)) <= 0:
        raise Q4StrictAuditError(f"{field} cold-launch timing is missing")


def _function(value: object, field: str, resources: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != FUNCTION_KEYS:
        raise Q4StrictAuditError(f"{field} function resource key set mismatch")
    if any(isinstance(value[k], bool) or not isinstance(value[k], int) or value[k] < 0 for k in FUNCTION_KEYS):
        raise Q4StrictAuditError(f"{field} function resources are invalid")
    if (
        value["binary_version"] != 86
        or value["local_bytes"] > int(resources["max_local_memory_bytes"])
        or value["registers_per_thread"] > int(resources["max_registers_per_thread"])
        or value["static_shared_bytes"] > int(resources["max_static_shared_bytes"])
        or value["launch_dynamic_shared_bytes"] > int(resources["max_dynamic_shared_bytes"])
        or value["launch_threads"] > value["max_threads_per_block"]
        or value["launch_dynamic_shared_bytes"] > value["max_dynamic_shared_bytes"]
    ):
        raise Q4StrictAuditError(f"{field} function resource limit failed")
    return value


def _resources(case: dict[str, Any], expected: dict[str, Any], policy: dict[str, Any], field: str) -> None:
    value = case.get("artifact_resources")
    if not isinstance(value, dict) or set(value) != RESOURCE_KEYS:
        raise Q4StrictAuditError(f"{field} artifact resource key set mismatch")
    integer_fields = RESOURCE_KEYS - {"module_load_wall_us", "native_function", "triton_function"}
    if any(isinstance(value[k], bool) or not isinstance(value[k], int) or value[k] < 0 for k in integer_fields):
        raise Q4StrictAuditError(f"{field} VRAM/resource values are invalid")
    if not isinstance(value["module_load_wall_us"], (int, float)) or float(value["module_load_wall_us"]) <= 0:
        raise Q4StrictAuditError(f"{field} module-load wall time is invalid")
    total = value["cuda_mem_total_bytes"]
    free = [value[k] for k in (
        "cuda_mem_free_before_buffers", "cuda_mem_free_after_buffers",
        "cuda_mem_free_before_module", "cuda_mem_free_after_module",
        "cuda_mem_free_after_first_launches", "cuda_mem_free_after_timing",
    )]
    if total <= 0 or any(v <= 0 or v > total for v in free):
        raise Q4StrictAuditError(f"{field} VRAM totals/free bytes are invalid")
    before, after_buf, before_mod, after_mod, after_first, after_time = free
    expected_deltas = {
        "observed_buffer_delta_bytes": max(0, before - after_buf),
        "observed_module_delta_bytes": max(0, before_mod - after_mod),
        "observed_first_launch_delta_bytes": max(0, after_mod - after_first),
        "observed_timing_delta_bytes": max(0, after_first - after_time),
        "observed_peak_delta_bytes": max(0, before - min(after_buf, after_mod, after_first, after_time)),
    }
    if any(value[k] != v for k, v in expected_deltas.items()):
        raise Q4StrictAuditError(f"{field} VRAM delta formula mismatch")
    n_in, n_out, n_tok = expected["n_in"], expected["n_out"], 512
    logical = {
        "weights_logical_bytes": n_out * (n_in // 32) * 18,
        "q8_logical_bytes": n_tok * (n_in // 32) * 36,
        "output_logical_bytes_per_buffer": n_out * n_tok * 4,
        "output_buffer_count": 2,
        "workspace_logical_bytes": int(policy["q4_mmq"]["q4_workspace_bytes"]),
    }
    if any(value[k] != v for k, v in logical.items()):
        raise Q4StrictAuditError(f"{field} logical resource formula mismatch")
    guarded = logical["weights_logical_bytes"] + logical["q8_logical_bytes"] + 2 * logical["output_logical_bytes_per_buffer"] + logical["workspace_logical_bytes"] + 5 * 2 * 256
    if value["guarded_allocation_bytes"] != guarded or value["cubin_bytes"] != expected["module_bytes"]:
        raise Q4StrictAuditError(f"{field} guarded allocation/cubin formula mismatch")
    native = _function(value["native_function"], f"{field}.native", policy["resources"])
    triton = _function(value["triton_function"], f"{field}.triton", policy["resources"])
    if (
        native["launch_threads"] != policy["q4_mmq"]["q4_native_launch_threads"]
        or native["launch_dynamic_shared_bytes"] != policy["q4_mmq"]["q4_native_dynamic_shared_bytes"]
        or native["block_m"] != 0 or native["block_n"] != 0
        or triton["registers_per_thread"] != expected["registers_per_thread"]
        or triton["launch_threads"] != expected["threads"]
        or triton["launch_dynamic_shared_bytes"] != expected["dynamic_shared_bytes"]
        or triton["block_m"] != expected["block_m"] or triton["block_n"] != expected["block_n"]
    ):
        raise Q4StrictAuditError(f"{field} native/Triton resource identity mismatch")


def _wrong_grid(case: dict[str, Any], expected: dict[str, Any], field: str) -> None:
    oracle = case.get("cpu_oracle")
    if not isinstance(oracle, dict):
        raise Q4StrictAuditError(f"{field} oracle is missing")
    if expected["shape_id"] != "k10240-m2560":
        return
    actual = oracle.get("wrong_grid_gpu_vs_correct")
    native = oracle.get("wrong_grid_gpu_vs_native")
    if (
        not isinstance(actual, dict)
        or actual.get("non_finite") != 0
        or actual.get("finite") != expected["n_out"] * 512
        or not isinstance(actual.get("max_abs"), (int, float))
        or not math.isfinite(float(actual["max_abs"])) or float(actual["max_abs"]) <= 0
        or not isinstance(actual.get("bitwise_different"), int)
        or actual["bitwise_different"] <= 0
        or oracle.get("wrong_grid_gpu_launched") is not True
        or any(oracle.get(k) != 0 for k in (
            "wrong_grid_gpu_canary_errors", "wrong_grid_gpu_input_mismatches",
            "wrong_grid_gpu_padding_errors",
        ))
        or not isinstance(native, dict)
        or native.get("non_finite") != 0
        or native.get("finite") != expected["n_out"] * 512
        or not isinstance(native.get("max_abs"), (int, float))
        or not math.isfinite(float(native["max_abs"])) or float(native["max_abs"]) <= 0
        or not isinstance(native.get("bitwise_different"), int)
        or native["bitwise_different"] <= 0
    ):
        raise Q4StrictAuditError(f"{field} actual GPU wrong-grid comparison/safety failed")


def audit(
    *, policy_path: Path, expected_policy_sha256: str,
    formal_matrix_path: Path, expected_formal_matrix_sha256: str,
    sanitizer_manifest_path: Path, expected_sanitizer_manifest_sha256: str,
    metadata_path: Path, expected_metadata_sha256: str,
    harness_source_path: Path, expected_harness_source_sha256: str,
    environment_witness_path: Path, expected_environment_witness_sha256: str,
    expected_min_kernel_speedup: str,
    expected_min_projected_e2e: str,
) -> dict[str, Any]:
    _expect_hash(policy_path, expected_policy_sha256, "policy")
    _expect_hash(formal_matrix_path, expected_formal_matrix_sha256, "formal matrix")
    _expect_hash(sanitizer_manifest_path, expected_sanitizer_manifest_sha256, "sanitizer manifest")
    _expect_hash(metadata_path, expected_metadata_sha256, "metadata")
    _expect_hash(harness_source_path, expected_harness_source_sha256, "harness source")
    _expect_hash(environment_witness_path, expected_environment_witness_sha256, "environment witness")
    policy = tomllib.loads(policy_path.read_text(encoding="utf-8"))
    if policy.get("schema") != 1 or policy.get("phase") != "A2" or policy.get("decision") != "gate-a" or policy.get("production_authority") is not False or policy.get("target_sm") != 86:
        raise Q4StrictAuditError("policy identity is invalid")
    for section in ("measurement", "noise", "gate", "resources", "q4_mmq"):
        if not isinstance(policy.get(section), dict):
            raise Q4StrictAuditError(f"policy {section} section is missing")
    reviewed_profile = validate_reviewed_floors(
        policy_path, expected_min_kernel_speedup, expected_min_projected_e2e
    )
    metadata = load_json(metadata_path)
    manifest = load_json(sanitizer_manifest_path)
    formal = load_json(formal_matrix_path)
    sanitizer_schema = manifest.get("schema")
    if sanitizer_schema == 2:
        manifest_identity = manifest.get("identity")
        manifest_metadata_sha256 = (
            manifest_identity.get("metadata_sha256")
            if isinstance(manifest_identity, dict) else None
        )
        manifest_policy_sha256 = (
            manifest_identity.get("policy_sha256")
            if isinstance(manifest_identity, dict) else None
        )
    else:
        manifest_metadata_sha256 = manifest.get("metadata_sha256")
        manifest_policy_sha256 = manifest.get("policy_sha256")
    if manifest_metadata_sha256 != expected_metadata_sha256 or formal.get("metadata_sha256") != expected_metadata_sha256:
        raise Q4StrictAuditError("raw evidence metadata authority differs")
    if manifest_policy_sha256 not in (None, expected_policy_sha256) or formal.get("policy_sha256") != expected_policy_sha256:
        raise Q4StrictAuditError("raw evidence policy authority differs")
    survivors = metadata.get("survivors")
    if not isinstance(survivors, list) or len(survivors) != 8:
        raise Q4StrictAuditError("metadata survivor set is incomplete")
    expected: dict[tuple[str, str], dict[str, Any]] = {}
    for item in survivors:
        tile, resources = item.get("tile"), item.get("resources")
        if not isinstance(tile, dict) or not isinstance(resources, dict):
            raise Q4StrictAuditError("metadata tile/resources missing")
        expected[(item["lab_label"], item["shape_id"])] = {
            **item, "block_m": tile["rows"], "block_n": tile["tokens"],
            "threads": item["num_warps"] * 32,
            "dynamic_shared_bytes": resources["dynamic_shared_bytes"],
            "registers_per_thread": resources["registers_per_thread"],
        }
    if set(expected) != {(label, shape) for label in LABELS for shape in SHAPES}:
        raise Q4StrictAuditError("metadata survivor Cartesian set differs")
    expansion_ids = {
        item["config_id"] for item in survivors if item["shape_id"] == "k2560-m10240"
    }
    contraction_ids = {
        item["config_id"] for item in survivors if item["shape_id"] == "k10240-m2560"
    }
    if (
        expansion_ids != set(policy["q4_mmq"]["q4_variant_allowed_expansion"])
        or contraction_ids != set(policy["q4_mmq"]["q4_variant_allowed_contraction"])
    ):
        raise Q4StrictAuditError("metadata config IDs differ from policy allowlists")
    builder = metadata.get("builder")
    q4 = policy["q4_mmq"]
    if (
        not isinstance(builder, dict)
        or builder.get("image_id") != q4["q4_builder_image_id"]
        or builder.get("python") != q4["q4_builder_python"]
        or builder.get("torch") != q4["q4_builder_torch"]
        or not isinstance(builder.get("triton"), str)
        or builder.get("triton") != f"{q4['q4_builder_triton']}-source-pin"
    ):
        raise Q4StrictAuditError("metadata builder differs from policy pins")
    sanitizer_binding: dict[str, Any] | None = None
    common_device: dict[str, Any] | None = None
    if sanitizer_schema == 2:
        try:
            sanitizer_binding = validate_schema2_matrix(
                sanitizer_manifest_path,
                policy_path,
                expected_metadata_sha256=expected_metadata_sha256,
                expected_harness_source_sha256=expected_harness_source_sha256,
            )
        except Q4SanitizerSchema2Error as error:
            raise Q4StrictAuditError(f"schema2 sanitizer rejected: {error}") from error
        common_device = validate_device(
            sanitizer_binding.get("device"), "schema2 sanitizer"
        )
    else:
        for entry in manifest.get("preflight", []):
            checks = entry.get("checks")
            if not isinstance(checks, dict) or set(checks) != PREFLIGHT_CHECKS or any(value is not True for value in checks.values()):
                raise Q4StrictAuditError("preflight exact check set/all-true requirement failed")
        if len(manifest.get("preflight", [])) != 8:
            raise Q4StrictAuditError("preflight count differs")
        harness_hashes: dict[str, set[str]] = {label: set() for label in LABELS}
        run_keys: set[tuple[str, str]] = set()
        for run in manifest.get("runs", []):
            label, tool = run.get("label"), run.get("tool")
            if label not in LABELS or tool not in TOOLS or (label, tool) in run_keys:
                raise Q4StrictAuditError("sanitizer Cartesian tool identity differs")
            run_keys.add((label, tool))
            log = resolve_path(run.get("log"), sanitizer_manifest_path, "sanitizer log")
            harness = resolve_path(run.get("harness_output"), sanitizer_manifest_path, "sanitizer harness")
            if log.name != f"compute-sanitizer-{tool}.log" or harness.name != f"{tool}-harness.json":
                raise Q4StrictAuditError("sanitizer tool-specific path identity differs")
            harness_hashes[label].add(str(run.get("harness_output_sha256")))
            text = log.read_text(encoding="utf-8")
            if text.count("========= COMPUTE-SANITIZER") != 1:
                raise Q4StrictAuditError("sanitizer header count differs")
            if tool == "memcheck":
                ok = text.count("========= LEAK SUMMARY: 0 bytes leaked in 0 allocations") == 1 and text.count("========= ERROR SUMMARY: 0 errors") == 1
            elif tool == "racecheck":
                ok = text.count("========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)") == 1 and "LEAK SUMMARY" not in text
            else:
                ok = text.count("========= ERROR SUMMARY: 0 errors") == 1 and "LEAK SUMMARY" not in text and "RACECHECK SUMMARY" not in text
            if not ok:
                raise Q4StrictAuditError(f"{label}/{tool} tool-specific summary differs")
            device = validate_device(load_json(harness).get("device"), f"{label}/{tool}")
            if common_device is None:
                common_device = device
            elif common_device != device:
                raise Q4StrictAuditError("sanitizer device identities differ")
        if run_keys != {(label, tool) for label in LABELS for tool in TOOLS} or any(len(value) != 4 for value in harness_hashes.values()):
            raise Q4StrictAuditError("sanitizer exact 16 Cartesian/unique harness requirement failed")
    formal_labels: set[str] = set()
    for run in formal.get("runs", []):
        label = run.get("label")
        if label not in LABELS or label in formal_labels:
            raise Q4StrictAuditError("formal label set differs")
        formal_labels.add(label)
        result = load_json(resolve_path(run.get("directory"), formal_matrix_path, "formal directory") / "stdout.json")
        device = validate_device(result.get("device"), f"formal/{label}")
        if device != common_device:
            raise Q4StrictAuditError("formal/sanitizer device identity differs")
        for case in result.get("cases", []):
            if case.get("n_tok") != 512:
                continue
            shape = case.get("shape_id")
            item = expected[(label, shape)]
            _timing(case, policy["noise"], f"{label}/{shape}")
            _resources(case, item, policy, f"{label}/{shape}")
            _wrong_grid(case, item, f"{label}/{shape}")
    if formal_labels != set(LABELS):
        raise Q4StrictAuditError("formal pair set differs")
    witness = load_json(environment_witness_path)
    expected_witness = {
        "schema": 1, "phase": "A2", "milestone": "bounded-variant-current-environment",
        "production_enabled": False, "target_sm": 86,
        "policy_sha256": expected_policy_sha256,
        "formal_matrix_sha256": expected_formal_matrix_sha256,
        "sanitizer_manifest_sha256": expected_sanitizer_manifest_sha256,
        "metadata_sha256": expected_metadata_sha256,
        "harness_source_sha256": expected_harness_source_sha256,
        "harness_exe_sha256": (
            sanitizer_binding["provenance"]["harness_exe_sha256"]
            if sanitizer_binding is not None else manifest.get("harness_exe_sha256")
        ),
    }
    if any(witness.get(k) != v for k, v in expected_witness.items()):
        raise Q4StrictAuditError("current environment witness authority differs")
    if validate_device(witness.get("device"), "environment witness") != common_device:
        raise Q4StrictAuditError("environment witness device differs")
    if witness.get("builder") != builder:
        raise Q4StrictAuditError("environment witness builder differs")
    expected_builder_provenance = {
        "triton_commit": q4["q4_builder_triton_commit"],
        "ptxas": q4["q4_builder_ptxas"],
        "cuobjdump": q4["q4_builder_cuobjdump"],
        "recipe_sha256": q4["q4_builder_recipe_sha256"],
        "linux_amd64_manifest": q4["q4_builder_linux_amd64_manifest"],
        "oci_config": q4["q4_builder_oci_config"],
        "verification_scope": "declaration-only-external-roots",
    }
    if witness.get("builder_provenance") != expected_builder_provenance:
        raise Q4StrictAuditError("environment witness full builder authority differs")
    expected_gate_authority = {
        "min_kernel_speedup_ratio": policy["gate"]["min_kernel_speedup_ratio"],
        "min_projected_e2e_improvement": policy["gate"]["min_projected_e2e_improvement"],
    }
    if witness.get("policy_gate_authority") != expected_gate_authority:
        raise Q4StrictAuditError("environment witness policy gate authority differs")
    if sanitizer_binding is not None:
        provenance = sanitizer_binding["provenance"]
        expected_sanitizer_authority = {
            "schema": 2,
            "runner_sha256": provenance["runner_sha256"],
            "compute_sanitizer_sha256": provenance["compute_sanitizer"]["sha256"],
            "compute_sanitizer_version": provenance["compute_sanitizer"]["version"],
            "cuobjdump_sha256": provenance["cuobjdump"]["sha256"],
            "cuobjdump_version": provenance["cuobjdump"]["version"],
            "nvidia_smi_sha256": provenance["nvidia_smi"]["sha256"],
            "nvidia_smi_version": provenance["nvidia_smi"]["version"],
        }
        if witness.get("sanitizer_authority") != expected_sanitizer_authority:
            raise Q4StrictAuditError("environment witness schema2 sanitizer authority differs")
    toolchain = witness.get("local_toolchain")
    required_toolchain = {
        "backend_abi", "target_sm", "nvcc", "nvcc_path", "nvcc_sha256",
        "cuobjdump", "cuobjdump_path", "cuobjdump_sha256",
        "compute_sanitizer_sha256", "compute_sanitizer_version",
    }
    expected_compute_sha = (
        sanitizer_binding["provenance"]["compute_sanitizer"]["sha256"]
        if sanitizer_binding is not None else manifest.get("compute_sanitizer_sha256")
    )
    expected_compute_version = (
        sanitizer_binding["provenance"]["compute_sanitizer"]["version"]
        if sanitizer_binding is not None else manifest.get("compute_sanitizer_version")
    )
    if not isinstance(toolchain, dict) or set(toolchain) != required_toolchain or toolchain.get("backend_abi") != 26 or toolchain.get("target_sm") != 86 or toolchain.get("compute_sanitizer_sha256") != expected_compute_sha or toolchain.get("compute_sanitizer_version") != expected_compute_version:
        raise Q4StrictAuditError("current environment witness toolchain differs")
    nvcc_path = Path(toolchain["nvcc_path"]).resolve()
    cuobjdump_path = Path(toolchain["cuobjdump_path"]).resolve()
    if (
        not nvcc_path.is_file() or not cuobjdump_path.is_file()
        or sha256_file(nvcc_path) != toolchain["nvcc_sha256"]
        or sha256_file(cuobjdump_path) != toolchain["cuobjdump_sha256"]
        or q4["q4_builder_ptxas"] not in toolchain["nvcc"]
        or q4["q4_builder_cuobjdump"] not in toolchain["cuobjdump"]
        or re.fullmatch(r"Cuda compilation tools, release 12\.9, V12\.9\.86; Build cuda_12\.9\.r12\.9/compiler\.36037853_0", toolchain["nvcc"]) is None
        or re.fullmatch(r"Cuda compilation tools, release 12\.9, V12\.9\.82; Build cuda_12\.9\.r12\.9/compiler\.36001619_0", toolchain["cuobjdump"]) is None
    ):
        raise Q4StrictAuditError("current environment witness tool binary/version differs")
    if sanitizer_binding is not None:
        cuobjdump_identity = sanitizer_binding["provenance"]["cuobjdump"]
        cuobjdump_version_compact = re.sub(
            r"\r?\nBuild ", "; Build ", cuobjdump_identity["version"]
        )
        if (
            str(cuobjdump_path) != cuobjdump_identity["path"]
            or toolchain["cuobjdump_sha256"] != cuobjdump_identity["sha256"]
            or toolchain["cuobjdump"] not in cuobjdump_version_compact
        ):
            raise Q4StrictAuditError("schema2 cuobjdump differs from current witness")
    return {
        "schema": 1,
        "policy_sha256": expected_policy_sha256,
        "formal_matrix_sha256": expected_formal_matrix_sha256,
        "sanitizer_manifest_sha256": expected_sanitizer_manifest_sha256,
        "metadata_sha256": expected_metadata_sha256,
        "harness_source_sha256": expected_harness_source_sha256,
        "environment_witness_sha256": expected_environment_witness_sha256,
        "device": common_device,
        "builder": metadata.get("builder"),
        "local_toolchain": toolchain,
        "reviewed_decision_profile": reviewed_profile,
        "sanitizer": sanitizer_binding,
    }
