#!/usr/bin/env python3
"""Finalize existing RMS->Q8 Gate-A evidence without executing GPU work."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

SANITIZERS = ("memcheck", "initcheck", "racecheck", "synccheck")


class RmsEvidenceError(ValueError):
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
                raise RmsEvidenceError(f"duplicate JSON key {key!r} in {path}")
            value[key] = item
        return value

    try:
        result = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicates)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RmsEvidenceError(f"read JSON {path}: {error}") from error
    if not isinstance(result, dict):
        raise RmsEvidenceError(f"JSON root is not an object: {path}")
    return result


def require_dict(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RmsEvidenceError(f"{field} must be an object")
    return value


def require_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RmsEvidenceError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise RmsEvidenceError(f"{field} must be finite")
    return number


def resolve_record_path(raw: object, owner: Path, field: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise RmsEvidenceError(f"{field} path is missing")
    path = Path(raw)
    if not path.is_absolute():
        path = owner.parent / path
    return path.resolve()


def add(violations: list[str], condition: bool, message: str) -> None:
    if not condition:
        violations.append(message)


def file_evidence(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RmsEvidenceError(f"evidence file is missing: {path}")
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def validate_metadata(
    metadata_path: Path, triton_source_path: Path, target_sm: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = load_json(metadata_path)
    if (
        metadata.get("schema") != 1
        or metadata.get("phase") != "A2"
        or metadata.get("candidate") != "rms_norm_q8_1_mmq"
        or metadata.get("production_enabled") is not False
        or metadata.get("target") != f"cuda:{target_sm}:32"
        or metadata.get("shape") != {"n_tok": 512, "width": 2560}
    ):
        raise RmsEvidenceError("RMS metadata identity/scope mismatch")
    source = require_dict(metadata.get("source"), "metadata.source")
    if source.get("sha256") != sha256_file(triton_source_path):
        raise RmsEvidenceError("Triton source hash differs from metadata")
    if Path(str(source.get("path", ""))).name != triton_source_path.name:
        raise RmsEvidenceError("Triton source name differs from metadata")
    variants = metadata.get("variants")
    if not isinstance(variants, list) or len(variants) != 2:
        raise RmsEvidenceError("RMS metadata must contain exactly two variants")
    modules: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_symbols: set[str] = set()
    metadata_root = metadata_path.parent.resolve()
    for index, variant_value in enumerate(variants):
        variant = require_dict(variant_value, f"metadata.variants[{index}]")
        variant_id = variant.get("variant_id")
        symbol = variant.get("symbol")
        if variant_id not in {"w4-s1", "w8-s1"} or variant_id in seen_ids:
            raise RmsEvidenceError("RMS metadata variant IDs are not exact w4/w8")
        if not isinstance(symbol, str) or not symbol or symbol in seen_symbols:
            raise RmsEvidenceError("RMS metadata symbols are missing or duplicate")
        seen_ids.add(variant_id)
        seen_symbols.add(symbol)
        module_path = resolve_record_path(variant.get("module"), metadata_path, "variant.module")
        try:
            module_path.relative_to(metadata_root)
        except ValueError as error:
            raise RmsEvidenceError("module path escapes metadata tree") from error
        module = file_evidence(module_path)
        if module["sha256"] != variant.get("module_sha256"):
            raise RmsEvidenceError(f"module hash mismatch for {variant_id}")
        if module["bytes"] != variant.get("module_bytes"):
            raise RmsEvidenceError(f"module size mismatch for {variant_id}")
        resources = require_dict(variant.get("resources"), f"{variant_id}.resources")
        if any(resources.get(key) != 0 for key in (
            "global_scratch_bytes", "profile_scratch_bytes", "local_memory_bytes"
        )):
            raise RmsEvidenceError(f"{variant_id} declares scratch or local memory")
        modules.append(
            {
                "variant_id": variant_id,
                "symbol": symbol,
                "num_warps": variant.get("num_warps"),
                "dynamic_shared_bytes": variant.get("dynamic_shared_bytes"),
                **module,
            }
        )
    if seen_ids != {"w4-s1", "w8-s1"}:
        raise RmsEvidenceError("RMS metadata variant set is incomplete")
    return metadata, sorted(modules, key=lambda item: item["variant_id"])


def validate_result(
    result: dict[str, Any],
    policy: dict[str, Any],
    metadata: dict[str, Any],
    *,
    formal: bool,
    prefix: str,
) -> tuple[bool, bool, bool, list[str]]:
    violations: list[str] = []
    workload = policy["workload"]
    measurement = policy["measurement"]
    correctness = policy["correctness"]
    noise = policy["noise"]
    resources_policy = policy["resources"]
    gate = policy["gate"]
    add(violations, result.get("schema") == 1, f"{prefix}: schema mismatch")
    add(violations, result.get("phase") == "A2", f"{prefix}: phase mismatch")
    add(violations, result.get("production_enabled") is False, f"{prefix}: production enabled")
    add(violations, result.get("target_sm") == policy["target_sm"], f"{prefix}: SM mismatch")
    add(
        violations,
        result.get("width") == workload["width"] and result.get("n_tok") == workload["n_tok"],
        f"{prefix}: workload shape mismatch",
    )
    add(violations, result.get("same_primary_context") is True, f"{prefix}: context differs")
    add(violations, result.get("same_stream") is True, f"{prefix}: stream differs")
    add(violations, result.get("separate_outputs") is True, f"{prefix}: outputs alias")
    add(violations, result.get("structural_ok") is True, f"{prefix}: structural checks failed")
    add(
        violations,
        result.get("all_canary_errors") == correctness["max_canary_errors"],
        f"{prefix}: canary corruption",
    )
    variants_value = result.get("variants")
    variants = variants_value if isinstance(variants_value, list) else []
    add(violations, len(variants) == 2, f"{prefix}: expected exactly two variants")
    metadata_by_symbol = {item["symbol"]: item for item in metadata["variants"]}
    seen_symbols: set[str] = set()
    measurement_violations: list[str] = []
    fast = False
    for index, variant_value in enumerate(variants):
        if not isinstance(variant_value, dict):
            violations.append(f"{prefix}: variant {index} is not an object")
            continue
        symbol = variant_value.get("symbol")
        expected = metadata_by_symbol.get(symbol)
        if expected is None or symbol in seen_symbols:
            violations.append(f"{prefix}: result variant symbol is unexpected or duplicate")
            continue
        seen_symbols.add(symbol)
        label = f"{prefix}/{expected['variant_id']}"
        add(violations, variant_value.get("warps") == expected["num_warps"], f"{label}: warp mismatch")
        add(
            violations,
            variant_value.get("dynamic_shared_bytes") == expected["dynamic_shared_bytes"],
            f"{label}: dynamic shared mismatch",
        )
        add(violations, variant_value.get("cubin_bytes") == expected["module_bytes"], f"{label}: cubin size mismatch")
        normalized = require_dict(variant_value.get("normalized_vs_native"), f"{label}.normalized")
        scales = require_dict(variant_value.get("q8_scales_vs_native"), f"{label}.scales")
        dequant = require_dict(variant_value.get("q8_dequant"), f"{label}.dequant")
        resources = require_dict(variant_value.get("resources"), f"{label}.resources")
        add(violations, require_number(normalized.get("max_abs"), f"{label}.max_abs") <= correctness["max_normalized_abs_vs_native"], f"{label}: normalized max_abs")
        add(violations, require_number(normalized.get("max_rel"), f"{label}.max_rel") <= correctness["max_normalized_rel_vs_native"], f"{label}: normalized max_rel")
        add(violations, normalized.get("non_finite") == correctness["max_non_finite"], f"{label}: normalized non-finite")
        add(violations, require_number(scales.get("max_abs"), f"{label}.scale_abs") <= correctness["max_q8_scale_abs_vs_native"], f"{label}: scale max_abs")
        add(violations, require_number(variant_value.get("q8_value_max_abs"), f"{label}.q8_value_max_abs") <= correctness["max_q8_value_abs_vs_native"], f"{label}: Q8 value delta")
        add(violations, require_number(variant_value.get("q8_byte_agreement"), f"{label}.agreement") >= correctness["min_q8_byte_agreement"], f"{label}: Q8 byte agreement")
        add(violations, require_number(dequant.get("max_abs"), f"{label}.dequant_abs") <= correctness["max_q8_dequant_abs"], f"{label}: dequant max_abs")
        add(violations, dequant.get("non_finite") == correctness["max_non_finite"], f"{label}: dequant non-finite")
        add(violations, variant_value.get("canary_errors") == correctness["max_canary_errors"], f"{label}: canary errors")
        add(violations, resources.get("registers_per_thread", 10**9) <= resources_policy["max_registers_per_thread"], f"{label}: register ceiling")
        add(violations, resources.get("static_shared_bytes", 10**9) <= resources_policy["max_static_shared_bytes"], f"{label}: static shared ceiling")
        add(violations, variant_value.get("dynamic_shared_bytes", 10**9) <= resources_policy["max_dynamic_shared_bytes"], f"{label}: dynamic shared ceiling")
        add(violations, resources.get("local_bytes") == resources_policy["max_local_memory_bytes"], f"{label}: local memory")
        timing = require_dict(variant_value.get("timing"), f"{label}.timing")
        speedup = require_number(timing.get("speedup_median"), f"{label}.speedup")
        fast = fast or speedup >= gate["min_kernel_speedup_ratio"]
        if formal:
            native_samples = timing.get("native_samples_us")
            triton_samples = timing.get("triton_samples_us")
            if not isinstance(native_samples, list) or len(native_samples) < measurement["min_samples_per_route"]:
                measurement_violations.append(f"{label}: native sample count")
            if not isinstance(triton_samples, list) or len(triton_samples) < measurement["min_samples_per_route"]:
                measurement_violations.append(f"{label}: Triton sample count")
            if require_number(timing.get("native_cv"), f"{label}.native_cv") > noise["max_native_cv"]:
                measurement_violations.append(f"{label}: native CV")
            if require_number(timing.get("triton_cv"), f"{label}.triton_cv") > noise["max_candidate_cv"]:
                measurement_violations.append(f"{label}: Triton CV")
            if require_number(timing.get("paired_mad_fraction"), f"{label}.mad") > noise["max_paired_mad_fraction"]:
                measurement_violations.append(f"{label}: paired MAD")
    add(violations, seen_symbols == set(metadata_by_symbol), f"{prefix}: metadata/result variants differ")
    correctness_ok = not violations
    if formal:
        if result.get("timing_clock") != "cuda-events":
            measurement_violations.append(f"{prefix}: timing clock is not CUDA events")
        if result.get("interleaved_schedule") != measurement["interleaved"]:
            measurement_violations.append(f"{prefix}: schedule is not ABBA/BAAB")
        if result.get("launches_per_sample") != measurement["launches_per_sample"]:
            measurement_violations.append(f"{prefix}: launches per sample mismatch")
    violations.extend(measurement_violations)
    return correctness_ok, not measurement_violations, fast, violations


def parse_sanitizer_log(path: Path, tool: str) -> tuple[dict[str, Any], dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="strict")
    if text.count("========= COMPUTE-SANITIZER") != 1:
        raise RmsEvidenceError(f"sanitizer banner missing or duplicate: {path}")
    success_summary = (
        "========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)"
        if tool == "racecheck"
        else "========= ERROR SUMMARY: 0 errors"
    )
    if text.count(success_summary) != 1:
        raise RmsEvidenceError(f"sanitizer zero-error summary missing or duplicate: {path}")
    if text.count("kernel-lab-phase: complete") != 1:
        raise RmsEvidenceError(f"kernel-lab completion marker missing or duplicate: {path}")
    json_lines = [line for line in text.splitlines() if line.startswith("{") and line.endswith("}")]
    if len(json_lines) != 1:
        raise RmsEvidenceError(f"sanitizer log must contain exactly one result JSON: {path}")
    try:
        embedded = json.loads(json_lines[0])
    except json.JSONDecodeError as error:
        raise RmsEvidenceError(f"invalid sanitizer result JSON in {path}") from error
    if not isinstance(embedded, dict):
        raise RmsEvidenceError(f"sanitizer result root is not an object: {path}")
    return embedded, file_evidence(path)


def finalize(
    *,
    policy_path: Path,
    metadata_path: Path,
    triton_source_path: Path,
    harness_source_path: Path,
    harness_exe_path: Path,
    result_path: Path,
    legacy_decision_path: Path,
    environment_witness_path: Path,
) -> dict[str, Any]:
    try:
        policy = tomllib.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise RmsEvidenceError(f"read policy: {error}") from error
    if policy.get("schema") != 1 or policy.get("phase") != "A2" or policy.get("target_sm") != 86:
        raise RmsEvidenceError("policy identity mismatch")
    if policy.get("production_authority") is not False:
        raise RmsEvidenceError("policy grants production authority")
    metadata, modules = validate_metadata(metadata_path, triton_source_path, policy["target_sm"])
    result = load_json(result_path)
    correctness_ok, measurement_ok, kernel_floor_met, violations = validate_result(
        result, policy, metadata, formal=True, prefix="formal"
    )
    legacy = load_json(legacy_decision_path)
    if (
        legacy.get("schema") != 1
        or legacy.get("decision") != "gate-a-candidate"
        or legacy.get("production_authority") is not False
        or legacy.get("final_gate_a_decision") is not False
        or legacy.get("candidate_admissible") is not True
        or legacy.get("kernel_floor_met") is not True
        or legacy.get("violations") != []
    ):
        raise RmsEvidenceError("legacy decision is not an admissible non-production candidate")
    if resolve_record_path(legacy.get("result"), legacy_decision_path, "legacy.result") != result_path.resolve():
        raise RmsEvidenceError("legacy decision result path mismatch")
    sanitizer_records = legacy.get("sanitizer")
    if not isinstance(sanitizer_records, dict) or set(sanitizer_records) != set(SANITIZERS):
        raise RmsEvidenceError("legacy decision lacks exact four-tool sanitizer mapping")
    sanitizer_output: dict[str, Any] = {}
    sanitizer_ok = True
    for tool in SANITIZERS:
        record = require_dict(sanitizer_records[tool], f"legacy.sanitizer.{tool}")
        if record.get("pass") is not True or record.get("exit_code") != 0:
            raise RmsEvidenceError(f"legacy {tool} sanitizer status did not pass")
        log_path = resolve_record_path(record.get("path"), legacy_decision_path, f"legacy.{tool}")
        embedded, log_evidence = parse_sanitizer_log(log_path, tool)
        log_correct, _, _, log_violations = validate_result(
            embedded, policy, metadata, formal=False, prefix=f"sanitizer/{tool}"
        )
        if not log_correct or log_violations:
            sanitizer_ok = False
            violations.extend(log_violations)
        sanitizer_output[tool] = {
            "path": log_evidence["path"],
            "sha256": log_evidence["sha256"],
            "bytes": log_evidence["bytes"],
            "exit_code": 0,
            "pass": log_correct and not log_violations,
        }
    witness = load_json(environment_witness_path)
    witness_provenance = require_dict(witness.get("provenance"), "environment witness provenance")
    device = require_dict(witness_provenance.get("device"), "environment witness device")
    toolchain = require_dict(
        witness_provenance.get("local_toolchain"), "environment witness toolchain"
    )
    if (
        device.get("sm") != 86
        or not isinstance(device.get("uuid"), str)
        or not isinstance(device.get("name"), str)
        or not isinstance(device.get("sm_count"), int)
        or not isinstance(device.get("driver_version"), int)
        or not isinstance(device.get("cuda_runtime_version"), int)
    ):
        raise RmsEvidenceError("environment witness device is incomplete or not SM86")
    if toolchain.get("target_sm") != 86 or toolchain.get("backend_abi") != 26:
        raise RmsEvidenceError("environment witness toolchain does not match SM86/ABI26")
    provenance = {
        "policy_sha256": sha256_file(policy_path),
        "metadata_sha256": sha256_file(metadata_path),
        "harness_source_sha256": sha256_file(harness_source_path),
        "harness_exe_sha256": sha256_file(harness_exe_path),
        "result_json_sha256": sha256_file(result_path),
        "triton_source_sha256": sha256_file(triton_source_path),
        "legacy_decision_sha256": sha256_file(legacy_decision_path),
        "environment_witness_sha256": sha256_file(environment_witness_path),
        "cubins": [
            {
                "variant_id": module["variant_id"],
                "symbol": module["symbol"],
                "path": module["path"],
                "sha256": module["sha256"],
                "bytes": module["bytes"],
            }
            for module in modules
        ],
        "device": device,
        "local_toolchain": toolchain,
        "builder": metadata.get("builder"),
        "sources": {
            "metadata": file_evidence(metadata_path),
            "triton": file_evidence(triton_source_path),
            "harness": file_evidence(harness_source_path),
            "harness_exe": file_evidence(harness_exe_path),
            "legacy_decision": file_evidence(legacy_decision_path),
            "environment_witness": file_evidence(environment_witness_path),
        },
    }
    candidate_ok = correctness_ok and measurement_ok and sanitizer_ok and kernel_floor_met and not violations
    return {
        "schema": 1,
        "decision": "gate-a-candidate",
        "candidate": "rms_norm_q8_1_mmq",
        "production_authority": False,
        "final_gate_a_decision": False,
        "correctness_admissible": correctness_ok,
        "measurement_admissible": measurement_ok,
        "sanitizer_admissible": sanitizer_ok,
        "provenance_admissible": True,
        "candidate_admissible": candidate_ok,
        "kernel_floor_met": kernel_floor_met,
        "projected_e2e_improvement": None,
        "projected_e2e_floor_met": False,
        "result": str(result_path.resolve()),
        "sanitizer": sanitizer_output,
        "provenance": provenance,
        "violations": violations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--triton-source", type=Path, required=True)
    parser.add_argument("--harness-source", type=Path, required=True)
    parser.add_argument("--harness-exe", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--legacy-decision", type=Path, required=True)
    parser.add_argument("--environment-witness", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {key: value.resolve() for key, value in vars(args).items() if isinstance(value, Path)}
    output = paths.pop("output")
    if output in paths.values():
        print("refusing to overwrite raw evidence", file=sys.stderr)
        return 2
    try:
        decision = finalize(
            policy_path=paths["policy"],
            metadata_path=paths["metadata"],
            triton_source_path=paths["triton_source"],
            harness_source_path=paths["harness_source"],
            harness_exe_path=paths["harness_exe"],
            result_path=paths["result"],
            legacy_decision_path=paths["legacy_decision"],
            environment_witness_path=paths["environment_witness"],
        )
    except (RmsEvidenceError, OSError, KeyError, TypeError) as error:
        print(f"RMS evidence finalization rejected: {error}", file=sys.stderr)
        return 2
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0 if decision["candidate_admissible"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
