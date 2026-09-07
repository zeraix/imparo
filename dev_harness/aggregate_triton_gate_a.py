#!/usr/bin/env python3
"""Prepare fail-closed Decision-A inputs without issuing a Gate A decision."""

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


HEX64 = re.compile(r"^[0-9a-f]{64}$")
PROFILE_ROW = re.compile(
    r"^\|\s*(2560|10240)\s*->\s*(2560|10240)\s*x\s*512\s*"
    r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([0-9.]+)\s*"
    r"\|\s*([0-9.]+)%\s*\|\s*([0-9.]+)\s*\|$"
)
SHAPES = {"k2560-m10240", "k10240-m2560"}
SANITIZERS = {"memcheck", "initcheck", "racecheck", "synccheck"}


class GateInputError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(value: object, field: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise GateInputError(f"{field} must be a lowercase SHA-256")
    return value


def resolve_bound_path(raw: object, owner: Path, field: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise GateInputError(f"{field} path is missing")
    path = Path(raw)
    if not path.is_absolute():
        path = owner.parent / path
    return path.resolve()


def load_json(path: Path) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise GateInputError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicates)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GateInputError(f"read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise GateInputError(f"JSON root is not an object: {path}")
    return value


def load_bound_json(path: Path, expected_sha256: str, field: str) -> dict[str, Any]:
    expected = require_hash(expected_sha256, field)
    actual = sha256_file(path)
    if actual != expected:
        raise GateInputError(f"{field} hash mismatch: expected {expected}, observed {actual}")
    return load_json(path)


def parse_phase_a1(path: Path, expected_sha256: str) -> dict[str, Any]:
    expected = require_hash(expected_sha256, "phase_a1_sha256")
    actual = sha256_file(path)
    if actual != expected:
        raise GateInputError(
            f"phase_a1_sha256 hash mismatch: expected {expected}, observed {actual}"
        )
    text = path.read_text(encoding="utf-8")
    if re.search(r"measured\s+logical\s+matmul\s+time", text, re.IGNORECASE) is None:
        raise GateInputError("Phase A1 evidence does not declare logical-matmat scope")
    shares: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        match = PROFILE_ROW.match(line.strip())
        if match is None:
            continue
        n_in, n_out, epilogue, calls, total_ms, share_percent, median_us = match.groups()
        if int(epilogue) != 0:
            continue
        shape = f"k{n_in}-m{n_out}"
        if shape not in SHAPES or shape in shares:
            raise GateInputError("Phase A1 epilogue-0 profile is duplicate or unexpected")
        share = float(share_percent) / 100.0
        if not 0.0 < share < 1.0:
            raise GateInputError(f"invalid Phase A1 share for {shape}")
        shares[shape] = share
        rows.append(
            {
                "shape_id": shape,
                "epilogue": 0,
                "calls": int(calls),
                "total_ms": float(total_ms),
                "logical_matmat_share": share,
                "profile_median_us": float(median_us),
            }
        )
    if set(shares) != SHAPES:
        raise GateInputError("Phase A1 evidence must contain exactly both epilogue-0 shapes")
    return {
        "path": str(path.resolve()),
        "sha256": actual,
        "scope": "share_of_measured_logical_matmat_time_not_whole_prefill",
        "rows": sorted(rows, key=lambda item: item["shape_id"]),
        "shares": shares,
    }


def add_check(violations: list[str], condition: bool, message: str) -> None:
    if not condition:
        violations.append(message)


def validate_candidate(
    family: str,
    decision_path: Path,
    decision_sha256: str,
    policy_sha256: str,
) -> dict[str, Any]:
    decision = load_bound_json(decision_path, decision_sha256, f"{family}_decision_sha256")
    violations: list[str] = []
    add_check(violations, decision.get("schema") == 1, "decision schema is not 1")
    add_check(
        violations,
        decision.get("decision") == "gate-a-candidate",
        "not a gate-a-candidate record",
    )
    add_check(
        violations,
        decision.get("production_authority") is False,
        "candidate record claims production authority",
    )
    add_check(
        violations,
        decision.get("final_gate_a_decision") is False,
        "candidate record claims a final Gate A decision",
    )
    expected_candidate = (
        "rms_norm_q8_1_mmq" if family == "rms_q8" else "q4_0_x_q8_1_mmq_epilogue0"
    )
    add_check(
        violations,
        decision.get("candidate") == expected_candidate,
        f"candidate identity is not {expected_candidate}",
    )
    add_check(
        violations,
        decision.get("correctness_admissible") is True,
        "correctness is not admissible",
    )
    add_check(
        violations,
        decision.get("measurement_admissible") is True,
        "formal measurement is not admissible",
    )
    add_check(
        violations,
        decision.get("candidate_admissible") is True,
        "candidate is not admissible",
    )
    add_check(
        violations,
        decision.get("kernel_floor_met") is True,
        "kernel speedup floor is not met",
    )
    add_check(violations, decision.get("violations") == [], "candidate violations are non-empty")

    provenance = decision.get("provenance")
    if not isinstance(provenance, dict):
        violations.append("complete provenance object is missing")
        provenance = {}
    add_check(
        violations,
        provenance.get("policy_sha256") == policy_sha256,
        "candidate policy hash differs from aggregation policy",
    )
    for field in (
        "metadata_sha256",
        "harness_source_sha256",
        "harness_exe_sha256",
        "result_json_sha256",
    ):
        try:
            require_hash(provenance.get(field), f"{family}.provenance.{field}")
        except GateInputError as error:
            violations.append(str(error))
    cubins = provenance.get("cubins")
    if not isinstance(cubins, list) or not cubins:
        violations.append("cubin provenance is missing")
    else:
        for index, cubin in enumerate(cubins):
            try:
                if not isinstance(cubin, dict):
                    raise GateInputError("entry is not an object")
                require_hash(cubin.get("sha256"), f"{family}.cubins[{index}].sha256")
                if not isinstance(cubin.get("bytes"), int) or cubin["bytes"] <= 0:
                    raise GateInputError("module byte count is not positive")
            except GateInputError as error:
                violations.append(f"invalid cubin provenance: {error}")
    device = provenance.get("device")
    if not isinstance(device, dict) or any(
        key not in device
        for key in ("uuid", "name", "sm", "sm_count", "driver_version", "cuda_runtime_version")
    ):
        violations.append("complete device provenance is missing")
        device = {}
    if not isinstance(provenance.get("local_toolchain"), dict):
        violations.append("local toolchain provenance is missing")
    if not isinstance(provenance.get("builder"), dict):
        violations.append("builder provenance is missing")

    sanitizer = decision.get("sanitizer")
    if not isinstance(sanitizer, dict) or set(sanitizer) != SANITIZERS:
        violations.append("exact four-tool sanitizer evidence is missing")
    else:
        for tool in sorted(SANITIZERS):
            entry = sanitizer[tool]
            if not isinstance(entry, dict):
                violations.append(f"{tool} sanitizer entry is not an object")
                continue
            add_check(
                violations,
                entry.get("pass") is True and entry.get("exit_code") == 0,
                f"{tool} sanitizer did not pass",
            )
            try:
                expected_log_hash = require_hash(
                    entry.get("sha256"), f"{family}.sanitizer.{tool}.sha256"
                )
                log_path = resolve_bound_path(
                    entry.get("path"), decision_path, f"{family}.sanitizer.{tool}"
                )
                actual_log_hash = sha256_file(log_path)
                if actual_log_hash != expected_log_hash:
                    violations.append(f"{tool} sanitizer log hash mismatch")
            except (GateInputError, OSError) as error:
                violations.append(str(error))

    result: dict[str, Any] | None = None
    try:
        result_path = resolve_bound_path(decision.get("result"), decision_path, f"{family}.result")
        expected_result_hash = require_hash(
            provenance.get("result_json_sha256"), f"{family}.provenance.result_json_sha256"
        )
        result = load_bound_json(result_path, expected_result_hash, f"{family}_result_sha256")
    except (GateInputError, OSError) as error:
        violations.append(str(error))

    return {
        "family": family,
        "decision_path": str(decision_path.resolve()),
        "decision_sha256": decision_sha256,
        "complete": not violations,
        "violations": violations,
        "device": device,
        "result": result,
    }


def q4_projection(
    result: dict[str, Any] | None,
    profile: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    floor = float(policy["gate"]["min_projected_e2e_improvement"])
    projection: dict[str, Any] = {
        "scope": "logical_matmat_share_proxy_not_measured_whole_prefill_share",
        "formula": (
            "sum_i(phase_a1_logical_matmat_share_i * "
            "(1 - triton_median_us_i / native_median_us_i))"
        ),
        "policy_whole_prefill_floor": floor,
        "combined_phase_a1_logical_matmat_share": sum(profile["shares"].values()),
        "components": [],
        "logical_matmat_projection": None,
        "logical_matmat_proxy_floor_met": False,
        "whole_prefill_fraction_measured": None,
        "projected_whole_prefill_improvement": None,
        "projected_whole_prefill_floor_evaluable": False,
        "projected_whole_prefill_floor_met": False,
        "caveat": (
            "Phase A1 percentages divide measured logical matmat time, not whole-prefill "
            "time; they cannot by themselves establish the 3% end-to-end floor."
        ),
    }
    if result is None:
        return projection
    cases = result.get("cases")
    if not isinstance(cases, list):
        return projection
    selected: dict[str, dict[str, Any]] = {}
    for case in cases:
        if not isinstance(case, dict) or case.get("n_tok") != 512:
            continue
        shape = case.get("shape_id")
        timing = case.get("timing")
        if shape in SHAPES and isinstance(timing, dict):
            if shape in selected:
                return projection
            selected[shape] = timing
    if set(selected) != SHAPES:
        return projection

    total = 0.0
    components: list[dict[str, Any]] = []
    for shape in sorted(SHAPES):
        timing = selected[shape]
        native = timing.get("native_median_us")
        triton = timing.get("triton_median_us")
        if not isinstance(native, (int, float)) or not isinstance(triton, (int, float)):
            return projection
        native = float(native)
        triton = float(triton)
        if not math.isfinite(native) or not math.isfinite(triton) or native <= 0 or triton <= 0:
            return projection
        share = float(profile["shares"][shape])
        local_savings = 1.0 - triton / native
        weighted = share * local_savings
        total += weighted
        components.append(
            {
                "shape_id": shape,
                "logical_matmat_share": share,
                "native_median_us": native,
                "triton_median_us": triton,
                "native_over_triton": native / triton,
                "local_savings_fraction": local_savings,
                "weighted_logical_matmat_savings": weighted,
            }
        )
    projection["components"] = components
    projection["logical_matmat_projection"] = total
    projection["logical_matmat_proxy_floor_met"] = total >= floor
    projection["whole_prefill_improvement_interval"] = (
        [0.0, total] if total >= 0.0 else [total, 0.0]
    )
    return projection


def aggregate(
    *,
    policy_path: Path,
    policy_sha256: str,
    phase_a1_path: Path,
    phase_a1_sha256: str,
    rms_decision_path: Path,
    rms_decision_sha256: str,
    q4_decision_path: Path,
    q4_decision_sha256: str,
) -> dict[str, Any]:
    expected_policy_hash = require_hash(policy_sha256, "policy_sha256")
    actual_policy_hash = sha256_file(policy_path)
    if actual_policy_hash != expected_policy_hash:
        raise GateInputError("policy SHA-256 mismatch")
    try:
        policy = tomllib.loads(policy_path.read_text(encoding="utf-8"))
        floor = policy["gate"]["min_projected_e2e_improvement"]
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError, KeyError) as error:
        raise GateInputError(f"read Gate A policy: {error}") from error
    if not isinstance(floor, (int, float)) or not 0.0 < float(floor) < 1.0:
        raise GateInputError("invalid projected end-to-end policy floor")

    profile = parse_phase_a1(phase_a1_path, phase_a1_sha256)
    rms = validate_candidate(
        "rms_q8", rms_decision_path, rms_decision_sha256, actual_policy_hash
    )
    q4 = validate_candidate(
        "q4_mmq", q4_decision_path, q4_decision_sha256, actual_policy_hash
    )
    same_device = bool(rms["device"] and rms["device"] == q4["device"])
    projection = q4_projection(q4["result"], profile, policy)
    blockers: list[str] = []
    if not rms["complete"]:
        blockers.append("RMS candidate correctness/sanitizer/provenance is incomplete")
    if not q4["complete"]:
        blockers.append("high-share Q4 candidate correctness/sanitizer/provenance is incomplete")
    if not same_device:
        blockers.append("candidate evidence device identities differ")
    if not projection["projected_whole_prefill_floor_evaluable"]:
        blockers.append("whole-prefill denominator is absent from current Phase A1 evidence")

    return {
        "schema": 1,
        "phase": "Decision-A-preparation",
        "production_authority": False,
        "final_gate_a_decision": False,
        "decision": "not-issued",
        "inputs": {
            "policy": {"path": str(policy_path.resolve()), "sha256": actual_policy_hash},
            "phase_a1": {key: value for key, value in profile.items() if key != "shares"},
            "rms_decision": {
                "path": str(rms_decision_path.resolve()),
                "sha256": rms_decision_sha256,
            },
            "q4_decision": {
                "path": str(q4_decision_path.resolve()),
                "sha256": q4_decision_sha256,
            },
        },
        "candidate_prerequisites": {
            "rms_q8_complete": rms["complete"],
            "q4_mmq_complete": q4["complete"],
            "same_device": same_device,
            "rms_q8_violations": rms["violations"],
            "q4_mmq_violations": q4["violations"],
        },
        "projection": projection,
        "rms_micro_win_cannot_authorize_gate_a": True,
        "high_share_q4_candidate_required": True,
        "decision_inputs_complete": not blockers,
        "blocking_reasons": blockers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--phase-a1", type=Path, required=True)
    parser.add_argument("--phase-a1-sha256", required=True)
    parser.add_argument("--rms-decision", type=Path, required=True)
    parser.add_argument("--rms-decision-sha256", required=True)
    parser.add_argument("--q4-decision", type=Path, required=True)
    parser.add_argument("--q4-decision-sha256", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = aggregate(
            policy_path=args.policy.resolve(),
            policy_sha256=args.policy_sha256,
            phase_a1_path=args.phase_a1.resolve(),
            phase_a1_sha256=args.phase_a1_sha256,
            rms_decision_path=args.rms_decision.resolve(),
            rms_decision_sha256=args.rms_decision_sha256,
            q4_decision_path=args.q4_decision.resolve(),
            q4_decision_sha256=args.q4_decision_sha256,
        )
    except (GateInputError, OSError) as error:
        print(f"Decision-A aggregation rejected: {error}", file=sys.stderr)
        return 2
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["decision_inputs_complete"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
