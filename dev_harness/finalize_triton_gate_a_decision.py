#!/usr/bin/env python3
"""Issue the fail-closed R2 Decision-A record from finalized RMS and Q4 evidence."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tomllib
from pathlib import Path
from typing import Any

try:
    from dev_harness.aggregate_triton_gate_a import (
        GateInputError,
        load_bound_json,
        parse_phase_a1,
        require_hash,
        sha256_file,
        validate_candidate,
    )
    from dev_harness.finalize_q4_gate_a_evidence import finalize as finalize_q4
    from dev_harness.audit_q4_gate_a_evidence import validate_reviewed_floors
except ModuleNotFoundError:
    from aggregate_triton_gate_a import (  # type: ignore[no-redef]
        GateInputError,
        load_bound_json,
        parse_phase_a1,
        require_hash,
        sha256_file,
        validate_candidate,
    )
    from finalize_q4_gate_a_evidence import finalize as finalize_q4  # type: ignore[no-redef]
    from audit_q4_gate_a_evidence import validate_reviewed_floors  # type: ignore[no-redef]


class DecisionAError(ValueError):
    pass


def decide_no_go_from_candidates(
    *, rms_candidate_admissible: bool, q4_record: dict[str, Any]
) -> str:
    """Return the mandatory No-Go reason only from complete candidate evidence."""
    if not rms_candidate_admissible:
        raise DecisionAError("RMS candidate evidence is incomplete")
    for field in (
        "correctness_admissible", "measurement_admissible",
        "sanitizer_admissible", "provenance_admissible",
    ):
        if q4_record.get(field) is not True:
            raise DecisionAError(f"Q4 {field} must be True")
    if q4_record.get("kernel_floor_met") is not False or q4_record.get("candidate_admissible") is not False:
        raise DecisionAError("Q4 record does not establish the mandatory kernel-floor failure")
    return "mandatory-high-share-q4-kernel-floor-failed"


def finalize(
    *, policy_path: Path, policy_sha256: str,
    phase_a1_path: Path, phase_a1_sha256: str,
    rms_decision_path: Path, rms_decision_sha256: str,
    q4_decision_path: Path, q4_decision_sha256: str,
    q4_formal_matrix_path: Path, q4_sanitizer_manifest_path: Path,
    q4_metadata_path: Path, q4_harness_source_path: Path,
    q4_environment_witness_path: Path,
    q4_formal_matrix_sha256: str, q4_sanitizer_manifest_sha256: str,
    q4_metadata_sha256: str, q4_harness_source_sha256: str,
    q4_environment_witness_sha256: str,
    expected_min_kernel_speedup: str,
    expected_min_projected_e2e: str,
) -> dict[str, Any]:
    expected_policy = require_hash(policy_sha256, "policy_sha256")
    if sha256_file(policy_path) != expected_policy:
        raise DecisionAError("policy SHA-256 mismatch")
    policy = tomllib.loads(policy_path.read_text(encoding="utf-8"))
    reviewed_profile = validate_reviewed_floors(
        policy_path, expected_min_kernel_speedup, expected_min_projected_e2e
    )
    profile = parse_phase_a1(phase_a1_path, phase_a1_sha256)

    rms = validate_candidate(
        "rms_q8", rms_decision_path, rms_decision_sha256, expected_policy
    )
    if not rms["complete"]:
        raise DecisionAError(
            "RMS finalized candidate is incomplete: " + "; ".join(rms["violations"])
        )

    q4_record = load_bound_json(
        q4_decision_path, q4_decision_sha256, "q4_decision_sha256"
    )
    sanitizer_manifest = load_bound_json(
        q4_sanitizer_manifest_path,
        q4_sanitizer_manifest_sha256,
        "q4_sanitizer_manifest_sha256",
    )
    metadata_raw = sanitizer_manifest.get("metadata")
    if metadata_raw is None and isinstance(sanitizer_manifest.get("identity"), dict):
        metadata_raw = sanitizer_manifest["identity"].get("metadata")
    if not isinstance(metadata_raw, str) or Path(metadata_raw).resolve() != q4_metadata_path.resolve():
        raise DecisionAError("Q4 metadata path differs from sanitizer authority")
    recomputed_q4 = finalize_q4(
        formal_matrix_path=q4_formal_matrix_path,
        sanitizer_manifest_path=q4_sanitizer_manifest_path,
        policy_path=policy_path,
        harness_source_path=q4_harness_source_path,
        environment_witness_path=q4_environment_witness_path,
        expected_policy_sha256=expected_policy,
        expected_formal_matrix_sha256=q4_formal_matrix_sha256,
        expected_sanitizer_manifest_sha256=q4_sanitizer_manifest_sha256,
        expected_metadata_sha256=q4_metadata_sha256,
        expected_harness_source_sha256=q4_harness_source_sha256,
        expected_environment_witness_sha256=q4_environment_witness_sha256,
        expected_min_kernel_speedup=expected_min_kernel_speedup,
        expected_min_projected_e2e=expected_min_projected_e2e,
    )
    if q4_record != recomputed_q4:
        raise DecisionAError("Q4 finalized decision differs from recomputed raw evidence")
    required_q4 = {
        "schema": 1,
        "decision": "gate-a-candidate",
        "candidate": "q4_0_x_q8_1_mmq_epilogue0",
        "production_authority": False,
        "reviewed_decision_profile": reviewed_profile,
        "final_gate_a_decision": False,
        "correctness_admissible": True,
        "measurement_admissible": True,
        "sanitizer_admissible": True,
        "provenance_admissible": True,
        "candidate_admissible": False,
        "kernel_floor_met": False,
    }
    for field, expected in required_q4.items():
        if q4_record.get(field) != expected:
            raise DecisionAError(f"Q4 {field} must be {expected!r}")
    reason_code = decide_no_go_from_candidates(
        rms_candidate_admissible=True, q4_record=q4_record
    )
    violations = q4_record.get("violations")
    if not isinstance(violations, list) or not any(
        isinstance(item, str)
        and item.startswith("mandatory high-share Q4 kernel speedup floor failed")
        for item in violations
    ):
        raise DecisionAError("Q4 mandatory high-share kernel-floor violation is missing")
    if rms["device"] != q4_record["provenance"]["device"]:
        raise DecisionAError("RMS and Q4 evidence device identities differ")

    best_label = q4_record["best_observed_pair"]["label"]
    best_pair = next(
        (
            item
            for item in q4_record["performance_matrix"]
            if item.get("label") == best_label
        ),
        None,
    )
    if (
        not isinstance(best_pair, dict)
        or set(best_pair.get("timings", {})) != set(profile["shares"])
    ):
        raise DecisionAError("Q4 best-pair timings do not cover both Phase A1 shapes")
    components: list[dict[str, Any]] = []
    proxy = 0.0
    for shape in sorted(profile["shares"]):
        timing = best_pair["timings"][shape]
        native = float(timing["native_median_us"])
        triton = float(timing["triton_median_us"])
        if not all(math.isfinite(value) and value > 0 for value in (native, triton)):
            raise DecisionAError("Q4 timing must be finite and positive")
        share = float(profile["shares"][shape])
        weighted = share * (1.0 - triton / native)
        proxy += weighted
        components.append(
            {
                "shape_id": shape,
                "logical_matmat_share": share,
                "native_median_us": native,
                "triton_median_us": triton,
                "native_over_triton": native / triton,
                "weighted_logical_matmat_savings": weighted,
            }
        )

    return {
        "schema": 1,
        "phase": "Decision-A",
        "decision": "gate-a-no-go",
        "reason_code": reason_code,
        "production_authority": False,
        "final_gate_a_decision": True,
        "gate_a_go": False,
        "stop_phase_b_c_d": True,
        "rms_candidate_admissible": True,
        "rms_micro_win_cannot_authorize_gate_a": True,
        "q4_candidate_admissible": False,
        "q4_kernel_floor_met": False,
        "mandatory_high_share_candidate_required": True,
        "projection": {
            "scope": "logical_matmat_share_proxy_not_measured_whole_prefill_share",
            "formula": "sum_i(logical_matmat_share_i * (1 - triton_median_us_i / native_median_us_i))",
            "components": components,
            "logical_matmat_projection": proxy,
            "whole_prefill_fraction_measured": None,
            "projected_whole_prefill_improvement": None,
            "projected_whole_prefill_floor": float(
                policy["gate"]["min_projected_e2e_improvement"]
            ),
            "projected_whole_prefill_floor_evaluable": False,
            "projected_whole_prefill_floor_met": False,
            "caveat": (
                "Phase A1 shares divide measured logical matmat time, not whole-prefill "
                "time. The 3% whole-prefill floor is not evaluable; that absence does "
                "not override the mandatory Q4 kernel-floor No-Go."
            ),
        },
        "inputs": {
            "policy": {"path": str(policy_path.resolve()), "sha256": expected_policy},
            "phase_a1": {
                "path": str(phase_a1_path.resolve()),
                "sha256": phase_a1_sha256,
            },
            "rms_decision": {
                "path": str(rms_decision_path.resolve()),
                "sha256": rms_decision_sha256,
            },
            "q4_decision": {
                "path": str(q4_decision_path.resolve()),
                "sha256": q4_decision_sha256,
            },
            "q4_formal_matrix": {
                "path": str(q4_formal_matrix_path.resolve()),
                "sha256": q4_formal_matrix_sha256,
            },
            "q4_sanitizer_manifest": {
                "path": str(q4_sanitizer_manifest_path.resolve()),
                "sha256": q4_sanitizer_manifest_sha256,
            },
            "q4_metadata": {
                "path": str(q4_metadata_path.resolve()),
                "sha256": q4_metadata_sha256,
            },
            "q4_harness_source": {
                "path": str(q4_harness_source_path.resolve()),
                "sha256": q4_harness_source_sha256,
            },
            "q4_environment_witness": {
                "path": str(q4_environment_witness_path.resolve()),
                "sha256": q4_environment_witness_sha256,
            },
        },
        "device": q4_record["provenance"]["device"],
        "no_go_reasons": [
            "mandatory high-share Q4 kernel speedup floor failed",
            "best tested Q4 Triton variant remains slower than native for both shapes",
        ],
        "next_action": (
            "Stop Phase B/C/D. Retain lab evidence; reopen Gate A only with materially "
            "new kernel, algorithm, compiler, or hardware evidence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "policy", "phase-a1", "rms-decision", "q4-decision",
        "q4-formal-matrix", "q4-sanitizer-manifest", "q4-harness-source",
        "q4-metadata", "q4-environment-witness",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in (
        "policy-sha256", "phase-a1-sha256", "rms-decision-sha256",
        "q4-decision-sha256",
        "q4-formal-matrix-sha256", "q4-sanitizer-manifest-sha256",
        "q4-metadata-sha256", "q4-harness-source-sha256",
        "q4-environment-witness-sha256",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-min-kernel-speedup", required=True)
    parser.add_argument("--expected-min-projected-e2e", required=True)
    args = parser.parse_args()
    try:
        decision = finalize(
            policy_path=args.policy.resolve(), policy_sha256=args.policy_sha256,
            phase_a1_path=args.phase_a1.resolve(),
            phase_a1_sha256=args.phase_a1_sha256,
            rms_decision_path=args.rms_decision.resolve(),
            rms_decision_sha256=args.rms_decision_sha256,
            q4_decision_path=args.q4_decision.resolve(),
            q4_decision_sha256=args.q4_decision_sha256,
            q4_formal_matrix_path=args.q4_formal_matrix.resolve(),
            q4_sanitizer_manifest_path=args.q4_sanitizer_manifest.resolve(),
            q4_metadata_path=args.q4_metadata.resolve(),
            q4_harness_source_path=args.q4_harness_source.resolve(),
            q4_environment_witness_path=args.q4_environment_witness.resolve(),
            q4_formal_matrix_sha256=args.q4_formal_matrix_sha256,
            q4_sanitizer_manifest_sha256=args.q4_sanitizer_manifest_sha256,
            q4_metadata_sha256=args.q4_metadata_sha256,
            q4_harness_source_sha256=args.q4_harness_source_sha256,
            q4_environment_witness_sha256=args.q4_environment_witness_sha256,
            expected_min_kernel_speedup=args.expected_min_kernel_speedup,
            expected_min_projected_e2e=args.expected_min_projected_e2e,
        )
    except (DecisionAError, GateInputError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"Decision-A finalization rejected: {error}", file=sys.stderr)
        return 2
    output = args.output.resolve()
    raw = {
        args.policy.resolve(), args.phase_a1.resolve(), args.rms_decision.resolve(),
        args.q4_decision.resolve(), args.q4_formal_matrix.resolve(),
        args.q4_sanitizer_manifest.resolve(), args.q4_harness_source.resolve(),
        args.q4_metadata.resolve(), args.q4_environment_witness.resolve(),
    }
    if output in raw:
        print("refusing to overwrite input evidence", file=sys.stderr)
        return 2
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
