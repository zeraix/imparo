#!/usr/bin/env python3
"""Finalize bounded Q4 formal/sanitizer evidence into a non-production candidate record."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tomllib
from pathlib import Path
from typing import Any

try:
    from dev_harness.audit_q4_gate_a_evidence import audit as strict_audit
except ModuleNotFoundError:
    from audit_q4_gate_a_evidence import audit as strict_audit  # type: ignore[no-redef]

try:
    from dev_harness.finalize_q4_sanitizer_evidence import (
        LABELS,
        TOOLS,
        Q4SanitizerEvidenceError,
        load_json,
        require_hash,
        resolve_path,
        sha256_file,
        validate_harness,
        validate_matrix,
    )
except ModuleNotFoundError:
    from finalize_q4_sanitizer_evidence import (  # type: ignore[no-redef]
        LABELS,
        TOOLS,
        Q4SanitizerEvidenceError,
        load_json,
        require_hash,
        resolve_path,
        sha256_file,
        validate_harness,
        validate_matrix,
    )


class Q4GateEvidenceError(ValueError):
    pass


def select_best_pair(
    formal: list[dict[str, Any]], floor: float
) -> tuple[str, float, dict[str, dict[str, float]], bool]:
    """Score each variant pair by its slower shape; other variants cannot veto it."""
    if not formal:
        raise Q4GateEvidenceError("formal performance matrix is empty")
    pair_scores = [
        (
            run["label"],
            min(timing["native_over_triton"] for timing in run["timings"].values()),
            run["timings"],
        )
        for run in formal
    ]
    best_label, best_score, best_timings = max(pair_scores, key=lambda item: item[1])
    return best_label, best_score, best_timings, best_score >= floor


def validate_formal_matrix(
    *,
    formal_matrix_path: Path,
    sanitizer_binding: dict[str, Any],
    policy_path: Path,
    harness_source_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        policy = tomllib.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise Q4GateEvidenceError(f"read policy: {error}") from error
    matrix = load_json(formal_matrix_path)
    measurement = policy["measurement"]
    expected_contract = {
        "warmup": measurement["warmup"],
        "pairs": measurement["abba_baab_pairs"],
        "launches_per_sample": measurement["launches_per_sample"],
        "samples_per_route": measurement["min_samples_per_route"],
        "sequential": True,
    }
    if (
        matrix.get("schema") != 1
        or matrix.get("phase") != "A2"
        or matrix.get("milestone") != "bounded-variant-formal-raw-matrix"
        or matrix.get("production_enabled") is not False
        or matrix.get("candidate_decision") is not False
        or matrix.get("execution_order") != list(LABELS)
        or matrix.get("formal_contract") != expected_contract
        or matrix.get("policy_sha256") != sha256_file(policy_path)
        or matrix.get("metadata_sha256")
        != sanitizer_binding["provenance"]["metadata_sha256"]
        or matrix.get("fixed_harness_exe_sha256")
        != sanitizer_binding["provenance"]["harness_exe_sha256"]
    ):
        raise Q4GateEvidenceError("formal matrix identity/contract/hash mismatch")
    preflight = sanitizer_binding["preflight"]
    expected = {(item["label"], item["shape_id"]): item for item in preflight}
    runs = matrix.get("runs")
    if not isinstance(runs, list) or len(runs) != 4:
        raise Q4GateEvidenceError("formal matrix must contain exactly four pair runs")
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    common_device: dict[str, Any] | None = None
    request = {
        "warmup": measurement["warmup"],
        "pairs": measurement["abba_baab_pairs"],
        "launches_per_sample": measurement["launches_per_sample"],
        "samples_per_route": measurement["min_samples_per_route"],
        "formal_contract": True,
    }
    for run_value in runs:
        if not isinstance(run_value, dict):
            raise Q4GateEvidenceError("formal run is not an object")
        label = run_value.get("label")
        if label not in LABELS or label in seen:
            raise Q4GateEvidenceError("formal run label is duplicate/unexpected")
        seen.add(label)
        if (
            run_value.get("exit_code") != 0
            or run_value.get("validation_pass") is not True
            or run_value.get("violations") != []
        ):
            raise Q4GateEvidenceError(f"{label} formal raw validation did not pass")
        directory = resolve_path(run_value.get("directory"), formal_matrix_path, "formal.directory")
        provenance_path = resolve_path(
            run_value.get("provenance"), formal_matrix_path, "formal.provenance"
        )
        require_hash(provenance_path, run_value.get("provenance_sha256"), "formal.provenance")
        provenance = load_json(provenance_path)
        hashes = provenance.get("hashes")
        command = provenance.get("command")
        if (
            provenance.get("schema") != 1
            or provenance.get("phase") != "A2"
            or provenance.get("milestone") != "bounded-variant-formal-raw"
            or provenance.get("production_enabled") is not False
            or provenance.get("candidate_decision") is not False
            or provenance.get("label") != label
            or provenance.get("exit_code") != 0
            or provenance.get("validation") != {"pass": True, "violations": []}
            or not isinstance(hashes, dict)
            or not isinstance(command, dict)
        ):
            raise Q4GateEvidenceError(f"{label} formal provenance identity mismatch")
        stdout_path = directory / "stdout.json"
        stderr_path = directory / "stderr.log"
        exit_path = directory / "exit-code.txt"
        require_hash(stdout_path, hashes.get("stdout_sha256"), f"{label}.stdout")
        require_hash(stderr_path, hashes.get("stderr_sha256"), f"{label}.stderr")
        if stderr_path.stat().st_size != 0 or exit_path.read_text(encoding="utf-8").strip() != "0":
            raise Q4GateEvidenceError(f"{label} formal stderr/exit evidence mismatch")
        if (
            hashes.get("policy_sha256") != sha256_file(policy_path)
            or hashes.get("metadata_sha256") != sanitizer_binding["provenance"]["metadata_sha256"]
            or hashes.get("source_sha256") != sha256_file(harness_source_path)
            or hashes.get("exe_sha256") != sanitizer_binding["provenance"]["harness_exe_sha256"]
        ):
            raise Q4GateEvidenceError(f"{label} formal source/tool hash mismatch")
        expected_by_shape = {shape: expected[(label, shape)] for shape in (
            "k2560-m10240", "k10240-m2560"
        )}
        expansion = expected_by_shape["k2560-m10240"]
        contraction = expected_by_shape["k10240-m2560"]
        if (
            hashes.get("expansion_cubin_sha256") != expansion["module_sha256"]
            or hashes.get("contraction_cubin_sha256") != contraction["module_sha256"]
            or command.get("exe") != sanitizer_binding["provenance"]["harness_exe"]
        ):
            raise Q4GateEvidenceError(f"{label} formal module/executable mismatch")
        arguments = command.get("arguments")
        expected_arguments = [
            expansion["module"],
            expansion["symbol"],
            contraction["module"],
            contraction["symbol"],
            str(request["warmup"]),
            str(request["pairs"]),
            str(request["launches_per_sample"]),
            str(expansion["block_m"]),
            str(expansion["block_n"]),
            str(expansion["threads"]),
            str(expansion["dynamic_shared_bytes"]),
            str(contraction["block_m"]),
            str(contraction["block_n"]),
            str(contraction["threads"]),
            str(contraction["dynamic_shared_bytes"]),
        ]
        if arguments != expected_arguments:
            raise Q4GateEvidenceError(f"{label} formal argv differs from exact pair contract")
        result = load_json(stdout_path)
        device = validate_harness(
            stdout_path,
            label=label,
            expected_by_shape=expected_by_shape,
            policy=policy,
            measurement_request=request,
            require_formal_timing=True,
        )
        if common_device is None:
            common_device = device
        elif common_device != device:
            raise Q4GateEvidenceError("formal pair device identities differ")
        timings: dict[str, dict[str, float]] = {}
        for case in result["cases"]:
            if case.get("n_tok") != 512:
                continue
            timing = case["timing"]
            shape = case["shape_id"]
            native = float(timing["native_median_us"])
            triton = float(timing["triton_median_us"])
            reported_native = run_value[
                "expansion_native_median_us"
                if shape == "k2560-m10240"
                else "contraction_native_median_us"
            ]
            reported_triton = run_value[
                "expansion_triton_median_us"
                if shape == "k2560-m10240"
                else "contraction_triton_median_us"
            ]
            if native != reported_native or triton != reported_triton:
                raise Q4GateEvidenceError(f"{label}/{shape} aggregate medians mismatch")
            timings[shape] = {
                "native_median_us": native,
                "triton_median_us": triton,
                "native_over_triton": native / triton,
            }
        if set(timings) != {"k2560-m10240", "k10240-m2560"}:
            raise Q4GateEvidenceError(f"{label} formal timing pair is incomplete")
        output.append(
            {
                "label": label,
                "provenance": str(provenance_path),
                "provenance_sha256": run_value["provenance_sha256"],
                "result": str(stdout_path),
                "result_sha256": hashes["stdout_sha256"],
                "timings": timings,
            }
        )
    if seen != set(LABELS) or common_device != sanitizer_binding["device"]:
        raise Q4GateEvidenceError("formal/sanitizer pair coverage or device identity differs")
    return sorted(output, key=lambda item: item["label"]), common_device or {}


def finalize(
    *,
    formal_matrix_path: Path,
    sanitizer_manifest_path: Path,
    policy_path: Path,
    harness_source_path: Path,
    environment_witness_path: Path,
    expected_policy_sha256: str,
    expected_formal_matrix_sha256: str,
    expected_sanitizer_manifest_sha256: str,
    expected_metadata_sha256: str,
    expected_harness_source_sha256: str,
    expected_environment_witness_sha256: str,
    expected_min_kernel_speedup: str,
    expected_min_projected_e2e: str,
) -> dict[str, Any]:
    sanitizer_manifest = load_json(sanitizer_manifest_path)
    if sanitizer_manifest.get("schema") == 2:
        identity = sanitizer_manifest.get("identity")
        if not isinstance(identity, dict) or not isinstance(identity.get("metadata"), str):
            raise Q4GateEvidenceError("schema2 sanitizer metadata authority is missing")
        metadata_path = Path(identity["metadata"]).resolve()
    else:
        metadata_path = Path(sanitizer_manifest["metadata"]).resolve()
    strict = strict_audit(
        policy_path=policy_path,
        expected_policy_sha256=expected_policy_sha256,
        formal_matrix_path=formal_matrix_path,
        expected_formal_matrix_sha256=expected_formal_matrix_sha256,
        sanitizer_manifest_path=sanitizer_manifest_path,
        expected_sanitizer_manifest_sha256=expected_sanitizer_manifest_sha256,
        metadata_path=metadata_path,
        expected_metadata_sha256=expected_metadata_sha256,
        harness_source_path=harness_source_path,
        expected_harness_source_sha256=expected_harness_source_sha256,
        environment_witness_path=environment_witness_path,
        expected_environment_witness_sha256=expected_environment_witness_sha256,
        expected_min_kernel_speedup=expected_min_kernel_speedup,
        expected_min_projected_e2e=expected_min_projected_e2e,
    )
    sanitizer = strict.get("sanitizer")
    if sanitizer is None:
        sanitizer = validate_matrix(sanitizer_manifest_path, policy_path)
    formal, device = validate_formal_matrix(
        formal_matrix_path=formal_matrix_path,
        sanitizer_binding=sanitizer,
        policy_path=policy_path,
        harness_source_path=harness_source_path,
    )
    policy = tomllib.loads(policy_path.read_text(encoding="utf-8"))
    sanitizer_schema = sanitizer_manifest.get("schema")
    sanitizer_authoritative = sanitizer_schema == 2 and sanitizer.get("schema") == 2
    floor = float(policy["gate"]["min_kernel_speedup_ratio"])
    best_label, best_pair_score, best_timings, kernel_floor_met = select_best_pair(
        formal, floor
    )
    observations = [
        (
            f"{label}/{shape} kernel speedup "
            f"{timing['native_over_triton']:.9f} < required {floor:.9f}"
        )
        for run in formal
        for label in (run["label"],)
        for shape, timing in run["timings"].items()
        if timing["native_over_triton"] < floor
    ]
    violations: list[str] = []
    if not kernel_floor_met:
        violations.append(
            "mandatory high-share Q4 kernel speedup floor failed; "
            f"best_pair={best_label} min(expansion,contraction)={best_pair_score:.9f}"
        )
    if not sanitizer_authoritative:
        violations.append(
            "legacy sanitizer manifest lacks exact runner/argv/stdout/stderr/exit authority; "
            "schema 2 variants-c evidence is required"
        )
    metadata = load_json(metadata_path)
    cubins = [
        {
            "shape": entry["shape_id"],
            "config_id": entry["config_id"],
            "symbol": entry["symbol"],
            "path": entry["module"],
            "sha256": entry["module_sha256"],
            "bytes": entry["module_bytes"],
        }
        for entry in sanitizer["preflight"]
    ]
    selected_runs = {
        run["tool"]: {
            "path": run["log"],
            "sha256": run["log_sha256"],
            "harness_output": run["harness_output"],
            "harness_output_sha256": run["harness_output_sha256"],
            "exit_code": 0,
            "pass": True,
        }
        for run in sanitizer["runs"]
        if run["label"] == best_label
    }
    if set(selected_runs) != set(TOOLS):
        raise Q4GateEvidenceError("best pair lacks selected four-tool sanitizer evidence")
    return {
        "schema": 1,
        "decision": "gate-a-candidate",
        "candidate": "q4_0_x_q8_1_mmq_epilogue0",
        "production_authority": False,
        "final_gate_a_decision": False,
        "correctness_admissible": True,
        "measurement_admissible": True,
        "sanitizer_admissible": sanitizer_authoritative,
        "provenance_admissible": sanitizer_authoritative,
        "candidate_admissible": kernel_floor_met and not violations and sanitizer_authoritative,
        "sanitizer_authority_schema": sanitizer_schema,
        "kernel_floor_met": kernel_floor_met,
        "kernel_speed_floor_ratio": floor,
        "reviewed_decision_profile": strict["reviewed_decision_profile"],
        "projected_e2e_improvement": None,
        "projected_e2e_floor_met": False,
        "selection_mode": "bounded-variant-matrix",
        "best_observed_pair": {
            "label": best_label,
            "pair_score_min_native_over_triton": best_pair_score,
            "shape_speedups": {
                shape: timing["native_over_triton"]
                for shape, timing in sorted(best_timings.items())
            },
            "configs": sanitizer["pairs"][best_label],
        },
        "kernel_floor_observations": observations,
        "performance_matrix": formal,
        "sanitizer": selected_runs,
        "sanitizer_matrix": sanitizer,
        "result": str(formal_matrix_path.resolve()),
        "provenance": {
            "policy_sha256": strict["policy_sha256"],
            "metadata_sha256": sanitizer["provenance"]["metadata_sha256"],
            "harness_source_sha256": sha256_file(harness_source_path),
            "harness_exe_sha256": sanitizer["provenance"]["harness_exe_sha256"],
            "result_json_sha256": strict["formal_matrix_sha256"],
            "sanitizer_manifest_sha256": strict["sanitizer_manifest_sha256"],
            "environment_witness_sha256": strict["environment_witness_sha256"],
            "cubins": cubins,
            "device": device,
            "local_toolchain": strict["local_toolchain"],
            "builder": metadata.get("builder"),
            "sanitizer_runner": sanitizer["provenance"].get("runner"),
            "sanitizer_runner_sha256": sanitizer["provenance"].get("runner_sha256"),
            "compute_sanitizer": sanitizer["provenance"].get("compute_sanitizer"),
            "cuobjdump": sanitizer["provenance"].get("cuobjdump"),
            "nvidia_smi": sanitizer["provenance"].get("nvidia_smi"),
        },
        "numeric_policy_status": policy["q4_mmq"]["q4_numeric_policy_status"],
        "violations": violations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-matrix", type=Path, required=True)
    parser.add_argument("--sanitizer-manifest", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--harness-source", type=Path, required=True)
    parser.add_argument("--environment-witness", type=Path, required=True)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--formal-matrix-sha256", required=True)
    parser.add_argument("--sanitizer-manifest-sha256", required=True)
    parser.add_argument("--metadata-sha256", required=True)
    parser.add_argument("--harness-source-sha256", required=True)
    parser.add_argument("--environment-witness-sha256", required=True)
    parser.add_argument("--expected-min-kernel-speedup", required=True)
    parser.add_argument("--expected-min-projected-e2e", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        decision = finalize(
            formal_matrix_path=args.formal_matrix.resolve(),
            sanitizer_manifest_path=args.sanitizer_manifest.resolve(),
            policy_path=args.policy.resolve(),
            harness_source_path=args.harness_source.resolve(),
            environment_witness_path=args.environment_witness.resolve(),
            expected_policy_sha256=args.policy_sha256,
            expected_formal_matrix_sha256=args.formal_matrix_sha256,
            expected_sanitizer_manifest_sha256=args.sanitizer_manifest_sha256,
            expected_metadata_sha256=args.metadata_sha256,
            expected_harness_source_sha256=args.harness_source_sha256,
            expected_environment_witness_sha256=args.environment_witness_sha256,
            expected_min_kernel_speedup=args.expected_min_kernel_speedup,
            expected_min_projected_e2e=args.expected_min_projected_e2e,
        )
    except (
        Q4GateEvidenceError,
        Q4SanitizerEvidenceError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        print(f"Q4 Gate-A finalization rejected: {error}", file=sys.stderr)
        return 2
    output = args.output.resolve()
    raw = {
        args.formal_matrix.resolve(),
        args.sanitizer_manifest.resolve(),
        args.policy.resolve(),
        args.harness_source.resolve(),
        args.environment_witness.resolve(),
    }
    if output in raw:
        print("refusing to overwrite raw evidence", file=sys.stderr)
        return 2
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0 if decision["candidate_admissible"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
