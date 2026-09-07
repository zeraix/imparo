#!/usr/bin/env python3
"""Verify a serialized one-prefill Nsight SQLite GPU-timeline denominator.

This tool is intentionally read-only. It does not run Nsight, launch CUDA, or
claim a host-wall end-to-end denominator. Its output can support a GPU-timeline
projection only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
import statistics
import sys
import tomllib
from pathlib import Path
from typing import Any


HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
ARGUMENT_KEY = re.compile(r"^PROCESS_0:ARGUMENT_(\d+)$")
COMMAND_KEY = re.compile(r"^PROCESS_(\d+):COMMAND$")
MIN_PROJECTED_E2E_FLOOR = 0.03


class DenominatorError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise DenominatorError(f"{field} must be a lowercase SHA-256")
    return value


def load_json(path: Path) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DenominatorError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    try:
        result = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicates
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DenominatorError(f"read JSON {path}: {error}") from error
    if not isinstance(result, dict):
        raise DenominatorError("profile evidence root must be an object")
    return result


def require_file_hash(path: Path, expected: object, field: str) -> str:
    expected_hash = require_sha256(expected, field)
    try:
        actual = sha256_file(path)
    except OSError as error:
        raise DenominatorError(f"hash {field} at {path}: {error}") from error
    if actual != expected_hash:
        raise DenominatorError(
            f"{field} mismatch: expected {expected_hash}, observed {actual}"
        )
    return actual


def canonical_token_hash(tokens: list[int]) -> str:
    encoded = json.dumps(tokens, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_policy_floor(path: Path, expected_sha256: str) -> tuple[str, float]:
    actual = require_file_hash(path, expected_sha256, "policy_sha256")
    try:
        policy = tomllib.loads(path.read_text(encoding="utf-8"))
        floor = float(policy["gate"]["min_projected_e2e_improvement"])
    except (
        OSError,
        UnicodeDecodeError,
        tomllib.TOMLDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        raise DenominatorError(f"read projected E2E policy floor: {error}") from error
    if not math.isfinite(floor) or floor < MIN_PROJECTED_E2E_FLOOR or floor >= 1.0:
        raise DenominatorError(
            "min_projected_e2e_improvement must remain at least 0.03"
        )
    return actual, floor


def sqlite_uri(path: Path) -> str:
    return path.resolve().as_uri() + "?mode=ro&immutable=1"


def read_capture_metadata(connection: sqlite3.Connection) -> dict[str, str]:
    required = {"META_DATA_CAPTURE", "CUPTI_ACTIVITY_KIND_KERNEL"}
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    if not required.issubset(tables):
        raise DenominatorError("Nsight SQLite lacks capture metadata or kernel table")
    values: dict[str, str] = {}
    for name, value in connection.execute(
        "SELECT name, value FROM META_DATA_CAPTURE "
        "WHERE name LIKE 'PROCESS_%'"
    ):
        name = str(name)
        if COMMAND_KEY.fullmatch(name) is None and ARGUMENT_KEY.fullmatch(name) is None:
            continue
        if name in values:
            raise DenominatorError(f"duplicate capture metadata key {name}")
        values[name] = "" if value is None else str(value)
    return values


def observed_workload(metadata: dict[str, str]) -> dict[str, Any]:
    commands = [
        (int(match.group(1)), value)
        for key, value in metadata.items()
        if (match := COMMAND_KEY.fullmatch(key)) is not None
    ]
    if commands != [(0, metadata.get("PROCESS_0:COMMAND", ""))] or not commands[0][1]:
        raise DenominatorError("capture must contain exactly one PROCESS_0 command")
    indexed: dict[int, str] = {}
    for key, value in metadata.items():
        match = ARGUMENT_KEY.fullmatch(key)
        if match is None:
            continue
        index = int(match.group(1))
        if index in indexed:
            raise DenominatorError(f"duplicate process argument index {index}")
        indexed[index] = value
    if not indexed or 0 not in indexed:
        raise DenominatorError("capture model argument is missing")
    expected_indices = list(range(max(indexed) + 1))
    if sorted(indexed) != expected_indices:
        raise DenominatorError("capture process argument indices are not contiguous")
    tokens: list[int] = []
    for index in expected_indices[1:]:
        raw = indexed[index]
        if re.fullmatch(r"0|[1-9][0-9]*", raw) is None:
            raise DenominatorError(f"token argument {index} is not canonical u32")
        token = int(raw)
        if token > 0xFFFFFFFF:
            raise DenominatorError(f"token argument {index} exceeds u32")
        tokens.append(token)
    return {
        "command": commands[0][1],
        "model_argument": indexed[0],
        "token_count": len(tokens),
        "token_ids_sha256": canonical_token_hash(tokens),
    }


def kernel_denominator(connection: sqlite3.Connection) -> dict[str, Any]:
    rows = list(
        connection.execute(
            "SELECT start, end, deviceId, contextId, streamId, globalPid "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start, end"
        )
    )
    if not rows:
        raise DenominatorError("Nsight capture contains no CUDA kernels")
    streams: set[int] = set()
    contexts: set[int] = set()
    devices: set[int] = set()
    processes: set[int] = set()
    total = 0
    previous_end: int | None = None
    first_start: int | None = None
    last_end: int | None = None
    idle = 0
    for start, end, device, context, stream, process in rows:
        if not isinstance(start, int) or not isinstance(end, int) or end <= start:
            raise DenominatorError("kernel interval is non-integer or non-positive")
        streams.add(int(stream))
        contexts.add(int(context))
        devices.add(int(device))
        if process is not None:
            processes.add(int(process))
        if previous_end is not None:
            if start < previous_end:
                raise DenominatorError("CUDA kernels overlap; serialized envelope invalid")
            idle += start - previous_end
        first_start = start if first_start is None else first_start
        previous_end = end
        last_end = end
        total += end - start
    if len(streams) != 1:
        raise DenominatorError("CUDA kernels use multiple streams")
    if len(contexts) != 1 or len(devices) != 1 or len(processes) > 1:
        raise DenominatorError("CUDA kernels span multiple device/context/process identities")
    assert first_start is not None and last_end is not None
    envelope = last_end - first_start
    if envelope != total + idle:
        raise DenominatorError("serialized kernel envelope accounting mismatch")
    return {
        "kernel_count": len(rows),
        "stream_id": next(iter(streams)),
        "context_id": next(iter(contexts)),
        "device_id": next(iter(devices)),
        "global_pid": next(iter(processes)) if processes else None,
        "first_kernel_start_ns": first_start,
        "last_kernel_end_ns": last_end,
        "total_cuda_kernel_work_ns": total,
        "serialized_kernel_envelope_ns": envelope,
        "serialized_idle_gap_ns": idle,
        "kernel_work_fraction_of_envelope": total / envelope,
        "overlap_observed": False,
        "single_stream": True,
    }


def parse_prefill_wall(
    path: Path,
    *,
    repeat_count: int,
    token_count: int,
    start_pos: int,
    split_state: dict[str, Any],
) -> dict[str, Any]:
    if repeat_count != 1:
        raise DenominatorError("formal Phase A1 denominator requires exactly one prefill")
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise DenominatorError(f"read prefill wall output: {error}") from error
    for number, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError as error:
            raise DenominatorError(
                f"invalid JSON line in prefill wall output at {number}: {error}"
            ) from error
        if not isinstance(value, dict) or value.get("event") != "prefill_wall":
            raise DenominatorError("unexpected JSON object in prefill wall output")
        records.append(value)
    if len(records) != repeat_count:
        raise DenominatorError("prefill wall record count mismatch")
    milliseconds: list[float] = []
    for rep, record in enumerate(records):
        wall_ms = record.get("prefill_wall_ms")
        if (
            record.get("schema") != 1
            or record.get("phase") != "A1"
            or record.get("lab_only") is not True
            or record.get("production_authority") is not False
            or record.get("rep") != rep
            or record.get("token_count") != token_count
            or record.get("start_pos") != start_pos
            or record.get("split_state") != split_state
            or not isinstance(wall_ms, (int, float))
            or not math.isfinite(float(wall_ms))
            or float(wall_ms) <= 0.0
        ):
            raise DenominatorError("prefill wall record identity/value mismatch")
        milliseconds.append(float(wall_ms))
    return {
        "records": records,
        "prefill_wall_ms": milliseconds,
        "whole_prefill_host_wall_ms_median": statistics.median(milliseconds),
    }


def verify(
    *,
    sqlite_path: Path,
    sqlite_sha256: str,
    executable_path: Path,
    executable_sha256: str,
    dll_path: Path,
    dll_sha256: str,
    model_path: Path,
    model_sha256: str,
    profile_evidence_path: Path,
    profile_evidence_sha256: str,
    prefill_wall_path: Path,
    prefill_wall_sha256: str,
    policy_path: Path,
    policy_sha256: str,
) -> dict[str, Any]:
    policy_hash, projected_floor = load_policy_floor(policy_path, policy_sha256)
    artifact_hashes = {
        "sqlite_sha256": require_file_hash(
            sqlite_path, sqlite_sha256, "sqlite_sha256"
        ),
        "executable_sha256": require_file_hash(
            executable_path, executable_sha256, "executable_sha256"
        ),
        "dll_sha256": require_file_hash(dll_path, dll_sha256, "dll_sha256"),
        "model_sha256": require_file_hash(
            model_path, model_sha256, "model_sha256"
        ),
        "profile_evidence_sha256": require_file_hash(
            profile_evidence_path,
            profile_evidence_sha256,
            "profile_evidence_sha256",
        ),
        "prefill_wall_sha256": require_file_hash(
            prefill_wall_path, prefill_wall_sha256, "prefill_wall_sha256"
        ),
        "policy_sha256": policy_hash,
    }
    evidence = load_json(profile_evidence_path)
    if (
        evidence.get("schema") != 1
        or evidence.get("phase") != "A1"
        or evidence.get("scope") != "one-prefill-gpu-timeline-denominator"
        or evidence.get("backend_abi") != 26
        or evidence.get("target_sm") != 86
        or not isinstance(evidence.get("head_commit"), str)
        or HEX40.fullmatch(evidence["head_commit"]) is None
    ):
        raise DenominatorError("profile evidence identity/scope mismatch")
    inputs = evidence.get("inputs")
    if not isinstance(inputs, dict):
        raise DenominatorError("profile evidence inputs are missing")
    for field in (
        "sqlite_sha256",
        "executable_sha256",
        "dll_sha256",
        "model_sha256",
        "prefill_wall_sha256",
    ):
        if inputs.get(field) != artifact_hashes[field]:
            raise DenominatorError(f"profile evidence {field} mismatch")
    logical = evidence.get("logical_matmat")
    if (
        not isinstance(logical, dict)
        or logical.get("scope") != "complete-native-imparo_cuda_matmat-cuda-events"
        or logical.get("collection_relationship")
        not in ("same-run", "identity-matched-separate-run")
        or not isinstance(logical.get("calls"), int)
        or logical["calls"] <= 0
        or not isinstance(logical.get("total_ns"), int)
        or logical["total_ns"] <= 0
    ):
        raise DenominatorError("logical-matmat profiler evidence is incomplete")
    try:
        connection = sqlite3.connect(sqlite_uri(sqlite_path), uri=True)
        connection.execute("PRAGMA query_only = ON")
        metadata = read_capture_metadata(connection)
        workload = observed_workload(metadata)
        denominator = kernel_denominator(connection)
    except sqlite3.Error as error:
        raise DenominatorError(f"read Nsight SQLite: {error}") from error
    finally:
        if "connection" in locals():
            connection.close()
    expected_workload = {
        "command": inputs.get("capture_command"),
        "model_argument": inputs.get("capture_model_argument"),
        "token_count": inputs.get("token_count"),
        "token_ids_sha256": inputs.get("token_ids_sha256"),
    }
    if workload != expected_workload:
        raise DenominatorError("capture command/model/token identity mismatch")
    expected_split = inputs.get("prefill_split_state")
    if not isinstance(expected_split, dict):
        raise DenominatorError("profile evidence prefill split state is missing")
    wall = parse_prefill_wall(
        prefill_wall_path,
        repeat_count=inputs.get("repeat_count"),
        token_count=inputs.get("prefill_token_count"),
        start_pos=inputs.get("prefill_start_pos"),
        split_state=expected_split,
    )
    logical_ns = logical["total_ns"]
    envelope_ns = denominator["serialized_kernel_envelope_ns"]
    if logical_ns > envelope_ns:
        raise DenominatorError("logical-matmat time exceeds GPU kernel envelope")
    fraction = logical_ns / envelope_ns
    host_prefill_ns = wall["whole_prefill_host_wall_ms_median"] * 1e6
    if logical_ns > host_prefill_ns:
        raise DenominatorError("logical-matmat time exceeds host prefill wall")
    host_fraction = logical_ns / host_prefill_ns
    same_run = logical["collection_relationship"] == "same-run"
    return {
        "schema": 1,
        "phase": "A1-denominator-verification",
        "production_authority": False,
        "final_gate_a_decision": False,
        "admissible_gpu_timeline_denominator": True,
        "scope": "gpu-timeline-projection-only-not-host-wall-end-to-end",
        "inputs": {
            "sqlite": str(sqlite_path.resolve()),
            "executable": str(executable_path.resolve()),
            "dll": str(dll_path.resolve()),
            "model": str(model_path.resolve()),
            "profile_evidence": str(profile_evidence_path.resolve()),
            "prefill_wall": str(prefill_wall_path.resolve()),
            "policy": str(policy_path.resolve()),
            "hashes": artifact_hashes,
            "head_commit": evidence["head_commit"],
            "backend_abi": 26,
            "target_sm": 86,
        },
        "workload": workload,
        "kernel_timeline": denominator,
        "logical_matmat": {
            "scope": logical["scope"],
            "collection_relationship": logical["collection_relationship"],
            "calls": logical["calls"],
            "total_ns": logical_ns,
            "fraction_of_whole_prefill_gpu_timeline": fraction,
            "fraction_of_whole_prefill_host_wall": host_fraction,
            "fraction_kind": (
                "same-run-measured"
                if same_run
                else "identity-matched-cross-run-estimate"
            ),
        },
        "prefill_wall": wall,
        "projection_boundary": {
            "gpu_timeline_fraction_measured": fraction,
            "whole_prefill_host_wall_fraction_measured": host_fraction,
            "whole_prefill_host_wall_denominator_available": True,
            "full_request_host_wall_end_to_end_denominator_available": False,
            "projected_whole_prefill_host_wall_improvement": None,
            "projected_whole_prefill_floor_evaluable": False,
            "projected_full_request_host_wall_improvement": None,
            "min_projected_e2e_improvement": projected_floor,
            "three_percent_policy_weakened": False,
            "caveat": (
                "The GPU denominator is the serialized first-to-last CUDA-kernel "
                "envelope for one bound prefill. The separately bound opt-in wall "
                "record is host wall for that prefill, not full-request wall. "
                "The logical numerator is "
                + ("from the same run. " if same_run else
                   "from a separately collected identity-matched run. ")
                + "A measured candidate savings ratio is still required before "
                "the 3% projected whole-prefill floor is evaluable."
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "sqlite",
        "executable",
        "dll",
        "model",
        "profile-evidence",
        "prefill-wall",
        "policy",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
        parser.add_argument(f"--{name}-sha256", required=True)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    try:
        report = verify(
            sqlite_path=arguments.sqlite,
            sqlite_sha256=arguments.sqlite_sha256,
            executable_path=arguments.executable,
            executable_sha256=arguments.executable_sha256,
            dll_path=arguments.dll,
            dll_sha256=arguments.dll_sha256,
            model_path=arguments.model,
            model_sha256=arguments.model_sha256,
            profile_evidence_path=arguments.profile_evidence,
            profile_evidence_sha256=arguments.profile_evidence_sha256,
            prefill_wall_path=arguments.prefill_wall,
            prefill_wall_sha256=arguments.prefill_wall_sha256,
            policy_path=arguments.policy,
            policy_sha256=arguments.policy_sha256,
        )
    except DenominatorError as error:
        print(f"denominator verification failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
