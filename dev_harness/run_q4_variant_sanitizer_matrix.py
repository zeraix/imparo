#!/usr/bin/env python3
"""Fail-closed Phase A2 Q4 variant sanitizer evidence runner."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
EXPECTED_POLICY_SHA256 = "b6572d7f48f695022eaf7b2bcba3881df938ce1f0457273e8b4800743f9f22ca"
EXPECTED_METADATA_SHA256 = "f4dd69db8475ec881f5bf5062d1a7d4626605b658bd7727233af077eef3d2607"
EXPECTED_EXE_SHA256 = "910b0355e0a576069824bb8d459d441b5ac02b8cd537c551a5e4476848a2aa42"
EXPECTED_SOURCE_SHA256 = "ed7860950f657d7e0affc37ed46729742fd5a34a4d572a0670d2cafaa56d0137"
LABELS = (
    "bm128-bn64-w8-s2",
    "bm64-bn128-w8-s2",
    "bm64-bn64-w8-s2",
    "bm64-bn64-w8-s1",
)
TOOLS = ("memcheck", "initcheck", "racecheck", "synccheck")
SHAPES = ("k2560-m10240", "k10240-m2560")
TAIL_TOKENS = (1, 127, 128, 129, 511, 512)
EXPECTED_KPARAM_OFFSETS = (0, 8, 16, 24, 32, 40, 44, 48, 56, 64)
EXPECTED_KPARAM_SIZES = (8, 8, 8, 8, 8, 4, 4, 4, 8, 8)
RELEVANT_ENVIRONMENT = (
    "CUDA_VISIBLE_DEVICES",
    "CUDA_MODULE_LOADING",
    "CUDA_CACHE_PATH",
    "CUDA_CACHE_DISABLE",
    "CUDA_FORCE_PTX_JIT",
    "CUDA_DISABLE_PTX_JIT",
    "CUDA_LAUNCH_BLOCKING",
    "IMPARO_CUDA_KERNEL_LAB",
)


class EvidenceError(RuntimeError):
    pass


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    ) as stream:
        json.dump(value, stream, indent=2, sort_keys=False)
        stream.write("\n")
        temporary = pathlib.Path(stream.name)
    os.replace(temporary, path)


def tool_identity(
    path: pathlib.Path, version_args: list[str], product: str
) -> dict[str, Any]:
    completed = subprocess.run(
        [str(path), *version_args], check=False, capture_output=True, text=True
    )
    version = (completed.stdout + completed.stderr).strip()
    if completed.returncode != 0 or not version or product.lower() not in version.lower():
        raise EvidenceError(f"{path.name} version identity failed")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "version_argv": [str(path.resolve()), *version_args],
        "version_exit_code": completed.returncode,
        "version_stdout": completed.stdout,
        "version_stderr": completed.stderr,
        "version": version,
    }


def relevant_environment(environment: dict[str, str]) -> dict[str, Any]:
    values = {name: environment.get(name) for name in RELEVANT_ENVIRONMENT}
    path_value = environment.get("PATH", "")
    values["PATH"] = path_value
    values["PATH_sha256"] = hashlib.sha256(path_value.encode("utf-8")).hexdigest()
    return values


def discover_tool_paths(
    environment: dict[str, str] | None = None,
) -> dict[str, pathlib.Path | None]:
    environment = os.environ if environment is None else environment
    cuda_roots: list[pathlib.Path] = []
    if environment.get("CUDA_PATH"):
        cuda_roots.append(pathlib.Path(environment["CUDA_PATH"]))
    nvcc = shutil.which("nvcc.exe", path=environment.get("PATH"))
    if nvcc:
        cuda_roots.append(pathlib.Path(nvcc).resolve().parent.parent)
    program_files = environment.get("ProgramFiles")
    if program_files:
        toolkit_parent = (
            pathlib.Path(program_files)
            / "NVIDIA GPU Computing Toolkit"
            / "CUDA"
        )
        if toolkit_parent.is_dir():
            cuda_roots.extend(
                sorted(
                    (path for path in toolkit_parent.glob("v*") if path.is_dir()),
                    reverse=True,
                )
            )
    unique_roots: list[pathlib.Path] = []
    for root in cuda_roots:
        resolved = root.resolve()
        if resolved not in unique_roots:
            unique_roots.append(resolved)

    def first_existing(candidates: list[pathlib.Path | None]) -> pathlib.Path | None:
        for candidate in candidates:
            if candidate is not None and candidate.is_file():
                return candidate.resolve()
        return None

    sanitizer_from_path = shutil.which(
        "compute-sanitizer.exe", path=environment.get("PATH")
    )
    cuobjdump_from_path = shutil.which("cuobjdump.exe", path=environment.get("PATH"))
    nvidia_smi_from_path = shutil.which("nvidia-smi.exe", path=environment.get("PATH"))
    system_root = pathlib.Path(environment["SystemRoot"]) if environment.get(
        "SystemRoot"
    ) else None
    return {
        "compute_sanitizer": first_existing(
            [
                *[
                    root / "compute-sanitizer" / "compute-sanitizer.exe"
                    for root in unique_roots
                ],
                pathlib.Path(sanitizer_from_path) if sanitizer_from_path else None,
            ]
        ),
        "cuobjdump": first_existing(
            [
                *[root / "bin" / "cuobjdump.exe" for root in unique_roots],
                pathlib.Path(cuobjdump_from_path) if cuobjdump_from_path else None,
            ]
        ),
        "nvidia_smi": first_existing(
            [
                pathlib.Path(nvidia_smi_from_path) if nvidia_smi_from_path else None,
                system_root / "System32" / "nvidia-smi.exe"
                if system_root is not None
                else None,
            ]
        ),
    }


def require_hash(path: pathlib.Path, expected: str, label: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise EvidenceError(f"{label} SHA256 mismatch: {actual} != {expected}")
    return actual


def load_inputs(
    policy_path: pathlib.Path, metadata_path: pathlib.Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    require_hash(policy_path, EXPECTED_POLICY_SHA256, "policy")
    require_hash(metadata_path, EXPECTED_METADATA_SHA256, "variant metadata")
    with policy_path.open("rb") as stream:
        policy = tomllib.load(stream)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        policy.get("schema") != 1
        or policy.get("phase") != "A2"
        or policy.get("decision") != "gate-a"
    ):
        raise EvidenceError("policy identity is not schema1/Phase A2/Gate A")
    if (
        metadata.get("schema") != 1
        or metadata.get("phase") != "A2"
        or metadata.get("production_enabled") is not False
        or metadata.get("target") != "cuda:86:32"
        or metadata.get("matrix", {}).get("bounded") is not True
    ):
        raise EvidenceError("variant metadata identity is invalid")
    return policy, metadata


def build_pairs(
    metadata: dict[str, Any], metadata_path: pathlib.Path
) -> list[dict[str, Any]]:
    survivors = metadata.get("survivors")
    if not isinstance(survivors, list) or len(survivors) != 8:
        raise EvidenceError("metadata must contain exactly eight survivors")
    root = metadata_path.parent.resolve()
    pairs: list[dict[str, Any]] = []
    seen_modules: set[pathlib.Path] = set()
    for label in LABELS:
        matches = [item for item in survivors if item.get("lab_label") == label]
        if len(matches) != 2:
            raise EvidenceError(f"{label} must contain exactly two shapes")
        by_shape = {item.get("shape_id"): item for item in matches}
        if tuple(shape for shape in SHAPES if shape in by_shape) != SHAPES:
            raise EvidenceError(f"{label} does not contain the exact shape pair")
        normalized: list[dict[str, Any]] = []
        for shape in SHAPES:
            item = by_shape[shape]
            module = (root / item["module"]).resolve()
            try:
                module.relative_to(root)
            except ValueError as error:
                raise EvidenceError(f"module escapes metadata root: {module}") from error
            if module in seen_modules:
                raise EvidenceError(f"duplicate module path: {module}")
            seen_modules.add(module)
            if not module.is_file() or sha256(module) != item["module_sha256"]:
                raise EvidenceError(f"module missing or SHA256 mismatch: {module}")
            if (
                not isinstance(item.get("module_bytes"), int)
                or item["module_bytes"] <= 0
                or module.stat().st_size != item["module_bytes"]
            ):
                raise EvidenceError(f"module size does not match metadata: {module}")
            block = item.get("block")
            tile = item.get("tile")
            resources = item.get("resources")
            if (
                block != [256, 1, 1]
                or not isinstance(tile, dict)
                or not isinstance(resources, dict)
                or tile.get("rows") not in (64, 128)
                or tile.get("tokens") not in (64, 128)
                or resources.get("dynamic_shared_bytes") not in (16384, 32768)
                or resources.get("local_memory_bytes") != 0
            ):
                raise EvidenceError(f"unsupported geometry/resources: {label}/{shape}")
            normalized.append(
                {
                    "label": label,
                    "shape": shape,
                    "module": str(module),
                    "module_sha256": item["module_sha256"],
                    "module_bytes": item["module_bytes"],
                    "symbol": item["symbol"],
                    "block_m": int(tile["rows"]),
                    "block_n": int(tile["tokens"]),
                    "threads": int(block[0]),
                    "dynamic_shared_bytes": int(resources["dynamic_shared_bytes"]),
                    "registers_per_thread": int(resources["registers_per_thread"]),
                    "local_memory_bytes": int(resources["local_memory_bytes"]),
                    "config_id": item["config_id"],
                    "identity_descriptor_sha256": item["identity_descriptor_sha256"],
                }
            )
        pairs.append(
            {"label": label, "expansion": normalized[0], "contraction": normalized[1]}
        )
    if len(seen_modules) != 8:
        raise EvidenceError("module path set is not exactly eight")
    return pairs


def parse_cuobjdump(
    text: str, *, symbol: str, threads: int, registers: int
) -> dict[str, Any]:
    elf_headers = re.findall(r"(?m)^64bit elf:.*$", text)
    function_sections = re.findall(r"(?m)^\.nv\.info\.([^\r\n]+)\r?$", text)
    info_match = re.search(
        rf"(?ms)^\.nv\.info\.{re.escape(symbol)}\r?\n"
        rf"(?P<body>.*?)(?=^\S|\Z)",
        text,
    )
    global_match = re.search(
        r"(?ms)^\.nv\.info\r?\n(?P<body>.*?)(?=^\S|\Z)", text
    )
    text_match = re.search(
        rf"(?m)^\.text\.{re.escape(symbol)}\r?\n"
        rf"bar\s*=\s*\d+\s+reg\s*=\s*(\d+)\s+"
        r"lmem\s*=\s*(\d+)\s+smem\s*=\s*(\d+)\s*$",
        text,
    )
    if (
        len(elf_headers) != 1
        or not re.search(r"\bsm=86\b", elf_headers[0])
        or function_sections != [symbol]
        or info_match is None
        or global_match is None
        or text_match is None
    ):
        raise EvidenceError(f"cuobjdump symbol scope failed for {symbol}")
    info_body = info_match.group("body")
    global_body = global_match.group("body")
    checks: dict[str, Any] = {
        "sm": 86,
        "elf_header_count": len(elf_headers),
        "function_info_sections": function_sections,
        "symbol_present": True,
    }
    cbank_sizes = re.findall(
        r"EIATTR_CBANK_PARAM_SIZE[\s\S]*?Value:\s+0x([0-9a-f]+)\b",
        info_body,
        flags=re.IGNORECASE,
    )
    checks["kparam_bytes"] = (
        int(cbank_sizes[0], 16) if len(cbank_sizes) == 1 else None
    )
    parameters = re.findall(
        r"EIATTR_KPARAM_INFO[\s\S]*?Ordinal\s*:\s*0x([0-9a-f]+)"
        r"\s+Offset\s*:\s*0x([0-9a-f]+)\s+Size\s*:\s*0x([0-9a-f]+)",
        info_body,
        flags=re.IGNORECASE,
    )
    decoded = sorted(
        (int(ordinal, 16), int(offset, 16), int(size, 16))
        for ordinal, offset, size in parameters
    )
    checks["kparam_count"] = len(parameters)
    checks["kparam_layout"] = decoded
    expected_layout = [
        (ordinal, EXPECTED_KPARAM_OFFSETS[ordinal], EXPECTED_KPARAM_SIZES[ordinal])
        for ordinal in range(10)
    ]
    requested_hex = f"0x{threads:x}"
    reqntid = re.findall(
        r"EIATTR_REQNTID[\s\S]*?Value:\s+(0x[0-9a-f]+)\s+"
        r"(0x[0-9a-f]+)\s+(0x[0-9a-f]+)",
        info_body,
        flags=re.IGNORECASE,
    )
    checks["reqntid"] = (
        threads
        if reqntid == [(requested_hex, "0x1", "0x1")]
        else None
    )
    regcounts = re.findall(
        rf"EIATTR_REGCOUNT[\s\S]*?function:\s*{re.escape(symbol)}\([^)]+\)"
        r"\s+register count:\s*(\d+)\b",
        global_body,
    )
    frames = re.findall(
        rf"EIATTR_FRAME_SIZE[\s\S]*?function:\s*{re.escape(symbol)}\([^)]+\)"
        r"\s+frame size:\s*0x([0-9a-f]+)\b",
        global_body,
        flags=re.IGNORECASE,
    )
    checks["registers_per_thread"] = (
        int(regcounts[0]) if len(regcounts) == 1 else None
    )
    checks["frame_bytes"] = int(frames[0], 16) if len(frames) == 1 else None
    checks["text_registers_per_thread"] = int(text_match.group(1))
    checks["local_memory_bytes"] = int(text_match.group(2))
    checks["static_shared_bytes"] = int(text_match.group(3))
    valid = (
        checks["sm"] == 86
        and checks["elf_header_count"] == 1
        and checks["function_info_sections"] == [symbol]
        and checks["symbol_present"]
        and checks["kparam_bytes"] == 72
        and checks["kparam_count"] == 10
        and decoded == expected_layout
        and checks["reqntid"] == threads
        and checks["registers_per_thread"] == registers
        and checks["text_registers_per_thread"] == registers
        and checks["frame_bytes"] == 0
        and checks["local_memory_bytes"] == 0
        and checks["static_shared_bytes"] == 0
    )
    if not valid:
        raise EvidenceError(f"cuobjdump inspection failed for {symbol}: {checks}")
    checks["passed"] = True
    return checks


def split_process_streams(
    process_stdout: str, process_stderr: str
) -> tuple[str, str, dict[str, Any]]:
    json_lines: list[str] = []
    diagnostic_stdout: list[str] = []
    for line in process_stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError:
                diagnostic_stdout.append(line)
                continue
            if not isinstance(value, dict):
                diagnostic_stdout.append(line)
                continue
            json_lines.append(stripped)
        else:
            diagnostic_stdout.append(line)
    if len(json_lines) != 1:
        raise EvidenceError(
            f"process streams contain {len(json_lines)} harness JSON objects, expected one"
        )
    diagnostics = [*diagnostic_stdout, *process_stderr.splitlines()]
    diagnostics_text = "\n".join(line for line in diagnostics if line.strip()) + "\n"
    harness_text = json_lines[0] + "\n"
    return (
        harness_text,
        diagnostics_text,
        {
            "mode": "windows-compute-sanitizer-mixed-stream-v1",
            "harness_json_objects": 1,
            "diagnostic_stdout_lines": len(
                [line for line in diagnostic_stdout if line.strip()]
            ),
            "diagnostic_stderr_lines": len(
                [line for line in process_stderr.splitlines() if line.strip()]
            ),
        },
    )


def sanitizer_summary(tool: str, diagnostics: str) -> dict[str, Any]:
    lines = [line.strip() for line in diagnostics.splitlines() if line.strip()]
    header = "========= COMPUTE-SANITIZER"
    header_count = lines.count(header)
    if tool == "racecheck":
        matches = re.findall(
            r"^========= RACECHECK SUMMARY:\s*(\d+) hazards displayed "
            r"\((\d+) errors, (\d+) warnings\)$",
            diagnostics,
            flags=re.MULTILINE,
        )
        expected_lines = {
            header,
            "========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)",
        }
        passed = (
            header_count == 1
            and matches == [("0", "0", "0")]
            and len(lines) == 2
            and set(lines) == expected_lines
        )
        return {
            "kind": "racecheck",
            "header_count": header_count,
            "summary_count": len(matches),
            "hazards": int(matches[0][0]) if len(matches) == 1 else None,
            "errors": int(matches[0][1]) if len(matches) == 1 else None,
            "warnings": int(matches[0][2]) if len(matches) == 1 else None,
            "unexpected_lines": sorted(set(lines) - expected_lines),
            "passed": passed,
        }
    matches = re.findall(
        r"^========= ERROR SUMMARY:\s*(\d+) errors$", diagnostics, flags=re.MULTILINE
    )
    leak_matches = re.findall(
        r"^========= LEAK SUMMARY:\s*(\d+) bytes leaked in (\d+) allocations$",
        diagnostics,
        flags=re.MULTILINE,
    )
    expected_lines = {header, "========= ERROR SUMMARY: 0 errors"}
    if tool == "memcheck":
        expected_lines.add("========= LEAK SUMMARY: 0 bytes leaked in 0 allocations")
        leak_ok = leak_matches == [("0", "0")]
        expected_line_count = 3
    else:
        leak_ok = not leak_matches
        expected_line_count = 2
    passed = (
        header_count == 1
        and matches == ["0"]
        and leak_ok
        and len(lines) == expected_line_count
        and set(lines) == expected_lines
    )
    return {
        "kind": tool,
        "header_count": header_count,
        "summary_count": len(matches),
        "errors": int(matches[0]) if len(matches) == 1 else None,
        "leak_summary_count": len(leak_matches),
        "leaked_bytes": int(leak_matches[0][0]) if len(leak_matches) == 1 else None,
        "leaked_allocations": (
            int(leak_matches[0][1]) if len(leak_matches) == 1 else None
        ),
        "unexpected_lines": sorted(set(lines) - expected_lines),
        "passed": passed,
    }


def _exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise EvidenceError(
            f"{label} keys differ: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _validate_comparison(
    value: Any, *, finite: int, label: str
) -> None:
    if not isinstance(value, dict):
        raise EvidenceError(f"{label} is not an object")
    _exact_keys(
        value,
        {
            "max_abs",
            "max_rel",
            "max_normalized_rel",
            "rms",
            "finite",
            "non_finite",
            "bitwise_different",
        },
        label,
    )
    for name in ("max_abs", "max_rel", "max_normalized_rel", "rms"):
        number = value[name]
        if not isinstance(number, (int, float)) or not 0 <= number < float("inf"):
            raise EvidenceError(f"{label}.{name} is not finite/nonnegative")
    if (
        value["finite"] != finite
        or value["non_finite"] != 0
        or not isinstance(value["bitwise_different"], int)
        or not 0 <= value["bitwise_different"] <= finite
    ):
        raise EvidenceError(f"{label} finite/nonfinite/difference counts failed")


def _validate_cpu_oracle(value: Any, *, shape: str, finite: int) -> None:
    if not isinstance(value, dict):
        raise EvidenceError("n_tok=512 CPU oracle is missing")
    for name in (
        "wrong_grid_gpu_canary_errors",
        "wrong_grid_gpu_input_mismatches",
        "wrong_grid_gpu_padding_errors",
    ):
        if value.get(name) != 0:
            raise EvidenceError(f"CPU oracle safety counter failed: {name}")
    if shape == "k10240-m2560":
        if (
            value.get("samples") != 8
            or value.get("seam_104_samples") != 2
            or value.get("seam_208_samples") != 2
            or value.get("no_seam_samples") != 4
            or value.get("wrong_grid_gpu_launched") is not True
            or value.get("wrong_grid_bitwise_different", 0) <= 0
            or value.get("seam_minus_one_bitwise_different", 0) <= 0
            or value.get("seam_plus_one_bitwise_different", 0) <= 0
            or value.get("directed_seam_tiles") != [2, 5]
            or value.get("adjacent_no_seam_tiles") != [1, 3, 4, 6]
            or value.get("directed_block_boundaries") != [103, 104, 207, 208]
        ):
            raise EvidenceError("contraction seam/wrong-grid oracle failed")
        _validate_comparison(
            value.get("wrong_grid_gpu_vs_correct"),
            finite=finite,
            label="wrong_grid_gpu_vs_correct",
        )
        if value["wrong_grid_gpu_vs_correct"]["bitwise_different"] <= 0:
            raise EvidenceError("wrong-grid GPU mutation did not alter output")
    else:
        if (
            value.get("samples") != 4
            or value.get("no_seam_samples") != 4
            or value.get("wrong_grid_gpu_launched") is not False
        ):
            raise EvidenceError("expansion no-seam CPU oracle failed")


def _validate_case(
    case: dict[str, Any], expected: dict[str, Any], token: int
) -> None:
    shape = expected["shape"]
    dimensions = {
        "k2560-m10240": (2560, 10240),
        "k10240-m2560": (10240, 2560),
    }
    n_in, n_out = dimensions[shape]
    if (
        case.get("shape_id") != shape
        or case.get("n_in") != n_in
        or case.get("n_out") != n_out
        or case.get("n_tok") != token
        or case.get("out_stride") != (n_out if token == 512 else n_out + 17)
        or case.get("workspace_bytes") != 1966080
        or case.get("fused_epilogue") is not False
    ):
        raise EvidenceError(f"case identity failed for {shape}/n_tok={token}")
    launch = case.get("triton_launch_config", {})
    if launch != {
        "bm": expected["block_m"],
        "bn": expected["block_n"],
        "threads": expected["threads"],
        "dynamic_shared_bytes": expected["dynamic_shared_bytes"],
    }:
        raise EvidenceError("harness launch geometry does not match actual argv")
    if case.get("native_tiles") != {"rows": 128, "tokens": 128}:
        raise EvidenceError("native tile identity failed")
    route = case.get("native_route")
    if route not in ("full-tile", "physical-stream-k"):
        raise EvidenceError("unknown native route")
    if token == 512 and route != "full-tile":
        raise EvidenceError("n_tok=512 did not use exact native full-tile route")
    if token != 512 and route != "physical-stream-k":
        raise EvidenceError("tail did not use native PhysicalStreamK route")
    logical = case.get("logical_tiles")
    physical = case.get("physical_tiles")
    efficiency = case.get("efficiency")
    if (
        not isinstance(logical, int)
        or logical <= 0
        or not isinstance(physical, int)
        or physical <= 0
        or not isinstance(efficiency, int)
        or not 0 < efficiency <= 100
    ):
        raise EvidenceError("native launch dimensions are invalid")
    expected_seams = (route == "full-tile" and efficiency < 90) or (
        route == "physical-stream-k" and logical % physical != 0
    )
    if case.get("numeric_seams") is not expected_seams:
        raise EvidenceError("numeric seam evidence does not match native rule")
    expected_grid = 30 if route == "full-tile" and efficiency < 90 else (
        physical if route == "physical-stream-k" else logical
    )
    if case.get("numeric_stream_grid") != expected_grid:
        raise EvidenceError("numeric stream grid does not match native route")
    finite = token * n_out
    _validate_comparison(case.get("comparison"), finite=finite, label="comparison")
    safety_names = (
        "native_input_mismatches",
        "triton_input_mismatches",
        "native_padding_errors",
        "triton_padding_errors",
        "canary_errors",
        "post_timing_native_input_mismatches",
        "post_timing_triton_input_mismatches",
        "post_timing_native_padding_errors",
        "post_timing_triton_padding_errors",
        "post_timing_canary_errors",
    )
    if any(case.get(name) != 0 for name in safety_names):
        raise EvidenceError(f"memory/input safety failed for {shape}/n_tok={token}")
    if token != 512:
        if any(
            case.get(name) is not None
            for name in (
                "cpu_oracle",
                "timing",
                "artifact_resources",
                "post_timing_comparison",
            )
        ):
            raise EvidenceError("tail case unexpectedly contains formal evidence")
        return
    _validate_cpu_oracle(case.get("cpu_oracle"), shape=shape, finite=finite)
    timing = case.get("timing")
    if (
        not isinstance(timing, dict)
        or timing.get("clock") != "cuda_event"
        or timing.get("schedule") != "ABBA_BAAB"
        or timing.get("warmup") != 1
        or timing.get("pairs") != 1
        or timing.get("launches_per_sample") != 1
        or timing.get("samples_per_route") != 2
        or len(timing.get("native_samples_us", [])) != 2
        or len(timing.get("triton_samples_us", [])) != 2
    ):
        raise EvidenceError("n_tok=512 timing smoke contract failed")
    resources = case.get("artifact_resources")
    if not isinstance(resources, dict) or resources.get("cubin_bytes") != expected[
        "module_bytes"
    ]:
        raise EvidenceError("artifact resource/module size binding failed")
    triton_function = resources.get("triton_function", {})
    native_function = resources.get("native_function", {})
    if (
        triton_function.get("registers_per_thread")
        != expected["registers_per_thread"]
        or triton_function.get("local_bytes") != 0
        or triton_function.get("binary_version") != 86
        or triton_function.get("launch_threads") != expected["threads"]
        or triton_function.get("launch_dynamic_shared_bytes")
        != expected["dynamic_shared_bytes"]
        or triton_function.get("block_m") != expected["block_m"]
        or triton_function.get("block_n") != expected["block_n"]
        or native_function.get("local_bytes") != 0
        or native_function.get("binary_version") != 86
        or native_function.get("launch_threads") != 256
    ):
        raise EvidenceError("runtime function resources do not bind launch/module")
    _validate_comparison(
        case.get("post_timing_comparison"),
        finite=finite,
        label="post_timing_comparison",
    )


def validate_harness_stdout(stdout: str, pair: dict[str, Any]) -> dict[str, Any]:
    lines = stdout.splitlines()
    if len(lines) != 1 or not lines[0].strip():
        raise EvidenceError("harness stdout must contain exactly one JSON line")
    data = json.loads(lines[0])
    _exact_keys(
        data,
        {
            "schema",
            "phase",
            "milestone",
            "production_enabled",
            "target_sm",
            "device",
            "same_primary_context",
            "same_stream",
            "visible_arguments",
            "hidden_arguments",
            "tail_tokens",
            "measurement_request",
            "policy_admissible",
            "metadata_kparam_preflight",
            "debug_exact_triton_native_required",
            "debug_exact_triton_native_observed",
            "debug_exact_triton_native_pass",
            "direct_debug_exit_requires_exact",
            "formal_pass",
            "cases",
            "structural_ok",
        },
        "harness top-level",
    )
    measurement = data.get("measurement_request")
    if (
        data.get("schema") != 1
        or data.get("phase") != "A2"
        or data.get("milestone") != "C0"
        or data.get("production_enabled") is not False
        or data.get("target_sm") != 86
        or data.get("same_primary_context") is not True
        or data.get("same_stream") is not True
        or data.get("visible_arguments") != 8
        or data.get("hidden_arguments") != 2
        or data.get("tail_tokens") != list(TAIL_TOKENS)
        or measurement
        != {
            "warmup": 1,
            "pairs": 1,
            "launches_per_sample": 1,
            "samples_per_route": 2,
            "formal_contract": False,
        }
        or data.get("policy_admissible") is not False
        or data.get("metadata_kparam_preflight") is not False
        or data.get("debug_exact_triton_native_required") is not False
        or data.get("debug_exact_triton_native_observed") is not True
        or data.get("direct_debug_exit_requires_exact") is not False
        or data.get("formal_pass") is not False
        or data.get("structural_ok") is not True
        or len(data.get("cases", [])) != 12
    ):
        raise EvidenceError("harness structural identity failed")
    device = data.get("device")
    if (
        not isinstance(device, dict)
        or set(device)
        != {
            "uuid",
            "name",
            "sm",
            "sm_count",
            "driver_version",
            "cuda_runtime_version",
        }
        or not isinstance(device["uuid"], str)
        or not (
            device["uuid"].startswith("GPU-")
            or re.fullmatch(r"[0-9a-fA-F]{32}", device["uuid"])
        )
        or not isinstance(device["name"], str)
        or not device["name"]
        or device["sm"] != 86
        or not isinstance(device["sm_count"], int)
        or device["sm_count"] <= 0
        or not isinstance(device["driver_version"], int)
        or device["driver_version"] <= 0
        or not isinstance(device["cuda_runtime_version"], int)
        or device["cuda_runtime_version"] <= 0
    ):
        raise EvidenceError("harness device identity is incomplete")
    expected_cases = [
        (variant, token)
        for variant in (pair["expansion"], pair["contraction"])
        for token in TAIL_TOKENS
    ]
    actual_identity = [
        (case.get("shape_id"), case.get("n_tok")) for case in data["cases"]
    ]
    expected_identity = [
        (variant["shape"], token) for variant, token in expected_cases
    ]
    if actual_identity != expected_identity or len(set(actual_identity)) != 12:
        raise EvidenceError("harness cases are not the exact ordered 2x6 Cartesian")
    for case, (expected, token) in zip(data["cases"], expected_cases):
        _validate_case(case, expected, token)
    return {
        "passed": True,
        "device": device,
        "case_count": len(data["cases"]),
        "case_identity": [
            {"shape": shape, "n_tok": token} for shape, token in actual_identity
        ],
        "measurement_request": measurement,
        "structural_ok": True,
    }


def exact_run_ids(pairs: list[dict[str, Any]]) -> list[str]:
    values = [f"{pair['label']}--{tool}" for pair in pairs for tool in TOOLS]
    if len(values) != 16 or len(set(values)) != 16:
        raise EvidenceError("run Cartesian product is not exactly 4x4")
    return values


def harness_args(pair: dict[str, Any]) -> list[str]:
    expansion = pair["expansion"]
    contraction = pair["contraction"]
    return [
        expansion["module"],
        expansion["symbol"],
        contraction["module"],
        contraction["symbol"],
        "1",
        "1",
        "1",
        str(expansion["block_m"]),
        str(expansion["block_n"]),
        str(expansion["threads"]),
        str(expansion["dynamic_shared_bytes"]),
        str(contraction["block_m"]),
        str(contraction["block_n"]),
        str(contraction["threads"]),
        str(contraction["dynamic_shared_bytes"]),
    ]


def gpu_snapshot(nvidia_smi: pathlib.Path) -> dict[str, Any]:
    fields = (
        "timestamp,uuid,name,driver_version,temperature.gpu,"
        "clocks.current.graphics,clocks.current.sm,clocks.current.memory,"
        "pstate,power.draw,utilization.gpu,utilization.memory,memory.used,memory.free"
    )
    completed = subprocess.run(
        [
            str(nvidia_smi),
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or len(completed.stdout.splitlines()) != 1:
        raise EvidenceError("nvidia-smi device snapshot failed")
    applications = subprocess.run(
        [
            str(nvidia_smi),
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if applications.returncode != 0:
        raise EvidenceError("nvidia-smi compute application snapshot failed")
    return {
        "captured_utc": utc_now(),
        "gpu_csv": completed.stdout.strip(),
        "compute_apps_csv": applications.stdout.strip().splitlines(),
    }


def write_partial(path: pathlib.Path, state: dict[str, Any]) -> None:
    partial = dict(state)
    partial["all_pass"] = False
    partial["complete"] = False
    atomic_json(path, partial)


def write_final(
    path: pathlib.Path, state: dict[str, Any], expected_run_ids: list[str]
) -> None:
    actual_run_ids = [record.get("run_id") for record in state.get("runs", [])]
    if (
        len(actual_run_ids) != 16
        or actual_run_ids != expected_run_ids
        or len(set(actual_run_ids)) != 16
        or len(state.get("modules", [])) != 8
        or not all(record.get("pass") is True for record in state["runs"])
    ):
        raise EvidenceError("refusing to create final manifest from incomplete/failed state")
    final = dict(state)
    final["complete"] = True
    final["all_pass"] = True
    atomic_json(path, final)


def run_matrix(args: argparse.Namespace) -> int:
    policy_path = args.policy.resolve()
    metadata_path = args.metadata.resolve()
    source = args.source.resolve()
    executable = args.executable.resolve()
    runner = pathlib.Path(__file__).resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise EvidenceError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    partial_path = output / "partial-manifest.json"
    final_path = output / "sanitizer-evidence-v2.json"
    policy, metadata = load_inputs(policy_path, metadata_path)
    del policy
    require_hash(executable, EXPECTED_EXE_SHA256, "fixed harness executable")
    require_hash(source, EXPECTED_SOURCE_SHA256, "harness source")
    pairs = build_pairs(metadata, metadata_path)
    run_ids = exact_run_ids(pairs)
    runner_hash = sha256(runner)
    sanitizer_identity = tool_identity(
        args.compute_sanitizer.resolve(), ["--version"], "Compute Sanitizer"
    )
    cuobjdump_identity = tool_identity(
        args.cuobjdump.resolve(), ["--version"], "cuobjdump"
    )
    nvidia_smi_identity = tool_identity(
        args.nvidia_smi.resolve(), ["--version"], "NVIDIA-SMI"
    )
    environment = os.environ.copy()
    environment["IMPARO_CUDA_KERNEL_LAB"] = "1"
    cuda_bin = str(args.compute_sanitizer.parent.parent / "bin")
    environment["PATH"] = cuda_bin + os.pathsep + environment.get("PATH", "")
    state: dict[str, Any] = {
        "schema": 2,
        "phase": "A2",
        "milestone": "q4-variant-sanitizer-matrix-v2",
        "production_enabled": False,
        "all_pass": False,
        "complete": False,
        "expected_run_ids": run_ids,
        "execution_order": {"labels": list(LABELS), "tools": list(TOOLS)},
        "identity": {
            "policy": str(policy_path),
            "policy_sha256": sha256(policy_path),
            "metadata": str(metadata_path),
            "metadata_sha256": sha256(metadata_path),
            "harness_source": str(source),
            "harness_source_sha256": sha256(source),
            "harness_executable": str(executable),
            "harness_executable_sha256": sha256(executable),
            "runner": str(runner),
            "runner_sha256": runner_hash,
            "compute_sanitizer": sanitizer_identity,
            "cuobjdump": cuobjdump_identity,
            "nvidia_smi": nvidia_smi_identity,
            "relevant_environment": relevant_environment(environment),
        },
        "device_snapshots": {"before": gpu_snapshot(args.nvidia_smi), "after": None},
        "modules": [],
        "runs": [],
    }
    write_partial(partial_path, state)
    preflight_dir = output / "preflight"
    preflight_dir.mkdir()
    seen_preflight_paths: set[pathlib.Path] = set()
    for pair in pairs:
        for variant in (pair["expansion"], pair["contraction"]):
            raw_path = (
                preflight_dir / f"{pair['label']}--{variant['shape']}.cuobjdump.txt"
            )
            if raw_path in seen_preflight_paths or raw_path.exists():
                raise EvidenceError(f"duplicate preflight path: {raw_path}")
            seen_preflight_paths.add(raw_path)
            completed = subprocess.run(
                [str(args.cuobjdump), "-elf", variant["module"]],
                check=False,
                capture_output=True,
                text=True,
            )
            raw_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
            if completed.returncode != 0:
                raise EvidenceError(f"cuobjdump failed for {variant['symbol']}")
            inspection = parse_cuobjdump(
                completed.stdout,
                symbol=variant["symbol"],
                threads=variant["threads"],
                registers=variant["registers_per_thread"],
            )
            state["modules"].append(
                {
                    **variant,
                    "cuobjdump_argv": [
                        str(args.cuobjdump),
                        "-elf",
                        variant["module"],
                    ],
                    "cuobjdump_raw": str(raw_path),
                    "cuobjdump_raw_sha256": sha256(raw_path),
                    "module_stat_bytes": pathlib.Path(variant["module"]).stat().st_size,
                    "cuobjdump_identity_sha256": cuobjdump_identity["sha256"],
                    "inspection": inspection,
                }
            )
            write_partial(partial_path, state)
    seen_run_paths: set[pathlib.Path] = set()
    run_index = 0
    for pair in pairs:
        for tool in TOOLS:
            run_id = f"{pair['label']}--{tool}"
            if run_id != run_ids[run_index]:
                raise EvidenceError("run order diverged from exact Cartesian identity")
            run_dir = output / "runs" / f"{run_index + 1:02d}--{run_id}"
            if run_dir in seen_run_paths or run_dir.exists():
                raise EvidenceError(f"duplicate run path: {run_dir}")
            seen_run_paths.add(run_dir)
            run_dir.mkdir(parents=True)
            actual_argv = [
                str(args.compute_sanitizer),
                "--tool",
                tool,
                "--error-exitcode",
                "86",
                *(("--leak-check", "full") if tool == "memcheck" else ()),
                str(executable),
                *harness_args(pair),
            ]
            started = utc_now()
            before = gpu_snapshot(args.nvidia_smi)
            completed = subprocess.run(
                actual_argv,
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            ended = utc_now()
            after = gpu_snapshot(args.nvidia_smi)
            raw_stdout_path = run_dir / "process-stdout.raw"
            raw_stderr_path = run_dir / "process-stderr.raw"
            stdout_path = run_dir / "stdout.json"
            stderr_path = run_dir / "stderr.log"
            exit_path = run_dir / "exit-code.txt"
            raw_stdout_path.write_text(completed.stdout, encoding="utf-8")
            raw_stderr_path.write_text(completed.stderr, encoding="utf-8")
            try:
                harness_stdout, diagnostics, normalization = split_process_streams(
                    completed.stdout, completed.stderr
                )
            except EvidenceError as error:
                harness_stdout = ""
                diagnostics = completed.stdout + completed.stderr
                normalization = {
                    "mode": "windows-compute-sanitizer-mixed-stream-v1",
                    "passed": False,
                    "error": str(error),
                }
            else:
                normalization["passed"] = True
            stdout_path.write_text(harness_stdout, encoding="utf-8")
            stderr_path.write_text(diagnostics, encoding="utf-8")
            exit_path.write_text(f"{completed.returncode}\n", encoding="utf-8")
            summary = sanitizer_summary(tool, diagnostics)
            try:
                harness = validate_harness_stdout(harness_stdout, pair)
            except (EvidenceError, json.JSONDecodeError) as error:
                harness = {"passed": False, "error": str(error)}
            passed = (
                completed.returncode == 0
                and summary["passed"]
                and harness["passed"]
            )
            command = {
                "schema": 1,
                "run_id": run_id,
                "actual_argv": actual_argv,
                "runner": str(runner),
                "runner_sha256": runner_hash,
                "compute_sanitizer": sanitizer_identity,
                "relevant_environment": relevant_environment(environment),
                "capture_normalization": normalization,
                "process_stdout_raw": str(raw_stdout_path),
                "process_stdout_raw_sha256": sha256(raw_stdout_path),
                "process_stderr_raw": str(raw_stderr_path),
                "process_stderr_raw_sha256": sha256(raw_stderr_path),
                "started_utc": started,
                "ended_utc": ended,
                "exit_code": completed.returncode,
            }
            command_path = run_dir / "command.json"
            atomic_json(command_path, command)
            record = {
                "run_id": run_id,
                "label": pair["label"],
                "tool": tool,
                "directory": str(run_dir),
                "command": str(command_path),
                "command_sha256": sha256(command_path),
                "actual_argv": actual_argv,
                "started_utc": started,
                "ended_utc": ended,
                "exit_code": completed.returncode,
                "stdout": str(stdout_path),
                "stdout_sha256": sha256(stdout_path),
                "stderr": str(stderr_path),
                "stderr_sha256": sha256(stderr_path),
                "process_stdout_raw": str(raw_stdout_path),
                "process_stdout_raw_sha256": sha256(raw_stdout_path),
                "process_stderr_raw": str(raw_stderr_path),
                "process_stderr_raw_sha256": sha256(raw_stderr_path),
                "capture_normalization": normalization,
                "exit_code_file": str(exit_path),
                "exit_code_file_sha256": sha256(exit_path),
                "device_before": before,
                "device_after": after,
                "summary": summary,
                "harness": harness,
                "pass": passed,
            }
            state["runs"].append(record)
            write_partial(partial_path, state)
            print(
                f"{run_index + 1:02d}/16 {run_id} exit={completed.returncode} "
                f"summary={summary['passed']} harness={harness['passed']} pass={passed}",
                flush=True,
            )
            run_index += 1
    state["device_snapshots"]["after"] = gpu_snapshot(args.nvidia_smi)
    if len(state["runs"]) != 16 or [r["run_id"] for r in state["runs"]] != run_ids:
        write_partial(partial_path, state)
        raise EvidenceError("matrix did not produce exactly sixteen runs")
    if not all(record["pass"] for record in state["runs"]):
        write_partial(partial_path, state)
        raise EvidenceError("matrix contains failed evidence; final manifest withheld")
    write_final(final_path, state, run_ids)
    print(f"manifest={final_path}", flush=True)
    print(f"manifest_sha256={sha256(final_path)}", flush=True)
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    discovered = discover_tool_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy", type=pathlib.Path, default=ROOT / "config/kernel-lab-policy.toml"
    )
    parser.add_argument(
        "--metadata",
        type=pathlib.Path,
        default=ROOT
        / "program-packs/lab/q4-q8-mmq-sm86/variants-b/lab-metadata.json",
    )
    parser.add_argument(
        "--source",
        type=pathlib.Path,
        default=ROOT / "crates/imparo-cuda/native/tests/kernel_lab_q4_q8.cu",
    )
    parser.add_argument(
        "--executable",
        type=pathlib.Path,
        default=ROOT / "artifacts/kernel-lab/q4-c0/kernel_lab_q4_q8_sm86.exe",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=ROOT / "artifacts/kernel-lab/q4-variants-sanitizer/variants-c",
    )
    parser.add_argument(
        "--compute-sanitizer",
        type=pathlib.Path,
        default=discovered["compute_sanitizer"],
    )
    parser.add_argument(
        "--cuobjdump",
        type=pathlib.Path,
        default=discovered["cuobjdump"],
    )
    parser.add_argument(
        "--nvidia-smi",
        type=pathlib.Path,
        default=discovered["nvidia_smi"],
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    for executable in (
        args.compute_sanitizer,
        args.cuobjdump,
        args.nvidia_smi,
        args.executable,
    ):
        if executable is None or not executable.is_file():
            raise EvidenceError(f"required executable missing: {executable}")
    return run_matrix(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except EvidenceError as error:
        print(f"q4 sanitizer v2: {error}", file=sys.stderr)
        raise SystemExit(2)
