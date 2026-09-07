#!/usr/bin/env python3
"""Build the bounded Phase A2 Q4_0 x Q8_1 MMQ SM86 variant matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACK_TOOL = ROOT / "tools" / "triton-pack"
sys.path.insert(0, str(PACK_TOOL))

from imparo_triton_pack.aot import AotSpec, TritonCompiler  # noqa: E402
from imparo_triton_pack.errors import BuildError  # noqa: E402


TARGET = "cuda:86:32"
N_TOK = 512
SIGNATURE = (
    "*i8",
    "*fp16",
    "*i8:16",
    "*fp32:16",
    "*fp32:16",
    "i32",
    "i32",
    "i32",
)
IDENTITY_DOMAIN = "imparo-lab-a2"
NUMERIC_CONTRACT = "numeric-seam-v1"
CANDIDATE = "q4_0_x_q8_1_mmq_epilogue0"


@dataclass(frozen=True)
class Variant:
    label: str
    block_m: int
    block_n: int
    num_warps: int
    num_stages: int
    rationale: str


@dataclass(frozen=True)
class Shape:
    shape_id: str
    n_in: int
    n_out: int
    source: str
    symbols: tuple[str, ...]


VARIANTS = (
    Variant(
        "bm128-bn64-w8-s2", 128, 64, 8, 2,
        "halve row CTAs and repeated Q8 reads while preserving accumulator lanes per thread",
    ),
    Variant(
        "bm64-bn128-w8-s2", 64, 128, 8, 2,
        "halve token CTAs and repeated Q4 reads while preserving accumulator lanes per thread",
    ),
    Variant(
        "bm64-bn64-w8-s2", 64, 64, 8, 2,
        "isolate warp and register-pressure effects at the baseline tile",
    ),
    Variant(
        "bm64-bn64-w8-s1", 64, 64, 8, 1,
        "isolate stage, shared-memory, and register cost from the eight-warp tile",
    ),
)

SHAPES = (
    Shape(
        "k2560-m10240",
        2560,
        10240,
        "q4_q8_mmq_variants_2560x10240.py",
        (
            "ip_cd1dd7eea1c5a04fa3fca1d1dbbde58075df25baad5830bf8d1fb179d1c7d5ed",
            "ip_040845b88cef1fe24b7900e33d507b81592c31ddc35ba8516ba7151c03741bc0",
            "ip_58bab621edc125e72372075f257ab39554b74abc49ff8855e6d836bbcdfd857c",
            "ip_2f16dc07d913306ecc0325a784eab13a396850af0b5c8bc1f8c59e6b69f7198b",
        ),
    ),
    Shape(
        "k10240-m2560",
        10240,
        2560,
        "q4_q8_mmq_variants_10240x2560.py",
        (
            "ip_5ee29234f32c69540442430ee69071ca81e4fa89fc48a10d6e32654c8e3c9b32",
            "ip_05e1dac74eaa31f6f6fd679867d0c92f3ad684d640d9797119078ed66d1280d4",
            "ip_99be389d8e9794a218708e277fcaabde58f53ce81e73c9518a250726a2ded6eb",
            "ip_6b4e87dd0bfbf0dd95feac24ca12b2a4c112665dc7bd352e202cb3e84c1c264d",
        ),
    ),
)

EXPECTED_KPARAMS = (
    (0, 0x00, 8),
    (1, 0x08, 8),
    (2, 0x10, 8),
    (3, 0x18, 8),
    (4, 0x20, 8),
    (5, 0x28, 4),
    (6, 0x2C, 4),
    (7, 0x30, 4),
    (8, 0x38, 8),
    (9, 0x40, 8),
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_descriptor(shape: Shape, variant: Variant) -> str:
    return "|".join(
        (
            IDENTITY_DOMAIN,
            CANDIDATE,
            "sm86",
            shape.shape_id,
            variant.label,
            NUMERIC_CONTRACT,
        )
    )


def expected_symbol(shape: Shape, variant: Variant) -> str:
    return "ip_" + sha256(canonical_descriptor(shape, variant).encode("utf-8"))


def verify_matrix() -> None:
    if any(len(shape.symbols) != len(VARIANTS) for shape in SHAPES):
        raise BuildError("each shape must define one symbol per bounded variant")
    symbols: set[str] = set()
    for shape in SHAPES:
        for variant, symbol in zip(VARIANTS, shape.symbols):
            if symbol != expected_symbol(shape, variant):
                raise BuildError(f"opaque symbol identity mismatch for {shape.shape_id}/{variant.label}")
            if symbol in symbols:
                raise BuildError("variant matrix contains a duplicate opaque symbol")
            symbols.add(symbol)
            if (
                variant.block_m > 128
                or variant.block_n > 128
                or 128 % variant.block_m
                or 128 % variant.block_n
                or shape.n_out % variant.block_m
            ):
                raise BuildError(f"variant crosses a native numerical tile: {variant.label}")


def verify_lab_cubin(cubin: bytes, expected_sm: int) -> None:
    if len(cubin) < 64 or cubin[:6] != b"\x7fELF\x02\x01":
        raise BuildError("LAB module is not little-endian ELF64")
    if cubin[7] != 51 or struct.unpack_from("<H", cubin, 18)[0] != 190:
        raise BuildError("LAB module is not a CUDA ELF")
    actual_sm = struct.unpack_from("<I", cubin, 48)[0] & 0xFF
    if actual_sm != expected_sm:
        raise BuildError(f"LAB module real-SM {actual_sm} is not SM{expected_sm}")


def parse_lab_abi_dump(
    dump: str, symbol: str, expected_threads: int
) -> tuple[tuple[int, int, int], ...]:
    heading = re.search(rf"(?m)^\.nv\.info\.{re.escape(symbol)}\s*$", dump)
    if heading is None:
        raise BuildError("cuobjdump omitted the LAB kernel info section")
    end = re.search(r"(?m)^\.nv\.[^\r\n]+$", dump[heading.end() :])
    section_end = heading.end() + end.start() if end is not None else len(dump)
    section = dump[heading.end() : section_end]
    size = re.search(
        r"Attribute:\s*EIATTR_CBANK_PARAM_SIZE.*?Value:\s*0x([0-9a-fA-F]+)",
        section,
        re.S,
    )
    if size is None or int(size.group(1), 16) != 0x48:
        raise BuildError("LAB kernel parameter bank is not the frozen 0x48 bytes")
    params = tuple(
        sorted(
            (
                int(ordinal, 16),
                int(offset, 16),
                int(item_size, 16),
            )
            for ordinal, offset, item_size in re.findall(
                r"Ordinal\s*:\s*0x([0-9a-fA-F]+)\s+"
                r"Offset\s*:\s*0x([0-9a-fA-F]+)\s+"
                r"Size\s*:\s*0x([0-9a-fA-F]+)",
                section,
            )
        )
    )
    if params != EXPECTED_KPARAMS:
        raise BuildError(f"LAB kernel parameter ABI differs: {params!r}")
    reqntid = re.search(
        r"Attribute:\s*EIATTR_REQNTID.*?Value:\s*0x([0-9a-fA-F]+)\s+"
        r"0x([0-9a-fA-F]+)\s+0x([0-9a-fA-F]+)",
        section,
        re.S,
    )
    expected = (expected_threads, 1, 1)
    if reqntid is None or tuple(int(item, 16) for item in reqntid.groups()) != expected:
        raise BuildError(f"LAB kernel does not require the frozen {expected_threads}x1x1 block")
    return params


def verify_lab_launch_abi(cubin: bytes, symbol: str, expected_threads: int) -> None:
    tool = os.environ.get("IMPARO_CUOBJDUMP", "cuobjdump")
    with tempfile.TemporaryDirectory(prefix="imparo-q4-q8-variants-abi-") as directory:
        path = Path(directory) / "module.cubin"
        path.write_bytes(cubin)
        try:
            process = subprocess.run(
                [tool, "--dump-elf", str(path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except OSError as error:
            raise BuildError(f"launch pinned cuobjdump for ABI query: {error}") from error
    if process.returncode != 0:
        raise BuildError("pinned cuobjdump ABI query failed: " + process.stderr.strip())
    parse_lab_abi_dump(process.stdout, symbol, expected_threads)


def build(output: Path, builder_image_id: str) -> dict[str, object]:
    verify_matrix()
    if "torch" in sys.modules:
        raise BuildError("LAB builder process already imported torch")
    if output.exists() and any(output.iterdir()):
        raise BuildError(f"refusing non-empty output directory: {output}")
    modules = output / "modules"
    modules.mkdir(parents=True, exist_ok=True)
    compiler = TritonCompiler(expected_version="3.8.0")
    survivors: list[dict[str, object]] = []
    rejections: list[dict[str, object]] = []
    sources: list[dict[str, str]] = []

    for shape in SHAPES:
        source = Path(__file__).with_name(shape.source).resolve()
        source_hash = sha256(source.read_bytes())
        sources.append({"path": shape.source, "sha256": source_hash})
        for variant, symbol in zip(VARIANTS, shape.symbols):
            spec = AotSpec(
                kernel_name=symbol,
                signature=SIGNATURE,
                num_warps=variant.num_warps,
                num_stages=variant.num_stages,
                manifest={},
            )
            started = time.perf_counter()
            result = compiler.compile(source, spec, TARGET)
            elapsed = time.perf_counter() - started
            if result.global_scratch_bytes or result.profile_scratch_bytes:
                raise BuildError(
                    f"{shape.shape_id}/{variant.label} uses non-zero hidden scratch"
                )
            verify_lab_cubin(result.cubin, 86)
            expected_threads = variant.num_warps * 32
            verify_lab_launch_abi(result.cubin, result.symbol, expected_threads)
            digest = sha256(result.cubin)
            resources = {
                "registers_per_thread": result.registers_per_thread,
                "dynamic_shared_bytes": result.static_shared_bytes,
                "local_memory_bytes": result.local_memory_bytes,
                "global_scratch_bytes": result.global_scratch_bytes,
                "profile_scratch_bytes": result.profile_scratch_bytes,
            }
            common = {
                "shape_id": shape.shape_id,
                "n_in": shape.n_in,
                "n_out": shape.n_out,
                "n_tok": N_TOK,
                "config_id": "cfg_" + symbol.removeprefix("ip_"),
                "lab_label": variant.label,
                "identity_descriptor_sha256": symbol.removeprefix("ip_"),
                "symbol": result.symbol,
                "tile": {"rows": variant.block_m, "tokens": variant.block_n},
                "num_warps": variant.num_warps,
                "num_stages": variant.num_stages,
                "grid_at_evidence_n_tok": [
                    shape.n_out // variant.block_m,
                    (N_TOK + variant.block_n - 1) // variant.block_n,
                    1,
                ],
                "grid_formula": [
                    f"ceil_div(n_out,{variant.block_m})",
                    f"ceil_div(n_tok,{variant.block_n})",
                    1,
                ],
                "block": [expected_threads, 1, 1],
                "compile_seconds": elapsed,
                "module_sha256": digest,
                "module_bytes": len(result.cubin),
                "resources": resources,
                "rationale": variant.rationale,
            }
            if result.local_memory_bytes:
                rejections.append(
                    {
                        **common,
                        "rejected": True,
                        "reason": "cuobjdump reported non-zero LOCAL+STACK spill bytes",
                    }
                )
                continue
            name = f"{shape.shape_id}-{symbol}.cubin"
            (modules / name).write_bytes(result.cubin)
            survivors.append(
                {
                    **common,
                    "module": f"modules/{name}",
                    "rejected": False,
                }
            )

    if "torch" in sys.modules:
        raise BuildError("LAB compilation imported torch")
    if not survivors:
        raise BuildError("all bounded Q4-Q8 variants were rejected")

    metadata: dict[str, object] = {
        "schema": 1,
        "phase": "A2",
        "milestone": "bounded-variant-resource-screen",
        "production_enabled": False,
        "candidate": CANDIDATE,
        "target": TARGET,
        "baseline_artifacts_unchanged": [
            "program-packs/lab/q4-q8-mmq-sm86/run-a",
            "program-packs/lab/q4-q8-mmq-sm86/run-b",
        ],
        "matrix": {
            "requested_configs": len(VARIANTS),
            "requested_modules": len(VARIANTS) * len(SHAPES),
            "surviving_modules": len(survivors),
            "rejected_modules": len(rejections),
            "bounded": True,
            "no_cartesian_expansion": True,
        },
        "sources": sources,
        "builder": {
            "image_identity_kind": "local_image_store_digest",
            "image_id": builder_image_id,
            "python": platform.python_version(),
            "triton": "3.8.0-source-pin",
            "torch": "absent",
        },
        "launch_abi": [
            {"ordinal": 0, "name": "w_qs", "type": "*i8", "alignment": 2, "alias": "weights+2"},
            {"ordinal": 1, "name": "w_d", "type": "*fp16", "alignment": 2, "alias": "weights"},
            {"ordinal": 2, "name": "x_qs", "type": "*i8", "alignment": 16, "alias": "q8"},
            {"ordinal": 3, "name": "x_d", "type": "*fp32", "alignment": 16, "alias": "q8+128"},
            {"ordinal": 4, "name": "y", "type": "*fp32", "alignment": 16},
            {"ordinal": 5, "name": "n_tok", "type": "i32", "supported_range": [1, 512], "evidence_value": 512},
            {"ordinal": 6, "name": "out_stride", "type": "i32"},
            {"ordinal": 7, "name": "numeric_stream_grid", "type": "i32"},
        ],
        "hidden_launch_abi": [
            {"ordinal": 8, "name": "global_scratch", "type": "device_pointer", "allocation_bytes": 0, "argument_value": "null", "argument_required": True},
            {"ordinal": 9, "name": "profile_scratch", "type": "device_pointer", "allocation_bytes": 0, "argument_value": "null", "argument_required": True},
        ],
        "numeric_contract": {
            "revision": NUMERIC_CONTRACT,
            "native_tile": [128, 128],
            "variant_tiles_must_divide_native_tile": True,
            "seam_formula": "element-derived native tile; runtime numeric_stream_grid",
            "fold_order": "suffix+prefix when seam is non-zero",
        },
        "survivors": survivors,
        "rejections": rejections,
    }
    (output / "lab-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--builder-image-id", required=True)
    args = parser.parse_args()
    try:
        metadata = build(args.out.resolve(), args.builder_image_id)
    except (BuildError, OSError, ValueError) as error:
        print(f"triton-kernel-lab q4-q8 variants: {error}", file=sys.stderr)
        return 2
    print(json.dumps(metadata, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
