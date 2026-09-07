#!/usr/bin/env python3
"""Build the two frozen Phase A2 Q4_0 x Q8_1 MMQ SM86 cubins."""

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


@dataclass(frozen=True)
class Shape:
    shape_id: str
    n_in: int
    n_out: int
    source: str
    symbol: str


SHAPES = (
    Shape(
        "k2560-m10240-n512",
        2560,
        10240,
        "q4_q8_mmq_2560x10240.py",
        "ip_a1da19c631b940e2f64b0e46f33bb0ced69e830472eae28b4c06560d2509fda6",
    ),
    Shape(
        "k10240-m2560-n512",
        10240,
        2560,
        "q4_q8_mmq_10240x2560.py",
        "ip_951d2021de1698086f0d5a89753fb2da8448d43a17a2e01753d9cb7c81039fc6",
    ),
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_lab_cubin(cubin: bytes, expected_sm: int) -> None:
    if len(cubin) < 64 or cubin[:6] != b"\x7fELF\x02\x01":
        raise BuildError("LAB module is not little-endian ELF64")
    if cubin[7] != 51 or struct.unpack_from("<H", cubin, 18)[0] != 190:
        raise BuildError("LAB module is not a CUDA ELF")
    actual_sm = struct.unpack_from("<I", cubin, 48)[0] & 0xFF
    if actual_sm != expected_sm:
        raise BuildError(f"LAB module real-SM {actual_sm} is not SM{expected_sm}")


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


def parse_lab_abi_dump(dump: str, symbol: str) -> tuple[tuple[int, int, int], ...]:
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
    if reqntid is None or tuple(int(item, 16) for item in reqntid.groups()) != (
        128,
        1,
        1,
    ):
        raise BuildError("LAB kernel does not require the frozen 128x1x1 block")
    return params


def verify_lab_launch_abi(cubin: bytes, symbol: str) -> None:
    tool = os.environ.get("IMPARO_CUOBJDUMP", "cuobjdump")
    with tempfile.TemporaryDirectory(prefix="imparo-q4-q8-abi-") as directory:
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
    parse_lab_abi_dump(process.stdout, symbol)


def build(output: Path, builder_image_id: str) -> dict[str, object]:
    if "torch" in sys.modules:
        raise BuildError("LAB builder process already imported torch")
    if output.exists() and any(output.iterdir()):
        raise BuildError(f"refusing non-empty output directory: {output}")
    modules = output / "modules"
    modules.mkdir(parents=True, exist_ok=True)
    compiler = TritonCompiler(expected_version="3.8.0")
    variants: list[dict[str, object]] = []
    sources: list[dict[str, str]] = []
    for shape in SHAPES:
        source = Path(__file__).with_name(shape.source).resolve()
        source_hash = sha256(source.read_bytes())
        sources.append({"path": shape.source, "sha256": source_hash})
        spec = AotSpec(
            kernel_name=shape.symbol,
            signature=SIGNATURE,
            num_warps=4,
            num_stages=2,
            manifest={},
        )
        started = time.perf_counter()
        result = compiler.compile(source, spec, TARGET)
        elapsed = time.perf_counter() - started
        if result.global_scratch_bytes or result.profile_scratch_bytes:
            raise BuildError(f"{shape.shape_id} uses non-zero hidden scratch")
        verify_lab_cubin(result.cubin, 86)
        verify_lab_launch_abi(result.cubin, result.symbol)
        name = f"{shape.symbol}.cubin"
        (modules / name).write_bytes(result.cubin)
        variants.append(
            {
                "shape_id": shape.shape_id,
                "n_in": shape.n_in,
                "n_out": shape.n_out,
                "n_tok": N_TOK,
                "variant_id": "bm64-bn64-w4-s2",
                "symbol": result.symbol,
                "module": f"modules/{name}",
                "module_sha256": sha256(result.cubin),
                "module_bytes": len(result.cubin),
                "compile_seconds": elapsed,
                "tile": {"rows": 64, "tokens": 64},
                "num_warps": 4,
                "num_stages": 2,
                "grid_at_evidence_n_tok": [shape.n_out // 64, (N_TOK + 63) // 64, 1],
                "grid_formula": ["n_out/64", "ceil_div(n_tok,64)", 1],
                "block": [128, 1, 1],
                "dynamic_shared_bytes": result.static_shared_bytes,
                "resources": {
                    "registers_per_thread": result.registers_per_thread,
                    "local_memory_bytes": result.local_memory_bytes,
                    "global_scratch_bytes": result.global_scratch_bytes,
                    "profile_scratch_bytes": result.profile_scratch_bytes,
                },
            }
        )
    if "torch" in sys.modules:
        raise BuildError("LAB compilation imported torch")
    metadata: dict[str, object] = {
        "schema": 1,
        "phase": "A2",
        "production_enabled": False,
        "candidate": "q4_0_x_q8_1_mmq_epilogue0",
        "target": TARGET,
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
            {"ordinal": 7, "name": "numeric_stream_grid", "type": "i32", "contract_by_shape": {"k2560-m10240-n512": "native_tiles=ceil(n_out/128)*ceil(n_tok/128); no seam", "k10240-m2560-n512": "runtime device SM count; RTX3060 evidence=30"}},
        ],
        "hidden_launch_abi": [
            {"ordinal": 8, "name": "global_scratch", "type": "device_pointer", "allocation_bytes": 0, "argument_value": "null", "argument_required": True},
            {"ordinal": 9, "name": "profile_scratch", "type": "device_pointer", "allocation_bytes": 0, "argument_value": "null", "argument_required": True},
        ],
        "weight_layout": {
            "format": "Q4_0",
            "record_bytes": 18,
            "scale": "w_d[(row*n_blocks+block)*9] from weights+0",
            "packed_bytes": "w_qs[(row*n_blocks+block)*18+0..15] from weights+2",
            "nibble_order": "k0..15 low, k16..31 high",
            "dequant": "(q-8)*d4",
            "order": "row-major blocks",
        },
        "activation_layout": {
            "format": "BlockQ8_1Mmq",
            "record_bytes": 144,
            "record_index": "(block32/4)*n_tok+tok",
            "qs": "int8[128]@0",
            "scales": "fp32[4]@128 (DS4 half-rounded producer)",
        },
        "output_layout": "dst[token*out_stride+row]",
        "native_control_by_shape": {
            "k2560-m10240-n512": {"symbol": "imparo_sm80_mmq::q4_q8_1_full_tile<false,4>", "numeric_seams": False},
            "k10240-m2560-n512": {"symbol": "imparo_sm80_mmq::q4_q8_1_full_tile<false,4,false,true,true>", "numeric_seams": True},
        },
        "variants": variants,
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
        print(f"triton-kernel-lab q4-q8: {error}", file=sys.stderr)
        return 2
    print(json.dumps(metadata, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
