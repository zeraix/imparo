#!/usr/bin/env python3
"""Build direct, non-production LAB-A cubins for exact SM86."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import struct
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACK_TOOL = ROOT / "tools" / "triton-pack"
sys.path.insert(0, str(PACK_TOOL))

from imparo_triton_pack.aot import AotSpec, TritonCompiler  # noqa: E402
from imparo_triton_pack.errors import BuildError  # noqa: E402


TARGET = "cuda:86:32"
WIDTH = 2560
N_TOK = 512
SIGNATURE = ("*fp32:16", "*fp32:16", "*fp32:16", "*i8:16", "*fp32:16", "fp32")
VARIANTS = (
    ("w4-s1", "ip_1f7730341ef8debbde72f83b24bc35b271fa3422b4bf10c8b7de43dded918834", 4),
    ("w8-s1", "ip_fd28e32eb73acb9f96ea19a08c73529e5cdeea92d60f3ec29ac80a43a0d0efcc", 8),
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_lab_cubin(cubin: bytes, expected_sm: int) -> None:
    """Verify executable target identity without applying production pack policy."""
    if len(cubin) < 64 or cubin[:6] != b"\x7fELF\x02\x01":
        raise BuildError("LAB module is not little-endian ELF64")
    if cubin[7] != 51:
        raise BuildError("LAB module does not use NVIDIA CUDA ELF OSABI")
    if struct.unpack_from("<H", cubin, 18)[0] != 190:
        raise BuildError("LAB module ELF machine is not EM_CUDA")
    actual_sm = struct.unpack_from("<I", cubin, 48)[0] & 0xFF
    if actual_sm != expected_sm:
        raise BuildError(f"LAB module real-SM {actual_sm} is not SM{expected_sm}")


def build(source: Path, output: Path, builder_image_id: str) -> dict[str, object]:
    if "torch" in sys.modules:
        raise BuildError("LAB-A builder process already imported torch")
    if output.exists() and any(output.iterdir()):
        raise BuildError(f"refusing non-empty output directory: {output}")
    modules = output / "modules"
    modules.mkdir(parents=True, exist_ok=True)
    compiler = TritonCompiler(expected_version="3.8.0")
    built: list[dict[str, object]] = []
    for variant_id, symbol, warps in VARIANTS:
        spec = AotSpec(
            kernel_name=symbol,
            signature=SIGNATURE,
            num_warps=warps,
            num_stages=1,
            manifest={},
        )
        started = time.perf_counter()
        result = compiler.compile(source, spec, TARGET)
        elapsed = time.perf_counter() - started
        if result.global_scratch_bytes or result.profile_scratch_bytes:
            raise BuildError(f"{variant_id} uses hidden scratch")
        verify_lab_cubin(result.cubin, expected_sm=86)
        module_name = f"{symbol}.cubin"
        (modules / module_name).write_bytes(result.cubin)
        built.append(
            {
                "variant_id": variant_id,
                "symbol": result.symbol,
                "module": f"modules/{module_name}",
                "module_sha256": sha256(result.cubin),
                "module_bytes": len(result.cubin),
                "compile_seconds": elapsed,
                "num_warps": warps,
                "num_stages": 1,
                "grid": [N_TOK, 1, 1],
                "block": [warps * 32, 1, 1],
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
        raise BuildError("LAB-A compilation imported torch")
    metadata: dict[str, object] = {
        "schema": 1,
        "phase": "A2",
        "production_enabled": False,
        "candidate": "rms_norm_q8_1_mmq",
        "target": TARGET,
        "shape": {"width": WIDTH, "n_tok": N_TOK},
        "source": {"path": source.name, "sha256": sha256(source.read_bytes())},
        "builder": {
            "image_identity_kind": "local_image_store_digest",
            "image_id": builder_image_id,
            "python": platform.python_version(),
            "triton": "3.8.0-source-pin",
            "torch": "absent",
        },
        "launch_abi": [
            {"ordinal": 0, "name": "src", "type": "*fp32", "alignment": 16},
            {"ordinal": 1, "name": "dst", "type": "*fp32", "alignment": 16},
            {"ordinal": 2, "name": "mul", "type": "*fp32", "alignment": 16},
            {"ordinal": 3, "name": "q8_bytes", "type": "*i8", "alignment": 16},
            {"ordinal": 4, "name": "q8_scales", "type": "*fp32", "alignment": 16, "alias": "q8_bytes+128"},
            {"ordinal": 5, "name": "eps", "type": "fp32"},
        ],
        "hidden_launch_abi": [
            {
                "ordinal": 6,
                "name": "global_scratch",
                "type": "device_pointer",
                "allocation_bytes": 0,
                "argument_value": "null",
                "argument_required": True,
            },
            {
                "ordinal": 7,
                "name": "profile_scratch",
                "type": "device_pointer",
                "allocation_bytes": 0,
                "argument_value": "null",
                "argument_required": True,
            },
        ],
        "q8_layout": {
            "record_bytes": 144,
            "record_index": "(block32/4)*n_tok+tok",
            "qs_offset": 0,
            "qs_bytes": 128,
            "scales_offset": 128,
            "scales": 4,
        },
        "variants": built,
    }
    encoded = (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8")
    (output / "lab-metadata.json").write_bytes(encoded)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).with_name("rms_q8.py"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--builder-image-id", required=True)
    args = parser.parse_args()
    try:
        metadata = build(args.source.resolve(), args.out.resolve(), args.builder_image_id)
    except (BuildError, OSError, ValueError) as error:
        print(f"triton-kernel-lab: {error}", file=sys.stderr)
        return 2
    print(json.dumps(metadata, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
