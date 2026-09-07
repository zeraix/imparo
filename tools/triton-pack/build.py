#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from imparo_triton_pack import BuildError, build_pack, load_toolchain_lock


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a pinned Triton AOT pack")
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--feasibility-source-pin",
        action="store_true",
        help=(
            "allow a source_pin_pre_release lock only for non-release "
            "infrastructure evidence; normal builds remain fail-closed"
        ),
    )
    args = parser.parse_args()
    try:
        result = build_pack(
            lock=load_toolchain_lock(args.lock),
            target=args.target,
            source=args.source,
            output=args.out,
            feasibility_source_pin=args.feasibility_source_pin,
        )
    except (BuildError, OSError) as error:
        print(f"triton-pack build: {error}", file=sys.stderr)
        return 2
    print(f"manifest_sha256={result.manifest_sha256}")
    print(f"module_sha256={result.module_sha256}")
    print(f"variant_id={result.variant_id}")
    print(f"config_id={result.config_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
