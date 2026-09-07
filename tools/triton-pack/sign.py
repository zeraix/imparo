#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from imparo_triton_pack.errors import BuildError
from imparo_triton_pack.signing import sign_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Sign exact Program Pack manifest bytes")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--key-id", required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    output = args.out or args.manifest.parent / "manifest.sig"
    try:
        sign_manifest(args.manifest, args.private_key, args.key_id, output)
    except (BuildError, OSError) as error:
        print(f"triton-pack sign: {error}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
