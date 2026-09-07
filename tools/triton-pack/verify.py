#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from imparo_triton_pack.errors import BuildError
from imparo_triton_pack.verify import verify_pack


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a data-only Program Pack")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--require-signature", action="store_true")
    parser.add_argument("--public-key", type=Path)
    parser.add_argument("--allow-feasibility-smoke", action="store_true")
    args = parser.parse_args()
    try:
        result = verify_pack(
            args.manifest,
            require_signature=args.require_signature,
            public_key=args.public_key,
            allow_feasibility_smoke=args.allow_feasibility_smoke,
        )
    except (BuildError, OSError) as error:
        print(f"triton-pack verify: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
