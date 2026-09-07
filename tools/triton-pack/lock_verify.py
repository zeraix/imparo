#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from imparo_triton_pack.common import file_sha256
from imparo_triton_pack.errors import BuildError
from imparo_triton_pack.lock import load_toolchain_lock


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify pinned AOT toolchain inputs")
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--triton-archive", type=Path)
    parser.add_argument("--ptxas-wheel", type=Path)
    parser.add_argument(
        "--builder-image-id",
        help="docker image inspect .Id (sha256:<64hex> or raw hex)",
    )
    args = parser.parse_args()
    try:
        lock = load_toolchain_lock(args.lock)
        checks: dict[str, object] = {
            "lock_sha256": lock.digest,
            "status": lock.raw["status"],
            "builder_image_sha256": lock.builder_image_sha256,
            "builder_image_identity_kind": lock.builder_image_identity_kind,
        }
        requirements = args.lock.resolve().with_name("requirements-build.lock")
        if file_sha256(requirements) != lock.raw["build_requirements_sha256"]:
            raise BuildError("requirements-build.lock digest differs from lock")
        checks["build_requirements"] = "verified"
        if args.triton_archive:
            if args.triton_archive.stat().st_size != lock.raw["triton"]["source_bytes"]:
                raise BuildError("Triton source archive size differs from lock")
            if file_sha256(args.triton_archive) != lock.raw["triton"]["source_sha256"]:
                raise BuildError("Triton source archive digest differs from lock")
            checks["triton_archive"] = "verified"
        if args.ptxas_wheel:
            if file_sha256(args.ptxas_wheel) != lock.raw["cuda"]["ptxas_wheel_sha256"]:
                raise BuildError("ptxas wheel digest differs from lock")
            checks["ptxas_wheel"] = "verified"
        if args.builder_image_id:
            actual = args.builder_image_id.removeprefix("sha256:")
            if lock.builder_image_sha256 is None:
                raise BuildError("builder image identity is absent from lock")
            if actual != lock.builder_image_sha256:
                raise BuildError("inspected builder image identity differs from lock")
            if (
                lock.builder_image_identity_kind
                != "oci_config_digest_local_feasibility"
            ):
                raise BuildError(
                    "docker image inspect .Id requires local config digest identity"
                )
            checks["builder_image"] = "verified"
    except (BuildError, OSError) as error:
        print(f"triton-pack lock verify: {error}", file=sys.stderr)
        return 2
    print(json.dumps(checks, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
