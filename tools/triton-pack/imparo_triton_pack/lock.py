from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import (
    canonical_json,
    exact_keys,
    load_json,
    require_int,
    require_revision,
    require_sha256,
    require_string,
    sha256,
)
from .errors import BuildError


@dataclass(frozen=True)
class ToolchainLock:
    raw: dict[str, Any]
    digest: str
    python_version: str
    triton_version: str
    triton_revision: str
    cuda_toolkit: str
    ptxas_version: str
    cuobjdump_version: str
    llvm_revision: str
    builder_image_sha256: str | None
    builder_image_identity_kind: str | None
    source_date_epoch: int

    def require_build_environment(
        self, *, feasibility_source_pin: bool = False
    ) -> None:
        status = self.raw["status"]
        if status != "release_ready" and not (
            feasibility_source_pin and status == "source_pin_pre_release"
        ):
            raise BuildError(
                "toolchain lock is not release-ready; Program Pack build is disabled"
            )
        actual_python = platform.python_version()
        if actual_python != self.python_version:
            raise BuildError(
                f"Python drift: lock={self.python_version}, actual={actual_python}"
            )
        if self.builder_image_sha256 is None:
            raise BuildError(
                "builder image digest is not pinned"
            )
        actual_image = os.environ.get("IMPARO_BUILDER_IMAGE_SHA256", "")
        if actual_image != self.builder_image_sha256:
            raise BuildError(
                "builder image identity is absent or differs from toolchain.lock"
            )
        actual_revision = os.environ.get("IMPARO_TRITON_SOURCE_REVISION", "")
        if actual_revision != self.triton_revision:
            raise BuildError(
                "Triton source revision is absent or differs from toolchain.lock"
            )
        actual_llvm = os.environ.get("IMPARO_TRITON_LLVM_REVISION", "")
        if actual_llvm != self.llvm_revision:
            raise BuildError(
                "Triton LLVM revision is absent or differs from toolchain.lock"
            )
        actual_ptxas = os.environ.get("IMPARO_PTXAS_VERSION", "")
        if actual_ptxas != self.ptxas_version:
            raise BuildError(
                "ptxas version is absent or differs from toolchain.lock"
            )
        actual_cuobjdump = os.environ.get("IMPARO_CUOBJDUMP_VERSION", "")
        if actual_cuobjdump != self.cuobjdump_version:
            raise BuildError(
                "cuobjdump version is absent or differs from toolchain.lock"
            )


def _object(value: Any, name: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BuildError(f"{name} must be an object")
    exact_keys(value, keys, name)
    return value


def load_toolchain_lock(path: Path) -> ToolchainLock:
    raw = load_json(path)
    exact_keys(
        raw,
        {
            "schema",
            "status",
            "source_date_epoch",
            "python",
            "triton",
            "llvm",
            "cuda",
            "builder",
            "build_requirements_sha256",
        },
        "toolchain lock",
    )
    if require_int(raw["schema"], "schema", 1) != 1:
        raise BuildError("toolchain lock schema must be 1")
    status = require_string(raw["status"], "status")
    if status not in {"source_pin_pre_release", "release_ready"}:
        raise BuildError("unknown toolchain lock status")
    source_date_epoch = require_int(
        raw["source_date_epoch"], "source_date_epoch", 1
    )
    require_sha256(
        raw["build_requirements_sha256"], "build_requirements_sha256"
    )

    python = _object(raw["python"], "python", {"version"})
    python_version = require_string(python["version"], "python.version")

    triton = _object(
        raw["triton"],
        "triton",
        {
            "version",
            "revision",
            "source_url",
            "source_bytes",
            "source_sha256",
            "official_tag",
        },
    )
    triton_version = require_string(triton["version"], "triton.version")
    triton_revision = require_revision(triton["revision"], "triton.revision")
    require_string(triton["source_url"], "triton.source_url")
    require_int(triton["source_bytes"], "triton.source_bytes", 1)
    require_sha256(triton["source_sha256"], "triton.source_sha256")
    official_tag = triton["official_tag"]
    if official_tag is not None and not isinstance(official_tag, str):
        raise BuildError("triton.official_tag must be null or a string")
    if official_tag is not None and official_tag != f"v{triton_version}":
        raise BuildError("Triton official tag does not match its version")

    llvm = _object(raw["llvm"], "llvm", {"revision", "ubuntu_x64_sha256"})
    llvm_revision = require_revision(llvm["revision"], "llvm.revision")
    require_sha256(llvm["ubuntu_x64_sha256"], "llvm.ubuntu_x64_sha256")

    cuda = _object(
        raw["cuda"],
        "cuda",
        {
            "toolkit",
            "ptxas_version",
            "cuobjdump_version",
            "ptxas_wheel",
            "ptxas_wheel_sha256",
        },
    )
    cuda_toolkit = require_string(cuda["toolkit"], "cuda.toolkit")
    ptxas_version = require_string(cuda["ptxas_version"], "cuda.ptxas_version")
    cuobjdump_version = require_string(
        cuda["cuobjdump_version"], "cuda.cuobjdump_version"
    )
    require_string(cuda["ptxas_wheel"], "cuda.ptxas_wheel")
    require_sha256(cuda["ptxas_wheel_sha256"], "cuda.ptxas_wheel_sha256")

    builder = _object(
        raw["builder"],
        "builder",
        {
            "image",
            "image_sha256",
            "image_identity_kind",
            "base_image",
            "base_image_sha256",
            "python_image",
        },
    )
    require_string(builder["image"], "builder.image")
    require_string(builder["base_image"], "builder.base_image")
    base_image = require_sha256(
        builder["base_image_sha256"], "builder.base_image_sha256"
    )
    require_string(builder["python_image"], "builder.python_image")
    image = builder["image_sha256"]
    identity_kind = builder["image_identity_kind"]
    if image is not None:
        image = require_sha256(image, "builder.image_sha256")
    if identity_kind is not None and identity_kind not in {
        "oci_config_digest_local_feasibility",
        "oci_manifest_digest",
    }:
        raise BuildError("unknown builder.image_identity_kind")
    if (image is None) != (identity_kind is None):
        raise BuildError(
            "builder image digest and identity kind must be present together"
        )
    if status == "release_ready" and image is None:
        raise BuildError("release-ready lock requires builder.image_sha256")
    if status == "release_ready" and identity_kind != "oci_manifest_digest":
        raise BuildError(
            "release-ready lock requires a pullable OCI manifest digest"
        )
    if (
        status == "source_pin_pre_release"
        and image is not None
        and identity_kind != "oci_config_digest_local_feasibility"
    ):
        raise BuildError(
            "source-pin builder identity must be a local OCI config digest"
        )
    if status == "release_ready" and image == base_image:
        raise BuildError("final builder image identity cannot equal its base image")

    return ToolchainLock(
        raw=raw,
        digest=sha256(canonical_json(raw)),
        python_version=python_version,
        triton_version=triton_version,
        triton_revision=triton_revision,
        cuda_toolkit=cuda_toolkit,
        ptxas_version=ptxas_version,
        cuobjdump_version=cuobjdump_version,
        llvm_revision=llvm_revision,
        builder_image_sha256=image,
        builder_image_identity_kind=identity_kind,
        source_date_epoch=source_date_epoch,
    )
