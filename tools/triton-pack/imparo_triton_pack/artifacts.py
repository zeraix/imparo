from __future__ import annotations

import copy
import datetime as dt
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .aot import AotSpec, Compiler, TritonCompiler, load_aot_spec, parse_target
from .common import (
    canonical_json,
    exact_keys,
    file_sha256,
    require_int,
    require_sha256,
    require_string,
    sha256,
)
from .elf import verify_release_cubin
from .errors import BuildError
from .lock import ToolchainLock

TOP_LEVEL_KEYS = {
    "pack_id",
    "pack_version",
    "distribution",
    "release_channel",
    "required_entitlement_features",
    "engine_api",
    "backend_abi",
    "driver_min",
    "math_mode",
    "choice_group",
    "variant",
}
VARIANT_KEYS = {
    "choice_group_id",
    "contract",
    "constraints",
    "effects",
    "scratch",
    "launch",
    "graph_capture",
    "graph_update_slots",
    "numerical_class",
    "determinism",
    "bit_affecting",
    "required_entitlement_features",
    "requires",
    "conflicts",
    "provides",
    "joint_with",
}
RECIPE_FILES = (
    "Dockerfile",
    "requirements-build.lock",
    "build.py",
    "imparo_triton_pack/__init__.py",
    "imparo_triton_pack/errors.py",
    "imparo_triton_pack/common.py",
    "imparo_triton_pack/lock.py",
    "imparo_triton_pack/aot.py",
    "imparo_triton_pack/elf.py",
    "imparo_triton_pack/artifacts.py",
)


@dataclass(frozen=True)
class BuildResult:
    output: Path
    manifest_sha256: str
    module_sha256: str
    variant_id: str
    config_id: str


def _validate_template(spec: AotSpec) -> tuple[dict[str, Any], dict[str, Any]]:
    template = spec.manifest
    exact_keys(template, TOP_LEVEL_KEYS, "IMPARO_AOT.manifest")
    if template["distribution"] != "community":
        raise BuildError("public builder only accepts community distribution")
    if template["math_mode"] not in {"strict", "fast"}:
        raise BuildError("math_mode must be strict or fast")
    require_int(template["driver_min"], "driver_min")
    for name in ("pack_id", "pack_version", "release_channel"):
        require_string(template[name], name)
    choice_group = template["choice_group"]
    variant = template["variant"]
    if not isinstance(choice_group, dict) or not isinstance(variant, dict):
        raise BuildError("choice_group and variant must be objects")
    exact_keys(variant, VARIANT_KEYS, "IMPARO_AOT.manifest.variant")
    if variant.get("choice_group_id") != choice_group.get("choice_group_id"):
        raise BuildError("variant and choice_group identities differ")
    if variant.get("contract") != choice_group.get("contract"):
        raise BuildError("variant and choice_group contracts differ")
    scratch = variant.get("scratch")
    if not isinstance(scratch, dict):
        raise BuildError("variant.scratch must be an object")
    require_int(scratch.get("max_bytes"), "variant.scratch.max_bytes")
    launch = variant.get("launch")
    if not isinstance(launch, dict):
        raise BuildError("variant.launch must be an object")
    expected_block = {"x": spec.num_warps * 32, "y": 1, "z": 1}
    if launch.get("block") != expected_block:
        raise BuildError(
            "variant.launch.block must match Triton num_warps * warp_size"
        )
    if launch.get("dynamic_shared_bytes") != {"kind": "const", "value": 0}:
        raise BuildError(
            "variant.launch.dynamic_shared_bytes must be constant zero"
        )
    return choice_group, variant


def _spdx(
    lock: ToolchainLock, module_path: str, module_sha: str, source_sha: str
) -> bytes:
    created = dt.datetime.fromtimestamp(
        lock.source_date_epoch, tz=dt.timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    namespace = f"https://zeraix.com/spdx/imparo-program-pack/{module_sha}"
    value = {
        "SPDXID": "SPDXRef-DOCUMENT",
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "name": "Imparo Community Program Pack",
        "documentNamespace": namespace,
        "creationInfo": {
            "created": created,
            "creators": ["Tool: imparo-triton-pack-1"],
        },
        "packages": [
            {
                "SPDXID": "SPDXRef-Triton",
                "name": "triton",
                "versionInfo": lock.triton_version,
                "downloadLocation": lock.raw["triton"]["source_url"],
                "licenseConcluded": "MIT",
                "licenseDeclared": "MIT",
                "checksums": [
                    {
                        "algorithm": "SHA256",
                        "checksumValue": lock.raw["triton"]["source_sha256"],
                    }
                ],
                "filesAnalyzed": False,
                "externalRefs": [
                    {
                        "referenceCategory": "PACKAGE-MANAGER",
                        "referenceType": "purl",
                        "referenceLocator": (
                            "pkg:github/triton-lang/triton@"
                            f"{lock.triton_revision}"
                        ),
                    }
                ],
            },
            {
                "SPDXID": "SPDXRef-LLVM",
                "name": "llvm-project-toolchain",
                "versionInfo": lock.llvm_revision,
                "downloadLocation": "https://github.com/llvm/llvm-project",
                "licenseConcluded": "Apache-2.0 WITH LLVM-exception",
                "licenseDeclared": "Apache-2.0 WITH LLVM-exception",
                "checksums": [
                    {
                        "algorithm": "SHA256",
                        "checksumValue": lock.raw["llvm"]["ubuntu_x64_sha256"],
                    }
                ],
                "filesAnalyzed": False,
            },
            {
                "SPDXID": "SPDXRef-CUDA-Toolkit",
                "name": "NVIDIA CUDA Toolkit",
                "versionInfo": lock.cuda_toolkit,
                "downloadLocation": lock.raw["builder"]["base_image"],
                "licenseConcluded": "LicenseRef-NVIDIA-CUDA",
                "licenseDeclared": "LicenseRef-NVIDIA-CUDA",
                "checksums": [
                    {
                        "algorithm": "SHA256",
                        "checksumValue": lock.raw["builder"]["base_image_sha256"],
                    }
                ],
                "filesAnalyzed": False,
            },
            {
                "SPDXID": "SPDXRef-ptxas",
                "name": "nvidia-cuda-nvcc-cu12-ptxas",
                "versionInfo": lock.ptxas_version,
                "downloadLocation": lock.raw["cuda"]["ptxas_wheel"],
                "licenseConcluded": "LicenseRef-NVIDIA-CUDA",
                "licenseDeclared": "LicenseRef-NVIDIA-CUDA",
                "checksums": [
                    {
                        "algorithm": "SHA256",
                        "checksumValue": lock.raw["cuda"]["ptxas_wheel_sha256"],
                    }
                ],
                "filesAnalyzed": False,
            },
        ],
        "files": [
            {
                "SPDXID": "SPDXRef-Module",
                "fileName": module_path,
                "checksums": [
                    {"algorithm": "SHA256", "checksumValue": module_sha}
                ],
                "licenseConcluded": "NOASSERTION",
                "copyrightText": "NOASSERTION",
            }
        ],
        "annotations": [
            {
                "annotationType": "OTHER",
                "annotator": "Tool: imparo-triton-pack-1",
                "annotationDate": created,
                "comment": f"AOT source SHA-256: {source_sha}",
            }
        ],
        "relationships": [
            {
                "spdxElementId": "SPDXRef-DOCUMENT",
                "relationshipType": "DESCRIBES",
                "relatedSpdxElement": "SPDXRef-Module",
            },
            {
                "spdxElementId": "SPDXRef-Module",
                "relationshipType": "GENERATED_FROM",
                "relatedSpdxElement": "SPDXRef-Triton",
            },
            {
                "spdxElementId": "SPDXRef-Module",
                "relationshipType": "GENERATED_FROM",
                "relatedSpdxElement": "SPDXRef-LLVM",
            },
            {
                "spdxElementId": "SPDXRef-Module",
                "relationshipType": "GENERATED_FROM",
                "relatedSpdxElement": "SPDXRef-CUDA-Toolkit",
            },
            {
                "spdxElementId": "SPDXRef-Module",
                "relationshipType": "GENERATED_FROM",
                "relatedSpdxElement": "SPDXRef-ptxas",
            },
        ],
    }
    return canonical_json(value)


def _notices() -> bytes:
    return (
        "Imparo Community Program Pack third-party notices\n\n"
        "Triton (https://github.com/triton-lang/triton) is licensed under MIT.\n"
        "LLVM components are licensed under Apache-2.0 WITH LLVM-exception.\n"
        "CUDA toolkit components are redistributed under NVIDIA's applicable "
        "CUDA toolkit terms.\n"
    ).encode("utf-8")


def _provenance(
    lock: ToolchainLock,
    target: str,
    source_sha: str,
    recipe_sha: str,
    module_sha: str,
    symbol: str,
) -> bytes:
    value = {
        "schema": 1,
        "toolchain_status": lock.raw["status"],
        "feasibility_only": lock.raw["status"] != "release_ready",
        "builder": {
            "id": "imparo.triton-pack.aot.v1",
            "image": lock.raw["builder"]["image"],
            "image_sha256": lock.builder_image_sha256,
            "identity_kind": lock.builder_image_identity_kind,
            "recipe_sha256": recipe_sha,
        },
        "invocation": {
            "target": target,
            "source_date_epoch": lock.source_date_epoch,
        },
        "materials": {
            "source_sha256": source_sha,
            "toolchain_lock_sha256": lock.digest,
            "triton_revision": lock.triton_revision,
            "llvm_revision": lock.llvm_revision,
            "ptxas_version": lock.ptxas_version,
            "cuobjdump_version": lock.cuobjdump_version,
        },
        "subject": {
            "module_sha256": module_sha,
            "symbol": symbol,
        },
        "reproducibility": {
            "timestamps_removed": True,
            "absolute_source_paths_rejected": True,
            "clean_build_identity_required": True,
        },
    }
    return canonical_json(value)


def _manifest(
    spec: AotSpec,
    lock: ToolchainLock,
    target: str,
    cubin: bytes,
    symbol: str,
    registers_per_thread: int,
    static_shared: int,
    local_memory: int,
    notices: bytes,
    sbom: bytes,
    provenance: bytes,
    recipe_sha: str,
) -> tuple[bytes, str, str]:
    sm = parse_target(target)
    choice_group, variant_template = _validate_template(spec)
    module_sha = sha256(cubin)
    module_path = f"modules/{module_sha}.cubin"
    config_input = {
        "schema": 1,
        "target": target,
        "kernel": spec.kernel_name,
        "signature": list(spec.signature),
        "num_warps": spec.num_warps,
        "num_stages": spec.num_stages,
        "variant": variant_template,
    }
    config_id = sha256(canonical_json(config_input))
    variant_id = sha256(
        b"imparo-program-variant-v1\0"
        + bytes.fromhex(module_sha)
        + bytes.fromhex(config_id)
    )
    variant = copy.deepcopy(variant_template)
    variant.update(
        {
            "variant_id": variant_id,
            "config_id": config_id,
            "module_id": module_sha,
            "symbol": symbol,
            "resources": {
                "registers_per_thread_max": registers_per_thread,
                "static_shared_bytes_max": static_shared,
                "dynamic_shared_bytes_max": 0,
                "local_memory_bytes_max": local_memory,
                "threads_per_block_max": spec.num_warps * 32,
            },
        }
    )
    template = spec.manifest
    manifest = {
        "schema": 1,
        "program_pack_abi": 1,
        "pack_id": template["pack_id"],
        "pack_version": template["pack_version"],
        "distribution": template["distribution"],
        "release_channel": template["release_channel"],
        "required_entitlement_features": template[
            "required_entitlement_features"
        ],
        "backend": "cuda",
        "engine_api": template["engine_api"],
        "backend_abi": template["backend_abi"],
        "target": {
            "sm": sm,
            "warp_size": 32,
            "driver_min": template["driver_min"],
            "math_mode": template["math_mode"],
        },
        "required_extensions": [],
        "optional_extensions": [],
        "toolchain": {
            "builder_image_sha256": lock.builder_image_sha256,
            "build_recipe_sha256": recipe_sha,
            "producer": {
                "kind": "triton",
                "triton_revision": lock.triton_revision,
                "triton_version": lock.triton_version,
                "python": lock.python_version,
                "cuda_toolkit": lock.cuda_toolkit,
            },
        },
        "modules": [
            {
                "id": module_sha,
                "file": module_path,
                "format": "cubin",
                "bytes": len(cubin),
                "sha256": module_sha,
            }
        ],
        "choice_groups": [choice_group],
        "variants": [variant],
        "notices_sha256": sha256(notices),
        "sbom_sha256": sha256(sbom),
        "provenance_sha256": sha256(provenance),
    }
    return canonical_json(manifest), variant_id, config_id


def build_pack(
    *,
    lock: ToolchainLock,
    target: str,
    source: Path,
    output: Path,
    compiler: Compiler | None = None,
    enforce_environment: bool = True,
    feasibility_source_pin: bool = False,
    recipe_root: Path | None = None,
) -> BuildResult:
    parse_target(target)
    if enforce_environment:
        lock.require_build_environment(
            feasibility_source_pin=feasibility_source_pin
        )
    source = source.resolve(strict=True)
    if output.exists():
        raise BuildError(f"output already exists: {output}")
    spec = load_aot_spec(source)
    active_compiler = compiler or TritonCompiler(lock.triton_version)
    result = active_compiler.compile(source, spec, target)
    if not result.cubin:
        raise BuildError("compiler returned an empty cubin")
    if result.symbol != spec.kernel_name:
        raise BuildError("compiler returned an unexpected kernel symbol")
    if result.global_scratch_bytes != 0 or result.profile_scratch_bytes != 0:
        raise BuildError(
            "non-zero Triton global_scratch/profile_scratch is forbidden"
        )
    if result.static_shared_bytes < 0 or result.static_shared_bytes > 262_144:
        raise BuildError("compiler reported invalid static shared memory")
    if not 0 <= result.registers_per_thread <= 255:
        raise BuildError("compiler reported invalid register usage")
    if not 0 <= result.local_memory_bytes <= 1_073_741_824:
        raise BuildError("compiler reported invalid local memory usage")
    verify_release_cubin(
        result.cubin, parse_target(target), str(source).encode("utf-8")
    )

    root = recipe_root or Path(__file__).resolve().parents[1]
    recipe_files = [root / relative for relative in RECIPE_FILES]
    missing = [
        relative
        for relative, path in zip(RECIPE_FILES, recipe_files)
        if not path.is_file()
    ]
    if missing:
        raise BuildError(f"builder recipe is incomplete: missing={missing}")
    recipe_material = bytearray(b"imparo-triton-pack-recipe-v1\0")
    for path in recipe_files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        recipe_material.extend(len(relative).to_bytes(8, "little"))
        recipe_material.extend(relative)
        recipe_material.extend(bytes.fromhex(file_sha256(path)))
    recipe_sha = sha256(bytes(recipe_material))

    source_sha = file_sha256(source)
    module_sha = sha256(result.cubin)
    module_path = f"modules/{module_sha}.cubin"
    notices = _notices()
    sbom = _spdx(lock, module_path, module_sha, source_sha)
    provenance = _provenance(
        lock,
        target,
        source_sha,
        recipe_sha,
        module_sha,
        result.symbol,
    )
    manifest, variant_id, config_id = _manifest(
        spec,
        lock,
        target,
        result.cubin,
        result.symbol,
        result.registers_per_thread,
        result.static_shared_bytes,
        result.local_memory_bytes,
        notices,
        sbom,
        provenance,
        recipe_sha,
    )

    output_parent = output.resolve().parent
    output_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.part-", dir=output_parent)
    )
    try:
        (staging / "modules").mkdir()
        (staging / module_path).write_bytes(result.cubin)
        (staging / "SBOM.spdx.json").write_bytes(sbom)
        (staging / "THIRD_PARTY_NOTICES").write_bytes(notices)
        (staging / "provenance.json").write_bytes(provenance)
        (staging / "manifest.json").write_bytes(manifest)
        os.replace(staging, output.resolve())
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return BuildResult(
        output=output.resolve(),
        manifest_sha256=sha256(manifest),
        module_sha256=module_sha,
        variant_id=variant_id,
        config_id=config_id,
    )
