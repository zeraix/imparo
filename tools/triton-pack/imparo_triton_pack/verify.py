from __future__ import annotations

import base64
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .common import (
    exact_keys,
    file_sha256,
    load_json,
    require_int,
    require_sha256,
    require_string,
    sha256,
)
from .elf import verify_release_cubin
from .errors import BuildError
from .signing import signature_message, validate_key_id

BASE_FILES = {
    "manifest.json",
    "SBOM.spdx.json",
    "THIRD_PARTY_NOTICES",
    "provenance.json",
}


def _walk(root: Path) -> set[str]:
    files: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise BuildError(f"pack contains symlink: {path.relative_to(root)}")
        if path.is_file():
            files.add(path.relative_to(root).as_posix())
    return files


def _signature_envelope(path: Path, manifest: bytes) -> bytes:
    envelope = load_json(path)
    exact_keys(
        envelope,
        {
            "schema",
            "domain",
            "algorithm",
            "key_id",
            "manifest_bytes",
            "manifest_sha256",
            "signature",
        },
        "manifest.sig",
    )
    if require_int(envelope["schema"], "manifest.sig.schema", 1) != 1:
        raise BuildError("manifest.sig schema must be 1")
    if envelope["domain"] != "imparo-program-pack-v1":
        raise BuildError("manifest.sig domain mismatch")
    if envelope["algorithm"] != "ed25519":
        raise BuildError("manifest.sig algorithm mismatch")
    validate_key_id(envelope["key_id"])
    if envelope["manifest_bytes"] != len(manifest):
        raise BuildError("manifest.sig length mismatch")
    if require_sha256(
        envelope["manifest_sha256"], "manifest.sig.manifest_sha256"
    ) != sha256(manifest):
        raise BuildError("manifest.sig digest mismatch")
    signature = require_string(envelope["signature"], "manifest.sig.signature")
    try:
        raw = base64.urlsafe_b64decode(signature + "=" * (-len(signature) % 4))
    except ValueError as error:
        raise BuildError("manifest.sig signature is not base64url") from error
    if len(raw) != 64 or base64.urlsafe_b64encode(raw).rstrip(b"=").decode() != signature:
        raise BuildError("manifest.sig signature is not canonical Ed25519 base64url")
    ordered = {
        "schema": envelope["schema"],
        "domain": envelope["domain"],
        "algorithm": envelope["algorithm"],
        "key_id": envelope["key_id"],
        "manifest_bytes": envelope["manifest_bytes"],
        "manifest_sha256": envelope["manifest_sha256"],
        "signature": envelope["signature"],
    }
    expected = json.dumps(
        ordered,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("ascii")
    if path.read_bytes() != expected:
        raise BuildError("manifest.sig is not canonical JSON")
    return raw


def _verify_ed25519(manifest: bytes, signature: bytes, public_key: Path) -> None:
    try:
        public_key = public_key.resolve(strict=True)
    except OSError as error:
        raise BuildError(f"read Ed25519 public key: {error}") from error
    with tempfile.TemporaryDirectory(prefix="imparo-signature-") as directory:
        signature_path = Path(directory) / "signature.bin"
        signature_path.write_bytes(signature)
        try:
            process = subprocess.run(
                [
                    "openssl",
                    "pkeyutl",
                    "-verify",
                    "-pubin",
                    "-rawin",
                    "-inkey",
                    str(public_key),
                    "-sigfile",
                    str(signature_path),
                ],
                input=signature_message(manifest),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        except OSError as error:
            raise BuildError(f"launch openssl: {error}") from error
    if process.returncode != 0:
        raise BuildError("Ed25519 Program Pack signature verification failed")


def verify_pack(
    manifest_path: Path,
    require_signature: bool = False,
    public_key: Path | None = None,
    allow_feasibility_smoke: bool = False,
) -> dict[str, Any]:
    if public_key is not None and not require_signature:
        raise BuildError("--public-key requires --require-signature")
    if require_signature and public_key is None:
        raise BuildError("--require-signature requires an Ed25519 --public-key")
    manifest_path = manifest_path.resolve(strict=True)
    if manifest_path.name != "manifest.json":
        raise BuildError("verifier input must be manifest.json")
    root = manifest_path.parent
    manifest_bytes = manifest_path.read_bytes()
    manifest = load_json(manifest_path)
    target = manifest.get("target")
    if not isinstance(target, dict):
        raise BuildError("manifest target must be an object")
    expected_sm = require_int(target.get("sm"), "target.sm", 1)
    if expected_sm not in {80, 86}:
        raise BuildError("manifest target.sm is not supported by this builder")
    modules = manifest.get("modules")
    if not isinstance(modules, list) or not modules:
        raise BuildError("manifest modules must be a non-empty array")
    expected = set(BASE_FILES)
    for module in modules:
        if not isinstance(module, dict):
            raise BuildError("manifest module entry must be an object")
        exact_keys(module, {"id", "file", "format", "bytes", "sha256"}, "module")
        module_path = require_string(module["file"], "module.file")
        if not module_path.startswith("modules/") or not module_path.endswith(".cubin"):
            raise BuildError("module path is not a data-only cubin path")
        if Path(module_path).as_posix() != module_path or ".." in Path(module_path).parts:
            raise BuildError("module path is not canonical")
        expected.add(module_path)
        path = root / module_path
        if not path.is_file() or path.is_symlink():
            raise BuildError(f"missing regular module {module_path}")
        actual = path.read_bytes()
        digest = file_sha256(path)
        if digest != require_sha256(module["sha256"], "module.sha256"):
            raise BuildError(f"module digest mismatch: {module_path}")
        if module["id"] != digest or Path(module_path).stem != digest:
            raise BuildError("module identity is not content-addressed")
        if module["format"] != "cubin" or module["bytes"] != len(actual):
            raise BuildError("module format/size mismatch")
        verify_release_cubin(actual, expected_sm)
    sidecars = {
        "THIRD_PARTY_NOTICES": "notices_sha256",
        "SBOM.spdx.json": "sbom_sha256",
        "provenance.json": "provenance_sha256",
    }
    for filename, field in sidecars.items():
        if file_sha256(root / filename) != require_sha256(manifest.get(field), field):
            raise BuildError(f"{filename} digest mismatch")
    provenance = load_json(root / "provenance.json")
    status = provenance.get("toolchain_status")
    feasibility = provenance.get("feasibility_only")
    if status not in {"source_pin_pre_release", "release_ready"}:
        raise BuildError("provenance toolchain_status is invalid")
    if not isinstance(feasibility, bool) or feasibility != (
        status == "source_pin_pre_release"
    ):
        raise BuildError("provenance feasibility boundary is inconsistent")
    if feasibility and not allow_feasibility_smoke:
        raise BuildError("feasibility-only pack is not admissible for release")
    builder = provenance.get("builder")
    toolchain = manifest.get("toolchain")
    if not isinstance(builder, dict) or not isinstance(toolchain, dict):
        raise BuildError("manifest/provenance toolchain identity is absent")
    if builder.get("image_sha256") != toolchain.get("builder_image_sha256"):
        raise BuildError("manifest/provenance builder image identities differ")
    signed = (root / "manifest.sig").is_file()
    if signed:
        expected.add("manifest.sig")
        signature = _signature_envelope(root / "manifest.sig", manifest_bytes)
        if public_key is not None:
            _verify_ed25519(manifest_bytes, signature, public_key)
    elif require_signature:
        raise BuildError("signed final pack requires manifest.sig")
    actual_files = _walk(root)
    if actual_files != expected:
        raise BuildError(
            f"pack inventory differs: missing={sorted(expected-actual_files)}, "
            f"extra={sorted(actual_files-expected)}"
        )
    for forbidden in (".py", ".ptx", ".ttir", ".ttgir", ".ll", ".bc", ".dll", ".so"):
        if any(path.lower().endswith(forbidden) for path in actual_files):
            raise BuildError(f"pack inventory contains forbidden {forbidden} content")
    return {
        "manifest_sha256": sha256(manifest_bytes),
        "module_count": len(modules),
        "signed": signed,
        "feasibility_only": feasibility,
    }
