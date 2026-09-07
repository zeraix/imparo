from __future__ import annotations

import base64
import json
import re
import subprocess
from pathlib import Path

from .common import sha256
from .errors import BuildError

KEY_ID_RE = re.compile(
    r"(?=.{3,128}\Z)(?=.*[._-])[a-z0-9](?:[a-z0-9]|[._-](?=[a-z0-9]))*[a-z0-9]\Z"
)


def validate_key_id(value: object) -> str:
    if not isinstance(value, str) or KEY_ID_RE.fullmatch(value) is None:
        raise BuildError(
            "key_id must be a 3..128 byte lowercase namespaced id with "
            "non-consecutive ._- separators"
        )
    return value


def signature_message(manifest: bytes) -> bytes:
    return (
        b"imparo-program-pack-v1\0"
        + len(manifest).to_bytes(8, "little")
        + bytes.fromhex(sha256(manifest))
        + manifest
    )


def sign_manifest(
    manifest_path: Path,
    private_key: Path,
    key_id: str,
    output: Path,
) -> None:
    key_id = validate_key_id(key_id)
    manifest = manifest_path.read_bytes()
    if output.resolve().parent != manifest_path.resolve().parent:
        raise BuildError("manifest.sig must be written beside manifest.json")
    if output.exists():
        raise BuildError("refusing to overwrite existing manifest.sig")
    try:
        process = subprocess.run(
            [
                "openssl",
                "pkeyutl",
                "-sign",
                "-rawin",
                "-inkey",
                str(private_key.resolve(strict=True)),
            ],
            input=signature_message(manifest),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as error:
        raise BuildError(f"launch openssl: {error}") from error
    if process.returncode != 0:
        detail = process.stderr.decode("utf-8", errors="replace").strip()
        raise BuildError(f"openssl Ed25519 signing failed: {detail}")
    if len(process.stdout) != 64:
        raise BuildError("openssl did not return a 64-byte Ed25519 signature")
    signature = base64.urlsafe_b64encode(process.stdout).rstrip(b"=").decode("ascii")
    envelope = {
        "schema": 1,
        "domain": "imparo-program-pack-v1",
        "algorithm": "ed25519",
        "key_id": key_id,
        "manifest_bytes": len(manifest),
        "manifest_sha256": sha256(manifest),
        "signature": signature,
    }
    # Rust trust.rs freezes this exact insertion order. Alphabetical JSON would
    # be valid JSON but is deliberately rejected as a non-canonical envelope.
    output.write_bytes(
        json.dumps(
            envelope,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("ascii")
    )
