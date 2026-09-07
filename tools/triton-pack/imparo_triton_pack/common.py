from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .errors import BuildError

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
SYMBOL_RE = re.compile(r"^ip_[0-9a-f]{64}$")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BuildError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise BuildError(f"read {path}: {error}") from error
    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_pairs
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BuildError(f"parse strict JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise BuildError(f"{path} must contain a JSON object")
    return value


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise BuildError(f"hash {path}: {error}") from error
    return digest.hexdigest()


def exact_keys(value: dict[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise BuildError(
            f"{name} keys differ: missing={missing}, unknown={unknown}"
        )


def require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise BuildError(f"{name} must be a non-empty string")
    return value


def require_int(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BuildError(f"{name} must be an integer >= {minimum}")
    return value


def require_sha256(value: Any, name: str) -> str:
    text = require_string(value, name)
    if not SHA256_RE.fullmatch(text) or text == "0" * 64:
        raise BuildError(f"{name} must be a non-zero lowercase SHA-256")
    return text


def require_revision(value: Any, name: str) -> str:
    text = require_string(value, name)
    if not REVISION_RE.fullmatch(text) or text == "0" * 40:
        raise BuildError(f"{name} must be a non-zero lowercase git revision")
    return text


def safe_relative_file(value: Any, name: str) -> str:
    text = require_string(value, name)
    path = Path(text)
    if (
        path.is_absolute()
        or len(path.parts) != 1
        or text in {".", ".."}
        or "/" in text
        or "\\" in text
    ):
        raise BuildError(f"{name} must be a single relative filename")
    return text
