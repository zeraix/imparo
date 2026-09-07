#!/usr/bin/env python3
"""Build a reviewed public mirror from an exact Git tree and file allowlist.

Dry-run validation is the default.  Source bytes are always read from Git blobs,
never from the index or working tree.  Apply first builds and verifies a sibling
staging tree, then replaces only files proven to belong to a managed destination.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import unicodedata
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence


REPORT_SCHEMA = 1
MARKER_SCHEMA = 1
RECOVERY_SCHEMA = 1
DEFAULT_ALLOWLIST = "dev_harness/public-export.allowlist"
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 128 * 1024 * 1024
MAX_FILES = 512
MAX_MARKER_BYTES = 4 * 1024 * 1024
MAX_LOCK_BYTES = 4096
ALLOWED_GIT_MODES = {"100644", "100755"}
WINDOWS_RESERVED = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}
FORBIDDEN_COMPONENTS = {
    ".git", ".claude", "target", "handoff", "private", "internal",
    "customer", "customers", "commercial", "artifacts", "results", "tmp",
}
FORBIDDEN_SUFFIXES = {
    ".cubin", ".ptx", ".fatbin", ".bin", ".dll", ".so", ".dylib",
    ".pdb", ".prof", ".log", ".pem", ".key", ".p12", ".pfx", ".exe",
    ".lib", ".gguf", ".onnx", ".safetensors",
    ".ncu-rep", ".nsys-rep",
}
FORBIDDEN_BASENAMES = {
    ".env", "credentials", "credentials.json", "id_dsa", "id_ecdsa",
    "id_ed25519", "id_rsa", "secrets.json",
}
BINARY_ALLOWLIST = {
    "assets/imparo-wordmark-black.png": (
        b"\x89PNG\r\n\x1a\n",
        b"IEND\xaeB\x60\x82",
        467655,
        "77f217e43ece7c5bed4a5e2aa2c83adf306f1406a3a3d8aec4b8756569f9f66a",
    ),
}
SECRET_PATTERNS = (
    (
        "pem-private-key",
        re.compile(r"-----BEGIN " + r"(?:[A-Z0-9][A-Z0-9 -]* )?PRIVATE KEY-----"),
    ),
    (
        "pgp-private-key",
        re.compile(r"-----BEGIN PGP " + r"PRIVATE KEY BLOCK-----"),
    ),
    (
        "putty-private-key",
        re.compile(r"(?m)^(?:PuTTY-" + r"User-Key-File-[0-9]+:|Private-" + r"Lines:)"),
    ),
    ("github-token", re.compile(r"(?<![A-Za-z0-9_])(?:github_pat_|gh[pousr]_)[A-Za-z0-9_]{20,}(?![A-Za-z0-9_])")),
    ("gitlab-token", re.compile(r"(?<![A-Za-z0-9_-])glpat-[A-Za-z0-9_-]{20,}(?![A-Za-z0-9_-])")),
    ("slack-token", re.compile(r"(?<![A-Za-z0-9-])xox[baprs]-[A-Za-z0-9-]{20,}(?![A-Za-z0-9-])")),
    ("aws-access-key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("google-api-key", re.compile(r"\bAIza[A-Za-z0-9_-]{35}\b")),
    ("openai-api-key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b")),
    (
        "assigned-secret",
        re.compile(
            r"(?i)\b(?:api[_-]?key|access[_-]?token|client[_-]?secret|password|passwd)"
            r"\s*[:=]\s*[\"']?[A-Za-z0-9+/=_-]{20,}"
        ),
    ),
)
PRIVATE_REFERENCE_PATTERNS = (
    (
        "private-repository-url",
        re.compile(
            r"(?i)(?:https?://|ssh://git@|git@)github\.com[/:][^\s\"'<>]*"
            r"(?:imparo-internal|imparo-optimization-packs|imparo-release-internal)\b"
        ),
    ),
    ("drive-absolute-path", re.compile(r"(?i)(?<![A-Za-z0-9_])[A-Z]:[\\/]")),
    ("posix-user-path", re.compile(r"/(?:Users|home)/[^/\s\"'<>]+/")),
    ("unc-path", re.compile(r"\\\\[^\\\s]+\\[^\\\s]+")),
    ("url-embedded-credentials", re.compile(r"(?i)\b(?:https?|ssh)://[^/\s:@]+:[^/\s@]+@")),
    ("private-hostname", re.compile(r"(?i)\b(?:[a-z0-9-]+\.)+(?:internal|corp)\b")),
    ("git-lfs-pointer", re.compile(r"(?m)^version https://git-lfs\.github\.com/spec/v1$")),
)
BINARY_TOKEN_PATTERNS = (
    *SECRET_PATTERNS,
    PRIVATE_REFERENCE_PATTERNS[0],
    PRIVATE_REFERENCE_PATTERNS[4],
)
BIDI_CONTROLS = {
    "\u202a", "\u202b", "\u202c", "\u202d", "\u202e",
    "\u2066", "\u2067", "\u2068", "\u2069",
}


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _strict_json_loads(text: str) -> Any:
    return json.loads(text, object_pairs_hook=_reject_duplicate_keys)


class PublicExportError(RuntimeError):
    def __init__(self, code: str, message: str, path: str | None = None):
        super().__init__(message)
        self.code = code
        self.path = path

    def record(self) -> dict[str, str]:
        record = {"code": self.code, "message": str(self)}
        if self.path is not None:
            # Rejected paths are untrusted and can themselves contain customer or
            # private-project names.  CI receives only a stable correlation digest;
            # successful reports may list paths because those bytes passed policy.
            record["path_sha256"] = hashlib.sha256(
                self.path.encode("utf-8", errors="surrogatepass")
            ).hexdigest()
        return record


@dataclass(frozen=True)
class TreeEntry:
    mode: str
    object_type: str
    oid: str
    path: str


@dataclass(frozen=True)
class ExportFile:
    path: str
    mode: str
    blob_oid: str
    data: bytes
    sha256: str

    def public_record(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "mode": self.mode,
            "blob_oid": self.blob_oid,
            "bytes": len(self.data),
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class ExportPlan:
    source: Path
    requested_tree: str
    tree_oid: str
    allowlist_path: str
    allowlist_blob_oid: str
    allowlist_sha256: str
    files: tuple[ExportFile, ...]

    def report(self, operation: str, destination: Path | None, applied: bool) -> dict[str, Any]:
        return {
            "schema": REPORT_SCHEMA,
            "ok": True,
            "operation": operation,
            "applied": applied,
            "tree_oid": self.tree_oid,
            "allowlist": self.allowlist_path,
            "allowlist_blob_oid": self.allowlist_blob_oid,
            "allowlist_sha256": self.allowlist_sha256,
            "destination_supplied": destination is not None,
            "file_count": len(self.files),
            "total_bytes": sum(len(item.data) for item in self.files),
            "files": [item.public_record() for item in self.files],
            "scans": {
                "path_policy": "passed",
                "binary_policy": "passed",
                "secret_patterns": "passed",
                "private_references": "passed",
            },
            "errors": [],
        }


def _git(source: Path, *args: str) -> bytes:
    environment = os.environ.copy()
    environment.update({"GIT_OPTIONAL_LOCKS": "0", "LC_ALL": "C"})
    try:
        completed = subprocess.run(
            ["git", "-C", str(source), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        # Git stderr commonly embeds checkout paths, remote URLs, and credential
        # helper details.  Never copy it into a CI-visible report.
        raise PublicExportError("git-error", "Git command failed") from error
    return completed.stdout


def _repo_root(source: Path) -> Path:
    raw = _git(source, "rev-parse", "--show-toplevel")
    try:
        return Path(raw.decode("utf-8", errors="strict").strip()).resolve(strict=True)
    except (UnicodeDecodeError, OSError) as error:
        raise PublicExportError("source-path", "repository root is not a valid path") from error


def _resolve_tree(source: Path, requested: str) -> str:
    if not requested or "\x00" in requested or "\n" in requested or "\r" in requested:
        raise PublicExportError("tree-revision", "tree revision is empty or contains controls")
    raw = _git(source, "rev-parse", "--verify", "--end-of-options", f"{requested}^{{tree}}")
    oid = raw.decode("ascii", errors="strict").strip()
    if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", oid) is None:
        raise PublicExportError("tree-oid", "Git returned an invalid tree object ID")
    return oid


def _tree_entries(source: Path, tree_oid: str) -> dict[str, TreeEntry]:
    raw = _git(source, "ls-tree", "-rz", "--full-tree", tree_oid)
    entries: dict[str, TreeEntry] = {}
    folded: dict[str, str] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            mode, object_type, raw_oid = metadata.split(b" ", 2)
            path = raw_path.decode("utf-8", errors="strict")
            entry = TreeEntry(
                mode.decode("ascii"),
                object_type.decode("ascii"),
                raw_oid.decode("ascii"),
                path,
            )
        except (UnicodeDecodeError, ValueError) as error:
            raise PublicExportError("tree-entry", "Git tree contains an undecodable entry") from error
        if path in entries:
            raise PublicExportError("tree-entry", "Git tree contains a duplicate path", path)
        entries[path] = entry
        folded_key = unicodedata.normalize("NFC", path).casefold()
        previous = folded.get(folded_key)
        if previous is not None and previous != path:
            # Record all collisions now; only selected paths are rejected later.
            folded[folded_key] = ""
        else:
            folded[folded_key] = path
    return entries


def validate_public_path(path: str) -> None:
    if not path or len(path.encode("utf-8")) > 4096:
        raise PublicExportError("path", "path is empty or too long", path)
    if path != unicodedata.normalize("NFC", path):
        raise PublicExportError("path-normalization", "path is not Unicode NFC", path)
    if path.startswith("/") or "\\" in path or "\x00" in path:
        raise PublicExportError("path", "path must be a relative POSIX path", path)
    pure = PurePosixPath(path)
    parts = pure.parts
    if not parts or str(pure) != path or any(part in {"", ".", ".."} for part in parts):
        raise PublicExportError("path", "path is not canonical", path)
    for part in parts:
        if len(part.encode("utf-8")) > 255:
            raise PublicExportError("path-portability", "path segment exceeds portable length", path)
        if any(ord(char) < 32 or ord(char) == 127 for char in part):
            raise PublicExportError("path-control", "path contains a control character", path)
        if part.startswith("-"):
            raise PublicExportError("path-portability", "path segment starts with a dash", path)
        if part.endswith((" ", ".")) or ":" in part:
            raise PublicExportError("path-portability", "path is unsafe on Windows", path)
        stem = part.split(".", 1)[0].casefold()
        if stem in WINDOWS_RESERVED:
            raise PublicExportError("path-portability", "path uses a reserved Windows name", path)
    lowered_parts = {part.casefold() for part in parts}
    forbidden = sorted(lowered_parts & FORBIDDEN_COMPONENTS)
    if forbidden:
        raise PublicExportError("private-path", f"path contains forbidden component {forbidden[0]}", path)
    basename = parts[-1].casefold()
    if basename in FORBIDDEN_BASENAMES or any(basename.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES):
        raise PublicExportError("private-path", "path has a forbidden sensitive/artifact name", path)


def _parse_allowlist(raw: bytes, path: str) -> list[str]:
    if raw.startswith(b"\xef\xbb\xbf"):
        raise PublicExportError("allowlist-encoding", "allowlist must not contain a BOM", path)
    if b"\r" in raw or not raw.endswith(b"\n"):
        raise PublicExportError("allowlist-format", "allowlist must use LF and end with LF", path)
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise PublicExportError("allowlist-encoding", "allowlist is not UTF-8", path) from error
    paths = text[:-1].split("\n")
    if not paths or any(not item for item in paths):
        raise PublicExportError("allowlist-format", "allowlist contains an empty line", path)
    for item in paths:
        validate_public_path(item)
    if len(paths) != len(set(paths)):
        raise PublicExportError("allowlist-duplicate", "allowlist contains duplicate paths", path)
    if len(paths) > MAX_FILES:
        raise PublicExportError("allowlist-size", "allowlist exceeds file-count ceiling", path)
    if paths != sorted(paths, key=lambda item: item.encode("utf-8")):
        raise PublicExportError("allowlist-order", "allowlist is not UTF-8 byte sorted", path)
    folded: dict[str, str] = {}
    for item in paths:
        key = unicodedata.normalize("NFC", item).casefold()
        if key in folded:
            raise PublicExportError(
                "allowlist-collision",
                "allowlist contains a portable path collision",
                item,
            )
        folded[key] = item
    return paths


def _scan_patterns(
    path: str,
    text: str,
    patterns: Iterable[tuple[str, re.Pattern[str]]] = (
        *SECRET_PATTERNS,
        *PRIVATE_REFERENCE_PATTERNS,
    ),
) -> None:
    for label, pattern in patterns:
        match = pattern.search(text)
        if match is not None:
            line = text.count("\n", 0, match.start()) + 1
            raise PublicExportError(label, f"blocked pattern at line {line}", path)


def _scan_text(path: str, data: bytes) -> None:
    if b"\x00" in data:
        raise PublicExportError("binary-file", "unapproved binary/NUL content", path)
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise PublicExportError("text-encoding", "text file is not UTF-8", path) from error
    if "\r" in text:
        raise PublicExportError("line-endings", "public text must use LF line endings", path)
    if any(character in text for character in BIDI_CONTROLS):
        raise PublicExportError("unicode-control", "text contains a bidirectional control", path)
    _scan_patterns(path, text)


def _scan_blob(path: str, data: bytes) -> None:
    binary_signature = BINARY_ALLOWLIST.get(path)
    if binary_signature is not None:
        prefix, suffix, expected_bytes, expected_sha256 = binary_signature
        if not data.startswith(prefix) or not data.endswith(suffix):
            raise PublicExportError("binary-signature", "approved binary has an invalid signature", path)
        if len(data) != expected_bytes or hashlib.sha256(data).hexdigest() != expected_sha256:
            raise PublicExportError("binary-digest", "approved binary does not match its reviewed digest", path)
        # Arbitrary compressed bytes frequently resemble short filesystem paths.
        # Binary scanning therefore uses only high-confidence credential/repository
        # tokens; text payloads retain the complete private-path policy above.
        _scan_patterns(path, data.decode("latin-1"), BINARY_TOKEN_PATTERNS)
        _scan_patterns(
            path, data.decode("utf-16le", errors="ignore"), BINARY_TOKEN_PATTERNS
        )
        return
    _scan_text(path, data)


def build_plan(
    source: Path | str,
    requested_tree: str = "HEAD",
    allowlist_path: str = DEFAULT_ALLOWLIST,
) -> ExportPlan:
    try:
        resolved_source = Path(source).resolve(strict=True)
    except OSError as error:
        raise PublicExportError("source-path", "source does not exist or is inaccessible") from error
    source_root = _repo_root(resolved_source)
    validate_public_path(allowlist_path)
    tree_oid = _resolve_tree(source_root, requested_tree)
    entries = _tree_entries(source_root, tree_oid)
    allowlist_entry = entries.get(allowlist_path)
    if allowlist_entry is None:
        raise PublicExportError("allowlist-missing", "allowlist is absent from the selected tree", allowlist_path)
    if allowlist_entry.object_type != "blob" or allowlist_entry.mode not in ALLOWED_GIT_MODES:
        raise PublicExportError("allowlist-mode", "allowlist is not a regular Git blob", allowlist_path)
    raw_allowlist = _git(source_root, "cat-file", "blob", allowlist_entry.oid)
    paths = _parse_allowlist(raw_allowlist, allowlist_path)
    if allowlist_path not in paths:
        raise PublicExportError("allowlist-self", "allowlist must export itself", allowlist_path)

    selected_folded: dict[str, str] = {}
    files: list[ExportFile] = []
    total_bytes = 0
    for path in paths:
        entry = entries.get(path)
        if entry is None:
            raise PublicExportError("missing-path", "allowlisted path is absent from the selected tree", path)
        if entry.object_type != "blob" or entry.mode not in ALLOWED_GIT_MODES:
            raise PublicExportError("git-mode", "allowlisted entry is not a regular non-symlink blob", path)
        folded = unicodedata.normalize("NFC", path).casefold()
        previous = selected_folded.get(folded)
        if previous is not None:
            raise PublicExportError("path-collision", "selected paths have a portable collision", path)
        selected_folded[folded] = path
        data = _git(source_root, "cat-file", "blob", entry.oid)
        if len(data) > MAX_FILE_BYTES:
            raise PublicExportError("file-size", "file exceeds public-export size ceiling", path)
        total_bytes += len(data)
        if total_bytes > MAX_TOTAL_BYTES:
            raise PublicExportError("total-size", "public export exceeds total size ceiling")
        _scan_blob(path, data)
        files.append(
            ExportFile(path, entry.mode, entry.oid, data, hashlib.sha256(data).hexdigest())
        )
    return ExportPlan(
        source=source_root,
        requested_tree=requested_tree,
        tree_oid=tree_oid,
        allowlist_path=allowlist_path,
        allowlist_blob_oid=allowlist_entry.oid,
        allowlist_sha256=hashlib.sha256(raw_allowlist).hexdigest(),
        files=tuple(files),
    )


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _is_reparse(path: Path) -> bool:
    try:
        value = path.lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISLNK(value.st_mode) or bool(
        getattr(value, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _durable_replace(source: Path, target: Path) -> None:
    if os.name == "nt":
        import ctypes

        move_file = ctypes.windll.kernel32.MoveFileExW
        move_file.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32]
        move_file.restype = ctypes.c_int
        replace_existing = 0x1
        write_through = 0x8
        if not move_file(str(source), str(target), replace_existing | write_through):
            raise ctypes.WinError()
        return
    os.replace(source, target)
    _fsync_directory(source.parent)
    if target.parent != source.parent:
        _fsync_directory(target.parent)


def _marker_path(destination: Path) -> Path:
    return destination.parent / f".{destination.name}.imparo-public-export.json"


def _lock_path(destination: Path) -> Path:
    return destination.parent / f".{destination.name}.imparo-public-export.lock"


def _recovery_path(destination: Path) -> Path:
    return destination.parent / f".{destination.name}.imparo-public-export.recovery.json"


@contextmanager
def _destination_lock(destination: Path) -> Iterable[None]:
    """Hold one stable advisory lock inode for every apply to this destination."""

    lock_path = _lock_path(destination)
    descriptor: int | None = None
    try:
        if _is_reparse(lock_path):
            raise PublicExportError("destination-lock", "destination lock is a reparse point")
        flags = os.O_RDWR | os.O_CREAT
        if os.name == "nt":
            flags |= getattr(os, "O_BINARY", 0) | getattr(os, "O_NOINHERIT", 0)
        else:
            flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(lock_path, flags, 0o600)
        descriptor_stat = os.fstat(descriptor)
        path_stat = lock_path.lstat()
        if (
            _is_reparse(lock_path)
            or not stat.S_ISREG(descriptor_stat.st_mode)
            or descriptor_stat.st_nlink != 1
            or descriptor_stat.st_size > MAX_LOCK_BYTES
            or (descriptor_stat.st_dev, descriptor_stat.st_ino) != (path_stat.st_dev, path_stat.st_ino)
        ):
            raise PublicExportError("destination-lock", "destination lock is not an owned regular file")
        if descriptor_stat.st_size == 0:
            os.write(descriptor, b"\0")
            os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        handle = os.fdopen(descriptor, "r+b", buffering=0)
        descriptor = None
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except PublicExportError:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        try:
            handle.close()
        except (NameError, OSError):
            pass
        raise PublicExportError("destination-locked", "another public export owns the destination") from error
    try:
        yield
    finally:
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _marker_payload(plan: ExportPlan) -> dict[str, Any]:
    return {
        "schema": MARKER_SCHEMA,
        "managed_by": "imparo-public-export",
        "source_tree": plan.tree_oid,
        "allowlist_sha256": plan.allowlist_sha256,
        "files": [
            {
                "path": item.path,
                "mode": item.mode,
                "bytes": len(item.data),
                "sha256": item.sha256,
            }
            for item in plan.files
        ],
    }


def _validate_marker_record(record: Any) -> None:
    if (
        not isinstance(record, dict)
        or set(record) != {"path", "mode", "bytes", "sha256"}
        or not isinstance(record.get("path"), str)
        or record.get("mode") not in ALLOWED_GIT_MODES
        or type(record.get("bytes")) is not int
        or record["bytes"] < 0
        or re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256", ""))) is None
    ):
        raise PublicExportError("destination-marker", "destination marker contains an invalid file record")
    validate_public_path(record["path"])


def _walk_destination(destination: Path, *, ignore_root_git: bool = True) -> set[str]:
    found: set[str] = set()
    for root, dirs, files in os.walk(destination, topdown=True, followlinks=False):
        root_path = Path(root)
        if ignore_root_git and root_path == destination and ".git" in dirs:
            if _is_reparse(destination / ".git"):
                raise PublicExportError("destination-reparse", "destination .git is a reparse point")
            dirs.remove(".git")
        for name in list(dirs):
            candidate = root_path / name
            if _is_reparse(candidate):
                raise PublicExportError("destination-reparse", "destination contains a reparse point")
        for name in files:
            candidate = root_path / name
            if _is_reparse(candidate):
                raise PublicExportError("destination-reparse", "destination contains a reparse point")
            relative = candidate.relative_to(destination).as_posix()
            validate_public_path(relative)
            found.add(relative)
    return found


def _load_marker(destination: Path) -> tuple[dict[str, Any], bytes] | None:
    marker = _marker_path(destination)
    if not marker.exists():
        return None
    if _is_reparse(marker) or not marker.is_file():
        raise PublicExportError("destination-marker", "destination marker is not a regular file")
    try:
        marker_size = marker.stat().st_size
    except OSError as error:
        raise PublicExportError("destination-marker", "destination marker cannot be inspected") from error
    if marker_size > MAX_MARKER_BYTES:
        raise PublicExportError("destination-marker", "destination marker exceeds size ceiling")
    raw = marker.read_bytes()
    try:
        payload = _strict_json_loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, ValueError, RecursionError) as error:
        raise PublicExportError("destination-marker", "destination marker is invalid JSON") from error
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema", "managed_by", "source_tree", "allowlist_sha256", "files"}
        or payload.get("schema") != MARKER_SCHEMA
        or payload.get("managed_by") != "imparo-public-export"
        or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", str(payload.get("source_tree", ""))) is None
        or re.fullmatch(r"[0-9a-f]{64}", str(payload.get("allowlist_sha256", ""))) is None
    ):
        raise PublicExportError("destination-marker", "destination marker has wrong authority")
    files = payload.get("files")
    if not isinstance(files, list) or len(files) > MAX_FILES:
        raise PublicExportError("destination-marker", "destination marker omits file inventory")
    for record in files:
        _validate_marker_record(record)
    return payload, raw


def _verify_inventory(
    destination: Path,
    records: Iterable[dict[str, Any]],
    *,
    ignore_root_git: bool = True,
) -> list[str]:
    expected: dict[str, dict[str, Any]] = {}
    for record in records:
        _validate_marker_record(record)
        path = record["path"]
        if path in expected:
            raise PublicExportError("destination-marker", "destination marker repeats a path", path)
        expected[path] = record
    actual = _walk_destination(destination, ignore_root_git=ignore_root_git)
    if actual != set(expected):
        raise PublicExportError("destination-dirty", "destination file set differs from its managed inventory")
    for path, record in expected.items():
        data = (destination / PurePosixPath(path)).read_bytes()
        if len(data) != record.get("bytes") or hashlib.sha256(data).hexdigest() != record.get("sha256"):
            raise PublicExportError("destination-dirty", "destination file differs from its managed inventory", path)
        if os.name != "nt":
            executable = bool((destination / PurePosixPath(path)).stat().st_mode & 0o111)
            if executable != (record["mode"] == "100755"):
                raise PublicExportError("destination-dirty", "destination executable mode differs from inventory", path)
    return sorted(expected, key=lambda item: item.encode("utf-8"))


def _record_matches(root: Path, record: dict[str, Any]) -> bool:
    path = root / PurePosixPath(record["path"])
    if not path.exists() or _is_reparse(path) or not path.is_file():
        return False
    try:
        data = path.read_bytes()
        if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
            return False
        if os.name != "nt":
            executable = bool(path.stat().st_mode & 0o111)
            if executable != (record["mode"] == "100755"):
                return False
    except OSError:
        return False
    return True


def _snapshot_records(destination: Path, paths: Iterable[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        candidate = destination / PurePosixPath(path)
        if _is_reparse(candidate) or not candidate.is_file():
            raise PublicExportError("destination-dirty", "destination inventory is not a regular file", path)
        data = candidate.read_bytes()
        executable = os.name != "nt" and bool(candidate.stat().st_mode & 0o111)
        records.append(
            {
                "path": path,
                "mode": "100755" if executable else "100644",
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return records


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    raw = (json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
    if len(raw) > MAX_MARKER_BYTES:
        raise PublicExportError("recovery-record", "recovery record exceeds size ceiling")
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        _durable_replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _adopt_clean_git_destination(destination: Path) -> list[str]:
    if not (destination / ".git").exists() or _is_reparse(destination / ".git"):
        raise PublicExportError("destination-unmanaged", "existing destination lacks a valid export marker")
    status_output = _git(destination, "status", "--porcelain=v1", "--untracked-files=all")
    if status_output:
        raise PublicExportError("destination-dirty", "unmarked destination Git worktree is not clean")
    raw_paths = _git(destination, "ls-files", "-z")
    try:
        tracked = [item.decode("utf-8", errors="strict") for item in raw_paths.split(b"\0") if item]
    except UnicodeDecodeError as error:
        raise PublicExportError("destination-path", "destination has a non-UTF-8 tracked path") from error
    for path in tracked:
        validate_public_path(path)
    actual = _walk_destination(destination)
    if actual != set(tracked):
        raise PublicExportError("destination-dirty", "destination contains ignored or untracked files")
    return sorted(tracked, key=lambda item: item.encode("utf-8"))


def _remove_empty_parents(path: Path, boundary: Path) -> None:
    current = path
    while current != boundary:
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent


def _write_stage(plan: ExportPlan, stage: Path) -> None:
    stage.mkdir(mode=0o700)
    for item in plan.files:
        destination = stage / PurePosixPath(item.path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(item.data)
        if os.name != "nt":
            destination.chmod(0o755 if item.mode == "100755" else 0o644)
    records = [
        {"path": item.path, "mode": item.mode, "bytes": len(item.data), "sha256": item.sha256}
        for item in plan.files
    ]
    _verify_inventory(stage, records)


def _load_recovery(destination: Path) -> dict[str, Any] | None:
    path = _recovery_path(destination)
    if not path.exists():
        return None
    if _is_reparse(path) or not path.is_file() or path.stat().st_size > MAX_MARKER_BYTES:
        raise PublicExportError("recovery-required", "recovery record is not a bounded regular file")
    try:
        payload = _strict_json_loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, ValueError, RecursionError) as error:
        raise PublicExportError("recovery-required", "recovery record is malformed") from error
    expected_keys = {
        "schema", "managed_by", "transaction", "destination_name", "stage_name",
        "backup_name", "marker_temp_name", "created_destination", "old_marker_sha256",
        "new_marker_sha256", "old_files", "new_files",
    }
    transaction = payload.get("transaction") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or set(payload) != expected_keys
        or payload.get("schema") != RECOVERY_SCHEMA
        or payload.get("managed_by") != "imparo-public-export"
        or re.fullmatch(r"[0-9a-f]{32}", str(transaction or "")) is None
        or payload.get("destination_name") != destination.name
        or payload.get("stage_name") != f".{destination.name}.staging-{transaction}"
        or payload.get("backup_name") != f".{destination.name}.rollback-{transaction}"
        or payload.get("marker_temp_name") != f".{destination.name}.marker-{transaction}"
        or type(payload.get("created_destination")) is not bool
        or payload.get("old_marker_sha256") is not None
        and re.fullmatch(r"[0-9a-f]{64}", str(payload.get("old_marker_sha256"))) is None
        or re.fullmatch(r"[0-9a-f]{64}", str(payload.get("new_marker_sha256", ""))) is None
    ):
        raise PublicExportError("recovery-required", "recovery record has wrong authority")
    for key in ("old_files", "new_files"):
        records = payload.get(key)
        if not isinstance(records, list) or len(records) > MAX_FILES:
            raise PublicExportError("recovery-required", "recovery inventory is unbounded")
        seen: set[str] = set()
        for record in records:
            try:
                _validate_marker_record(record)
            except PublicExportError as error:
                raise PublicExportError("recovery-required", "recovery inventory is malformed") from error
            if record["path"] in seen:
                raise PublicExportError("recovery-required", "recovery inventory repeats a path")
            seen.add(record["path"])
    return payload


def _marker_digest(destination: Path) -> str | None:
    marker = _marker_path(destination)
    if not marker.exists():
        return None
    if _is_reparse(marker) or not marker.is_file() or marker.stat().st_size > MAX_MARKER_BYTES:
        raise PublicExportError("recovery-required", "destination marker is not recoverable")
    return hashlib.sha256(marker.read_bytes()).hexdigest()


def _cleanup_transaction(destination: Path, payload: dict[str, Any]) -> None:
    old_paths = {record["path"] for record in payload["old_files"]}
    new_paths = {record["path"] for record in payload["new_files"]}
    for key, allowed in (("stage_name", new_paths), ("backup_name", old_paths)):
        directory = destination.parent / payload[key]
        if directory.exists():
            if _is_reparse(directory) or not directory.is_dir():
                raise PublicExportError("recovery-required", "transaction directory is unsafe")
            found = _walk_destination(directory, ignore_root_git=False)
            if not found.issubset(allowed):
                raise PublicExportError("recovery-required", "transaction directory contains unknown files")
            shutil.rmtree(directory)
            _fsync_directory(directory.parent)
    marker_temp = destination.parent / payload["marker_temp_name"]
    if marker_temp.exists():
        if _is_reparse(marker_temp) or not marker_temp.is_file():
            raise PublicExportError("recovery-required", "transaction marker temporary is unsafe")
        marker_temp.unlink()
        _fsync_directory(marker_temp.parent)
    _recovery_path(destination).unlink()
    _fsync_directory(destination.parent)


def _recover_pending(destination: Path) -> str | None:
    payload = _load_recovery(destination)
    if payload is None:
        return None
    current_marker = _marker_digest(destination)
    if current_marker == payload["new_marker_sha256"]:
        if not destination.exists() or not destination.is_dir() or _is_reparse(destination):
            raise PublicExportError("recovery-required", "committed destination is unavailable")
        try:
            _verify_inventory(destination, payload["new_files"])
            _cleanup_transaction(destination, payload)
        except (OSError, PublicExportError) as error:
            raise PublicExportError("recovery-required", "committed transaction cleanup is incomplete") from error
        return "committed"
    if current_marker != payload["old_marker_sha256"]:
        raise PublicExportError("recovery-required", "destination marker is ambiguous")

    old = {record["path"]: record for record in payload["old_files"]}
    new = {record["path"]: record for record in payload["new_files"]}
    backup = destination.parent / payload["backup_name"]
    try:
        if not destination.exists():
            if old:
                destination.mkdir(mode=0o755)
            elif not payload["created_destination"]:
                raise PublicExportError("recovery-required", "prior destination disappeared")
        for path, record in new.items():
            if path in old:
                continue
            target = destination / PurePosixPath(path)
            if target.exists() or _is_reparse(target):
                if not _record_matches(destination, record):
                    raise PublicExportError("recovery-required", "new-only file became ambiguous", path)
                target.unlink()
                _fsync_directory(target.parent)
                _remove_empty_parents(target.parent, destination)
        for path, record in old.items():
            target = destination / PurePosixPath(path)
            backup_file = backup / PurePosixPath(path)
            if backup_file.exists() or _is_reparse(backup_file):
                if not _record_matches(backup, record):
                    raise PublicExportError("recovery-required", "backup file failed its pinned digest", path)
                if target.exists() or _is_reparse(target):
                    if _record_matches(destination, record):
                        continue
                    new_record = new.get(path)
                    if new_record is None or not _record_matches(destination, new_record):
                        raise PublicExportError("recovery-required", "destination file became ambiguous", path)
                    target.unlink()
                    _fsync_directory(target.parent)
                target.parent.mkdir(parents=True, exist_ok=True)
                _durable_replace(backup_file, target)
            elif not _record_matches(destination, record):
                raise PublicExportError("recovery-required", "old file is missing from destination and backup", path)
        if old:
            _verify_inventory(destination, payload["old_files"])
        elif destination.exists():
            if _walk_destination(destination):
                raise PublicExportError("recovery-required", "new destination contains unknown files")
            if payload["created_destination"]:
                destination.rmdir()
                _fsync_directory(destination.parent)
        if _marker_digest(destination) != payload["old_marker_sha256"]:
            raise PublicExportError("recovery-required", "old marker was not restored")
        _cleanup_transaction(destination, payload)
    except (OSError, PublicExportError) as error:
        raise PublicExportError("recovery-required", "automatic rollback is incomplete; recovery state was retained") from error
    return "rolled-back"


def _apply_plan_locked(plan: ExportPlan, destination: Path) -> None:
    recovery = _recover_pending(destination)
    # A recovered transaction may belong to an older invocation.  Cleanup makes
    # that state stable, then this invocation must still apply its own exact plan.

    marker_state = _load_marker(destination)
    if destination.exists():
        if marker_state is not None:
            old_paths = _verify_inventory(destination, marker_state[0]["files"])
            old_records = list(marker_state[0]["files"])
        elif not any(destination.iterdir()):
            old_paths = []
            old_records = []
        else:
            old_paths = _adopt_clean_git_destination(destination)
            old_records = _snapshot_records(destination, old_paths)
    else:
        if marker_state is not None:
            raise PublicExportError("destination-marker", "orphan destination marker exists")
        old_paths = []
        old_records = []

    transaction = uuid.uuid4().hex
    stage = destination.parent / f".{destination.name}.staging-{transaction}"
    backup = destination.parent / f".{destination.name}.rollback-{transaction}"
    marker = _marker_path(destination)
    marker_temp = destination.parent / f".{destination.name}.marker-{transaction}"
    new_records = [
        {"path": item.path, "mode": item.mode, "bytes": len(item.data), "sha256": item.sha256}
        for item in plan.files
    ]
    marker_bytes = (
        json.dumps(_marker_payload(plan), ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")
    old_marker_sha256 = hashlib.sha256(marker_state[1]).hexdigest() if marker_state is not None else None
    payload = {
        "schema": RECOVERY_SCHEMA,
        "managed_by": "imparo-public-export",
        "transaction": transaction,
        "destination_name": destination.name,
        "stage_name": stage.name,
        "backup_name": backup.name,
        "marker_temp_name": marker_temp.name,
        "created_destination": not destination.exists(),
        "old_marker_sha256": old_marker_sha256,
        "new_marker_sha256": hashlib.sha256(marker_bytes).hexdigest(),
        "old_files": old_records,
        "new_files": new_records,
    }
    _write_stage(plan, stage)
    backup.mkdir(mode=0o700)
    _atomic_json(_recovery_path(destination), payload)
    if _load_recovery(destination) != payload:
        raise PublicExportError("recovery-required", "recovery record did not survive publication")
    try:
        if not destination.exists():
            destination.mkdir(mode=0o755)
            _fsync_directory(destination.parent)
        for path in old_paths:
            source_path = destination / PurePosixPath(path)
            backup_path = backup / PurePosixPath(path)
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            _durable_replace(source_path, backup_path)
            _remove_empty_parents(source_path.parent, destination)
        for item in plan.files:
            staged_path = stage / PurePosixPath(item.path)
            target_path = destination / PurePosixPath(item.path)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if target_path.exists() or _is_reparse(target_path):
                raise PublicExportError("destination-collision", "new export collides with destination content", item.path)
            _durable_replace(staged_path, target_path)
        _verify_inventory(destination, new_records)
        with marker_temp.open("xb") as handle:
            handle.write(marker_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        _durable_replace(marker_temp, marker)
        if _marker_digest(destination) != payload["new_marker_sha256"]:
            raise PublicExportError("destination-marker", "installed marker digest mismatch")
        _verify_inventory(destination, new_records)
        _cleanup_transaction(destination, payload)
    except BaseException as error:
        try:
            recovered = _recover_pending(destination)
        except BaseException as recovery_error:
            raise PublicExportError(
                "recovery-required",
                "automatic rollback is incomplete; recovery state was retained",
            ) from recovery_error
        if recovered == "committed":
            return
        if isinstance(error, PublicExportError):
            raise error
        if isinstance(error, OSError):
            raise PublicExportError("filesystem-error", "public export filesystem operation failed") from error
        raise


def apply_plan(plan: ExportPlan, destination: Path | str) -> None:
    try:
        destination = Path(destination).resolve(strict=False)
        source = plan.source.resolve(strict=True)
        if destination == destination.parent or destination == Path.home().resolve(strict=False):
            raise PublicExportError("destination-scope", "refusing a root or home destination")
        if _is_relative_to(destination, source) or _is_relative_to(source, destination):
            raise PublicExportError("destination-scope", "source and destination must not contain each other")
        parent = destination.parent
        if not parent.exists() or not parent.is_dir() or _is_reparse(parent):
            raise PublicExportError("destination-parent", "destination parent must be an existing real directory")
        if destination.exists() and (not destination.is_dir() or _is_reparse(destination)):
            raise PublicExportError("destination-type", "destination is not a real directory")
        with _destination_lock(destination):
            _apply_plan_locked(plan, destination)
    except PublicExportError:
        raise
    except OSError as error:
        raise PublicExportError("filesystem-error", "public export filesystem operation failed") from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--tree", default="HEAD", help="Git revision resolved once to a tree object")
    parser.add_argument("--allowlist", default=DEFAULT_ALLOWLIST, help="repo-relative allowlist path in the selected tree")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="validate and report without writing (default)")
    mode.add_argument("--apply", action="store_true", help="apply a verified export to --dest")
    parser.add_argument("--dest", type=Path, help="managed destination; required with --apply")
    parser.add_argument("--json", action="store_true", help="emit one machine-readable JSON report")
    return parser


def _emit(payload: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
        return
    if payload.get("ok"):
        print(f"OK {payload['operation']}: {payload['file_count']} files, tree {payload['tree_oid']}")
        if payload.get("destination_supplied"):
            print("destination: supplied")
    else:
        error = payload["errors"][0]
        print(f"FAILED {error['code']}: {error['message']}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    operation = "apply" if args.apply else "dry-run"
    if args.apply and args.dest is None:
        parser.error("--apply requires --dest")
    if not args.apply and args.dest is not None:
        parser.error("--dest is valid only with --apply")
    try:
        plan = build_plan(args.source, args.tree, args.allowlist)
        destination = args.dest.resolve(strict=False) if args.dest is not None else None
        if args.apply:
            apply_plan(plan, destination)
        _emit(plan.report(operation, destination, args.apply), args.json)
        return 0
    except PublicExportError as error:
        payload = {
            "schema": REPORT_SCHEMA,
            "ok": False,
            "operation": operation,
            "applied": False,
            "errors": [error.record()],
        }
        _emit(payload, args.json)
        return 1
    except (OSError, UnicodeError) as error:
        sanitized = PublicExportError("filesystem-error", "public export filesystem operation failed")
        payload = {
            "schema": REPORT_SCHEMA,
            "ok": False,
            "operation": operation,
            "applied": False,
            "errors": [sanitized.record()],
        }
        _emit(payload, args.json)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
