#!/usr/bin/env python3
"""Verify an exported public tree against its detached exporter report."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Sequence

try:
    from . import public_export
except ImportError:
    import public_export


def _load_report(path: Path) -> dict[str, Any]:
    if public_export._is_reparse(path) or not path.is_file():
        raise public_export.PublicExportError("artifact-report", "artifact report is not a regular file")
    try:
        size = path.stat().st_size
        if size > public_export.MAX_MARKER_BYTES:
            raise public_export.PublicExportError("artifact-report", "artifact report exceeds size ceiling")
        payload = public_export._strict_json_loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, ValueError, RecursionError) as error:
        raise public_export.PublicExportError("artifact-report", "artifact report is malformed") from error
    expected = {
        "schema", "ok", "operation", "applied", "tree_oid", "allowlist",
        "allowlist_blob_oid", "allowlist_sha256", "destination_supplied", "file_count",
        "total_bytes", "files", "scans", "errors",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != expected
        or payload.get("schema") != public_export.REPORT_SCHEMA
        or type(payload.get("schema")) is not int
        or payload.get("ok") is not True
        or payload.get("operation") != "apply"
        or payload.get("applied") is not True
        or payload.get("destination_supplied") is not True
        or payload.get("errors") != []
        or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", str(payload.get("tree_oid", ""))) is None
        or not isinstance(payload.get("tree_oid"), str)
        or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", str(payload.get("allowlist_blob_oid", ""))) is None
        or not isinstance(payload.get("allowlist_blob_oid"), str)
        or re.fullmatch(r"[0-9a-f]{64}", str(payload.get("allowlist_sha256", ""))) is None
        or not isinstance(payload.get("allowlist_sha256"), str)
        or not isinstance(payload.get("allowlist"), str)
        or type(payload.get("file_count")) is not int
        or type(payload.get("total_bytes")) is not int
    ):
        raise public_export.PublicExportError("artifact-report", "artifact report has wrong authority")
    files = payload.get("files")
    if not isinstance(files, list) or not files or len(files) > public_export.MAX_FILES:
        raise public_export.PublicExportError("artifact-report", "artifact inventory is unbounded")
    seen: set[str] = set()
    total = 0
    for record in files:
        if not isinstance(record, dict) or set(record) != {
            "path", "mode", "blob_oid", "bytes", "sha256",
        }:
            raise public_export.PublicExportError("artifact-report", "artifact file record is malformed")
        public_export._validate_marker_record(
            {key: record[key] for key in ("path", "mode", "bytes", "sha256")}
        )
        if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", str(record.get("blob_oid", ""))) is None:
            raise public_export.PublicExportError("artifact-report", "artifact blob ID is invalid")
        if record["path"] in seen:
            raise public_export.PublicExportError("artifact-report", "artifact inventory repeats a path")
        seen.add(record["path"])
        total += record["bytes"]
        if record["bytes"] > public_export.MAX_FILE_BYTES:
            raise public_export.PublicExportError("artifact-report", "artifact file exceeds size ceiling")
    if (
        payload["file_count"] != len(files)
        or payload["total_bytes"] != total
        or total > public_export.MAX_TOTAL_BYTES
    ):
        raise public_export.PublicExportError("artifact-report", "artifact totals do not match inventory")
    scans = payload.get("scans")
    if not isinstance(scans, dict) or scans != {
        "path_policy": "passed",
        "binary_policy": "passed",
        "secret_patterns": "passed",
        "private_references": "passed",
    }:
        raise public_export.PublicExportError("artifact-report", "artifact scan claims are incomplete")
    public_export.validate_public_path(payload["allowlist"])
    allowlist_record = next((record for record in files if record["path"] == payload["allowlist"]), None)
    if (
        allowlist_record is None
        or allowlist_record["sha256"] != payload["allowlist_sha256"]
        or allowlist_record["blob_oid"] != payload["allowlist_blob_oid"]
    ):
        raise public_export.PublicExportError("artifact-report", "allowlist is not bound to artifact inventory")
    return payload


def verify(root: Path, report_path: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    if public_export._is_reparse(root) or not root.is_dir():
        raise public_export.PublicExportError("artifact-root", "artifact root is not a real directory")
    payload = _load_report(report_path.resolve(strict=True))
    records = [
        {key: record[key] for key in ("path", "mode", "bytes", "sha256")}
        for record in payload["files"]
    ]
    public_export._verify_inventory(root, records, ignore_root_git=False)
    allowlist_bytes = (root / payload["allowlist"]).read_bytes()
    allowlisted_paths = public_export._parse_allowlist(allowlist_bytes, payload["allowlist"])
    reported_paths = [record["path"] for record in payload["files"]]
    if allowlisted_paths != reported_paths:
        raise public_export.PublicExportError("artifact-report", "allowlist and artifact inventory differ")
    for record in payload["files"]:
        data = (root / record["path"]).read_bytes()
        public_export._scan_blob(record["path"], data)
        framed = f"blob {len(data)}\0".encode("ascii") + data
        algorithm = "sha1" if len(record["blob_oid"]) == 40 else "sha256"
        if hashlib.new(algorithm, framed).hexdigest() != record["blob_oid"]:
            raise public_export.PublicExportError("artifact-report", "artifact Git blob ID mismatch")
    return {
        "schema": 1,
        "ok": True,
        "tree_oid": payload["tree_oid"],
        "allowlist_sha256": payload["allowlist_sha256"],
        "file_count": payload["file_count"],
        "total_bytes": payload["total_bytes"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = verify(args.root, args.report)
    except public_export.PublicExportError as error:
        result = {"schema": 1, "ok": False, "errors": [error.record()]}
        print(json.dumps(result, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
        return 1
    except (OSError, UnicodeError):
        error = public_export.PublicExportError("filesystem-error", "artifact verification failed")
        result = {"schema": 1, "ok": False, "errors": [error.record()]}
        print(json.dumps(result, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
        return 1
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
