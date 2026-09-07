#!/usr/bin/env python3
"""Fail-closed release-boundary checks for AOT Program Packs and runtimes.

This module deliberately uses only the Python standard library.  Python is a CI/build
tool here; it is not imported or shipped by the Imparo runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Sequence


MAX_PACK_FILES = 72
MAX_PACK_BYTES = 768 * 1024 * 1024
ROOT_FILES = {
    "manifest.json",
    "manifest.sig",
    "SBOM.spdx.json",
    "THIRD_PARTY_NOTICES",
    "provenance.json",
}
FORBIDDEN_SUFFIXES = {
    ".py", ".pyc", ".pyo", ".ptx", ".ttir", ".ttgir", ".ll", ".bc",
    ".dll", ".so", ".dylib", ".exe", ".pdb", ".lib", ".a", ".fatbin",
}
FORBIDDEN_ELF_SECTION_PREFIXES = (
    ".debug", ".zdebug", ".line", ".stab", ".llvm", ".nv_debug", ".nv.debug",
)
FORBIDDEN_CUBIN_MARKERS = (
    b"ttir", b"ttgir", b"llvm ir", b"target triple", b".version ", b".target sm_",
)
ABSOLUTE_SOURCE_PATH = re.compile(
    rb"(?i)(?:[a-z]:[\\/]|\\\\[^\\\x00]+\\|"
    rb"/(?:home|users|workspace|builds|__w|tmp|root|opt|mnt|var/tmp|private/tmp|runner)/)"
    rb"[^\x00\r\n\t ]{2,}"
)
FORBIDDEN_RUNTIME_IMPORT = re.compile(
    rb"(?i)(?:lib)?(?:python(?:\d+(?:\.\d+)?)?|torch(?:_cpu|_cuda)?|c10|triton)"
    rb"(?:\.dll|\.so(?:\.\d+)*|\.dylib)"
)
FORBIDDEN_PROCESS_MARKER = re.compile(
    rb"(?i)(?:^|[\x00/\x5c])(?:python(?:\d+(?:\.\d+)?)?(?:\.exe)?|ptxas(?:\.exe)?|"
    rb"nvcc(?:\.exe)?|triton)(?:\x00|$)"
)
FORBIDDEN_CARGO_PACKAGES = {
    "pyo3", "pyo3-build-config", "pyo3-ffi", "python3-sys", "tch", "torch-sys",
    "triton", "triton-sys",
}


class ReleaseBoundaryError(RuntimeError):
    def __init__(self, code: str, message: str, path: str | None = None):
        super().__init__(message)
        self.code = code
        self.path = path


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path, maximum: int = 8 * 1024 * 1024) -> Any:
    raw = path.read_bytes()
    if not raw or len(raw) > maximum or raw.startswith(b"\xef\xbb\xbf"):
        raise ReleaseBoundaryError("json-size", "JSON is empty, oversized, or has a BOM", path.name)
    try:
        text = raw.decode("utf-8")
        return json.loads(text, object_pairs_hook=_strict_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ReleaseBoundaryError("json-format", "JSON is not strict UTF-8", path.name) from error


def _portable_relative(path: str) -> None:
    candidate = PurePosixPath(path)
    if (
        not path
        or "\\" in path
        or candidate.is_absolute()
        or any(part in {"", ".", ".."} for part in candidate.parts)
        or any(ord(char) < 0x20 or ord(char) == 0x7F for char in path)
    ):
        raise ReleaseBoundaryError("pack-path", "pack path is not portable and relative", path)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _elf_metadata(raw: bytes, path: str) -> tuple[list[str], int]:
    if len(raw) < 64 or raw[:4] != b"\x7fELF":
        raise ReleaseBoundaryError("cubin-format", "module is not an ELF cubin", path)
    if raw[4] != 2 or raw[5] != 1 or raw[7] != 51:
        raise ReleaseBoundaryError(
            "cubin-format", "cubin must be ELF64 little-endian with NVIDIA CUDA OSABI", path
        )
    try:
        header = struct.unpack_from("<16sHHIQQQIHHHHHH", raw, 0)
        if header[2] != 190:
            raise ValueError("ELF machine is not CUDA")
        section_offset, flags, section_size, section_count, names_index = (
            header[6], header[7], header[11], header[12], header[13]
        )
        if section_size < 64 or section_count == 0 or names_index >= section_count:
            raise ValueError("invalid section table")
        if section_offset + section_size * section_count > len(raw):
            raise ValueError("truncated section table")

        def section(index: int) -> tuple[int, int, int]:
            offset = section_offset + section_size * index
            name, _, _, _, data_offset, data_size, _, _, _, _ = struct.unpack_from(
                "<IIQQQQIIQQ", raw, offset
            )
            if data_offset + data_size > len(raw):
                raise ValueError("truncated section")
            return name, data_offset, data_size

        _, names_offset, names_size = section(names_index)
        names = raw[names_offset:names_offset + names_size]
        result = []
        for index in range(section_count):
            name_offset, _, _ = section(index)
            if name_offset >= len(names):
                raise ValueError("section name outside table")
            end = names.find(b"\0", name_offset)
            if end < 0:
                raise ValueError("unterminated section name")
            result.append(names[name_offset:end].decode("ascii"))
        return result, flags & 0xFF
    except (struct.error, UnicodeDecodeError, ValueError) as error:
        raise ReleaseBoundaryError("cubin-format", "cubin has an invalid ELF section table", path) from error


def _scan_cubin(raw: bytes, path: str, expected_sm: int) -> None:
    names, binary_sm = _elf_metadata(raw, path)
    if binary_sm != expected_sm:
        raise ReleaseBoundaryError(
            "cubin-sm", f"cubin ELF target SM{binary_sm} differs from manifest SM{expected_sm}", path
        )
    for name in names:
        if name.lower().startswith(FORBIDDEN_ELF_SECTION_PREFIXES):
            raise ReleaseBoundaryError("cubin-debug", "cubin contains debug or IR sections", path)
    lowered = raw.lower()
    if any(marker in lowered for marker in FORBIDDEN_CUBIN_MARKERS):
        raise ReleaseBoundaryError("cubin-ir", "cubin contains PTX or compiler IR markers", path)
    if ABSOLUTE_SOURCE_PATH.search(raw):
        raise ReleaseBoundaryError("cubin-source-path", "cubin contains an absolute build/source path", path)


def _verify_sbom(value: Any) -> None:
    if not isinstance(value, dict) or value.get("spdxVersion") != "SPDX-2.3":
        raise ReleaseBoundaryError("sbom-format", "SBOM must be an SPDX-2.3 JSON document")
    if value.get("dataLicense") != "CC0-1.0" or value.get("SPDXID") != "SPDXRef-DOCUMENT":
        raise ReleaseBoundaryError("sbom-format", "SBOM document identity/license is incomplete")
    creation = value.get("creationInfo")
    if not isinstance(creation, dict) or not creation.get("creators"):
        raise ReleaseBoundaryError("sbom-format", "SBOM creationInfo.creators is required")
    packages = value.get("packages")
    if not isinstance(packages, list) or not packages:
        raise ReleaseBoundaryError("sbom-packages", "SBOM must inventory at least one package")
    for package in packages:
        if not isinstance(package, dict) or not package.get("name") or not package.get("SPDXID"):
            raise ReleaseBoundaryError("sbom-packages", "SBOM package identity is incomplete")
        licenses = (package.get("licenseDeclared"), package.get("licenseConcluded"))
        if not any(isinstance(item, str) and item not in {"", "NONE", "NOASSERTION"} for item in licenses):
            raise ReleaseBoundaryError("sbom-license", "every SBOM package needs a declared license")


def scan_pack(
    root: Path,
    expected_sm: int | None = None,
    *,
    allow_feasibility_smoke: bool = False,
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ReleaseBoundaryError("pack-root", "pack root is not a directory")
    files: dict[str, bytes] = {}
    for entry in root.rglob("*"):
        if entry.is_symlink() or not (entry.is_file() or entry.is_dir()):
            raise ReleaseBoundaryError("pack-entry", "pack contains a link or special entry")
        if entry.is_dir():
            relative_directory = entry.relative_to(root).as_posix()
            if relative_directory != "modules":
                raise ReleaseBoundaryError("pack-inventory", "pack contains an extra directory", relative_directory)
            continue
        relative = entry.relative_to(root).as_posix()
        _portable_relative(relative)
        if entry.suffix.lower() in FORBIDDEN_SUFFIXES and not relative.startswith("modules/"):
            raise ReleaseBoundaryError("pack-host-code", "pack contains source, IR, or host code", relative)
        raw = entry.read_bytes()
        files[relative] = raw
    if len(files) > MAX_PACK_FILES or sum(map(len, files.values())) > MAX_PACK_BYTES:
        raise ReleaseBoundaryError("pack-size", "pack inventory exceeds release ceilings")
    missing = ROOT_FILES - files.keys()
    if missing:
        raise ReleaseBoundaryError("pack-inventory", "pack is missing required release sidecars")

    manifest = _load_json(root / "manifest.json")
    if not isinstance(manifest, dict) or manifest.get("schema") != 1:
        raise ReleaseBoundaryError("manifest", "manifest schema is not Program Pack v1")
    target = manifest.get("target")
    if not isinstance(target, dict) or target.get("warp_size") != 32:
        raise ReleaseBoundaryError("manifest-target", "manifest target is incomplete")
    if target.get("sm") not in {80, 86}:
        raise ReleaseBoundaryError("manifest-sm", "Step 4 release target is not an admitted SM lane")
    if expected_sm is not None and target.get("sm") != expected_sm:
        raise ReleaseBoundaryError("manifest-sm", "manifest does not match the CI SM lane")

    modules = manifest.get("modules")
    if not isinstance(modules, list) or not modules:
        raise ReleaseBoundaryError("manifest-modules", "manifest has no modules")
    declared: set[str] = set()
    for module in modules:
        if not isinstance(module, dict):
            raise ReleaseBoundaryError("manifest-modules", "manifest module is malformed")
        path = module.get("file")
        if not isinstance(path, str) or not re.fullmatch(r"modules/[0-9a-f]{64}\.cubin", path):
            raise ReleaseBoundaryError("manifest-module-path", "module path is not content addressed")
        if path in declared or path not in files:
            raise ReleaseBoundaryError("manifest-module-path", "module is duplicate or missing", path)
        declared.add(path)
        raw = files[path]
        if module.get("bytes") != len(raw) or module.get("sha256") != _sha256(raw):
            raise ReleaseBoundaryError("manifest-module-hash", "module bytes/hash mismatch", path)
        _scan_cubin(raw, path, target["sm"])
    if {path for path in files if path.startswith("modules/")} != declared:
        raise ReleaseBoundaryError("pack-inventory", "pack has an undeclared module")
    if set(files) != ROOT_FILES | declared:
        raise ReleaseBoundaryError("pack-inventory", "pack has files outside the release inventory")

    sidecars = {
        "sbom_sha256": "SBOM.spdx.json",
        "notices_sha256": "THIRD_PARTY_NOTICES",
        "provenance_sha256": "provenance.json",
    }
    for field, path in sidecars.items():
        if manifest.get(field) != _sha256(files[path]):
            raise ReleaseBoundaryError("sidecar-hash", f"{path} is not bound by manifest", path)
    _verify_sbom(_load_json(root / "SBOM.spdx.json"))
    provenance = _load_json(root / "provenance.json")
    if not isinstance(provenance, dict) or not provenance:
        raise ReleaseBoundaryError("provenance", "provenance must be a nonempty JSON object")
    toolchain_status = provenance.get("toolchain_status")
    feasibility_only = provenance.get("feasibility_only")
    if (
        toolchain_status not in {"source_pin_pre_release", "release_ready"}
        or not isinstance(feasibility_only, bool)
        or feasibility_only != (toolchain_status == "source_pin_pre_release")
    ):
        raise ReleaseBoundaryError(
            "provenance-status", "provenance toolchain status/feasibility marker is absent or inconsistent"
        )
    if feasibility_only and not allow_feasibility_smoke:
        raise ReleaseBoundaryError(
            "feasibility-only", "source-pin feasibility pack is forbidden at the release boundary"
        )
    notices = files["THIRD_PARTY_NOTICES"]
    if len(notices) < 32 or not re.search(rb"(?i)(?:copyright|license|notice)", notices):
        raise ReleaseBoundaryError("notices", "THIRD_PARTY_NOTICES is empty or non-substantive")

    signature = _load_json(root / "manifest.sig", maximum=64 * 1024)
    key_id = signature.get("key_id") if isinstance(signature, dict) else None
    valid_key_id = (
        isinstance(key_id, str)
        and 3 <= len(key_id.encode("utf-8")) <= 128
        and re.fullmatch(r"[a-z0-9][a-z0-9._-]*[a-z0-9]", key_id) is not None
        and re.search(r"[._-]", key_id) is not None
        and re.search(r"[._-]{2}", key_id) is None
    )
    if (
        not isinstance(signature, dict)
        or signature.get("schema") != 1
        or signature.get("domain") != "imparo-program-pack-v1"
        or signature.get("algorithm") != "ed25519"
        or not valid_key_id
        or not isinstance(signature.get("signature"), str)
        or re.fullmatch(r"[A-Za-z0-9_-]{86}", signature["signature"]) is None
    ):
        raise ReleaseBoundaryError("signature", "signature envelope is malformed")
    manifest_raw = files["manifest.json"]
    if signature.get("manifest_bytes") != len(manifest_raw) or signature.get("manifest_sha256") != _sha256(manifest_raw):
        raise ReleaseBoundaryError("signature", "signature envelope does not bind manifest bytes")
    canonical_signature = (
        f'{{"schema":1,"domain":"imparo-program-pack-v1","algorithm":"ed25519",'
        f'"key_id":"{signature["key_id"]}","manifest_bytes":{len(manifest_raw)},'
        f'"manifest_sha256":"{_sha256(manifest_raw)}","signature":"{signature["signature"]}"}}'
    ).encode("ascii")
    if files["manifest.sig"] != canonical_signature:
        raise ReleaseBoundaryError("signature", "signature envelope is not canonical for the Rust trust boundary")
    return {
        "ok": True,
        "files": len(files),
        "bytes": sum(map(len, files.values())),
        "sm": target.get("sm"),
        "feasibility_only": feasibility_only,
    }


def parse_cargo_tree_packages(text: str) -> set[str]:
    packages: set[str] = set()
    for line in text.splitlines():
        cleaned = re.sub(r"^[\s│├└─]+", "", line).strip()
        match = re.match(r"([A-Za-z0-9_.-]+)\s+v\d", cleaned)
        if match:
            packages.add(match.group(1).lower())
    return packages


def scan_cargo_runtime(repo: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "cargo", "tree", "--workspace", "--locked", "--edges", "normal,build",
            "--prefix", "none", "--format", "{p}",
        ],
        cwd=repo, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if result.returncode:
        raise ReleaseBoundaryError("cargo-tree", "cargo tree failed; dependency gate cannot be skipped")
    packages = parse_cargo_tree_packages(result.stdout)
    forbidden = sorted(packages & FORBIDDEN_CARGO_PACKAGES)
    if forbidden:
        raise ReleaseBoundaryError("runtime-dependency", "forbidden Python/Torch/Triton Cargo dependency: " + ", ".join(forbidden))
    return {"ok": True, "packages": len(packages)}


def scan_runtime_binary(path: Path) -> dict[str, Any]:
    raw = path.resolve(strict=True).read_bytes()
    import_match = FORBIDDEN_RUNTIME_IMPORT.search(raw)
    if import_match:
        raise ReleaseBoundaryError("runtime-import", "binary contains a Python/Torch/Triton dynamic-library dependency", path.name)
    process_match = FORBIDDEN_PROCESS_MARKER.search(raw)
    if process_match:
        raise ReleaseBoundaryError("runtime-compiler", "binary embeds a Python/compiler process marker", path.name)
    return {"ok": True, "bytes": len(raw), "sha256": _sha256(raw)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pack = sub.add_parser("pack")
    pack.add_argument("--root", required=True, type=Path)
    pack.add_argument("--expected-sm", type=int, choices=(80, 86))
    pack.add_argument(
        "--allow-feasibility-smoke",
        action="store_true",
        help="admit a provenance-marked source-pin pack for smoke validation only",
    )
    cargo = sub.add_parser("cargo")
    cargo.add_argument("--repo", type=Path, default=Path("."))
    binary = sub.add_parser("binary")
    binary.add_argument("paths", nargs="+", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "pack":
            report: Any = scan_pack(
                args.root,
                args.expected_sm,
                allow_feasibility_smoke=args.allow_feasibility_smoke,
            )
        elif args.command == "cargo":
            report = scan_cargo_runtime(args.repo)
        else:
            report = {str(path): scan_runtime_binary(path) for path in args.paths}
    except (OSError, ReleaseBoundaryError) as error:
        code = getattr(error, "code", "filesystem")
        print(json.dumps({"ok": False, "code": code, "message": str(error)}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
