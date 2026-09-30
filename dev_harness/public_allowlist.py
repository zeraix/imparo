#!/usr/bin/env python3
"""The public-export allowlist, generated from a deny list.

dev_harness/public-export.denylist names what stays private; every other file Git tracks is
public. This script writes dev_harness/public-export.allowlist from it (--write) or fails
when the committed allowlist differs from what the deny list gives (--check, the default).
A new file is therefore public unless a deny line covers it, and public_export.py still
reviews and exports one exact file list.

Deny-list lines: a line ending in / names a directory; any other line is a path or an
fnmatch pattern over the whole path; a line starting with ! makes matching paths public
again. The last line that matches a path decides, as in .gitignore. # starts a comment.
"""
from __future__ import annotations

import argparse
import fnmatch
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DENYLIST = "dev_harness/public-export.denylist"
ALLOWLIST = "dev_harness/public-export.allowlist"


def parse_denylist(text: str) -> list[tuple[bool, str]]:
    """(public_again, pattern) per rule line, in file order."""
    rules = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        keep = line.startswith("!")
        rules.append((keep, line[1:] if keep else line))
    return rules


def _matches(pattern: str, path: str) -> bool:
    if pattern.endswith("/"):
        return path.startswith(pattern)
    return fnmatch.fnmatchcase(path, pattern)


def is_denied(rules: list[tuple[bool, str]], path: str) -> bool:
    denied = False
    for keep, pattern in rules:
        if _matches(pattern, path):
            denied = not keep
    return denied


def tracked_files(root: Path) -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"], capture_output=True, check=True
    ).stdout
    return [p.decode("utf-8") for p in out.split(b"\0") if p]


def generate(root: Path = ROOT) -> tuple[bytes, list[str]]:
    """The allowlist bytes, and the deny lines that match no tracked file."""
    rules = parse_denylist((root / DENYLIST).read_text(encoding="utf-8"))
    files = tracked_files(root)
    public = [p for p in files if not is_denied(rules, p)]
    for required in (ALLOWLIST, DENYLIST):
        if required not in public:
            public.append(required)
    public = sorted(set(public), key=lambda p: p.encode("utf-8"))
    unused = [pattern for keep, pattern in rules
              if not any(_matches(pattern, p) for p in files)]
    return ("\n".join(public) + "\n").encode("utf-8"), unused


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true",
                        help="rewrite the allowlist instead of checking it")
    args = parser.parse_args(argv)
    data, unused = generate()
    for pattern in unused:
        print(f"note: deny line matches no tracked file: {pattern}", file=sys.stderr)
    target = ROOT / ALLOWLIST
    count = data.count(b"\n")
    if args.write:
        target.write_bytes(data)
        print(f"wrote {ALLOWLIST}: {count} files")
        return 0
    if target.read_bytes() != data:
        print(f"{ALLOWLIST} differs from {DENYLIST}; run: "
              f"python3 dev_harness/public_allowlist.py --write", file=sys.stderr)
        return 1
    print(f"OK {ALLOWLIST}: {count} files, matches {DENYLIST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
