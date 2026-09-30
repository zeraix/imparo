#!/bin/sh
# Compatibility wrapper for the fixed-tree, allowlist-only public exporter.
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON_BIN=${PYTHON:-python3}
# The allowlist is generated from public-export.denylist; refuse a stale one.
"$PYTHON_BIN" "$SCRIPT_DIR/public_allowlist.py"
exec "$PYTHON_BIN" "$SCRIPT_DIR/public_export.py" "$@"
