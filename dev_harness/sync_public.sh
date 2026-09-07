#!/bin/sh
# Compatibility wrapper for the fixed-tree, allowlist-only public exporter.
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON_BIN=${PYTHON:-python3}
exec "$PYTHON_BIN" "$SCRIPT_DIR/public_export.py" "$@"
