#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
TECTONIC="${TECTONIC:-/Users/briannaughton/latex_local/.tools/tectonic}"
[ -x "$TECTONIC" ] || TECTONIC="$(command -v tectonic || true)"
[ -n "${TECTONIC:-}" ] || { echo "Tectonic not found. Install Tectonic (tectonic-typesetting.github.io) or set TECTONIC=/path/to/tectonic"; exit 1; }
"$TECTONIC" "$ROOT/main.tex" --outdir "$ROOT"
mv "$ROOT/main.pdf" "$ROOT/certified-inference-note.pdf"
echo "Built $ROOT/certified-inference-note.pdf"
