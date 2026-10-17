#!/usr/bin/env bash
# Compatibility entry point: all submission/preflight policy lives in run_eval.py.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "$ROOT/.venv/bin/python" "$ROOT/run_eval.py" "${TASK:-ppi_pdl1}" "$@"
