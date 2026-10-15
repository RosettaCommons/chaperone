#!/usr/bin/env bash
# Build US-align into the active venv's bin, where tm_score_rna resolves it.
#
# US-align is a single C++ translation unit, not a Python package, so it is not
# installed by `uv pip install` and does not survive a fresh clone. Run this
# once per checkout (and in CI) before using the `tm_score_rna` block.
set -euo pipefail

PREFIX="${1:-${VIRTUAL_ENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/.venv}}"
BIN_DIR="${PREFIX}/bin"
SRC_DIR="$(mktemp -d)"
trap 'rm -rf "${SRC_DIR}"' EXIT

git clone --depth 1 https://github.com/pylelab/USalign "${SRC_DIR}/USalign"
g++ -O3 -ffast-math -o "${SRC_DIR}/USalign/USalign" "${SRC_DIR}/USalign/USalign.cpp"
mkdir -p "${BIN_DIR}"
install -m 0755 "${SRC_DIR}/USalign/USalign" "${BIN_DIR}/USalign"

echo "Installed $("${BIN_DIR}/USalign" -v 2>&1 | head -1) to ${BIN_DIR}/USalign"
