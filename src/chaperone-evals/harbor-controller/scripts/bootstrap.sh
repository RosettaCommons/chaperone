#!/usr/bin/env bash
set -euo pipefail

CONTROLLER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# The controller is src/chaperone-evals/harbor-controller of the Chaperone
# checkout it evaluates, and shares that checkout's environment. The nested
# project src/chaperone-evals depends on the checkout's `pipelines` package by
# path and adds the harness dependencies.
REPO_ROOT=$(cd "$CONTROLLER_ROOT/../../.." && pwd)
CHAPERONE_PYTHON_VERSION=${CHAPERONE_PYTHON_VERSION:-3.12}

install_into_checkout() {
  local checkout=$1 target=$2
  cd "$checkout"
  # Login images may provide /usr/bin/python3.12 even when that path does not
  # exist on compute nodes. Requiring uv's user-managed interpreter keeps each
  # checkout-local virtualenv executable through the shared home filesystem.
  if [[ ! -x .venv/bin/python ]]; then
    uv venv --python "$CHAPERONE_PYTHON_VERSION" --managed-python
  fi
  # Install on top of whatever is already there. `uv sync` would prune extras
  # that were installed for other work.
  uv pip install --python .venv/bin/python -e "$target" --override overrides.txt
}

if [[ ! -f "$REPO_ROOT/pyproject.toml" ]]; then
  echo "Chaperone checkout not found: $REPO_ROOT" >&2
  exit 2
fi
install_into_checkout "$REPO_ROOT" src/chaperone-evals
# The controller's entry points run through this portable link to the
# environment; it is not committed.
ln -sfn ../../../.venv "$CONTROLLER_ROOT/.venv"

PIPELINES_REPO=${PIPELINES_REPO:-$REPO_ROOT}
if [[ "$PIPELINES_REPO" != "$REPO_ROOT" ]]; then
  # Held-out scoring runs in the scoring checkout's own environment.
  if [[ ! -f "$PIPELINES_REPO/pyproject.toml" ]]; then
    echo "Chaperone checkout not found: $PIPELINES_REPO" >&2
    exit 2
  fi
  install_into_checkout "$PIPELINES_REPO" '.'
fi
printf '%s\n' "Harness source is intentionally not configured at bootstrap time."
printf '%s\n' "Each harness launch must provide --pipelines-path explicitly."
