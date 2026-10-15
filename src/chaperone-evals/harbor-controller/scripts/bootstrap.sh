#!/usr/bin/env bash
set -euo pipefail

CONTROLLER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PROJECT_ROOT=$(cd "$CONTROLLER_ROOT/.." && pwd)
CHAPERONE_PYTHON_VERSION=${CHAPERONE_PYTHON_VERSION:-3.12}
CHAPERONE_REPO=${CHAPERONE_REPO:-$(cd "$PROJECT_ROOT/../.." && pwd)}
CHAPERONE_REPO=$(cd "$CHAPERONE_REPO" && pwd)
if [[ ! -f "$CHAPERONE_REPO/pyproject.toml" || ! -d "$CHAPERONE_REPO/src/chaperone" ]]; then
  echo "Chaperone checkout not found: $CHAPERONE_REPO" >&2
  exit 2
fi

# Managed interpreters and both environments must be on shared compute storage.
# Keep controller dependencies out of the public scientific checkout's package.
for checkout in "$PROJECT_ROOT" "$CHAPERONE_REPO"; do
  if [[ ! -x "$checkout/.venv/bin/python" ]]; then
    uv venv "$checkout/.venv" --python "$CHAPERONE_PYTHON_VERSION" --managed-python
  fi
done
uv pip install --python "$PROJECT_ROOT/.venv/bin/python" -e "$CHAPERONE_REPO" -e "$PROJECT_ROOT[test]"
uv pip install --python "$CHAPERONE_REPO/.venv/bin/python" -e "$CHAPERONE_REPO"
ln -sfn ../.venv "$CONTROLLER_ROOT/.venv"
printf '%s\n' "Controller installed. If selecting another scoring checkout, keep CHAPERONE_REPO exported."
printf '%s\n' "Each harness launch also requires --chaperone-path for its agent source snapshot."
