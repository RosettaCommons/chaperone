#!/usr/bin/env bash

# Harbor sources this before every command executed in the task container.
WORKSPACE=${HARBOR_HOST_WORKSPACE:-/workspace}
export HOME="$WORKSPACE/.home"
export TMPDIR="$WORKSPACE/.tmp"
export XDG_CACHE_HOME="$HOME/.cache"
export XDG_CONFIG_HOME="$HOME/.config"
export MPLCONFIGDIR="$XDG_CONFIG_HOME/matplotlib"
export PYTHONPYCACHEPREFIX="$WORKSPACE/.pycache"
export PYTHONPATH="$WORKSPACE/pipelines/src${PYTHONPATH:+:$PYTHONPATH}"
export PIPELINES_COMPUTE_BACKEND=auto
export PIPELINES_CONTAINER_BACKEND=apptainer

mkdir -p \
  "$HOME/.cache" \
  "$HOME/.config" \
  "$HOME/.local" \
  "$MPLCONFIGDIR" \
  "$PYTHONPYCACHEPREFIX" \
  "$TMPDIR" \
  "$WORKSPACE/submission"

if [[ -f "$WORKSPACE/pipelines/.env" ]]; then
  set -a
  source "$WORKSPACE/pipelines/.env"
  [[ ! -f "$WORKSPACE/pipelines/.env.ipd" ]] || \
    source "$WORKSPACE/pipelines/.env.ipd"
  set +a
  export PIPELINES_CONTAINER_BACKEND=apptainer
fi

export QHOP=false
export PATH="/opt/harbor-codex/bin:/opt/harbor-codex/codex-path:/opt/pipelines/.venv/bin:$PATH"
export CODEX_MANAGED_BY_NPM=1
