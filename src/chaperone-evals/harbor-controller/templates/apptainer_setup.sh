#!/usr/bin/env bash

# Harbor sources this before every command executed in the task container.
WORKSPACE=${HARBOR_HOST_WORKSPACE:-/workspace}
export HOME="$WORKSPACE/.home"
export TMPDIR="$WORKSPACE/.tmp"
export XDG_CACHE_HOME="$HOME/.cache"
export XDG_CONFIG_HOME="$HOME/.config"
export MPLCONFIGDIR="$XDG_CONFIG_HOME/matplotlib"
export PYTHONPYCACHEPREFIX="$WORKSPACE/.pycache"
export PYTHONPATH="$WORKSPACE/chaperone/src${PYTHONPATH:+:$PYTHONPATH}"
export CHAPERONE_COMPUTE_BACKEND=auto
export CHAPERONE_CONTAINER_BACKEND=apptainer

mkdir -p \
  "$HOME/.cache" \
  "$HOME/.config" \
  "$HOME/.local" \
  "$MPLCONFIGDIR" \
  "$PYTHONPYCACHEPREFIX" \
  "$TMPDIR" \
  "$WORKSPACE/submission"

if [[ -f "$WORKSPACE/chaperone/.env" ]]; then
  set -a
  source "$WORKSPACE/chaperone/.env"
  [[ ! -f "$WORKSPACE/chaperone/.env.ipd" ]] || \
    source "$WORKSPACE/chaperone/.env.ipd"
  set +a
  export CHAPERONE_CONTAINER_BACKEND=apptainer
fi

export QHOP=false
export PATH="/opt/harbor-codex/bin:/opt/harbor-codex/codex-path:/opt/pipelines/.venv/bin:$PATH"
export CODEX_MANAGED_BY_NPM=1
