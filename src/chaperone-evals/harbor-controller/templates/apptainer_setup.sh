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

# Invoke the sanitized source with the configured image interpreter. Older digs
# images still contain the old CLI and must not select that baked-in code.
mkdir -p "$WORKSPACE/.bin"
cat > "$WORKSPACE/.bin/chaperone" <<'CHAPERONE_CLI'
#!/usr/bin/env bash
exec {{WORKER_PYTHON}} -m chaperone_cli.main "$@"
CHAPERONE_CLI
chmod 700 "$WORKSPACE/.bin/chaperone"
export QHOP=false
export PATH="/opt/harbor-codex/bin:/opt/harbor-codex/codex-path:$WORKSPACE/.bin:$PATH"
export CODEX_MANAGED_BY_NPM=1
