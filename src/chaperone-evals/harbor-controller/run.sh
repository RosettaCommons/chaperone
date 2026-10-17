#!/usr/bin/env bash
set -euo pipefail

ROOT=${HARBOR_CONTROLLER_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
MODE=${1:-local}
ARM=${2:-harness}
PROMPT_FILE=${3:-${PROMPT_FILE:-}}
if [[ -n "$PROMPT_FILE" ]]; then
  PROMPT_FILE=$(realpath "$PROMPT_FILE")
fi

if [[ "$ARM" == both ]]; then
  PAIR_NAME=${JOB_NAME:-chaperone-codex-both-$(date -u +%H%M%SZ)}
  JOB_NAME="${PAIR_NAME}-baseline" "$0" "$MODE" baseline "$PROMPT_FILE"
  JOB_NAME="${PAIR_NAME}-harness" "$0" "$MODE" harness "$PROMPT_FILE"
  exit
fi
if [[ "$MODE" != local && "$MODE" != job ]]; then
  echo "usage: $0 {local|job} {baseline|harness|both} [extra-prompt.md]" >&2
  exit 2
fi
if [[ "$ARM" != baseline && "$ARM" != harness ]]; then
  echo "arm must be baseline, harness, or both" >&2
  exit 2
fi

MODEL=${MODEL:-openai/gpt-5.6-sol}
REASONING_EFFORT=${REASONING_EFFORT:-medium}
ATTEMPTS=${ATTEMPTS:-1}
CONCURRENCY=${CONCURRENCY:-1}
GPUS=${GPUS:-0}
APPTAINER_IMAGE=${APPTAINER_IMAGE:-${APPTAINER_BUILD_SOURCE:-}}

# Harbor defaults Codex to OPENAI_API_KEY authentication, even when the host
# Codex CLI is logged in through ChatGPT. Reuse that existing login when an API
# key (or an explicit auth file) was not supplied, instead of injecting an
# empty key and failing with HTTP 401.
if [[ -z "${OPENAI_API_KEY:-}" && -z "${CODEX_AUTH_JSON_PATH:-}" ]]; then
  CODEX_DEFAULT_AUTH_PATH="$HOME/.codex/auth.json"
  if [[ -s "$CODEX_DEFAULT_AUTH_PATH" ]]; then
    export CODEX_FORCE_AUTH_JSON=1
  else
    echo "Codex authentication is unavailable: set OPENAI_API_KEY or run 'codex login'." >&2
    exit 2
  fi
fi

if [[ ! "$ATTEMPTS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ATTEMPTS must be a positive integer" >&2
  exit 2
fi
if [[ ! "$CONCURRENCY" =~ ^[1-9][0-9]*$ ]]; then
  echo "CONCURRENCY must be a positive integer" >&2
  exit 2
fi
if [[ ! "$GPUS" =~ ^[0-9]+$ ]] || (( GPUS > 5 )); then
  echo "GPUS must be an integer from 0 through 5" >&2
  exit 2
fi
STAMP=$(date -u +%H%M%SZ)
JOB_NAME=${JOB_NAME:-chaperone-codex-${ARM}-${STAMP}}
if [[ ! "$JOB_NAME" =~ ^[A-Za-z0-9_.-]{1,128}$ ]]; then
  echo "JOB_NAME must contain only letters, digits, dot, underscore, and hyphen (maximum 128 characters)" >&2
  exit 2
fi
TASK=${HARBOR_TASK_DIR:-$ROOT/jobs/task-definitions/ppi_pdl1_test}
if [[ -z "$PROMPT_FILE" && -f "$TASK/prompt.md" ]]; then
  PROMPT_FILE="$TASK/prompt.md"
fi

if [[ ! -f "$TASK/task.toml" || ! -f "$TASK/instruction.md" || ! -d "$TASK/environment" ]]; then
  echo "Harbor task wrapper is incomplete: $TASK" >&2
  exit 2
fi
if [[ "$ARM" == harness ]]; then
  if [[ -z "${WORKSPACE_SEED:-}" ]]; then
    echo "WORKSPACE_SEED is required; launch harness runs through run_eval.py with an explicit --chaperone-path." >&2
    exit 2
  fi
  SNAPSHOT=$WORKSPACE_SEED
  TASK_INPUTS=${TASK_INPUTS:-$TASK/inputs}
  # The skill and executable source must be the same sanitized bundle. Never
  # accept an ambient second path that could reintroduce repository artifacts.
  HARNESS_SKILL=$SNAPSHOT
  if [[ ! -d "$SNAPSHOT" ]]; then
    echo "Chaperone workspace seed not found: $SNAPSHOT" >&2
    exit 2
  fi
  if [[ ! -d "$TASK_INPUTS" ]]; then
    echo "Evaluation inputs directory not found: $TASK_INPUTS" >&2
    exit 2
  fi
  if [[ ! -f "$HARNESS_SKILL/SKILL.md" ]]; then
    echo "Chaperone harness skill not found: $HARNESS_SKILL/SKILL.md" >&2
    exit 2
  fi
fi

ARGS=(
  run
  --path "$TASK"
  --agent codex
  --model "$MODEL"
  --ak "reasoning_effort=$REASONING_EFFORT"
  --disable-verification
  --n-attempts "$ATTEMPTS"
  --n-concurrent "$CONCURRENCY"
  --job-name "$JOB_NAME"
  --jobs-dir "${HARBOR_JOBS_DIR:-$ROOT/jobs}/results"
)
if [[ -n "$APPTAINER_IMAGE" ]]; then
  APPTAINER_IMAGE=$(realpath "$APPTAINER_IMAGE")
  if [[ ! -f "$APPTAINER_IMAGE" || "$APPTAINER_IMAGE" != *.sif ]]; then
    echo "APPTAINER_IMAGE must name an existing .sif file: $APPTAINER_IMAGE" >&2
    exit 2
  fi
fi
if [[ "$ARM" == harness ]]; then
  ARGS+=(
    --env environments.apptainer:ApptainerEnvironment
    --ek "arm=harness"
    --ek "workspace_seed=$SNAPSHOT"
    --ek "task_inputs=$TASK_INPUTS"
    --ek "apptainer_gpus=$GPUS"
  )
  if [[ -n "$APPTAINER_IMAGE" ]]; then
    ARGS+=(--ek "apptainer_image=$APPTAINER_IMAGE")
  fi
  ARGS+=(--skill "$HARNESS_SKILL")
else
  # This policy-only subclass delegates execution to Harbor's stock
  # Singularity lifecycle while pinning and auditing the clean base image.
  export APPTAINER_CLEANENV=1
  export SINGULARITY_CLEANENV=1
  ARGS+=(--env environments.baseline:BaselineSingularityEnvironment)
fi
if [[ -n "$PROMPT_FILE" ]]; then
  ARGS+=(--extra-instruction-path "$PROMPT_FILE")
fi

if [[ "$MODE" == job ]]; then
  echo "The legacy 'job' mode is disabled because it bypasses the guarded Slurm workflow." >&2
  echo "Use ./run_eval.py; its outer allocation is CPU-only and each arm brokers child compute." >&2
  exit 2
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [[ ${DRY_RUN:-0} == 1 ]]; then
  printf '%q ' "$ROOT/.venv/bin/python" "$ROOT/scripts/harbor_cli.py" "${ARGS[@]}"
  printf '\nOuter Apptainer GPU passthrough: %s\n' "$GPUS"
  exit
fi

set +e
"$ROOT/.venv/bin/python" "$ROOT/scripts/harbor_cli.py" "${ARGS[@]}"
HARBOR_STATUS=$?
set -e
if (( HARBOR_STATUS != 0 )); then
  exit "$HARBOR_STATUS"
fi
exec "$ROOT/.venv/bin/python" "$ROOT/scripts/check_harbor_result.py" \
  --result "${HARBOR_JOBS_DIR:-$ROOT/jobs}/results/$JOB_NAME/result.json"
