#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# Run output goes to scratch by default; a home quota cannot absorb the ~30 GB
# of per-trial bootstrap, let alone an abandoned run's container .tmp.
if [[ -z "${HARBOR_JOBS_DIR:-}" ]]; then
  if [[ -d "/net/scratch/$USER" ]]; then
    HARBOR_JOBS_DIR="/net/scratch/$USER/chaperone-evals/runs"
  else
    HARBOR_JOBS_DIR="$ROOT/jobs"
  fi
fi
export HARBOR_JOBS_DIR
ARM=${ARM:-harness}
TASK=${TASK:-biotin_binder_design}
MODEL=${MODEL:-openai/gpt-5.6-sol}
REASONING_EFFORT=${REASONING_EFFORT:-medium}
ATTEMPTS=${ATTEMPTS:-1}
CONCURRENCY=${CONCURRENCY:-1}
MEMORY_GB=${MEMORY_GB:-64}
CPU_PARTITION=${SLURM_CPU_PARTITION:-cpu}
APPTAINER_BUILD_SOURCE=${APPTAINER_BUILD_SOURCE:-/projects/ml/pipelines/containers/pipelines_agent_hardened_20260816.sif}
CHAPERONE_PATH=
WORKSPACE_SEED=

usage() {
  echo "usage: $0 [--arm baseline|harness] [--chaperone-path PATH] [--model MODEL] [--reasoning-effort EFFORT] [--attempts N] [--concurrency N]" >&2
}

while (( $# )); do
  case "$1" in
    --arm) ARM=${2:?--arm requires a value}; shift 2 ;;
    --chaperone-path) CHAPERONE_PATH=${2:?--chaperone-path requires a value}; shift 2 ;;
    --model) MODEL=${2:?--model requires a value}; shift 2 ;;
    --reasoning-effort) REASONING_EFFORT=${2:?--reasoning-effort requires a value}; shift 2 ;;
    --attempts) ATTEMPTS=${2:?--attempts requires a value}; shift 2 ;;
    --concurrency) CONCURRENCY=${2:?--concurrency requires a value}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "$ARM" != baseline && "$ARM" != harness ]]; then
  echo "--arm must be baseline or harness" >&2
  exit 2
fi
if [[ "$ARM" == baseline && -n "$CHAPERONE_PATH" ]]; then
  echo "--chaperone-path is not accepted by the baseline arm" >&2
  exit 2
fi
if [[ ! "$ATTEMPTS" =~ ^[1-9][0-9]*$ || ! "$CONCURRENCY" =~ ^[1-9][0-9]*$ || ! "$MEMORY_GB" =~ ^[1-9][0-9]*$ ]]; then
  echo "attempts, concurrency, and MEMORY_GB must be positive integers" >&2
  exit 2
fi
if [[ "$APPTAINER_BUILD_SOURCE" != *://* && ! -f "$APPTAINER_BUILD_SOURCE" ]]; then
  echo "Apptainer build source not found: $APPTAINER_BUILD_SOURCE" >&2
  exit 2
fi
if [[ ! -x "$ROOT/.venv/bin/harbor" || ! -x "$ROOT/.venv/bin/python" ]]; then
  echo "Controller environment is incomplete; run ./scripts/bootstrap.sh first." >&2
  exit 2
fi
for required in \
  "$ROOT/run.sh" \
  "$ROOT/scripts/run_slurm_node.sh" \
  "$ROOT/scripts/harbor_cli.py" \
  "$ROOT/scripts/slurm_gpu_guard.py" \
  "$ROOT/../evals_task/$TASK/prompt.md"; do
  if [[ ! -f "$required" ]]; then
    echo "Required controller file is missing: $required" >&2
    exit 2
  fi
done
for shell_script in \
  "$ROOT/submit_slurm.sh" \
  "$ROOT/run.sh" \
  "$ROOT/scripts/run_slurm_node.sh"; do
  if ! bash -n "$shell_script"; then
    echo "Refusing to submit because shell syntax validation failed: $shell_script" >&2
    exit 2
  fi
done
if ! "$ROOT/.venv/bin/python" -m py_compile \
  "$ROOT/launch.py" \
  "$ROOT/environments/apptainer.py" \
  "$ROOT/scripts/harbor_cli.py" \
  "$ROOT/scripts/slurm_gpu_guard.py" \
  "$ROOT/scripts/check_harbor_result.py"; then
  echo "Refusing to submit because Python syntax validation failed." >&2
  exit 2
fi
if ! command -v sbatch >/dev/null 2>&1; then
  echo "Slurm sbatch is unavailable on the submission host." >&2
  exit 2
fi
if ! command -v apptainer >/dev/null 2>&1; then
  echo "Apptainer is unavailable on the submission host." >&2
  exit 2
fi
if [[ "$APPTAINER_BUILD_SOURCE" == *.sif ]] && ! apptainer inspect "$APPTAINER_BUILD_SOURCE" >/dev/null 2>&1; then
  echo "Apptainer build source is not a valid image: $APPTAINER_BUILD_SOURCE" >&2
  exit 2
fi
if [[ "$ARM" == harness ]]; then
  if [[ -z "$CHAPERONE_PATH" ]]; then
    echo "--chaperone-path is required for the harness arm" >&2
    exit 2
  fi
  WORKSPACE_SEED=$(
    "$ROOT/.venv/bin/python" "$ROOT/scripts/snapshot_harness.py" \
      --chaperone-path "$CHAPERONE_PATH" \
      --output-root "$HARBOR_JOBS_DIR/workspace-seeds"
  )
fi
if [[ -z "${OPENAI_API_KEY:-}" && -z "${CODEX_AUTH_JSON_PATH:-}" && ! -s "$HOME/.codex/auth.json" ]]; then
  echo "Codex authentication is unavailable: set OPENAI_API_KEY or run 'codex login'." >&2
  exit 2
fi
if [[ "$ARM" == harness ]]; then
  printf 'Chaperone source: %s\n' "$CHAPERONE_PATH"
  printf 'Sanitized source bundle: %s\n' "$WORKSPACE_SEED"
fi

STAMP=$(date -u +%H%M%SZ)
JOB_NAME=${JOB_NAME:-chaperone-codex-${ARM}-slurm-${STAMP}}
if [[ ! "$JOB_NAME" =~ ^[A-Za-z0-9_.-]{1,100}$ ]]; then
  echo "JOB_NAME must contain only letters, digits, dot, underscore, and hyphen (maximum 100 characters)" >&2
  exit 2
fi
mkdir -p "$HARBOR_JOBS_DIR/slurm"

export ARM MODEL REASONING_EFFORT ATTEMPTS CONCURRENCY MEMORY_GB
export APPTAINER_BUILD_SOURCE JOB_NAME
if [[ "$ARM" == harness ]]; then
  export WORKSPACE_SEED
fi
printf -v WORKER_COMMAND '%q' "$ROOT/scripts/run_slurm_node.sh"

EXPORT_CANDIDATES=(
  HOME PATH USER LOGNAME SHELL LANG LC_ALL
  ARM MODEL REASONING_EFFORT ATTEMPTS CONCURRENCY
  MEMORY_GB
  APPTAINER_BUILD_SOURCE JOB_NAME WORKSPACE_SEED KEEP_RUN_IMAGE
  OPENAI_API_KEY CODEX_AUTH_JSON_PATH CODEX_FORCE_AUTH_JSON
  HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy
  SSL_CERT_FILE REQUESTS_CA_BUNDLE
)
EXPORT_NAMES=()
for name in "${EXPORT_CANDIDATES[@]}"; do
  if [[ -v "$name" ]]; then
    EXPORT_NAMES+=("$name")
  fi
done
SLURM_EXPORTS=$(IFS=,; echo "${EXPORT_NAMES[*]}")

SBATCH_ARGS=(
  sbatch
  --partition="$CPU_PARTITION"
  --time=24:00:00
  --mem="${MEMORY_GB}G"
  --cpus-per-task=4
  --job-name="$JOB_NAME"
  --output="$HARBOR_JOBS_DIR/slurm/%x-%j.out"
  --error="$HARBOR_JOBS_DIR/slurm/%x-%j.err"
  --export="$SLURM_EXPORTS"
  --wrap="$WORKER_COMMAND"
)

if [[ ${DRY_RUN:-0} == 1 ]]; then
  printf '%q ' "${SBATCH_ARGS[@]}"
  printf '\nOuter allocation: partition=%s time=24:00:00 memory=%sG GPUs=0\n' "$CPU_PARTITION" "$MEMORY_GB"
  printf '%s\n' 'Chaperone GPU policy: explicitly request 1-5 gpu:small GPUs per allocation (controller-pinned to A4000)'
  exit 0
fi

exec "${SBATCH_ARGS[@]}"
