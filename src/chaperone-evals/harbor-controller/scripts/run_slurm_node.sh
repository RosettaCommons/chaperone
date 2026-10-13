#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE=${APPTAINER_BUILD_SOURCE:?Launch through run_eval.py to select an explicit site and image}
NODE_TMP_ROOT=${SLURM_TMPDIR:-/tmp}

if [[ ! -d "$NODE_TMP_ROOT" ]]; then
  echo "Node-local temporary directory does not exist: $NODE_TMP_ROOT" >&2
  exit 2
fi
if [[ "$SOURCE" != *://* ]]; then
  SOURCE=$(realpath "$SOURCE")
  if [[ ! -f "$SOURCE" ]]; then
    echo "Apptainer build source not found: $SOURCE" >&2
    exit 2
  fi
fi

SAFE_JOB_NAME=${JOB_NAME:-chaperone-codex}
SAFE_JOB_NAME=${SAFE_JOB_NAME//[^A-Za-z0-9_.-]/_}
RUN_IMAGE_DIR="${HARBOR_JOBS_DIR:-$ROOT/jobs}/images/${SAFE_JOB_NAME}-${SLURM_JOB_ID:-manual}"
RUN_IMAGE="$RUN_IMAGE_DIR/chaperone_agent.sif"
APPTAINER_TMPDIR=
OWN_RUN_IMAGE=1
LOCAL_RUN_SCRIPT=

cleanup() {
  if [[ -n "${APPTAINER_TMPDIR:-}" && -d "$APPTAINER_TMPDIR" ]]; then
    /usr/bin/rm -rf -- "$APPTAINER_TMPDIR"
  fi
  if [[ "$OWN_RUN_IMAGE" == 1 && ${KEEP_RUN_IMAGE:-0} != 1 && -d "$RUN_IMAGE_DIR" ]]; then
    /usr/bin/rm -rf -- "$RUN_IMAGE_DIR"
  fi
  if [[ -n "${LOCAL_RUN_SCRIPT:-}" && -f "$LOCAL_RUN_SCRIPT" ]]; then
    /usr/bin/rm -f -- "$LOCAL_RUN_SCRIPT"
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ "$SOURCE" == *.sif && -f "$SOURCE" ]]; then
  RUN_IMAGE="$SOURCE"
  OWN_RUN_IMAGE=0
  echo "Using assigned Apptainer image directly: $RUN_IMAGE"
elif [[ -f "$RUN_IMAGE" ]] && apptainer inspect "$RUN_IMAGE" >/dev/null 2>&1; then
  echo "Reusing valid per-run Apptainer image: $RUN_IMAGE"
else
  if [[ -e "$RUN_IMAGE_DIR" ]]; then
    echo "Incomplete per-run image directory exists: $RUN_IMAGE_DIR" >&2
    exit 2
  fi
  mkdir -p "$RUN_IMAGE_DIR"
  APPTAINER_TMPDIR=$(mktemp -d "$NODE_TMP_ROOT/harbor-apptainer-build-${SLURM_JOB_ID:-manual}.XXXXXX")
  export APPTAINER_TMPDIR
  echo "Building per-run Apptainer image on $(hostname): $RUN_IMAGE"
  echo "Using node-local build scratch: $APPTAINER_TMPDIR"
  apptainer build "$RUN_IMAGE" "$SOURCE"
fi
apptainer inspect "$RUN_IMAGE" >/dev/null
if [[ -n "$APPTAINER_TMPDIR" ]]; then
  /usr/bin/rm -rf -- "$APPTAINER_TMPDIR"
  APPTAINER_TMPDIR=
  unset APPTAINER_TMPDIR
fi

echo "Starting Harbor with CPU-only outer allocation; Chaperone GPU work uses guarded sbatch."
echo "Evaluation task inputs: ${TASK_INPUTS:-unset}"
export APPTAINER_IMAGE="$RUN_IMAGE"
export GPUS=0
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  export JOB_NAME="${JOB_NAME:-chaperone-codex}-${SLURM_JOB_ID}"
fi
# Bash may lazily read a script for the lifetime of a day-long job. Stage the
# small controller entry point on node-local storage so a transient NFS handle
# loss at shutdown cannot turn a scientifically successful run into a failure.
LOCAL_RUN_SCRIPT=$(mktemp "$NODE_TMP_ROOT/harbor-run-${SLURM_JOB_ID:-manual}.XXXXXX.sh")
/usr/bin/cp -- "$ROOT/run.sh" "$LOCAL_RUN_SCRIPT"
/usr/bin/chmod 0500 "$LOCAL_RUN_SCRIPT"
export HARBOR_CONTROLLER_ROOT="$ROOT"
"$LOCAL_RUN_SCRIPT" local "${ARM:-harness}"
