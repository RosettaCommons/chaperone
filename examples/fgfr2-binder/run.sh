#!/usr/bin/env bash
# Launch the FGFR2 binder pipeline. The orchestrator runs inside a `srun` CPU
# allocation (login-node ulimits are too low for polars), and submits the
# RFD3/MPNN/AF3 jobs to the GPU partition.
#
# Usage:
#   ./examples/fgfr2-binder/run.sh                    # test  (1 design end-to-end, 1 batch * 1 size)
#   ./examples/fgfr2-binder/run.sh small              # 10 backbones (no test overrides — uses script defaults)
#   ./examples/fgfr2-binder/run.sh pilot              # ~200 backbones
#   ./examples/fgfr2-binder/run.sh prod               # full default scale
#
# Knobs (env vars):
#   GRES=gpu:a100:1   PARTITION=gpu   ORCH_TIME=2:00:00   ORCH_MEM=8G
#   ORCH_PARTITION=cpu                  # partition for the orchestrator job;
#                                       # default: $CHAPERONE_DEFAULT_PARTITION, else
#                                       # the cluster default
#   RUNDIR=/path/to/custom/dir         # override default runs/<mode> location
#   BG=1                                # detach (sbatch); script returns the job id
#                                       # and you can close your terminal
#
# Outputs go to <rundir>/run.log.
set -euo pipefail

MODE="${1:-test}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Don't pass --partition or --gres globally by default. Each block asks for a CPU
# or GPU job (RFD3/AF3 a GPU, MPNN none) and the site settings (CHAPERONE_GPU_PARTITION,
# CHAPERONE_GRES_MAP, ...) turn that into this cluster's partition and gres.
# If you DO pass them:
#   1. A global --partition applies to GPU blocks only and strips the GPU type from
#      their gres (e.g. gpu:large:1 → gpu:1). If your cluster requires a type, also
#      pass GRES.
#   2. A global --gres applies to GPU blocks only; CPU blocks like MPNN keep none.
PARTITION="${PARTITION:-}"
GRES="${GRES:-}"
ORCH_PARTITION="${ORCH_PARTITION-${CHAPERONE_DEFAULT_PARTITION:-}}"
ORCH_PARTITION_FLAG=()
if [ -n "$ORCH_PARTITION" ]; then
    ORCH_PARTITION_FLAG=(--partition="$ORCH_PARTITION")
fi
ORCH_TIME="${ORCH_TIME:-2:00:00}"
ORCH_MEM="${ORCH_MEM:-8G}"

# Outputs nest under the example directory so everything for one campaign
# stays together. $SCRIPT_DIR/runs/ is on shared FS (NFS) so SLURM workers can
# read/write back to it.
RUNS_BASE="${RUNS_BASE:-$SCRIPT_DIR/runs}"
case "$MODE" in
    test)
        RUNDIR="${RUNDIR:-$RUNS_BASE/test}"
        EXTRA_ARGS=("--test")
        ;;
    small)
        # Uses the script's pipeline.py defaults (n_batches=10, batch_size=1 → 10 designs).
        RUNDIR="${RUNDIR:-$RUNS_BASE/small}"
        EXTRA_ARGS=()
        ;;
    pilot)
        RUNDIR="${RUNDIR:-$RUNS_BASE/pilot}"
        EXTRA_ARGS=("--rfd3.n_batches=20" "--rfd3.batch_size=10")
        ;;
    prod)
        RUNDIR="${RUNDIR:-$RUNS_BASE/prod}"
        # NOTE: a site-provided backfill partition suits production, but a
        # global --partition strips the GPU type from each block's gres. Set both,
        # e.g. PARTITION=gpu-bf GRES=gpu:large:1, if your cluster needs a type.
        ORCH_TIME="${ORCH_TIME:-12:00:00}"
        EXTRA_ARGS=()
        ;;
    *)
        echo "Usage: $0 [test|pilot|prod]" >&2
        exit 2
        ;;
esac

mkdir -p "$RUNDIR"
LOG="$RUNDIR/run.log"

cat <<EOF
Mode:       $MODE
Rundir:     $RUNDIR
Log:        $LOG
Orch alloc: ${ORCH_PARTITION:-(cluster default)}  $ORCH_TIME  $ORCH_MEM
Job target: ${PARTITION:-(site/per-block partition)}  ${GRES:-(site/per-block gres)}
---
EOF

# Only forward --partition / --gres when explicitly set so we don't override
# the per-block defaults and site settings.
PIPELINE_ARGS=("${EXTRA_ARGS[@]}" --rundir="$RUNDIR")
if [ -n "$PARTITION" ]; then
    PIPELINE_ARGS+=(--partition="$PARTITION")
fi
if [ -n "$GRES" ]; then
    PIPELINE_ARGS+=(--gres="$GRES")
fi

cd "$REPO_ROOT"

if [ -n "${BG:-}" ]; then
    # Submit orchestrator as a detached SLURM batch job. The user can close the
    # terminal afterwards; check progress via `squeue -j <jobid>` and `tail -f $LOG`.
    SBATCH_SCRIPT="$RUNDIR/orchestrator.sbatch"
    cat > "$SBATCH_SCRIPT" <<EOF
#!/bin/bash
${ORCH_PARTITION:+#SBATCH --partition=$ORCH_PARTITION}
#SBATCH --time=$ORCH_TIME
#SBATCH --mem=$ORCH_MEM
#SBATCH --cpus-per-task=4
#SBATCH --job-name=fgfr2-orch
#SBATCH --output=$LOG
#SBATCH --error=$LOG
set -euo pipefail
cd $REPO_ROOT
./examples/fgfr2-binder/pipeline.py ${PIPELINE_ARGS[@]@Q}
EOF
    job_id=$(sbatch --parsable "$SBATCH_SCRIPT")
    echo "Submitted orchestrator job: $job_id"
    echo "Log:    $LOG"
    echo "Status: squeue -j $job_id"
    echo "Tail:   tail -f $LOG"
    exit 0
fi

srun ${ORCH_PARTITION_FLAG[@]+"${ORCH_PARTITION_FLAG[@]}"} --time="$ORCH_TIME" --mem="$ORCH_MEM" --cpus-per-task=4 \
    ./examples/fgfr2-binder/pipeline.py \
    "${PIPELINE_ARGS[@]}" \
    2>&1 | tee "$LOG"

status="${PIPESTATUS[0]}"
echo "---"
echo "Pipeline exited with status $status. Log: $LOG"
exit "$status"
