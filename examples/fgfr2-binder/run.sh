#!/usr/bin/env bash
# Launch the FGFR2 binder pipeline. The orchestrator runs inside a `srun` CPU
# allocation (login-node ulimits are too low for polars), and submits the
# RFD3/MPNN/AF3 jobs to the GPU partition.
#
# Usage:
#   ./examples/fgfr2-binder/run.sh                    # test  (1 design end-to-end, 1 batch * 1 size)
#   ./examples/fgfr2-binder/run.sh small              # 10 backbones (no test overrides — uses script defaults)
#   ./examples/fgfr2-binder/run.sh pilot              # ~200 backbones on gpu
#   ./examples/fgfr2-binder/run.sh prod               # full default scale on gpu-bf
#
# Knobs (env vars):
#   GRES=gpu:a100:1   PARTITION=gpu   ORCH_TIME=2:00:00   ORCH_MEM=8G
#   RUNDIR=/path/to/custom/dir         # override default runs/<mode> location
#   BG=1                                # detach (sbatch); script returns the job id
#                                       # and you can close your terminal
#
# Outputs go to <rundir>/run.log.
set -euo pipefail

MODE="${1:-test}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Don't pass --partition or --gres globally by default. Each block's _resources
# already declares the right partition+gres (RFD3/AF3 → gpu:a4000:1, MPNN →
# cpu with no gres). Two known gotchas if you DO pass them:
#   1. A global --partition strips the GPU type from each block's default gres
#      (e.g. gpu:a4000:1 → gpu:1) — clusters that require an explicit type reject this.
#   2. A global --gres is bolted onto MPNN's cpu submission, producing an
#      invalid partition=cpu + gres=gpu:a4000:1 combo.
# Set PARTITION/GRES env vars only when you genuinely need to retarget
# (e.g. PARTITION=gpu-bf for prod), and accept that you may need per-block tuning.
PARTITION="${PARTITION:-}"
GRES="${GRES:-}"
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
        # NOTE: gpu-bf would be ideal here, but a global --partition strips
        # block-default GPU types and clusters that require an explicit type
        # reject the resulting gpu:1. Keep block defaults for now; if you need
        # gpu-bf, set both PARTITION=gpu-bf and GRES=gpu:a4000:1 (this re-
        # introduces the MPNN-leak issue — work around with per-block job_args).
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
Orch alloc: cpu  $ORCH_TIME  $ORCH_MEM
Job target: ${PARTITION:-(per-block partition)}  ${GRES:-(per-block gres)}
---
EOF

# Only forward --partition / --gres when explicitly set so we don't override
# the per-block defaults that this cluster needs.
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
#SBATCH --partition=cpu
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

srun --partition=cpu --time="$ORCH_TIME" --mem="$ORCH_MEM" --cpus-per-task=4 \
    ./examples/fgfr2-binder/pipeline.py \
    "${PIPELINE_ARGS[@]}" \
    2>&1 | tee "$LOG"

status="${PIPESTATUS[0]}"
echo "---"
echo "Pipeline exited with status $status. Log: $LOG"
exit "$status"
