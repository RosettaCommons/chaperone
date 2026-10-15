"""Scheduler authentication belongs exclusively to the trusted host broker."""

from pathlib import Path

BROKER_COMMANDS = ("sbatch", "scancel", "scontrol", "srun", "squeue", "sacct")
# Workers receive allocation metadata, never bearer credentials or client config.
WORKER_SLURM_ENV = (
    "SLURM_JOB_ID",
    "SLURM_JOBID",
    "SLURM_ARRAY_JOB_ID",
    "SLURM_ARRAY_TASK_ID",
    "SLURM_ARRAY_TASK_COUNT",
    "SLURM_ARRAY_TASK_MIN",
    "SLURM_ARRAY_TASK_MAX",
    "SLURM_JOB_NAME",
    "SLURM_JOB_NODELIST",
    "SLURM_NODELIST",
    "SLURM_JOB_NUM_NODES",
    "SLURM_NNODES",
    "SLURM_NTASKS",
    "SLURM_NTASKS_PER_NODE",
    "SLURM_TASKS_PER_NODE",
    "SLURM_PROCID",
    "SLURM_LOCALID",
    "SLURM_NODEID",
    "SLURM_CPUS_PER_TASK",
    "SLURM_CPUS_ON_NODE",
    "SLURM_JOB_CPUS_PER_NODE",
    "SLURM_JOB_GPUS",
    "SLURM_STEP_GPUS",
    "SLURM_GPUS",
    "SLURM_GPUS_ON_NODE",
    "SLURM_GPUS_PER_TASK",
    "SLURM_GPUS_PER_NODE",
    "SLURM_SUBMIT_DIR",
    "SLURM_RESTART_COUNT",
)
SCHEDULER_PATHS = (
    "/etc/slurm",
    "/etc/slurm-llnl",
    "/etc/munge",
    "/run/munge",
    "/run/slurm",
    "/run/slurmd",
    "/run/sackd",
    "/var/run/munge",
    "/var/run/slurm",
    "/var/spool/slurm",
    "/var/spool/slurmd",
    "/usr/lib/slurm",
    "/usr/lib/x86_64-linux-gnu/libmunge.so.2",
    *(
        f"/usr/bin/{name}"
        for name in (*BROKER_COMMANDS, "sinfo", "sstat", "munge", "unmunge")
    ),
)


def validate_runtime_binds(entries):
    """Reject direct, aliased, and ancestor mounts of scheduler credentials/code."""
    protected = {Path(p) for p in SCHEDULER_PATHS}
    protected |= {p.resolve() for p in protected}
    for entry in entries:
        for raw in [entry] if isinstance(entry, str) else entry:
            for candidate in {Path(raw), Path(raw).resolve()}:
                if any(
                    candidate == p
                    or candidate.is_relative_to(p)
                    or p.is_relative_to(candidate)
                    for p in protected
                ):
                    raise ValueError(
                        f"Scheduler authentication/runtime bind is forbidden: {raw}; use the broker"
                    )
