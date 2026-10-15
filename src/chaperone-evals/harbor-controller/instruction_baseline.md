Follow the appended task prompt. It is the authority for the scientific goal,
acceptance criteria, independence requirements, and `designs.csv` schema. You
have {{AGENT_HOURS}} hours to deliver exactly {{REQUIRED_N}} genuine designs.

## Baseline environment

This is the clean-room baseline arm. Work only in the otherwise blank writable
`/workspace`; the task's public files are in `inputs/`. Do not search the host
filesystem for evaluation assets or access another arm's workspace.

No scientific software, weights, databases, caches, examples, or environment
is supplied. Public network access and generic development tools are available.
Choose and obtain suitable open-source tools, and keep all source, setup files,
environments, weights, caches, scripts, logs, intermediates, and results under
`/workspace`. The workflow must be reproducible from that directory without
host-provided scientific assets or inaccessible services.

The outer process has four CPU cores and no GPU. Run only lightweight CPU work
locally; send larger, uncertain-memory, or GPU work to guarded Slurm.

## Guarded Slurm

Write each job script beneath `/workspace` and submit it with
`sbatch /workspace/JOB.sh`. Every script must use `#SBATCH --option=value`
directives for `--job-name`, `--output`, `--error`, `--time`,
`--mem`, and `--cpus-per-task`, and all referenced paths must remain beneath
`/workspace`.

{{SLURM_POLICY}}

- Workers use the same clean baseline image and `/workspace`; install every
  dependency they need there.
- `squeue` and `scancel` are trial-scoped. `sacct`, `sstat`, `sinfo`,
  `scontrol`, raw `srun`, and `salloc` are unavailable.

Preserve the job scripts and implement a self-contained, rerunnable workflow
covering generation, prediction, scoring, filtering, diversity checks, and
final validation. Use the public task criteria as internal gates, begin with a
small end-to-end test, and spend the compute ceiling only where results justify
it. The held-out scorer is unavailable until after submission; do not try to
locate or reconstruct it.

## Submission

Write the final table to `submission/designs.csv`. The task prompt's columns
are a strict allowlist: do not add metrics, structures, coordinates, or paths.
Submit only genuine designs satisfying its independence rules; if the full set
does not exist by the deadline, submit the genuine partial set without padding.
This sequence table is the only evaluation deliverable. Harbor retains the
transcript, and the controller measures compute and publishes audit metadata.
Legacy auxiliary files are ignored.
