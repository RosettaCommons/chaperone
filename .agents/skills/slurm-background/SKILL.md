---
name: slurm-background
description: "Submit BACKGROUND (detached, fire-and-forget) SLURM work that survives disconnect — for pilots, production, campaigns, long runs, and fanned-out job arrays. Covers the Chaperone ExecutionEngine, raw sbatch, and detaching a long driver (nohup/tmux), plus how to monitor and cancel. Use when the user says 'submit a job', 'run in background', 'production/pilot run', 'kick off a campaign', 'sbatch', 'fire and forget', or 'don't babysit it'. For a live shell on a node, use the slurm-interactive skill instead."
allowed-tools: ["Bash", "Read", "Grep", "Glob", "Write", "Edit"]
---

# SLURM Background Submission

Configure your cluster's partition names, GPU types, and shared run directories in `.env.local` or a file selected by `CHAPERONE_ENV_FILE`.

Run work **detached** on the cluster. Prefer one autonomous campaign driver that
contains its bounded theories, stages, scorer calls, and promotion rules. The
engine owns its inner jobs. Detached work survives disconnect and scales to many
jobs without reasoning-thread polling.

## When to use background (vs interactive)

Use **background** (this skill) when:
- The run is long — pilot / production / campaign (hundreds → 100k+ backbones).
- The Chaperone engine fans work into many chunks / parallel blocks.
- You don't want to babysit it and it must survive your session ending.

Use **interactive** (`slurm-interactive` skill) when you're iterating command-by-command on a node (testing a block, debugging on-GPU).

**Rule of thumb: iterating → interactive; scaling → background.**

## Three ways to submit

### 1. Chaperone ExecutionEngine (the primary path)

The engine submits each block to SLURM via submitit, auto-chunking rows across jobs. Pass `job_args` as a dict or a `JobArgs`; only the fields you set override the block's defaults:

```python
engine = ExecutionEngine(rundir="/path/to/shared/scratch/run")
engine.run(blocks, job_args={"mem_gb": 32, "time_limit": "24:00:00",
                             "rows_per_job": 10})
```

Leave `partition` and `gres` out unless you need to change them: the block's CPU/GPU class and the site settings (below) already give a valid request on your cluster.

| `job_args` field | Default | Notes |
|---|---|---|
| `partition` | block's class, mapped by the site | A real partition name, passed to `sbatch` as-is. `None` means `CHAPERONE_DEFAULT_PARTITION`, else the cluster default. |
| `gpu_class` | block default | Portable GPU size, `small` or `large`. The site maps it to a `gres` (`CHAPERONE_GRES_MAP`); unmapped it is `gpu:1`. |
| `gres` | from `gpu_class` | A literal `--gres` string, passed to `sbatch` unchecked, so valid GPU types are whatever your cluster registers. Beats `gpu_class`. |
| `account` / `qos` | site default | `--account` / `--qos`. Defaults: `CHAPERONE_DEFAULT_ACCOUNT` / `CHAPERONE_DEFAULT_QOS`. |
| `extra_sbatch` | site default | List of extra long `sbatch` options, e.g. `["--reservation=mine"]`. Replaces `CHAPERONE_DEFAULT_EXTRA_SBATCH`. |
| `mem_gb` | block default (8) | per job |
| `time_limit` | block default (`5:00:00`) | HH:MM:SS |
| `rows_per_job` / `num_chunks` | block default | parallel granularity |
| `cpus_per_task` | 1 | |
| `exclude` | None | comma-separated nodes to avoid |

Precedence, highest first: CLI flags (`--partition`, `--gres`, `--account`, `--qos`, `--extra_sbatch`), `job_args`, the block's `_resources`, the site settings. `--partition` and `--gres` only affect GPU blocks. A `--partition` override drops the GPU type from the block's `gres` (`gpu:a100:1` becomes `gpu:1`), so pass `--gres` too if your cluster requires a type.

### Site configuration

A site sets these once, in `.env.local` or a file selected by `CHAPERONE_ENV_FILE`. With none set, nothing cluster-specific is submitted: no `--partition`, and `--gres=gpu:1` for GPU blocks.

| Setting | Meaning |
|---|---|
| `CHAPERONE_DEFAULT_PARTITION` | Partition for CPU blocks, bare `JobArgs()`, and GPU blocks without a GPU partition |
| `CHAPERONE_GPU_PARTITION` | Partition for GPU blocks |
| `CHAPERONE_GRES_MAP` | JSON map from size class to `--gres`, e.g. `'{"small": "gpu:a4000:1", "large": "gpu:a6000:1"}'`. Key `default` covers unlisted classes. |
| `CHAPERONE_DEFAULT_ACCOUNT`, `CHAPERONE_DEFAULT_QOS` | Default `--account`, `--qos` |
| `CHAPERONE_DEFAULT_EXTRA_SBATCH` | Extra long `sbatch` options, shell-split |

Find your cluster's values with `sinfo -o "%P %G %l"`. A map set in the environment is merged key by key with one in a `.env*` file.


- **Parallel metric overlays:** `engine.run_parallel(...)` runs independent update blocks concurrently. Sequential design pipelines use `engine.run([...])` with a list of blocks.
- `engine.run` **blocks until the cluster jobs finish** (it is orchestrating them). To run the *driver itself* outside the model loop, use the Chaperone monitor's `launch_and_wait` tool (see #3).
- **Re-running resumes:** completed chunks are cached; only missing/failed chunks re-run. Use `--force_exec` to redo everything. (See the `scripting` skill for the full test → pilot → production flow.)

### 2. Raw `sbatch` (custom / non-chaperone scripts)

> Use this only for non-Chaperone scripts, and only where manual submission is allowed. When a task prompt or harness instruction says Chaperone/Submitit owns job submission, do not call `sbatch`, `srun` or `salloc` yourself; use the engine (section 1).

```bash
sbatch --partition=<gpu-partition> --gres=gpu:1 --mem=32G --time=24:00:00 \
       -J myjob -o logs/%x_%j.out --wrap "source .venv/bin/activate && ./run.sh"
```
Or a script with `#SBATCH` directives + `sbatch script.sh`. Returns a job id immediately; the job runs detached. Use a **job array** (`--array=0-99`) for many independent tasks.

### 3. Launch and wait for a long-running driver

When a pipeline driver runs for hours and submits and waits on many cluster
jobs, have it write a completion marker on success and a failure marker on
failure. Call `campaign-monitor.launch_and_wait` when the runtime provides the monitor, or
`chaperone-monitor.launch_and_wait` with a standalone `chaperone monitor-mcp`
server. Pass an argv array, the driver cwd, and at least one completion marker.
The tool launches the driver without a shell and blocks outside the model loop.
Do not improvise `nohup`, `setsid`, shell backgrounding, or a terminal-managed
process, and do not wait on each inner engine job.

Start the argv array with the active Chaperone Python/CLI or a workspace
executable. Do not use `/usr/bin/env` to prefix environment assignments: the
monitor rejects generic wrapper executables. Express smoke/pilot/production
selection through driver CLI flags or defaults instead.

For a non-agent shell or another process supervisor, an already-running process
can instead be observed with the equivalent CLI:

```bash
chaperone wait --process-id <pid> \
  --completion-marker <run>/driver.done \
  --failure-marker <run>/driver.failed
```

For a standalone Slurm submission, call `campaign-monitor.wait` when the runtime provides the monitor, or
`chaperone-monitor.wait` with a standalone server, passing `job_ids`; the CLI
equivalent is `chaperone wait --job-id <jobid>`. A marker-only wait is also
valid. The waiter keeps no registration or monitor state and returns only on
completion, failure, or an explicit timeout. Never implement a reasoning-loop
polling cadence.

## Choose a partition

List available partitions and GPU types with `sinfo -o "%P %G %l"`. Use your
site's recommended interactive or normal queue for bounded tests and pilots.
For production, use a preemptible/backfill queue only when the workload tolerates
interruptions. Re-invoking the engine resumes incomplete chunks.

## Diagnose and manage after a wait wake-up

Use these commands for bounded diagnosis or cancellation after the outer
waiter reports a terminal result. Do not run them on a fixed interval, do not
use `tail -f` from the reasoning thread, and never wake the model every five
minutes merely to report unchanged state.

```bash
squeue -u $USER                            # what's queued/running
squeue -u $USER -o "%i %j %T %P %M %N"     # detailed (id, name, state, partition, time, node)
sacct -j <jobid> -o JobID,State,Elapsed,MaxRSS,NodeList   # post-hoc accounting
scancel <jobid>                            # cancel one
scancel -u $USER -n <name>                 # cancel by name
tail -n 80 <rundir>/.pipeline/.../*.out       # logs (chaperone writes per-chunk logs under the rundir)
```
