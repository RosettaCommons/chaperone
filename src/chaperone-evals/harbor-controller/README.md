# Harbor controller

This directory launches Chaperone evaluations through Harbor on Slurm. Run it
from a login/submission host; `run_eval.py` is the supported entry point.

Evaluation tasks live in `../evals_task/<task-name>/` and provide the model prompt
and public inputs. The held-out scorer (`score/`) and reference submission
(`example/`) sit in the same task folder, are read only by the controller, and are
never exposed to the model. `CHAPERONE_EVALS_PRIVATE` moves them elsewhere (see
`../README.md`).

## How a run works

1. The controller validates the task, authentication, containers, Slurm, and
   the requested Chaperone source.
2. It submits a CPU-only outer Slurm job that runs Harbor and Codex inside an
   isolated Apptainer workspace.
3. Work submitted by the agent goes through a guarded Slurm broker. The trial
   can access only its workspace and explicitly assigned scientific assets.
4. After Harbor exits, a detached host-side watcher audits each finished trial
   and runs the task's held-out scorer for successful submissions. Scientific
   scoring blocks may submit their own Slurm workers.

There are two evaluation arms:

- `harness` receives task inputs plus a sanitized copy of the Chaperone source
  and agent skills. It is the default arm.
- `baseline` receives task inputs in a clean workspace without Chaperone
  source, skills, or scientific assets.

Harness launches require `--chaperone-path`. The controller copies only
allowlisted source and instruction files into a content-addressed bundle under
`$HARBOR_JOBS_DIR/workspace-seeds/`, then copies that bundle into each trial as
`/workspace/chaperone`. Git metadata, tests, examples, evaluations, generated
outputs, and prior submissions are excluded. There is no persistent
`harbor-controller/snapshots/` directory or default source checkout.

## One-time setup

From `src/chaperone-evals`:

```bash
./harbor-controller/scripts/bootstrap.sh
```

Then enter the controller directory and provide authentication before launching:

```bash
cd harbor-controller
export OPENAI_API_KEY=...
```

An existing `codex login` is also accepted. The submission host must provide
`uv`, Apptainer, and the Slurm commands `sbatch`, `squeue`, and `sacct`.
Bootstrap installs `../pyproject.toml` (the checkout plus the controller's
dependencies) into the checkout's own `.venv` and links this directory to it; no virtual environment is shared between users.

## Select a cluster

The default profile is portable and requires your own runtime images and asset
paths. Pass `--site-config /absolute/path/to/site.json` using
`harbor-controller/sites/portable.example.json` (relative to the eval project)
as a starting point. On digs, use `--site digs` to retain the existing images,
GPU usage accounting and A4000 policy. Add a site JSON override to select another
GPU constraint such as B4000. Optional `env_file` imports scientific settings from
a supplied `.env.ipd`; the controller JSON owns the scheduler policy.

See the [cluster configuration and image build guide](https://github.com/RosettaCommons/chaperone/blob/main/docs/site/evaluations.md#cluster-configuration)
for the full configuration, image recipes, and external-cluster prerequisites.

## Launch a job

Harness:

```bash
./run_eval.py rna_binder \
  --arm harness --site-config /absolute/path/to/site.json \
  --chaperone-path /absolute/path/to/chaperone \
  --run-dir /absolute/path/to/eval-runs \
  --run-name rna-binder-harness
```

Baseline:

```bash
./run_eval.py rna_binder \
  --arm baseline --site-config /absolute/path/to/site.json \
  --run-name rna-binder-baseline
```

Do not pass `--chaperone-path` to the baseline arm. By default, the launcher
returns after submitting Harbor and starting the detached scoring watcher. Add
`--wait` to stay attached until both finish, or `--dry-run` to run preflight
without submitting.

Common options:

| Option | Purpose |
|---|---|
| `--model`, `--reasoning-effort` | Override the default model or effort. |
| `--attempts N`, `--concurrency N` | Run multiple Harbor attempts. |
| `--run-dir PATH` | Choose the shared root for results, logs, wrappers, and snapshots. |
| `--required-n N` | Change the required submission size for both prompt and scorer. |
| `--memory-gb N`, `--time-limit HH:MM:SS` | Change the outer Slurm allocation (memory defaults to 64 GB). |
| `--no-score` | Skip automatic held-out scoring. |

Run `./run_eval.py --help` for the complete option list.

Inside a harness trial, detached work uses the `campaign-monitor.wait` MCP tool.
It is provided by the Chaperone harness and blocks outside the model loop until
the Slurm job, process marker, or explicit timeout is terminal. Baseline trials
receive no monitoring MCP server.

The same Chaperone-owned MCP server exposes `campaign-monitor.inspect`. It
returns bounded `summary`, `failures`, `funnel`, or `candidates` views derived
from the campaign's ordinary manifests, execution records, logs, score tables,
and submissions. Inspection is read-only and creates no controller or campaign
ledger state.

## Monitor and collect results

The launcher prints the outer Slurm job ID, result directory, and scoring
watcher receipt. Use the printed paths; by default the run root is
`/net/scratch/$USER/chaperone-evals/runs` when that scratch directory exists,
otherwise it is `harbor-controller/jobs`. Pass `--run-dir PATH` to choose
another run root. `HARBOR_JOBS_DIR` remains available as an environment default;
the CLI flag takes precedence.

Useful locations beneath the run root are:

```text
slurm/<run-name>-<job-id>.{out,err}
host-scorers/<run-name>-<job-id>.{json,out,err}
results/<run-name>-<job-id>/<trial>/
```

The final trial artifacts are:

```text
artifacts/workspace/submission/designs.csv
controller/audit.json
controller/run_meta.json
scoring/score.json
scoring/score_per_design.csv
```

Check the outer allocation with `squeue -j <job-id>`. A successfully scored run
is complete when the detached watcher has exited and `scoring/score.json`
exists for the trial.
