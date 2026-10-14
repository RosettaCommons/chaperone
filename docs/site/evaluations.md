# Running Evaluations

The evaluation harness in `src/chaperone-evals/` measures how well a coding agent
completes a protein or nucleic-acid design task. It runs the same task twice:

- **`harness`**: the agent gets the task inputs and a sanitized copy of Chaperone with
  its agent skills.
- **`baseline`**: the agent gets the task inputs in a clean workspace, with no Chaperone.

Each submission is then scored against held-out criteria. Comparing the two arms shows
what Chaperone adds.

## Tasks

| Task | Design problem |
|------|----------------|
| `antibody_cdr` | Design the six CDR loops of an antibody on a fixed framework, against hen egg-white lysozyme |
| `biotin_binder_design` | De novo mini-protein binders for a modified biotin ligand |
| `dna_binder` | De novo mini-proteins that read a specific DNA sequence through the major groove |
| `enzyme_glycosidase` | De novo retaining glycosidase around a six-residue catalytic theozyme |
| `enzyme_protease` | De novo zinc metalloprotease around a HEXXH motif |
| `nanoparticle_c3` | Protomer sequences that assemble into a C3-symmetric homotrimer |
| `peptide_binder` | De novo binders for a flexible 15-residue peptide |
| `ppi_pdl1` | De novo binders for human PD-L1 |
| `ribozyme_design` | RNA sequences that rebuild a hammerhead ribozyme's catalytic core into a single-chain fold |
| `rna_binder` | De novo mini-proteins that bind a structured RNA stem-loop |

Each task is a folder under `src/chaperone-evals/evals_task/<task>/`. `prompt.md` and
`inputs/` are exactly what the agent receives. `score/` and `example/` (the scorer and
a reference submission) live in a separate private repository on the controller
host and are never shown to the agent. The prompt
states what a design must achieve. The scorer applies its own gates on top, and does
not disclose them.

## Before you start

You need a Slurm cluster with Apptainer, and a submission host that provides `uv`,
`apptainer`, `sbatch`, `squeue` and `sacct`. Run everything from the login or
submission host.

The scorers, reference submissions and audit patterns are held in the private
[Ubiquinone-dot/chaperone-evals repository](https://github.com/Ubiquinone-dot/chaperone-evals).
Clone it on the trusted controller, outside this checkout and every agent/worker
mount, verify its checksums, and configure the controller:

```bash
umask 077
git clone https://github.com/Ubiquinone-dot/chaperone-evals.git /private/chaperone-evals
(cd /private/chaperone-evals && sha256sum --check SHA256SUMS)
export CHAPERONE_EVALS_PRIVATE=/private/chaperone-evals/evals_task
```

Access requires authorization to the private repository. The private Harbor Hub
dataset `ubiquinone-dot/chaperone-evals@release-split-20261014` is an alternative;
download and restore instructions are in `src/chaperone-evals/README.md`.
Keep private data and registry credentials outside all agent mounts. Removing
answers from the current Chaperone tree does not remove copies in old commits
or retained pull-request refs; clean those before making the repository public.

## One-time setup

```bash
cd src/chaperone-evals
./harbor-controller/scripts/bootstrap.sh
```

Bootstrap installs the harness project (`uv pip install -e src/chaperone-evals`) into the
checkout's `.venv`, on top of anything already installed, and links the controller to
it. That project depends on the checkout's own `chaperone-pipelines` package, so there is no
separate environment. Then authenticate:

```bash
export OPENAI_API_KEY=...        # or use an existing `codex login`
```

The controller lives inside the Chaperone checkout it evaluates and uses that
checkout for scoring. Set `CHAPERONE_REPO` to score against a different one.

## Cluster configuration

The default `--site portable` has no digs images or scientific mounts. Copy
[`portable.example.json`](https://github.com/RosettaCommons/chaperone/blob/main/src/chaperone-evals/harbor-controller/sites/portable.example.json)
to a local file, replace its paths and resource names, and pass
`--site-config /absolute/path/to/site.json`. All image, asset, checkout, and run
paths must be visible on the submission host and compute nodes. The example is
a starting configuration, not a complete set of assets for every task.

The JSON selects `agent_image`, `worker_image`, `baseline_image`, the interpreter
inside the scientific images (`worker_python`), and the scheduler policy:
`cpu_partition`, `gpu_partition`, `gpu_type`, `gpu_constraint`, `account`, `qos`,
`max_total_gpus`, and `max_requested_gpus_per_submission`. An empty CPU partition omits `--partition`; an empty GPU partition inherits the
CPU partition, matching Chaperone core settings. Empty `gpu_type` requests `gpu:N`; empty constraint omits
`--constraint`. The controller alone applies account, QoS and constraint.

`runtime_binds` is the complete read-only allowlist of host runtime/asset paths;
entries are absolute paths or `[source, target]` pairs. Include `/dev/fuse` and the
required scientific assets. Slurm authentication stays on the host: native Slurm
clients, libraries, configuration and MUNGE sockets are rejected as runtime mounts.
Agent scheduler requests use the trial-scoped broker. `environment`
contains allowlisted scientific settings. `capabilities` names the paths used by
preflight to verify the task's required tools; declaring a capability does not
install it or mount it. Paths must also be covered by `runtime_binds`. Never mount
a parent of the private task/scorer tree. Broad `/shared`, `/projects`, or home
mounts would defeat evaluation isolation.

An optional JSON `env_file` reads allowlisted scientific settings from a supplied
`.env.ipd` or another dotenv file. JSON `environment` overrides those settings.
Workspace-control settings remain excluded.
Scheduler fields in the JSON remain authoritative so workers and the submission
guard agree. Core Chaperone also accepts an untracked `.env.ipd` in the checkout
root, independently of the eval policy.

For the previous digs setup, pass `--site digs`. This retains the existing agent
and baseline images, asset paths, CPU/GPU queues, `gpu:small:N`, the **A4000**
constraint, five concurrently reserved GPUs per trial, and thirty requested GPUs
per submission across all array tasks. To use B4000s instead, supply an override
JSON with `{"gpu_constraint": "B4000"}` after confirming the scheduler's feature
name and using images compatible with that hardware. `--site digs --site-config
...` merges overrides into the digs preset. `CHAPERONE_EVAL_SITE` and
`CHAPERONE_EVAL_SITE_CONFIG` provide equivalent defaults for all controller entry
points. The controller passes these to its Slurm launcher and external scorer.

Both profiles retain trial-scoped cancellation, guarded submission, GPU accounting,
array limits, baseline image auditing, and private scorer isolation. The instructions
shown to each agent are generated from the selected policy. `--image` overrides the
selected arm's image and `--worker-image` overrides the scoring/scientific worker image.

Scoring reads `HBPLUS_COPY`, `CCD_MIRROR_PATH`, `PYROSETTA_PATH`, `X3DNA_BIN_DIR`, and
`X3DNA_DIR` from the selected profile, with shell overrides, and verifies the paths
before starting. Biotin requires only `HBPLUS_COPY` from that list; other configured
paths are checked when supplied. Other tasks retain the full preflight requirements.
No checkout `.env.ipd` is required. Scientific tasks still need
licensed tools, model weights, and their block containers; the controller cannot
make those external assets available by itself.

If the login host cannot execute Apptainer, set `"runtime_probe_on_compute": true`
in the site JSON. Runtime preflight then uses a short CPU-only `srun` allocation
in the configured CPU partition/account/QoS, with temporary files under the shared
run directory. The main controller still launches from the submission host.

### Building runtime images outside digs

From the repository root on a Linux host with Apptainer build support:

```bash
apptainer build --fakeroot agent.sif src/chaperone-evals/harbor-controller/containers/agent.def
apptainer build --fakeroot baseline.sif src/chaperone-evals/harbor-controller/containers/baseline.def
```

These recipes copy only framework packages into the scientific image and no
repository files into the baseline. Set both `agent_image` and `worker_image` to
the resulting scientific SIF and `baseline_image` to the clean SIF. The scientific recipe includes non-setuid Apptainer for nested block containers,
following its [official installation guide](https://apptainer.org/docs/admin/1.4/installation.html).
The compute nodes must permit unprivileged user namespaces and FUSE. Native Linux
Codex must be installed on the host; the controller mounts its executable.
Build scientific block containers separately with `chaperone install`; the runtime
image does not include model containers or weights. Both supplied recipes were
built on Linux, and the
fresh scientific image passed clean framework imports, block discovery and the
CPU functional smoke covering cache, metrics, selection and CatCIF roundtrips.
Nested Apptainer execution also passed with `/dev/fuse` bound, matching the
controller's default runtime binds. Preserve `/dev/fuse` when supplying a custom
`runtime_binds` list.
These checks do not constitute a full LLM agent trial or a scientific benchmark.
Validate nested GPU execution and the complete evaluation on your target cluster
before using it for benchmark results.

A trial whose agent already wrote `submission/designs.csv` is scored even if Harbor's
result record is failed or missing, because the agent's work is finished at that point.

## Run an evaluation

Start with a small smoke run:

```bash
cd src/chaperone-evals/harbor-controller

./run_eval.py biotin_binder_design \
  --arm harness --site-config /absolute/path/to/site.json \
  --chaperone-path /absolute/path/to/chaperone \
  --reasoning-effort low \
  --required-n 10 \
  --run-name biotin-harness-smoke
```

For the baseline arm, leave out `--chaperone-path`:

```bash
./run_eval.py biotin_binder_design \
  --arm baseline --site-config /absolute/path/to/site.json \
  --reasoning-effort low \
  --required-n 10 \
  --run-name biotin-baseline-smoke
```

The launcher checks your setup, submits a CPU-only Slurm job that runs the agent in an
isolated Apptainer workspace, and starts a detached watcher that scores the result
when the agent finishes. It returns once both are accepted.

| Option | Purpose |
|--------|---------|
| `--arm {harness,baseline}` | Which arm to run. The harness arm needs `--chaperone-path`. |
| `--chaperone-path PATH` | Chaperone working tree handed to the agent (harness arm only). |
| `--chaperone-repo PATH` | Host checkout used for scoring. Never shown to the agent. |
| `--required-n N` | Number of designs to submit. Sets both the prompt and the scorer. Lower it for smoke tests. |
| `--attempts N`, `--concurrency N` | Repeat or parallelize independent trials. |
| `--model`, `--reasoning-effort` | Override the model and effort. |
| `--run-dir PATH` | Root for results, logs and snapshots. |
| `--run-name NAME` | Name for the run. |
| `--memory-gb N`, `--time-limit HH:MM:SS` | Size the outer allocation (64 GB by default). |
| `--exclude-nodes LIST` | Slurm nodes to avoid for every GPU job in the trial. |
| `--private-tasks-path PATH` | Use scorers and audit patterns from another `evals_task/` directory. Same as `CHAPERONE_EVALS_PRIVATE`. |
| `--no-score` | Skip the held-out scorer. |
| `--skip-image-digests` | Record image identity without hashing the images. Hashing is cached, so this only avoids the first launch's one-time cost. |
| `--wait` | Stay attached until the run and its scoring finish. |
| `--dry-run` | Check the configuration without submitting. |

`./run_eval.py --help` lists every option.

## Results

The launcher prints the Slurm job id, the result directory and the scoring watcher's
receipt. Portable runs default to `harbor-controller/jobs`; digs uses
`/net/scratch/$USER/chaperone-evals/runs` when available. `--run-dir` overrides it. Under the run root:

```text
slurm/<run-name>-<job-id>.{out,err}
host-scorers/<run-name>-<job-id>.{json,out,err}
results/<run-name>-<job-id>/<trial>/
```

A finished trial holds:

```text
artifacts/workspace/submission/designs.csv    what the agent submitted
controller/audit.json
controller/run_meta.json
scoring/score.json                            the verdict
scoring/score_per_design.csv                  one row per design
```

Check the allocation with `squeue -j <job-id>`. A run is complete when the watcher has
exited and `scoring/score.json` exists. A non-null `harness_fault` in that file means
the scientific oracle did not produce a valid structure and the trial is void. Do not
count it as an ordinary zero-pass trial.

Every launch also writes a provenance record to `<run root>/manifests/<run-name>-<timestamp>.json`, outside everything the agent can read. It holds the controller commit and dirty flag, the Harbor version, the image digests, the sanitized bundle's hash, the hashes of the rendered prompt and instruction and of the public inputs, the resolved arguments, and the scorer's hash. It never contains the private tree's location or the scorer's path. After submission it also records the Slurm job id and result directory.

Helper scripts live in `harbor-controller/scripts/`:

```bash
./scripts/eval_status.py <pattern>             # progress of matching runs
./score_submission.py --job <run-dir> --task <task>   # score a finished run by hand
./scripts/rescore.py --task <task> --run-dir <run-dir>   # score trials that have no result yet
```

## What the agent can see

The harness arm's workspace is built fresh for each launch from an allowlist. It holds
`src/blocks`, `src/chaperone` and `src/chaperone_cli`, the agent skills, the block
reference and a few root files. It contains no Git history, tests, examples, campaigns,
earlier outputs, and nothing under `src/chaperone-evals/`. The bundle is validated
again after it is built, and rejected if any file matches a pattern from
`audit_hints.json`.

Compute is capped by the selected site policy across all Slurm jobs and arrays
in a trial (digs defaults to five concurrent `gpu:small` GPUs). Held-out scoring jobs are not counted against the agent.

## Troubleshooting

| Message | Cause and fix |
|---------|---------------|
| `No held-out scorers found` | The task folders have no `score/`. Set `CHAPERONE_EVALS_PRIVATE` or pass `--private-tasks-path`. |
| `Held-out audit patterns not found` | The tree has no `audit_hints.json`. A harness bundle cannot be built without it. |
| `Public and private inputs/ differ` | `CHAPERONE_EVALS_PRIVATE` points at a tree whose `inputs/` no longer match the ones in this repository. Sync them. |
| `No scorer for task` | The task name is wrong or has no `score/score.py`. |
| `Scoring site-config preflight failed` | A site tool or data path in the selected eval profile is missing or not executable on this host. Fix the path, or override it in your shell. |
| `Refusing to overwrite existing scoring folder` | The trial already has a `scoring/` folder. `rescore.py` only handles unscored trials; add `--clear-empty-scoring` if the folder is empty. |

## Tests

```bash
cd src/chaperone-evals
PYTHONPATH=harbor-controller ../../.venv/bin/python -m pytest harbor-controller/tests scoring
```

This needs `pytest` (the harness project's `test` extra). The repository's own `pytest` only collects
`tests/`, so it never runs these.
