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
a reference submission) stay on the host and are never shown to the agent. The prompt
states what a design must achieve. The scorer applies its own gates on top, and does
not disclose them.

## Before you start

You need a Slurm cluster with Apptainer, and a submission host that provides `uv`,
`apptainer`, `sbatch`, `squeue` and `sacct`. Run everything from the login or
submission host.

The scorers, reference submissions and audit patterns ship in the task folders, so
no extra checkout is needed. Keep this repository private while they are in it,
because an agent with network access could read anything public. To keep them
elsewhere, point `CHAPERONE_EVALS_PRIVATE` at a separate `evals_task/` directory.

## One-time setup

```bash
cd src/chaperone-evals
./harbor-controller/scripts/bootstrap.sh
```

Bootstrap installs the harness project (`uv pip install -e src/chaperone-evals`) into the
checkout's `.venv`, on top of anything already installed, and links the controller to
it. That project depends on the checkout's own `pipelines` package, so there is no
separate environment. Then authenticate:

```bash
export OPENAI_API_KEY=...        # or use an existing `codex login`
```

The controller lives inside the Chaperone checkout it evaluates and uses that
checkout for scoring. Set `PIPELINES_REPO` to score against a different one.

Scoring reads the site tool locations (`HBPLUS_COPY`, `CCD_MIRROR_PATH`, `PYROSETTA_PATH`,
`X3DNA_BIN_DIR`, `X3DNA_DIR`) from the checkout's checked-in `.env` and `.env.ipd`, and checks
that they exist before it starts. A value set in your shell wins. The sanitized scoring
snapshot carries no `.env` files, so if the checkout it is linked to is not the one with
the site config, set `PIPELINES_SITE_CONFIG_REPO` to that checkout.

A trial whose agent already wrote `submission/designs.csv` is scored even if Harbor's
result record is failed or missing, because the agent's work is finished at that point.

## Run an evaluation

Start with a small smoke run:

```bash
cd src/chaperone-evals/harbor-controller

./run_eval.py biotin_binder_design \
  --arm harness \
  --pipelines-path /absolute/path/to/chaperone \
  --reasoning-effort low \
  --required-n 10 \
  --run-name biotin-harness-smoke
```

For the baseline arm, leave out `--pipelines-path`:

```bash
./run_eval.py biotin_binder_design \
  --arm baseline \
  --reasoning-effort low \
  --required-n 10 \
  --run-name biotin-baseline-smoke
```

The launcher checks your setup, submits a CPU-only Slurm job that runs the agent in an
isolated Apptainer workspace, and starts a detached watcher that scores the result
when the agent finishes. It returns once both are accepted.

| Option | Purpose |
|--------|---------|
| `--arm {harness,baseline}` | Which arm to run. The harness arm needs `--pipelines-path`. |
| `--pipelines-path PATH` | Chaperone working tree handed to the agent (harness arm only). |
| `--pipelines-repo PATH` | Host checkout used for scoring. Never shown to the agent. |
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
receipt. By default runs go under `/net/scratch/$USER/chaperone-evals/runs` when that
exists, otherwise `harbor-controller/jobs`; `--run-dir` overrides it. Under the run root:

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

Compute is capped too: a trial can reserve at most five `gpu:small` GPUs at a time
across all its Slurm jobs. Held-out scoring jobs are not counted against the agent.

## Troubleshooting

| Message | Cause and fix |
|---------|---------------|
| `No held-out scorers found` | The task folders have no `score/`. Set `CHAPERONE_EVALS_PRIVATE` or pass `--private-tasks-path`. |
| `Held-out audit patterns not found` | The tree has no `audit_hints.json`. A harness bundle cannot be built without it. |
| `Public and private inputs/ differ` | `CHAPERONE_EVALS_PRIVATE` points at a tree whose `inputs/` no longer match the ones in this repository. Sync them. |
| `No scorer for task` | The task name is wrong or has no `score/score.py`. |
| `Scoring site-config preflight failed` | A site tool or data path named in `.env` / `.env.ipd` is missing or not executable on this host. Fix the path, or override it in your shell. |
| `Could not locate the trusted .env and .env.ipd site config` | Set `PIPELINES_SITE_CONFIG_REPO` to a checkout that has both files. |
| `Refusing to overwrite existing scoring folder` | The trial already has a `scoring/` folder. `rescore.py` only handles unscored trials; add `--clear-empty-scoring` if the folder is empty. |

## Tests

```bash
cd src/chaperone-evals
PYTHONPATH=harbor-controller ../../.venv/bin/python -m pytest harbor-controller/tests scoring
```

This needs `pytest` (the harness project's `test` extra). The repository's own `pytest` only collects
`tests/`, so it never runs these.
