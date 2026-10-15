# chaperone-evals

Agent evaluation harness for Chaperone. It launches a coding agent on a design
task, with or without Chaperone available, then scores what the agent submits
against held-out criteria.

Each shipped task folder contains only `prompt.md` and `inputs/`. Held-out
scorers, reference answers and audit fingerprints are stored in the private
[chaperone-evals repository](https://github.com/Ubiquinone-dot/chaperone-evals)
and a private Harbor Hub snapshot, accessible only to the trusted controller.

## Layout

```text
harbor-controller/       launcher, isolated environments, sanitizer, audit
scoring/                 submission contract, gates and reporting shared by scorers
evals_task/<task>/       prompt.md and inputs/ (what the agent receives)
pyproject.toml           controller dependencies; bootstrap selects the Chaperone checkout
```

The folder is self-contained: the repository's own `pyproject.toml` does not know
about it, and nothing here is part of the wheel.

Ten tasks are included: `antibody_cdr`, `biotin_binder_design`, `dna_binder`,
`enzyme_glycosidase`, `enzyme_protease`, `nanoparticle_c3`, `peptide_binder`,
`ppi_pdl1`, `ribozyme_design` and `rna_binder`.

## Held-out files

Only `prompt.md` and `inputs/` reach the agent. `score/`, `example/` and
`audit_hints.json` are read by the controller on the host and are never copied into
the rendered agent task, the harness bundle or the baseline workspace. Tests cover each
path.

Point the controller at a separate private directory laid
out like `evals_task/` with `CHAPERONE_EVALS_PRIVATE=<dir>` or
`run_eval.py --private-tasks-path`. Launch checks that its `inputs/` match the
ones in this repository. `audit_hints.json` is a JSON list of
`{"label": ..., "pattern": <regex>, "ignore_case": true}` objects.

Launching the harness arm fails if `audit_hints.json` is missing, and scoring
fails if a scorer is missing. Launch with `--no-score` to run without the scorer;
the audit patterns are still required for the harness arm.

## Isolation model

The two arms receive different environments.

- `harness` receives the task inputs plus a freshly built, history-free bundle of
  this checkout. The bundle holds only `src/blocks`, `src/chaperone` and
  `src/chaperone_cli`, the agent skills, the block reference and a few root files.
  Git metadata, tests, examples, campaigns, prior outputs and **everything else
  under `src/`, including this directory**, are never copied or even walked.
- `baseline` receives the task inputs in a clean workspace without Chaperone.

After the bundle is built it is validated again, and the audit rejects any
allowlisted file that contains a fingerprint from `audit_hints.json`.

Exclusion keeps held-out material out of the agent's workspace. It does not stop an
agent that has network access from reading public worked examples. Treat those
examples as open-book assistance or use a network-restricted benchmark. Before
making this repository public, remove historical answer copies from all published
Git refs as well; the deletion commit alone does not sanitize history.

## Private evaluation repository

Clone [Ubiquinone-dot/chaperone-evals](https://github.com/Ubiquinone-dot/chaperone-evals)
on the trusted controller, outside the Chaperone checkout and every agent/worker
mount:

```bash
umask 077
git clone https://github.com/Ubiquinone-dot/chaperone-evals.git /private/chaperone-evals
cd /private/chaperone-evals
sha256sum --check SHA256SUMS
export CHAPERONE_EVALS_PRIVATE=/private/chaperone-evals/evals_task
```

Return to this evaluation project before running the controller commands below.
The private repository contains all ten task trees and their audit fingerprints;
the controller and shared scoring implementation remain here. Keep that repository
private. Git changes do not automatically update the published Harbor snapshot.

## Private Harbor dataset (alternative download)

The private package is `ubiquinone-dot/chaperone-evals`. Its ten task packages
contain prompts/inputs; held-out controller data is a dataset-level archive,
not an agent environment asset. Download only on the trusted controller:

```bash
harbor dataset download ubiquinone-dot/chaperone-evals@release-split-20261014 \
  --output-dir /private/download
# Use the download directory returned by Harbor:
python harbor-controller/scripts/private_dataset.py restore \
  --dataset /private/download/chaperone-evals --output /private/evals_task
export CHAPERONE_EVALS_PRIVATE=/private/evals_task
```

Use the Chaperone `run_eval.py` entry point afterward. The package requires the
configured cluster runtimes and external scorer; its standalone Harbor verifier
fails explicitly rather than issuing an unmeasured reward. Keep Hub credentials,
downloads and restored data outside all agent and scientific-asset mounts.
The exporter/restorer validates the file inventory and SHA-256 checksums.

Public CI checks structural isolation using synthetic data. Run the corpus audit
with `CHAPERONE_EVALS_PRIVATE` configured before launching real evaluations.

### Scheduler boundary

Only the host broker has Slurm authentication. Agent status queries (`sacct`,
`squeue`) are scoped to registered trial jobs. Agent and worker mounts reject
scheduler directories, authentication sockets, libraries and native clients,
including aliases and enclosing directories. Workers receive allocation metadata
from an explicit environment allowlist; they cannot submit nested Slurm work or
invoke Submitit's native `scontrol requeue`. The outer driver owns retries.
Site-provided images and scientific assets remain trusted inputs: do not bake
credentials into images or expose broad shared code/cache trees containing secrets.

## Setup

Bootstrap creates a separate `src/chaperone-evals/.venv` for the controller
and installs the containing Chaperone checkout into its own `.venv` for host-side
scoring. Set `CHAPERONE_REPO` before bootstrap to select another scoring checkout;
keep that variable exported when launching. Both environments use a managed
Python interpreter on shared storage. Existing environments are not pruned.

```bash
cd src/chaperone-evals
./harbor-controller/scripts/bootstrap.sh     # install the controller and selected Chaperone checkout
export OPENAI_API_KEY=...       # or use an existing `codex login`
```

The submission host needs `uv`, Apptainer, `sbatch`, `squeue` and `sacct`.
Set `CHAPERONE_REPO` to evaluate a different Chaperone checkout than the one
containing this directory.

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

## Run an evaluation

```bash
cd src/chaperone-evals/harbor-controller

# harness arm: the agent gets this checkout
./run_eval.py biotin_binder_design \
  --arm harness --site-config /absolute/path/to/site.json \
  --chaperone-path /absolute/path/to/chaperone \
  --reasoning-effort low \
  --required-n 10 \
  --run-name biotin-harness-smoke

# baseline arm: no Chaperone
./run_eval.py biotin_binder_design --arm baseline --site-config /absolute/path/to/site.json --required-n 10 \
  --run-name biotin-baseline-smoke
```

`./run_eval.py --help` lists every option, `--dry-run` checks the configuration
only, and `--no-score` skips the held-out scorer. See
[harbor-controller/README.md](harbor-controller/README.md) for the controller's
internals and failure recovery.

A non-null `harness_fault` in a trial's `scoring/score.json` means the scientific
oracle did not produce a valid assembly and the run is void; never aggregate it as
an ordinary zero-pass trial.

## Tests

```bash
cd src/chaperone-evals
PYTHONPATH=harbor-controller .venv/bin/python -m pytest harbor-controller/tests scoring
```

The environment needs `pytest` (this project's `test` extra). The repository's own `pytest`
only collects `tests/`, so it never runs these.
