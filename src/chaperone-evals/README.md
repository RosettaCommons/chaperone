# chaperone-evals

Agent evaluation harness for Chaperone. It launches a coding agent on a design
task, with or without Chaperone available, then scores what the agent submits
against held-out criteria.

Each task folder holds what the agent receives (`prompt.md`, `inputs/`) and what it
never sees (`score/`, `example/`). Keep this repository private while the scorers
are in it: an agent with network access can read anything that is public.

## Layout

```text
harbor-controller/       launcher, isolated environments, sanitizer, audit
scoring/                 submission contract, gates and reporting shared by scorers
evals_task/<task>/       prompt.md and inputs/ (what the agent receives), score/
                         (held-out scorer) and example/ (reference submission)
evals_task/audit_hints.json   fingerprints the sanitizer must never find in a bundle
pyproject.toml           depends on the checkout's `pipelines` package by path and
                         adds the controller's dependencies
```

The folder is self-contained: the repository's own `pyproject.toml` does not know
about it, and nothing here is part of the wheel.

Ten tasks are included: `antibody_cdr`, `biotin_binder_design`, `dna_binder`,
`enzyme_glycosidase`, `enzyme_protease`, `nanoparticle_c3`, `peptide_binder`,
`ppi_pdl1`, `ribozyme_design` and `rna_binder`.

## Held-out files

Only `prompt.md` and `inputs/` reach the agent. `score/`, `example/` and
`audit_hints.json` are read by the controller on the host and are never copied into
the Harbor task, the harness bundle or the baseline workspace. Tests cover each
path.

To keep them somewhere else, point the controller at a separate directory laid
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
  this checkout. The bundle holds only `src/blocks`, `src/pipelines` and
  `src/pipelines_cli`, the agent skills, the block reference and a few root files.
  Git metadata, tests, examples, campaigns, prior outputs and **everything else
  under `src/`, including this directory**, are never copied or even walked.
- `baseline` receives the task inputs in a clean workspace without Chaperone.

After the bundle is built it is validated again, and the audit rejects any
allowlisted file that contains a fingerprint from `audit_hints.json`.

Exclusion keeps held-out material out of the agent's workspace. It does not stop an
agent that has network access from reading a public repository. If this repository
is ever made public, move `score/`, `example/` and `audit_hints.json` to a private
tree first.

## Setup

The controller lives inside the Chaperone checkout it evaluates and shares its
environment. Bootstrap installs this folder's project, which brings in the
checkout's `pipelines` package (editable) and the controller's dependencies, on top
of what is already in the checkout's `.venv`, and links the controller to it.

```bash
cd src/chaperone-evals
./harbor-controller/scripts/bootstrap.sh     # uv pip install -e src/chaperone-evals  (from the repository root)
export OPENAI_API_KEY=...       # or use an existing `codex login`
```

The submission host needs `uv`, Apptainer, `sbatch`, `squeue` and `sacct`.
Set `PIPELINES_REPO` to evaluate a different Chaperone checkout than the one
containing this directory.

## Run an evaluation

```bash
cd src/chaperone-evals/harbor-controller

# harness arm: the agent gets this checkout
./run_eval.py biotin_binder_design \
  --arm harness \
  --pipelines-path /absolute/path/to/chaperone \
  --reasoning-effort low \
  --required-n 10 \
  --run-name biotin-harness-smoke

# baseline arm: no Chaperone
./run_eval.py biotin_binder_design --arm baseline --required-n 10 \
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
PYTHONPATH=harbor-controller ../../.venv/bin/python -m pytest harbor-controller/tests scoring
```

The environment needs `pytest` (this project's `test` extra). The repository's own `pytest`
only collects `tests/`, so it never runs these.
