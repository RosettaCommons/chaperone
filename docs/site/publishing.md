# Package release and portable installation

The distribution name is **chaperone-pipelines**. Python imports remain
`chaperone` and `blocks`; the executable remains `chaperone`. The unrelated
`chaperone` distribution on PyPI must not be installed into this environment.
Production PyPI publication is not enabled by this repository's workflows.

## TestPyPI

The `TestPyPI` workflow runs the complete clean-install workflow before uploading
its wheel and source distribution. Trigger it manually after the workflow is on
main, or push a `testpypi-v<version>` tag on the reviewed commit. Increment the
version in `pyproject.toml` before uploading changed artifacts: TestPyPI does not
allow replacing an existing file. A rehearsal upload does not release to PyPI.

For token-free uploads, configure a **pending Trusted Publisher** on TestPyPI:

- Project: `chaperone-pipelines`
- GitHub owner: `RosettaCommons`
- Repository: `chaperone`
- Workflow filename: `testpypi.yml`
- Environment: `testpypi`

Linking a TestPyPI account to GitHub is separate from configuring this publisher.
See [TestPyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/).
Use GitHub's `testpypi` environment to restrict who can approve uploads.

A maintainer may instead upload validated local artifacts using a TestPyPI token
supplied to Twine through its environment or a credential manager. Never commit
credentials. Set the repository explicitly to `https://test.pypi.org/legacy/`.

Install the rehearsal package without resolving scientific dependencies from
TestPyPI or mixing package indexes:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Optional: keep the controller's Torch CPU-only on Linux.
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
mkdir -p /tmp/chaperone-download
python -m pip download --no-deps --only-binary=:all: \
  --index-url https://test.pypi.org/simple/ \
  --dest /tmp/chaperone-download 'chaperone-pipelines==1.5.0'
python -m pip install --index-url https://pypi.org/simple/ \
  /tmp/chaperone-download/chaperone_pipelines-1.5.0-py3-none-any.whl
python -m pip check
chaperone --help
chaperone list-blocks
```

Use a fresh download directory for another version. `cloud`, `rna`, `ligand`,
and development extras remain available; e.g. install the downloaded wheel
with `[cloud]` to add Modal. Examples and model weights are not part of the wheel.

## Clean-install release gate

`.github/workflows/ci.yml` builds and checks wheel and sdist metadata, installs
each into a fresh Python 3.12 environment using public dependencies, and runs
CLI discovery, installed runtime-asset checks, a CPU pipeline with disk cache
replay, and focused portability regressions. Smoke execution runs outside the
checkout without `PYTHONPATH`; regression tests disable pytest's source-path
injection. CI also executes a CPU Docker command without NVIDIA support.

The initial CI target is Linux / Python 3.12. This is not a claim that every
registered scientific tool has been validated on every platform. GPU smoke
evidence for CoreWeave and Modal is recorded in [release validation](release-validation.md).
No cloud jobs or paid GPU jobs are launched by pull-request CI.

## Choose an execution path

- **CPU Python blocks:** use `CHAPERONE_COMPUTE_BACKEND=local`; no container
  runtime, model weights, or GPU are required.
- **Local Docker:** install Docker; install NVIDIA Container Toolkit only for
  GPU blocks. Generated CPU commands request no GPUs; generated GPU commands
  restrict Docker to the worker's `CUDA_VISIBLE_DEVICES` assignment.
- **Slurm / Apptainer:** use `CHAPERONE_COMPUTE_BACKEND=slurm` and configure
  the site's partition, account, and GPU resource mapping. Apptainer GPU commands
  preserve the scheduler's mask through `--cleanenv`.
- **Modal:** install the `cloud` extra, authenticate with `modal setup`, and use
  `CHAPERONE_COMPUTE_BACKEND=modal` or `cloud=True` on compiled blocks. Local
  Docker/Apptainer is unnecessary. Model-specific weight/access requirements
  still apply.

Write user settings to `.env.local` or an explicit `CHAPERONE_ENV_FILE`.
Compute backend values are single choices, not slash-separated priority lists.
Direct `Image.build_command()` callers request GPU access with `gpu=True`;
engine-managed blocks derive that choice from their declared resources.
Explicit custom command templates remain the caller's responsibility.

When upgrading container recipes, rebuild with `chaperone install <block> --force`:
existing local image names are not yet invalidated by recipe fingerprints.
AF3 weights, PyRosetta, and some external tools have separate access or licensing
requirements; a registry entry alone does not indicate they are installed.
The validated Foundry image remains pinned to AtomWorks 2, independently of the
controller's AtomWorks 3 dependency.

## Later production release

After reviewing TestPyPI installation and the PR, configure a distinct production
PyPI publisher and add an explicitly approved production publication workflow.
The current workflow's only upload endpoint is TestPyPI.
