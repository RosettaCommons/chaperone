---
name: setup
description: Setup skill for installing and checking Chaperone functionality.
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write", "Edit"]
---

# Setup Skill

Guide new users through Chaperone environment configuration.

## Trigger

When the user asks to "setup Chaperone", "configure environment", "get started", or "install Chaperone".

## Steps

### 1. Install & Check Python Environment

First check whether uv is installed (`which uv`). If not, use `AskUserQuestion` to ask:
- **Install uv now** — Run `curl -LsSf https://astral.sh/uv/install.sh | sh` and then `source $HOME/.local/bin/env` (or restart shell)
- **I'll install it myself** — Stop and let the user handle it

Do not proceed until `uv` is available.

Then check if the venv exists and Chaperone is importable:
```bash
source .venv/bin/activate
python -c "import chaperone; print('chaperone OK')"
```

If the venv doesn't exist or the import fails, create and install:
```bash
uv venv --python=3.12.3
source .venv/bin/activate
uv pip install -e '.[all]' --override overrides.txt
```

### 2. Select site configuration

Portable defaults apply when no site file is supplied. Do not infer the site
from the existence of `/projects` or `/net`.

For IPD, accept the user's `.env.ipd` in the checkout root (untracked); it is
loaded automatically when present. Alternatively export `CHAPERONE_SITE=digs`
before starting Python to select packaged IPD defaults. Users can point
`CHAPERONE_ENV_FILE` at a shared or external dotenv file. An existing supplied
site file is authorization to use its settings; do not ask again.

Verify configured containers/weights exist before skipping their installation.
Continue the compute checks: a configured path alone does not prove availability.

### 3. Detect Container Backends

```bash
which apptainer 2>/dev/null && apptainer --version || echo "apptainer: not found"
which docker 2>/dev/null && docker --version || echo "docker: not found"
```

Local container execution needs Docker or Apptainer. Modal runs remotely and needs neither installed locally; install the `cloud` extra and authenticate with `modal setup` for that path. CPU Python blocks need no container runtime.

### 4. Configure Container Backend

Check auto-detection result:
```bash
source .venv/bin/activate
python -c "from chaperone import get_settings; print(f'Auto-detected backend: {get_settings().CHAPERONE_CONTAINER_BACKEND}')"
```

Use `AskUserQuestion` to ask the user which container backend they want to use (showing what was auto-detected as default). If they want to override, write it to the project `.env.local` file (see step 5 for the write pattern).

### 5. Configure Compute Backend

Detect available compute resources:
```bash
which sinfo 2>/dev/null && echo "SLURM: available" || echo "SLURM: not found"
nvidia-smi 2>/dev/null && echo "Local GPU: available" || echo "Local GPU: not found"
```

Check current setting:
```bash
source .venv/bin/activate
python -c "from chaperone import get_settings; print(f'Current compute backend: {get_settings().CHAPERONE_COMPUTE_BACKEND}')"
```

Select one compute backend, using the user's existing preference when supplied. Show what was auto-detected in the descriptions:
- **Auto (recommended)** — Automatically detect and use the best available backend at runtime
- **Local** — Run directly on this machine (detected: {local GPU status})
- **Cluster (detected: {SLURM status})** — Submit jobs to SLURM
- **Modal** — Run compiled blocks on cloud GPUs (`CHAPERONE_COMPUTE_BACKEND=modal`)

After the user selects, write their preference to `.env.local` (gitignored, user-specific overrides). Use one of `auto`, `local`, `slurm`, or `modal`. Slash-separated priority lists apply only to the container backend; they are invalid for the compute backend. Use `CHAPERONE_`-prefixed variable names.

For Slurm, configure `CHAPERONE_DEFAULT_PARTITION` (CPU),
`CHAPERONE_GPU_PARTITION`, and `CHAPERONE_GRES_MAP` in `.env.local`. For example:

```dotenv
CHAPERONE_DEFAULT_PARTITION=batch
CHAPERONE_GPU_PARTITION=accelerated
CHAPERONE_GRES_MAP='{"small":"gpu:a100:1","large":"gpu:h100:1"}'
```

With these unset, blocks omit `--partition` and GPU blocks request `gpu:1`.
IPD uses partitions `cpu`/`gpu` and types `gpu:small:1`/`gpu:large:1`.
Process `CHAPERONE_GRES_MAP` overrides merge by key with dotenv settings.
`CHAPERONE_DEFAULT_ACCOUNT` and `CHAPERONE_DEFAULT_QOS` are optional.
Inspect resolved settings and run the smallest relevant block before a campaign.

**Important: `.env` file structure:**
- `.env` — generic defaults, checked into git (do NOT write user-specific values here)
- `.env.ipd` — optional supplied IPD settings, gitignored, loaded when present
- `CHAPERONE_SITE=digs` — optional packaged IPD preset
- `~/.env.chaperone` / `CHAPERONE_ENV_FILE` — optional user/shared overrides
- `.env.local` — user-specific overrides, gitignored (write overrides here)

To update or add a setting in `.env.local`:
```bash
# Replace existing line or append if not present
if grep -q '^CHAPERONE_COMPUTE_BACKEND=' .env.local 2>/dev/null; then
    sed -i 's/^CHAPERONE_COMPUTE_BACKEND=.*/CHAPERONE_COMPUTE_BACKEND=local/' .env.local
else
    echo 'CHAPERONE_COMPUTE_BACKEND=local' >> .env.local
fi
```

Similarly for container backend overrides:
```bash
if grep -q '^CHAPERONE_CONTAINER_BACKEND=' .env.local 2>/dev/null; then
    sed -i 's/^CHAPERONE_CONTAINER_BACKEND=.*/CHAPERONE_CONTAINER_BACKEND=docker/' .env.local
else
    echo 'CHAPERONE_CONTAINER_BACKEND=docker' >> .env.local
fi
```

Verify the settings were persisted:
```bash
source .venv/bin/activate
python -c "from chaperone import get_settings; s = get_settings(); print(f'Compute: {s.CHAPERONE_COMPUTE_BACKEND}, Container: {s.CHAPERONE_CONTAINER_BACKEND}')"
```

### 5b. Check External Dependencies & Data

Check for external data mirrors, weight caches, and binaries that various blocks depend on. These are **optional** — only needed for specific blocks. At IPD a supplied `.env.ipd` or `CHAPERONE_SITE=digs` supplies these paths. Verify they exist.

Run this check to see what's configured:
```bash
source .venv/bin/activate
python -c "
import os
deps = {
    'CCD_MIRROR_PATH':        ('CCD ligand dictionary',        'Ligand-aware structure parsing (atomworks)'),
    'PDB_MIRROR_PATH':        ('PDB structure mirror',         'Local PDB lookups'),
    'HF_HUB_CACHE':           ('HuggingFace model cache',      'ESM2, BoltzGen weight downloads'),
    'FOUNDRY_CHECKPOINT_DIRS': ('Foundry model checkpoints',   'RFD3, RF3, FoundryMPNN'),
    'BOLTZ_CACHE_DIR':         ('Boltz2 model cache',          'Boltz2, ProteinHunter'),
    'DUNBRACK_LIBRARY_COPY':   ('Dunbrack rotamer library',    'Rotamer-dependent scoring'),
    'HBPLUS_COPY':             ('HBPLUS binary',               'Hydrogen bond metrics (hbond block)'),
    'ROSETTA_BIN':             ('Rosetta binary',              'Ligand DDG calculations'),
    'PYROSETTA_PATH':          ('PyRosetta setup directory',   'PyRosetta-backed blocks (ligand_ddg_soft, rotamer_relax)'),
    'FOLDSEEK_EXEC':           ('Foldseek binary',             'Structure search (foldseek block)'),
}
for var, (name, usage) in deps.items():
    val = os.environ.get(var)
    status = f'{val}' if val else 'NOT SET'
    print(f'{var}: {status}  # {name} — {usage}')
"
```

Present the results as a table showing what's set and what's missing. Then use `AskUserQuestion` to let the user configure the missing ones they care about:

**For `CCD_MIRROR_PATH`** (needed for ligand-aware operations):
- **Download CCD now** — Download from RCSB (~2.5 GB compressed):
  ```bash
  mkdir -p ~/.chaperone/ccd
  curl -L "https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz" -o ~/.chaperone/ccd/components.cif.gz
  gunzip ~/.chaperone/ccd/components.cif.gz
  ```
  Then write `CCD_MIRROR_PATH=$HOME/.chaperone/ccd` to `.env.local`.
- **I have a local copy** — Ask for the path, write to `.env.local`.
- **Skip** — Only needed for ligand-aware operations.

**For `HF_HUB_CACHE`** (HuggingFace weights for ESM2, BoltzGen):
- Default location is `~/.cache/huggingface/hub/`. If the user has a custom cache location, write it to `.env.local`. Otherwise the HuggingFace default works fine — no action needed.

**For `FOUNDRY_CHECKPOINT_DIRS`** (RFD3, RF3, FoundryMPNN weights):
- These are downloaded automatically when blocks are first run. If the user has pre-downloaded weights, ask for the path and write to `.env.local`.

**For `BOLTZ_CACHE_DIR`** (Boltz2 weights):
- Defaults to `~/.boltz`. Only set if the user has a custom location.

**For binaries** (`HBPLUS_COPY`, `ROSETTA_BIN`, `FOLDSEEK_EXEC`):
- These are external tools that must be installed separately. Report which are found/missing and note which blocks require them:
  - `HBPLUS_COPY` — needed by `hbond_metrics` block
  - `ROSETTA_BIN` — needed by `ligand_ddg` block
  - `FOLDSEEK_EXEC` — needed by `foldseek` block. Check if `foldseek` is on PATH: `which foldseek`
- Don't prompt to install these unless the user specifically needs them. Just report status.

**For `PYROSETTA_PATH`** (PyRosetta-backed blocks):
- This should point at the PyRosetta `setup/` directory (for example `/path/to/pyrosetta/setup`).
- At IPD this is configured by the supplied `.env.ipd` or the digs preset; elsewhere, either ask the user for their existing path and write it to `.env.local`, or install the helper extra with `uv pip install -e '.[pyrosetta]' --override overrides.txt` and then run `python -c "import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()"`.

Write any user-provided paths to `.env.local` using the same pattern as step 5:
```bash
if grep -q '^VAR_NAME=' .env.local 2>/dev/null; then
    sed -i '' 's|^VAR_NAME=.*|VAR_NAME=/new/path|' .env.local
else
    echo 'VAR_NAME=/new/path' >> .env.local
fi
```

### 6. Check Block & Container Status

```bash
source .venv/bin/activate
chaperone list-blocks
```

Show the user which blocks are available and which containers are installed vs missing.

### 7. Install core containers (full install)

When the user asks for a full install or evaluation-ready setup, use this preset;
their request already authorizes the selected builds. Reuse verified installed
images. Do not ask again for individual tools or rebuild everything with `--force`.

```bash
source .venv/bin/activate
chaperone install --full --jobs=2 --dry_run
chaperone install --full --jobs=2
```

The preset builds **Foundry** (RFD3, RFD3NA, RF3, FoundryMPNN), **LigandMPNN**
(ProteinMPNN, LigandMPNN, SolubleMPNN), **AlphaFold3**, **ESMFold2**, **Boltz2**,
**Chai-1**, and **ColabFold**. It omits PLMs (E1/ESM2), SimpleFold, legacy diffusion,
and specialized tools. Additional names extend the preset, e.g.
`chaperone install --full tmol --jobs=2`. Named installs retain a serial default.
Unknown names fail before any builds start; aliases are built once. Failures are
reported together, and a rerun skips successful images. Each build retains its
rendered recipe in the container directory.

Run builds on a machine/allocation with Apptainer build privileges or a Docker
daemon, outbound network, and sufficient disk, RAM and CPUs. Start with two builds;
Apex compilation and large CUDA image layers are expensive. `--jobs` bounds builds,
not every compiler thread. Apptainer compression is capped at two threads/1 GiB per
image. On a cluster that forbids execution on login nodes, use a CPU allocation
via the Slurm skill. Do not build containers inside a benchmark agent trial.

**Containers alone are not a full evaluation installation.** Complete the
prerequisite table and runtime-image instructions in `docs/site/evaluations.md`:

- Install the separate harness project with its bootstrap script; authenticate
  the host Codex executable with the intended model provider.
- Build `agent.sif` (also usable as `worker_image`); build `baseline.sif` only when
  running a baseline arm. These are separate from scientific tool containers.
- Stage model weights before the trial. AF3 weights require authorized access;
  some other downloads need accepted terms/authentication. Verify cache mounts
  and driver support (current ESMFold2 uses CUDA 13).
- Configure Slurm partitions/GPU classes, runtime binds, task capabilities and
  writable per-trial caches. Verify nested Apptainer with `/dev/fuse` on a GPU node.
- Keep private eval data and provider/download credentials on the trusted
  controller. Never bind their parent directories into agent or worker containers.
- Install the task's scoring dependencies. Biotin needs AF3 weights/container and
  HBPLUS; the controller stages pinned PLIP/OpenBabel/lxml. Other task preflights
  also require CCD, PyRosetta and X3DNA paths. Optional scientific containers do
  not supply those external assets.
- Run controller `--dry-run`, then one real single-design trial and scoring. A
  container listing or successful import does not prove end-to-end readiness.

### 7b. Warm RFD3's compilation cache on the target GPU

For a full install including GPU warmup, combine the phases:

```bash
chaperone install --full --jobs=2 --warmup_rfd3 \
  --warmup_cache_dir="$HOME/.cache/chaperone/rfd3" \
  --warmup_run_dir=/absolute/shared/path/rfd3-warmup
```

This builds containers first, then submits warmup through the engine. Add
`--warmup_local` only when the installer is already running in a GPU allocation.
A failed build prevents warmup; retrying skips completed images.


Use the pinned public Foundry's supported `compile_model` interface. Do not select
a private `foundry_fast.sif`, change the model weights, or copy PR #351's unsupported
kernel/shape/worker flags. Its persistent-cache idea is applicable, but its private
runtime is not the current public recipe.

```bash
mkdir -p "$HOME/.cache/chaperone/rfd3"
chaperone warmup-rfd3 \
  --cache-dir "$HOME/.cache/chaperone/rfd3" \
  --rundir /absolute/shared/path/rfd3-warmup
```

The normal execution engine submits one GPU job using site settings. Add `--local`
only within an existing GPU allocation. This runs two short batches through the
compiled model; outputs are installation smoke artifacts, not design-quality
samples. `--inputs /path/to/public-inputs.json` and `--batch-size N` can warm a
representative workload. Keep benchmark-private inputs/answers out of shared caches.

Persist `RFD3_COMPILE_CACHE_DIR=/absolute/cache/path` in `.env.local` and use
`RFD3(compile_model=True)` in subsequent pipelines. The directory must already
exist, be writable, and allow execution of compiler probes (`TMPDIR` is routed
there because container `/tmp` can be non-executable); it is bound explicitly for local Apptainer/Docker execution.
The cache does not contain model weights. GPU architecture, software version,
shape, and diffusion batch size affect reuse; new combinations may compile again.
Cold compile can take several minutes and is not always worthwhile for a short
job. Leaving `compile_model=None` retains the installed runtime's defaults.

For evaluation, use a writable trial-local cache (or a controller-prepared copy),
never a shared mutable cache across contestants. The setup command does not relax
the evaluation mount allowlist or automatically expose a user's cache.

### 8. Optional Extras

Use a single `AskUserQuestion` with `multiSelect: true` to ask which optional extras to install:
- **Install pre-commit hooks (recommended)** — Runs ruff format + ruff check automatically before each commit (uses the `pre-commit` package)
- **Open documentation when finished** — Build and open the Chaperone docs site in the browser
- **Register Jupyter kernel** — Makes the Chaperone venv available as a Jupyter kernel named "Chaperone"

Then execute the selected options:

**Pre-commit hooks:**
```bash
source .venv/bin/activate
pre-commit install
```

**Open documentation:**
```bash
source .venv/bin/activate
uv pip install -e '.[docs]' --override overrides.txt
python docs/site/gen_block_docs.py
mkdocs build && mkdocs serve &
sleep 1 && open http://127.0.0.1:8000
```

**Jupyter kernel:**
```bash
source .venv/bin/activate
python -m ipykernel install --user --name chaperone --display-name "Chaperone"
```

### 9. Summary
Summarize to the user the available tools:
```bash
source .venv/bin/activate
chaperone list-blocks
```

Then present what's possible, grouped into two categories:

**Proteins you can design**: If an `./examples/` directory exists, list its subdirectories so the user can navigate there for more information; this distribution may ship without one, in which case skip this line.
**Workflows** (general patterns the library supports):
- **Sequence optimization** — take existing designs/outputs and run MPNN + prediction sweeps
- **Parameter sweeps** — sweep any block's parameters or chain multiple blocks together
- **Multi-state validation** — predict apo, holo, and ligand-bound states in parallel branches
- **Progressive filtering** — multi-round design with increasingly strict thresholds

If you actually started the documentation server, report its address, normally **http://127.0.0.1:8000** where they can browse the full block reference with primary/secondary parameter tiers, syntax docs, and guides.
