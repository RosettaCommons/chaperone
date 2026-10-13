---
name: setup
description: Setup skill for installing and checking pipelines functionality.
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write", "Edit"]
---

# Setup Skill

Guide new users through pipelines environment configuration.

## Trigger

When the user asks to "setup pipelines", "configure environment", "get started", or "install pipelines".

## Steps

### 1. Install & Check Python Environment

First check whether uv is installed (`which uv`). If not, use `AskUserQuestion` to ask:
- **Install uv now** — Run `curl -LsSf https://astral.sh/uv/install.sh | sh` and then `source $HOME/.local/bin/env` (or restart shell)
- **I'll install it myself** — Stop and let the user handle it

Do not proceed until `uv` is available.

Then check if the venv exists and pipelines is importable:
```bash
source .venv/bin/activate
python -c "import pipelines; print('pipelines OK')"
```

If the venv doesn't exist or the import fails, create and install:
```bash
uv venv --python=3.12.3
source .venv/bin/activate
uv pip install -e '.[all]' --override overrides.txt
```

### 2. Detect IPD Environment

Check if the user is at the IPD:
```bash
[ -d /projects/ml/pipelines ] && echo "IPD: yes" || echo "IPD: no"
```

If `/projects/ml/pipelines/` exists, the user is on an IPD machine where containers, compute backends, and block installations are already managed centrally. **Skip directly to step 9 (Jupyter kernel).** Do not run steps 3–8.

### 3. Detect Container Backends

```bash
which apptainer 2>/dev/null && apptainer --version || echo "apptainer: not found"
which docker 2>/dev/null && docker --version || echo "docker: not found"
```

Report which backends are available. If neither is found, warn the user that container-based blocks (RFD3, MPNN, AF3, etc.) won't work. Suggest to install docker if not.

### 4. Configure Container Backend

Check auto-detection result:
```bash
source .venv/bin/activate
python -c "from pipelines import get_settings; print(f'Auto-detected backend: {get_settings().CONTAINER_BACKEND}')"
```

Use `AskUserQuestion` to ask the user which container backend they want to use (showing what was auto-detected as default). If they want to override, write it to the project `.env` file (see step 5 for the write pattern).

### 5. Configure Compute Backend

Detect available compute resources:
```bash
which sinfo 2>/dev/null && echo "SLURM: available" || echo "SLURM: not found"
nvidia-smi 2>/dev/null && echo "Local GPU: available" || echo "Local GPU: not found"
```

Check current setting:
```bash
source .venv/bin/activate
python -c "from pipelines import get_settings; print(f'Current compute backend: {get_settings().COMPUTE_BACKEND}')"
```

Use `AskUserQuestion` with `multiSelect: true` to ask the user which compute resources are available to them. Show what was auto-detected in the descriptions:
- **Auto (recommended)** — Automatically detect and use the best available backend at runtime
- **Local** — Run directly on this machine (detected: {local GPU status})
- **Cluster (detected: {SLURM status})** — Submit jobs to SLURM

After the user selects, write their preference to `.env.local` (gitignored, user-specific overrides). If multiple backends are selected, join them with `/` as a priority list (e.g. `PIPELINES_COMPUTE_BACKEND=local/slurm`). Use `PIPELINES_`-prefixed variable names.

**Important: `.env` file structure:**
- `.env` — generic defaults, checked into git (do NOT write user-specific values here)
- `.env.ipd` — IPD-specific paths/mirrors, checked into git, auto-loaded when at IPD
- `.env.local` — user-specific overrides, gitignored (write overrides here)

To update or add a setting in `.env.local`:
```bash
# Replace existing line or append if not present
if grep -q '^PIPELINES_COMPUTE_BACKEND=' .env.local 2>/dev/null; then
    sed -i 's/^PIPELINES_COMPUTE_BACKEND=.*/PIPELINES_COMPUTE_BACKEND=local/' .env.local
else
    echo 'PIPELINES_COMPUTE_BACKEND=local' >> .env.local
fi
```

Similarly for container backend overrides:
```bash
if grep -q '^PIPELINES_CONTAINER_BACKEND=' .env.local 2>/dev/null; then
    sed -i 's/^PIPELINES_CONTAINER_BACKEND=.*/PIPELINES_CONTAINER_BACKEND=docker/' .env.local
else
    echo 'PIPELINES_CONTAINER_BACKEND=docker' >> .env.local
fi
```

Verify the settings were persisted:
```bash
source .venv/bin/activate
python -c "from pipelines import get_settings; s = get_settings(); print(f'Compute: {s.COMPUTE_BACKEND}, Container: {s.CONTAINER_BACKEND}')"
```

### 5b. Check External Dependencies & Data

Check for external data mirrors, weight caches, and binaries that various blocks depend on. These are **optional** — only needed for specific blocks. At IPD these are all pre-configured via `.env.ipd`.

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
  mkdir -p ~/.pipelines/ccd
  curl -L "https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz" -o ~/.pipelines/ccd/components.cif.gz
  gunzip ~/.pipelines/ccd/components.cif.gz
  ```
  Then write `CCD_MIRROR_PATH=$HOME/.pipelines/ccd` to `.env.local`.
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
- At IPD this is preconfigured in `.env.ipd`; elsewhere, either ask the user for their existing path and write it to `.env.local`, or install the helper extra with `uv pip install -e '.[pyrosetta]' --override overrides.txt` and then run `python -c "import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()"`.

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
pipelines list-blocks
```

Show the user which blocks are available and which containers are installed vs missing.

### 7. Install Containers

Use `AskUserQuestion` to ask the user which blocks they want to install containers for (based on what's not yet installed from step 6).

Run installation:
```bash
source .venv/bin/activate
pipelines install <selected_names>
```

### 8. Optional Extras

Use a single `AskUserQuestion` with `multiSelect: true` to ask which optional extras to install:
- **Install pre-commit hooks (recommended)** — Runs ruff format + ruff check automatically before each commit (uses the `pre-commit` package)
- **Open documentation when finished** — Build and open the pipelines docs site in the browser
- **Register Jupyter kernel** — Makes the pipelines venv available as a Jupyter kernel named "Pipelines"

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
python -m ipykernel install --user --name pipelines --display-name "Pipelines"
```

### 9. Summary
Summarize to the user the available tools:
```bash
source .venv/bin/activate
pipelines list-blocks
```

Then present what's possible, grouped into two categories:

**Proteins you can design**: If an `./examples/` directory exists, list its subdirectories so the user can navigate there for more information; this distribution may ship without one, in which case skip this line.
**Workflows** (general patterns the library supports):
- **Sequence optimization** — take existing designs/outputs and run MPNN + prediction sweeps
- **Parameter sweeps** — sweep any block's parameters or chain multiple blocks together
- **Multi-state validation** — predict apo, holo, and ligand-bound states in parallel branches
- **Progressive filtering** — multi-round design with increasingly strict thresholds

Remind the user that the documentation site is running at **http://127.0.0.1:8000** where they can browse the full block reference with primary/secondary parameter tiers, syntax docs, and guides.
