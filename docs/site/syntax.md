# Chaperone Syntax Reference

This document provides detailed syntax and usage patterns for the Chaperone library.

## ExecutionEngine

The `ExecutionEngine` is the main entry point for running pipelines.

```python
from chaperone import ExecutionEngine

engine = ExecutionEngine(rundir="output/")
```

### Constructor

```python
ExecutionEngine(
    rundir="my_run",              # Required (str | Path | None)
    reinit=True,                  # True = fresh manifest, False = load existing
    run_args_default=RuntimeArgs(mode="local"),  # Default RuntimeArgs for all run() calls
    job_args_default=JobArgs(...),               # Default JobArgs for all run() calls
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rundir` | str \| Path \| None | `None` | Output directory. Falls back to `CHAPERONE_RUNDIR` env var / `.env` file; relative paths resolve under `CHAPERONE_ROOT` (default `/net/scratch/{user}/chaperone`) |
| `reinit` | bool | `True` | `True` = fresh manifest (disk-cached parquets still reused); `False` = load existing manifest to resume/extend a run |
| `run_args_default` | RuntimeArgs \| dict \| None | `None` | Fallback `RuntimeArgs` when `run()` is called without `run_args` |
| `job_args_default` | JobArgs \| dict \| None | `None` | Fallback `JobArgs` when `run()` is called without `job_args` |

### Running Blocks

```python
from chaperone import ExecutionEngine, RuntimeArgs, JobArgs

engine = ExecutionEngine(rundir="output/")

# Basic usage — kwargs are forwarded to the block constructor
engine.run("rfd3", inputs="input.json", n_batches=4)

# With execution configuration
engine.run(
    "rfd3",
    inputs="input.json",
    n_batches=4,
    job_args=JobArgs(gpu_class="small", mem_gb=16),
    run_args=RuntimeArgs(mode="local"),
)

# Chain multiple blocks explicitly
h = engine.run("rfd3", inputs="input.json", n_batches=4)
h = engine.run("mpnn", data=h)
engine.run("af3", data=h)

# Get final results
handler = engine.compile()
```

### `engine.run()` Signature

```python
engine.run(
    block,                         # str | BlockFunction | Type[BlockFunction]
    run_args=None,                 # RuntimeArgs | dict | None
    data=None,                     # Any (DataFrame, PipelineDataHandler, etc.)
    job_args=None,                 # JobArgs | dict | None
    **kwargs,                      # Forwarded to block constructor (when block is a str or type)
) -> PipelineDataHandler
```

**Key behavior:**
- When `block` is a string (e.g. `"rfd3"`), `**kwargs` are passed to the block constructor
- When `block` is an already-instantiated `BlockFunction`, `**kwargs` must be empty
- `data` is required for non-source blocks; accepts packed/unpacked DataFrames
- Returns a `PipelineDataHandler` (not the engine)

### Compute Backend & Mode Resolution

The `CHAPERONE_COMPUTE_BACKEND` setting determines the default execution mode when `RuntimeArgs.mode` is not explicitly provided:

| Backend | Default Mode | Description |
|---------|-------------|-------------|
| `auto` (default) | auto-detected | Local GPU → `"local"`, SLURM available → `"cluster_compiled"`, fallback → `"local"` |
| `slurm` | `"cluster_compiled"` | Always submit to HPC cluster |
| `local` | `"local"` | Always run locally |

An explicit `mode` in `RuntimeArgs` always takes precedence over the backend setting.

### GPU Auto-Detection

When a block requests a GPU and local CUDA GPUs are detected, the engine automatically routes to local execution instead of SLURM (controlled by `use_local_gpus=True`, the default). To force SLURM submission even when local GPUs are available (e.g., when you have a GPU via `salloc` but want to submit batch jobs), set `use_local_gpus=False`:

```python
engine.run("rfd3", run_args=RuntimeArgs(use_local_gpus=False))
```

Or via CLI: `--nouse_local_gpus`

### Parallel Submission: `engine.submit()` + `gather()`

For running independent blocks concurrently (e.g. multiple MPNN variants), use `submit()` and `gather()`:

```python
from blocks import Blocks

MPNN = Blocks.get("mpnn")

protein_mpnn = MPNN(name="protein_mpnn", model_type="protein_mpnn")
ligand_mpnn = MPNN(name="ligand_mpnn", model_type="ligand_mpnn")
soluble_mpnn = MPNN(name="soluble_mpnn", model_type="soluble_mpnn")

# Submit all three in parallel
f1 = engine.submit(protein_mpnn, data=h)
f2 = engine.submit(ligand_mpnn, data=h)
f3 = engine.submit(soluble_mpnn, data=h)

# Gather results (concatenates handlers)
h = engine.gather(f1, f2, f3)
```

Each submitted block must have a unique `name`. `submit()` has the same signature as `run()` but returns a future; `gather()` collects and concatenates results.

### Cycling: `engine.cycle()`

For iterative refinement loops (e.g. MPNN → oracle → repeat):

```python
N_CYCLES = 5

h = engine.run("rfd3", inputs=str(config), batch_size=32, n_batches=10)

for cycle_idx, cycle_engine in enumerate(engine.cycle("opt", N_CYCLES)):
    cycle_engine.run("mpnn", data=h, model_type="ligand_mpnn", num_seq_per_pdb=1)
    h = cycle_engine.run("af3", num_diffusion_samples=5)

    df = h.unpack()
    print(f"Cycle {cycle_idx}: mean plddt={df['af3.plddt'].mean():.2f}")
```

Common pattern: `rfd3 → cycle(mpnn → af3)`. Each cycle creates a separate sub-engine with its own manifest.

## JobArgs

SLURM job configuration for submitting to the cluster.

```python
from chaperone import JobArgs

job_args = JobArgs(
    gpu_class="large",    # GPU size; the site maps it to a gres (see below)
    mem_gb=16,            # Memory per job in GB (default: 8)
    time_limit="04:00:00", # Time limit HH:MM:SS (default: "5:00:00")
    account="myaccount",  # --account (default: CHAPERONE_DEFAULT_ACCOUNT)
    cpus_per_task=4,      # CPU cores allocated per task
    job_name="my_job",    # Custom SLURM job name
    rows_per_job=10,      # Rows per job (overrides block default)
    exclude_nodes="node1,node2",  # Nodes to exclude
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `partition` | str \| None | `None` | SLURM partition name, passed to `sbatch` as-is, or `"local"` for local execution. `None` uses `CHAPERONE_DEFAULT_PARTITION`, else the cluster default |
| `mem_gb` | int | `8` | Memory per job in GB |
| `time_limit` | str | `"5:00:00"` | Time limit in HH:MM:SS format |
| `gpu_class` | str \| None | `None` | Portable GPU size (`"small"`, `"large"`). The site maps it to a gres through `CHAPERONE_GRES_MAP`, else `gpu:1`. Ignored when `gres` is set |
| `gres` | str \| None | `None` | SLURM gres, passed to `sbatch` unchecked (e.g., `"gpu:1"`). Valid GPU types depend on the cluster |
| `account` | str \| None | `None` | SLURM `--account`. `None` uses `CHAPERONE_DEFAULT_ACCOUNT` |
| `qos` | str \| None | `None` | SLURM `--qos`. `None` uses `CHAPERONE_DEFAULT_QOS` |
| `extra_sbatch` | list[str] \| None | `None` | Extra long `sbatch` options, e.g. `["--reservation=mine"]`. `None` uses `CHAPERONE_DEFAULT_EXTRA_SBATCH`; a list replaces it |
| `cpus_per_task` | int | `1` | CPU cores allocated per task/job |
| `job_name` | str \| None | `None` | Custom SLURM job name |
| `rows_per_job` | int \| None | `None` | Rows per job (overrides block default) |
| `exclude_nodes` | str \| None | `None` | Comma-separated nodes to exclude |

### Slurm Site Configuration

A block does not name cluster resources. Its `_resources` says `partition="gpu"` or `"cpu"` (a resource *class*, not a partition name) and, for GPU blocks, `gpu_class="small"` or `"large"`. The site settings turn those into a real partition and `--gres`. Set them in `.env.local`, or `.env.ipd` at IPD:

| Setting | Meaning |
|---------|---------|
| `CHAPERONE_DEFAULT_PARTITION` | Partition for CPU blocks and for any job that names none |
| `CHAPERONE_GPU_PARTITION` | Partition for GPU blocks (falls back to `CHAPERONE_DEFAULT_PARTITION`) |
| `CHAPERONE_GRES_MAP` | JSON map from size class to `--gres`, e.g. `'{"small": "gpu:a4000:1", "large": "gpu:a6000:1"}'`. The key `default` covers other classes |
| `CHAPERONE_DEFAULT_ACCOUNT` / `CHAPERONE_DEFAULT_QOS` | Default `--account` / `--qos` |
| `CHAPERONE_DEFAULT_EXTRA_SBATCH` | Extra long `sbatch` options, shell-split, e.g. `'--reservation=mine --mail-type=END'` |

With none set, a GPU block submits with `--gres=gpu:1` and no `--partition`, and a CPU block with neither. That is valid when the cluster's default partition has the GPUs; if it does not, set `CHAPERONE_GPU_PARTITION`. List a cluster's partitions and GPU types with `sinfo -o "%P %G"`.

Chaperone does not check GPU types. Whatever `gres` reaches `sbatch` unchanged, so the cluster decides what is valid: IPD registers size classes (`gpu:small:1`, `gpu:large:1`) and rejects model names such as `gpu:a4000:1`, while most other clusters register model names or accept only `gpu:1`. A map given in the environment is merged key by key with one in a `.env*` file.

An optional, user-supplied `.env.ipd` in the checkout root is loaded automatically; the public repository does not track it. Without that file, installation uses portable defaults. Alternatively, select the packaged IPD preset explicitly with `CHAPERONE_SITE=digs`. The IPD preset sets `CHAPERONE_DEFAULT_PARTITION=cpu`, `CHAPERONE_GPU_PARTITION=gpu` and the map `small` -> `gpu:small:1`, `large` -> `gpu:large:1`, so blocks request exactly what they always did.

Configuration files are read from lower to higher priority: repository `.env`, an explicitly selected `CHAPERONE_SITE` preset, optional repository `.env.ipd`, `~/.env.chaperone`, a file named by `CHAPERONE_ENV_FILE`, then repository `.env.local`. Process environment variables take precedence. `CHAPERONE_GRES_MAP` merges keys across these sources, so overriding `small` does not erase the site's `large` mapping. Keep private credentials out of committed configuration.

### JobArgs Resolution

Final `JobArgs` are resolved by merging layers. Highest priority first: CLI flags (`--partition`, `--gres`, `--account`, `--qos`, `--extra_sbatch`), then the user `JobArgs`, then `block._resources`, then the site settings above. Class names in `block._resources` (`"cpu"`, `"gpu"`, `gpu_class`) are mapped by the site; values you pass in `job_args` or on the CLI are used verbatim. Between the user and block layers, only user fields that differ from `JobArgs()` defaults override. The effective user `JobArgs` is: `job_args` passed to `run()`, falling back to `job_args_default` from the engine constructor if `job_args` is `None`. Note that `job_args_default` and `job_args` are **not** merged with each other — `job_args` fully replaces `job_args_default` when provided.

## RuntimeArgs

Runtime execution configuration.

```python
from chaperone import RuntimeArgs

run_args = RuntimeArgs(
    mode="local",            # Execution mode
    recompile=False,         # Force recompilation of outputs
    allow_errors=False,      # Continue on chunk errors
    force_exec=False,        # Force full re-execution (prepare, execute, compile), ignoring all caches
    skip_cache=False,        # Skip step-level cache
    skip_chunk_cache=False,  # Skip chunk-level cache
    use_local_gpus=True,     # Auto-detect and use local GPUs instead of SLURM
    max_retries=2,           # Max retries for failed chunks (SLURM only)
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | str | `"cluster_compiled"` | Execution mode (see below). When not set explicitly, resolved from `CHAPERONE_COMPUTE_BACKEND` |
| `recompile` | bool | `False` | Force recompilation even if outputs exist (forces `mode="local"`) |
| `allow_errors` | bool | `False` | Continue execution if some chunks fail |
| `force_exec` | bool | `False` | Force full re-execution (prepare, execute, compile), bypassing all step and chunk caches. Unlike `recompile`, preserves the current execution mode |
| `skip_cache` | bool | `False` | Skip step-level caching |
| `skip_chunk_cache` | bool | `False` | Skip chunk-level caching |
| `use_local_gpus` | bool | `True` | Auto-detect and use local GPUs instead of submitting to SLURM. Set `False` to force SLURM submission |
| `max_retries` | int | `2` | Max retries for failed SLURM chunks (0 for local mode) |

**Note:** Setting `recompile=True` automatically forces `mode="local"`. Use `force_exec=True` instead if you want to bypass all caches while keeping the current execution mode (e.g. SLURM).

### Execution Modes

| Mode | Prepare | Execute | Compile | Use Case |
|------|---------|---------|---------|----------|
| `local` | local | local | local | Development, debugging |
| `cluster_executed` | local | SLURM | local | GPU jobs with local compile |
| `cluster_compiled` | local | SLURM | SLURM | Full SLURM execution (default) |
| `cluster_only` | SLURM | SLURM | SLURM | Everything on cluster |

## Block Configuration

### RFD3 (RFdiffusion3)

```python
engine.run(
    "rfd3",
    inputs={
        "design-1": {
            "contig": "50-80",           # Diffuse 50-80 residue protein
            "input": "path/to/pdb",      # Optional: fixed scaffold
            "ligand": "HAX,OAA",         # Optional: ligand res names
        }
    },
    n_batches=4,      # Number of diffusion batches
    batch_size=8,     # Designs per batch
)
```

### Filter — `expr` boolean language

The `filter` block prunes rows by metric thresholds. For anything beyond a flat
AND, prefer the `expr` string DSL over the older `filters` / `filter_groups`
dicts — it lets you write `and` / `or` / `not` with parentheses directly instead
of hand-distributing an OR-of-ANDs into named groups.

```python
engine.run(
    "filter",
    name="pocket_filter",
    expr=f"""
        {c}.min_pae_interaction <= 10
        and {c}_hbond.num_hbonds >= 3
        and {c}_contact_ms >= 280
        and (   ({c}_hbond.nicotinamide >= 2 and {c}_aromatic.num_aromatic_neighbors >= 1)
             or ({c}_hbond.nicotinamide >= 1 and {c}_aromatic.num_aromatic_neighbors >= 2) )
        and (   {c}_rasa.ligand_atom_rasa.C21 >= 0.1
             or {c}_rasa.ligand_atom_rasa.C22 >= 0.1
             or {c}_rasa.ligand_atom_rasa.O14 >= 0.1 )
    """,
)
```

Grammar (precedence low → high, standard `or` < `and` < `not`):

| Element | Notes |
|---------|-------|
| **operators** | `>=` `<=` `>` `<` `==` `!=` (all metrics coerced to float) |
| **booleans** | `and` `or` `not`, grouped with `( … )` |
| **metric key** | bare `block.inner_key` — splits on the **first dot only**, exactly like the `filters` dict keys. Namespaced keys (`opt/cycle_0/af3.min_pae_interaction`) and flat inner keys with dots (`rasa.ligand_atom_rasa.C21`) both work. Use `` `back-tick quotes` `` for keys with other characters (spaces, leading digits). |
| **numbers** | int/float, negatives allowed (`ddg <= -1.5`) |

Semantics and interop:

- Compiles to a single Polars mask — no per-row Python, same speed as `filters`.
- **Null / missing metric** → Polars three-valued logic: `True or null` still keeps the row (so an OR-pool survives a missing branch), but a row whose overall mask is null is dropped. A key that resolves to all-null (typo, or nested >1 level deep) logs a warning and drops every row.
- `expr` may be combined with `filters` / `filter_groups` (AND-ed together). **Quantile** filtering has no DSL form — keep it on a `filters` `MetricFilter(quantile=…)`.

Programmatic access: `from blocks.util_blocks.filter import compile_filter_expression` returns the `pl.Expr` for a string, useful for testing or reuse.

## PipelineDataHandler

The universal data container for all pipeline operations. Every block takes a handler as input and returns a handler as output.

### Schema

Each row has two columns:

| Column | Type | Description |
|--------|------|-------------|
| `structures` | `List[Struct]` | Ordered lineage of structures (one entry per step) |
| `metrics` | `Utf8` (JSON) | Accumulated metrics from all steps |

Each structure entry contains `name` (step name), `path` (structure file), and `metadata` (residue annotations like motif, interface, etc.).

### Creating Handlers

| Method | Use Case |
|--------|----------|
| `PipelineDataHandler.from_dir(dir, name, recursive=False)` | Load all structures from a directory |
| `PipelineDataHandler.from_paths(paths, name)` | Load specific structure files |
| `PipelineDataHandler.from_file(path, allow_sequences=False)` | Load from parquet or CSV |
| `PipelineDataHandler.from_sequences(data)` | Load data with sequence columns (auto-detects `sequence`, `seq`, `*_sequence`) |
| `PipelineDataHandler.load(source, allow_sequences=False)` | Smart loader: file, dir, handler, or DataFrame |
| `PipelineDataHandler(data)` | Constructor: accepts `list[dict]`, DataFrame, Path, or handler |

```python
from chaperone import PipelineDataHandler

# From a directory of structures
handler = PipelineDataHandler.from_dir("designs/", name="input")

# From explicit file paths
handler = PipelineDataHandler.from_paths(["design_1.pdb", "design_2.cif"], name="input")

# From a previous step's parquet
handler = PipelineDataHandler.from_file("output/.pipeline/rfd3.parquet")

# From a CSV with sequences (auto-creates CIF files)
handler = PipelineDataHandler.from_sequences("sequences.csv")

# Build rows programmatically
from chaperone.data.schema import make_row
handler = PipelineDataHandler([
    make_row(name="input", path="design_1.pdb", metadata={"motif": ["A10"]}),
    make_row(name="input", path="design_2.pdb"),
])
```

### Inspecting and Transforming

```python
len(handler)                        # Number of designs
handler[0]                          # (structures, metrics) tuple
handler[0:5]                        # New handler with first 5 rows
handler_a + handler_b               # Vertical concatenation

df = handler.unpack()               # Flatten to pandas DataFrame
df = handler.unpack(to_pandas=False) # Flatten to polars
handler.write_parquet("out.parquet") # Persist to parquet
handler.save_csv("out.csv")         # Save as unpacked CSV
handler.subset_by_name("rfd3")      # Keep only structures matching name

# Iterate over loaded structures
for loaded in handler.get_data(which="last", load=True):
    print(loaded.path, loaded.atom_array, loaded.metadata)
```

### Feeding Data to Blocks

The `data` parameter on `engine.run()` accepts anything the constructor accepts — handlers, paths, DataFrames, list of dicts:

```python
engine.run("mpnn", data=handler)
engine.run("mpnn", data="previous_step.parquet")
engine.run("mpnn", data=[make_row(name="input", path="design.pdb")])
```

Source blocks (e.g. `rfd3`) don't need `data=` — they create their own input from their config.

### Explicit Block Config and Data Passing

```python
from blocks import Blocks

MPNNBlock = Blocks.get('mpnn')  # or Blocks['mpnn']
mpnn_round2 = MPNNBlock(name='mpnn_round2', num_seq_per_pdb=16)

df = handler.unpack(all=True)  # unpacks all columns from nested structure
df_filtered = df.head(10)
engine.run(
    mpnn_round2,  # explicit block config
    data=df_filtered  # explicit data — auto-repacked into pipeline schema
)
```

## Full Example: RFD3 → MPNN → AF3 → RMSD

A complete pipeline script using explicit block creation. Adapted from the boltzgen benchmark.

```python
"""
RFD3 Binder Design Pipeline

Generates backbone designs with RFD3, sequences with LigandMPNN,
refolds with AF3, and evaluates with RMSD.

Usage:
    # Local (single design, local GPU):
    python run_pipeline.py

    # Cluster (100 designs):
    python run_pipeline.py --scale
"""
import sys
from pathlib import Path
from chaperone import ExecutionEngine, RuntimeArgs, JobArgs
from chaperone.data import SELECTIONS
from blocks import Blocks

# ── Configuration ──────────────────────────────────────────────────────────
SCALE = "--scale" in sys.argv

SCRIPT_DIR = Path(__file__).resolve().parent
NUM_DESIGNS = 100 if SCALE else 1

if SCALE:
    RUNDIR = SCRIPT_DIR / "out_full"
    RUN_ARGS = RuntimeArgs(mode="cluster_only")
else:
    RUNDIR = SCRIPT_DIR / "out_pilot"
    RUN_ARGS = RuntimeArgs(mode="local")

GPU_JOB = JobArgs(gpu_class="small", mem_gb=32, time_limit="08:00:00")

# ── Block setup ────────────────────────────────────────────────────────────
RFD3Block = Blocks.get("rfd3")
MPNNBlock = Blocks.get("mpnn")
AF3Block = Blocks.get("af3")
RMSDBlock = Blocks.get("rmsd")

rfd3 = RFD3Block(
    inputs={
        "binder": {
            "contig": "50-80",
            "input": "path/to/target.pdb",
            "ligand": "LIG",
        }
    },
    n_batches=NUM_DESIGNS // 8,
    batch_size=8,
)
mpnn = MPNNBlock(model_type="ligand_mpnn", num_seq_per_pdb=8)
af3 = AF3Block(read_most_confident_model=False)
rmsd = RMSDBlock(
    alignment_methods=[SELECTIONS.BKBN],
    rmsd_targets=[SELECTIONS.BKBN, SELECTIONS.LIGANDS],
)

# ── Run pipeline ───────────────────────────────────────────────────────────
engine = ExecutionEngine(rundir=str(RUNDIR), run_args_default=RUN_ARGS)

h = engine.run(rfd3, job_args=GPU_JOB)
h = engine.run(mpnn, data=h)
h = engine.run(af3, data=h, job_args=GPU_JOB)
engine.run(rmsd, data=h)

# ── Export results ─────────────────────────────────────────────────────────
handler = engine.compile()
df = handler.unpack()
df.to_csv(str(RUNDIR / "pipeline.csv"), index=False)
print(f"Saved {df.shape[0]} rows → {RUNDIR / 'pipeline.csv'}")
```

### Script recommendations

- Avoid over-specifying the main script on the first pass; keep it as clean as possible. E.g. do not specify a custom job_args - assume that the default is the best

---

## `chaperone.init_cli()` — CLI Flags for Standalone Scripts

Standalone pipeline scripts can opt into CLI flag parsing with a single `init_cli()` call.
This registers all block flags and global runtime flags, then parses `sys.argv`.

### Basic Usage

```python
from chaperone import init_cli, ExecutionEngine

init_cli()  # registers flags, parses argv, sets up logging

engine = ExecutionEngine(rundir="my_run")
engine.run("rfd3", n_batches=10)  # --rfd3.n_batches=50 overrides this
```

```bash
python script.py --rfd3.n_batches=50 --recompile --rundir=/tmp/test
python script.py --partition=gpu --gres=gpu:a100:1  # retarget all GPU blocks (use your cluster's names)
python script.py --help  # prints all available flags
```

### Priority Order (highest wins)

```
CLI flags (--rfd3.n_batches=50)  >  script kwargs (n_batches=10)  >  block Pydantic defaults
```

### Global Flags

| Flag | Description |
|------|-------------|
| `--rundir` | Override output directory (wins over script argument) |
| `--recompile` | Force recompilation from existing sandbox directories |
| `--force_exec` | Force execution even if cached results exist |
| `--local` | Use local execution instead of SLURM |
| `--nouse_local_gpus` | Force SLURM submission even when local GPUs are available |
| `--pickup` | Pick up parquet from existing run directory |
| `--partition=<str>` | SLURM partition for all GPU blocks, or `local` (CPU blocks are unaffected). If unset, uses the site's GPU partition (`CHAPERONE_GPU_PARTITION`), then `CHAPERONE_DEFAULT_PARTITION`, then the cluster default. Overriding it drops the GPU type from the block's gres (`gpu:a100:1` becomes `gpu:1`) unless `--gres` is also given. IPD example: `gpu-bf` |
| `--gres=<str>` | SLURM `--gres` for all GPU blocks, passed to `sbatch` as-is (e.g. `gpu:1`, `gpu:a100:1`). Valid GPU types depend on the cluster. If unset, uses each block's size class mapped through `CHAPERONE_GRES_MAP`, else `gpu:1` |
| `--account=<str>`, `--qos=<str>` | SLURM `--account` / `--qos` for all blocks. If unset, `CHAPERONE_DEFAULT_ACCOUNT` / `CHAPERONE_DEFAULT_QOS` |
| `--extra_sbatch=<str>` | Extra long `sbatch` options for all blocks, space-separated and quoted, e.g. `--extra_sbatch='--reservation=mine --mail-type=END'` |

### Shebang for Auto-Activation

Scripts can use this shebang to run without manually activating the venv:

```python
#!/usr/bin/env -S bash -c '"${CHAPERONE_REPO:-.}/.venv/bin/python" "$0" "$@"'
from chaperone import init_cli, ExecutionEngine
init_cli()
...
```

Set `CHAPERONE_REPO` in your shell profile to point to the Chaperone repository root, or run from the repo directory.
