---
name: scripting
description: Expert guidance on writing, running, and testing standalone Chaperone scripts. Use when the user wants to write a pipeline script, debug CLI flags, understand the build process (test → pilot → production), set up cycling, or investigate how external tool scripts work.
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write", "Edit"]
---

# Chaperone Scripting

Guide for writing, running, and testing standalone Python scripts that use the Chaperone framework. Chaperone uses a code-based execution environment instead of a dedicated MCP for each tool. You are expected to know how to easily write, manage and debug scripts and Chaperone code using the guidance provided as well as the repository structure.

A trimmed copy of the repository may omit optional directories such
as `examples/` and `tests/`. Check that optional paths exist before including
them in searches; use `src/`, `docs/site/`, and the task workspace when they are
the only material supplied.

You can both write Chaperone code as single-call python scripts for flexibility:

```bash
source .venv/bin/activate && python -c "..."
```

Or build the pipeline as a standalone script. For scripts, there are additional tools like the shebang line and `init_cli`. In general, scripts follow a simple structure (see below for more details):

## Chaperone scripts

Chaperone scripts function as configs but can be modified / tested easily by supplying command line args - you could also modify the script parameters directly, but using cli arguments will save you the extra steps to manage the file state.
They should be succinct, with minimal unnecessary overrides. For example:
- Don't over-specify the block parameters and specify the necessary ones.
- Bundle multiple steps into one (supplying lists of blocks instead of single blocks to each `engine.run` call) to maximize the amount of work each job submission step does.
- DO create a specific 'files' directory for bespoke input files such as theozyme.pdb, af3_json_template.json, binder_target.pdb etc.

Every script should start with this shebang so it auto-activates the Chaperone venv (remember to `chmod +x`):

```python
#!/usr/bin/env -S bash -c '"${CHAPERONE_REPO:-.}/.venv/bin/python" "$0" "$@"'
"""Short description of what this script does."""
from pathlib import Path
from blocks import Blocks
from blocks.util_blocks.filter import MetricFilter
from chaperone import ExecutionEngine, init_cli
from chaperone.data.schema import SELECTIONS

init_cli()  # MUST be called before any engine usage

SCRIPT_DIR = Path(__file__).resolve().parent
FILES_DIR = SCRIPT_DIR / "files"

RFD3, MPNN, AF3, RMSD, Filter = Blocks.getall("rfd3", "mpnn", "af3", "rmsd", "filter")

# ── Block configuration ─────────────────────────────────────────────────

blocks = [
    RFD3(inputs=str(FILES_DIR / "config.json"), n_batches=10, batch_size=10),
    Filter(name="rfd3_filter", filters={...}),
    MPNN(),
    AF3(diffusion_batch_size=5),
    RMSD(
        alignment_methods=[SELECTIONS.BKBN, SELECTIONS.CA],
        rmsd_targets=[SELECTIONS.BKBN, SELECTIONS.CA],
    ),
    Filter(name="af3_filter", filters={...}),
]

# ── Execute ──────────────────────────────────────────────────────────────

engine = ExecutionEngine(rundir=SCRIPT_DIR / "output")
handler = engine.run(blocks)
df = handler.unpack()
print(df)
```

### Key rules

1. The shebang line can be used to make environment activation easier to run it as: `./script.py <flags>`
2. `init_cli()` must be called at the top of the file. This registers all block flags + global flags, sets up `absl` to handle args, and sets up logging.
3. Use `SCRIPT_DIR` for relative paths — scripts should be portable.
4. Set `rundir` to where production outputs go (e.g. `/net/scratch/...`).
5. CLI flags override everything — script kwargs are just defaults for the production run.
6. **Split config from execution**: use `Blocks.getall()` to load block classes, configure all blocks as a list, then pass the list to a single `engine.run(blocks)` call. Only break into multiple `engine.run()` calls when branching logic is required (e.g. `engine.cycle`, conditional filtering, `engine.run_parallel`).
7. **Raw structures are not tables**: the plain `ingest` block loads `.csv` or
   `.parquet`; do not pass it a lone PDB/CIF path or a structure-only directory.
   Start a structure-backed flow with a standard handler instead:
   `PipelineDataHandler([make_row(name="input", path=str(structure), design_id="input")])`.
   Use an ingest compiler only when its structure-compilation behavior is
   specifically required.

---

### `init_cli()`

`init_cli()` enables CLI flag overrides without the `chaperone run` entry point:
```python
from chaperone import init_cli, ExecutionEngine
init_cli()  # registers all block flags + global flags, parses sys.argv, sets up logging
engine = ExecutionEngine(rundir="my_run")
engine.run("rfd3", n_batches=10)  # --rfd3.n_batches=50 overrides this
```

**Shebang** — use this so scripts automatically use the Chaperone venv:
```
#!/usr/bin/env -S bash -c '"${CHAPERONE_REPO:-.}/.venv/bin/python" "$0" "$@"'
```
Requires GNU coreutils `env` with `-S` support. Portable fallback: `#!/usr/bin/env python`.

**Override priority** (highest wins):
1. CLI flags (`--rfd3.n_batches=1`)
2. CLI config (`--config=overrides.yaml`)
3. Script kwargs (`engine.run("rfd3", n_batches=10)`) / pre-instantiated block fields
4. Script YAML defaults (`init_cli(defaults_yaml="defaults.yaml")`)
5. Pydantic field defaults

### Global flags

| Flag | Effect |
|------|--------|
| `--rundir=/path` | Override the script's `rundir` |
| `--recompile` | Force recompilation from sandbox (skip execution, just re-collect outputs) |
| `--force_exec` | Force full re-execution, ignoring all caches |
| `--local` | Use local execution instead of SLURM |
| `--use_local_gpus` | Auto-detect and use local GPUs |
| `--nouse_local_gpus` | Force SLURM even if local GPUs exist |
| `--pickup` | Pick up parquet from existing run directory |

**Global flags:** `--rundir`, `--recompile`, `--force_exec`, `--local`, `--nouse_local_gpus`, `--partition`, `--gres`. Run any pipeline script with `--help` for the full flag list.

### Block-specific flags

Any Pydantic field on a block can be overridden: `--{block_name}.{field}={value}`

```bash
./script.py --test  # Shorthand for --config=tests/test_overrides.yaml
./script.py --rfd3.n_batches=1 --rfd3.batch_size=1
./script.py --mpnn.num_seq_per_pdb=4
./script.py --af3.num_diffusion_samples=1
./script.py --protein_hunter.num_designs=5
```

### Priority order

**CLI flags > script kwargs > block Pydantic defaults**

So `engine.run("rfd3", n_batches=10)` in the script is overridden by `--rfd3.n_batches=1` on the command line.

---

## Production vs Testing

The core pattern: write scripts with production / pilot defaults, then override at the CLI for quick local testing.

### Production script

```python
RFD3, MPNN, AF3 = Blocks.getall("rfd3", "mpnn", "af3")

blocks = [
    RFD3(inputs=str(DATA_DIR / "rfd3_config.json"), batch_size=32, n_batches=100),
]

engine = ExecutionEngine(rundir="/net/scratch/user/fad_binders")
engine.run(blocks)

# Cycling requires multiple engine.run calls (branching logic)
for cycle_engine in engine.cycle("opt", 5):
    cycle_engine.run("mpnn", model_type="ligand_mpnn", num_seq_per_pdb=4)
    cycle_engine.run("af3", diffusion_batch_size=5)
```

## Build process (How to build scripts robustly)

In general, adopt a three-stage build-process:
1. Test the script on a minimal compute task (one design for everything) to ensure everything looks good. Ideally the user has provided you a GPU (and use `--use_local_gpus`), and you use a temporary directory (`--rundir=/tmp/test_fad`).
2. Pilot the script to probe pass rates. Pilot runs are often on partition `gpu` or `gpu-bf` depending on urgency.
3. Production scale design: taking pilot example further or scaling the raw number of designs generated to get better hits. Production runs are almost always on partition `gpu-bf`

Steps 2. and 3. can sometimes be merged depending on user requests and constraints.

### Test the same script locally (1 design, local GPU, temp dir)

```bash
./script.py --test --rundir=/tmp/test_fad --use_local_gpus
```

This applies test overrides (1 design per block), reroutes output to `/tmp/test_fad`, and uses local GPUs to verify the pipeline works end-to-end before committing to a large production run.

### Cycling (Iterative Refinement)

Use `engine.cycle()` for MPNN -> Oracle loops:

Common pattern: `rfd3 -> cycle(mpnn -> af3)`.

```python
N_CYCLES = 5

engine.run("rfd3", inputs=str(config), batch_size=32, n_batches=10)

records=[]
for cycle_idx, cycle_engine in enumerate(engine.cycle("opt", N_CYCLES)):
    cycle_engine.run("mpnn", model_type="ligand_mpnn", num_seq_per_pdb=1)
    handler = cycle_engine.run("af3", num_diffusion_samples=5)

    # Optional: inspect per-cycle results
    df = handler.unpack()
    pldtt=df['af3.plddt'].mean()
    print(f"Cycle {cycle_idx}: {len(df)} designs, mean plddt={plddt:.2f}")
    records.append({"cycle": cycle_idx, "plddt", plddt})
```

### Finishing up testing

When finished, report to the user the test results, data in the outputs etc., but also include what the directory structure looks like (by printing the engine)

---

## Using external scripts

When a block wraps an external tool (e.g. RFD3, MPNN, RF3, AF3), its `create_command` builds CLI arguments for that tool's inner script. If you're unsure how the inner script's parameters work — **actively investigate**:
1. Clone the upstream repo (check `_script_spec.source_url`) into `/tmp/` and read the source
2. Look at the tool's `--help` output (via the container or local install)
3. Read the tool's README or docs (via `source_url` or web search)
4. Check the `pyproject.toml` `[project.scripts]` for CLI entry points

Do NOT guess parameter names or semantics — verify them. If you discover a discrepancy between the block's parameters and the inner tool's actual interface, fix it and PR the fix. This is how the repo stays accurate as upstream tools evolve.

## Shareable Pipeline Scripts

When creating scripts for others to run:

- **Shebang**: Use `#!/usr/bin/env -S bash -c '"${CHAPERONE_REPO:-.}/.venv/bin/python" "$0" "$@"'`
- **All paths must be absolute** — input parquets, rundirs, output paths
- **Permissions**: `chmod +x script.py` and `chmod -R g+w` on any output directories so `baker` group members can write
- **Container mounts**: `/home` and `/mnt/home` resolve differently — containers may only mount `/mnt/home`. Ensure blocks bind both (see AF3 container binds)
- **Simple Run command**: `CHAPERONE_REPO=/home/user/chaperone /home/user/chaperone/.venv/bin/python /path/to/script.py`
