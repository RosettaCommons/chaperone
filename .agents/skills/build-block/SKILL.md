---
name: build-block
description: Build a new block implementation for a computational tool, including container definition and block code. Use when the user says "Build a block for this tool" and provides a URL or documentation.
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write", "Edit", "WebFetch", "Agent"]
---

# Build Block Skill

End-to-end guide for creating a new Chaperone block from a tool's documentation or repository URL.

## Overview

Building a block has 5 phases:
1. **Research** - Understand the tool's installation, CLI, inputs/outputs
2. **Container** - Create a container definition and verify it builds/runs
3. **Block** - Implement the block Python class
4. **Input Documentation** - Research and document the input config format
5. **Validate** - Test the block and prepare a PR

---

## Phase 1: Research the Tool

### 1a. Gather Documentation

If the user provides a URL:
- Clone the repo to a temporary location: `/tmp/block-build-{tool_name}`
- Read the README, installation docs, and example usage
- Identify: base image needs, dependencies, GPU requirements, CLI interface, input/output formats

If the user provides documentation text or a README directly, parse that instead.

### 1b. Identify Key Information

Extract and summarize to the user:
- **Tool name**: Short lowercase name for the block (e.g., `rfd3`, `esmfold`)
- **Installation method**: pip, conda, source build, etc.
- **GPU required**: Yes/No, which CUDA version
- **CLI entry point**: The command to run the tool (e.g., `python run.py`, `rf3 predict`)
- **Input format**: What files/arguments the tool expects (PDB, FASTA, JSON, etc.):
    - How to convert the data from the handler into them appropriataly (if the block takes designs as inputs; like MPNN, prediction or scoring), or;
    - How to appropriately write the prepare_inputs block when there isn't handler (see "rfd3" example).
- **Output format**: What the tool produces (PDB, CIF, scores JSON, etc.)
- **Dependencies**: Key packages (PyTorch, JAX, etc.)

Use `AskUserQuestion` to confirm the tool name, verify GPU needs, and ask about any ambiguities before proceeding.

### 1c. Find a Similar Existing Block

Look at existing blocks for one with a similar pattern:

| Pattern | Example Block | When to Use |
|---------|--------------|-------------|
| GPU inference, JSON/config input | `rfd3` (`rfdiffusion3.py`) | Diffusion models, generative tools |
| GPU inference, PDB-in PDB-out | `af3` (`af3.py`) | Structure prediction |
| CPU/GPU, PDB-in multi-PDB-out | `mpnn` (`mpnn.py`) | Sequence design, inverse folding |
| CPU-only utility | `annotate` (`annotate.py`) | Analysis, scoring |

Read the chosen reference block in full — it will be the template.

For most cases you will want to write a single file (if the tool's container is self-contained, like boltzgen) or you may want to write a directory like `foundry` which hosts multiple independent models tied to the same container. If it is not obvious, ask the user what their preference is.

---

## Phase 2: Create the Container Definition

### 2a. Write the Container Image

Create a new file for the container. Choose the location based on whether the tool is standalone or part of a family:

- **Standalone tool**: `src/blocks/networks/{tool_name}.py` (container + block in same file, or separate `container.py` if complex)
- **Tool family** (e.g., foundry tools): `src/blocks/networks/{family}/container.py`

Use the `Image` builder pattern from `chaperone.containers`. For example:

```python
from chaperone.containers import Image
from blocks.constants import CUDA_INSTALL  # Only if GPU tool needs CUDA

{TOOL}_CONTAINER = (
    Image("{tool_name}")
    .base_image("ubuntu:24.04")  # or nvidia/cuda:12.6.0-devel-ubuntu24.04 for GPU
    .packages(
        "build-essential", "git", "wget", "curl", "ca-certificates",
        "python3.12", "python3.12-dev", "python3.12-venv", "python3-pip",
        # Add tool-specific system packages here
    )
    .run(*CUDA_INSTALL)  # Only if GPU tool
    .run(
        "python3.12 -m venv /opt/{tool_name}-env",
        ". /opt/{tool_name}-env/bin/activate && pip install --upgrade pip setuptools wheel",
    )
    .run(
        # Clone and install the tool
        "cd /opt && git clone {repo_url} {tool_name}",
        ". /opt/{tool_name}-env/bin/activate && cd /opt/{tool_name} && pip install -e .",
    )
    .run("apt-get clean && rm -rf /var/lib/apt/lists/*")
    .env(
        PATH="/opt/{tool_name}-env/bin:$PATH",
        VIRTUAL_ENV="/opt/{tool_name}-env",
        # Add CUDA env vars if GPU tool:
        # CUDA_HOME="/usr/local/cuda",
        # LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH",
    )
    .label(Version="v1.0.0")
    .runscript('"$@"')
    .template_command(
        exec_mode="exec",
        binds=[("/mnt/home", "/mnt/home")],
        env={},  # Add runtime env vars if needed
    )
)
```

### 2b. Build the Container

Try building the container (selects apptainer or docker in the background):

```bash
source .venv/bin/activate
chaperone install <tool_name>
```

### 2c. Test the Container

Run an example command from the tool's documentation inside the container:

```bash
# Apptainer
apptainer exec --nv /tmp/{tool_name}.sif {tool_entry_point} --help

# Docker
docker run --gpus all chaperone/{tool_name}:latest {tool_entry_point} --help
```

Verify the tool runs and responds correctly. If it fails, debug and fix the container definition.

**Ask the user** if they want to proceed to building the block after the container is verified.

---

## Phase 3: Implement the Block

### 3a. Create the Block File

Create `src/blocks/networks/{tool_name}.py` (or within a subdirectory if it belongs to a family).

Every `CompiledBlockFunction` block needs:

1. **Class attributes**: `_name`, `_description`, `_image`, `_resources`, `_pipeline_inputs`, `_inline=False`, `_chunked=True`
2. **Pydantic Fields**: Block-specific configuration parameters
3. **`command_template`** property: Returns the base container command
4. **`prepare_inputs(handler, sandbox_dir)`**: Validates/transforms input handler
5. **`create_command(handler, sandbox_dir, chunk_id)`**: Builds the full CLI command for a chunk
6. **`compile(sandbox_dir)`**: Collects outputs into a `PipelineDataHandler`

Template:

```python
import os
import json
import logging
from typing import Optional
from pydantic import Field
from chaperone import CompiledBlockFunction, PipelineDataHandler, JobArgs
from chaperone.data import make_row
from chaperone.common import glob_designs_in_dir

logger = logging.getLogger(__name__)

# Import or define the container here
from blocks.networks.{module}.container import {TOOL}_CONTAINER
# OR define inline:
# {TOOL}_CONTAINER = Image("{tool_name}").base_image(...)...

class {ToolName}Block(CompiledBlockFunction):
    """{Tool description}."""

    _name = "{tool_name}"
    _description = "{Short description}"
    _image = {TOOL}_CONTAINER
    _resources = JobArgs(
        partition="gpu",        # or "cpu"
        gres="gpu:a4000:1",    # omit for CPU-only
        mem_gb=16,
        time_limit="04:00:00",
        rows_per_job=100,
    )
    _pipeline_inputs = ["last"]  # [] if de novo (no input handler needed)
    _inline = False
    _chunked = True

    # === Block-specific parameters as Pydantic Fields ===
    # Example:
    # num_samples: int = Field(8, description="Number of samples to generate")

    @property
    def command_template(self) -> list[str]:
        return self._image.build_command("{cli_entry_point}")

    def prepare_inputs(self, handler, sandbox_dir):
        """Validate and prepare inputs."""
        # For blocks that require input structures:
        assert handler is not None, "{tool_name} requires input structures"
        return handler
        # For de novo blocks that don't need input:
        # return PipelineDataHandler([{Column.STRUCTURES: [], Column.METRICS: "{}"}])

    def create_command(self, handler, sandbox_dir, chunk_id) -> list[str]:
        """Build the CLI command for this chunk."""
        sandbox_dir = os.path.abspath(sandbox_dir)

        # Collect input paths from handler
        input_paths = [data.path for data in handler.get_data(which="last")]

        # Build command
        cmd = self.command_template + [
            "--output_dir", sandbox_dir,
            # Add tool-specific arguments here
        ]

        # Add input files
        for path in input_paths:
            cmd.extend(["--input", path])

        return cmd

    def compile(self, sandbox_dir):
        """Collect outputs into a PipelineDataHandler."""
        paths = glob_designs_in_dir(dir=sandbox_dir)

        if not paths:
            raise RuntimeError(
                f"{self._name} produced no output files in {sandbox_dir}. "
                f"Check logs for errors."
            )

        rows = []
        for path in paths:
            # Read metrics from sidecar JSON if the tool produces one
            metrics = {}
            meta_path = path.replace(".pdb", ".json").replace(".cif", ".json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    metrics = json.load(f)

            rows.append(
                make_row(
                    name=self.name,
                    path=path,
                    metadata={},   # design-level metadata
                    metrics={self.name: metrics},
                )
            )

        return PipelineDataHandler(rows)
```

### 3b. Verify Block Discovery

Check that the block is discovered by the lazy registry:

```bash
source .venv/bin/activate
python -c "from blocks import Blocks; print(Blocks['{tool_name}'])"
```

### 3c. Test the Block

Write a minimal test or run the block via the engine:

```python
source .venv/bin/activate
python -c "
from chaperone import ExecutionEngine
engine = ExecutionEngine(rundir='/tmp/test_{tool_name}')
# If the block needs input, provide a handler:
# from chaperone import PipelineDataHandler
# handler = PipelineDataHandler.from_dir('path/to/pdbs', name='input')
# result = engine.run('{tool_name}', data=handler)

# If de novo:
# result = engine.run('{tool_name}', run_args={...})
print(result)
"
```

There will be test data under ./tests/test_data

---

## Phase 4: Input Documentation

For **design blocks** (generative tools that take a config/JSON as input rather than upstream PDB files), create an `input.md` file documenting how to prepare input configs. This is critical because design tools often have bespoke, complex input formats that users need to understand.

### 4a. Research the Input Format

Thoroughly investigate the tool's input format by:
1. Reading the upstream repo's README, docs, and example configs
2. Reading the actual input parsing code (e.g. `data/`, `utils/`, `parser.py`) to understand all supported fields, types, and defaults
3. Collecting example input files from the repo (e.g. `examples/`, `configs/`, `input_jsons/`)

Focus on:
- **Config schema**: All top-level and nested fields, their types, required vs optional, defaults
- **Entity/input types**: What kinds of inputs are supported (protein sequences, ligands, DNA, etc.)
- **Special syntax**: Masking conventions, file path prefixes, CCD codes, etc.
- **Constraints and limitations**: What combinations are valid, max sizes, unsupported features

### 4b. Write `input.md`

Create the file at the block's directory: `src/blocks/networks/{category}/{tool_name}/input.md`

The file should cover, in order:

1. **Overview** — One-line summary of the input format (e.g. "JSON list of jobs following AF3 Server format")
2. **Config structure** — The top-level JSON/YAML schema with a template showing all fields
3. **Entity/input types** — Detailed section for each supported input type, with:
   - Field table (name, type, default, description)
   - Masking or special syntax conventions
   - Concrete JSON/YAML snippets
4. **Advanced features** — Covalent bonds, constraints, templates, or other optional features
5. **Complete examples** — 5-8 realistic, copy-pasteable examples covering common use cases:
   - Simplest case (e.g. unconditional generation)
   - Each major conditioning type (ligand, DNA, RNA, etc.)
   - Partial design / mixed fixed+designable
   - Multiple jobs in one file
6. **Known limitations** — Numbered list of things that don't work or aren't supported

**Style guidelines:**
- Focus on **config preparation**, not CLI arguments (those belong in the block's docstring/README)
- Include CLI argument tables only for parameters that directly affect input interpretation (e.g. `experiment=designable` vs `diverse`)
- Use tables for field documentation, code blocks for examples
- Note any downstream gotchas (e.g. "ligands are named `l01` in output PDB — downstream AF3 needs `specified_ccd` mapping")
- Keep examples realistic — use actual protein lengths, real ligand SMILES/CCD codes

### 4c. When to Skip

Skip this phase for blocks that:
- Take upstream PDB/CIF files as input (prediction blocks, MPNN, scoring) — their input format is the `PipelineDataHandler`
- Have trivially simple inputs (e.g. a single FASTA file)
- Are utility blocks (filter, annotate, ingest)

The rule of thumb: if a user needs to **author a config file** to use the block, write `input.md`.

---

## Phase 5: Validate and PR

### 5a. Run Tests

```bash
source .venv/bin/activate
pytest tests/ -v -x
```

Ensure no existing tests are broken.

### 5b. Summary

Show the user a summary of all created/modified files:
- Container definition file
- Block implementation file
- Input documentation (`input.md`) if applicable
- Any test files

### 5c. Offer to Create PR

Ask the user if they want to commit and create a PR. If yes, follow the standard commit and PR creation workflow.

---

## Key References

| File | Purpose |
|------|---------|
| `src/chaperone/containers.py` | `Image` builder class |
| `src/blocks/constants.py` | `CUDA_INSTALL` and shared constants |
| `src/chaperone/block_config/compiled.py` | `CompiledBlockFunction` base class |
| `src/chaperone/block_config/base.py` | `BlockFunction` base class |
| `src/chaperone/data/handler.py` | `PipelineDataHandler` |
| `src/chaperone/data/schema.py` | `make_row()`, `Column`, `merge_rows()` |
| `src/chaperone/common.py` | `glob_designs_in_dir()` |
| `src/blocks/__init__.py` | Lazy block registry (AST scanning) |
| `src/blocks/networks/foundry/rfdiffusion3.py` | Reference: complex GPU block |
| `src/blocks/networks/mpnn.py` | Reference: simpler PDB-in/PDB-out block |
| `src/blocks/networks/prediction/alphafold3.py` | Reference: structure prediction block |

## Common Pitfalls

- **Block `_name` must be unique** across all blocks — the registry raises `ValueError` on duplicates
- **`_name` must be a simple string assignment** (e.g., `_name = "foo"`) — the AST scanner only finds literal assignments
- **Container image ref uses `${CONTAINER:name}`** placeholder — resolved at runtime by `_resolve_command()` in `CompiledBlockFunction`
- **`glob_designs_in_dir()`** finds `.pdb`, `.cif`, and `.cif.gz` files recursively
- **`make_row()`** creates a properly structured row dict with `structures` and `metrics` columns
- **Don't forget `_chunked = True`** if the block should support parallel execution across chunks
- **Match `_pipeline_inputs`** to whether the block is de novo (`[]`) or requires upstream data (`["last"]`)
- **Generative steps must seed/name chunks differently** — For generative blocks (e.g., `boltzgen`, `rfd3`) that are used as the first step in a pipeline, each chunk's `chunk_id` should be used to seed the random number generator or differentiate output naming. If chunks are not seeded differently, multiple chunks for a generative first step will produce identical outputs, defeating the purpose of parallelization. Use the `chunk_id` in `create_command()` to pass a unique seed or prefix to the tool's CLI.
