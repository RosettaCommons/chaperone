# Pipelines

A framework for making structural biology tools interoperable.

Pipelines provides a unified interface for running protein design and prediction tools
(RFdiffusion3, AlphaFold3, ProteinMPNN, Boltz-2, and more) as composable blocks with
automatic job orchestration, data lineage tracking, and reproducible execution.

## Quick Links

- **[Simple Example](simple.md)** — Minimal pipeline walkthrough (RFD3 → MPNN → AF3)
- **[All Blocks](blocks-reference.md)** — Auto-generated parameter docs for every block, with primary/secondary tier labels
- **[Syntax Reference](syntax.md)** — ExecutionEngine, PipelineDataHandler, JobArgs, RuntimeArgs API reference
- **[Validation](validation.md)** — Score any structure predictor on the PDB validation cohort
- **[Evaluations](evaluations.md)** — Run the agent evaluation harness on the design tasks

## Getting Started

```bash
# Install
uv venv --python=3.12.3
source .venv/bin/activate
uv pip install -e '.[all]' --override overrides.txt

# Check which blocks and containers are available
pipelines list-blocks
```

## Architecture

Each pipeline follows a composable block pattern:

1. **Backbone generation** (`rfd3`, `rfd1`, `disco`) — generate diverse backbones
2. **Filtering** (`filter`) — prune low-quality backbones early
3. **Sequence design** (`mpnn`, `foundry_mpnn`, `abmpnn_poe`) — inverse fold sequences
4. **Structure prediction** (`af3`, `rf3`, `boltz2`) — validate with a prediction oracle
5. **Metrics + filtering** (`rmsd`, `rasa`, `hbond`, `filter`) — assess quality and select top designs

Advanced patterns include **cycling** (iterative MPNN → oracle loops), **parallel submission** (`engine.submit()` + `gather()`), and **branching** (multiple prediction configurations).

See the [Simple Example](simple.md) for a complete pipeline, and [Claude Skills](skills.md) for the guided workflows (`/setup` configures containers and compute).
