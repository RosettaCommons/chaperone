# Chaperone

A framework for making structural biology tools interoperable.

Chaperone provides a unified interface for running protein design and prediction tools
(RFdiffusion3, AlphaFold3, ProteinMPNN, Boltz-2, and more) as composable blocks with
automatic job orchestration, data lineage tracking, and reproducible execution.

## Quick Links

- **[Simple Example](simple.md)** — Minimal pipeline walkthrough (RFD3 → MPNN → AF3)
- **[Core Examples](core-examples.md)** — Linked scripts, benchmark objectives, and validation limits
- **[Agent Skills](skills.md)** — Guides for setup, planning, scripting, and debugging with your coding agent
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
chaperone list-blocks
```

## Architecture

Each pipeline follows a composable block pattern:

1. **Backbone generation** (`rfd3`, `rfd1`, `disco`) — generate diverse backbones
2. **Filtering** (`filter`) — prune low-quality backbones early
3. **Sequence design** (`mpnn`, `foundry_mpnn`, `abmpnn_poe`) — inverse fold sequences
4. **Structure prediction** (`af3`, `rf3`, `boltz2`) — validate with a prediction oracle
5. **Metrics + filtering** (`rmsd`, `rasa`, `hbond`, `filter`) — assess quality and select top designs

Advanced patterns include **cycling** (iterative MPNN → oracle loops), **parallel submission** (`engine.submit()` + `gather()`), and **branching** (multiple prediction configurations).

See the [Simple Example](simple.md) for a pipeline walkthrough and [Agent Skills](skills.md) for guided workflows. Skills are Markdown instructions that Claude Code, Codex, and other agents can read; automatic discovery and slash commands depend on the agent installation. They still require working containers, weights, and compute access.

Chaperone-authored code is distributed under the [MIT license](https://github.com/RosettaCommons/chaperone/blob/main/LICENSE). [Bundled third-party code and example outputs](https://github.com/RosettaCommons/chaperone/blob/main/THIRD_PARTY_NOTICES.md), upstream tools, and model weights retain their own terms and access requirements. See the [installation instructions](https://github.com/RosettaCommons/chaperone#installation-manual) to configure your own system, or supply an untracked `.env.ipd` for an IPD installation.
