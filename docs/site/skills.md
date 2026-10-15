# Agent Skills

Chaperone includes **agent skills**: focused guides for planning, running, and debugging protein-design workflows. Each guide is a Markdown file with YAML metadata at `.agents/skills/<name>/SKILL.md`. The guides describe repository conventions and scientific workflows that an agent can follow with access to the repository and the required tools.

## Using skills with your agent

Start with the repository's [agent guide](https://github.com/RosettaCommons/chaperone/blob/main/AGENTS.md) and [skills index](https://github.com/RosettaCommons/chaperone/blob/main/.agents/skills/INDEX.md), then load the single skill that matches the task. You can ask in plain language, for example:

> Read AGENTS.md and .agents/skills/INDEX.md, then use .agents/skills/design/SKILL.md to plan a tested pipeline for my target.

Automatic skill discovery and commands such as `/design` depend on the agent application. If your application does not discover this directory, point it directly to the relevant `SKILL.md`. Slash names below are shorthand for the guides, not a promise that every application registers those commands. Tool names and optional permissions metadata may also need adapting to your application's capabilities.

Skills provide instructions; they do not install the tools or grant access to compute. Running a workflow still requires the relevant Chaperone environment, containers, model assets, and execution backend. Some guides have additional requirements: Rosetta for ligand parameters, a running PyMOL XML-RPC server for visualization. Use the allocation commands and GPU resource names supported by your cluster.

The inventory below follows the [skills shipped in this repository](https://github.com/RosettaCommons/chaperone/tree/main/.agents/skills). Consult each source guide for its full procedure and prerequisites.

## Chaperone-specific skills

Skills that directly concern writing, running, or debugging Chaperone code.

| Skill | Purpose | Typical triggers |
|-------|---------|------------------|
| [`/setup`](#setup) | Install and check Chaperone functionality | "set up Chaperone", "install", "check my environment" |
| [`/design`](#design) | Plan and convert specs into clean, tested pipeline code | "how do I design this protein?", "convert this notebook into a pipeline" |
| [`/scripting`](#scripting) | Writing, running, and testing standalone pipeline scripts | "write a pipeline script", "debug CLI flags", "test → pilot → production" |
| [`/blocks-reference`](#blocks-reference) | Choose and compare blocks by task, modality, cost and limitation | "which tool", "alternative to", "compare oracles" |
| [`/auto-research`](#auto-research) | Iteratively improve a weak or stalled workflow against a scorer | "improve this pipeline", "stalled campaign" |
| [`/build-block`](#build-block) | Build a new block implementation from a tool's URL/docs | "build a block for this tool" |
| [`/debugging-inputs`](#debugging-inputs) | Troubleshoot input prep issues (chain IDs, ligands, parsing) | "RMSD errors", "AF3 input mismatches", "unexpected chain assignments" |

## Protein design and process skills

General skills for protein design workflows and peripheral tooling.

| Skill | Purpose | Typical triggers |
|-------|---------|------------------|
| [`/rosetta-params`](#rosetta-params) | Generate/debug Rosetta `.params` files for ligands | "make a params file", "rotamer generation", "ligand charges" |
| [`/order-collection`](#order-collection) | Package a campaign into a handoff bundle | "collect orders", "bundle designs", "prepare handoff" |
| [`/visualize`](#visualize) | Send structures and commands to PyMOL via XML-RPC | "visualize", "show in PyMOL", "view structure" |
| [`/slurm-interactive`](#slurm-interactive) | Persistent interactive SLURM session (GPU/CPU) using your cluster’s allocation command | "get me a gpu", "allocate a gpu/cpu", "gpu session", "qlogin" |
| [`/slurm-background`](#slurm-background) | Submit detached SLURM jobs for pilots/production/campaigns | "submit a job", "run in background", "production run", "sbatch" |
| [`/notebook-editing`](#notebook-editing) | Edit, create, and execute Jupyter notebooks | any `.ipynb` interaction |

---

## Skill details

### `/setup`

Guides new users through Chaperone environment configuration: venv creation, `uv` installs, `.env` setup, container backend checks, and a smoke test.

Use this when first cloning the repo, when collaborators hit install errors, or when the main branch migrates and environments need updating.

### `/design`

The top-level planning skill. Given a user specification (a target protein, a notebook to convert, or an open-ended "how do I design X?"), it walks through the four pipeline levels — code, integration, pilot, campaign — and produces a plan that maps to existing blocks and examples.

### `/scripting`

Expert guidance on standalone pipeline scripts: the shebang line, `init_cli()`, CLI flag overrides, `engine.run()` chaining, `engine.cycle()` for iterative refinement, and the test → pilot → production build process. Also covers investigating inner-tool CLIs when a `CompiledBlockFunction` wraps an external script.

### `/blocks-reference`

Choose and compare blocks by scientific task, modality, speed, cost, evidence quality and known limitations: a design-funnel block, an oracle or metric, or a faster alternative to a slow block.

### `/auto-research`

Iteratively improve a weak, uncertain or stalled workflow using your thresholds or an existing Chaperone-compatible scorer. Turns thresholds into one tested `score.py`, then runs bounded pilots before scaling.

### `/build-block`

End-to-end guide for adding a new block from a tool's repo or documentation URL. Covers container definition with `chaperone.containers.Image`, block code layout, `ScriptSpec`, input/output helpers, and registering the block for lazy discovery.

### `/debugging-inputs`

Diagnostic reference for the "silent" input failures that surface downstream: chain-ID mismatches, ligand residue numbering collisions, RMSD pairing failures, AF3 JSON schema errors, and structure parsing edge cases.

### `/rosetta-params`

Everything needed to generate and debug Rosetta small-molecule `.params` files: molfile → params conversion, rotamer libraries, partial-charge assignment, and common failure modes at the Rosetta/PyRosetta boundary.

### `/order-collection`

Packages a completed campaign into a handoff bundle: a compressed archive of prediction structures + confidence data, a reference CSV of metrics, and a human-readable summary for the wet lab.

### `/visualize`

Sends structures (CIF, PDB, pickle) and PyMOL commands to a running PyMOL session over XML-RPC. Supports cartoon styling, selections, and coloring. Useful for on-the-fly inspection of design outputs without leaving the CLI.

### `/slurm-interactive`

Holds a persistent **interactive** allocation on a GPU or CPU node using the cluster’s allocation command inside a `tmux` session, so the agent runs commands live in the *same* shell on the node (one allocation, one job step — not a job per command). Useful for fast iteration on a sequence of blocks, on-node debugging, and quick probes before scaling up. **Iterating → interactive.**

### `/slurm-background`

Submits **detached** Slurm work for pilots, production, and campaigns using the Chaperone `ExecutionEngine` or standalone `sbatch`. Long-running drivers use the monitor's `launch_and_wait` tool. After submitting a standalone Slurm job, call the monitor's `wait` tool directly once with its job ID; the engine manages its own inner jobs.

The monitor is exposed as `campaign-monitor` under Harbor or by the standalone `chaperone monitor-mcp` server. Follow the source guide's launch and monitoring procedure rather than detaching drivers with shell backgrounding or adding polling loops. This workflow requires those monitor tools to be available to the agent.

### `/notebook-editing`

Conventions for editing `.ipynb` files as the source of truth (no parallel `.md`), and for rerunning notebooks via `papermill` when changes warrant a fresh execution.

---

## Adding a new skill

Skills live in `.agents/skills/<name>/SKILL.md` with YAML frontmatter:

```yaml
---
name: my-skill
description: Describe the task this skill handles and when an agent should use it.
---
```

Keep the description **action-oriented** and include concrete situations where the guide applies. Document required tools, environment assumptions, validation steps, and expected outputs in the body. Add application-specific metadata only when needed, and explain any dependencies on a particular agent runtime.

After adding or changing a skill, update `.agents/skills/INDEX.md` and this page together. Keep the guide as the source of truth for detailed commands; this page should summarize its purpose and prerequisites.
