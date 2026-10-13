# Claude Skills

Chaperone ships with a collection of [Claude Code](https://claude.com/claude-code) **skills** — domain-specific guides and workflows that Claude loads on demand. Skills live under `.agents/skills/` in the repo and are invoked either automatically (when the user's request matches the skill description) or explicitly via `/<skill-name>`.

Each skill provides focused context for a particular task so Claude can act with the right conventions, commands, and guardrails without flooding its main context window.

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
| [`/order-designs`](#order-designs) | Generate DNA eblock orders from design PDBs | "order these designs", "generate eblocks" |
| [`/order-collection`](#order-collection) | Package a campaign into a handoff bundle | "collect orders", "bundle designs", "prepare handoff" |
| [`/visualize`](#visualize) | Send structures and commands to PyMOL via XML-RPC | "visualize", "show in PyMOL", "view structure" |
| [`/slurm-interactive`](#slurm-interactive) | Persistent interactive SLURM session (GPU/CPU) via qlogin | "get me a gpu", "allocate a gpu/cpu", "gpu session", "qlogin" |
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

End-to-end guide for adding a new block from a tool's repo or documentation URL. Covers container definition (HPCCM), block code layout, `ScriptSpec`, input/output helpers, and registering the block for lazy discovery.

### `/debugging-inputs`

Diagnostic reference for the "silent" input failures that surface downstream: chain-ID mismatches, ligand residue numbering collisions, RMSD pairing failures, AF3 JSON schema errors, and structure parsing edge cases.

### `/rosetta-params`

Everything needed to generate and debug Rosetta small-molecule `.params` files: molfile → params conversion, rotamer libraries, partial-charge assignment, and common failure modes at the Rosetta/PyRosetta boundary.

### `/order-designs`

Generates DNA eblock orders from design PDBs using the JohnBercow Golden Gate cloning pipeline on IPD digs. Handles codon optimization, BsaI site removal, and eblock CSV output.

### `/order-collection`

Packages a completed campaign into a handoff bundle: a compressed archive of prediction structures + confidence data, a reference CSV of metrics, and a human-readable summary for the wet lab.

### `/visualize`

Sends structures (CIF, PDB, pickle) and PyMOL commands to a running PyMOL session over XML-RPC. Supports cartoon styling, selections, and coloring. Useful for on-the-fly inspection of design outputs without leaving the CLI.

### `/slurm-interactive`

Holds a persistent **interactive** allocation on a GPU or CPU node via `qlogin` inside a `tmux` session, so Claude runs commands live in the *same* shell on the node (one allocation, one job step — not a job per command). Useful for fast iteration on a sequence of blocks, on-node debugging, and quick probes before scaling up. **Iterating → interactive.**

### `/slurm-background`

Submits **detached** SLURM work that survives disconnect — for pilots, production, and campaigns. Covers the Chaperone `ExecutionEngine` (`job_args`, `submit`/`gather`), raw `sbatch`, detaching a long driver with `nohup`/`tmux`, and how to monitor/cancel. **Scaling → background.**

### `/notebook-editing`

Conventions for editing `.ipynb` files as the source of truth (no parallel `.md`), and for rerunning notebooks via `papermill` when changes warrant a fresh execution.

---

## Adding a new skill

Skills live in `.agents/skills/<name>/SKILL.md` with YAML frontmatter:

```yaml
---
name: my-skill
description: One-line description Claude uses to decide when to load this skill.
allowed-tools: ["Read", "Grep", "Glob", "Bash"]
---
```

Keep the description **action-oriented** and list concrete trigger phrases — Claude matches user intent against this field. After adding a skill, update the tables above so users can discover it.
