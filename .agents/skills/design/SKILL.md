---
name: design
description: Comprehensive planning agent for converting any input, user specification or otherwise into clean, tested, reproducible and pipelines code. Use this skill whenever a user mentions creating a pipeline, or designing a protein. For example; converting an existing notebook into a comprehensive pipeline plan, for interactively asking questions about an existing design specification, or for "How can I design this protein?"
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write", "Task", "Edit"]
---

# Pipeline Builder Skill

Comprehensive planning agent for converting any input, user specification or otherwise into clean, tested, reproducible and pipelines code.

This skill is divided into two stages:
- Planning phase: interactively asking the user relevant questions about their task
- Testing and implementing phase: robustly testing.

## Phase 1: Building the plan

Extends the conventional `planning_agent` workflow. Below are details for:
- Creating a plan from notebooks
- Creating a plan from a design specification or general user input "I need to design an enzyme...".

Keep an implementation plan when it helps the user reproduce a normal design
project. A task that specifies its own deliverable does not need a packaged campaign plan.

### Before creating the plan

You should consider **to what extent to implement, test, experiment and scale the pipeline before finishing**. There are exactly pipeline 4 levels to this with increasing computational cost and robustness:
1. Implement code - *no integration testing, but still check file / executables availability, run smaller tests.*
2. Integration test plan - *check if local gpu is available; if available, strongly recommended this option above no testing, if it's not available then you will grab a gpu once needed (not submit to cloud/cluster)*. What checks should be run at this scale to ensure the outputs and metrics are correct? 
3. Run pilot experiments. Get preliminary results (~100-1000 backbone scale), sweeping necessary different configurations. If selected, you should include a discussion of which parameters to test out. Beyond the integration test, local gpus should generally not be used, in cases where it's ambiguous (~100 designs total for a pilot), ask for clarification whether the local or slurm gpu should be used.
4. Run campaign-scale experiments. Produce fully designed and evaluated *orderable* computational designs.

Each point should **always** follow the other; never run a campaign-scale run without a pilot run (etc.)

You MUST use the `AskUserQuestion` tool to gather the following details before creating *any* plan. Ask all questions in a single `AskUserQuestion` call with multiple questions. If no user is available to answer (an autonomous run, or the task prompt already fixes these choices), do not ask: use the task's own instructions, otherwise the recommended option for each question.

**Question 1 — Output location** (header: "Location"):
- `examples/[pipeline_name]` — Standard examples directory (Recommended)
- `/net/scratch/$USER` - Standard for large runs
Options should also include "Other" (automatically provided by AskUserQuestion).

**Question 2 — Implementation depth** (header: "Depth"):
- `Code only` — Implement code, check file/executable availability, no GPU testing
- `Integration test` — Code + end-to-end test on GPU (Recommended if GPU available)
- `Pilot experiments` — Code + test + ~100-1000 backbone sweep with parameter exploration
- `Campaign scale` — Full production run with orderable designs

Each level always includes all previous levels; never run campaign without pilot (etc.)

**Question 3 — Output format** (header: "Format"):
- `Hybrid` — pipeline.py + analysis.ipynb (Recommended)
- `Monolithic` — Single pipeline.py with inline analysis
- `Multi-script` — Separate scripts per phase

The standard recommended Hybrid setup:
```
[pipeline_name]/
├── campaign-plan.md              # Optional design plan for a normal project
├── files/{af3_ptm_template.json, rfd3_input.json, ligand.cif}  # Source files
├── rundir/{final_designs.csv}    # Final aggregated outputs.
├── rundir/                       # Pilot/integration test outputs
├── pipeline.py                   # Launch and compilation script for full pipeline
└── analysis.ipynb                # Notebook for compiling everything
```

**Question 4 — Conversion exactness** (header: "Conversion", only ask if a notebook/existing code is being converted):
- `Modernize` — Port to standard pipelines blocks and patterns (Recommended)
- `Preserve` — Keep custom logic as close to original as possible

## Recommendations for 1. - Implementing code.

**Notebook conversion recommendations:**
- Use the `/notebook-editing` skill to effectively read the notebooks (using `jupytext`)
- Use Bash(`pipelines list-blocks`) to see available blocks you can use
- Document which parameters from the messy notebook map to block parameters
- Create an analysis table to map each cell or cell group to it's function (see below)

**Create an analysis of the tools required/available:**
```markdown
## Overview
[Brief description of the notebook's purpose]

## Block Availability Analysis
| Functionality | Block | Status | Action Needed |
|--------------|-------|--------|---------------|
| RFD3 diffusion | rfd3 | Available | Look up params |
| MPNN sequence | mpnn | Available | Look up params |
| Custom oracle | ? | NOT AVAILABLE | CREATE NEW BLOCK |
| ...           | ...  | ...       | ...           |

**Subagent use:** You should specify in your plan whether and where to spin off subagents. 
If the user inputs / notebook has multiple pieces of custom software to convert into pipelines blocks and test, you should plan to spin off a subagent for each block.
For example, if the user's pipeline has / requires "RPXDock" and "BoltzGen" - both of which have complicated input and output streams, and will require testing, you can plan to implement both in the background in parallel with 2 subagents working on the output pipeline.
This will speed things up significantly.

## Recommendations for 2. - Running the integration test.

If an `examples/` directory exists, see its worked pipelines for the standard layout. Use `/slurm-interactive` to get a temporary gpu for testing. Only allocate the gpu once you need to start testing on gpu.
For integration tests, you should report to the user where the outputs are. Preferably they exist under the folder with the other code. E.g. `examples/[pipeline_name]/rundir/{rfd3,mpnn,etc.}`.

## Recommendations for 3. - Running pilot-scale tests.

This is the exploratory phase of the design phase and is intending for testing / probing potentially better parameters / steps for the pipeline.
There are different levels of doing this:
1. Input-level sweeps: what configurations of the inputs are best for this task?
2. Block-level sweeps: sweeping. Can be useful for squeezing out a bit more from the tools.

See the `/auto-research` skill if the pilots stall or the scorer shows no signal.

## Recommendations for 4. - Running at campaign scale.

In general, you should **never** run the campaign scale test without explicitly asking to do so. Typically the previous steps (1,2 and 3) should provide you with sufficient confidence in the pipeline to run at scale, giving you enough to propose a plan for the campaign-scale run.

You should use the `/order-designs` skill after the final designs have been made, to convert the outputs into directly orderable genes. This skill will also detail any additional necessary procedures to appropriately log the outputs and summary of the pipeline.
