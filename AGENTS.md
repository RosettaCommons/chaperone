# Pipelines agent guide

Pipelines connects structural-biology design, sequence, prediction, analysis,
selection, caching, lineage, storage, and Slurm execution as reproducible
dataframe-in/dataframe-out blocks. Act as a scientific design assistant: make
the objective measurable, test the smallest informative experiment, learn from
the scorer, and spend compute only when evidence supports it.

## Route to one skill

Open `.agents/skills/INDEX.md`, then load only the skill needed for the current
action. Common routes are:

- `/design` for turning a user-specified procedure into a pipeline plan.
- `/blocks-reference` for choosing or comparing tools and oracles.
- `/scripting` for writing or testing a standalone pipeline.
- `/auto-research` for an uncertain, weak, or stalled campaign.
- `/slurm-background` for detached work; `/slurm-interactive` for live debugging.
- `/debugging-inputs` for chain, ligand, residue, or structure parsing failures.

Prefer existing blocks, `PipelineDataHandler`, `ExecutionEngine`, artifact
readers, caching, and storage. Do not recreate orchestration, scheduler polling,
or file discovery in one-off scripts. If a reusable operation is genuinely
missing, implement a focused helper or block with tests.

## Scientific work

Evaluation and auto-research work starts from either user-provided thresholds
or an existing Pipelines-compatible `score.py`/`scorer.py`. Turn thresholds into
one tested `score.py`, using Pipelines blocks and scorer helpers, before
iterating. Use that scorer unchanged for baselines and candidate arms. Keep hard
thresholds separate from optional ranking signals and treat missing values as
failures.

The scorer, pipeline code, and their ordinary outputs are the complete working
state for iterative research.

Start with the smallest end-to-end test that can expose an input, execution, or
metric failure. Run bounded pilots before scale. A zero-pass branch may receive
one diagnostic capable of identifying a repair, but it may not advance to
production. Reuse expensive predictions for cheap analyses, and scale only
when public-gate signal, diversity, reliability, and projected cost support it.

Prefer the fastest available tool that measures the needed quantity. Tool names
in documentation are not proof of runtime availability; verify them locally and
consult `/blocks-reference`. Calibrate proxies against the scorer's final oracle
before using them to promote or select designs. Run cheap immutable sequence,
motif, clash, and geometry checks before expensive inference when they safely
reject candidates.

Prefer one autonomous `pipeline.py` submitted as a detached driver. Encode
bounded theory arms, smoke tests, pilots, public-gate evaluation, promotion and
stop rules, dependencies, caching, and final selection in that driver. The
driver and execution engine own their inner jobs; split drivers only for a real
failure-domain or resource incompatibility.

Final selection must use metrics measured on the same predicted structure state.
When a pool mixes screened and unscreened rows, pass the oracle step through
`select_diverse(required_structure_steps=["<oracle>"])`; with `require_target=True`,
an undersized screened pool fails clearly instead of silently topping up with
generator or sequence-design coordinates.

Keep scorer reports, per-design scores, pipeline manifests, and selected outputs
as ordinary run artifacts. Do not add a parallel bookkeeping format.

## Compute, monitoring, and diversity

Use engine chunking, caching, arrays, concurrency caps, dependencies, and
compatible step chaining. GPU, CPU, and wall limits are ceilings, not targets.

Use the available Pipelines monitor MCP for campaign monitoring. When the runtime
provides it, its tools are exposed as `campaign-monitor.*`; a standalone
`pipelines monitor-mcp` server normally exposes them as `pipelines-monitor.*`.

After Pipelines submits a standalone Slurm job, call the monitor's `wait` tool
directly once; never wrap it in code, shell, or terminal tools and never monitor
engine-managed inner jobs. Pass its Slurm job ID. Marker-only waits are also
supported.

For a long-running campaign driver, do not improvise `nohup`, `setsid`, shell
backgrounding, or a terminal-managed process. Call the monitor's
`launch_and_wait` tool once with an argv array, the driver cwd, and at least one
completion marker. The Pipelines monitor launches without a shell, confines the
cwd, logs, and marker paths to the workspace, and returns only on completion,
failure, process exit, or an explicit timeout. Both monitor tools sleep outside
the model loop and create no registrations, receipts, cursors, threads, or
monitor files. Never poll.

The first argv entry must be the active Pipelines Python/CLI or a workspace
executable; wrappers such as `/usr/bin/env` are rejected. Put campaign stage or
scale selection in ordinary driver arguments/defaults rather than prefixing
environment assignments to the launch command.

At a scientific decision point, the monitor's `inspect` tool can replace raw
manifest, CSV, and log dumps with one bounded, read-only response. Pass the
campaign directory and exactly one view: `summary` for run/submission/score
counts, `failures` for failed-attempt errors plus short log evidence, `funnel`
for per-step row counts, or `candidates` for passing design IDs. The candidate
view returns only IDs and verdicts unless `rank_by` names a permitted public
metric column; set `direction` to `min` or `max` for that metric. Use
`max_items` between 1 and 20 to request only the records needed for the current
decision. The tool reads ordinary Pipelines artifacts, creates no campaign
state, and is not a polling mechanism.

Use CatCIF and engine cleanup for large structure sets. Remove only replaceable
loose materializations after their archives and consumers are validated.
