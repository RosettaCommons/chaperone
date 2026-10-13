---
name: auto-research
description: Iteratively improve a weak, uncertain, or stalled Chaperone design workflow using user-provided thresholds or an existing Chaperone-compatible scorer.
---

# Auto research

Maximize diverse scorer-passing designs per unit time and compute. Repository
instructions in `AGENTS.md` remain authoritative.

## Start from one scorer

Accept either:

- user-provided thresholds or acceptance criteria; or
- an existing `score.py`/`scorer.py` that uses Chaperone for downstream scoring.

When the user supplies thresholds, translate them once into a tested `score.py`.
Use Chaperone metric/oracle blocks and filter expressions. The scorer must retain every candidate, treat missing or failed metrics as
failures, report each threshold and the joint pass result, and write per-design
scores. If the user supplies a scorer, use its CLI and outputs directly rather
than wrapping it in another format.

`score.py`, `pipeline.py`, and their normal run outputs are the complete working
state.

If a requested threshold cannot be measured with the available inputs and
blocks, resolve that measurement gap before compute. Never silently replace it
with a proxy. Assume at most five concurrent GPUs unless the user gives another
limit. Read `references/design-priors.md` only when its protein-design priors are
relevant to the current bottleneck.

## Diagnose and choose an experiment

Run the smallest end-to-end baseline through the scorer. Locate the dominant
loss or uncertainty using attempted, completed, failed, per-threshold pass
counts, joint passers, diversity, and cost. Distinguish an engineering failure
from evidence against a scientific mechanism.

Choose a bounded experiment with a hypothesis, parameter change, affected
thresholds, falsifier, candidate count, seeds, resource estimate, and a matched
control when attribution needs one. Keep this in `pipeline.py` or a short nearby
comment; do not serialize it into a separate record.

For comparable numerical settings, batch a few plausible values at the same
small fidelity. Avoid blind Cartesian sweeps. Change one conceptual factor when
attribution matters; use a compact factorial only when the interaction itself
is the hypothesis.

## Execute and improve

Prefer one autonomous `pipeline.py` containing the bounded arms, scorer calls,
promotion rules, caching, dependencies, and final selection. Use the appropriate
Slurm skill for a detached run and make one stateless wait call for the driver,
not its engine-managed inner jobs.

Score every baseline and arm with the same `score.py`. Compare joint passers,
per-threshold losses, diversity, execution failures, and actual cost. Promote
only scorer-passing arms whose evidence supports more compute. For comparable
arms, keep roughly the best fraction and increase fidelity; a zero-pass arm may
receive one diagnostic capable of identifying a repair but must not scale.

After each round, promote, diagnose, test a different mechanism, pivot, or stop.
Two rounds without improvement or useful information require a structural pivot
or current-literature review, not another nearby mutation. Replicate a surprising
result when uncertainty could reverse promotion.

End when the selected outputs pass the scorer, no feasible experiment can change
the conclusion, or the next informative step exceeds the compute limit. Preserve
the scorer outputs and pipeline artifacts, including negative results; do not
manufacture an additional reporting layer.
