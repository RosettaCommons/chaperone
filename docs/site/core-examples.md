# Core Examples

The worked scripts live in the repository's
[`examples/` folder](https://github.com/RosettaCommons/chaperone/tree/main/examples).
The links below open their source and run instructions. They demonstrate pipeline
composition; inclusion here does not certify scientific performance or successful
execution on every supported cluster.

| Example | What it demonstrates or measures |
|---|---|
| [FGFR2 binder campaign](https://github.com/RosettaCommons/chaperone/tree/main/examples/fgfr2-binder) | Backbone generation, sequence design, AF3 refolding, and interface confidence ranking with ipSAE for FGFR2. |
| [PPI benchmark](https://github.com/RosettaCommons/chaperone/tree/main/examples/benchmarks/ppi) | Two inverse-folding backends in an RFD3 → MPNN → RF3 workflow for the insulin receptor; compare refolding RMSD and interface confidence under matched inputs and sampling. |
| [DISCO benchmark](https://github.com/RosettaCommons/chaperone/tree/main/examples/benchmarks/disco) | Heme B ligand-conditioned co-design followed by AF3; measures backbone and ligand pose recovery after refolding. |
| [Active-site motif benchmark](https://github.com/RosettaCommons/chaperone/tree/main/examples/benchmarks/ame) | Enzyme design around catalytic motifs; measures recovery of motif sidechain geometry after sequence-only AF3 folding. |
| [Prediction validation cohort](validation.md) | Structure predictors evaluated against reference PDB assemblies with lDDT and PEPPR metrics, with explicit target and chain coverage. |
| [CatCIF storage example](https://github.com/RosettaCommons/chaperone/tree/main/examples/tests/catcif-storage) | Archiving, reading, and materializing structures; an engineering demonstration, not a scientific benchmark. |

## What “benchmark” means here

A benchmark script specifies a task, an input set, and metrics to compare a
workflow or model variant. It is not itself a results report. Design benchmarks
measure computational proxies such as refolding consistency and motif recovery;
they do not measure experimental binding or catalytic activity. The prediction
cohort measures agreement with reference structures. Record model/checkpoint
versions, seeds, sample counts, failures, and compute cost alongside scores when
comparing runs.

The PPI scripts make a comparison between FoundryMPNN and the `mpnn` block (public
LigandMPNN) possible, but their presence does not establish matching sequence
distributions or equivalent scientific performance. A controlled comparison must
reuse the same generated backbones and scoring protocol and include enough
samples to estimate uncertainty.

## Validation status and first runs

See the dated [release validation record](release-validation.md) for completed
local checks, GPU smoke evidence, pending runs, and reproducible commands.

These GPU campaigns are not run in routine CI. No release-wide, fresh-cluster
end-to-end validation is asserted for this list. Unit tests and small integration
checks cover different questions from full scientific benchmarking. Where an
example has run evidence, consult its README and retained results for the exact
hardware, inputs, and versions; otherwise treat it as requiring validation.

Install the relevant containers and weights, follow the example's preparation
steps, and start with `--test` or a small bounded input. Inspect outputs and
missing metrics before a pilot or full campaign. A failed or zero-pass pilot
needs a diagnosis, not a larger sample count. See the [simple example](simple.md)
and [Slurm site settings](syntax.md#slurm-site-configuration).

## Why contribute an example?

An example records a reusable composition of existing blocks, including its
inputs, settings, and acceptance criteria. It helps both people and coding agents
find a relevant starting point. Agents can read examples, but adding one is not
required for an agent to discover blocks or create a new workflow; the skills
index and block reference serve those purposes.

Contribute a focused script when it adds a distinct task or useful composition.
Include a README with prerequisites, preparation commands, a small test command,
expected output fields, scientific metrics, and an honest record of what was
actually tested. Keep large generated outputs and site credentials out of the
repository, and put generally reusable operations into blocks rather than
copying orchestration into each example.
