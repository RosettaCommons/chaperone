# Biotin binder

## Objective and inputs

Generate a protein pocket around the bundled linker-bearing biotin analogue,
sequence it, and evaluate the holo and apo structures separately. The input
`files/biotin_all.pdb` and `files/biotin.json` retain the original ligand atom
names and hydrogen-bond, burial, and linker-exposure conditioning.

## Pipeline

```text
RFD3 → contact/RASA filters → LigandMPNN → soft ligand DDG triage
     → AF3 holo → ligand remapping → contact/H-bond/RASA filters
     → AF3 apo → backbone RMSD/pLDDT filters
```

RFD3 defaults to two backbones; MPNN makes two sequences per backbone.
The original stringent gates may eliminate every candidate in a smoke run.
An empty results table is a valid scientific outcome, but does not verify
execution of the downstream stages that were skipped.

## Run

Configure the Foundry, LigandMPNN, and AlphaFold 3 containers and their model
weights. AlphaFold 3 requires separately licensed parameters. The
`ligand_ddg_soft` block also requires PyRosetta in its worker environment.
Install HBPLUS and set `HBPLUS_COPY` to its executable for hydrogen-bond
fixing during MPNN and the final hydrogen-bond metric. Set
`MOLFILE_TO_PARAMS_SCRIPT` to Rosetta’s `molfile_to_params.py` (with its
`rosetta_py` helpers) for on-demand ligand parameter generation.

```bash
python examples/biotin-binder/pipeline.py --test --rundir=/absolute/path/biotin-smoke
```

Use `--local --use_local_gpus` inside a GPU allocation. Configure site resource
mapping rather than embedding a particular cluster's partition in the recipe.

## Gates and interpretation

The source recipe requires zero ligand clashes, non-loop fraction ≥ 0.2,
minimum ligand distance ≥ 2 Å, and aggregate core contact score ≥ 110.
At least one linker atom C14/C15/C16 must have RASA ≥ 0.1. Core burial is an
RFD3 conditioning request; the aggregate contact score does **not** prove
that every core atom is buried. DDG triage retains the lowest 10% per parent.

Holo filtering requires interaction PAE ≤ 1.2, at least three reported hydrogen
bonds, and contact score ≥ 120. Apo filtering requires backbone RMSD ≤ 2 Å and
pLDDT ≥ 85. These computational criteria do not establish binding or affinity.
The DDG value is a model-dependent score, not a measured binding free energy.

## Outputs and validation

`results.csv`, engine manifests, intermediate structures, and per-block metrics
are written beneath `--rundir`. Ligand remapping makes predicted ligand names
consistent before atom-specific metrics run. CPU configuration tests cover the
ported recipe; bounded GPU execution is described below.

Adapted from the original pipelines `examples/biotin-binder` at revision
`95ef0b6fd87eb27e1e73d2cfaa1482a39a0a132f`.

MPNN returns side-chain-packed structures, with one packing sample per designed
sequence; refolding consumes those sequences and structure context.

## GPU execution check

A bounded CoreWeave H200 diagnostic completed RFD3, one side-chain-packed
LigandMPNN sequence, holo AF3, ligand remapping, hydrogen-bond/contact/RASA/RMSD
metrics, and apo AF3/RMSD. The same packed design also completed five PyRosetta
DDG replicates on CPU, with a finite trimmed-mean score. Generated and predicted
coordinates were finite. This used native AtomWorks 2 RFD3, the public MPNN
container, and an existing AF3 image with Chaperone's public inference runner;
it was not a fresh container build test.

The diagnostic deliberately evaluated later stages even when a quality gate
would stop the normal recipe. This candidate failed the holo gates: two hydrogen
bonds (required three) and minimum interaction PAE 1.69 (maximum 1.2). Execution
success therefore does not imply a successful biotin binder. Holo backbone RMSD
was 1.09 Å; apo-versus-holo backbone RMSD was 0.40 Å.

Packed-output checks also verified that the sampled FASTA matches the packed
protein sequence and all 21 biotin heavy atoms retain their identities and
coordinates. After restoring the five fixed protein residues during compilation,
all 39 observed fixed heavy-atom coordinates match the staged input exactly.
