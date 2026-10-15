# DNA binder

## Objective and inputs

Design a protein against the bundled full-atom double-stranded DNA duplex
CGCGAATTCGCG, based on PDB 1BNA. The target PDB uses chains A and B, each numbered
from 1. The recipe validates sequence, numbering, and hydrogen-bond conditioning
atoms before generating a run-local RFD3 input JSON.

## Pipeline

```text
RFD3 → geometry filter → LigandMPNN → AF3 → backbone RMSD/filter
     → LigandMPNN → AF3 → tighter backbone RMSD/filter
```

The protein has 120–150 residues. Both DNA strands are fixed context; the
sense strand supplies major-groove donor/acceptor conditioning. The origin is
the fixed duplex center of mass. The predecessor marked the whole duplex as
hotspots and inferred an outward direction from those same atoms; this gives a
zero-length direction and nonfinite coordinates in both the original Foundry
implementation and PR #397. Using the supported COM origin repairs this
geometry error while keeping the hotspot and hydrogen-bond selections. RFD3 generates
two backbones and each MPNN round makes two sequences by default.

## Run

Install the public RFD3 checkpoint and configure Foundry, LigandMPNN and
AlphaFold 3 containers/weights. The public RFD3 model supports nucleic-acid
binder generation; a private DNA checkpoint is not required. AlphaFold 3
parameters require their own license/access.

```bash
python examples/dna-binder/pipeline.py --test --rundir=/absolute/path/dna-smoke
```

Optional `--dna_checkpoint=/absolute/path/model.ckpt` overrides the installed
RFD3 checkpoint. To change the target, supply both `--dna_target` and
`--dna_sequence`; this script validates existing coordinates and does not
construct a new duplex. Use `--local --use_local_gpus` inside a GPU allocation.

## Gates and interpretation

The first filter requires zero chain breaks and non-loop fraction ≥ 0.2.
Round one requires ipTM ≥ 0.6, minimum interaction PAE ≤ 2.5 and backbone RMSD
≤ 8 Å. Round two tightens these to 0.7, 1.25 and 3 Å. These are inherited
computational gates, not evidence of sequence-specific DNA binding. Specificity
against competing DNA sequences is not evaluated by this example.
`min_pae_interaction` is the minimum over all chain pairs, so it can describe
the DNA–DNA interface rather than the protein–DNA interface. These inherited
gates therefore do not establish binder-specific interface confidence; inspect
the chain-pair confidence matrices before promoting a design.

## Outputs and validation

`results.csv` and the generated `inputs/dna_binder.json` are under the run
directory, together with engine manifests and intermediate structures. Empty
survivor tables are allowed; a smoke run that stops at a gate does not verify
later inference stages. CPU tests check target preparation and block settings;
Bounded GPU execution is described below.

The public Foundry `models/rfd3/docs/examples/na_binder_design.md` documents
using `rfd3_latest.ckpt` for dsDNA binder generation. This port adapts original
pipelines `examples/dna-binder` at revision
`95ef0b6fd87eb27e1e73d2cfaa1482a39a0a132f`.

MPNN returns side-chain-packed structures, with one packing sample per designed
sequence; refolding consumes those sequences and structure context.

## GPU execution check

After the COM-origin repair, a CoreWeave H200 diagnostic completed one RFD3
backbone (123 protein residues and two 12-nucleotide DNA strands), packed
LigandMPNN, AF3/RMSD, and a second packed MPNN → AF3/RMSD round. DNA context was
preserved through packing, and generated/predicted coordinates and RMSDs were
finite. This used native AtomWorks 2 RFD3, the public MPNN image, and an existing
AF3 image with Chaperone's public runner; fresh image builds were not tested.

Quality filtering was bypassed only to exercise both rounds. The candidate
failed the first-round ipTM gate (0.09 versus 0.6 required); the second diagnostic
round also failed its confidence gates. Backbone RMSDs were 0.59 Å and 0.84 Å.
These results verify execution, not DNA binding or sequence specificity.

Both packed MPNN rounds matched their sampled protein FASTA sequences and
preserved all 24 fixed DNA residues, including atom identities and coordinates.
The checks covered 486 nonprotein heavy atoms in round one and 494 after AF3
preparation in round two; displacement from each round's own input was zero.
