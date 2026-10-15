# AME — Active-site Motif Evaluation Benchmark

Measures how well AF3 recovers catalytic motif and ligand geometry after RFD3
backbone generation and LigandMPNN sequence design.

## Benchmark inputs

- **AME-1:** 41 MCSA active sites in [`files/ame-1.json`](files/ame-1.json),
  with structures in `files/mcsa_41/`.
- **AME-2:** four tasks in [`files/ame-2.json`](files/ame-2.json), with
  structures in `files/ame2_inputs/`:

| Task | Ligand |
|---|---|
| `ornithine_decarboxylase` | `ORN` |
| `dera_aldolase` | `HPD` |
| `glycosidase` | `L:1` |
| `phosphotriesterase` | `YYE` |

[`pipeline.py`](pipeline.py) runs both suites, then combines their generated
backbones for shared evaluation. [`pipeline_full.py`](pipeline_full.py) runs
AME-1 only, requesting one generation batch per job. Both default to 100
backbones per input and eight designed sequences per backbone.

The phosphotriesterase fixture retains its molecular coordinates and atom IDs.
Its virtual `ORI` marker has normalized PDB element/charge fields and follows
the molecular atoms so readers accept the serial-number ordering.

## Pipeline

```text
rfd3 → rotamer_metrics → ligand_mpnn → af3 (with ligands) → rmsd
```

- **RFD3:** generates backbones around the supplied catalytic motifs.
- **Rotamer metrics:** scores motif sidechains using the Dunbrack library.
- **LigandMPNN:** designs sequences with ligand context and fixed motif identities.
- **AF3:** predicts the designed sequences with their ligand context retained.
- **RMSD:** compares backbone, selected motif atoms, and ligand atoms after
  backbone alignment, pairing selected atoms by name.

CPU metrics and GPU models run as separate engine steps. These scripts report
metrics for the generated cohort; they do not apply a scientific acceptance
filter.

## Run a bounded check first

Configure Foundry/RFD3, LigandMPNN, AF3 and their model weights, plus
`DUNBRACK_LIBRARY_COPY` pointing to the Dunbrack rotamer-library directory.
AF3 weights require upstream access. Custom ligand chemistry must be compatible
with AF3's CCD or supplied through an appropriate SMILES/template configuration.

From the repository root, with the Chaperone environment active:

```bash
python examples/benchmarks/ame/pipeline.py --test --rundir="$PWD/runs/ame-smoke"
python examples/benchmarks/ame/pipeline_full.py --test --rundir="$PWD/runs/ame1-smoke"
```

Append `--local` when running inside an allocated GPU node. Otherwise configure
Slurm for your cluster and let the engine submit the steps.

`--test` selects `M0024_1nzy` from AME-1 and, for `pipeline.py`, `dera_aldolase`
from AME-2. It requests one backbone and one sequence per selected input, with
one AF3 sample per prediction input. It does not establish that the entire
benchmark cohort has completed inference or meets a quality threshold.

## Reported metrics

- `rmsd.bkbn_align.bkbn_rmsd`: backbone recovery.
- `rmsd.bkbn_align.motif_allatom_rmsd`: recovery of the selected catalytic atoms.
- `rmsd.bkbn_align.ligand_rmsd`: ligand placement after backbone alignment.

The ligand selector is `SELECTIONS.LIGANDS`; its serialized value is `"ligand"`.
Motif RMSD covers the atoms selected by each input specification, which may
include more than sidechain tip atoms. Ligand-containing AF3 predictions are
not an apo-only folding benchmark. Interpret metrics alongside atom coverage
and the input chemistry.

## Adapt an input

1. Prepare a theozyme PDB/CIF containing your catalytic residues and ligand.
2. Add a specification following [`files/ame-2.json`](files/ame-2.json).
3. Set `select_fixed_atoms`, the motif selection, and the desired length range.
4. Make its `input` path relative to the JSON file and verify the residue and
   atom names exist in that structure.
5. Run a bounded check with your task selected before increasing sample counts.
