# AME — Active-site Motif Evaluation Benchmark

Evaluates de novo enzyme design by measuring how well AF3 recovers catalytic
motif geometry after RFD3 backbone generation and LigandMPNN sequence design.

## Benchmark tasks

| Task | Ligand | Motif residues | Description |
|------|--------|----------------|-------------|
| glycosidase | L:1 (glycoside) | 6 residues | Glycoside hydrolase active site |
| dera_aldolase | HPD | 6 residues | DERA aldolase active site |

## Pipeline

```
rfd3 -> filter -> ligand_mpnn -> af3 (apo) -> rmsd (tip-atom)
```

- **RFD3**: generates backbones around the theozyme motif with fixed catalytic atoms
- **Filter**: removes designs with chain breaks, ligand clashes, or poor loop content
- **LigandMPNN**: sequence design with ligand context and fixed motif sequences
- **AF3 (apo)**: structure prediction with ligands removed for clean motif evaluation
- **RMSD**: backbone-aligned tip-atom RMSD — primary success metric

## Key metric

`rmsd.bkbn_align.motif_allatom_rmsd` — measures how faithfully AF3 reproduces the
catalytic residue sidechain geometry after folding only from sequence.

## Extending to ligand RMSD

To also evaluate ligand placement fidelity, add a second AF3 branch that retains
the ligand and compute `SELECTIONS.LIGANDS` RMSD. This requires specifying the
ligand SMILES or CCD code for each task in the AF3 configuration.

## Adapting for your own enzyme

1. Prepare a theozyme PDB with your catalytic motif and ligand
2. Create an RFD3 input JSON following the format in `files/ame_tasks.json`
3. Adjust `select_fixed_atoms` to specify which sidechain atoms define your motif
4. Update `length` range for your target protein size
