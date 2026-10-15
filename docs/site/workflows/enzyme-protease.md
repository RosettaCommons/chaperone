# Zinc protease design

This recipe scaffolds catalytic motifs around a zinc-containing site and a ten-residue substrate peptide, then designs enzyme sequences and predicts apo and complex structures.

```text
RFD3 → geometry filter → LigandMPNN → RF3 refinement
                                    ├→ select enzyme A → AF3 apo → filter
                                    └→ restore complex → AF3 complex → filter
```

## Inputs and prerequisites

The [source recipe](https://github.com/RosettaCommons/chaperone/tree/main/examples/enzyme-protease) provides workflow code and RFD3 configuration templates. Custom structures and chemistry templates are **not bundled** pending redistribution permission. Supply an authorized input directory containing `protease.json` and `GPRVVERHQS78.pdb`; copy the RFD3 JSON from the example and resolve its structure paths there. The configuration expects enzyme chain A, substrate chain B and a `ZnO` ligand. Adapt the contigs, residue selections and chemistry together for a different input.

Install Foundry, LigandMPNN, and AlphaFold 3 runtimes using the [container guide](../containers.md), including the RFD3 and RF3 checkpoints and licensed AF3 weights. Site configuration supplies the scheduler and GPU requests.

The public port uses **regular RF3**, with substrate chain B selected as a template and ground-truth conformer, 50 diffusion steps, and five recycles. This workflow uses the public Foundry container and checkpoint.

## Run

```bash
python examples/enzyme-protease/pipeline.py --input_dir=/path/to/authorized/inputs --test --rundir=/absolute/path/protease-smoke
```

Use `--local --use_local_gpus` inside an allocated GPU session. `--test` requests one backbone per contig specification. Defaults request two backbones per specification, two side-chain-packed MPNN sequences per backbone (one pack each), one RF3 refinement cycle with one diffusion sample, and one sample per AF3 branch.

## Chemistry, branching, and selection

LigandMPNN designs chain A, retaining motif sequences and omitting new cysteine/histidine choices elsewhere. The apo branch explicitly crops to enzyme chain A before prediction; removing ligands alone would leave the peptide present. RMSD for this branch compares the cropped enzyme with its prediction.

The complex branch restores the last RF3 structure for the surviving designs, retaining apo-screen metrics. Its JSON input maps the actual `ZnO` residue name to `[Zn+2].O`: this is a disconnected zinc/water representation, **not an explicit zinc–hydroxide bond or a coordination constraint**. Interpret metal chemistry accordingly.

Generation checks reject chain breaks and clashes and require non-loop structure and ligand separation. Apo selection requires pLDDT ≥80 and CA RMSD ≤2 Å; complex selection requires pLDDT ≥80 and backbone RMSD ≤2.5 Å. `results.csv` and the per-step run artifacts retain outputs and provenance. An empty early screen stops the workflow before subsequent predictions.

## Validation limits

Earlier H200 diagnostics exercised this workflow with inputs that are no longer
redistributed. They demonstrated execution, not a passing enzyme design. Tests
now use synthetic structures to check AF3 input assembly, residue remapping and
motif coverage. A new user-supplied input needs its own bounded GPU validation;
these tests do not establish catalytic activity or reproduce the historical campaign.
