# Zinc protease design

This recipe scaffolds catalytic motifs around a zinc-containing site and a ten-residue substrate peptide, then designs enzyme sequences and predicts apo and complex structures.

```text
RFD3 → geometry filter → LigandMPNN → RF3 refinement
                                    ├→ select enzyme A → AF3 apo → filter
                                    └→ restore complex → AF3 complex → filter
```

## Inputs and prerequisites

The [source recipe](https://github.com/RosettaCommons/chaperone/tree/main/examples/enzyme-protease) bundles `GPRVVERHQS78.pdb` and two alternative contig arrangements. Both retain the substrate peptide as chain B and target a total specified length of 170–200 residues.

Install Foundry, LigandMPNN, and AlphaFold 3 runtimes using the [container guide](../containers.md), including the RFD3 and RF3 checkpoints and licensed AF3 weights. Site configuration supplies the scheduler and GPU requests.

The public port uses **regular RF3**, with substrate chain B selected as a template and ground-truth conformer, 50 diffusion steps, and five recycles. The original internal `rf3_dev` wrapper exposed additional auto-templating and classifier-free-guidance controls; this is a supported public replacement, not a numerical reproduction of that development model.

## Run

```bash
python examples/enzyme-protease/pipeline.py --test --rundir=/absolute/path/protease-smoke
```

Use `--local --use_local_gpus` inside an allocated GPU session. `--test` requests one backbone per contig specification. Defaults request two backbones per specification, two side-chain-packed MPNN sequences per backbone (one pack each), one RF3 refinement cycle with one diffusion sample, and one sample per AF3 branch.

## Chemistry, branching, and selection

LigandMPNN designs chain A, retaining motif sequences and omitting new cysteine/histidine choices elsewhere. The apo branch explicitly crops to enzyme chain A before prediction; removing ligands alone would leave the peptide present. RMSD for this branch compares the cropped enzyme with its prediction.

The complex branch restores the last RF3 structure for the surviving designs, retaining apo-screen metrics. Its JSON input maps the actual `ZnO` residue name to `[Zn+2].O`: this is a disconnected zinc/water representation, **not an explicit zinc–hydroxide bond or a coordination constraint**. Interpret metal chemistry accordingly.

Generation checks reject chain breaks and clashes and require non-loop structure and ligand separation. Apo selection requires pLDDT ≥80 and CA RMSD ≤2 Å; complex selection requires pLDDT ≥80 and backbone RMSD ≤2.5 Å. `results.csv` and the per-step run artifacts retain outputs and provenance. An empty early screen stops the workflow before subsequent predictions.

## Verification and scientific limits

A bounded H200 check completed RFD3 (`protease_contig1`, one backbone), LigandMPNN (one sequence and one side-chain pack), regular RF3 refinement, both AF3 branches, and their metric/filter blocks. RF3 and both AF3 branches were rerun after fixed-coordinate preservation was enabled. The generated backbone passed its geometry gate, but neither AF3 branch passed: apo pLDDT was 54.9 with CA RMSD 16.96 Å, and complex pLDDT was 65.0 with backbone RMSD 18.60 Å. The complex branch was exercised diagnostically after apo rejection. These results demonstrate runtime coverage, not a successful protease design. The second contig arrangement was configuration-tested only.

A packed-output audit preserved all fixed enzyme/substrate residue identities, all 212 observed fixed protein heavy atoms (including all 81 substrate atoms), and both ligand atoms. With coordinate preservation enabled, observed fixed protein coordinates match staged inputs exactly; differences from the original higher-precision RFD3 CIF remain below 0.001 Å. Packed residue sequences match the sampled FASTA. This matters because the upstream packer reconstructs geometry even for fixed sequence positions; Chaperone restores those observed fixed coordinates by default unless explicit repacking is requested.

Regression tests parse the bundled complex, verify that both enzyme and peptide enter the complex AF3 input, verify the ligand SMILES mapping, and check that the apo input contains only chain A. The diagnostic deliberately exercised both AF3 branches despite the failed apo screen; the ordinary recipe stops when no designs survive. Runtime success does not establish proteolytic activity, selectivity, or reproduction of the original campaign's results.

Adapted from the original `baker-laboratory/pipelines` example at source snapshot `95ef0b6`.
