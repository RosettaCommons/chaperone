# Glycosidase active-site design

This recipe scaffolds six catalytic residues and a substrate from a bundled theozyme, designs sequences with LigandMPNN, and evaluates the resulting protein in three AlphaFold 3 states.

```text
RFD3 → geometry filter → LigandMPNN → AF3 apo → apo filter
                                  ↘ AF3 covalent intermediate → PTM filter
                                  ↘ AF3 separate substrate → ligand filter
```

MPNN emits designed, side-chain-packed structures. Each later branch restores the shared MPNN structure while preserving survivors and metrics from the previous branch. It does not feed predicted apo coordinates into the covalent-intermediate predictor.

## Inputs and prerequisites

The [source recipe](https://github.com/RosettaCommons/chaperone/tree/main/examples/enzyme-glycosidase) bundles `theozyme.pdb`, two RFD3 specifications (`ori` and `full`), and separate AF3 JSON templates for the intermediate and substrate. `ori` adds burial/exposure conditioning; `full` retains the catalytic-tip constraints. Both generate 130–220-residue scaffolds.

Install the Foundry, LigandMPNN, and AlphaFold 3 runtimes using the [container guide](../containers.md). Configure model checkpoints and the licensed AF3 weights for your installation. No laboratory filesystem paths or scheduler partitions are embedded in the recipe.

The PTM template defines the glycosyl–glutamate intermediate as a custom CCD residue. `add_ptm_at_mapped_residues={"L:1": "A86"}` resolves the **original theozyme residue A86** through RFD3's index map before attaching it to the generated sequence. The separate-ligand template instead adds substrate chain B. These are different chemical representations and should not be treated as interchangeable measurements.

## Run

From an installed source checkout:

```bash
python examples/enzyme-glycosidase/pipeline.py --test --rundir=/absolute/path/glycosidase-smoke
```

On an allocated GPU append `--local --use_local_gpus`. Otherwise the engine submits scheduler jobs using your site configuration. `--test` requests one design **per input specification**, not one design total. Defaults use one batch of two backbones per specification and two sequences per backbone with one side-chain pack per sequence; predictions use one sample each.

## Selection and outputs

The generation filter checks chain breaks, clashes, non-loop fraction, ligand separation, motif insertion RMSD, and conjoined residues. Apo selection requires pLDDT ≥80 and backbone RMSD ≤2 Å. The PTM branch requires common-backbone RMSD ≤2 Å and reference-motif RMSD ≤2 Å, with all six catalytic residues and all 38 theozyme atoms matched; ligand selection requires backbone RMSD ≤2 Å and ligand pLDDT ≥80.

The run directory contains per-step structures, metrics, lineage, and `results.csv`. If a branch has no survivors, the script writes its empty result and stops. An empty result does not demonstrate that later AF3 branches executed.

The modified glutamate appears as `HETATM` in AF3 output. The backbone screen pairs the common **unmodified** backbone atoms. A separate `af3_ptm_catalytic` metric aligns the six-residue active site directly to the input theozyme, matches atoms by mapped residue and atom name (including the modified glutamate), and requires exactly six residues / 38 atoms. This replaces the historical PTM all-atom comparison, which could silently exclude the modified residue; the two RMSDs use different alignments and must not be conflated. The inherited 2 Å cutoff is retained as an illustrative screen, not as a calibrated threshold equivalent to the original backbone-aligned motif metric. The reference-motif score measures internal active-site agreement after its own fit, while the separate backbone score measures the overall fold.

## Verification and scientific limits

A bounded H200 check completed RFD3 (`ori`, one backbone), LigandMPNN (one sequence and one side-chain pack), all three AF3 branches (one prediction each), and their metric/filter blocks. The minimal sample passed **none** of the generation or AF3 branch gates; downstream predictions were run separately for execution coverage. This is GPU runtime evidence, not a successful enzyme-design campaign. The `full` conditioning variant was configuration-tested only.

The packed-output audit checked all six motif residue identities, all 62 observed motif heavy atoms, and all 22 ligand atoms. With coordinate preservation enabled, fixed atoms match staged input coordinates exactly; differences from the higher-precision RFD3 CIF stay below 0.001 Å because of PDB staging. Packed residue sequences match the sampled FASTA. Designable residues remain packed, and newly completed missing atoms retain their predicted positions.

Regression tests exercise both actual bundled AF3 templates, catalytic-residue remapping, and the covalent bond definition. The first GPU generation attempt with the preserved theozyme produced an RDKit ligand-conformer warning: Foundry retained the constrained motif coordinates but fell back to zero ligand reference coordinates. The PDB supplies connectivity without explicit bond orders, and the historical separate-ligand CCD specifies single bonds throughout. Confirm the intended substrate chemistry before using this recipe for a scientific campaign; those historical representations have not been chemically reinterpreted in the port. The completed diagnostic exercised all three branches despite the failed generation screen; bypassing filters measured runtime coverage only. Neither confidence nor motif RMSD establishes glycosidase activity, substrate specificity, or a reaction rate.

Adapted from the original `baker-laboratory/pipelines` example at source snapshot `95ef0b6`.
