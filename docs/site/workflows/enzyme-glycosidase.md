# Glycosidase active-site design

This recipe scaffolds six catalytic residues and a substrate from a user-supplied theozyme, designs sequences with LigandMPNN, and evaluates the resulting protein in three AlphaFold 3 states.

```text
RFD3 → geometry filter → LigandMPNN → AF3 apo → apo filter
                                  ↘ AF3 covalent intermediate → PTM filter
                                  ↘ AF3 separate substrate → ligand filter
```

MPNN emits designed, side-chain-packed structures. Each later branch restores the shared MPNN structure while preserving survivors and metrics from the previous branch. It does not feed predicted apo coordinates into the covalent-intermediate predictor.

## Inputs and prerequisites

The [source recipe](https://github.com/RosettaCommons/chaperone/tree/main/examples/enzyme-glycosidase) provides workflow code and RFD3 configuration templates. Custom structures and chemistry templates are **not bundled** pending redistribution permission. Supply an authorized input directory containing `glycosidase.json`, `theozyme.pdb`, `af3_template_ptm.json` and `af3_template_ligand.json`; copy the RFD3 JSON from the example and resolve its structure paths there. The recipe expects six catalytic residues including GLU A86, a covalent intermediate named `L:1`, and separate substrate chain B. The catalytic filter expects 38 matched atoms. Adapt these selections and coverage thresholds together for a different theozyme.

Install the Foundry, LigandMPNN, and AlphaFold 3 runtimes using the [container guide](../containers.md). Configure model checkpoints and the licensed AF3 weights for your installation. No laboratory filesystem paths or scheduler partitions are embedded in the recipe.

The PTM template defines the glycosyl–glutamate intermediate as a custom CCD residue. `add_ptm_at_mapped_residues={"L:1": "A86"}` resolves the **original theozyme residue A86** through RFD3's index map before attaching it to the generated sequence. The separate-ligand template instead adds substrate chain B. These are different chemical representations and should not be treated as interchangeable measurements.

## Run

From an installed source checkout:

```bash
python examples/enzyme-glycosidase/pipeline.py --input_dir=/path/to/authorized/inputs --test --rundir=/absolute/path/glycosidase-smoke
```

On an allocated GPU append `--local --use_local_gpus`. Otherwise the engine submits scheduler jobs using your site configuration. `--test` requests one design **per input specification**, not one design total. Defaults use one batch of two backbones per specification and two sequences per backbone with one side-chain pack per sequence; predictions use one sample each.

## Selection and outputs

The generation filter checks chain breaks, clashes, non-loop fraction, ligand separation, motif insertion RMSD, and conjoined residues. Apo selection requires pLDDT ≥80 and backbone RMSD ≤2 Å. The PTM branch requires common-backbone RMSD ≤2 Å and reference-motif RMSD ≤2 Å, with all six catalytic residues and all 38 theozyme atoms matched; ligand selection requires backbone RMSD ≤2 Å and ligand pLDDT ≥80.

The run directory contains per-step structures, metrics, lineage, and `results.csv`. If a branch has no survivors, the script writes its empty result and stops. An empty result does not demonstrate that later AF3 branches executed.

The modified glutamate appears as `HETATM` in AF3 output. The backbone screen pairs the common **unmodified** backbone atoms. A separate `af3_ptm_catalytic` metric aligns the six-residue active site directly to the input theozyme, matches atoms by mapped residue and atom name (including the modified glutamate), and requires exactly six residues / 38 atoms. This replaces the historical PTM all-atom comparison, which could silently exclude the modified residue; the two RMSDs use different alignments and must not be conflated. The inherited 2 Å cutoff is retained as an illustrative screen, not as a calibrated threshold equivalent to the original backbone-aligned motif metric. The reference-motif score measures internal active-site agreement after its own fit, while the separate backbone score measures the overall fold.

## Validation limits

Earlier H200 diagnostics exercised this workflow with inputs that are no longer
redistributed. They demonstrated execution, not a passing enzyme design. Tests
now use synthetic structures to check AF3 input assembly, residue remapping and
motif coverage. A new user-supplied input needs its own bounded GPU validation;
these tests do not establish catalytic activity or reproduce the historical campaign.
