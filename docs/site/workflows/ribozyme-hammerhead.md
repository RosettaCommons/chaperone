# Hammerhead ribozyme rescaffolding

Design a 60-nucleotide RNA around six catalytic nucleotides from the
Schistosoma hammerhead crystal structure, PDB 2OEU, and its active-site Mn²⁺.
RFD3NA chooses the positions of the unindexed motif within one designed RNA
chain. AF3 then predicts the RNA and ion, and the metrics compare the predicted
motif with the crystal reference.

```text
RFD3NA → motif/clash filter → AF3 → crystal motif RMSD
       → catalytic distances and angle → geometry filter
```

## Inputs

The source directory is
[`examples/ribozyme-hammerhead/`](https://github.com/RosettaCommons/chaperone/tree/main/examples/ribozyme-hammerhead).
`files/2oeu_motif.pdb` contains A20–22, A36, B6–7 and Mn.
`files/2oeu_native.cif` is the full native reference for optional controls.
The unindexed groups are joined with commas, not `/0`: introducing a molecule
break changes the connectivity requested of the model. Input JSON is generated
under the run directory, leaving the bundled files unchanged.

## Dependencies and bounded run

Install the `foundry` and `alphafold3` images. Stage the public RFD3NA checkpoint
(`foundry install rfd3na --checkpoint-dir /absolute/path/checkpoints`) and set
`FOUNDRY_CHECKPOINT_DIRS`. Obtain AF3 model parameters under their applicable
terms and set `ALPHAFOLD3_WEIGHTS_DIR`.

```bash
python examples/ribozyme-hammerhead/pipeline.py --test --rundir=/absolute/path/hammerhead-smoke
```

Append `--local` inside an allocated GPU session. Otherwise the engine submits
jobs through your Slurm site settings. The default produces two 60-nt designs;
`--test` requests one, followed by one AF3 sample if the first filter passes.
The RFD3NA cloud backend is not currently enabled: use a configured cluster or
local GPU with the staged checkpoint.

## Measurements and selection

The generation filter requires motif insertion RMSD ≤1 Å, zero conjoined
residues, and zero reported interresidue backbone/sidechain clashes. These checks
do not establish complete phosphodiester connectivity.

After AF3, the example requires crystal-reference RNA motif RMSD ≤2 Å,
C17 O2′–C1.1 P attack distance ≤4 Å, and O2′–P–O5′ angle ≥140°.
General-acid/base distances and Mn–A9/Mn–G10.1 distances are reported for
inspection. Mn is excluded from the RNA motif RMSD and its distances are not
selection gates. These settings preserve the source example's computational
criteria; they are not a validated predictor of cleavage activity.

Normal engine manifests, structures, metric tables, and `results.csv` are
written to the run directory. An empty table means no design passed. If the
first filter rejects the sole smoke design, AF3 has not been tested by that run.

## Validation scope

CPU tests verify that every atom addressed by the catalytic measurements exists
in the bundled motif and that the generated specification requests one 60-nt
RNA molecule. A bounded NVIDIA H200 run using the public RFD3NA 1190 checkpoint
completed 200 diffusion steps (seed 42), producing 60 RNA residues and one Mn.
Motif insertion RMSD was 0.257 Å, with zero conjoined residues, zero reported
backbone clashes, and one reported sidechain clash. That candidate fails the
example’s clash gate. A diagnostic continued through AF3 and both metric blocks
to verify execution, without changing the shipped selection gates. AF3 seed 42 / one sample completed
with 1,285 finite-coordinate atoms, 60 RNA residues, and one Mn. All catalytic
measurements were finite. Mean pLDDT was 82.13; crystal motif RMSD was 5.166 Å,
attack distance 4.046 Å, and attack angle 73.12°. This candidate also fails the
refold gates. There was **no accepted design** in this one-sample check.

Generation used the native AtomWorks 3 Foundry environment and checkpoint 1190;
refolding used the existing AF3 image with this repository's maintained runner,
not a newly built public container. AF3 model inference took 37.8 seconds after
featurisation on the H200. The ordinary run artifacts include `generation.csv`,
`refolding.csv`, structures, manifests, and logs under the validation run's
`rna` directory. This verifies GPU model execution, output compilation, motif
mapping, and scoring; it does not certify a fresh container installation.

Neither a successful model execution nor these geometry filters establish RNA catalytic activity, stability, or experimental
function. Check O3′–P connectivity and native controls before a scientific pilot.

Adapted from `baker-laboratory/pipelines`, main snapshot `95ef0b6`.
Historical success rates from that source are not evidence for this public port.
