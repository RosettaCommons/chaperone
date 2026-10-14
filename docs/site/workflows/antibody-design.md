# Antibody loop redesign with regular RFD3

This public example remodels **one CDR-H3 loop** in an existing antibody Fv–CD28
complex with the regular public RFD3 checkpoint. It then designs the loop sequence
with ProteinMPNN and refolds the complex with AF3. It does not require the private
antibody modelhub checkpoint or abMPNN-PoE used by the original example.

## Inputs and scope

The bundled structure is cropped from **PDB 8S6Z**, as distributed with Foundry's
binder-design tutorial. Chain A is the antibody heavy variable domain, B the light
variable domain, and C CD28. Constant domains, glycans, waters, and other
nonprotein components are omitted. This is a simplified structural example;
glycan-dependent binding is outside its scope.

The source author residue labels include insertion codes. The bundled PDB uses
consecutive residue numbers within each chain; `files/residue_mapping.json`
records every source label. The eight-residue H3 segment corresponding to source
heavy-chain residues 95–102 is regenerated. All remaining coordinates and sequences
are supplied as fixed context. These labels describe this fixture, not a general
antibody numbering algorithm.

```text
RFD3: fixed Fv/CD28 context + eight-residue H3 gap
  → ProteinMPNN: heavy chain only, fixed motif sequences preserved
  → AF3: one complex prediction
  → backbone RMSD and prediction confidence
```

RFD3 is a general model. Successful execution does not show equivalence to the
antibody-tuned model, native binding, developability, or preservation of affinity.
The script reports metrics without inventing an antibody acceptance threshold.

## Run

Install Chaperone and configure the `foundry`, `ligandmpnn`, and `alphafold3`
runtimes and their model weights. AF3 weights require authorized access.
From the checkout root:

```bash
python examples/antibody-design/pipeline.py --test --rundir=runs/antibody-smoke
```

Add `--local` inside an existing GPU allocation. The default is already one
backbone, one sequence and one AF3 sample. Outputs and lineage are stored in the
run directory; `results.csv` contains the final metrics.

For another antibody, prepare its structure and RFD3 contig explicitly. Preserve
framework and antigen residues as fixed motifs, and update chain selections if
needed. Replacing only the structure path does not update the fixture's loop
numbering.

## Validation

Regression tests check the input crop, contiguous numbering, exact loop gap,
fixed-framework sequence policy, packed output and smoke configuration.

A one-design test completed on a CoreWeave NVIDIA H200: regular RFD3 (200 steps,
seed 42), ProteinMPNN with one side-chain pack, AF3 with one sample, and backbone
RMSD. The packed structure contained 2,684 finite-coordinate atoms; all three
chain sequences matched the sampled FASTA, and framework, light-chain and antigen
sequences were unchanged. AF3 reported mean pLDDT 83.5 and complex backbone RMSD
6.23 Å. This is execution evidence, not evidence of a good antibody design.

The native packer moved some fixed atoms despite `repack_everything=0`.
After Chaperone's coordinate-preservation correction, all 2,628 observed heavy
atoms across the 340 fixed residues exactly matched the staged input PDB.
Sampled and packed sequences still matched. The sequence-only AF3 input was
unchanged, so its prediction was reused.

The test used the native Foundry AtomWorks 2 environment at revision `0999dce`,
the public LigandMPNN image, and an existing AF3 image with Chaperone's current
public runner and licensed weights. It did not rebuild every release container.
The full example path executed without scientific gate bypasses; this reporting
example does not define an antibody success threshold.

[Example source](https://github.com/RosettaCommons/chaperone/tree/main/examples/antibody-design)
