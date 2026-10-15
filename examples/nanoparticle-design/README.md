# C3 nanoparticle scaffold design

A bounded RFdiffusion v1 → I3 RPXdock → tied ProteinMPNN → AF3 workflow.
It generates a C3 scaffold around a fixed antigen stub, docks its asymmetric
unit into an icosahedral cage, and refolds the selected **trimer**.

## Inputs and dependencies

The historical motif/stalk structures are not bundled pending redistribution
permission. Supply your authorized copies in `--input_dir`; the code remains
available. For different stubs, update the contigs and motif residue selections.

- `dn5_stump.pdb` in your input directory: fixed motif, residues 1–7 of chains A/B/C.
- `dn5b_stalk.pdb` in the same directory: antigen stalk in the same reference frame.
- RFdiffusion v1 weights: configure `RFDIFFUSION_WEIGHTS_DIR`.
- Separately obtained AF3 weights: configure `ALPHAFOLD3_WEIGHTS_DIR`.
- Apptainer or Docker and a supported GPU for RFdiffusion/AF3; RPXdock runs on CPU.

From the repository root, in the installed Chaperone environment:

```bash
chaperone install rfdiffusion1
chaperone install rpxdock
chaperone install ligandmpnn
chaperone install alphafold3
python examples/nanoparticle-design/pipeline.py --input_dir=/path/to/authorized/inputs --test --rundir=/absolute/path/nanoparticle-smoke
```

Use `chaperone list-available` to inspect available image names. Add `--local`
inside an existing GPU allocation; otherwise the engine submits to Slurm using
your site settings. No private container is required. The
RPXdock recipe builds pinned public RPXdock/willutil sources and bundles their
`small_ilv_h` scoring tables. Its first build compiles native extensions.

## Steps and acceptance criteria

1. Generate one 157-residue-per-chain C3 scaffold, keeping the seven-residue motif.
2. Align the stalk through the RFdiffusion motif mapping. Reject more than eight
   stalk CAs within 3 Å of designed scaffold CAs (motif residues excluded).
3. Require C3 CA agreement within 2 Å; orient the axis to z, preserve the
   asymmetric unit's radial displacement, residue identities, and motif metadata.
4. Dock into I3. RFdiffusion's undesigned residues are glycine placeholders,
   so this example sets `ignored_aas=""`: RPX scores N/CA/C-derived backbone
   frames for them instead of excluding glycine. These frames use the upstream
   Rosetta-derived mean centroid offset; output residue identities and atoms
   are retained. The historical score threshold has not been calibrated for
   this placeholder-inclusive scoring protocol. Emit the highest-scoring model as a true C3 subgroup, with its
   corresponding score and model index. Require RPX score ≥70.
5. Design two sequences (one with `--test`) with corresponding residues tied
   across all three chains and antigen motif identities fixed. MPNN returns
   a packed all-atom structure for each sequence (`output_mode="structures"`,
   one sidechain pack per design).
6. Refold with one AF3 sample; compare against the docked trimer. Require
   backbone RMSD ≤2 Å and mean pLDDT ≥90.

The run directory contains normal engine manifests, structures, score tables,
and `results.csv`. A one-design run may yield no passing candidates. An early
filter rejection means later stages have **not** been exercised. Increase scale
only after checking the small run's outputs and resource use.

## Scope

CPU regression tests cover rigid-transform invariance, aligned clash counting,
C3 geometry, motif/lineage preservation, model-score matching, and rejection of
invalid docking outputs. CLI tests resolve the bounded configuration without
running the neural models. A native CPU integration smoke on the upstream
`C3_1na0-1_1.pdb.gz` fixture completed an I3 search (beam size 100) and Chaperone
compilation: one 357-CA trimer, three mapped motif residues, retained parent
lineage, and RPX score 0.03 (below the selection threshold). This used Python
3.12, NumPy 1.26.4, and the pinned RPXdock/willutil sources. The container recipe
itself still needs a build/run check on the deployment system.

A subsequent CoreWeave H200 check completed public RFdiffusion v1 (`86507b6`,
Base checkpoint, 50 steps) for the full 471-residue C3 scaffold. Its 157-residue
asymmetric unit had zero stalk clashes and C3 CA RMSD 0.000638 Å. A native CPU
I3 RPXdock search (beam 100, placeholder-inclusive scoring) produced a 471-CA
trimer with 21 mapped motif residues and score 0.14, below the example gate.
The RFD1 stack used PyTorch 2.4.0+cu124, DGL 2.4.0+cu124, NumPy 1.26.4, and
e3nn 0.5.1.

The same candidate then completed tied ProteinMPNN on CUDA, one sidechain pack,
and one AF3 sample (seed 42) on the H200. Packed and predicted structures each
contain 3,825 finite-coordinate atoms. All three 157-residue sequences are
identical; all 21 fixed motif residue identities and all 84 existing input motif
atom coordinates were preserved exactly through MPNN. The public runner
compiled AF3 pLDDT 77.11 and docked-trimer backbone RMSD 4.007 Å. AF3 model
inference took 47.5 seconds after featurisation.

This was an execution diagnostic: it continued past the RPX gate to exercise
packing and refolding. It fails both the RPX and AF3 selection criteria, so no
accepted nanoparticle design is claimed. Native public RFD1/RPX installations,
the public LigandMPNN image, and an existing AF3 image with the maintained
Chaperone runner were used. Fresh image builds remain a separate validation
step. Ordinary engine artifacts include `rpx.csv`, `diagnostic.csv`, structures,
and logs for this run.

AF3 evaluates the trimer, not assembly of the 60-subunit cage. The source recipe's
N-terminus check, Rosetta symmetry wiggle, and full-cage antigen/contact checks
are not implemented here. These require additional validation before selecting
experimental nanoparticles; a higher RPX score is not a substitute. The user-supplied
small scoring tables and historical threshold are example settings, not a
calibrated assembly-quality predictor.

Adapted from the `baker-laboratory/pipelines` nanoparticle example (main snapshot
`95ef0b6`), based on Cyrus Haas's nanoparticle workflow and the
[RFdiffusion work](https://doi.org/10.1038/s41586-023-06415-8).
The two input structures retain their source coordinates. The public helpers
correct the predecessor's unaligned clash calculation, recentering of the
monomer, motif loss, and rank/model-index confusion.
