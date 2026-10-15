# RFOptimization input and output

`rfoptimization` refines an existing PDB/CIF complex through the pinned public
[Foundry RFO implementation](https://github.com/RosettaCommons/foundry/tree/69d5f735f272b78b6b23c230abba849f18315234/models/rfo).
The RFO image uses AtomWorks 2.2.1: the upstream RFO code still
imports selection APIs removed in AtomWorks 3. The other Foundry blocks use
their own image. A guarded compatibility repair disables geometry-based cyclic
bond guessing for RFO’s sequence-only placeholder coordinates, which otherwise
creates false closure bonds in linear proteins. It also sets the training
residue-embedding cache to `None`, so results cannot silently depend on a
lab-only cache being present on a particular cluster.

Its stages are RF3 gradient-guided sequence optimization **or** Boltz-2 prediction,
followed by upstream ProteinMPNN/LigandMPNN redesign. Each input receives an
independent run directory and branch/MPNN seed. Upstream RF3/Boltz sampling
is not fully seeded by this driver, so this is not an exact-reproducibility guarantee.

## Installation

```bash
chaperone install rfoptimization
```

The image contains three separate environments with pinned Foundry, Boltz-2,
and LigandMPNN source/weights. It does not need a local Foundry development
checkout, nested Apptainer, a specific cluster filesystem, or a named GPU model.
Select scheduler partitions/GPU requests through normal Chaperone configuration.
The block sets Foundry’s supported `DISABLE_CUEQUIVARIANCE=1` switch for native
PyTorch attention: the fused triangle kernel fails in this RF3 gradient path.

For RF3 cycles, download public confidence-enabled RF3 weights (`foundry install
rf3` in a Foundry environment), and supply `rf3_checkpoint` or configure
`FOUNDRY_CHECKPOINT_DIRS`. For Boltz cycles, set `boltz_cache` to a writable cache;
Boltz downloads missing public assets there. Existing `BOLTZ_CACHE_DIR` is honored.

## Input contract

Use a `PipelineDataHandler` containing structures. Chain **A** must be the
protein to redesign. Other chains are interaction partners. The RF3 cycling
preset chooses contacts with chain **B**; use an A/B protein complex for this
preset. Rename chains explicitly before calling the block. The wrapper orders chain A
first because the RF3 optimizer selects its binder by position. RFO can renumber
residues during prediction; input residue numbers are not output constraints. Standard amino acids
and complete N/CA/C/O backbones are required in chain A. PDB staging preserves
residue identifiers subject to the usual PDB limits.

| Field | Default | Meaning |
| --- | --- | --- |
| `total_cycles` | 10 | Number of completed prediction/optimization + MPNN cycles |
| `backprop_fraction` | 0.5 | Per-cycle probability of RF3; 0 = Boltz only, 1 = RF3 only |
| `mpnn_model_type` | `protein_mpnn` | Or `ligand_mpnn` for ligand-aware redesign |
| `mpnn_temperature` | 0.1 | Positive sequence sampling temperature |
| `seed` | 42 | Base seed; independent offsets for chunks and inputs |
| `rf3_checkpoint` | configured Foundry checkpoint | Public confidence-enabled RF3 checkpoint |
| `boltz_cache` | `BOLTZ_CACHE_DIR` or `~/.boltz` | Writable model/CCD cache |
| `rf3_config` | upstream `cycle_ppi.yaml` | Optional optimizer YAML; requires a confidence loss |
| `msa_paths` | `{}` | Chain ID → A3M file path; no MSA server is contacted |
| `templates` | `{}` | Chain ID → structure path; RF3 branch only |
| `cyclic` | false | Cyclic chain A annotation in Boltz only |

```python
from chaperone import ExecutionEngine, PipelineDataHandler
from blocks.networks.rfoptimization.block import RFOptimizationBlock

data = PipelineDataHandler.from_paths(["starting_complex.cif"], name="input")
engine = ExecutionEngine(rundir="runs/rfo")
result = engine.run(
    RFOptimizationBlock(total_cycles=1, backprop_fraction=1.0,
                        rf3_checkpoint="/path/to/rf3_checkpoint.ckpt"),
    data=data,
)
```

For a Boltz-only diagnostic use `backprop_fraction=0.0`. To compare both routes,
run each explicitly in separate directories: a single mixed cycle cannot
exercise both models.

## Outputs and limitations

1. The returned structure is the **final MPNN-redesigned complex**. Upstream RFO
   removes obsolete side chains at mutations; it does **not** repack all side
   chains. Run a packing/refolding step before using atom-level geometry or
   final-sequence confidence as selection metrics.
2. RFO records contain scores of the prediction **before** its subsequent MPNN
   redesign. They are retained in `records_path`, and are not published as
   confidence metrics for the returned structure. `completed_cycles` is the
   block's completion metric. Final PDBs have unique published basenames and
   parent-linked design IDs. All intermediate predictions remain ordinary run
   artifacts under `design_<n>/output/recycle_<n>/`.
3. Only the first MPNN sample is carried forward upstream; this wrapper requests
   exactly one sample per cycle. Other chains and ligands are retained by MPNN.
4. Chaperone motif masks are rejected explicitly: RFO chooses its own fixed
   residues and cannot guarantee that an upstream motif mask is respected.
5. `cyclic=True` does not impose a closure bond in RF3. The general protein
   preset is not a complete enzyme/covalent-ligand/cyclic-peptide setup.
6. Public RF3 checkpoints differ from historical research checkpoints. A small
   GPU execution verifies software compatibility, not the paper's success
   rates, experimental binding, or improvement over its starting design.

The [RFO paper](https://www.biorxiv.org/content/10.64898/2026.09.04.749184v2)
includes protein binders (including FGFR2), cyclic peptides, small-molecule
binders, and enzyme active-site scaffolds. Independent AF3 refolding/filtering
is outside this block.
