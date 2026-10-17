# BindCraft2 binder design

`bindcraft2` wraps [PacesaLab/BindCraft2](https://github.com/PacesaLab/BindCraft2),
pinned to revision `c0ee638d11bc010817c2c97ee58ed5c323e664e2` (package 1.0.3).
It runs JAX/AlphaFold2 binder optimization, the bundled ProteinMPNN redesign,
and validation with AlphaFold2 models separate from optimization. Returned structures are accepted, refolded
**target–binder complexes with side chains**, accompanied by their native scores.

This initial interface supports the upstream `binder` modality against **one
protein chain per input**. It does not expose scaffold-constrained designs, peptides,
multistate/multichain targets, ligand targets, or upstream parameter sweeps.
Classic `bindcraft` remains a separate block with a different runtime.

## Installation and weights

```bash
chaperone install bindcraft2
```

The container uses Python 3.12, JAX 0.11.0 with CUDA 12, and Biotite 1.6.0. It
requires an NVIDIA GPU and a compatible host driver. It does not depend on
Foundry, AtomWorks, PyRosetta, a private cluster filesystem, or private weights.
Chaperone can read its returned CIFs using AtomWorks 3.

ProteinMPNN weights are bundled upstream. AlphaFold2 parameters are downloaded
on first use into `weights_dir/alphafold` (default `~/.cache/bindcraft2/alphafold`);
the archive is approximately 5.3 GB. For an offline compute node, fetch them
before submission using the installed container:

```bash
mkdir -p "$HOME/.cache/bindcraft2"
apptainer exec \
  --bind "$HOME/.cache/bindcraft2:$HOME/.cache/bindcraft2" \
  --env "BINDCRAFT_WEIGHTS=$HOME/.cache/bindcraft2" \
  "$HOME/.chaperone/containers/bindcraft2.sif" bindcraft fetch-weights
```

Adjust the image path if `CHAPERONE_CONTAINER_DIRS` points elsewhere. Existing
AlphaFold2/ColabFold weights can be reused by passing `af2_params_dir` or setting
`BINDCRAFT_AF2_PARAMS`. The directory must contain `params_model_1_ptm.npz`,
`params_model_2_ptm.npz`, and `params_model_1_multimer_v3.npz` through
`params_model_5_multimer_v3.npz`; a nested `params/` directory is also accepted.
Explicit missing checkpoints fail rather than triggering an implicit download.

BindCraft2 uses its own
[Source-Available License (Hosting-Restricted)](https://github.com/PacesaLab/BindCraft2/blob/c0ee638d11bc010817c2c97ee58ed5c323e664e2/LICENSE).
The source and license remain inside the container. Model weights retain their
upstream terms.

## Example: PD-L1

Download the upstream
[hPDL1 PDB](https://github.com/PacesaLab/BindCraft2/blob/c0ee638d11bc010817c2c97ee58ed5c323e664e2/settings/target/structures/hPDL1.pdb)
to `inputs/hPDL1.pdb`. These hotspots use that file's residue numbering:

```python
from blocks import Blocks
from chaperone import ExecutionEngine, PipelineDataHandler

engine = ExecutionEngine(rundir="runs/bindcraft2-pdl1")
targets = PipelineDataHandler.from_paths(["inputs/hPDL1.pdb"], name="target")
designs = engine.run(
    Blocks["bindcraft2"](
        chain="A",
        hotspots="54,56,66,115",
        lengths=(60, 100),
        number_of_final_designs=1,
        max_trajectories=10,
        seed=42,
        # af2_params_dir="/your/alphafold/parameters",
    ),
    data=targets,
)
```

The block stages only `chain`; other input chains are deliberately excluded.
The selected chain must contain standard protein residues with complete
N/CA/C/O backbones, integer residue IDs, no insertion codes, and no ligands.
Normalize/crop the target explicitly before use. Hotspots are comma-separated
residue numbers or inclusive ranges (`54,66-70`) and must exist in that chain.
Chaperone motif constraints are rejected because this interface cannot enforce
them during design.

## Budgets, settings, and outputs

- Each input runs in its own engine chunk on one GPU. Keep `rows_per_job=1`;
  combining targets in one chunk is rejected. Upstream multi-GPU workers and
  speculative next-length compilation are disabled. The seed increments by
  chunk ID. Identical seeds do not imply bitwise reproducibility across hardware.
- `max_trajectories=10` bounds attempts; it does not guarantee an accepted design.
  If fewer than `number_of_final_designs` pass, the step fails and leaves upstream
  trajectory/refold logs for diagnosis. Choose a larger budget only after a pilot.
- Native scientific defaults and quality filters are preserved. `overrides`
  accepts selected settings such as `design_recycles`, `validation_recycles`,
  `sequence_candidates`, `mpnn_variant`, stage step counts, confidence thresholds,
  and `filters`. It rejects control settings that would bypass staging, resource
  limits, or accepted-only collection. Upstream validates scientific setting values.
- Outputs come exclusively from `3_Ranked/!_Ranked.csv` and its matching complex
  CIFs. Trajectories, rejected refolds, monomers, and optional relaxed coordinates
  are not returned. Missing files, wrong sequences, non-finite coordinates, and
  lost chains fail compilation.
- Output chain **A** is the target and **B** is the binder, irrespective of the
  input chain label. Structure metadata records the fixed-target `motif` and
  source input; `metrics.bindcraft2.provenance` records chain roles, parent
  identity, seed, settings path, and upstream revision. Structures receive
  unique published names so independent chunks do not collide downstream.
- Metrics under `bindcraft2` describe the returned refolded complex. Preserve
  this structure state when comparing scores. Refolding can move the target;
  unchanged target sequence does not mean unchanged coordinates.
- Passing the configured filters is a computational result, not experimental
  binding evidence. A shortened run with relaxed thresholds tests execution
  only. Use an independent final oracle before selecting candidates.
