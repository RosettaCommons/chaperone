# PPI Binder Design — Benchmark

De novo design of a protein binder for the human insulin receptor
(PDB: 4ZXB), run through two parallel inverse-folding backends on the
same RFD3 scaffolds and the same RF3 validation. This directory fuses
two `pipeline_*.py` variants that share `files/` and this README so
metrics between the two stacks are directly comparable on an identical
target.

Based on the [RFD3 PPI design tutorial](https://rosettacommons.github.io/foundry/models/rfd3/tutorials/ppi_design_tutorial.html).

## Variants

| File | MPNN backend | Notes |
|---|---|---|
| `pipeline_foundry.py` | `foundry_mpnn` (production foundry fork) | Canonical all-foundry PPI stack. |
| `pipeline_legacy_mpnn.py` | legacy `mpnn` block (fused_mpnn) | End-to-end cloud smoke bed for the legacy MPNN stack. |

Both pipelines share the same block sequence; the only difference is
which MPNN block is plugged in at the inverse-folding step:

```
rfd3 → filter → mpnn → rf3 → rmsd → filter
```

## Input

- **files/4zxb_cropped.pdb** — Cropped insulin receptor structure from PDB 4ZXB.
- **files/rfd3_config.json** — RFD3 binder design spec with hotspot residues E64, E88, E96.

## Running on-prem (SLURM)

```bash
./examples/benchmarks/ppi/pipeline_foundry.py     --config=tests/test_overrides.yaml
./examples/benchmarks/ppi/pipeline_legacy_mpnn.py --config=tests/test_overrides.yaml
```

Both use `FOUNDRY_CONTAINER` for RFD3 and RF3. The foundry variant
also uses it for MPNN; the legacy variant auto-selects the
institutional modelhub SIF at
`/projects/ml/modelhub/apptainer/frozen_modelhub_datahub_cifutils_2025-02-05.sif`
when it exists on disk, otherwise falls back to the cloud-buildable
`modelhub_cloud` container (see `src/blocks/networks/mpnn.py`).

## Running on cloud (Modal)

Shared prerequisites:
- `~/.modal.toml` populated (`modal setup` once).
- `FOUNDRY_CHECKPOINT_DIRS` pointing at a local foundry checkpoint tree.

Legacy-only additional prerequisite:
- `MPNN_WEIGHTS_DIR` pointing at a local LigandMPNN/fused_mpnn weight
  tree with at least `ligand_mpnn_model_weights/` and `packer_weights/`.

```bash
# All-foundry:
PIPELINES_COMPUTE_BACKEND=modal \
FOUNDRY_CHECKPOINT_DIRS=/abs/path/to/foundry/weights \
./examples/benchmarks/ppi/pipeline_foundry.py --config=tests/test_overrides.yaml

# Legacy mpnn:
PIPELINES_COMPUTE_BACKEND=modal \
FOUNDRY_CHECKPOINT_DIRS=/abs/path/to/foundry/weights \
MPNN_WEIGHTS_DIR=/abs/path/to/mpnn/weights \
./examples/benchmarks/ppi/pipeline_legacy_mpnn.py --config=tests/test_overrides.yaml
```

First cloud runs rebuild container images (foundry ~30–60 min,
modelhub_cloud ~5–10 min); subsequent runs hit Modal's image cache.
Weights upload once to `/weights/foundry/checkpoints` and
`/weights/mpnn` on the data volume; re-runs are near-instant.

## Apple Silicon status

- **RFD3**: Runs on MPS natively.
- **Foundry MPNN**: Forced to CPU on MPS (GatherND crash, auto-wrapped).
- **RF3**: Not yet supported on Apple Silicon (MLIR reshape crash in
  the diffusion transformer). Use CUDA or CPU on Linux.

## Porting to a new target

1. Replace `files/4zxb_cropped.pdb` with your target structure.
2. Update `files/rfd3_config.json`: contig string, hotspots, chain selection.
3. Adjust filter thresholds in the chosen `pipeline_*.py` as needed.

## Notes

- `baker-laboratory/fused_mpnn` is a private repo; cloud builds of
  `modelhub_cloud` require git credentials with access on whoever's
  running `pipelines install` or the first cloud job.
- See `docs/cloud.md` for the full cloud-readiness matrix and setup
  guide, and `tests/test_cloud_smoke.py` for per-block Modal smoke
  tests (`test_foundry_mpnn_modal_smoke`, `test_legacy_mpnn_modal_smoke`,
  `test_rfd3_modal_smoke`, `test_rf3_modal_smoke`).
