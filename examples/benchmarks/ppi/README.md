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
| `pipeline_mpnn.py` | `mpnn` block (LigandMPNN) | Needs the `ligandmpnn_public` image (`chaperone install ligandmpnn_public`). |

Both pipelines share the same block sequence; the only difference is
which MPNN block is plugged in at the inverse-folding step:

```
rfd3 → filter → mpnn → rf3 → rmsd → filter
```

## Input

- **files/4zxb_cropped.pdb** — Cropped insulin receptor structure from PDB 4ZXB.
- **files/rfd3_config.json** — RFD3 binder design spec with hotspot residues E64, E88, E96.

## Running (Slurm or local)

```bash
./examples/benchmarks/ppi/pipeline_foundry.py     --test
./examples/benchmarks/ppi/pipeline_mpnn.py --test
```

Both use `FOUNDRY_CONTAINER` for RFD3 and RF3. The foundry variant
also uses it for MPNN; the `mpnn` variant runs LigandMPNN from the
`ligandmpnn_public` image (see `src/blocks/networks/mpnn.py`).
Set `CHAPERONE_COMPUTE_BACKEND` to `slurm` or `local` to force a mode;
containers run with Apptainer or Docker. Cloud (Modal) execution is not
supported.

## Apple Silicon status

- **RFD3**: Runs on MPS natively.
- **Foundry MPNN**: Forced to CPU on MPS (GatherND crash, auto-wrapped).
- **RF3**: Not yet supported on Apple Silicon (MLIR reshape crash in
  the diffusion transformer). Use CUDA or CPU on Linux.

## Porting to a new target

1. Replace `files/4zxb_cropped.pdb` with your target structure.
2. Update `files/rfd3_config.json`: contig string, hotspots, chain selection.
3. Adjust filter thresholds in the chosen `pipeline_*.py` as needed.
