#!/usr/bin/env -S bash -c '"${CHAPERONE_REPO:-.}/.venv/bin/python" "$0" "$@"'
"""End-to-end exercise of ``auto_store_format="catcif"`` + ``auto_store_cleanup=True``.

Pipeline:
    [RFD3 (unconditional 80-mer, 2×5) -> MPNN (2 seqs/backbone)]  # one GPU job
    -> [AF3 (refolding, 1 diffusion sample)]                      # separate job
    -> RMSD (bkbn + CA, inline)

Chaining `[rfd3, mpnn]` in one ``engine.run`` call keeps the GPU allocation
across both steps so there's no re-queue penalty between backbone generation
and sequence design. Catcif stays per-chunk under ``rundir/<step>/`` —
``handler.store("all", dir)`` at the end packages everything into a single
archive per step for handoff.
"""

from __future__ import annotations

from pathlib import Path

from blocks import Blocks
from chaperone import ExecutionEngine, init_cli
from chaperone.data import SELECTIONS

SCRIPT_DIR = Path(__file__).resolve().parent
FILES_DIR = SCRIPT_DIR / "files"
RUNDIR = SCRIPT_DIR / "rundir"

# Partition and GPU type come from the blocks and the CHAPERONE_* site settings.
# On IPD, run this on backfill with ``--partition=gpu-bf``.
GPU = dict(time_limit="1:00:00")

RFD3, MPNN, AF3, RMSD = Blocks.getall("rfd3", "mpnn", "af3", "rmsd")


if __name__ == "__main__":
    init_cli()

    engine = ExecutionEngine(
        rundir=RUNDIR,
        auto_store_format="catcif",
        auto_store_cleanup=True,
    )

    # Stage 1: chain RFD3 + MPNN in a single SLURM job per chunk (shared GPU,
    # no re-allocation between steps). 2 chunks × 1 batch of 5 = 10 backbones,
    # each fed straight into MPNN which produces 2 sequences per backbone.
    h = engine.run(
        [
            RFD3(inputs=str(FILES_DIR / "rfd3_uncond.json"), batch_size=5, n_batches=2),
            MPNN(num_seq_per_pdb=2),
        ],
        job_args={
            **GPU,
            "mem_gb": 16,
            "rows_per_job": 1,
            "job_name": "catcif_rfd3_mpnn",
        },
    )

    # Stage 2: AF3 refolding in its own job — rows_per_job=5 spreads 20
    # designs across 4 chunks.
    h = engine.run(
        AF3(diffusion_batch_size=1, num_diffusion_samples=1),
        data=h,
        job_args={**GPU, "mem_gb": 16, "rows_per_job": 5, "job_name": "catcif_af3"},
    )

    # RMSD: inline (no store, no chunks).
    handler = engine.run(
        RMSD(
            alignment_methods=[SELECTIONS.BKBN, SELECTIONS.CA],
            rmsd_targets=[SELECTIONS.BKBN, SELECTIONS.CA],
        ),
        data=h,
    )

    # Optional on-demand consolidation: merge per-chunk archives into one
    # ``<step>.catcif`` per step under ``rundir/packaged/``. Per-chunk files
    # remain in place under ``rundir/<step>/``.
    handler.store("all", engine.rundir / "packaged")

    # Optional: extract every archived entry back to loose ``.cif`` files
    # under ``rundir/materialized/<step>/<entry_id>.cif`` — useful for handoff
    # or quick PyMOL loading without a catcif-aware client.
    handler.materialize(engine.rundir / "materialized", expand_catcif=True)

    print(f"\nFinal handler: {len(handler)} rows, steps = {handler.names}")
    print(handler.unpack())
