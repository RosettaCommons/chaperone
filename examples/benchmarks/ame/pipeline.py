#!/usr/bin/env -S bash -c '"${PIPELINES_REPO:-.}/.venv/bin/python" "$0" "$@"'
"""Active-site Motif Evaluation (AME) benchmark pipeline.

Evaluates de novo enzyme design quality by measuring how well AF3 recovers
the catalytic motif geometry after backbone generation and sequence design.
Runs two benchmark suites through a shared evaluation pipeline:

  AME-1: 41 MCSA enzyme active sites (dialect-1 format, fixed-length 180 res)
  AME-2: 4 enzyme tasks — glycosidase, aldolase, decarboxylase, phosphotriesterase

Pipeline (per suite):
  rfd3 -> [concat] -> rotamer_metrics -> ligand_mpnn -> af3 -> rmsd (tip-atom)

Tip-atom RMSD (motif_allatom) after AF3 folding is the primary success metric.
"""

from pathlib import Path

from blocks import Blocks
from chaperone import ExecutionEngine, init_cli
from chaperone.data.schema import SELECTIONS

init_cli()

SCRIPT_DIR = Path(__file__).resolve().parent
FILES_DIR = SCRIPT_DIR / "files"

RFD3, MPNN, AF3, RMSD, RotamerMetrics = Blocks.getall(
    "rfd3", "mpnn", "af3", "rmsd", "rotamer_metrics"
)

# ── Block configuration ─────────────────────────────────────────────────

rfd3_ame1 = RFD3(
    name="rfd3/ame1",
    inputs=str(FILES_DIR / "ame-1.json"),
    n_batches=10,
    batch_size=10,
)

rfd3_ame2 = RFD3(
    name="rfd3/ame2",
    inputs=str(FILES_DIR / "ame-2.json"),
    n_batches=10,
    batch_size=10,
)

shared_blocks = [
    RotamerMetrics(
        backend="dunbrack",
        motif_only=True,
    ),
    MPNN(
        model_type="ligand_mpnn",
        keep_motif_sequences_fixed=True,
        num_seq_per_pdb=8,
        args={"omit_AA": "XC"},
    ),
    AF3(
        read_most_confident_model=True,
    ),
    RMSD(
        which=[0, -1],
        alignment_methods=[SELECTIONS.BKBN],
        rmsd_targets=[SELECTIONS.BKBN, SELECTIONS.MOTIF_ALLATOM, SELECTIONS.LIGANDS],
        select_mask_by_name_pairing=[
            SELECTIONS.BKBN,
            SELECTIONS.MOTIF_ALLATOM,
            SELECTIONS.LIGANDS,
        ],
    ),
]

# ── Execute ──────────────────────────────────────────────────────────────

engine = ExecutionEngine(rundir=SCRIPT_DIR / "rundir", reinit=True)

# Run both AME suites through RFD3 separately
h_ame1 = engine.run(rfd3_ame1)
h_ame2 = engine.run(rfd3_ame2)

# Concatenate and feed through shared evaluation pipeline
handler = engine.run(shared_blocks, data=h_ame1 + h_ame2)

df = handler.unpack()
print(df)
