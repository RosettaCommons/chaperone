#!/usr/bin/env -S bash -c '"${CHAPERONE_REPO:-.}/.venv/bin/python" "$0" "$@"'
"""PPI binder design using the all-foundry stack (RFD3 + MPNN + RF3).

De novo design of a protein binder for the human insulin receptor (PDB: 4ZXB)
using only tools from the RosettaCommons Foundry suite. The pipeline generates
backbones with RFD3, redesigns sequences with Foundry MPNN, and validates
with RosettaFold3 structure prediction.
Pipeline: rfd3 -> filter -> foundry_mpnn -> rf3 -> rmsd -> filter
"""

from pathlib import Path

from blocks import Blocks
from blocks.util_blocks.filter import MetricFilter
from chaperone import ExecutionEngine, init_cli
from chaperone.data.schema import SELECTIONS

init_cli()

SCRIPT_DIR = Path(__file__).resolve().parent
FILES_DIR = SCRIPT_DIR / "files"

# ── Block configuration ─────────────────────────────────────────────────

RFD3, FoundryMPNN, RF3, RMSD, Filter = Blocks.getall(
    "rfd3", "foundry_mpnn", "rf3", "rmsd", "filter"
)

blocks = [
    # Generate binder backbones against insulin receptor
    RFD3(
        inputs=str(FILES_DIR / "rfd3_config.json"),
        ckpt_path=None,
        n_batches=10,
        batch_size=8,
    ),
    # Remove designs with chain breaks or excessive clashes
    Filter(
        name="rfd3_filter",
        filters={
            "rfd3.n_chainbreaks": MetricFilter(max=0),
        },
    ),
    # Redesign sequences with Foundry MPNN (ligand-aware)
    FoundryMPNN(
        model_type="ligand_mpnn",
        num_seq_per_pdb=4,
        keep_motif_sequences_fixed=True,
    ),
    # Validate with RosettaFold3 structure prediction
    RF3(),
    # Check backbone RMSD between designed and predicted structures
    RMSD(
        alignment_methods=[SELECTIONS.BKBN],
        rmsd_targets=[SELECTIONS.BKBN, SELECTIONS.MOTIF_ALLATOM],
    ),
    # Keep only designs that refold well with good interface metrics
    Filter(
        name="rf3_filter",
        filters={
            "rf3.iptm": MetricFilter(min=0.7),
            "rmsd.bkbn_align.bkbn_rmsd": MetricFilter(max=3.0),
        },
    ),
]

# ── Execute ──────────────────────────────────────────────────────────────

engine = ExecutionEngine(rundir=SCRIPT_DIR / "rundir", reinit=True)
handler = engine.run(blocks)

df = handler.unpack()
print(df)
