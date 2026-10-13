#!/usr/bin/env -S bash -c '"${CHAPERONE_REPO:-.}/.venv/bin/python" "$0" "$@"'
"""PPI binder design using RFD3 + public LigandMPNN (``mpnn`` block) + RF3.

Sibling of ``pipeline_foundry.py`` in this directory that swaps
Foundry MPNN for the ``mpnn`` block (public LigandMPNN). Same binder
target (human insulin receptor, 4ZXB) and shared ``files/`` so metrics
between the two inverse-folding backends are directly comparable on
identical RFD3 scaffolds and RF3 validation.

Pipeline: rfd3 -> filter -> mpnn -> rf3 -> rmsd -> filter

Run:
    FOUNDRY_CHECKPOINT_DIRS=/abs/path/to/foundry/weights \
    ./examples/benchmarks/ppi/pipeline_mpnn.py --test

The ``mpnn`` block runs from the ``ligandmpnn_public`` image; install it with
``chaperone install ligandmpnn_public``. Choose the model with ``model_type``
(``ligand_mpnn``, ``protein_mpnn``, ``soluble_mpnn``) and the checkpoint
version with ``noise_level``; see ``src/blocks/networks/mpnn.py``.
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

RFD3, MPNN, RF3, RMSD, Filter = Blocks.getall("rfd3", "mpnn", "rf3", "rmsd", "filter")

blocks = [
    # Generate binder backbones against insulin receptor.
    RFD3(
        inputs=str(FILES_DIR / "rfd3_config.json"),
        ckpt_path=None,
        n_batches=10,
        batch_size=8,
    ),
    # Remove designs with chain breaks or excessive clashes.
    Filter(
        name="rfd3_filter",
        filters={
            "rfd3.n_chainbreaks": MetricFilter(max=0),
        },
    ),
    # LigandMPNN — ligand-aware inverse folding (public dauparas/LigandMPNN).
    MPNN(
        model_type="ligand_mpnn",
        num_seq_per_pdb=4,
        keep_motif_sequences_fixed=True,
    ),
    # Validate with RosettaFold3 structure prediction.
    RF3(),
    # Check backbone RMSD between designed and predicted structures.
    RMSD(
        alignment_methods=[SELECTIONS.BKBN],
        rmsd_targets=[SELECTIONS.BKBN, SELECTIONS.MOTIF_ALLATOM],
    ),
    # Keep only designs that refold well with good interface metrics.
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
