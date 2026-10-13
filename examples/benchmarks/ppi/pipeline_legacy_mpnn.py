#!/usr/bin/env -S bash -c '"${PIPELINES_REPO:-.}/.venv/bin/python" "$0" "$@"'
"""PPI binder design using RFD3 + legacy (fused_mpnn) MPNN + RF3.

Sibling of ``pipeline_foundry.py`` in this directory that swaps
Foundry MPNN for the legacy ``mpnn`` block (fused_mpnn /
ProteinMPNN+LigandMPNN fork). Same binder target (human insulin
receptor, 4ZXB) and shared ``files/`` so metrics between the two
inverse-folding backends are directly comparable on identical RFD3
scaffolds and RF3 validation.

Pipeline: rfd3 -> filter -> mpnn (legacy) -> rf3 -> rmsd -> filter

Cloud runs:
    PIPELINES_COMPUTE_BACKEND=modal \
    FOUNDRY_CHECKPOINT_DIRS=/abs/path/to/foundry/weights \
    MPNN_WEIGHTS_DIR=/abs/path/to/fused_mpnn_weights \
    ./examples/benchmarks/ppi/pipeline_legacy_mpnn.py \
        --config=tests/test_overrides.yaml

The legacy mpnn block auto-selects the on-prem SIF when it exists on
disk, otherwise falls back to the cloud-buildable ``modelhub_cloud``
variant (git_clone of baker-laboratory/fused_mpnn). See
``src/blocks/networks/mpnn.py`` for the selection logic and
``docs/cloud.md`` for the full cloud setup.
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
    # Legacy MPNN — ligand-aware inverse folding. Uses fused_mpnn's
    # LigandMPNN model via the modelhub container (on-prem SIF or the
    # cloud-buildable variant, auto-selected at import time).
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
