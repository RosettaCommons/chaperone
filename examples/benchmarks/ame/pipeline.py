#!/usr/bin/env -S bash -c '"${CHAPERONE_REPO:-.}/.venv/bin/python" "$0" "$@"'
"""Active-site Motif Evaluation (AME) benchmark pipeline.

Evaluates de novo enzyme design quality by measuring how well AF3 recovers
the catalytic motif geometry after backbone generation and sequence design.
Runs AME-1 and optionally a user-supplied AME-2 suite:

  AME-1: 41 MCSA enzyme active sites (dialect-1 format, fixed-length 180 res)
  AME-2: 4 enzyme tasks — glycosidase, aldolase, decarboxylase, phosphotriesterase

Pipeline (per suite):
  rfd3 -> [concat] -> rotamer_metrics -> ligand_mpnn -> af3 -> rmsd (tip-atom)

Tip-atom RMSD (motif_allatom) after AF3 folding is the primary success metric.
"""

from pathlib import Path

from blocks import Blocks
from chaperone import ExecutionEngine, get_runtime_overrides, init_cli
from chaperone.data.schema import SELECTIONS

from absl import flags

flags.DEFINE_string("ame2_inputs", None, "Optional authorized AME-2 JSON with paths to your motif structures.")
init_cli()

SCRIPT_DIR = Path(__file__).resolve().parent
FILES_DIR = SCRIPT_DIR / "files"
TEST_MODE = get_runtime_overrides().get("test", False)

RFD3, MPNN, AF3, RMSD, RotamerMetrics = Blocks.getall(
    "rfd3", "mpnn", "af3", "rmsd", "rotamer_metrics"
)

# ── Block configuration ─────────────────────────────────────────────────

rfd3_ame1 = RFD3(
    name="rfd3/ame1",
    inputs=str(FILES_DIR / "ame-1.json"),
    json_keys_subset=["M0024_1nzy"] if TEST_MODE else [],
    n_batches=10,
    batch_size=10,
)

rfd3_ame2 = None
if flags.FLAGS.ame2_inputs:
    rfd3_ame2 = RFD3(
        name="rfd3/ame2",
        inputs=flags.FLAGS.ame2_inputs,
        json_keys_subset=["dera_aldolase"] if TEST_MODE else [],
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

# Keep CPU rotamer work separate from GPU prediction allocations.
handler = engine.run(rfd3_ame1)
if rfd3_ame2 is not None:
    handler = handler + engine.run(rfd3_ame2)
for block in shared_blocks:
    handler = engine.run(block, data=handler)

df = handler.unpack()
print(df)
