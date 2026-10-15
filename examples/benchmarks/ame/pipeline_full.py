#!/usr/bin/env -S bash -c '"${CHAPERONE_REPO:-.}/.venv/bin/python" "$0" "$@"'
"""AME-1 benchmark — full production run.

Runs AME-1 (41 MCSA active sites) through:
  rfd3 (1 * 100 = 100 designs/site, fanned out to 100 SLURM jobs)
    -> rotamer_metrics -> ligand_mpnn (8 seqs/design)
    -> af3 (with ligand) -> rmsd

Backbone-aligned RMSDs are computed for backbone, motif tip-atoms, and
ligand atoms (AF3 keeps ligands by default — `remove_ligands=False`).

Shared blocks are split into per-block engine.run calls so each block
uses its own _resources (RotamerMetrics→CPU, MPNN→GPU, AF3→GPU, RMSD→CPU).
A single chained engine.run picks up the leading block's CPU resources,
which would starve AF3 of a GPU.
"""

from pathlib import Path

from blocks import Blocks
from chaperone import ExecutionEngine, get_runtime_overrides, init_cli
from chaperone.data.schema import SELECTIONS

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
    n_batches=100,
    batch_size=1,
)

rotamer_metrics = RotamerMetrics(
    backend="dunbrack",
    motif_only=True,
)

mpnn = MPNN(
    model_type="ligand_mpnn",
    keep_motif_sequences_fixed=True,
    num_seq_per_pdb=8,
    args={"omit_AA": "XC"},
)

af3 = AF3(
    read_most_confident_model=True,
)

rmsd = RMSD(
    which=[0, -1],
    alignment_methods=[SELECTIONS.BKBN],
    rmsd_targets=[SELECTIONS.BKBN, SELECTIONS.MOTIF_ALLATOM, SELECTIONS.LIGANDS],
    select_mask_by_name_pairing=[
        SELECTIONS.BKBN,
        SELECTIONS.MOTIF_ALLATOM,
        SELECTIONS.LIGANDS,
    ],
)

# ── Execute ──────────────────────────────────────────────────────────────

engine = ExecutionEngine(
    rundir=SCRIPT_DIR / "rundir_full",
    reinit=True,
)

rfd3_job_args = {"rows_per_job": 1}
data = engine.run(rfd3_ame1, job_args=rfd3_job_args)
data = engine.run(rotamer_metrics, data=data)
data = engine.run(mpnn, data=data)
data = engine.run(af3, data=data)
handler = engine.run(rmsd, data=data)

df = handler.unpack()
print(df)
