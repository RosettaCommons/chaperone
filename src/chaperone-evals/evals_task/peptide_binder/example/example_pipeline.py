#!/usr/bin/env -S bash -c '"${CHAPERONE_REPO:-.}/.venv/bin/python" "$0" "$@"'
"""Generate 96 EIN2 site-1 peptide binder sequences with RFD3, MPNN, and AF3.

Reference implementation for the ``peptide-binder`` eval. RFD3 diffuses a
binder around the *fixed* peptide pose in ``inputs/target.pdb``, MPNN designs
one sequence per backbone with the peptide held fixed, and AF3 refolds the
complex. Designs are then filtered on the pipeline's own (non-held-out)
criteria and ranked by interface PAE; if fewer than 96 clear every filter the
table is topped up with the best-ranked remainder, and the number that actually
cleared is printed. The submission is a calibration reference — it is not
required to pass the held-out scorer.

Production (one SLURM GPU):

    ./example_pipeline.py --nouse_local_gpus

One-design smoke test:

    ./example_pipeline.py --required_n=1 --rfd3.n_batches=1 --rfd3.batch_size=1 \
        --rundir=/tmp/peptide_binder_smoke --nouse_local_gpus
"""

import subprocess
from pathlib import Path

import polars as pl
from absl import flags

from blocks import Blocks
from chaperone import ExecutionEngine, init_cli
from chaperone.data.schema import SELECTIONS

flags.DEFINE_integer(
    "required_n",
    96,
    "Expected sequence count. Override to 1 only for a smoke test.",
)
FLAGS = flags.FLAGS

init_cli()

SCRIPT_DIR = Path(__file__).resolve().parent
INPUTS_DIR = SCRIPT_DIR.parent / "inputs"
TARGET_PDB = (INPUTS_DIR / "target.pdb").resolve()
RFD3_CONFIG = (INPUTS_DIR / "config.json").resolve()

BINDER_CHAIN = "A"
TARGET_CHAIN = "B"

# Pipeline-side filters. These are the designer's own criteria — deliberately
# not the held-out scorer's thresholds.
IPTM_MIN = 0.80
MIN_PAE_MAX = 2.0
SELF_CONSISTENCY_RMSD_MAX = 2.0

GPU_ARGS = {
    "partition": "gpu",
    "gres": "gpu:small:1",
    "mem_gb": 32,
    "time_limit": "8:00:00",
    "rows_per_job": 10_000,  # one chunk, one GPU
}

RFD3, MPNN, AF3, RMSD = Blocks.getall("rfd3", "mpnn", "af3", "rmsd")

# 144 candidates for 96 slots: enough headroom to absorb MPNN sequence
# collisions and still rank rather than merely collect.
rfd3 = RFD3(
    inputs=str(RFD3_CONFIG),
    n_batches=18,
    batch_size=8,
    jitter_ori_sigma=1.0,
    seed_offset=42,
)
design_and_fold = [
    MPNN(
        num_seq_per_pdb=1,
        model_type="protein_mpnn",
        keep_motif_sequences_fixed=True,  # the peptide is the motif; never redesign it
    ),
    AF3(diffusion_batch_size=1),
    RMSD(
        name="sc_rmsd",
        which=[0, -1],
        alignment_methods=[SELECTIONS.BKBN],
        rmsd_targets=[SELECTIONS.BKBN],
    ),
]

engine = ExecutionEngine(rundir=SCRIPT_DIR / "rundir", reinit=False)
backbones = engine.run(rfd3, job_args={**GPU_ARGS, "rows_per_job": rfd3.n_batches})
folded = engine.run(design_and_fold, data=backbones, job_args=GPU_ARGS)

frame = folded.unpack(all=True)
required = ["af3.iptm", "af3.min_pae_interaction", "sc_rmsd.bkbn_align.bkbn_rmsd"]
missing = [column for column in required if column not in frame.columns]
if missing:
    raise RuntimeError(f"pipeline output is missing columns {missing}")
if frame[required].isna().any().any():
    raise RuntimeError(
        "NaN metrics in the design run; refusing to rank designs on missing data:\n"
        f"{frame[required].isna().sum()}"
    )

sequences = []
for index, loaded in enumerate(folded.get_data(which="last", load=True)):
    sequence = loaded.sequence.get(BINDER_CHAIN)
    if not sequence:
        raise ValueError(
            f"design {index} has no binder sequence on chain {BINDER_CHAIN}"
        )
    sequences.append(sequence)
frame["sequence"] = sequences

# One row per unique binder sequence, keeping its best-ranked instance.
frame = frame.sort_values("af3.min_pae_interaction").drop_duplicates("sequence")

cleared = (
    (frame["af3.iptm"] >= IPTM_MIN)
    & (frame["af3.min_pae_interaction"] <= MIN_PAE_MAX)
    & (frame["sc_rmsd.bkbn_align.bkbn_rmsd"] <= SELF_CONSISTENCY_RMSD_MAX)
)
n_cleared = int(cleared.sum())
print(
    f"{len(frame)} unique designs; {n_cleared} cleared every pipeline filter "
    f"(iptm>={IPTM_MIN}, min_pae<={MIN_PAE_MAX}, "
    f"bkbn_rmsd<={SELF_CONSISTENCY_RMSD_MAX})"
)
if n_cleared < FLAGS.required_n:
    print(
        f"Only {n_cleared} cleared; topping up to {FLAGS.required_n} with the "
        "best-ranked remaining designs (interface PAE ascending)."
    )

# Filter-passers first, then the best of the rest — both already PAE-sorted.
ordered = pl.concat(
    [
        pl.from_pandas(frame[cleared][["sequence"]]),
        pl.from_pandas(frame[~cleared][["sequence"]]),
    ]
)
if len(ordered) < FLAGS.required_n:
    raise RuntimeError(
        f"Only {len(ordered)} unique sequences available, need {FLAGS.required_n}. "
        "Increase --rfd3.n_batches rather than padding the table."
    )

submission = ordered.head(FLAGS.required_n).with_row_index("row")
submission = submission.select(
    pl.format("ein2_{}", pl.col("row").cast(pl.Utf8).str.zfill(4)).alias("design_id"),
    pl.col("sequence"),
)
if submission["sequence"].n_unique() != len(submission):
    raise ValueError("Submission contains duplicate binder sequences")

out_path = SCRIPT_DIR / "submission.csv"
submission.write_csv(out_path)
subprocess.run(["chmod", "-R", "g+w", str(engine.rundir)], check=True)
subprocess.run(["chmod", "g+w", str(out_path)], check=True)
print(engine)
print(submission)
print(
    f"\nWrote {len(submission)} EIN2 site-1 binder sequences to {out_path} "
    f"({n_cleared} of them cleared every pipeline filter)"
)
