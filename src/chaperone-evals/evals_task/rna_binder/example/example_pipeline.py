#!/usr/bin/env -S bash -c '"${CHAPERONE_REPO:-.}/.venv/bin/python" "$0" "$@"'
"""Generate 96 RNA-hairpin binder sequences with RFD3, LigandMPNN, and AF3.

The RNA target (`../inputs/target.pdb`, chain B) is held fixed while RFD3
diffuses a mini-protein around the loop face, LigandMPNN designs its sequence
with the RNA present as nucleic-acid context, and AF3 re-folds the complex so
the pool can be ranked before it is truncated to the submitted 96.

Production defaults diffuse 160 backbones and design one sequence per backbone.
The reference submission is deliberately *ranked*, not *filtered*: the top 96
designs are always submitted so the held-out thresholds stay calibratable, and
the number that actually cleared every design-side filter is printed.

Single-design smoke test:

    ./example_pipeline.py --required_n=1 --rundir=/tmp/rna_smoke \
        --rfd3.n_batches=1 --rfd3.batch_size=1 --nouse_local_gpus
"""

import subprocess
from pathlib import Path

import polars as pl
from absl import flags

from blocks import Blocks
from blocks.util_blocks.filter import MetricFilter
from chaperone import ExecutionEngine, init_cli

flags.DEFINE_integer(
    "required_n",
    96,
    "Expected sequence count. Override to 1 only for a smoke test.",
)
FLAGS = flags.FLAGS

init_cli()

SCRIPT_DIR = Path(__file__).resolve().parent
INPUTS_DIR = SCRIPT_DIR.parent / "inputs"
RFD3_CONFIG = (INPUTS_DIR / "config.json").resolve()

BINDER_CHAIN = "A"
TARGET_CHAIN = "B"

# One SLURM GPU for the whole benchmark. `rows_per_job` is set to the full row
# count of each step so nothing fans out across additional cards.
# 160 backbones for a 96-design submission: enough margin that the RFD3 filter
# and sequence de-duplication cannot leave the pool short.
N_BATCHES = 20
BATCH_SIZE = 8

GPU_ARGS = {
    "partition": "gpu",
    "gres": "gpu:small:1",
    "mem_gb": 32,
    "time_limit": "8:00:00",
}

# Design-side filters. These are the *pipeline's* own quality bar, chosen
# independently of the held-out scorer's thresholds; they rank the pool, they
# do not truncate it.
MIN_PLDDT = 80.0
MAX_MIN_PAE_INTERACTION = 6.0
MAX_LOOP_FRACTION = 0.45

RFD3, MPNN, AF3, Filter = Blocks.getall("rfd3", "mpnn", "af3", "filter")

rfd3 = RFD3(
    inputs=str(RFD3_CONFIG),
    n_batches=N_BATCHES,
    batch_size=BATCH_SIZE,
    seed_offset=42,
)
rfd3_filter = Filter(
    name="rfd3_filter",
    filters={
        "rfd3.n_chainbreaks": MetricFilter(max=0),
        "rfd3.non_loop_fraction": MetricFilter(min=0.2),
    },
)
# LigandMPNN sees the RNA as nucleic-acid context, so the designed surface is
# aware of the hairpin it has to read out. protein_mpnn would ignore it.
mpnn = MPNN(num_seq_per_pdb=1, model_type="ligand_mpnn")
af3 = AF3(name="rank_af3", diffusion_batch_size=1)

engine = ExecutionEngine(rundir=SCRIPT_DIR / "rundir", reinit=True)
backbones = engine.run(
    [rfd3, rfd3_filter],
    job_args={**GPU_ARGS, "rows_per_job": N_BATCHES},
)
sequenced = engine.run(
    mpnn, data=backbones, job_args={**GPU_ARGS, "rows_per_job": 1000}
)
folded = engine.run(af3, data=sequenced, job_args={**GPU_ARGS, "rows_per_job": 1000})

# ── Rank, then submit the top `required_n` ───────────────────────────────

frame = folded.unpack(all=True)
# Read the sequence off the *folded* handler, not the MPNN one: the AF3 block
# sorts its rows by length before chunking, so the two tables are not in the
# same order and zipping them would silently mislabel every design.
sequences = [
    loaded.sequence.get(BINDER_CHAIN)
    for loaded in folded.get_data(which="last", load=True)
]
if len(sequences) != len(frame):
    raise RuntimeError(
        f"Read {len(sequences)} sequences but the folded table has "
        f"{len(frame)} rows; the two are not aligned"
    )
missing = [i for i, s in enumerate(sequences) if not s]
if missing:
    raise ValueError(f"Designs {missing[:10]} have no chain-{BINDER_CHAIN} sequence")
frame["sequence"] = sequences

required = ["rank_af3.plddt", "rank_af3.min_pae_interaction", "rfd3.loop_fraction"]
for column in required:
    if column not in frame.columns:
        raise KeyError(f"Ranking column {column!r} missing from the pipeline output")
    if not frame[column].notna().all():
        raise ValueError(f"Ranking column {column!r} contains NaN; refusing to rank")

frame["cleared_filters"] = (
    (frame["rank_af3.plddt"] >= MIN_PLDDT)
    & (frame["rank_af3.min_pae_interaction"] <= MAX_MIN_PAE_INTERACTION)
    & (frame["rfd3.loop_fraction"] <= MAX_LOOP_FRACTION)
)
n_cleared = int(frame["cleared_filters"].sum())

frame = frame.drop_duplicates(subset="sequence")
# Best interface first, then best fold; the cleared-filter designs float to the
# top naturally, and anything below the bar is only used to top up to 96.
frame = frame.sort_values(
    ["cleared_filters", "rank_af3.min_pae_interaction", "rank_af3.plddt"],
    ascending=[False, True, False],
).reset_index(drop=True)

if len(frame) < FLAGS.required_n:
    raise RuntimeError(
        f"Only {len(frame)} unique sequences survived the pipeline; "
        f"{FLAGS.required_n} are required. Increase --rfd3.n_batches."
    )
top = frame.iloc[: FLAGS.required_n]

submission = pl.DataFrame(
    {
        "design_id": [f"rna_{i:04d}" for i in range(len(top))],
        "sequence": top["sequence"].tolist(),
    }
)
out_path = SCRIPT_DIR / "submission.csv"
submission.write_csv(out_path)
subprocess.run(["chmod", "-R", "g+w", str(engine.rundir)], check=False)
subprocess.run(["chmod", "g+w", str(out_path)], check=False)

print(engine)
print(submission)
print(
    f"\n{n_cleared} / {len(frame)} unique designs cleared every design-side "
    f"filter (plddt >= {MIN_PLDDT}, min_pae_interaction <= "
    f"{MAX_MIN_PAE_INTERACTION}, loop_fraction <= {MAX_LOOP_FRACTION}); "
    f"the submitted {len(submission)} are the best-ranked of the pool."
)
print(f"Wrote {len(submission)} RNA-binder sequences to {out_path}")
