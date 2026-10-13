#!/usr/bin/env -S bash -c '"${CHAPERONE_REPO:-.}/.venv/bin/python" "$0" "$@"'
"""Reference pipeline for the `dna-binder` eval — 96 major-groove DNA binders.

Diffuses protein backbones around the supplied dsDNA duplex (`inputs/target.pdb`,
chains B/C) with major-groove hbond conditioning, designs sequences with
LigandMPNN (DNA is the ligand context), refolds with AF3, and writes the 96
best-ranked unique sequences to `submission.csv`.

Ranking is by AF3 ipTM, deliberately **not** by `af3.min_pae_interaction`:
with three chains present, that metric is the minimum over *all* chain pairs
and is dominated by the DNA:DNA duplex pair (~1.5 Å even for designs that never
touch the DNA). See `../score/readme.md`.

This is a *reference* implementation. It is not tuned to pass the held-out
scorer; it exists so the scorer's thresholds are calibratable.

Smoke test (one design, SLURM GPU):

    ./example_pipeline.py --test --required_n=1 \
        --rundir=/tmp/dna_binder_smoke --nouse_local_gpus

Production (96 sequences, one SLURM GPU):

    ./example_pipeline.py --nouse_local_gpus
"""

import re
import subprocess
from pathlib import Path

import polars as pl
from absl import flags

from blocks import Blocks
from blocks.util_blocks.filter import MetricFilter
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
RFD3_CONFIG = (INPUTS_DIR / "config.json").resolve()

CANONICAL_AA = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
MIN_BINDER_LENGTH = 40

# One SLURM GPU for the whole campaign: `rows_per_job` is set high enough that
# every stage runs as a single chunk rather than fanning out.
GPU_ARGS = {
    "partition": "gpu",
    "gres": "gpu:small:1",
    "mem_gb": 32,
    "time_limit": "8:00:00",
    "rows_per_job": 4096,
}

RFD3, MPNN, AF3, RMSD, Filter = Blocks.getall("rfd3", "mpnn", "af3", "rmsd", "filter")

# 128 backbones × 2 sequences leaves headroom for the post-diffusion filter and
# for duplicate sequences, so 96 unique designs survive to the submission.
design = [
    RFD3(
        inputs=str(RFD3_CONFIG),
        n_batches=16,
        batch_size=8,
        inference_sampler={"step_scale": 1.5, "cfg_scale": 1.0},
        seed_offset=42,
    ),
    Filter(
        name="rfd3_filter",
        filters={
            "rfd3.n_chainbreaks": MetricFilter(max=0),
            "rfd3.non_loop_fraction": MetricFilter(min=0.2),
        },
    ),
    MPNN(name="mpnn", num_seq_per_pdb=2, model_type="ligand_mpnn"),
    AF3(name="af3"),
    RMSD(
        name="af3_rmsd",
        alignment_methods=[SELECTIONS.BKBN],
        rmsd_targets=[SELECTIONS.BKBN],
    ),
]

engine = ExecutionEngine(rundir=SCRIPT_DIR / "rundir", reinit=True)
scored = engine.run(design, job_args=GPU_ARGS)

# ── Rank and select ──────────────────────────────────────────────────────

frame = scored.unpack(all=True)


def binder_sequence(loaded) -> str:
    """The designed protein chain of a design, whatever chain ID RFD3 gave it."""
    candidates = [
        (chain, seq)
        for chain, seq in (loaded.sequence or {}).items()
        if isinstance(seq, str)
        and len(seq) >= MIN_BINDER_LENGTH
        and CANONICAL_AA.fullmatch(seq)
    ]
    if not candidates:
        raise ValueError(
            f"No protein chain found in {loaded.path}; chains="
            f"{sorted((loaded.sequence or {}).keys())}"
        )
    return max(candidates, key=lambda item: len(item[1]))[1]


sequences = [
    binder_sequence(loaded) for loaded in scored.get_data(which="last", load=True)
]
if len(sequences) != len(frame):
    raise RuntimeError(
        f"Recovered {len(sequences)} sequences for {len(frame)} rows; refusing to "
        "align them by position."
    )
frame["sequence"] = sequences

for column in ("af3.iptm", "af3.plddt", "af3_rmsd.bkbn_align.bkbn_rmsd"):
    if column not in frame.columns:
        raise RuntimeError(f"Pipeline output is missing {column}")
    if frame[column].isna().any():
        raise RuntimeError(f"{column} has NaN values; refusing to rank on it")

# Design-time quality flags. They order the table but never truncate it: the
# submission must carry exactly `required_n` rows even when few designs are good.
frame["design_pass"] = (
    (frame["af3.iptm"] >= 0.6)
    & (frame["af3.plddt"] >= 80.0)
    & (frame["af3_rmsd.bkbn_align.bkbn_rmsd"] <= 3.0)
)

ranked = (
    frame.sort_values(["design_pass", "af3.iptm"], ascending=[False, False])
    .drop_duplicates(subset="sequence", keep="first")
    .reset_index(drop=True)
)

n_available = len(ranked)
n_design_pass = int(ranked["design_pass"].sum())
if n_available < FLAGS.required_n:
    raise RuntimeError(
        f"Only {n_available} unique sequences available, need {FLAGS.required_n}. "
        "Increase rfd3.n_batches or mpnn.num_seq_per_pdb."
    )

selected = ranked.head(FLAGS.required_n)
n_topped_up = max(0, FLAGS.required_n - n_design_pass)

submission = pl.DataFrame(
    {
        "design_id": [f"dna_{index:04d}" for index in range(len(selected))],
        "sequence": selected["sequence"].tolist(),
    }
)
if submission["sequence"].n_unique() != len(submission):
    raise ValueError("Submission contains duplicate binder sequences")

out_path = SCRIPT_DIR / "submission.csv"
submission.write_csv(out_path)
subprocess.run(["chmod", "-R", "g+w", str(engine.rundir)], check=False)
subprocess.run(["chmod", "g+w", str(out_path)], check=False)

print(engine)
print(submission)
print(
    f"\n{len(frame)} designs folded, {n_available} unique sequences, "
    f"{n_design_pass} cleared the design-time flags "
    f"(af3.iptm >= 0.6, af3.plddt >= 80, backbone RMSD <= 3 A)."
)
if n_topped_up:
    print(
        f"Only {n_design_pass} designs cleared those flags, so the table was "
        f"topped up with the {n_topped_up} best-ranked remaining designs by "
        "af3.iptm. The reference submission is deliberately not filtered down "
        "to a passing set."
    )
print(f"Wrote {len(submission)} DNA-binder sequences to {out_path}")
