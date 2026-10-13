#!/usr/bin/env -S bash -c '"${CHAPERONE_REPO:-.}/.venv/bin/python" "$0" "$@"'
"""Generate 96 PD-L1 binder sequences with RFD3, MPNN, and AF3 validation.

Production defaults generate 96 independent RFD3 backbones and one sequence
per backbone.  For the local one-design smoke test:

    ./example_pipeline.py --test --required_n=1 \
        --rundir=./rundir --local --use_local_gpus
"""

import subprocess
from pathlib import Path

import polars as pl
from absl import flags

from blocks import Blocks
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
TARGET_PDB = (INPUTS_DIR / "target.pdb").resolve()
RFD3_CONFIG = (INPUTS_DIR / "config.json").resolve()

RFD3, MPNN, AF3 = Blocks.getall("rfd3", "mpnn", "af3_with_target_template")

rfd3 = RFD3(
    inputs=str(RFD3_CONFIG),
    n_batches=12,
    batch_size=8,
    jitter_ori_sigma=1.0,
    seed_offset=42,
)
mpnn = MPNN(
    num_seq_per_pdb=1,
    model_type="protein_mpnn",
    keep_motif_sequences_fixed=True,
)
af3 = AF3(
    name="final",
    target_chain="B",
    full_receptor_pdb=str(TARGET_PDB),
    diffusion_batch_size=1,
)

engine = ExecutionEngine(rundir=SCRIPT_DIR / "rundir", reinit=True)
backbones = engine.run(rfd3)
sequenced = engine.run(mpnn, data=backbones)
engine.run(af3, data=sequenced)

records = []
for index, loaded in enumerate(sequenced.get_data(which="last", load=True)):
    sequence = loaded.sequence.get("A")
    if not sequence:
        raise ValueError(f"MPNN design {index} has no binder sequence on chain A")
    records.append(
        {
            "design_id": f"pdl1_{index:04d}",
            "sequence": sequence,
        }
    )

submission = pl.DataFrame(records)
if len(submission) != FLAGS.required_n:
    raise RuntimeError(
        f"Expected {FLAGS.required_n} sequences, found {len(submission)}. "
        "Do not submit a partial sequence table."
    )
if submission["sequence"].n_unique() != len(submission):
    raise ValueError("Submission contains duplicate binder sequences")

out_path = SCRIPT_DIR / "submission.csv"
submission.write_csv(out_path)
subprocess.run(["chmod", "-R", "g+w", str(engine.rundir)], check=True)
subprocess.run(["chmod", "g+w", str(out_path)], check=True)
print(engine)
print(submission)
print(f"\nWrote {len(submission)} PD-L1 binder sequences to {out_path}")
