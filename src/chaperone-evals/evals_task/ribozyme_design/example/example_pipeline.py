#!/usr/bin/env -S bash -c '"${CHAPERONE_REPO:-.}/.venv/bin/python" "$0" "$@"'
"""Reference submission for the hammerhead rescaffolding eval.

Hands RFdiffusion3NA only the six catalytic nucleotides of the full-length
Schistosoma hammerhead (PDB 2OEU) plus its active-site Mn2+, unindexed, and
asks it to build a new fixed-length RNA around them. Survivors are refolded
with AF3 and ranked on the crystal-referenced metrics; the top `required_n`
are written as the submission.

    rfd3na -> connectivity+placement filter -> af3 -> metrics -> rank -> submit

The submission is deliberately *ranked*, not *filtered*: the top designs are
always submitted so the held-out thresholds stay calibratable, and the number
that actually cleared every design-side filter is printed.

Smoke test:
    ./example_pipeline.py --required_n=1 --rundir=/tmp/rz01_smoke \
        --rfd3na.n_batches=1 --rfd3na.batch_size=1
"""

import json
import shutil
import subprocess
from pathlib import Path

import polars as pl
from absl import flags

from blocks import Blocks
from blocks.metrics.sequence_identity_gate import nw_identity
from blocks.networks.foundry.rfd3na_input import build_motif_spec, write_rfd3na_json
from blocks.util_blocks.filter import MetricFilter
from chaperone import ExecutionEngine, init_cli

flags.DEFINE_integer("required_n", 96, "Designs to submit. Override to 1 to smoke test.")
FLAGS = flags.FLAGS

init_cli()

SCRIPT_DIR = Path(__file__).resolve().parent
INPUTS_DIR = SCRIPT_DIR.parent / "inputs"
MOTIF = (INPUTS_DIR / "2oeu_motif.pdb").resolve()

# 2OEU -> hammerhead numbering: A20=G8, A21=A9, A22=G10.1, A36=G12,
# B6=C17 (nucleophile), B7=C1.1 (leaving group). A20-22 are tied because they
# are a contiguous trinucleotide in the native; the other groups are free to
# land anywhere. All six come from two native strands but are rescaffolded into
# a single chain, i.e. a cis-acting ribozyme.
#
# Groups are separated by commas, NEVER "/0". Inside `unindex` a "/0" puts the
# groups on separate molecules and the model then declines to bond them into
# the designed chain: measured over 105 designs, that one character moves the
# median insertion RMSD from 3.63 A to 0.35 A.
UNINDEX = ["A20-22", "A36", "B6-7"]
LIGAND = "MN"
LENGTH = "60-60"
DESIGN_CHAIN = "A"
MAX_PAIRWISE_IDENTITY = 0.70

CATALYTIC = {
    "general_base_dist": ["A36.N1", "B6.O2'"],  # native 3.26 A
    "general_acid_dist": ["A20.O2'", "B7.O5'"],  # native 3.51 A
    "inline_attack_dist": ["B6.O2'", "B7.P"],  # native 3.30 A
    "inline_attack_angle": ["B6.O2'", "B7.P", "B7.O5'"],  # native 152.0 deg
    "metal_a9_dist": ["L:MN.MN", "A21.OP2"],  # native 2.01 A
    "metal_g10_dist": ["L:MN.MN", "A22.N7"],  # native 2.08 A
}

# 79.3% of diffused designs clear insertion_rmsd <= 1 A and only ~1% of those
# refold well, so oversample hard: the refold is the expensive step and the
# filters below are what make it affordable.
N_BATCHES = 40
BATCH_SIZE = 25

AF3_GPU_ARGS = {
    "partition": "gpu",
    "gres": "gpu:a4000:1",
    "mem_gb": 32,
    "time_limit": "8:00:00",
}

RFD3NA, AF3, Connectivity, RefMotifRMSD, MotifGeometry, Filter = Blocks.getall(
    "rfd3na", "af3", "rna_connectivity", "reference_motif_rmsd",
    "motif_geometry", "filter",
)

engine = ExecutionEngine(rundir=FLAGS.rundir or SCRIPT_DIR / "rundir")

spec_json = write_rfd3na_json(
    {
        "hammerhead": build_motif_spec(
            MOTIF, UNINDEX, nucleic="RNA", length=LENGTH, ligand=LIGAND
        )
    },
    engine.rundir / "rfd3na_input.json",
)

backbones = engine.run(
    [
        RFD3NA(inputs=spec_json, n_batches=N_BATCHES, batch_size=BATCH_SIZE),
        # Real O3'-P connectivity, measured on the diffusion output. This is the
        # scarcest criterion in the whole protocol (26% at best) and is NOT what
        # rfd3na.n_chainbreaks reports - that metric uses protein backbone
        # geometry and is meaningless for RNA.
        Connectivity(name="connectivity", which=[0]),
        Filter(
            name="placement_filter",
            filters={
                "connectivity.n_breaks": MetricFilter(max=0),
                "rfd3na.insertion_rmsd": MetricFilter(max=1.0),
                "rfd3na.n_conjoined_residues": MetricFilter(max=0),
                "rfd3na.n_clashing.interresidue_clashes_w_backbone": MetricFilter(max=0),
                "rfd3na.n_clashing.interresidue_clashes_w_sidechain": MetricFilter(max=0),
            },
        ),
    ],
    # Keep RFD3NA's gpu-bf/gpu:small default. Harbor accepts this exact
    # Pipelines-native pair and maps it to an A4000-constrained allocation.
    job_args={"rows_per_job": 4},
)

# MN is a CCD code, so AF3 carries the ion through without a SMILES override.
# ligands=[] EXCLUDES it from the RMSD: AF3 lands this ion ~36 A off even on the
# native, which alone turns a 0.49 A control into 3.08 A.
folded = engine.run(
    [
        AF3(),
        RefMotifRMSD(reference=str(MOTIF), ligands=[]),
        MotifGeometry(measurements=CATALYTIC),
    ],
    data=backbones,
    job_args={**AF3_GPU_ARGS, "rows_per_job": 200},
)

# ── Rank, then submit the top `required_n` ───────────────────────────────

frame = folded.unpack(all=True)

# The motif map is read off the SAME row as the sequence. Do not zip two
# separately-ordered tables: the AF3 block sorts rows by length before
# chunking, so a positional join silently mislabels every design.
sequences, motif_maps = [], []
for loaded in folded.get_data(which=[0], load=True):
    design = loaded[0] if isinstance(loaded, tuple) else loaded
    index_map = json.loads(design.metadata["diffused_index_map"])
    sequences.append(design.sequence[DESIGN_CHAIN])
    motif_maps.append(
        json.dumps(
            {
                reference: int(str(placed)[1:]) - 1  # design residue id -> 0-based
                for reference, placed in index_map.items()
                if not reference.startswith("L:")
            }
        )
    )
if len(sequences) != len(frame):
    raise RuntimeError(
        f"Read {len(sequences)} sequences but the folded table has {len(frame)} rows"
    )
frame["sequence"] = sequences
frame["motif_map"] = motif_maps

for column in ("af3.plddt", "reference_motif_rmsd.motif_rmsd",
               "motif_geometry.inline_attack_dist",
               "motif_geometry.inline_attack_angle"):
    if column not in frame.columns:
        raise KeyError(f"Ranking column {column!r} missing from the pipeline output")

frame["cleared_filters"] = (
    (frame["reference_motif_rmsd.motif_rmsd"] <= 2.0)
    & (frame["motif_geometry.inline_attack_dist"] <= 4.0)
    & (frame["motif_geometry.inline_attack_angle"] >= 140.0)
    & (frame["af3.plddt"] >= 70.0)
)
n_cleared = int(frame["cleared_filters"].sum())

frame = frame.drop_duplicates(subset="sequence")
# Best site reconstruction first; the designs that cleared every gate float to
# the top naturally and anything below the bar only tops the submission up.
frame = frame.sort_values(
    ["cleared_filters", "reference_motif_rmsd.motif_rmsd", "af3.plddt"],
    ascending=[False, True, False],
).reset_index(drop=True)

selected_indices = []
selected_sequences = []
for index, sequence in zip(frame.index, frame["sequence"]):
    if all(
        nw_identity(sequence, selected) <= MAX_PAIRWISE_IDENTITY
        for selected in selected_sequences
    ):
        selected_indices.append(index)
        selected_sequences.append(sequence)
    if len(selected_indices) == FLAGS.required_n:
        break

if len(selected_indices) < FLAGS.required_n:
    raise RuntimeError(
        f"Only {len(selected_indices)} pairwise-diverse designs survived from "
        f"{len(frame)} unique candidates; {FLAGS.required_n} are required. "
        "Increase --rfd3na.n_batches."
    )
top = frame.loc[selected_indices]

submission = pl.DataFrame(
    {
        "design_id": [f"rz01_{i:04d}" for i in range(len(top))],
        "sequence": top["sequence"].tolist(),
        "motif_map": top["motif_map"].tolist(),
    }
)
out_path = SCRIPT_DIR / "submission.csv"
submission.write_csv(out_path)
top_design_path = SCRIPT_DIR / "top_design.cif"
shutil.copyfile(top.iloc[0]["af3.path"], top_design_path)
subprocess.run(["chmod", "-R", "g+w", str(engine.rundir)], check=False)

print(engine)
print(submission)
print(
    f"\n{n_cleared} / {len(frame)} unique designs cleared every design-side "
    f"gate; the submitted {len(submission)} are the best-ranked of the pool."
)
print(f"Wrote {len(submission)} ribozyme designs to {out_path}")
print(f"Wrote the top-ranked refold to {top_design_path}")
