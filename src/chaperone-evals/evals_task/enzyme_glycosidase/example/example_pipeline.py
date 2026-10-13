#!/usr/bin/env -S bash -c '"${CHAPERONE_REPO:-.}/.venv/bin/python" "$0" "$@"'
"""Generate 96 de novo glycosidase sequences plus their catalytic motif maps.

RFdiffusion3 scaffolds the six-residue theozyme of ``inputs/theozyme.pdb``
around the ``L:1`` substrate, a strict backbone filter removes broken or
clashing scaffolds, and LigandMPNN designs sequences with the catalytic
residues held fixed.

The submission carries ``motif_index_map`` because a sequence-only table gives
the scorer no way to know which residues are catalytic. The map is *not*
hand-written: RFD3 records where each theozyme position landed in the diffused
scaffold as ``metadata["diffused_index_map"]`` (``{"A86": "A45", ...}``), and
this script renames those keys to the benchmark's reference IDs (``E86``) and
emits the CSV spelling via ``format_motif_index_map``.

The whole run is sized to fit one GPU as a single chunk.

One-design smoke test (uses SLURM; drop ``--nouse_local_gpus`` to use a free
local GPU):

    ./example_pipeline.py --required_n=1 --rfd3.n_batches=1 \
        --rfd3.batch_size=1 --mpnn.num_seq_per_pdb=1 \
        --rundir=/tmp/glyco_smoke --nouse_local_gpus

Production (96 designs, one SLURM GPU):

    ./example_pipeline.py --nouse_local_gpus
"""

from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from pathlib import Path

import polars as pl
from absl import flags

from blocks import Blocks
from blocks.util_blocks.filter import MetricFilter
from chaperone import ExecutionEngine, init_cli
from scoring import THREE_TO_ONE, format_motif_index_map

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
SUBMISSION = SCRIPT_DIR / "submission.csv"

# theozyme residue key (as RFD3 sees it) -> benchmark reference position ID.
# Read directly off inputs/theozyme.pdb; keep in sync with score/score.py.
REFERENCE_ID_BY_THEOZYME_RESIDUE = {
    "A77": "Y77",
    "A86": "E86",
    "A88": "Y88",
    "A122": "R122",
    "A136": "Q136",
    "A177": "E177",
}
DESIGN_CHAIN = "A"

# One GPU, one chunk: rows_per_job is set above the row count of each step so
# the engine never fans out across nodes.
GPU_ARGS = {
    "partition": "gpu",
    "gres": "gpu:small:1",
    "mem_gb": 32,
    "time_limit": "8:00:00",
    "rows_per_job": 10_000,
}

RFD3, Filter, MPNN = Blocks.getall("rfd3", "filter", "mpnn")

blocks = [
    # 24 batches x 8 designs x 2 input specs ("ori" and "full") = 384 backbones.
    RFD3(inputs=str(RFD3_CONFIG), n_batches=24, batch_size=8, seed_offset=7),
    Filter(
        name="rfd3_filter",
        filters={
            "rfd3.n_chainbreaks": MetricFilter(max=0),
            "rfd3.n_clashing.ligand_clashes": MetricFilter(max=0),
            "rfd3.non_loop_fraction": MetricFilter(min=0.2),
            "rfd3.n_clashing.ligand_min_distance": MetricFilter(min=2.0),
            # The core example uses 0.5 Å here. On this theozyme that keeps
            # almost nothing (0/2 in the smoke test, where the better design
            # scored 0.88 Å), and a reference pipeline that cannot fill 96 rows
            # is useless. 1.0 Å still rejects a motif that was not really
            # rebuilt, and matches the scorer's motif-RMSD gate.
            "rfd3.insertion_rmsd": MetricFilter(max=1.0),
            "rfd3.n_conjoined_residues": MetricFilter(max=0),
        },
    ),
    MPNN(model_type="ligand_mpnn", num_seq_per_pdb=2, keep_motif_sequences_fixed=True),
]


def motif_index_map(design, diffused_index_map) -> dict[str, str]:
    """Rename RFD3's theozyme->scaffold map to the benchmark's contract.

    Two translations are needed, and skipping either silently produces a map
    that fails the scorer's residue-type check:

    1. Keys: RFD3 names motif positions by their theozyme residue ID (``A86``);
       the benchmark names them by residue type + number (``E86``).
    2. Values: RFD3's values are residue *IDs* of the diffused backbone, which
       are not guaranteed to start at 1 or to be contiguous. The submission
       contract indexes into ``sequence``, so each value is converted to a
       1-based position in that sequence.
    """
    from biotite.structure import get_residue_starts

    if isinstance(diffused_index_map, str):
        diffused_index_map = json.loads(diffused_index_map)
    if not diffused_index_map:
        raise ValueError("RFD3 design carries no diffused_index_map")

    sequence = design.sequence.get(DESIGN_CHAIN)
    if not sequence:
        raise ValueError(f"{design.path} has no chain {DESIGN_CHAIN} sequence")
    chain = design.atom_array[design.atom_array.chain_id == DESIGN_CHAIN]
    residue_ids = [int(r) for r in chain.res_id[get_residue_starts(chain)]]
    if len(residue_ids) != len(sequence):
        raise ValueError(
            f"chain {DESIGN_CHAIN} has {len(residue_ids)} residues but "
            f"{len(sequence)} sequence letters"
        )
    position_of = {residue_id: i + 1 for i, residue_id in enumerate(residue_ids)}

    mapping: dict[str, str] = {}
    for residue, reference in REFERENCE_ID_BY_THEOZYME_RESIDUE.items():
        if residue not in diffused_index_map:
            raise ValueError(
                f"diffused_index_map {diffused_index_map} has no entry for "
                f"theozyme position {residue} ({reference})"
            )
        target = str(diffused_index_map[residue])
        if target[0] != DESIGN_CHAIN:
            raise ValueError(f"{reference} landed on chain {target[0]}, not A")
        position = position_of.get(int(target[1:]))
        if position is None:
            raise ValueError(f"{reference} maps to {target}, absent from the design")
        expected = THREE_TO_ONE[reference_residue_name(reference)]
        if sequence[position - 1] != expected:
            raise ValueError(
                f"{reference} maps to A{position} which is "
                f"{sequence[position - 1]}, expected {expected}"
            )
        mapping[reference] = f"{DESIGN_CHAIN}{position}"
    return mapping


def reference_residue_name(reference: str) -> str:
    """Three-letter name of a reference position, from its one-letter prefix."""
    one_letter = reference[0]
    for three, one in THREE_TO_ONE.items():
        if one == one_letter:
            return three
    raise ValueError(f"unknown reference position {reference!r}")


def main() -> None:
    engine = ExecutionEngine(rundir=SCRIPT_DIR / "rundir", reinit=False)
    designed = engine.run(blocks, job_args=GPU_ARGS)

    # Group by backbone so the submission spreads across scaffolds before it
    # takes a second sequence from any one of them.
    by_backbone: dict[str, list[dict]] = defaultdict(list)
    order: list[str] = []
    for index, (backbone, design) in enumerate(
        designed.get_data(which=[0, -1], load=True)
    ):
        sequence = design.sequence.get("A")
        if not sequence:
            raise ValueError(f"MPNN design {index} has no chain A sequence")
        if backbone.path not in by_backbone:
            order.append(backbone.path)
        by_backbone[backbone.path].append(
            {
                "sequence": sequence,
                "motif_index_map": format_motif_index_map(
                    motif_index_map(design, backbone.metadata.get("diffused_index_map"))
                ),
            }
        )

    records: list[dict] = []
    seen: set[str] = set()
    depth = 0
    while len(records) < FLAGS.required_n and any(
        len(by_backbone[path]) > depth for path in order
    ):
        for path in order:
            if len(records) >= FLAGS.required_n:
                break
            if len(by_backbone[path]) <= depth:
                continue
            candidate = by_backbone[path][depth]
            if candidate["sequence"] in seen:
                continue
            seen.add(candidate["sequence"])
            records.append({"design_id": f"glyco_{len(records):04d}", **candidate})
        depth += 1

    if len(records) != FLAGS.required_n:
        raise RuntimeError(
            f"Only {len(records)} unique sequences available from "
            f"{len(order)} filtered backbones; need {FLAGS.required_n}. "
            "Scale up the run (--rfd3.n_batches)."
        )

    submission = pl.DataFrame(records)
    if submission["sequence"].n_unique() != len(submission):
        raise ValueError("Submission contains duplicate sequences")
    submission.write_csv(SUBMISSION)

    subprocess.run(["chmod", "-R", "g+w", str(engine.rundir)], check=False)
    subprocess.run(["chmod", "g+w", str(SUBMISSION)], check=False)
    print(engine)
    print(submission)
    print(
        f"\nWrote {len(submission)} glycosidase sequences to {SUBMISSION} "
        f"from {len(order)} filtered backbones."
    )


if __name__ == "__main__":
    main()
