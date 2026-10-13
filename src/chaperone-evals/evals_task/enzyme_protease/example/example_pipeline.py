#!/usr/bin/env -S bash -c '"${CHAPERONE_REPO:-.}/.venv/bin/python" "$0" "$@"'
"""Reference pipeline for the ``enzyme-protease`` eval.

Scaffolds the HEXXH zinc site plus the downstream Tyr/Glu pair against the
fibrin Aa substrate peptide, then exports a sequence-only submission carrying
the ``motif_index_map`` the held-out scorer needs.

Pipeline::

    rfd3 -> rfd3_filter -> mpnn -> af3 (holo self-validation, ranking only)

The submission's ``motif_index_map`` is *derived*, never guessed: RFD3 emits a
``diffused_index_map`` per design ({"A293": "A45", ...}) that names where every
reference residue of ``inputs/theozyme.pdb`` landed in the generated backbone.
Each mapped output residue is converted to its 1-based ordinal within the
designed chain A (the numbering the scorer rebuilds from the submitted
sequence), and the amino acid actually present there is checked against the
reference residue type. A design whose motif cannot be reconstructed and
verified is dropped with a loud count, never patched up.

Smoke test (one design, one contig, SLURM GPU)::

    ./example_pipeline.py --required_n=1 --nouse_local_gpus \
        --rfd3.n_batches=1 --rfd3.batch_size=1 \
        --rfd3.json_keys_subset=protease_contig1 \
        --rundir=/mnt/home/<user>/eval_runs/protease_smoke

Keep any ``--rundir`` under ``/mnt/home``. ``/scratch/<user>/<jobid>`` is
node-local, and the tool containers bind only ``/mnt/home`` and ``/projects``,
so a rundir elsewhere fails with empty SLURM logs and no usable error.

Production (96 designs, one SLURM GPU)::

    ./example_pipeline.py --nouse_local_gpus
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import polars as pl
from absl import flags

from blocks import Blocks
from blocks.util_blocks.filter import MetricFilter
from chaperone import ExecutionEngine, init_cli
from scoring import THREE_TO_ONE, format_motif_index_map, motif_rmsd
from chaperone.io import parse_to_atom_array

flags.DEFINE_integer(
    "required_n",
    96,
    "Required submission size. Override to 1 only for a smoke test.",
)
FLAGS = flags.FLAGS

init_cli()

SCRIPT_DIR = Path(__file__).resolve().parent
FILES_DIR = SCRIPT_DIR / "files"
SUBMISSION = SCRIPT_DIR / "submission.csv"
THEOZYME_PDB = (SCRIPT_DIR.parent / "inputs" / "theozyme.pdb").resolve()

DESIGN_CHAIN = "A"
REFERENCE_CHAIN = "A"  # the theozyme chain the reference positions live on

# Must match score/score.py exactly — the contract, not a preference.
MOTIF_REFERENCE_IDS = ("H293", "E294", "H297", "E316", "Y377", "E378")
MOTIF_RESIDUE_TYPES = {
    "H293": "HIS",
    "E294": "GLU",
    "H297": "HIS",
    "E316": "GLU",
    "Y377": "TYR",
    "E378": "GLU",
}

# One SLURM GPU for the whole chain; rows_per_job keeps every step in a
# single chunk so the run never fans out across the cluster.
GPU_ARGS = {
    "partition": "gpu",
    "gres": "gpu:small:1",
    "mem_gb": 32,
    "time_limit": "8:00:00",
    "rows_per_job": 1000,
}

RFD3, Filter, MPNN, AF3 = Blocks.getall("rfd3", "filter", "mpnn", "af3")

# ── Block configuration ──────────────────────────────────────────────────────

blocks = [
    # 2 contig permutations x n_batches x batch_size designs. 12 batches
    # measured 93/192 through rfd3_filter — short of 96, so the default is
    # sized for the observed ~48 % survival with margin.
    RFD3(
        inputs=str(FILES_DIR / "config.json"),
        n_batches=18,
        batch_size=8,
        inference_sampler={"s_jitter_origin": 1},
    ),
    Filter(
        name="rfd3_filter",
        filters={
            "rfd3.n_chainbreaks": MetricFilter(max=0),
            "rfd3.n_clashing.ligand_clashes": MetricFilter(max=0),
            "rfd3.n_clashing.interresidue_clashes_w_backbone": MetricFilter(max=0),
            "rfd3.n_clashing.interresidue_clashes_w_sidechain": MetricFilter(max=0),
            "rfd3.non_loop_fraction": MetricFilter(min=0.2),
            "rfd3.n_clashing.ligand_min_distance": MetricFilter(min=2.0),
        },
    ),
    MPNN(
        model_type="ligand_mpnn",
        num_seq_per_pdb=1,
        keep_motif_sequences_fixed=True,
        args={
            "temperature": 0.3,
            "omit_AA": "CH",
            "chains_to_design": "A",
        },
    ),
    # Self-validation only: used to rank which designs are submitted. The
    # held-out scorer re-folds every submitted sequence independently.
    AF3(name="holo", diffusion_batch_size=1),
]

# ── Execute ──────────────────────────────────────────────────────────────────

engine = ExecutionEngine(rundir=SCRIPT_DIR / "rundir", reinit=False)
handler = engine.run(blocks, job_args=GPU_ARGS)
frame = handler.unpack(all=True)

# ── Derive the motif index map ───────────────────────────────────────────────


def chain_residues(structure_path: str, chain: str) -> tuple[list[int], str]:
    """``(res_ids, one_letter_sequence)`` for ``chain``, in file order."""
    from biotite.structure import get_residue_starts

    atoms = parse_to_atom_array(structure_path)
    atoms = atoms[atoms.chain_id == chain]
    if atoms.array_length() == 0:
        raise ValueError(f"{structure_path} has no chain {chain}")
    starts = get_residue_starts(atoms)
    res_ids = [int(r) for r in atoms.res_id[starts]]
    sequence = "".join(
        THREE_TO_ONE.get(str(name).upper(), "X") for name in atoms.res_name[starts]
    )
    return res_ids, sequence


def derive_motif_index_map(
    diffused_index_map: dict[str, str],
    res_ids: list[int],
    sequence: str,
) -> dict[str, str]:
    """Map every reference position onto a 1-based index of the design chain.

    Raises when a reference position is absent from the RFD3 map, landed on a
    different chain, or carries the wrong amino acid — all of which mean the
    correspondence cannot be trusted.
    """
    ordinal = {res_id: index + 1 for index, res_id in enumerate(res_ids)}
    mapping: dict[str, str] = {}
    for reference in MOTIF_REFERENCE_IDS:
        source = f"{REFERENCE_CHAIN}{reference[1:]}"
        target = diffused_index_map.get(source)
        if target is None:
            raise ValueError(f"diffused_index_map has no entry for {source}")
        chain, res_id = target[0], int(target[1:])
        if chain != DESIGN_CHAIN:
            raise ValueError(f"{source} mapped to chain {chain}, not {DESIGN_CHAIN}")
        if res_id not in ordinal:
            raise ValueError(f"{source} mapped to {target}, absent from the design")
        index = ordinal[res_id]
        expected = THREE_TO_ONE[MOTIF_RESIDUE_TYPES[reference]]
        actual = sequence[index - 1]
        if actual != expected:
            raise ValueError(
                f"{reference} landed on {DESIGN_CHAIN}{index} which is {actual}, "
                f"expected {expected}"
            )
        mapping[reference] = f"{DESIGN_CHAIN}{index}"
    return mapping


records: list[dict] = []
rejected: dict[str, int] = {}
for row_index in range(len(frame)):
    row = frame.iloc[row_index]
    raw_map = row["rfd3.metadata.diffused_index_map"]
    if not raw_map:
        raise ValueError(
            f"row {row_index} has no rfd3.metadata.diffused_index_map; the motif "
            "correspondence cannot be reconstructed"
        )
    diffused_index_map = json.loads(raw_map) if isinstance(raw_map, str) else raw_map

    # The diffused_index_map is expressed in RFD3's output numbering, so the
    # designed structure must still carry that numbering for the lookup to
    # mean anything. Check it rather than assume it.
    res_ids, sequence = chain_residues(row["mpnn.path"], DESIGN_CHAIN)
    rfd3_res_ids, _ = chain_residues(row["rfd3.path"], DESIGN_CHAIN)
    if res_ids != rfd3_res_ids:
        raise ValueError(
            f"row {row_index}: mpnn renumbered chain {DESIGN_CHAIN} relative to "
            "rfd3; the diffused_index_map lookup would be meaningless"
        )
    try:
        mapping = derive_motif_index_map(diffused_index_map, res_ids, sequence)
    except ValueError as exc:
        rejected[str(exc)] = rejected.get(str(exc), 0) + 1
        continue
    if "X" in sequence:
        reason = "non-canonical residue in chain A"
        rejected[reason] = rejected.get(reason, 0) + 1
        continue

    # Self-measured motif fidelity on our own fold. The reference geometry is a
    # supplied input, so ranking on it is fair game — and it is the metric the
    # task calls headline, so selecting on it is what a designer should do.
    records.append(
        {
            "design_id": f"protease_{row_index:04d}",
            "sequence": sequence,
            "motif_index_map": format_motif_index_map(mapping),
            "_motif_rmsd": motif_rmsd(row["holo.path"], THEOZYME_PDB, mapping),
            "_min_pae_interaction": float(row["holo.min_pae_interaction"]),
            "_plddt": float(row["holo.plddt"]),
        }
    )

print(f"{len(frame)} designs folded; {len(records)} carry a verified motif map")
if rejected:
    print("Rejected (motif map could not be verified):")
    for reason, count in sorted(rejected.items()):
        print(f"  {count:4d}  {reason}")

# ── Select and write the submission ──────────────────────────────────────────

RANKED = {"_motif_rmsd": True, "_min_pae_interaction": True, "_plddt": False}

pool = pl.DataFrame(records)
if len(pool) == 0:
    raise RuntimeError("No design produced a verifiable motif index map")
for column in RANKED:
    if not np.isfinite(pool[column].to_numpy()).all():
        raise ValueError(f"{column} has non-finite values; refusing to rank")

# Rank-sum over the three scored quantities so selection favours designs that
# are plausible on every gate, not extreme on one.
pool = pool.unique(subset=["sequence"], keep="first")
rank_sum = np.zeros(len(pool))
for column, ascending in RANKED.items():
    values = pool[column].to_numpy()
    order = np.argsort(values if ascending else -values, kind="stable")
    ranks = np.empty(len(pool))
    ranks[order] = np.arange(len(pool))
    rank_sum += ranks
pool = pool.with_columns(pl.Series("_rank_sum", rank_sum)).sort(
    ["_rank_sum", "_motif_rmsd"]
)

if len(pool) < FLAGS.required_n:
    raise RuntimeError(
        f"Only {len(pool)} unique verified designs available; need "
        f"{FLAGS.required_n}. Scale up with --rfd3.n_batches."
    )

selected = pool.head(FLAGS.required_n)
for column in RANKED:
    values = selected[column].to_numpy()
    print(
        f"selected {column.lstrip('_')}: min {values.min():.3f} "
        f"median {np.median(values):.3f} max {values.max():.3f}"
    )
submission = selected.drop([*RANKED, "_rank_sum"])
submission.write_csv(SUBMISSION)

subprocess.run(["chmod", "-R", "g+w", str(engine.rundir)], check=False)
subprocess.run(["chmod", "g+w", str(SUBMISSION)], check=False)
print(engine)
print(submission)
print(f"\nWrote {len(submission)} protease sequences to {SUBMISSION}")
