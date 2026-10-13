#!/usr/bin/env -S bash -c '"${PIPELINES_REPO:-.}/.venv/bin/python" "$0" "$@"'
"""Held-out scorer for sequence-only C3 homo-oligomer submissions.

Each submitted protomer sequence is replicated into three identical chains
A/B/C (the ``oligomer_copies`` half of the submission contract), folded as a
trimer with AF3, and gated on the *global* geometry of the prediction — whether
it is actually C3 — before anything local is measured.

Usage:
    ./score.py --designs=/absolute/path/to/submission.csv --nouse_local_gpus
"""

from __future__ import annotations

from pathlib import Path

from absl import flags

from blocks import Blocks
from pipelines import init_cli
from scoring import (
    SequenceSubmissionSpec,
    ca_radius_of_gyration,
    define_score_flags,
    run_sequence_scorer,
)

define_score_flags()
FLAGS = flags.FLAGS

init_cli()

EVAL_NAME = "nanoparticle-c3"
ORACLE = "af3"

SCRIPT_DIR = Path(__file__).resolve().parent

POINT_GROUP = "C3"
N_COPIES = 3
PROTOMER_CHAIN = "A"
# C3 has three equivalent protomer-protomer interfaces; A|B is scored as the
# representative one. Any asymmetry between them is already caught upstream by
# the symmetry gates, which is what makes scoring one interface sufficient.
NEIGHBOUR_CHAIN = "B"

# ── Thresholds (every one is justified in readme.md) ─────────────────────────
ANGLE_DEVIATION_MAX = 5.0  # degrees from the ideal 120
SUBUNIT_RMSD_MAX = 1.0  # A, protomers must be the same fold
IPTM_MIN = 0.85
PLDDT_MIN = 85.0
CONTACT_MS_MIN = 600.0  # A^2, per interface
FRACTION_LOOP_MAX = 0.35
PROTOMER_CA_RG_MAX = 14.0  # A, rejects extended coiled-coil rods
BB_CLASHES_MAX = 0
DDG_MAX = -35.0  # soft

ANGLE_COL = "score_symmetry.angle_deviation_deg"
SUBUNIT_RMSD_COL = "score_symmetry.subunit_rmsd"
AXIS_RMSD_COL = "score_symmetry.axis_rmsd"
IPTM_COL = "score_af3.iptm"
PLDDT_COL = "score_af3.plddt"
CONTACT_MS_COL = "score_contact_ms"  # contact_ms returns a bare float
FRACTION_LOOP_COL = "score_secondary_structure.fraction_loop"
PROTOMER_CA_RG_COL = "protomer.ca_radius_of_gyration"
CLASH_COL = "score_clash.n_bb_clashes"
DDG_COL = "score_ddg.ddg"

# A C3 trimer is a 3x-larger fold than a mini-binder, so the oracle gets the
# large-GPU allocation; rows_per_job keeps the run to a single GPU job.
GPU_ARGS = {
    "partition": "gpu",
    "gres": "gpu:large:1",
    "mem_gb": 64,
    "time_limit": "8:00:00",
    "rows_per_job": 1000,
}

# The CPU metric blocks default to 4 GB, which OOM-kills on 96 trimers: a chained
# step takes the FIRST block's resources, and every structure here is a
# ~210-residue three-chain assembly rather than a single mini-binder. Set
# explicitly rather than relying on any one block's default.
CPU_ARGS = {
    "partition": "cpu",
    "mem_gb": 32,
    "time_limit": "4:00:00",
    "rows_per_job": 1000,
}


def measure_protomer(handler, submission):
    """Protomer length and predicted-chain compactness per design.

    Radius of gyration is measured from chain A's C-alpha coordinates in the
    held-out AF3 prediction.  Values are attached via ``with_metrics`` (joined
    on ``design_id``, never by row position) so they are gated and summarized
    like block-produced metric columns.
    """
    frame = handler.unpack(all=True)
    values = {}
    for design_id, length, path in zip(
        frame["design_id"],
        frame["sequence_A"].str.len(),
        frame["score_af3.path"],
    ):
        try:
            radius = ca_radius_of_gyration(path, chain_id=PROTOMER_CHAIN)
        except Exception as exc:
            raise RuntimeError(
                f"protomer compactness failed for {design_id} ({path}): {exc}"
            ) from exc
        values[design_id] = {
            "protomer_length": int(length),
            "protomer": {"ca_radius_of_gyration": radius},
        }
    return values


def main() -> None:
    spec = SequenceSubmissionSpec(
        required_n=FLAGS.required_n,
        chains=(PROTOMER_CHAIN,),
        oligomer_copies=N_COPIES,
        max_pairwise_identity=0.70,
    )

    AF3, Symmetry, ContactMS, SecondaryStructure, Clash, PPIDDG = Blocks.getall(
        ORACLE, "symmetry", "contact_ms", "secondary_structure", "clash", "ppi_ddg"
    )

    run_sequence_scorer(
        eval_name=EVAL_NAME,
        validation_oracle=ORACLE,
        designs=FLAGS.designs,
        out=FLAGS.out,
        per_design_out=FLAGS.per_design_out,
        required_n=FLAGS.required_n,
        run_meta=FLAGS.run_meta,
        run_root=SCRIPT_DIR / "score_runs",
        spec=spec,
        oracle=AF3(name="score_af3", diffusion_batch_size=1),
        oracle_job_args=GPU_ARGS,
        metrics=[
            Symmetry(
                name="score_symmetry",
                point_group=POINT_GROUP,
                chains=list(spec.assembly_chains),
            ),
            ContactMS(
                name="score_contact_ms",
                binder_chains=[PROTOMER_CHAIN],
                target_chains=[NEIGHBOUR_CHAIN],
            ),
            SecondaryStructure(
                name="score_secondary_structure", chains=[PROTOMER_CHAIN]
            ),
            Clash(name="score_clash"),
            PPIDDG(
                name="score_ddg",
                binder_chain=PROTOMER_CHAIN,
                target_chain=NEIGHBOUR_CHAIN,
                relax_rounds=1,
                replicates=1,
            ),
        ],
        # Chained CPU metric steps take the FIRST block's resources; every
        # structure is a ~210-residue trimer, so 32 GB is set explicitly to
        # avoid the 4 GB default OOM-killing on 96 assemblies.
        metric_job_args=CPU_ARGS,
        measure=measure_protomer,
        # Gates are filter-expression strings (the repo's FilterBlock DSL); each
        # threshold lives in the expression that enforces it. Global C3 geometry
        # is gated before anything local. ``ddg`` is soft.
        gates={
            "angle_deviation": f"{ANGLE_COL} <= {ANGLE_DEVIATION_MAX}",
            "subunit_rmsd": f"{SUBUNIT_RMSD_COL} <= {SUBUNIT_RMSD_MAX}",
            "iptm": f"{IPTM_COL} >= {IPTM_MIN}",
            "plddt": f"{PLDDT_COL} >= {PLDDT_MIN}",
            "contact_ms": f"{CONTACT_MS_COL} >= {CONTACT_MS_MIN}",
            "fraction_loop": f"{FRACTION_LOOP_COL} < {FRACTION_LOOP_MAX}",
            "compactness": f"{PROTOMER_CA_RG_COL} <= {PROTOMER_CA_RG_MAX}",
            "clashes": f"{CLASH_COL} == {BB_CLASHES_MAX}",
            "ddg": f"{DDG_COL} < {DDG_MAX}",
        },
        soft=["ddg"],
        metric_columns=[
            ANGLE_COL,
            SUBUNIT_RMSD_COL,
            AXIS_RMSD_COL,
            IPTM_COL,
            PLDDT_COL,
            CONTACT_MS_COL,
            FRACTION_LOOP_COL,
            PROTOMER_CA_RG_COL,
            CLASH_COL,
            DDG_COL,
            "protomer_length",
        ],
        extra={
            "point_group": POINT_GROUP,
            "oligomer_copies": N_COPIES,
            "assembly_chains": list(spec.assembly_chains),
            "scored_interface": f"{PROTOMER_CHAIN}|{NEIGHBOUR_CHAIN}",
            "compactness_chain": PROTOMER_CHAIN,
        },
    )


if __name__ == "__main__":
    main()
