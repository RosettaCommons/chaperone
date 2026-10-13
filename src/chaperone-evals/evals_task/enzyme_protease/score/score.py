#!/usr/bin/env -S bash -c '"${CHAPERONE_REPO:-.}/.venv/bin/python" "$0" "$@"'
"""Held-out scorer for de novo zinc-metalloprotease submissions.

A submission is sequences plus a ``motif_index_map`` naming where each
reference catalytic position landed. The scorer co-folds every design with the
supplied substrate peptide (chain B) and the Zn site (chain C), then measures:

  motif fidelity   — all-atom RMSD of the mapped catalytic residues vs the
                     reference geometry (the headline metric)
  fold confidence  — AF3 pLDDT of the fold
  substrate engagement — design↔substrate interface PAE, chain-restricted via
                     ``ipsae.min_pae_interface`` (A vs B). AF3's own
                     ``min_pae_interaction`` is a minimum over *all* chain
                     pairs and is unusable as a gate here — see readme.md.

Self-reported metrics are never trusted; every sequence is re-folded here.

Usage:
    ./score.py --designs=/absolute/path/to/submission.csv
"""

from __future__ import annotations

from pathlib import Path

from absl import flags

from blocks import Blocks
from chaperone import init_cli
from scoring import (
    SequenceSubmissionSpec,
    define_score_flags,
    load_context_chains,
    merge_submission_columns,
    motif_rmsd,
    run_sequence_scorer,
)

define_score_flags()
FLAGS = flags.FLAGS

init_cli()

EVAL_NAME = "enzyme-protease"
ORACLE = "af3"

SCRIPT_DIR = Path(__file__).resolve().parent
THEOZYME_PDB = (SCRIPT_DIR.parent / "inputs" / "theozyme.pdb").resolve()

DESIGN_CHAIN = "A"
SUBSTRATE_CHAIN = "B"
METAL_CHAIN = "C"
CONTEXT_CHAINS = (SUBSTRATE_CHAIN, METAL_CHAIN)

# Reference catalytic positions, in the reference enzyme's own numbering.
# HEXXH (H293/E294/H297) + the Tyr/Glu pair and the third Zn ligand.
MOTIF_REFERENCE_IDS = ("H293", "E294", "H297", "E316", "Y377", "E378")
MOTIF_RESIDUE_TYPES = {
    "H293": "HIS",
    "E294": "GLU",
    "H297": "HIS",
    "E316": "GLU",
    "Y377": "TYR",
    "E378": "GLU",
}

# Thresholds are held out. NOTE: uncalibrated — no reference campaign has been
# run through this scorer yet, so these are starting points, not settled
# criteria.
MOTIF_RMSD_MAX = 1.5
MIN_PLDDT = 80.0
INTERFACE_PAE_MAX = 10.0

GPU_ARGS = {
    "partition": "gpu",
    "gres": "gpu:small:1",
    "mem_gb": 32,
    "time_limit": "6:00:00",
}
CPU_ARGS = {"partition": "local"}


def measure_motif_rmsd(handler, submission):
    """All-atom RMSD of each design's mapped catalytic residues vs the reference.

    Returned as a ``motif.rmsd`` metric so the gate can address it like any
    other. The map is per design and joined on ``design_id`` — never by row
    position, since the scored handler comes back in filesystem order.
    """
    frame = merge_submission_columns(
        handler.unpack(all=True), submission, ["motif_index_map"]
    )
    values = {}
    for design_id, path, motif_map in zip(
        frame["design_id"], frame["holo.path"], frame["motif_index_map"]
    ):
        try:
            rmsd = motif_rmsd(path, THEOZYME_PDB, motif_map)
        except Exception as exc:
            raise RuntimeError(f"motif RMSD failed for {design_id}: {exc}") from exc
        values[design_id] = {"motif": {"rmsd": rmsd}}
    return values


def main() -> None:
    AF3, ContactMS, IpSAE = Blocks.getall("af3", "contact_ms", "ipsae")

    run_sequence_scorer(
        eval_name=EVAL_NAME,
        validation_oracle=ORACLE,
        designs=FLAGS.designs,
        out=FLAGS.out,
        per_design_out=FLAGS.per_design_out,
        required_n=FLAGS.required_n,
        run_meta=FLAGS.run_meta,
        run_root=SCRIPT_DIR / "score_runs",
        spec=SequenceSubmissionSpec(
            required_n=FLAGS.required_n,
            chains=(DESIGN_CHAIN,),
            motif_reference_ids=MOTIF_REFERENCE_IDS,
            motif_residue_types=MOTIF_RESIDUE_TYPES,
            max_pairwise_identity=0.70,
        ),
        # The substrate peptide and the Zn site are supplied by the scorer, never
        # by the submitter, and are co-folded with every design.
        context=load_context_chains(THEOZYME_PDB, CONTEXT_CHAINS),
        oracle=AF3(name="holo"),
        oracle_job_args=GPU_ARGS,
        metrics=[
            ContactMS(
                name="holo_contact_ms",
                binder_chains=[DESIGN_CHAIN],
                target_chains=[SUBSTRATE_CHAIN],
            ),
            # Chain-restricted design<->substrate PAE. AF3's own
            # `min_pae_interaction` is the minimum over ALL off-diagonal chain
            # pairs, so on this three-chain fold (A design, B substrate, C Zn) it
            # frequently reports the trivially confident B:C substrate-metal pair
            # instead of the interface this benchmark cares about. ipsae's
            # `min_pae_interface` is restricted to binder x target.
            IpSAE(
                name="holo_ipsae",
                binder_chain=DESIGN_CHAIN,
                target_chains=[SUBSTRATE_CHAIN],
            ),
        ],
        metric_job_args=CPU_ARGS,
        measure=measure_motif_rmsd,
        # Gates are filter-expression strings (the repo's FilterBlock DSL); each
        # threshold lives in the expression that enforces it.
        gates={
            "motif_fidelity": f"motif.rmsd <= {MOTIF_RMSD_MAX}",
            "plddt": f"holo.plddt >= {MIN_PLDDT}",
            "substrate_interface": (
                f"holo_ipsae.min_pae_interface <= {INTERFACE_PAE_MAX}"
            ),
        },
        metric_columns=[
            "motif.rmsd",
            "holo.plddt",
            "holo_ipsae.min_pae_interface",
            # Reported for provenance only. NOT gated: see the note above and
            # score/readme.md — it is chain-agnostic on a three-chain fold.
            "holo.min_pae_interaction",
            "holo_contact_ms",
        ],
        extra={"reference": THEOZYME_PDB.name},
    )


if __name__ == "__main__":
    main()
