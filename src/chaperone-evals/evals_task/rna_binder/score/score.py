#!/usr/bin/env -S bash -c '"${PIPELINES_REPO:-.}/.venv/bin/python" "$0" "$@"'
"""Held-out scorer for sequence-only RNA-hairpin binder submissions.

Supplies the canonical RNA hairpin (`../inputs/target.pdb`, chain B) as
co-folded context, predicts every complex with plain AF3, and gates on fold
quality, interface confidence, DSSR/SNAP base-edge readout and stacking,
buried surface, RNA conformation, and backbone sterics.

`af3_with_target_template` is deliberately NOT used: it only injects templates
into `entry["protein"]` and silently skips a DNA/RNA target chain, so the
target would enter unrestrained and the templating would be a no-op.

Usage:
    ./score.py --designs=/absolute/path/to/submission.csv --nouse_local_gpus
"""

from __future__ import annotations

from pathlib import Path

from absl import flags

from blocks import Blocks
from pipelines import init_cli
from pipelines.data.schema import SELECTIONS
from scoring import (
    SequenceSubmissionSpec,
    define_score_flags,
    load_context_chains,
    run_sequence_scorer,
)

define_score_flags()
FLAGS = flags.FLAGS

init_cli()

EVAL_NAME = "rna-binder"
ORACLE = "af3"

SCRIPT_DIR = Path(__file__).resolve().parent
TARGET_PDB = (SCRIPT_DIR.parent / "inputs" / "target.pdb").resolve()

BINDER_CHAIN = "A"
TARGET_CHAIN = "B"

# ── Thresholds ───────────────────────────────────────────────────────────
# See readme.md for the rationale behind every number, including the native
# 1EC6 reference values these were calibrated against.
PLDDT_MIN = 85.0
MIN_PAE_INTERACTION_MAX = 2
TOTAL_HBOND_MIN = 8
MINOR_HBOND_MIN = 2
STACKING_MIN = 2
NT_AA_MIN = 12
CONTACT_MS_MIN = 250.0
BB_CLASHES_MAX = 0

# Target preservation is gated on the base-paired STEM, not the whole hairpin.
# DSSR annotates this target as a 4-bp stem (G1-C20, A2-U19, G3-C18, G4-C17,
# plus a non-canonical A5.C16) closing a 12-nt loop. Across the 160-design
# reference run the stem backbone sits at 0.75 Å median / 1.13 Å max while the
# loop moves 3.48 Å median / 6.04 Å max, so a whole-molecule gate would score
# AF3's loop-conformer sampling rather than whether the design deformed its
# target. The current 1.0 Å limit is intentionally tighter than the
# dna-binder duplex gate; a melted or re-threaded stem lands far outside it.
# See readme.md for the full distribution.
STEM_RESIDUES = (1, 2, 3, 4, 5, 16, 17, 18, 19, 20)
LOOP_RESIDUES = tuple(range(6, 16))
NUCLEOTIDE_BACKBONE_ATOMS = ("P", "O5'", "C5'", "C4'", "C3'", "O3'")
STEM_RMSD_MAX = 1.0

# contact_ms returns a bare float, so its column is the instance name itself.
CONTACT_MS_COL = "score_contact_ms"
# Whole-hairpin RMSD from the rmsd block. Reported as a diagnostic, not gated.
RNA_RMSD_COL = (
    f"score_rmsd.{SELECTIONS.NUCLEOTIDE_BKBN}_align.{SELECTIONS.NUCLEOTIDE}_rmsd"
)

GPU_ARGS = {
    "partition": "gpu",
    "gres": "gpu:small:1",
    "mem_gb": 32,
    "time_limit": "8:00:00",
    "rows_per_job": 1000,
}


def _region_backbone_rmsd(prediction_path: str) -> tuple[float, float]:
    """(stem, loop) RNA backbone RMSD of a prediction against the supplied target.

    Superimposes each region on its own atoms, so the two numbers separate
    "did the base-paired stem survive" from "where did the flexible loop go".
    Atoms are paired by (res_id, atom_name) because the crystal 5' nucleotide
    carries no phosphate while AF3 predicts one.
    """
    import numpy as np
    from biotite.structure import superimpose

    from pipelines.io import parse_to_atom_array

    prediction = parse_to_atom_array(prediction_path)
    target = parse_to_atom_array(TARGET_PDB)
    predicted_rna = prediction[prediction.chain_id == TARGET_CHAIN]
    if not len(predicted_rna):
        raise ValueError(f"{prediction_path} has no chain {TARGET_CHAIN}")

    def region(atoms, residues):
        mask = np.isin(atoms.res_id, residues) & np.isin(
            atoms.atom_name, NUCLEOTIDE_BACKBONE_ATOMS
        )
        return atoms[mask]

    def rmsd(reference, predicted):
        ref_keys = [
            (int(r), str(n)) for r, n in zip(reference.res_id, reference.atom_name)
        ]
        pred_keys = [
            (int(r), str(n)) for r, n in zip(predicted.res_id, predicted.atom_name)
        ]
        shared = set(pred_keys)
        common = [key for key in ref_keys if key in shared]
        if len(common) < 3:
            raise ValueError(
                f"only {len(common)} paired backbone atoms for {prediction_path}"
            )
        paired_reference = reference[[ref_keys.index(k) for k in common]]
        paired_predicted = predicted[[pred_keys.index(k) for k in common]]
        fitted, _ = superimpose(paired_reference, paired_predicted)
        deviation = fitted.coord - paired_reference.coord
        return float(np.sqrt((deviation**2).sum(axis=1).mean()))

    return (
        rmsd(region(target, STEM_RESIDUES), region(predicted_rna, STEM_RESIDUES)),
        rmsd(region(target, LOOP_RESIDUES), region(predicted_rna, LOOP_RESIDUES)),
    )


def measure_region_rmsd(handler, submission):
    """Stem/loop RNA backbone RMSD of each prediction against the supplied target.

    The stem number is the gate; the loop number is reported so the
    whole-hairpin RMSD (from the rmsd block) stays interpretable rather than
    hidden. Returned under the ``rna`` namespace so ``rna.stem_rmsd`` /
    ``rna.loop_rmsd`` become first-class metrics the gates and metric_columns
    address like any other. Keyed on ``design_id`` — never row position, since
    the scored handler comes back in filesystem order.
    """
    frame = handler.unpack(all=True)
    values = {}
    for design_id, path in zip(frame["design_id"], frame["score_af3.path"]):
        try:
            stem_rmsd, loop_rmsd = _region_backbone_rmsd(path)
        except Exception as exc:
            raise RuntimeError(f"region RMSD failed for {design_id}: {exc}") from exc
        values[design_id] = {"rna": {"stem_rmsd": stem_rmsd, "loop_rmsd": loop_rmsd}}
    return values


def main() -> None:
    AF3, DSSR, ContactMS, RMSD, Clash = Blocks.getall(
        ORACLE, "dssr_snap_metrics", "contact_ms", "rmsd", "clash"
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
        spec=SequenceSubmissionSpec(
            required_n=FLAGS.required_n,
            chains=(BINDER_CHAIN,),
            max_pairwise_identity=0.70,
        ),
        # The canonical RNA hairpin (chain B) is supplied by the scorer, never by
        # the submitter, and co-folded with every design.
        context=load_context_chains(TARGET_PDB, [TARGET_CHAIN]),
        oracle=AF3(name="score_af3", diffusion_batch_size=1),
        oracle_job_args=GPU_ARGS,
        metrics=[
            # DSSR/SNAP is the only nucleic-acid-aware interface descriptor in
            # the repo; hbond_metrics / plip / rasa / ppi_ddg are protein and
            # small-molecule only and would not see the RNA at all.
            DSSR(name="score_dssr"),
            ContactMS(
                name="score_contact_ms",
                binder_chains=[BINDER_CHAIN],
                target_chains=[TARGET_CHAIN],
            ),
            # Compares the supplied target (row structure 0) with the AF3
            # prediction (last). Nucleotide selections are not in the block
            # defaults, and name-pairing is required because the crystal 5'
            # residue carries no phosphate while AF3 predicts one.
            RMSD(
                name="score_rmsd",
                alignment_methods=[SELECTIONS.NUCLEOTIDE_BKBN],
                rmsd_targets=[SELECTIONS.NUCLEOTIDE],
                select_mask_by_name_pairing=[
                    SELECTIONS.NUCLEOTIDE,
                    SELECTIONS.NUCLEOTIDE_BKBN,
                ],
                resolve_symmetries=False,
            ),
            Clash(name="score_clash"),
        ],
        # Stem/loop decomposition of the RNA backbone. The stem number is the
        # gate (rna_stem_conformation); the loop number is reported alongside the
        # whole-hairpin RMSD from the rmsd block.
        measure=measure_region_rmsd,
        # Gates are filter-expression strings (the repo's FilterBlock DSL); each
        # threshold lives in the expression that enforces it. The stacking gate
        # is the OR of the two DSSR pseudo counts, equivalent to the sum >= 1 for
        # non-negative integer counts.
        gates={
            "plddt": f"score_af3.plddt >= {PLDDT_MIN}",
            "min_pae_interaction": (
                f"score_af3.min_pae_interaction <= {MIN_PAE_INTERACTION_MAX}"
            ),
            "total_hbond": f"score_dssr.Total_hbond >= {TOTAL_HBOND_MIN}",
            "minor_hbond": f"score_dssr.Minor_hbond >= {MINOR_HBOND_MIN}",
            "stacking": (
                f"score_dssr.pseudo_pairs >= {STACKING_MIN} "
                f"or score_dssr.pseudo_stacks >= {STACKING_MIN}"
            ),
            "nt_aa": f"score_dssr.nt_aa >= {NT_AA_MIN}",
            "contact_ms": f"{CONTACT_MS_COL} >= {CONTACT_MS_MIN}",
            "rna_stem_conformation": f"rna.stem_rmsd <= {STEM_RMSD_MAX}",
            "bb_clashes": f"score_clash.n_bb_clashes <= {BB_CLASHES_MAX}",
        },
        # A missing RMSD column means the nucleotide mask never matched, and a NaN
        # SNAP metric means x3dna-snap did not run. Both are upstream bugs, not
        # failing designs, so require_finite_metrics must stop the score rather
        # than tolerate them.
        metric_columns=[
            "score_af3.plddt",
            "score_af3.min_pae_interaction",
            "score_dssr.Total_hbond",
            "score_dssr.Minor_hbond",
            "score_dssr.Major_hbond",
            "score_dssr.pseudo_pairs",
            "score_dssr.pseudo_stacks",
            "score_dssr.nt_aa",
            CONTACT_MS_COL,
            RNA_RMSD_COL,
            "rna.stem_rmsd",
            "rna.loop_rmsd",
            "score_clash.n_bb_clashes",
        ],
        extra={"target": TARGET_PDB.name, "target_source_pdb": "1EC6 chain B"},
    )


if __name__ == "__main__":
    main()
