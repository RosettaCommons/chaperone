#!/usr/bin/env -S bash -c '"${PIPELINES_REPO:-.}/.venv/bin/python" "$0" "$@"'
"""Held-out scorer for hammerhead ribozyme rescaffolding submissions.

The public submission is RNA sequence plus a 0-based motif map. The scorer
materializes RNA-typed inputs with its own Mn2+, refolds every design with AF3,
and measures the six mapped catalytic residues against the 2OEU crystal and
against absolute in-line attack geometry. Submitted coordinates and
self-reported metrics are never accepted.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from absl import flags
from blocks import Blocks
from pipelines import init_cli

from scoring import SequenceSubmissionSpec, define_score_flags, run_sequence_scorer

define_score_flags()
FLAGS = flags.FLAGS

init_cli()

EVAL_NAME = "ribozyme-hammerhead"
ORACLE = "af3"
SCORER_CACHE_VERSION = "rz01-v3-dssr-t6-resilient"

SCRIPT_DIR = Path(__file__).resolve().parent
INPUTS_DIR = SCRIPT_DIR.parent / "inputs"
ASSETS_DIR = SCRIPT_DIR / "assets"
MOTIF = (INPUTS_DIR / "2oeu_motif.pdb").resolve()
NATIVE = (ASSETS_DIR / "2oeu_native.cif").resolve()
HOMOLOGS = (ASSETS_DIR / "homologs.fasta").resolve()

DESIGN_CHAIN = "A"
LIGANDS = ("MN",)
LENGTH_RANGE = (50, 90)
MAX_PAIRWISE_IDENTITY = 0.70

MOTIF_REFERENCE_IDS = ("A20", "A21", "A22", "A36", "B6", "B7")
MOTIF_RESIDUE_TYPES = {
    "A20": "G",
    "A21": "A",
    "A22": "G",
    "A36": "G",
    "B6": "C",
    "B7": "C",
}
MOTIF_CONTIGUITY_GROUPS = (("A20", "A21", "A22"), ("B6", "B7"))

CATALYTIC = {
    "general_base_dist": ["A36.N1", "B6.O2'"],
    "general_acid_dist": ["A20.O2'", "B7.O5'"],
    "inline_attack_dist": ["B6.O2'", "B7.P"],
    "inline_attack_angle": ["B6.O2'", "B7.P", "B7.O5'"],
    "metal_a9_dist": ["L:MN.MN", "A21.OP2"],
    "metal_g10_dist": ["L:MN.MN", "A22.N7"],
}

# Starting thresholds from the measured 10,000-design RZ-01 campaign. The
# structural triple admitted 23 designs and the native control clears it at
# 0.49 A motif RMSD / 164.3 degrees. The novelty threshold is the measured
# shuffled-null maximum for the shipped 503-reference set; the guide's 0.40
# placeholder is below the observed noise floor and would reject every design.
MOTIF_RMSD_MAX = 2.0
INLINE_DIST_MAX = 4.0
INLINE_ANGLE_MIN = 140.0
PLDDT_MIN = 70.0
NOVELTY_MAX_IDENTITY = 0.563

GATES = {
    "motif_rmsd": f"score_motif_rmsd.motif_rmsd <= {MOTIF_RMSD_MAX}",
    "inline_attack_dist": (
        f"score_geometry.inline_attack_dist <= {INLINE_DIST_MAX}"
    ),
    "inline_attack_angle": (
        f"score_geometry.inline_attack_angle >= {INLINE_ANGLE_MIN}"
    ),
    "plddt": f"score_af3.plddt >= {PLDDT_MIN}",
    "novelty": f"score_novelty.max_identity < {NOVELTY_MAX_IDENTITY}",
}

GPU_ARGS = {
    "partition": "gpu",
    "gres": "gpu:small:1",
    "constraint": "A4000",
    "mem_gb": 32,
    "time_limit": "8:00:00",
    "rows_per_job": 200,
}
_EXCLUDED_NODES = os.environ.get("HARBOR_EXCLUDED_NODES", "").strip()
if _EXCLUDED_NODES:
    if not re.fullmatch(r"[A-Za-z0-9_,.\[\]-]+", _EXCLUDED_NODES):
        raise ValueError("HARBOR_EXCLUDED_NODES contains unsupported characters")
    GPU_ARGS["exclude"] = _EXCLUDED_NODES

# Every post-fold metric is CPU-only: motif geometry/confidence, US-align,
# global sequence identity, and DSSR. Keep the scorer's existing memory/time
# ceilings without reserving an idle GPU for that chained metric job.
METRIC_ARGS = {
    "partition": "cpu",
    "mem_gb": 32,
    "time_limit": "8:00:00",
    "rows_per_job": 200,
}

METRIC_COLUMNS = [
    "score_af3.plddt",
    "score_motif_rmsd.motif_rmsd",
    "score_geometry.general_base_dist",
    "score_geometry.general_acid_dist",
    "score_geometry.inline_attack_dist",
    "score_geometry.inline_attack_angle",
    "score_geometry.metal_a9_dist",
    "score_geometry.metal_g10_dist",
    "score_confidence.plddt_motif",
    "score_confidence.pae_motif_scaffold_mean",
    "score_tm.tm_ref",
    "score_novelty.max_identity",
    "score_novelty.max_intra_submission_identity",
]

# T6 is report-only and deliberately excluded from METRIC_COLUMNS: a missing
# DSSR annotation must not void otherwise complete primary scientific scores.
# DSSR can annotate more than one partner per residue, so the vector includes
# partner counts as well as one-bit residue status.
T6_REPORT_COLUMNS = [
    "t6.n_pairs_total",
    "t6.n_paired",
    "t6.fraction_paired",
    "t6.n_partner_native_type",
    "t6.n_partner_in_motif",
    "t6.n_native_lw_match",
    "t6.fraction_native_lw_match",
    "t6.n_native_classification_match",
    "t6.fraction_native_classification_match",
    *[
        f"t6.{metric}_{reference_id}"
        for reference_id in MOTIF_REFERENCE_IDS
        for metric in (
            "paired",
            "n_partners",
            "partner_native_type",
            "partner_in_motif",
            "native_lw_match",
            "native_classification_match",
        )
    ],
]

T6_DIAGNOSTIC_COLUMNS = [
    *T6_REPORT_COLUMNS,
    "t6.status",
    "t6.error_type",
    "t6.error",
    *[
        f"t6.pair_classifications_{reference_id}"
        for reference_id in MOTIF_REFERENCE_IDS
    ],
]


def _as_plain_list(value):
    """Normalize nested vector payloads without treating missing values as data."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _decode_pair_classifications(value):
    if isinstance(value, str):
        if not value:
            return []
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    return _as_plain_list(value)


def _report_metadata(frame):
    """Static target metadata plus the report-only T6 residue summary."""
    if "t6.status" in frame:
        t6_status = frame["t6.status"].fillna("error").astype(str)
        complete = t6_status.eq("ok")
    else:
        # Backward-compatible with cached T6 outputs created before explicit
        # status fields were added: a completed legacy row is successful.
        complete = frame.index.to_series().map(lambda _: True)

    native_details = {}
    if "t6.native_pair_details" in frame:
        for value in frame.loc[complete, "t6.native_pair_details"]:
            if not isinstance(value, dict):
                continue
            native_details = {
                str(reference_id): _as_plain_list(pair_details)
                for reference_id, pair_details in value.items()
            }
            break

    def sum_complete(column):
        if column not in frame:
            return 0
        values = frame.loc[complete, column]
        return int(values.fillna(0).sum())

    failed_designs = []
    for _, row in frame.loc[~complete].iterrows():
        failed_designs.append(
            {
                "design_id": str(row["design_id"]),
                "error_type": str(row.get("t6.error_type", "")),
                "error": str(row.get("t6.error", "")),
            }
        )

    n_complete = int(complete.sum())
    n_failed = len(frame) - n_complete
    t6_summary_status = "ok" if n_failed == 0 else "partial" if n_complete else "error"
    motif_rmsd_only_passers = frame[
        frame["score_motif_rmsd.motif_rmsd"] < MOTIF_RMSD_MAX
    ]
    return {
        "target": "RZ-01",
        "target_source_pdb": "2OEU",
        "scorer_cache_version": SCORER_CACHE_VERSION,
        "novelty_reference_count": 503,
        # Preserve a motif-reconstruction-only alternate pass rate even when
        # the conjunction of the five primary gates is empty.
        "motif_rmsd_only_pass": {
            "threshold": f"score_motif_rmsd.motif_rmsd < {MOTIF_RMSD_MAX}",
            "n_pass": len(motif_rmsd_only_passers),
            "pass_rate": len(motif_rmsd_only_passers) / len(frame),
            "design_ids": [
                str(value) for value in motif_rmsd_only_passers["design_id"]
            ],
        },
        "t6": {
            "role": "report_only",
            "detector": "x3dna-dssr",
            "status": t6_summary_status,
            "n_complete": n_complete,
            "n_failed": n_failed,
            "failed_designs": failed_designs,
            "paired_by_residue": {
                reference_id: sum_complete(f"t6.paired_{reference_id}")
                for reference_id in MOTIF_REFERENCE_IDS
            },
            "native_lw_match_by_residue": {
                reference_id: sum_complete(f"t6.native_lw_match_{reference_id}")
                for reference_id in MOTIF_REFERENCE_IDS
            },
            "native_pair_details": native_details,
            "pair_classifications_by_design": {
                str(row["design_id"]): {
                    reference_id: _decode_pair_classifications(
                        row.get(f"t6.pair_classifications_{reference_id}")
                    )
                    for reference_id in MOTIF_REFERENCE_IDS
                }
                for _, row in frame.loc[complete].iterrows()
            },
        },
    }


def main() -> None:
    (
        AF3,
        RefMotifRMSD,
        MotifGeometry,
        MotifConfidence,
        TMScoreRNA,
        NoveltyGate,
        DSSRMotifBasePairing,
    ) = (
        Blocks.getall(
            ORACLE,
            "reference_motif_rmsd",
            "motif_geometry",
            "motif_confidence",
            "tm_score_rna",
            "sequence_identity_gate",
            "dssr_motif_base_pairing",
        )
    )

    run_root = (
        Path(FLAGS.rundir).expanduser().resolve()
        if FLAGS.rundir
        else SCRIPT_DIR / "score_runs" / SCORER_CACHE_VERSION
    )
    run_sequence_scorer(
        eval_name=EVAL_NAME,
        validation_oracle=ORACLE,
        designs=FLAGS.designs,
        out=FLAGS.out,
        per_design_out=FLAGS.per_design_out,
        required_n=FLAGS.required_n,
        run_meta=FLAGS.run_meta,
        run_root=run_root,
        spec=SequenceSubmissionSpec(
            required_n=FLAGS.required_n,
            chains=(DESIGN_CHAIN,),
            molecule_type="rna",
            sequence_length_range=LENGTH_RANGE,
            allow_motif_indices=False,
            motif_reference_ids=MOTIF_REFERENCE_IDS,
            motif_residue_types=MOTIF_RESIDUE_TYPES,
            motif_contiguity_groups=MOTIF_CONTIGUITY_GROUPS,
            motif_map_format="zero_based_indices",
            max_pairwise_identity=MAX_PAIRWISE_IDENTITY,
        ),
        ligands=LIGANDS,
        oracle=AF3(name="score_af3"),
        oracle_job_args=GPU_ARGS,
        metrics=[
            # AF3 does not place Mn2+ reliably even for the native sequence.
            # Excluding it is what makes the native motif RMSD 0.49 A instead
            # of 3.08 A; metal contacts remain reported diagnostics below.
            RefMotifRMSD(
                name="score_motif_rmsd", reference=str(MOTIF), ligands=[]
            ),
            MotifGeometry(name="score_geometry", measurements=CATALYTIC),
            MotifConfidence(name="score_confidence"),
            # Global fold similarity is diagnostic only: the campaign sits at
            # the RNA random-pair baseline and its catalytic hits do not recover
            # the native global fold.
            TMScoreRNA(name="score_tm", reference=str(NATIVE)),
            NoveltyGate(
                name="score_novelty",
                references=str(HOMOLOGS),
                threshold=NOVELTY_MAX_IDENTITY,
                compute_intra_identity=True,
            ),
            # T6 asks whether each mapped catalytic nucleotide is paired in the
            # AF3 refold and whether partner type / DSSR classification matches
            # any annotation in native 2OEU. It never gates: the prior Biotite
            # native-refold control reached only five of six, and the DSSR
            # distribution still needs its own calibration.
            DSSRMotifBasePairing(
                name="t6",
                reference=str(NATIVE),
                residues=list(MOTIF_REFERENCE_IDS),
                continue_on_error=True,
            ),
        ],
        metric_job_args=METRIC_ARGS,
        gates=GATES,
        metric_columns=METRIC_COLUMNS,
        diagnostic_columns=T6_DIAGNOSTIC_COLUMNS,
        extra=_report_metadata,
    )


if __name__ == "__main__":
    main()
