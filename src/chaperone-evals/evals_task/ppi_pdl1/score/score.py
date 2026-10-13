#!/usr/bin/env -S bash -c '"${CHAPERONE_REPO:-.}/.venv/bin/python" "$0" "$@"'
"""Held-out scorer for sequence-only human PD-L1 binder submissions.

Supplies the canonical PD-L1 target, predicts every complex with AF3, and
gates on interface confidence, binder secondary structure, and PPI ddG.

Usage:
    ./score.py --designs=/absolute/path/to/submission.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from absl import flags

from blocks import Blocks
from chaperone import init_cli
from scoring import (
    SequenceSubmissionSpec,
    define_score_flags,
    metric_series,
    run_sequence_scorer,
)

define_score_flags()
FLAGS = flags.FLAGS

init_cli()

EVAL_NAME = "ppi-pdl1"
ORACLE = "af3_with_target_template"

SCRIPT_DIR = Path(__file__).resolve().parent
TARGET_PDB = (SCRIPT_DIR.parent / "inputs" / "target.pdb").resolve()

BINDER_CHAIN = "A"
TARGET_CHAIN = "B"

IPAE_MAX = 7.0
MIN_PAE_MAX = 1.25
FRACTION_LOOP_MAX = 0.30
DDG_MAX = -25.0
MIN_PAE_GOOD = MIN_PAE_MAX
MIN_PAE_EXCELLENT = 1.0

FRACTION_LOOP_COL = "score_secondary_structure.fraction_loop"
DDG_COL = "score_ddg.ddg"
IPAE_COL = "iface.ipae"
MIN_PAE_COL = "iface.min_pae"

# The DIGS `gpu` partition rejects a typeless `gpu:1` request ("submitting to a
# regular gpu queue but did not specify a GPU type"), so the type is mandatory.
GPU_ARGS = {
    "partition": "gpu",
    "gres": "gpu:small:1",
    "mem_gb": 32,
    "time_limit": "6:00:00",
    # g2132 currently advertises an available GPU whose memory is already in
    # use. The cluster TaskProlog rejects allocations there before the worker
    # starts, and Submitit receives no result pickle to drive normal retry.
    "exclude": "g2132",
}


def _interface_pae(structure_path: str) -> tuple[float, float]:
    """(mean, min) symmetric binder↔target PAE from AF3's confidence output."""
    path = Path(structure_path)
    suffix = "_model.cif"
    if not path.name.endswith(suffix):
        raise ValueError(f"AF3 output does not end in {suffix}: {path}")
    confidence_path = path.with_name(path.name[: -len(suffix)] + "_confidences.json")
    if not confidence_path.exists():
        raise FileNotFoundError(
            f"Missing AF3 confidences for {path}: {confidence_path}"
        )

    data = json.loads(confidence_path.read_text())
    pae = np.asarray(data.get("pae"), dtype=float)
    chain_ids = np.asarray(data.get("token_chain_ids"), dtype=str)
    if pae.shape != (len(chain_ids), len(chain_ids)):
        raise ValueError(
            f"PAE shape {pae.shape} does not match {len(chain_ids)} token chains"
        )

    binder = np.flatnonzero(chain_ids == BINDER_CHAIN)
    target = np.flatnonzero(chain_ids == TARGET_CHAIN)
    if binder.size == 0 or target.size == 0:
        raise ValueError(
            f"AF3 confidence output is missing chain {BINDER_CHAIN} or {TARGET_CHAIN}"
        )

    values = np.concatenate(
        [pae[np.ix_(binder, target)].ravel(), pae[np.ix_(target, binder)].ravel()]
    )
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Interface PAE contains no values or non-finite values")
    return float(values.mean()), float(values.min())


def measure_interface_pae(handler, submission):
    """Symmetric binder↔target interface PAE, read from AF3's own confidences.

    Returned as ``iface.ipae`` / ``iface.min_pae`` metrics so the gates address
    them like any other. This is the hand-rolled computation this benchmark
    calibrated against — deliberately NOT the ipsae block, whose numbers differ.
    The AF3 output path is joined on ``design_id``, never by row position.
    """
    frame = handler.unpack(all=True)
    values = {}
    for design_id, path in zip(frame["design_id"], frame["score_af3.path"]):
        try:
            ipae, min_pae = _interface_pae(path)
        except Exception as exc:
            raise RuntimeError(f"interface PAE failed for {design_id}: {exc}") from exc
        values[design_id] = {"iface": {"ipae": ipae, "min_pae": min_pae}}
    return values


def report_extra(frame):
    """Report min-PAE count summaries in addition to the per-design gate."""
    min_pae = metric_series(frame, MIN_PAE_COL)
    return {
        "n_min_pae_good": int((min_pae <= MIN_PAE_GOOD).sum()),
        "n_min_pae_excellent": int((min_pae <= MIN_PAE_EXCELLENT).sum()),
        "target": TARGET_PDB.name,
    }


def main() -> None:
    AF3, SecondaryStructure, PPIDDG = Blocks.getall(
        ORACLE, "secondary_structure", "ppi_ddg"
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
        oracle=AF3(
            name="score_af3",
            target_chain=TARGET_CHAIN,
            full_receptor_pdb=str(TARGET_PDB),
            diffusion_batch_size=1,
        ),
        oracle_job_args=GPU_ARGS,
        metrics=[
            SecondaryStructure(name="score_secondary_structure", chains=[BINDER_CHAIN]),
            PPIDDG(
                name="score_ddg",
                binder_chain=BINDER_CHAIN,
                target_chain=TARGET_CHAIN,
                relax_rounds=1,
                replicates=1,
            ),
        ],
        # Symmetric interface PAE is hand-rolled from AF3's confidence JSON.
        measure=measure_interface_pae,
        # Gates are filter-expression strings (the repo's FilterBlock DSL); each
        # threshold lives in the expression that enforces it.
        gates={
            "ipae": f"{IPAE_COL} <= {IPAE_MAX}",
            "min_pae": f"{MIN_PAE_COL} <= {MIN_PAE_MAX}",
            "fraction_loop": f"{FRACTION_LOOP_COL} <= {FRACTION_LOOP_MAX}",
            "ddg": f"{DDG_COL} <= {DDG_MAX}",
        },
        metric_columns=[IPAE_COL, MIN_PAE_COL, FRACTION_LOOP_COL, DDG_COL],
        extra=report_extra,
    )


if __name__ == "__main__":
    main()
