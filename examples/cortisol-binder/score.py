"""Shared cortisol gates for cycling, final filtering, and score reports."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from blocks.util_blocks.filter import MetricFilter
from chaperone.evals import (
    Criterion,
    ScoreReport,
    evaluate_criteria,
    write_score_outputs,
)

CYCLE_MAX_PAE = 2.0
CYCLE_MIN_HBONDS = 2
FINAL_MAX_PAE = 1.25
FINAL_MIN_HBONDS = 4
FINAL_MIN_PLDDT = 90.0


def gate_filters(af3_step: str, hbond_step: str, *, final: bool = False) -> dict:
    # Finite bounds also reject NaN/inf in the engine's filter.
    filters = {
        f"{af3_step}.min_pae_interaction": MetricFilter(
            min=0, max=FINAL_MAX_PAE if final else CYCLE_MAX_PAE
        ),
        f"{hbond_step}.num_hbonds": MetricFilter(
            min=FINAL_MIN_HBONDS if final else CYCLE_MIN_HBONDS,
            max=sys.float_info.max,
        ),
    }
    if final:
        filters[f"{af3_step}.plddt"] = MetricFilter(min=FINAL_MIN_PLDDT, max=100)
    return filters


def score_frame(frame: pd.DataFrame, af3_step: str, hbond_step: str, *, final=False):
    """Evaluate only the named prediction state; missing metrics always fail."""
    frame = frame.copy()
    if "design_id" not in frame:
        frame["design_id"] = pd.Series(index=frame.index, dtype="string")
    criteria = []
    filters = gate_filters(af3_step, hbond_step, final=final)
    for column, bounds in filters.items():
        values = pd.to_numeric(
            frame.get(column, pd.Series(np.nan, index=frame.index)), errors="coerce"
        )
        frame[column] = values
        passed = np.isfinite(values) & values.between(bounds.min, bounds.max)
        criteria.append(Criterion(column.rsplit(".", 1)[-1], passed))
    return evaluate_criteria(frame, criteria)


def write_scores(handler, outdir: Path, af3_step: str, hbond_step: str, *, final=False):
    """Save ordinary Chaperone per-design scores and a canonical report."""
    outdir.mkdir(parents=True, exist_ok=True)
    frame, counts = score_frame(handler.unpack(), af3_step, hbond_step, final=final)
    filters = gate_filters(af3_step, hbond_step, final=final)
    report = ScoreReport(
        eval="cortisol_final" if final else "cortisol_cycle",
        n_reported=len(frame),
        required_n=1,
        contract_ok=True,
        validation_oracle=af3_step,
        n_pass=int(frame["pass"].sum()),
        pass_rate=float(frame["pass"].mean()) if len(frame) else 0.0,
        n_soft_pass=int(frame["soft_pass"].sum()),
        soft_pass_rate=float(frame["soft_pass"].mean()) if len(frame) else 0.0,
        per_criterion=counts,
        thresholds={k: v.model_dump(exclude_none=True) for k, v in filters.items()},
    )
    write_score_outputs(
        report,
        frame,
        out=outdir / "report.json",
        per_design_out=outdir / "per_design.csv",
        metric_columns=list(filters),
    )
    return frame
