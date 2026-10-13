#!/usr/bin/env -S bash -c '"${PIPELINES_REPO:-.}/.venv/bin/python" "$0" "$@"'
"""Run the core biotin example and export a submission of selected binders.

The eval contract is a fixed count (96), not "however many passed". Filter
survivors are always taken first; if there are fewer than the required count,
the table is topped up with the best remaining AF3-scored designs, ranked by
holo interface quality. The printed summary says how many rows actually cleared
every filter, so a thin run is never mistaken for a strong one.

For a one-sequence local smoke test:

    ./example_pipeline.py --test --required_n=1 --local --use_local_gpus
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import polars as pl

from pipelines import PipelineDataHandler

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
CORE_PIPELINE = (REPO_ROOT / "examples" / "biotin-binder" / "pipeline.py").resolve()
DEFAULT_RUNDIR = (SCRIPT_DIR / "rundir").resolve()
SUBMISSION = (SCRIPT_DIR / "submission.csv").resolve()

# Design pools, best first. Each is a step of the core pipeline.
POOLS = ("apo_filter", "af3_final_filter", "af3_final_rmsd")

# Ranking for the top-up pools: a confident, well-satisfied, well-packed
# interface first. (column, ascending)
RANK_BY = [
    ("af3_final.min_pae_interaction", True),
    ("af3_final_hbond.num_hbonds", False),
    ("af3_final_contact_ms", False),
]


def _ranked_handler(rundir: Path, step: str) -> PipelineDataHandler | None:
    """Load a pipeline step, ranked best-first. ``None`` if it produced nothing."""
    output = rundir / step / "output.parquet"
    if not output.is_file():
        return None
    handler = PipelineDataHandler.from_file(output)
    if len(handler) == 0:
        return None

    frame = handler.unpack(all=True)
    columns = [(c, asc) for c, asc in RANK_BY if c in frame.columns]
    if not columns:
        return handler
    order = frame.sort_values(
        by=[c for c, _ in columns], ascending=[asc for _, asc in columns]
    ).index
    packed = handler.to_pandas().iloc[order].reset_index(drop=True)
    return PipelineDataHandler(packed)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--required_n", type=int, default=96)
    parser.add_argument("--rundir", type=Path, default=DEFAULT_RUNDIR)
    parser.add_argument(
        "--skip_pipeline",
        action="store_true",
        help="Reuse an existing rundir instead of re-running the core pipeline.",
    )
    args, pipeline_args = parser.parse_known_args()

    rundir = args.rundir.resolve()
    if not args.skip_pipeline:
        subprocess.run(
            [sys.executable, str(CORE_PIPELINE), f"--rundir={rundir}", *pipeline_args],
            check=True,
        )

    records: list[dict] = []
    seen: set[str] = set()
    n_fully_passing = 0

    for step in POOLS:
        if len(records) >= args.required_n:
            break
        handler = _ranked_handler(rundir, step)
        if handler is not None:
            for loaded in handler.get_data(which="last", load=True):
                if len(records) >= args.required_n:
                    break
                sequence = loaded.sequence.get("A")
                if not sequence or sequence in seen:
                    continue
                seen.add(sequence)
                records.append(
                    {
                        "design_id": f"biotin_{len(records):04d}",
                        "sequence": sequence,
                    }
                )
        if step == POOLS[0]:
            n_fully_passing = len(records)

    if len(records) < args.required_n:
        raise RuntimeError(
            f"Only {len(records)} unique sequences available across {list(POOLS)}; "
            f"need {args.required_n}. Scale up the run (--rfd3.n_batches)."
        )

    submission = pl.DataFrame(records)
    if submission["sequence"].n_unique() != len(submission):
        raise ValueError("Selected submission contains duplicate sequences")
    submission.write_csv(SUBMISSION)

    subprocess.run(["chmod", "-R", "g+w", str(rundir)], check=True)
    subprocess.run(["chmod", "g+w", str(SUBMISSION)], check=True)
    print(submission)
    print(
        f"\nWrote {len(submission)} biotin-binder sequences to {SUBMISSION}\n"
        f"  {n_fully_passing} cleared every filter; "
        f"{len(submission) - n_fully_passing} are ranked top-ups."
    )


if __name__ == "__main__":
    main()
