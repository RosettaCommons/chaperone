#!/usr/bin/env -S bash -c 'exec "$(dirname "$0")/../.venv/bin/python" "$0" "$@"'
"""Re-run the held-out scorer over trials whose scoring did not complete.

`run_eval.py` scores automatically once its Slurm allocation finishes, but that
pass is a single shot: it refuses to overwrite an existing `scoring/` folder and
aborts the whole run on the first one it meets. When scoring fails for a reason
unrelated to the agents — a broken interpreter, a container that would not
start, a scheduler outage — the designs are already on disk and only the scorer
needs rerunning. That is what this does.

Classification is per trial:

  score  - no `scoring/` folder; run the scorer
  skip   - `scoring/score.json` exists; already scored, leave it alone
  clear  - `scoring/` exists without a `score.json`; wreckage of a failed
           attempt, removed first with --clear-empty-scoring, then scored

A trial needing `clear` without that flag is reported as blocked and the run
exits non-zero, so a stale folder is never silently skipped.

    ./scripts/rescore.py --task ppi_pdl1 --dry-run \\
      --run-dir /absolute/path/to/harbor-run-directory

Scoring submits GPU work and takes hours per trial; run it detached.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(os.path.abspath(__file__)).parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from task_paths import task_scorer  # noqa: E402
from run_eval import (  # noqa: E402
    DEFAULT_HARNESS_IMAGE,
    DEFAULT_PIPELINES_REPO,
    _scan_trials,
    _score_trial,
    resolve_task,
)

SCORE = "score"
SKIP = "skip"
CLEAR = "clear"


def _classify(trial: Path) -> str:
    scoring = trial / "scoring"
    if not scoring.exists():
        return SCORE
    if (scoring / "score.json").is_file():
        return SKIP
    return CLEAR


def _plan(run_dir: Path) -> list[tuple[Path, str]]:
    """Successful trials in a run folder, paired with what they need."""
    if not run_dir.is_dir():
        raise SystemExit(f"Run folder not found: {run_dir}")
    # Deliberately not _successful_trials(): its NFS settle loop exists for the
    # moment a run finishes, and it raises when a run has no successful trial.
    # Here a run whose trials are all still in flight is normal, not an error.
    successful, failures, _ = _scan_trials(run_dir)
    if failures:
        print(f"  ignoring {len(failures)} failed trial(s): {', '.join(failures)}")
    return [(trial, _classify(trial)) for trial in successful]


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-run the held-out scorer over unscored Harbor trials."
    )
    parser.add_argument("--task", required=True, help="Folder name under evals_task")
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        required=True,
        dest="run_dirs",
        metavar="PATH",
        help="A Harbor run folder; repeat for several.",
    )
    parser.add_argument("--pipelines-repo", type=Path, default=DEFAULT_PIPELINES_REPO)
    parser.add_argument(
        "--worker-image",
        "--scoring-image",
        dest="worker_image",
        type=Path,
        default=DEFAULT_HARNESS_IMAGE,
        help=(
            "Pipelines image used by scientific Slurm workers; the scorer "
            "itself runs in the host Pipelines .venv"
        ),
    )
    parser.add_argument("--required-n", type=_positive_integer, default=96)
    parser.add_argument(
        "--clear-empty-scoring",
        action="store_true",
        help=(
            "Delete a scoring/ folder that holds no score.json before rescoring "
            "it. A folder containing a score.json is never removed."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    task = resolve_task(args.task)
    scorer = task_scorer(task.name)
    pipelines_repo = Path(os.path.abspath(args.pipelines_repo.expanduser()))
    image = args.worker_image.expanduser().resolve()

    print(f"Task: {task}")
    print(f"Pipelines repo: {pipelines_repo}")
    print(f"Scientific worker image: {image}", flush=True)

    planned: list[tuple[Path, str]] = []
    blocked: list[Path] = []
    for run_dir in args.run_dirs:
        run_dir = run_dir.expanduser().resolve()
        print(f"\nRun: {run_dir}")
        for trial, action in _plan(run_dir):
            if action == CLEAR and not args.clear_empty_scoring:
                blocked.append(trial)
                print(f"  BLOCKED {trial.name}: scoring/ exists without a score.json")
                continue
            print(f"  {action:<6} {trial.name}")
            if action != SKIP:
                planned.append((trial, action))

    if blocked:
        raise SystemExit(
            f"\n{len(blocked)} trial(s) have a stale scoring/ folder. Re-run with "
            "--clear-empty-scoring to discard those partial attempts."
        )
    if not planned:
        print("\nNothing to score.")
        return
    print(f"\n{len(planned)} trial(s) to score.", flush=True)
    if args.dry_run:
        print("Dry run; nothing was scored.")
        return

    failed: list[str] = []
    for trial, action in planned:
        if action == CLEAR:
            scoring = trial / "scoring"
            # Re-check rather than trust the plan: scoring is long, and the
            # run's own controller may have written a report in the meantime.
            if (scoring / "score.json").is_file():
                print(f"Skipping {trial.name}: it was scored since the plan was made.")
                continue
            print(f"Clearing stale scoring folder: {scoring}", flush=True)
            shutil.rmtree(scoring)
        try:
            _score_trial(
                scorer, trial, pipelines_repo, image, required_n=args.required_n
            )
        except SystemExit as exc:
            failed.append(trial.name)
            print(f"Scoring {trial.name} failed: {exc}", file=sys.stderr, flush=True)

    if failed:
        raise SystemExit(f"Scoring failed for: {', '.join(failed)}")
    print("\nAll planned trials scored.")


if __name__ == "__main__":
    main()
