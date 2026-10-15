#!/usr/bin/env python3
"""Return nonzero unless a Harbor job result completed without trial errors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


class ResultError(RuntimeError):
    """A Harbor result is missing, malformed, incomplete, or unsuccessful."""


def _validate_submission(trial_dir: Path) -> None:
    candidates = (
        trial_dir / "workspace" / "submission",
        trial_dir / "artifacts" / "workspace" / "submission",
    )
    submission = next(
        (path for path in candidates if (path / "designs.csv").is_file()), None
    )
    if submission is None:
        raise ResultError(
            f"Successful trial has no submission/designs.csv: {trial_dir}"
        )

    try:
        if not (submission / "designs.csv").read_text().strip():
            raise ResultError(f"Submission designs.csv is empty: {submission}")
    except OSError as exc:
        raise ResultError(f"Could not read submitted designs: {exc}") from exc


def validate_result(path: Path) -> str:
    try:
        result = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ResultError(f"Harbor result was not written: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultError(f"Could not read Harbor result {path}: {exc}") from exc
    if not isinstance(result, dict):
        raise ResultError(f"Harbor result is not a JSON object: {path}")

    total = result.get("n_total_trials")
    stats = result.get("stats")
    if not isinstance(total, int) or total < 1 or not isinstance(stats, dict):
        raise ResultError(f"Harbor result has invalid trial statistics: {path}")

    fields = {
        name: stats.get(name)
        for name in (
            "n_completed_trials",
            "n_errored_trials",
            "n_running_trials",
            "n_pending_trials",
            "n_cancelled_trials",
        )
    }
    if not all(isinstance(value, int) for value in fields.values()):
        raise ResultError(f"Harbor result has incomplete trial statistics: {path}")

    failures = []
    if fields["n_completed_trials"] != total:
        failures.append(f"completed={fields['n_completed_trials']} of total={total}")
    for name in (
        "n_errored_trials",
        "n_running_trials",
        "n_pending_trials",
        "n_cancelled_trials",
    ):
        if fields[name]:
            failures.append(f"{name.removeprefix('n_')}={fields[name]}")

    evals = stats.get("evals")
    if isinstance(evals, dict):
        for eval_name, eval_stats in evals.items():
            if isinstance(eval_stats, dict) and eval_stats.get("n_errors"):
                failures.append(f"{eval_name}.errors={eval_stats['n_errors']}")

    if failures:
        raise ResultError("Harbor job failed: " + ", ".join(failures))

    successful_trials = []
    for candidate in path.parent.iterdir():
        trial_result_path = candidate / "result.json"
        if not candidate.is_dir() or not trial_result_path.is_file():
            continue
        try:
            trial_result = json.loads(trial_result_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if (
            isinstance(trial_result, dict)
            and trial_result.get("exception_info") is None
            and trial_result.get("agent_execution") is not None
        ):
            successful_trials.append(candidate)
    if not successful_trials:
        raise ResultError(
            "Harbor reported completion but no clean agent trial was found"
        )
    for trial_dir in successful_trials:
        _validate_submission(trial_dir)
    return (
        f"Harbor result validated: {total} trial(s), no errors, "
        f"{len(successful_trials)} valid submission(s)"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(validate_result(args.result.expanduser().resolve()))
    except ResultError as exc:
        print(exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
