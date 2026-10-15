#!/usr/bin/env python3
"""Tests for converting Harbor trial statistics into a process exit status."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.check_harbor_result import ResultError, validate_result


def _result(*, total: int = 1, completed: int = 1, errors: int = 0) -> dict:
    return {
        "n_total_trials": total,
        "stats": {
            "n_completed_trials": completed,
            "n_errored_trials": errors,
            "n_running_trials": 0,
            "n_pending_trials": 0,
            "n_cancelled_trials": 0,
            "evals": {"codex": {"n_trials": completed - errors, "n_errors": errors}},
        },
    }


class HarborResultTest(unittest.TestCase):
    def _write(self, root: Path, value: object) -> Path:
        path = root / "result.json"
        path.write_text(json.dumps(value))
        return path

    def _write_trial(self, root: Path) -> None:
        trial = root / "biotin_binder_design_test__test"
        submission = trial / "workspace" / "submission"
        submission.mkdir(parents=True)
        (trial / "result.json").write_text(
            json.dumps({"exception_info": None, "agent_execution": {"ok": True}})
        )
        sequences = ["A" * (index + 1) + "C" for index in range(10)]
        (submission / "designs.csv").write_text(
            "sequence\n" + "\n".join(sequences) + "\n"
        )

    def test_accepts_clean_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_trial(root)
            message = validate_result(self._write(root, _result()))
            self.assertIn("no errors", message)

    def test_leaves_task_specific_csv_validation_to_scorer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_trial(root)
            designs = next(root.glob("*/workspace/submission/designs.csv"))
            designs.write_text("candidate\none\n")
            message = validate_result(self._write(root, _result()))
            self.assertIn("valid submission", message)

    def test_rejects_trial_error_even_when_harbor_completed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self._write(Path(temporary), _result(errors=1))
            with self.assertRaisesRegex(ResultError, "errored_trials=1"):
                validate_result(path)

    def test_rejects_missing_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ResultError, "was not written"):
                validate_result(Path(temporary) / "result.json")


if __name__ == "__main__":
    unittest.main()
