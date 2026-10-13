from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_eval
import score_submission
from task_paths import PRIVATE_TASKS_ENV

TEST_SITE_ENV = {
    name: f"/trusted/{name.lower()}" for name in run_eval._SCORING_SITE_ENV_NAMES
}


class ScoreSubmissionTest(unittest.TestCase):
    def _task(self, root: Path, name: str) -> Path:
        task = root / name
        (task / "score").mkdir(parents=True)
        (task / "score" / "score.py").write_text("print('score')\n")
        return task

    def test_rescue_accepts_task_specific_sequence_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            designs = Path(temporary) / "designs.csv"
            designs.write_text("sequence_H,sequence_L\nAAAA,CCCC\n")
            score_submission._readable_designs(designs)

    def test_infers_task_from_trial_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tasks = root / "tasks"
            self._task(tasks, "antibody_cdr")
            job = root / "job"
            trial = job / "trial"
            trial.mkdir(parents=True)
            (trial / "config.json").write_text(
                json.dumps({"task_name": "local/antibody_cdr"})
            )
            with patch.dict(os.environ, {PRIVATE_TASKS_ENV: str(tasks)}):
                self.assertEqual(score_submission._task_from_job(job), "antibody_cdr")

    def test_task_resolution_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tasks = Path(temporary) / "tasks"
            self._task(tasks, "ppi_pdl1")
            with (
                patch.dict(os.environ, {PRIVATE_TASKS_ENV: str(tasks)}),
                self.assertRaisesRegex(SystemExit, "No scorer"),
            ):
                score_submission._resolve_task("../ppi_pdl1")

    def test_selects_delivered_submission_without_finalized_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary) / "job"
            trial = job / "trial"
            submission = trial / "workspace" / "submission"
            submission.mkdir(parents=True)
            (trial / "config.json").write_text("{}")
            (submission / "designs.csv").write_text("sequence\nAAAA\n")

            self.assertEqual(score_submission._select_trial(job, None), trial)

    def test_clean_trial_is_preferred_over_delivered_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary) / "job"
            delivered = job / "delivered"
            delivered_submission = delivered / "workspace" / "submission"
            delivered_submission.mkdir(parents=True)
            (delivered / "config.json").write_text("{}")
            (delivered_submission / "designs.csv").write_text("sequence\nAAAA\n")

            completed = job / "completed"
            completed.mkdir()
            (completed / "config.json").write_text("{}")
            (completed / "result.json").write_text(
                json.dumps({"exception_info": None, "agent_execution": {"ok": True}})
            )

            self.assertEqual(score_submission._select_trial(job, None), completed)

    def test_rescue_scoring_enables_exact_nested_rundir_bind(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tasks = root / "tasks"
            task = self._task(tasks, "biotin_binder_design")
            (task / "inputs").mkdir()
            (task / "inputs" / "acceptance_contract.json").write_text(
                json.dumps(
                    {
                        "required_n": 1,
                        "scorer_sha256": "a" * 64,
                    }
                )
            )
            job = root / "job"
            trial = job / "trial"
            submission = trial / "artifacts" / "workspace" / "submission"
            submission.mkdir(parents=True)
            (trial / "config.json").write_text(
                json.dumps({"task_name": "local/biotin_binder_design"})
            )
            (trial / "result.json").write_text(
                json.dumps({"exception_info": None, "agent_execution": {"ok": True}})
            )
            (submission / "designs.csv").write_text("sequence\nAAAA\n")
            repo = root / "chaperone"
            (repo / "src").mkdir(parents=True)
            host_python = repo / ".venv" / "bin" / "python"
            host_python.parent.mkdir(parents=True)
            host_python.touch()
            image = root / "worker.sif"
            image.touch()
            retry = trial / "scoring" / "retry"

            argv = [
                "score_submission.py",
                "--job",
                str(job),
                "--task",
                "biotin_binder_design",
                "--chaperone-repo",
                str(repo),
                "--worker-image",
                str(image),
                "--required-n",
                "1",
                "--out",
                str(retry / "score.json"),
                "--rundir",
                str(retry / "score_run"),
            ]
            with (
                patch.dict(os.environ, {PRIVATE_TASKS_ENV: str(tasks)}),
                patch.object(sys, "argv", argv),
                patch.object(
                    score_submission, "_host_scoring_python", return_value=host_python
                ),
                patch.object(score_submission, "_prepare_scoring_packages"),
                patch.object(
                    score_submission,
                    "_scoring_site_environment",
                    return_value=TEST_SITE_ENV.copy(),
                ),
                patch.object(
                    score_submission.ApptainerEnvironment,
                    "_prepare_runtime_identity",
                ),
                patch.object(
                    score_submission.ApptainerEnvironment,
                    "_prepare_slurm_python",
                ) as prepare_slurm,
                patch.object(score_submission, "_run_score_command") as run_score,
            ):
                score_submission.main()

            environment = run_score.call_args.kwargs["environment"]
            command = run_score.call_args.args[0]
            self.assertEqual(
                environment[score_submission._SCORING_RUNDIR_BIND_ENV], "true"
            )
            self.assertEqual(environment["RUNDIR"], str(retry / "score_run"))
            self.assertEqual(
                environment[score_submission._SCORING_USALIGN_ENV],
                str(repo / ".venv" / "bin" / "USalign"),
            )
            expected_jax_cache = retry / "score_run" / ".cache" / "af3_jax"
            self.assertEqual(
                environment[score_submission._SCORING_AF3_JAX_CACHE_ENV],
                str(expected_jax_cache),
            )
            for name in run_eval._SCORING_SITE_ENV_NAMES:
                self.assertEqual(environment[name], TEST_SITE_ENV[name])
            for name, value in score_submission._CONTROLLER_THREAD_ENV.items():
                self.assertEqual(environment[name], value)
            self.assertTrue(expected_jax_cache.is_dir())
            self.assertIn(
                f"--per_design_out={retry / 'score_per_design.csv'}",
                command,
            )
            prepare_slurm.assert_called_once()
            self.assertEqual(
                prepare_slurm.call_args.args[0], retry / ".runtime" / "slurm-python"
            )
            self.assertEqual(
                prepare_slurm.call_args.kwargs["site_environment"],
                {
                    **TEST_SITE_ENV,
                    score_submission._SCORING_AF3_JAX_CACHE_ENV: str(
                        expected_jax_cache
                    ),
                },
            )
            self.assertIn(
                f"--run_meta={trial / 'controller' / 'run_meta.json'}",
                command,
            )


if __name__ == "__main__":
    unittest.main()
