from __future__ import annotations

import json
import io
import hashlib
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run_eval

TEST_SITE_ENV = {
    name: f"/trusted/{name.lower()}" for name in run_eval._SCORING_SITE_ENV_NAMES
}


def _fake_site_environment(_repo):
    return TEST_SITE_ENV.copy()


class RunEvalTest(unittest.TestCase):
    def test_runtime_manifest_reports_optional_capabilities_without_requiring_them(
        self,
    ) -> None:
        manifest = run_eval._runtime_capability_manifest(None)
        capabilities = manifest["capabilities"]
        names = {item["name"] for item in capabilities}
        self.assertGreaterEqual(
            names,
            {
                *run_eval._HARNESS_RUNTIME_CAPABILITIES,
                "campaign-monitor",
            },
        )
        self.assertTrue(
            all(isinstance(item["available"], bool) for item in capabilities)
        )

    def test_harness_requires_stateless_monitor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task = self._task(Path(temporary))
            required = run_eval._required_harness_capabilities(task, None)

        self.assertEqual(required, {"campaign-monitor"})

    def test_runtime_manifest_uses_chaperone_campaign_monitor_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            seed = Path(temporary)
            backend = seed / "src" / "chaperone_cli" / "job_wait.py"
            backend.parent.mkdir(parents=True)
            backend.write_text(
                "def wait_for_completion(): pass\n"
                "def launch_and_wait(): pass\n"
                "def inspect_campaign(): pass\n"
                "def serve_mcp(): pass\n"
            )
            manifest = run_eval._runtime_capability_manifest(seed)

        monitor = next(
            item
            for item in manifest["capabilities"]
            if item["name"] == "campaign-monitor"
        )
        self.assertTrue(monitor["available"])
        self.assertEqual(monitor["version"], "chaperone-monitor-v2.1")
        self.assertEqual(monitor["tools"], ["wait", "launch_and_wait", "inspect"])

    def test_shared_job_parents_are_precreated_concurrently(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            jobs_root = Path(temporary) / "harness-vs-baseline"
            with patch.object(run_eval, "JOBS_ROOT", jobs_root):
                with ThreadPoolExecutor(max_workers=8) as executor:
                    list(
                        executor.map(
                            lambda _: run_eval._prepare_shared_job_directories(),
                            range(32),
                        )
                    )

            for name in run_eval._SHARED_JOB_DIRECTORIES:
                self.assertTrue((jobs_root / name).is_dir(), name)

    def test_harness_requires_explicit_chaperone_path(self) -> None:
        stderr = io.StringIO()
        with (
            patch.object(
                run_eval.sys,
                "argv",
                ["run_eval.py", "ppi_pdl1", "--arm", "harness", "--dry-run"],
            ),
            redirect_stderr(stderr),
            self.assertRaises(SystemExit) as raised,
        ):
            run_eval.main()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("requires an explicit --chaperone-path", stderr.getvalue())

    def test_baseline_rejects_chaperone_path(self) -> None:
        stderr = io.StringIO()
        with (
            patch.object(
                run_eval.sys,
                "argv",
                [
                    "run_eval.py",
                    "ppi_pdl1",
                    "--arm",
                    "baseline",
                    "--chaperone-path",
                    "/tmp/chaperone",
                    "--dry-run",
                ],
            ),
            redirect_stderr(stderr),
            self.assertRaises(SystemExit) as raised,
        ):
            run_eval.main()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("not accepted by the baseline arm", stderr.getvalue())

    def test_memory_policy_option_has_been_removed(self) -> None:
        stderr = io.StringIO()
        with (
            patch.object(
                run_eval.sys,
                "argv",
                [
                    "run_eval.py",
                    "ppi_pdl1",
                    "--arm",
                    "baseline",
                    "--memory-policy",
                    "ledger",
                    "--dry-run",
                ],
            ),
            redirect_stderr(stderr),
            self.assertRaises(SystemExit) as raised,
        ):
            run_eval.main()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("unrecognized arguments", stderr.getvalue())

    def test_external_scoring_refuses_a_slurm_submission_host(self) -> None:
        with (
            patch.dict(run_eval.os.environ, {"SLURM_JOB_ID": "12345"}),
            self.assertRaisesRegex(SystemExit, "submission host"),
        ):
            run_eval._preflight(SimpleNamespace(no_score=False), Path("/unused/task"))

    def test_agent_runtime_probe_has_private_writable_runtime_directories(self) -> None:
        completed = SimpleNamespace(returncode=0, stdout="", stderr="")
        with (
            patch.object(run_eval, "_find_codex_vendor", return_value=Path("/vendor")),
            patch.object(run_eval.subprocess, "run", return_value=completed) as run,
        ):
            run_eval._probe_agent_runtime(Path("/agent.sif"), "harness")

        command = run.call_args.args[0]
        private_tmp_bind = next(
            argument
            for argument in command
            if argument.endswith(":/tmp") and argument != "/vendor:/tmp"
        )
        host_tmp, container_tmp = private_tmp_bind.rsplit(":", 1)
        self.assertTrue(Path(host_tmp).name.startswith("harbor-agent-runtime-"))
        self.assertEqual(container_tmp, "/tmp")
        shell = command[-1]
        self.assertIn("mkdir -p /tmp/home/.cache", shell)
        self.assertIn("HOME=/tmp/home", shell)
        self.assertIn("TMPDIR=/tmp", shell)
        self.assertIn("MPLCONFIGDIR=/tmp/home/.config/matplotlib", shell)
        self.assertIn("OPENBLAS_NUM_THREADS=1", shell)
        self.assertIn("OMP_NUM_THREADS=1", shell)

    def _task(
        self,
        root: Path,
        name: str = "example_task",
        prompt: str = "Do the evaluation.\n",
    ) -> Path:
        task = root / name
        (task / "inputs").mkdir(parents=True)
        (task / "score").mkdir()
        (task / "prompt.md").write_text(prompt)
        (task / "score" / "score.py").write_text("print('score')\n")
        return task

    def test_resolves_exact_task_folder(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = self._task(root)
            self.assertEqual(run_eval.resolve_task("example_task", root), task)

    def test_rejects_task_path_instead_of_folder_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._task(root)
            with self.assertRaisesRegex(SystemExit, "Task name must start"):
                run_eval.resolve_task("../example_task", root)

    def test_materializes_harbor_wrapper_with_task_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = self._task(root / "tasks")
            instruction = root / "instruction.md"
            instruction.write_text(
                "Shared instructions: {{REQUIRED_N}} designs, "
                "{{AGENT_HOURS}} hours under campaigns/{{TASK_NAME}}.\n"
            )
            setup = root / "setup.sh"
            setup.write_text("export TEST_SETUP=1\n")
            controller = root / "controller"
            image = root / "worker.sif"
            image.touch()
            with (
                patch.object(run_eval, "ROOT", controller),
                patch.object(run_eval, "JOBS_ROOT", controller / "jobs"),
                patch.dict(run_eval.ARM_INSTRUCTIONS, {"harness": instruction}),
                patch.object(run_eval, "ENVIRONMENT_SETUP", setup),
            ):
                wrapper = run_eval.materialize_harbor_task(
                    task, "test-run", "harness", image
                )
            self.assertEqual(
                wrapper, controller / "jobs" / "task-definitions" / "test-run"
            )
            self.assertIn(
                'name = "local/example_task"',
                (wrapper / "task.toml").read_text(),
            )
            self.assertEqual(
                (wrapper / "instruction.md").read_text(),
                "Shared instructions: 96 designs, 12 hours under "
                "campaigns/example_task.\n",
            )
            task_toml = (wrapper / "task.toml").read_text()
            self.assertIn("timeout_sec = 43200", task_toml)
            self.assertIn('HARBOR_AGENT_TIMEOUT_SEC = "43200"', task_toml)
            self.assertNotIn("MEMORY_POLICY", task_toml)
            self.assertIn('CHAPERONE_CONTAINER_BIND_RUNDIR = "true"', task_toml)
            self.assertIn('CHAPERONE_CONTAINER_BIND_WORKSPACE = "true"', task_toml)
            self.assertIn(
                'command = "/opt/pipelines/.venv/bin/python"',
                task_toml,
            )
            self.assertIn(
                'args = ["-m", "chaperone_cli.main", "monitor-mcp", '
                '"--server-name", "campaign-monitor"]',
                task_toml,
            )
            self.assertNotIn("/opt/harbor-controller", task_toml)
            generated_setup = wrapper / "environment" / "files" / "setup.sh"
            self.assertEqual(generated_setup.read_text(), "export TEST_SETUP=1\n")
            self.assertTrue(generated_setup.stat().st_mode & 0o100)
            self.assertTrue((wrapper / "tests" / "test.sh").stat().st_mode & 0o100)

    def test_wrapper_never_carries_the_scorer_or_reference_submission(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = self._task(root / "tasks")
            (task / "inputs" / "target.pdb").write_text("ATOM\n")
            (task / "example").mkdir()
            (task / "example" / "submission.csv").write_text("sequence\nMKV\n")
            instruction = root / "instruction.md"
            instruction.write_text("{{REQUIRED_N}} {{AGENT_HOURS}} {{TASK_NAME}}\n")
            setup = root / "setup.sh"
            setup.write_text("true\n")
            image = root / "worker.sif"
            image.touch()
            for arm in ("harness", "baseline"):
                with (
                    patch.object(run_eval, "ROOT", root / "controller"),
                    patch.object(run_eval, "JOBS_ROOT", root / "controller" / "jobs"),
                    patch.dict(run_eval.ARM_INSTRUCTIONS, {arm: instruction}),
                    patch.object(run_eval, "ENVIRONMENT_SETUP", setup),
                ):
                    wrapper = run_eval.materialize_harbor_task(
                        task, f"run-{arm}", arm, image
                    )
                names = {
                    path.relative_to(wrapper).as_posix()
                    for path in wrapper.rglob("*")
                    if path.is_file()
                }
                self.assertTrue(any(name.endswith("target.pdb") for name in names))
                for held_out in ("score.py", "submission.csv"):
                    self.assertFalse(
                        [name for name in names if name.endswith(held_out)], names
                    )

    def test_renders_required_count_into_task_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = self._task(
                root / "tasks",
                prompt="Submit exactly {{REQUIRED_N}} designs.\n",
            )
            instruction = root / "instruction.md"
            instruction.write_text(
                "Submit {{REQUIRED_N}} in {{AGENT_HOURS}} hours at "
                "campaigns/{{TASK_NAME}}.\n"
            )
            setup = root / "setup.sh"
            setup.write_text("export TEST_SETUP=1\n")
            image = root / "worker.sif"
            image.touch()
            with (
                patch.object(run_eval, "JOBS_ROOT", root / "jobs"),
                patch.dict(run_eval.ARM_INSTRUCTIONS, {"harness": instruction}),
                patch.object(run_eval, "ENVIRONMENT_SETUP", setup),
            ):
                wrapper = run_eval.materialize_harbor_task(
                    task, "rendered", "harness", image, required_n=10
                )

            self.assertEqual(
                (wrapper / "prompt.md").read_text(),
                "Submit exactly 10 designs.\n",
            )
            self.assertNotIn("{{", (wrapper / "instruction.md").read_text())

    def test_materializes_pinned_public_contract_and_capabilities(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = self._task(root / "tasks")
            scorer = task / "inputs" / "public_score.py"
            scorer.write_text("print('public score')\n")
            digest = hashlib.sha256(scorer.read_bytes()).hexdigest()
            (task / "inputs" / "acceptance_contract.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "campaign_id": "example_task",
                        "scorer_sha256": digest,
                        "required_n": 96,
                        "gates": [],
                    }
                )
            )
            instruction = root / "instruction.md"
            instruction.write_text("{{REQUIRED_N}} {{AGENT_HOURS}} {{TASK_NAME}}\n")
            setup = root / "setup.sh"
            setup.write_text("export TEST_SETUP=1\n")
            image = root / "worker.sif"
            image.touch()
            with (
                patch.object(run_eval, "JOBS_ROOT", root / "jobs"),
                patch.dict(run_eval.ARM_INSTRUCTIONS, {"harness": instruction}),
                patch.object(run_eval, "ENVIRONMENT_SETUP", setup),
            ):
                wrapper = run_eval.materialize_harbor_task(
                    task, "rendered", "harness", image, required_n=7
                )

            rendered = json.loads(
                (wrapper / "inputs" / "acceptance_contract.json").read_text()
            )
            capabilities = json.loads(
                (wrapper / "inputs" / "runtime_capabilities.json").read_text()
            )
            assert rendered["required_n"] == 7
            assert rendered["scorer_sha256"] == digest
            assert {item["name"] for item in capabilities["capabilities"]} >= {
                "campaign-monitor",
            }

    def test_materialized_harness_has_no_project_memory_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = self._task(root / "tasks")
            instruction = root / "instruction.md"
            instruction.write_text("{{REQUIRED_N}} {{AGENT_HOURS}} {{TASK_NAME}}\n")
            setup = root / "setup.sh"
            setup.write_text("export TEST_SETUP=1\n")
            image = root / "worker.sif"
            image.touch()
            with (
                patch.object(run_eval, "JOBS_ROOT", root / "jobs"),
                patch.dict(run_eval.ARM_INSTRUCTIONS, {"harness": instruction}),
                patch.object(run_eval, "ENVIRONMENT_SETUP", setup),
            ):
                wrapper = run_eval.materialize_harbor_task(
                    task,
                    "no-memory",
                    "harness",
                    image,
                )

            assert (wrapper / "instruction.md").read_text() == ("96 12 example_task\n")
            task_toml = (wrapper / "task.toml").read_text()
            assert "MEMORY_POLICY" not in task_toml
            capabilities = json.loads(
                (wrapper / "inputs" / "runtime_capabilities.json").read_text()
            )["capabilities"]
            by_name = {item["name"]: item for item in capabilities}
            assert "campaign-memory" not in by_name
            assert "campaign-cli" not in by_name

    def test_materialized_baseline_has_no_harness_runtime_features(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = self._task(root / "tasks")
            instruction = root / "baseline.md"
            instruction.write_text("{{REQUIRED_N}} in {{AGENT_HOURS}} hours\n")
            image = root / "worker.sif"
            image.touch()
            with (
                patch.object(run_eval, "JOBS_ROOT", root / "jobs"),
                patch.dict(run_eval.ARM_INSTRUCTIONS, {"baseline": instruction}),
            ):
                wrapper = run_eval.materialize_harbor_task(
                    task, "baseline", "baseline", image
                )

            task_toml = (wrapper / "task.toml").read_text()
            assert run_eval.DEFAULT_CONTROLLER_MEMORY_GB == 64
            assert "memory_mb = 65536" in task_toml
            assert "MEMORY_POLICY" not in task_toml
            assert "campaign-monitor" not in task_toml
            assert "[[environment.mcp_servers]]" not in task_toml
            assert not (wrapper / "inputs" / "runtime_capabilities.json").exists()

    def test_finds_submission_without_hard_coding_row_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trial = Path(temporary)
            submission = trial / "workspace" / "submission"
            submission.mkdir(parents=True)
            designs = submission / "designs.csv"
            designs.write_text("candidate\nonly-one\n")
            self.assertEqual(run_eval._submission(trial), (designs, None))

    def test_scoring_prefers_the_frozen_artifact_submission(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trial = Path(temporary)
            live = trial / "workspace" / "submission"
            frozen = trial / "artifacts" / "workspace" / "submission"
            live.mkdir(parents=True)
            frozen.mkdir(parents=True)
            (live / "designs.csv").write_text("sequence\nLIVE\n")
            frozen_designs = frozen / "designs.csv"
            frozen_designs.write_text("sequence\nFROZEN\n")

            self.assertEqual(run_eval._submission(trial), (frozen_designs, None))

    def test_submission_terminal_report_must_match_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            submission = Path(temporary)
            designs = submission / "designs.csv"
            designs.write_text("sequence\nAAAA\nCCCC\n")
            (submission / "terminal_report.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "status": "complete",
                        "required_n": 2,
                        "n_submitted": 2,
                        "delivery_valid": True,
                        "target_count_met": True,
                        "public_gate_pass_count": 0,
                        "errors": [],
                    }
                )
            )

            self.assertIsNone(run_eval._submission_terminal_error(designs, 2))

            designs.write_text("sequence_A\nAAAA\nCCCC\n")
            self.assertIsNone(run_eval._submission_terminal_error(designs, 2))

            designs.write_text("sequence,sequence_A\nAAAA,AAAA\nCCCC,CCCC\n")
            self.assertIn(
                "not both",
                run_eval._submission_terminal_error(designs, 2),
            )

            designs.write_text(
                "design_id,sequence_H,sequence_L\n"
                "first,AAAA,CCCC\n"
                "second,GGGG,TTTT\n"
            )
            self.assertIsNone(
                run_eval._submission_terminal_error(
                    designs,
                    2,
                    sequence_columns=("sequence_H", "sequence_L"),
                )
            )

            designs.write_text("design_id,sequence_H\nfirst,AAAA\nsecond,GGGG\n")
            self.assertIn(
                "sequence_L",
                run_eval._submission_terminal_error(
                    designs,
                    2,
                    sequence_columns=("sequence_H", "sequence_L"),
                ),
            )

    def test_biotin_lineage_requires_distinct_de_novo_backbones(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            submission = Path(temporary)
            designs = submission / "designs.csv"
            designs.write_text("sequence\nAAAA\nCCCC\n")
            backbones = submission / "backbones"
            backbones.mkdir()

            def write_backbone(
                backbone: Path, residue_name: str, spacing: float, y: float
            ) -> None:
                lines = []
                serial = 1
                for residue_index in range(1, 5):
                    base = residue_index * spacing
                    for atom_name, offset in (("N", 0.0), ("CA", 1.3), ("C", 2.5)):
                        lines.append(
                            f"ATOM  {serial:5d}  {atom_name:<3s} {residue_name:>3s} "
                            f"A{residue_index:4d}    {base + offset:8.3f}"
                            f"{y:8.3f}{0.0:8.3f}  1.00 20.00           "
                            f"{atom_name[0]:>2s}"
                        )
                        serial += 1
                backbone.write_text("\n".join(lines) + "\nEND\n")

            backbone_paths = []
            for index, residue_name in enumerate(("ALA", "CYS")):
                backbone = backbones / f"design-{index}.pdb"
                write_backbone(backbone, residue_name, 3.8 + index * 0.2, index * 0.3)
                backbone_paths.append(backbone)
            records = [
                {
                    "sequence": sequence,
                    "backbone_path": str(path.relative_to(submission)),
                    "backbone_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "generator": "rfd3",
                    "generator_version": "1",
                    "run_id": "pilot",
                    "seed": index,
                    "parent_sequence": None,
                    "parent_backbone_sha256": None,
                }
                for index, (sequence, path) in enumerate(
                    zip(("AAAA", "CCCC"), backbone_paths, strict=True)
                )
            ]
            (submission / "lineage.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records)
            )

            self.assertIsNone(run_eval._independent_lineage_error(designs))
            write_backbone(backbone_paths[1], "CYS", 3.8, 0.0)
            records[1]["backbone_sha256"] = hashlib.sha256(
                backbone_paths[1].read_bytes()
            ).hexdigest()
            (submission / "lineage.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records)
            )
            self.assertIn(
                "duplicate backbone",
                run_eval._independent_lineage_error(designs),
            )

    def test_public_gate_count_must_match_pinned_scorer_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            submission = Path(temporary)
            designs = submission / "designs.csv"
            designs.write_text("sequence\nAAAA\n")
            (submission / "public_metrics.csv").write_text(
                "sequence,min_interaction_pae,linker_max_rsasa,"
                "protein_ligand_hbonds,core_contact_ms,core_aromatic_neighbors\n"
                "AAAA,1.0,0.2,3,120,1\n"
            )
            contract = (
                run_eval.EVALS_ROOT
                / "evals_task"
                / "biotin_binder_design"
                / "inputs"
                / "acceptance_contract.json"
            )
            scorer = contract.parent / "public_score.py"
            if not scorer.is_file():
                self.skipTest(
                    "legacy public-score fixture is not part of this checkout"
                )
            run_eval.subprocess.run(
                [
                    run_eval.sys.executable,
                    str(scorer),
                    str(submission / "public_metrics.csv"),
                    "--output",
                    str(submission / "public_score.json"),
                ],
                check=True,
            )

            self.assertIsNone(run_eval._public_score_error(designs, 1, 1, contract))
            forged = json.loads((submission / "public_score.json").read_text())
            forged["per_design"][0]["metrics"] = {"fake_gate": "not-a-number"}
            forged["per_design"][0]["gates"] = {"fake_gate": True}
            forged["gate_counts"] = {"fake_gate": {"passed": 1, "total": 1}}
            (submission / "public_score.json").write_text(json.dumps(forged))
            self.assertIn(
                "controller recomputation",
                run_eval._public_score_error(designs, 1, 1, contract),
            )
            run_eval.subprocess.run(
                [
                    run_eval.sys.executable,
                    str(scorer),
                    str(submission / "public_metrics.csv"),
                    "--output",
                    str(submission / "public_score.json"),
                ],
                check=True,
            )
            self.assertIn(
                "does not match",
                run_eval._public_score_error(designs, 1, 0, contract),
            )

    def test_incomplete_submission_is_structured_but_not_scorable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            submission = Path(temporary)
            designs = submission / "designs.csv"
            designs.write_text("sequence\nAAAA\n")
            (submission / "terminal_report.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "status": "contract_incomplete",
                        "required_n": 2,
                        "n_submitted": 1,
                        "delivery_valid": True,
                        "target_count_met": False,
                        "public_gate_pass_count": 0,
                        "errors": ["one genuine design available"],
                    }
                )
            )

            self.assertEqual(
                run_eval._submission_terminal_error(designs, 2),
                "submission contract incomplete: 1/2 designs",
            )

    def test_selects_all_successful_trials_for_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            for name in ("attempt-one", "attempt-two"):
                trial = job / name
                trial.mkdir()
                (trial / "result.json").write_text(
                    json.dumps(
                        {"exception_info": None, "agent_execution": {"ok": True}}
                    )
                )
            self.assertEqual(
                [path.name for path in run_eval._successful_trials(job)],
                ["attempt-one", "attempt-two"],
            )

    def test_failed_outer_trial_with_delivered_submission_is_scoreable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            trial = job / "failed-after-delivery"
            submission = trial / "workspace" / "submission"
            submission.mkdir(parents=True)
            (submission / "designs.csv").write_text("sequence\nAAAA\n")
            (trial / "result.json").write_text(
                json.dumps(
                    {
                        "exception_info": {"exception_type": "WorkerLostError"},
                        "agent_execution": {"started_at": "2026-08-20T00:00:00Z"},
                    }
                )
            )

            self.assertEqual(run_eval._successful_trials(job), [trial])

    def test_delivered_submission_without_result_record_is_scoreable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            trial = job / "no-result"
            submission = trial / "artifacts" / "workspace" / "submission"
            submission.mkdir(parents=True)
            (submission / "designs.csv").write_text("sequence\nAAAA\n")

            self.assertEqual(run_eval._scan_trials(job), ([trial], [], False))

    def test_failed_trial_without_a_delivery_stays_failed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            trial = job / "never-delivered"
            (trial / "workspace" / "submission").mkdir(parents=True)
            (trial / "result.json").write_text(
                json.dumps(
                    {
                        "exception_info": {"exception_type": "AgentTimeoutError"},
                        "agent_execution": {"started_at": "2026-08-20T00:00:00Z"},
                    }
                )
            )

            self.assertEqual(
                run_eval._scan_trials(job), ([], ["never-delivered"], True)
            )

    def test_watcher_scores_delivery_after_outer_job_failure(self) -> None:
        from scripts import score_after_eval

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            argv = [
                "score_after_eval.py",
                "--task",
                "demo_task",
                "--run-dir",
                str(run_dir),
                "--harbor-job-id",
                "12345",
            ]
            with (
                patch("sys.argv", argv),
                patch.object(score_after_eval, "resolve_task"),
                patch.object(
                    score_after_eval,
                    "_wait_for_job",
                    side_effect=SystemExit("Slurm job 12345 ended with state FAILED."),
                ),
                patch.object(score_after_eval, "_score") as score,
            ):
                score_after_eval.main()

            score.assert_called_once()

    def test_failed_trial_still_receives_controller_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = root / "tasks" / "ppi_pdl1"
            (task / "score").mkdir(parents=True)
            (task / "score" / "score.py").touch()
            job = root / "job"
            trial = job / "failed-attempt"
            trial.mkdir(parents=True)
            (trial / "result.json").write_text(
                json.dumps(
                    {
                        "exception_info": {"exception_type": "AgentTimeoutError"},
                        "agent_execution": {"started_at": "2026-08-18T00:00:00Z"},
                    }
                )
            )

            with self.assertRaisesRegex(SystemExit, "No successful Harbor trials"):
                run_eval._score(
                    task,
                    job,
                    root / "chaperone",
                    root / "worker.sif",
                    required_n=96,
                )

            audit = json.loads((trial / "controller" / "audit.json").read_text())
            self.assertFalse(audit["contract_ok"])
            self.assertIn(
                "submission CSV is unreadable",
                audit["submission"]["contract_errors"][0],
            )

    @patch.object(run_eval.subprocess, "run")
    def test_host_scoring_python_probes_chaperone_environment(self, run) -> None:
        run.return_value.returncode = 0
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "chaperone"
            python = repo / ".venv" / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.touch()
            (repo / "src").mkdir()

            self.assertEqual(run_eval._host_scoring_python(repo), python)

        command = run.call_args.args[0]
        environment = run.call_args.kwargs["env"]
        self.assertEqual(command[0], str(python))
        self.assertIn("import atomworks", command[2])
        self.assertIn("import scoring", command[2])
        self.assertIn("register_flags()", command[2])
        self.assertEqual(
            environment["PYTHONPATH"],
            run_eval.os.pathsep.join((str(repo / "src"), str(run_eval.EVALS_ROOT))),
        )
        for name, value in run_eval._CONTROLLER_THREAD_ENV.items():
            self.assertEqual(environment[name], value)

    @staticmethod
    def _site_tools(root: Path) -> dict[str, Path]:
        hbplus = root / "software" / "hbplus"
        hbplus.parent.mkdir()
        hbplus.touch(mode=0o700)
        paths = {
            "HBPLUS_COPY": hbplus,
            "CCD_MIRROR_PATH": root / "ccd",
            "PYROSETTA_PATH": root / "pyrosetta",
            "X3DNA_BIN_DIR": root / "x3dna" / "bin",
            "X3DNA_DIR": root / "x3dna" / "home",
        }
        for name, path in paths.items():
            if name != "HBPLUS_COPY":
                path.mkdir(parents=True)
        for executable in ("x3dna-dssr", "x3dna-snap"):
            (paths["X3DNA_BIN_DIR"] / executable).touch(mode=0o700)
        return paths

    def test_scoring_site_environment_uses_canonical_config_for_frozen_repo(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical = root / "chaperone"
            canonical_venv = canonical / ".venv"
            canonical_venv.mkdir(parents=True)
            frozen = root / "frozen-chaperone"
            frozen.mkdir()
            (frozen / ".venv").symlink_to(canonical_venv)
            tools = self._site_tools(root)
            (canonical / ".env").write_text(
                f"PYROSETTA_PATH={tools['PYROSETTA_PATH']}\n"
                f"X3DNA_BIN_DIR={tools['X3DNA_BIN_DIR']}\n"
                f"X3DNA_DIR={tools['X3DNA_DIR']}\n"
                "OPENAI_API_KEY=must-not-cross-the-boundary\n"
            )
            (canonical / ".env.ipd").write_text(
                f"HBPLUS_COPY={tools['HBPLUS_COPY']}\n"
                f"CCD_MIRROR_PATH={tools['CCD_MIRROR_PATH']}\n"
            )

            resolved = run_eval._scoring_site_environment(frozen, {})

        self.assertEqual(set(resolved), set(run_eval._SCORING_SITE_ENV_NAMES))
        self.assertEqual(resolved["HBPLUS_COPY"], str(tools["HBPLUS_COPY"]))
        self.assertEqual(resolved["CCD_MIRROR_PATH"], str(tools["CCD_MIRROR_PATH"]))
        self.assertNotIn("OPENAI_API_KEY", resolved)

    def test_scoring_site_environment_prefers_the_shell_over_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tools = self._site_tools(root)
            site = root / "site"
            site.mkdir()
            (site / ".env").write_text("PYROSETTA_PATH=/from/config\n")
            (site / ".env.ipd").write_text("HBPLUS_COPY=/from/config\n")
            shell = {name: str(path) for name, path in tools.items()}
            shell["CHAPERONE_SITE_CONFIG_REPO"] = str(site)

            resolved = run_eval._scoring_site_environment(root / "repo", shell)

        self.assertEqual(resolved, {name: str(path) for name, path in tools.items()})

    def test_scoring_site_environment_requires_every_setting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            site = Path(temporary)
            (site / ".env").write_text("PYROSETTA_PATH=/somewhere\n")
            (site / ".env.ipd").touch()
            environment = {"CHAPERONE_SITE_CONFIG_REPO": str(site)}

            with self.assertRaisesRegex(SystemExit, "HBPLUS_COPY"):
                run_eval._scoring_site_environment(site, environment)

    def test_scoring_site_environment_needs_a_config_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            empty = Path(temporary)
            environment = {"CHAPERONE_SITE_CONFIG_REPO": str(empty)}

            with self.assertRaisesRegex(SystemExit, "Could not locate"):
                run_eval._scoring_site_environment(empty, environment)

    def test_scoring_site_environment_requires_all_preflighted_paths(self) -> None:
        values = {name: "/missing" for name in run_eval._SCORING_SITE_ENV_NAMES}
        with self.assertRaisesRegex(SystemExit, "site-config preflight failed"):
            run_eval._preflight_scoring_site_environment(values)

    @patch.object(run_eval, "_run_score_command")
    @patch.object(run_eval.ApptainerEnvironment, "_prepare_slurm_python")
    @patch.object(run_eval.ApptainerEnvironment, "_prepare_runtime_identity")
    @patch.object(run_eval, "_prepare_scoring_packages")
    @patch.object(run_eval, "_host_scoring_python")
    @patch.object(run_eval, "_scoring_site_environment", new=_fake_site_environment)
    def test_score_trial_runs_scorer_on_host(
        self, host_python, _packages, _identity, _slurm_python, run_score
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "chaperone"
            (repo / "src").mkdir(parents=True)
            python = repo / ".venv" / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.touch()
            host_python.return_value = python
            scorer = root / "score.py"
            scorer.write_text("print('score')\n")
            image = root / "worker.sif"
            image.touch()
            trial = root / "trial"
            submission = trial / "workspace" / "submission"
            submission.mkdir(parents=True)
            (submission / "designs.csv").write_text("sequence\nAAAA\n")

            # The controller records the row-count failure, but still invokes
            # the held-out scorer so it owns the normal contract result.
            run_eval._score_trial(scorer, trial, repo, image, required_n=2)

            command = run_score.call_args.args[0]
            environment = run_score.call_args.kwargs["environment"]
            self.assertEqual(command[:2], [str(python), str(scorer.resolve())])
            self.assertNotIn("apptainer", command)
            self.assertEqual(
                environment["CHAPERONE_PLIP_PYTHONPATH"],
                str(trial / "scoring" / ".runtime" / "python-packages"),
            )
            for name in run_eval._SCORING_SITE_ENV_NAMES:
                self.assertEqual(environment[name], TEST_SITE_ENV[name])
            for name, value in run_eval._CONTROLLER_THREAD_ENV.items():
                self.assertEqual(environment[name], value)
            self.assertEqual(
                environment[run_eval._SCORING_USALIGN_ENV],
                str(repo / ".venv" / "bin" / "USalign"),
            )
            self.assertEqual(environment[run_eval._SCORING_RUNDIR_BIND_ENV], "true")
            expected_rundir = (trial / "scoring" / "score_run").absolute()
            self.assertEqual(environment["RUNDIR"], str(expected_rundir))
            self.assertIn(f"--rundir={expected_rundir}", command)
            self.assertIn(
                f"--per_design_out={trial / 'scoring' / 'score_per_design.csv'}",
                command,
            )
            self.assertIn(
                f"--run_meta={trial / 'controller' / 'run_meta.json'}",
                command,
            )
            self.assertIn("--required_n=2", command)
            self.assertFalse(
                json.loads((trial / "controller" / "audit.json").read_text())[
                    "contract_ok"
                ]
            )
            self.assertTrue(expected_rundir.is_dir())
            expected_jax_cache = expected_rundir / ".cache" / "af3_jax"
            self.assertEqual(
                environment[run_eval._SCORING_AF3_JAX_CACHE_ENV],
                str(expected_jax_cache),
            )
            self.assertTrue(expected_jax_cache.is_dir())
            self.assertEqual(
                _slurm_python.call_args.kwargs["site_environment"],
                {
                    **TEST_SITE_ENV,
                    run_eval._SCORING_AF3_JAX_CACHE_ENV: str(expected_jax_cache),
                },
            )
            self.assertNotEqual(
                environment[run_eval._SCORING_AF3_JAX_CACHE_ENV],
                "/projects/ml/pipelines/weights/af3_cache",
            )

    @patch.object(run_eval, "_run_score_command")
    @patch.object(run_eval.ApptainerEnvironment, "_prepare_slurm_python")
    @patch.object(run_eval.ApptainerEnvironment, "_prepare_runtime_identity")
    @patch.object(run_eval, "_prepare_scoring_packages")
    @patch.object(run_eval, "_host_scoring_python")
    def test_score_trial_preserves_site_environment_overrides(
        self, host_python, _packages, _identity, slurm_python, run_score
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "chaperone"
            (repo / "src").mkdir(parents=True)
            python = repo / ".venv" / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.touch()
            host_python.return_value = python
            scorer = root / "score.py"
            scorer.write_text("print('score')\n")
            image = root / "worker.sif"
            image.touch()
            trial = root / "trial"
            submission = trial / "workspace" / "submission"
            submission.mkdir(parents=True)
            (submission / "designs.csv").write_text("sequence\nAAAA\n")
            site = root / "site-config"
            site.mkdir()
            (site / ".env").touch()
            (site / ".env.ipd").touch()

            overrides = {
                name: f"/custom/{name.lower()}"
                for name in run_eval._SCORING_SITE_ENV_NAMES
            }
            with (
                patch.dict(
                    run_eval.os.environ,
                    {**overrides, "CHAPERONE_SITE_CONFIG_REPO": str(site)},
                ),
                patch.object(run_eval, "_preflight_scoring_site_environment"),
            ):
                run_eval._score_trial(scorer, trial, repo, image, required_n=1)

            environment = run_score.call_args.kwargs["environment"]
            worker_environment = slurm_python.call_args.kwargs["site_environment"]
            for name, value in overrides.items():
                self.assertEqual(environment[name], value)
                self.assertEqual(worker_environment[name], value)

    @patch.object(run_eval, "_run_score_command")
    @patch.object(run_eval.ApptainerEnvironment, "_prepare_slurm_python")
    @patch.object(run_eval.ApptainerEnvironment, "_prepare_runtime_identity")
    @patch.object(run_eval, "_prepare_scoring_packages")
    @patch.object(run_eval, "_host_scoring_python")
    @patch.object(run_eval, "_scoring_site_environment", new=_fake_site_environment)
    def test_score_trial_creates_cache_before_building_worker_wrapper(
        self, host_python, _packages, _identity, slurm_python, _run_score
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "chaperone"
            (repo / "src").mkdir(parents=True)
            python = repo / ".venv" / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.touch()
            host_python.return_value = python
            scorer = root / "score.py"
            scorer.write_text("print('score')\n")
            image = root / "worker.sif"
            image.touch()
            trial = root / "trial"
            submission = trial / "workspace" / "submission"
            submission.mkdir(parents=True)
            (submission / "designs.csv").write_text("sequence\nAAAA\n")
            seen: list[bool] = []

            def wrapper_is_built(*_args, **kwargs):
                cache = Path(
                    kwargs["site_environment"][run_eval._SCORING_AF3_JAX_CACHE_ENV]
                )
                seen.append(cache.is_dir() and cache.parent.parent.name == "score_run")

            slurm_python.side_effect = wrapper_is_built
            run_eval._score_trial(scorer, trial, repo, image, required_n=1)

        self.assertEqual(seen, [True])

    @patch.object(run_eval, "_run_score_command")
    @patch.object(run_eval.ApptainerEnvironment, "_prepare_slurm_python")
    @patch.object(run_eval.ApptainerEnvironment, "_prepare_runtime_identity")
    @patch.object(run_eval, "_prepare_scoring_packages")
    @patch.object(run_eval, "_host_scoring_python")
    @patch.object(run_eval, "_scoring_site_environment", new=_fake_site_environment)
    def test_designs_only_96_row_trial_is_audited_and_scored(
        self, host_python, _packages, _identity, _slurm_python, run_score
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "chaperone"
            (repo / "src").mkdir(parents=True)
            python = repo / ".venv" / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.touch()
            host_python.return_value = python
            scorer = root / "score.py"
            scorer.write_text("print('score')\n")
            image = root / "worker.sif"
            image.touch()
            trial = root / "trial"
            submission = trial / "artifacts" / "workspace" / "submission"
            submission.mkdir(parents=True)
            (submission / "designs.csv").write_text("sequence\n" + "AAAA\n" * 96)
            transcript = trial / "agent" / "codex.txt"
            transcript.parent.mkdir()
            transcript.write_text(
                json.dumps(
                    {
                        "type": "turn.completed",
                        "usage": {"input_tokens": 100, "output_tokens": 10},
                    }
                )
                + "\n"
            )
            registry = trial / ".runtime" / "slurm-broker" / "submissions.jsonl"
            registry.parent.mkdir(parents=True)
            registry.touch()

            run_eval._score_trial(scorer, trial, repo, image)

            audit = json.loads((trial / "controller" / "audit.json").read_text())
            self.assertTrue(audit["contract_ok"])
            self.assertTrue(audit["audit_ok"])
            self.assertEqual(audit["submission"]["row_count"], 96)
            self.assertTrue(run_score.called)
            self.assertIn(
                f"--run_meta={trial / 'controller' / 'run_meta.json'}",
                run_score.call_args.args[0],
            )

    @patch.object(run_eval, "_run_score_command")
    @patch.object(run_eval.ApptainerEnvironment, "_prepare_slurm_python")
    @patch.object(run_eval.ApptainerEnvironment, "_prepare_runtime_identity")
    @patch.object(run_eval, "_prepare_scoring_packages")
    @patch.object(run_eval, "_host_scoring_python")
    @patch.object(run_eval, "_scoring_site_environment", new=_fake_site_environment)
    def test_score_trials_isolate_per_design_outputs(
        self, host_python, _packages, _identity, _slurm_python, run_score
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "chaperone"
            (repo / "src").mkdir(parents=True)
            python = repo / ".venv" / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.touch()
            host_python.return_value = python
            scorer = root / "score.py"
            scorer.write_text("print('score')\n")
            image = root / "worker.sif"
            image.touch()

            trials = [root / "trial-one", root / "trial-two"]
            for trial in trials:
                submission = trial / "workspace" / "submission"
                submission.mkdir(parents=True)
                (submission / "designs.csv").write_text("sequence\nAAAA\n")
                run_eval._score_trial(scorer, trial, repo, image, required_n=1)

            per_design_args = {
                argument
                for call in run_score.call_args_list
                for argument in call.args[0]
                if argument.startswith("--per_design_out=")
            }
            self.assertEqual(
                per_design_args,
                {
                    f"--per_design_out={trial / 'scoring' / 'score_per_design.csv'}"
                    for trial in trials
                },
            )

    @patch.object(run_eval.subprocess, "run")
    @patch.object(run_eval.shutil, "which", return_value="/usr/bin/uv")
    def test_stages_locked_scoring_packages(self, _which, run) -> None:
        run.return_value.returncode = 0
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "packages"
            run_eval._prepare_scoring_packages(target)

        command = run.call_args.args[0]
        self.assertEqual(command[0:3], ["/usr/bin/uv", "pip", "install"])
        self.assertIn("--no-deps", command)
        for package in run_eval._SCORING_PACKAGES:
            self.assertIn(package, command)

        environment = run.call_args.kwargs["env"]
        for name, value in {
            **run_eval._CONTROLLER_THREAD_ENV,
            **run_eval._SCORING_UV_ENV,
        }.items():
            self.assertEqual(environment[name], value)

    @patch.object(run_eval.subprocess, "run")
    @patch.object(run_eval.shutil, "which", return_value="/usr/bin/uv")
    def test_scoring_package_staging_publishes_once_after_success(
        self, _which, run
    ) -> None:
        run.return_value.returncode = 0
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "packages"
            run_eval._prepare_scoring_packages(target)
            self.assertTrue((target / run_eval._SCORING_PACKAGE_MARKER).is_file())
            run_eval._prepare_scoring_packages(target)

        run.assert_called_once()

    @patch.object(run_eval.subprocess, "run")
    @patch.object(run_eval.shutil, "which", return_value="/usr/bin/uv")
    def test_failed_scoring_package_staging_preserves_previous_target(
        self, _which, run
    ) -> None:
        run.return_value.returncode = 1
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "packages"
            target.mkdir()
            sentinel = target / "existing"
            sentinel.write_text("keep")
            with self.assertRaisesRegex(SystemExit, "Failed to stage"):
                run_eval._prepare_scoring_packages(target)
            self.assertEqual(sentinel.read_text(), "keep")
            self.assertFalse(list(target.parent.glob(f".{target.name}.staging-*")))

    @patch.object(run_eval.subprocess, "run")
    @patch.object(run_eval.shutil, "which", return_value="/usr/bin/uv")
    def test_incomplete_scoring_packages_are_replaced_atomically(
        self, _which, run
    ) -> None:
        run.return_value.returncode = 0
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "packages"
            target.mkdir()
            (target / "plip").mkdir()
            (target / "stale").write_text("left by an interrupted install")

            run_eval._prepare_scoring_packages(target)

            self.assertTrue((target / run_eval._SCORING_PACKAGE_MARKER).is_file())
            self.assertFalse((target / "stale").exists())
            self.assertEqual(
                sorted(path.name for path in target.parent.iterdir()),
                ["packages", "uv-cache"],
            )

    @patch.object(run_eval.subprocess, "run")
    @patch.object(run_eval.shutil, "which", return_value="/usr/bin/uv")
    def test_scoring_package_staging_holds_the_per_user_lock(self, _which, run) -> None:
        held: list[bool] = []

        def probe(*_args, **_kwargs):
            lock = (
                Path(run_eval.tempfile.gettempdir())
                / f"chaperone-evals-scoring-packages-{run_eval.os.getuid()}.lock"
            )
            descriptor = run_eval.os.open(lock, run_eval.os.O_RDWR)
            try:
                run_eval.fcntl.flock(
                    descriptor, run_eval.fcntl.LOCK_EX | run_eval.fcntl.LOCK_NB
                )
                held.append(False)
            except BlockingIOError:
                held.append(True)
            finally:
                run_eval.os.close(descriptor)
            return SimpleNamespace(returncode=0)

        run.side_effect = probe
        with tempfile.TemporaryDirectory() as temporary:
            run_eval._prepare_scoring_packages(Path(temporary) / "packages")

        self.assertEqual(held, [True])

    @patch.object(run_eval.subprocess, "run")
    @patch.object(run_eval.shutil, "which", return_value="/usr/bin/uv")
    def test_scoring_package_staging_isolates_concurrent_uv_caches(
        self, _which, run
    ) -> None:
        run.return_value.returncode = 0
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            targets = [
                root / "trial-one" / "python-packages",
                root / "trial-two" / "python-packages",
            ]
            with patch.dict(run_eval.os.environ, {"UV_CACHE_DIR": "/shared/uv"}):
                for target in targets:
                    run_eval._prepare_scoring_packages(target)

            cache_dirs = [
                call.kwargs["env"]["UV_CACHE_DIR"] for call in run.call_args_list
            ]
            self.assertEqual(
                cache_dirs,
                [str(target.parent / "uv-cache") for target in targets],
            )
            self.assertEqual(len(set(cache_dirs)), len(targets))
            self.assertTrue(all(Path(path).is_dir() for path in cache_dirs))

    @patch.object(run_eval.subprocess, "run")
    def test_restarts_host_scorer_and_reuses_chunks(self, run) -> None:
        run.side_effect = [
            run_eval.subprocess.CompletedProcess([], 1),
            run_eval.subprocess.CompletedProcess([], 0),
        ]

        run_eval._run_score_command(
            ["python", "score.py"],
            cwd=Path("/tmp"),
            environment={},
            trial_name="trial-one",
        )

        self.assertEqual(run.call_count, 2)

    def test_parses_parsable_and_human_slurm_output(self) -> None:
        self.assertEqual(run_eval._parse_job_id("12345;cluster\n"), "12345")
        self.assertEqual(run_eval._parse_job_id("Submitted batch job 67890\n"), "67890")

    def test_external_scorer_waits_for_harbor_exit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = self._task(root / "tasks", "ppi_pdl1")
            repo = root / "chaperone"
            image = root / "worker.sif"
            args = SimpleNamespace(
                chaperone_repo=repo,
                worker_image=image,
                required_n=96,
                poll_seconds=30.0,
            )
            with (
                patch.object(run_eval, "ROOT", root / "controller"),
                patch.object(run_eval, "JOBS_ROOT", root / "jobs"),
            ):
                command = run_eval._external_scoring_command(
                    args,
                    task,
                    "ppi-high",
                    "12345",
                    root / "jobs" / "results" / "ppi-high-12345",
                )

        self.assertNotIn("sbatch", command)
        self.assertEqual(command[0], str(root / "controller" / ".venv/bin/python"))
        self.assertIn(str(root / "controller" / "scripts/score_after_eval.py"), command)
        self.assertEqual(command[command.index("--harbor-job-id") + 1], "12345")
        self.assertEqual(command[command.index("--poll-seconds") + 1], "30.0")
        self.assertEqual(command[command.index("--task") + 1], "ppi_pdl1")

    def test_external_scorer_is_detached_and_writes_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            jobs = Path(temporary) / "jobs"
            process = SimpleNamespace(pid=4321)
            with (
                patch.object(run_eval, "JOBS_ROOT", jobs),
                patch.object(
                    run_eval.subprocess, "Popen", return_value=process
                ) as popen,
            ):
                launched, receipt = run_eval._launch_external_scorer(
                    ["python", "score_after_eval.py"], "biotin-high", "12345"
                )

            self.assertIs(launched, process)
            self.assertTrue(receipt.is_file())
            payload = json.loads(receipt.read_text())
            self.assertEqual(payload["mode"], "external-host-watcher")
            self.assertEqual(payload["pid"], 4321)
            self.assertEqual(payload["harbor_job_id"], "12345")
            kwargs = popen.call_args.kwargs
            self.assertTrue(kwargs["start_new_session"])
            self.assertTrue(kwargs["close_fds"])
            self.assertIs(kwargs["stdin"], run_eval.subprocess.DEVNULL)

    def test_public_contract_is_pinned_by_harbor_job_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            jobs = Path(temporary) / "jobs"
            inputs = jobs / "task-definitions" / "biotin-run" / "inputs"
            inputs.mkdir(parents=True)
            (inputs / "acceptance_contract.json").write_text('{"version": 1}\n')
            (inputs / "public_score.py").write_text("print('score')\n")
            with patch.object(run_eval, "JOBS_ROOT", jobs):
                pinned = run_eval._pin_public_contract("biotin-run", "12345")
                assert pinned is not None
                self.assertEqual(pinned.read_text(), '{"version": 1}\n')
                (inputs / "acceptance_contract.json").write_text('{"version": 2}\n')
                with self.assertRaisesRegex(SystemExit, "different content"):
                    run_eval._pin_public_contract("biotin-run", "12345")

    def test_submission_exports_the_shared_jobs_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = self._task(root / "tasks", "ppi_pdl1")
            harbor_task = root / "task-definitions" / "ppi-low"
            image = root / "agent.sif"
            workspace = root / "snapshot"
            args = SimpleNamespace(
                arm="harness",
                model="openai/gpt-5.6-sol",
                reasoning_effort="low",
                attempts=1,
                concurrency=1,
                image=image,
                workspace_seed=workspace,
            )
            jobs_root = root / "scratch" / "runs"
            with (
                patch.object(run_eval, "JOBS_ROOT", jobs_root),
                patch.dict(run_eval.os.environ, {"HARNESS_SKILL": "/leaked/skill"}),
            ):
                environment = run_eval._submission_environment(
                    args, task, harbor_task, "ppi-low"
                )

        self.assertEqual(environment["HARBOR_JOBS_DIR"], str(jobs_root))
        self.assertEqual(environment["HARBOR_AGENT_TIMEOUT_SEC"], "43200")
        self.assertNotIn("MEMORY_POLICY", environment)
        self.assertNotIn("CHAPERONE_MEMORY_POLICY", environment)
        self.assertNotIn("HARNESS_SKILL", environment)

    @patch.object(run_eval, "_run_capture")
    def test_normalizes_annotated_sacct_state(self, capture) -> None:
        capture.side_effect = [
            run_eval.subprocess.CompletedProcess([], 0, "", ""),
            run_eval.subprocess.CompletedProcess([], 0, "CANCELLED by 2597|\n", ""),
        ]
        self.assertEqual(run_eval._slurm_state("12345"), "CANCELLED")

    @patch.object(run_eval.time, "sleep")
    @patch.object(run_eval.subprocess, "run")
    def test_slurm_capture_retries_transient_process_pressure(self, run, sleep) -> None:
        completed = run_eval.subprocess.CompletedProcess(
            ["squeue"], returncode=0, stdout="RUNNING\n", stderr=""
        )
        run.side_effect = [
            BlockingIOError(run_eval.errno.EAGAIN, "Resource temporarily unavailable"),
            completed,
        ]

        self.assertIs(run_eval._run_capture(["squeue"]), completed)
        sleep.assert_called_once_with(0.1)


if __name__ == "__main__":
    unittest.main()
