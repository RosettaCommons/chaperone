#!/usr/bin/env python3
"""Unit tests for the guarded sbatch resource parser."""

from __future__ import annotations

import importlib.util
import json
import socket
import tempfile
import threading
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "slurm_gpu_guard.py"
SPEC = importlib.util.spec_from_file_location("slurm_gpu_guard", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GUARD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GUARD)


class SlurmGpuGuardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.workspace = root / "workspace"
        self.workspace.mkdir()
        (self.workspace / ".home").mkdir()
        (self.workspace / ".tmp").mkdir()
        self.slurm_python = root / "runtime" / "slurm-python"
        self.slurm_python.parent.mkdir()
        self.slurm_python.write_text("#!/bin/bash\n")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _script(
        self,
        *,
        partition: str = "gpu",
        gres: str | None = "gpu:small:2",
        body: str | None = None,
        extra_directive: str = "",
    ) -> Path:
        log_dir = self.workspace / "logs" / "%j"
        script = self.workspace / "logs" / ".submission_file_test.sh"
        script.parent.mkdir(exist_ok=True)
        gres_line = f"#SBATCH --gres={gres}\n" if gres is not None else ""
        if body is None:
            body = (
                "export SUBMITIT_EXECUTOR=slurm\n"
                f"srun --unbuffered --output {log_dir}/%j_%t_log.out "
                f"--error {log_dir}/%j_%t_log.err --export=ALL "
                f"{self.slurm_python} -u -m submitit.core._submit {log_dir}\n"
            )
        script.write_text(
            "#!/bin/bash\n\n"
            "# Parameters\n"
            "#SBATCH --cpus-per-task=2\n"
            f"#SBATCH --error={log_dir}/%j_0_log.err\n"
            f"{gres_line}"
            "#SBATCH --job-name=submitit\n"
            "#SBATCH --mem=8GB\n"
            "#SBATCH --nodes=1\n"
            "#SBATCH --open-mode=append\n"
            f"#SBATCH --output={log_dir}/%j_0_log.out\n"
            f"#SBATCH --partition={partition}\n"
            "#SBATCH --signal=USR2@90\n"
            "#SBATCH --time=60\n"
            "#SBATCH --wckey=submitit\n"
            f"{extra_directive}"
            "\n# command\n"
            f"{body}"
        )
        return script

    def _prepare(self, script: Path) -> tuple[list[str], int, str | None]:
        return GUARD.prepare_submission(
            [str(script)],
            self.workspace,
            self.workspace,
            self.slurm_python,
        )

    def _generic_script(
        self,
        *,
        partition: str = "gpu",
        gres: str | None = "gpu:small:1",
        array: str | None = None,
    ) -> Path:
        script = self.workspace / "job.sh"
        gres_line = f"#SBATCH --gres={gres}\n" if gres is not None else ""
        array_line = f"#SBATCH --array={array}\n" if array is not None else ""
        script.write_text(
            "#!/bin/bash\n"
            "#SBATCH --job-name=baseline-test\n"
            "#SBATCH --output=logs/%A_%a.out\n"
            "#SBATCH --error=logs/%A_%a.err\n"
            f"#SBATCH --partition={partition}\n"
            "#SBATCH --time=01:00:00\n"
            "#SBATCH --mem=8GB\n"
            "#SBATCH --cpus-per-task=2\n"
            f"{gres_line}"
            f"{array_line}"
            "echo ok\n"
        )
        return script

    def _prepare_generic(
        self, script: Path
    ) -> tuple[list[str], int, str, dict[str, str]]:
        return GUARD.prepare_generic_submission(
            [str(script)],
            self.workspace,
            self.workspace,
            Path("/workspace"),
            self.workspace.parent / "scheduler-logs",
        )

    def test_cpu_job(self) -> None:
        self.assertEqual(GUARD.requested_gpus(["--mem=8G", "job.sh"]), 0)

    def test_gres_in_script(self) -> None:
        script = "#!/bin/bash\n#SBATCH --nodes=2\n#SBATCH --gres=gpu:small:2\necho ok\n"
        self.assertEqual(GUARD.requested_gpus([], [script]), 4)
        self.assertEqual(GUARD.validate_request([], [script], 5), 4)

    def test_gpus_per_task(self) -> None:
        self.assertEqual(GUARD.requested_gpus(["--ntasks=2", "--gpus-per-task=2"]), 4)

    def test_short_gpu_option(self) -> None:
        self.assertEqual(GUARD.requested_gpus(["-G4"]), 4)

    def test_accepts_explicit_small_gpu_class(self) -> None:
        self.assertEqual(GUARD.validate_request(["--gres=gpu:small:4"], [], 5), 4)

    def test_rejects_other_gpu_type(self) -> None:
        with self.assertRaisesRegex(GUARD.GuardError, "only gpu:small:N"):
            GUARD.validate_request(["--gres=gpu:a4000:1"], [], 5)

    def test_rejects_mixed_gpu_classes(self) -> None:
        with self.assertRaisesRegex(GUARD.GuardError, "only gpu:small:N"):
            GUARD.validate_request(["--gres=gpu:small:1,gpu:a4000:1"], [], 5)

    def test_rejects_untyped_gpu(self) -> None:
        with self.assertRaisesRegex(GUARD.GuardError, "found untyped"):
            GUARD.validate_request(["--gpus=1"], [], 5)

    def test_accepts_exactly_five(self) -> None:
        self.assertEqual(GUARD.validate_request(["--gres=gpu:small:5"], [], 5), 5)

    def test_rejects_six(self) -> None:
        with self.assertRaisesRegex(GUARD.GuardError, "at most 5 GPUs total"):
            GUARD.validate_request(["--gres=gpu:small:6"], [], 5)

    def test_rejects_aggregate_over_limit(self) -> None:
        with self.assertRaises(GUARD.GuardError):
            GUARD.validate_request(["--nodes=3", "--gres=gpu:small:2"], [], 5)

    def test_per_socket_requires_bound(self) -> None:
        with self.assertRaisesRegex(GUARD.GuardError, "sockets-per-node"):
            GUARD.requested_gpus(["--gpus-per-socket=1"])

    def test_prepares_submitit_gpu_script_for_real_scheduler(self) -> None:
        script = self._script()
        argv, count, submitted_script = self._prepare(script)
        self.assertEqual(count, 2)
        self.assertEqual(submitted_script, script.read_text())
        self.assertEqual(
            argv,
            [
                f"--chdir={self.workspace}",
                "--export=NONE",
                "--gres=gpu:small:2",
                "--constraint=A4000",
            ],
        )

    def test_prepares_baseline_gpu_script_with_a4000_constraint(self) -> None:
        argv, count, submitted_script, _ = self._prepare_generic(self._generic_script())
        self.assertEqual(count, 1)
        self.assertEqual(submitted_script, self._generic_script().read_text())
        self.assertIn("--partition=gpu", argv)
        self.assertIn("--gres=gpu:small:1", argv)
        self.assertIn("--constraint=A4000", argv)

    def test_prepares_baseline_cpu_script_without_gpu_flags(self) -> None:
        argv, count, _, _ = self._prepare_generic(
            self._generic_script(partition="cpu", gres=None)
        )
        self.assertEqual(count, 0)
        self.assertIn("--partition=cpu", argv)
        self.assertFalse(any(value.startswith("--gres=") for value in argv))

    def test_rejects_noncanonical_baseline_gpu_requests(self) -> None:
        cases = (
            ("gpu", "gpu:a4000:1", "only gpu:small:N"),
            ("gpu", "gpu:1", "found untyped"),
            ("gpu-bf", "gpu:small:1", "partition must be cpu or gpu"),
            ("cpu", "gpu:small:1", "GPU baseline jobs must use"),
        )
        for partition, gres, message in cases:
            with self.subTest(partition=partition, gres=gres):
                with self.assertRaisesRegex(GUARD.GuardError, message):
                    self._prepare_generic(
                        self._generic_script(partition=partition, gres=gres)
                    )

    def test_baseline_array_gpu_limit_uses_concurrency(self) -> None:
        self._prepare_generic(self._generic_script(array="0-9%5"))
        with self.assertRaisesRegex(GUARD.GuardError, "could reserve 6 GPUs"):
            self._prepare_generic(self._generic_script(array="0-9%6"))

    def test_prepares_submitit_cpu_script_without_gpu_flags(self) -> None:
        script = self._script(partition="cpu", gres=None)
        argv, count, submitted_script = self._prepare(script)
        self.assertEqual(count, 0)
        self.assertEqual(submitted_script, script.read_text())
        self.assertEqual(
            argv,
            [f"--chdir={self.workspace}", "--export=NONE"],
        )

    def test_accepts_cpu_gpus_per_node_zero_compatibility(self) -> None:
        script = self._script(
            partition="cpu",
            gres=None,
            extra_directive="#SBATCH --gpus-per-node=0\n",
        )
        argv, count, _ = self._prepare(script)
        self.assertEqual(count, 0)
        self.assertEqual(argv, [f"--chdir={self.workspace}", "--export=NONE"])

    def test_rejects_nonzero_gpus_per_node(self) -> None:
        script = self._script(
            partition="gpu",
            extra_directive="#SBATCH --gpus-per-node=1\n",
        )
        with self.assertRaisesRegex(GUARD.GuardError, "typed --gres"):
            self._prepare(script)

    def test_accepts_version_only(self) -> None:
        self.assertEqual(
            GUARD.prepare_submission(
                ["--version"],
                self.workspace,
                self.workspace,
                self.slurm_python,
            ),
            (["--version"], 0, None),
        )

    def test_rejects_wrap_and_direct_cli_payloads(self) -> None:
        for argv in (
            ["--wrap=cat /etc/passwd"],
            ["--partition=gpu", "job.sh"],
            ["--version", "extra"],
        ):
            with self.subTest(argv=argv), self.assertRaises(GUARD.GuardError):
                GUARD.prepare_submission(
                    argv,
                    self.workspace,
                    self.workspace,
                    self.slurm_python,
                )

    def test_rejects_heterogeneous_payload(self) -> None:
        with self.assertRaisesRegex(GUARD.GuardError, "heterogeneous"):
            GUARD.prepare_submission(
                [str(self._script()), ":", str(self._script())],
                self.workspace,
                self.workspace,
                self.slurm_python,
            )

    def test_rejects_script_outside_workspace(self) -> None:
        outside = self.workspace.parent / "outside.sh"
        outside.write_text("#!/bin/bash\n")
        with self.assertRaisesRegex(GUARD.GuardError, "assigned workspace"):
            self._prepare(outside)

    def test_rejects_stdin_script(self) -> None:
        with self.assertRaisesRegex(GUARD.GuardError, "one workspace file"):
            GUARD.prepare_submission(
                [],
                self.workspace,
                self.workspace,
                self.slurm_python,
                "#!/bin/bash\n",
            )

    def test_rejects_unsafe_directive(self) -> None:
        for directive in (
            "--chdir=/projects/ml",
            "--constraint=A4000",
            "--container=/tmp/image.sif",
            "--export=ALL",
            "--export-file=/tmp/environment",
            "--input=/etc/passwd",
        ):
            with self.subTest(directive=directive):
                script = self._script(extra_directive=f"#SBATCH {directive}\n")
                with self.assertRaisesRegex(GUARD.GuardError, "unsafe or unsupported"):
                    self._prepare(script)

    def test_accepts_submitit_array_nice_and_exclude(self) -> None:
        script = self._script(
            extra_directive=(
                "#SBATCH --array=0-9%2\n"
                "#SBATCH --exclude=g[1101-1102]\n"
                "#SBATCH --nice=100\n"
            )
        )
        _, count, _ = self._prepare(script)
        self.assertEqual(count, 2)

    def test_merges_required_node_exclusion(self) -> None:
        self.assertEqual(
            GUARD._merged_excluded_nodes("g[1101-1102]", "g1120"),
            "g[1101-1102],g1120",
        )
        self.assertEqual(GUARD._merged_excluded_nodes(None, "g1120"), "g1120")
        with self.assertRaisesRegex(GUARD.GuardError, "unsupported --exclude"):
            GUARD._merged_excluded_nodes(None, "g1120;uname")

    def test_accepts_array_using_exactly_five_total_gpus(self) -> None:
        script = self._script(
            gres="gpu:small:1",
            extra_directive="#SBATCH --array=0-11%5\n",
        )
        _, count, _ = self._prepare(script)
        self.assertEqual(count, 1)

    def test_rejects_array_over_five_total_gpus(self) -> None:
        script = self._script(
            gres="gpu:small:1",
            extra_directive="#SBATCH --array=0-11%12\n",
        )
        with self.assertRaisesRegex(GUARD.GuardError, "could reserve 12 GPUs"):
            self._prepare(script)

    def test_rejects_unthrottled_array_over_total(self) -> None:
        script = self._script(
            gres="gpu:small:2",
            extra_directive="#SBATCH --array=0-2\n",
        )
        with self.assertRaisesRegex(GUARD.GuardError, "could reserve 6 GPUs"):
            self._prepare(script)

    def test_accepts_submitit_srun_without_export_flag(self) -> None:
        script = self._script()
        script.write_text(script.read_text().replace(" --export=ALL", ""))
        _, count, _ = self._prepare(script)
        self.assertEqual(count, 2)

    def test_rejects_output_outside_workspace(self) -> None:
        script = self._script()
        text = script.read_text().replace(
            f"#SBATCH --error={self.workspace}/logs", "#SBATCH --error=/tmp"
        )
        script.write_text(text)
        with self.assertRaisesRegex(GUARD.GuardError, "must stay inside"):
            self._prepare(script)

    def test_rejects_shell_metacharacters_in_srun_path(self) -> None:
        script = self._script()
        text = script.read_text().replace(
            f"--output {self.workspace}/logs", "--output $(touch /tmp/pwn)/logs"
        )
        script.write_text(text)
        with self.assertRaisesRegex(GUARD.GuardError, "unsafe characters"):
            self._prepare(script)

    def test_rejects_extra_executable_body(self) -> None:
        valid = self._script().read_text()
        script = self.workspace / "logs" / ".submission_file_extra.sh"
        script.write_text(valid + "touch owned\n")
        with self.assertRaisesRegex(GUARD.GuardError, "body must contain only"):
            self._prepare(script)

    def test_rejects_non_bash_shebang(self) -> None:
        script = self._script()
        script.write_text(script.read_text().replace("#!/bin/bash", "#!/bin/sh", 1))
        with self.assertRaisesRegex(GUARD.GuardError, "exact #!/bin/bash"):
            self._prepare(script)

    def test_rejects_wrong_slurm_python(self) -> None:
        script = self._script()
        script.write_text(
            script.read_text().replace(str(self.slurm_python), "/usr/bin/python")
        )
        with self.assertRaisesRegex(GUARD.GuardError, "controller-provided"):
            self._prepare(script)

    def test_accepts_alias_to_controller_slurm_python(self) -> None:
        alias = self.slurm_python.parent / "slurm-python-alias"
        alias.symlink_to(self.slurm_python)
        script = self._script()
        script.write_text(
            script.read_text().replace(str(self.slurm_python), str(alias))
        )
        self._prepare(script)

    def test_rejects_pickle_path_outside_workspace(self) -> None:
        script = self._script()
        text = script.read_text()
        old = f"submitit.core._submit {self.workspace}/logs/%j"
        script.write_text(text.replace(old, "submitit.core._submit /tmp/stolen"))
        with self.assertRaisesRegex(GUARD.GuardError, "must stay inside"):
            self._prepare(script)

    def test_rejects_model_specific_gpu_type(self) -> None:
        with self.assertRaisesRegex(GUARD.GuardError, "only gpu:small:N"):
            self._prepare(self._script(gres="gpu:a4000:1"))

    def test_rejects_gpu_bf_partition(self) -> None:
        with self.assertRaisesRegex(GUARD.GuardError, "partition must be cpu or gpu"):
            self._prepare(self._script(partition="gpu-bf", gres="gpu:small:1"))

    def test_rejects_untyped_submitit_gpu(self) -> None:
        with self.assertRaisesRegex(GUARD.GuardError, "found untyped"):
            self._prepare(self._script(gres="gpu:1"))

    def test_accepts_five_gpus_in_submitit_script(self) -> None:
        argv, count, _ = self._prepare(self._script(gres="gpu:small:5"))
        self.assertEqual(count, 5)
        self.assertIn("--gres=gpu:small:5", argv)
        self.assertIn("--constraint=A4000", argv)

    def test_broker_uses_validated_snapshot_and_minimal_environment(self) -> None:
        root = self.workspace.parent
        capture = root / "capture.json"
        fake_sbatch = root / "fake-sbatch"
        fake_sbatch.write_text(
            "#!/usr/bin/python3\n"
            "import json, os, sys\n"
            f"with open({str(capture)!r}, 'w') as handle:\n"
            "    json.dump({'argv': sys.argv[1:], 'env': dict(os.environ), "
            "'stdin': sys.stdin.read()}, handle)\n"
            "print('Submitted batch job 123')\n"
        )
        fake_sbatch.chmod(0o700)
        script = self._script()
        expected_script = script.read_text()
        socket_path = root / "broker.sock"
        server = GUARD._BrokerServer(
            str(socket_path),
            str(fake_sbatch),
            str(fake_sbatch),
            str(fake_sbatch),
            str(fake_sbatch),
            str(fake_sbatch),
            self.workspace,
            self.slurm_python,
            5,
            "small",
            exclude_nodes="g1120",
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            request = {
                "argv": [str(script)],
                "cwd": str(self.workspace),
                "stdin": "",
                "env": {
                    "GITHUB_TOKEN": "must-not-leak",
                    "HOME": "/tmp/attacker-home",
                    "PATH": "/tmp/attacker-bin",
                    "HBPLUS_COPY": "/science/hbplus",
                    "CCD_MIRROR_PATH": "/science/ccd",
                    "PIPELINES_EXPERIMENT_FLAG": "enabled",
                },
                "sbatch_env": {},
            }
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.connect(str(socket_path))
                GUARD._send_message(connection, request)
                response = GUARD._receive_message(connection)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.assertEqual(response["returncode"], 0)
        captured = json.loads(capture.read_text())
        self.assertEqual(
            captured["argv"],
            [
                f"--chdir={self.workspace}",
                "--export=CCD_MIRROR_PATH,HBPLUS_COPY,HOME,LANG,LC_ALL,PATH,PIPELINES_EXPERIMENT_FLAG,TMPDIR",
                "--gres=gpu:small:2",
                "--constraint=A4000",
                "--exclude=g1120",
            ],
        )
        self.assertEqual(captured["stdin"], expected_script)
        self.assertNotIn("GITHUB_TOKEN", captured["env"])
        self.assertEqual(captured["env"]["HOME"], str(self.workspace / ".home"))
        self.assertEqual(captured["env"]["TMPDIR"], str(self.workspace / ".tmp"))
        self.assertEqual(captured["env"]["PATH"], "/usr/bin:/bin")
        self.assertEqual(captured["env"]["HBPLUS_COPY"], "/science/hbplus")
        self.assertEqual(captured["env"]["CCD_MIRROR_PATH"], "/science/ccd")
        self.assertEqual(captured["env"]["PIPELINES_EXPERIMENT_FLAG"], "enabled")

    def test_forwarded_environment_rejects_non_string_values(self) -> None:
        with self.assertRaisesRegex(GUARD.GuardError, "must be strings"):
            GUARD._forwarded_environment({"HBPLUS_COPY": None})

    def test_forwarded_environment_includes_scoring_site_paths(self) -> None:
        expected = {
            name: f"/science/{name.lower()}"
            for name in (
                "HBPLUS_COPY",
                "CCD_MIRROR_PATH",
                "PYROSETTA_PATH",
                "X3DNA_BIN_DIR",
                "X3DNA_DIR",
            )
        }
        self.assertEqual(GUARD._forwarded_environment(expected), expected)

    def test_forwarded_environment_ignores_untrusted_overrides(self) -> None:
        self.assertEqual(
            GUARD._forwarded_environment(
                {
                    "HOME": "/escape",
                    "PATH": "/escape/bin",
                    "LD_PRELOAD": "/escape/library.so",
                    "HBPLUS_COPY": "/science/hbplus",
                }
            ),
            {"HBPLUS_COPY": "/science/hbplus"},
        )

    def test_broker_enforces_total_across_overlapping_submissions(self) -> None:
        root = self.workspace.parent
        submission_count = root / "submission-count"
        active = root / "active"
        active.write_text("yes")
        fake_sbatch = root / "fake-sbatch"
        fake_sbatch.write_text(
            "#!/usr/bin/python3\n"
            "from pathlib import Path\n"
            f"path = Path({str(submission_count)!r})\n"
            "count = int(path.read_text()) + 1 if path.exists() else 1\n"
            "path.write_text(str(count))\n"
            "print(f'Submitted batch job {100 + count}')\n"
        )
        fake_sbatch.chmod(0o700)
        fake_squeue = root / "fake-squeue"
        fake_squeue.write_text(
            "#!/usr/bin/python3\n"
            "import sys\n"
            "from pathlib import Path\n"
            f"if Path({str(active)!r}).exists():\n"
            "    jobs = next(arg for arg in sys.argv if arg.startswith('--jobs='))\n"
            "    print('\\n'.join(jobs.split('=', 1)[1].split(',')))\n"
            "else:\n"
            "    print('slurm_load_jobs error: Invalid job id specified', file=sys.stderr)\n"
            "    raise SystemExit(1)\n"
        )
        fake_squeue.chmod(0o700)
        socket_path = root / "total-broker.sock"
        server = GUARD._BrokerServer(
            str(socket_path),
            str(fake_sbatch),
            str(fake_sbatch),
            str(fake_sbatch),
            str(fake_sbatch),
            str(fake_squeue),
            self.workspace,
            self.slurm_python,
            5,
            "small",
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def request() -> dict[str, object]:
            payload = {
                "argv": [str(self._script(gres="gpu:small:3"))],
                "cwd": str(self.workspace),
                "stdin": "",
                "env": {},
                "sbatch_env": {},
            }
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.connect(str(socket_path))
                GUARD._send_message(connection, payload)
                return GUARD._receive_message(connection)

        try:
            first = request()
            overlapping = request()
            active.unlink()
            after_completion = request()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.assertEqual(first["returncode"], 0)
        self.assertEqual(overlapping["returncode"], 2)
        self.assertIn("already reserves 3 GPU(s)", overlapping["stderr"])
        self.assertEqual(after_completion["returncode"], 0)
        self.assertEqual(submission_count.read_text(), "2")

    def test_control_commands_are_scoped_to_trial_jobs(self) -> None:
        fake = self.workspace.parent / "fake-slurm"
        fake.write_text("#!/bin/sh\nexit 0\n")
        fake.chmod(0o700)
        socket_path = self.workspace.parent / "control-broker.sock"
        server = GUARD._BrokerServer(
            str(socket_path),
            str(fake),
            str(fake),
            str(fake),
            str(fake),
            str(fake),
            self.workspace,
            self.slurm_python,
            5,
            "small",
        )
        try:
            server.register_submission("Submitted batch job 123\n", "owned-job")
            self.assertEqual(server.validate_scancel(["123"]), ["123"])
            self.assertEqual(server.validate_scancel(["123_4"]), ["123_4"])
            self.assertEqual(
                server.validate_scancel(["-n", "owned-job"]),
                ["-n", "owned-job"],
            )
            with self.assertRaisesRegex(GUARD.GuardError, "only jobs created"):
                server.validate_scancel(["999"])
            with self.assertRaisesRegex(GUARD.GuardError, "only jobs created"):
                server.validate_scancel(["-n", "someone-elses-job"])
            self.assertEqual(
                server.validate_squeue(
                    ["--noheader", "--job", "123", "--format", "%T"]
                ),
                ["--noheader", "--jobs=123", "--format", "%T"],
            )
            self.assertEqual(
                server.validate_squeue(
                    ["--noheader", "--job", "123_4", "--format", "%T"]
                ),
                ["--noheader", "--jobs=123_4", "--format", "%T"],
            )
            with self.assertRaisesRegex(GUARD.GuardError, "only jobs created"):
                server.validate_squeue(["--noheader", "--job", "999", "--format", "%T"])
            with self.assertRaisesRegex(GUARD.GuardError, "exact trial-owned"):
                server.validate_squeue(
                    ["--noheader", "--job", "123", "--format", "%all"]
                )
            with self.assertRaisesRegex(GUARD.GuardError, "exact trial-owned"):
                server.validate_squeue(["--job", "123"])
            self.assertEqual(
                server.validate_scontrol(["show", "config"]),
                ["show", "config"],
            )
            with self.assertRaisesRegex(GUARD.GuardError, "query-only"):
                server.validate_scontrol(["update", "JobId=123", "Partition=gpu"])
        finally:
            server.server_close()

    def test_submission_registry_is_durable_and_restored_after_restart(self) -> None:
        fake = self.workspace.parent / "fake-slurm-registry"
        fake.write_text("#!/bin/sh\nexit 0\n")
        fake.chmod(0o700)
        registry = self.workspace.parent / "runtime" / "submissions.jsonl"

        first = GUARD._BrokerServer(
            str(self.workspace.parent / "registry-broker-one.sock"),
            str(fake),
            str(fake),
            str(fake),
            str(fake),
            str(fake),
            self.workspace,
            self.slurm_python,
            5,
            "small",
            submission_registry=registry,
        )
        try:
            first.register_submission(
                "Submitted batch job 456\n", "durable-job", gpu_reservation=2
            )
            # Repeated acknowledgements never duplicate the durable record.
            first.register_submission(
                "Submitted batch job 456\n", "durable-job", gpu_reservation=2
            )
        finally:
            first.server_close()

        records = [json.loads(line) for line in registry.read_text().splitlines()]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["job_id"], "456")
        self.assertEqual(records[0]["gpu_reservation"], 2)

        second = GUARD._BrokerServer(
            str(self.workspace.parent / "registry-broker-two.sock"),
            str(fake),
            str(fake),
            str(fake),
            str(fake),
            str(fake),
            self.workspace,
            self.slurm_python,
            5,
            "small",
            submission_registry=registry,
        )
        try:
            self.assertEqual(second.validate_scancel(["456"]), ["456"])
            self.assertEqual(second._gpu_reservations, {"456": 2})
        finally:
            second.server_close()

    def test_cleanup_cancels_registered_child_jobs(self) -> None:
        capture = self.workspace.parent / "cancelled.txt"
        fake = self.workspace.parent / "fake-scancel"
        fake.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$@\" > {str(capture)!r}\n")
        fake.chmod(0o700)
        fake_squeue = self.workspace.parent / "fake-squeue"
        fake_squeue.write_text("#!/bin/sh\nprintf '123\\n321\\n'\n")
        fake_squeue.chmod(0o700)
        socket_path = self.workspace.parent / "cleanup-broker.sock"
        server = GUARD._BrokerServer(
            str(socket_path),
            str(fake),
            str(fake),
            str(fake),
            str(fake),
            str(fake_squeue),
            self.workspace,
            self.slurm_python,
            5,
            "small",
        )
        try:
            server.register_submission("Submitted batch job 321\n", "job-a")
            server.register_submission("Submitted batch job 123\n", "job-b")
            server.cancel_registered_jobs()
            self.assertEqual(capture.read_text().splitlines(), ["123", "321"])
        finally:
            server.server_close()

    def test_cleanup_cancels_only_active_registered_child_jobs(self) -> None:
        capture = self.workspace.parent / "active-cancelled.txt"
        fake_scancel = self.workspace.parent / "active-fake-scancel"
        fake_scancel.write_text(
            f"#!/bin/sh\nprintf '%s\\n' \"$@\" > {str(capture)!r}\n"
        )
        fake_scancel.chmod(0o700)
        fake_squeue = self.workspace.parent / "active-fake-squeue"
        fake_squeue.write_text("#!/bin/sh\nprintf '321\\n'\n")
        fake_squeue.chmod(0o700)
        socket_path = self.workspace.parent / "active-cleanup-broker.sock"
        server = GUARD._BrokerServer(
            str(socket_path),
            str(fake_scancel),
            str(fake_scancel),
            str(fake_scancel),
            str(fake_scancel),
            str(fake_squeue),
            self.workspace,
            self.slurm_python,
            5,
            "a4000",
        )
        try:
            server.register_submission("Submitted batch job 321\n", "active-job")
            server.register_submission("Submitted batch job 123\n", "finished-job")
            server.cancel_registered_jobs()
            self.assertEqual(capture.read_text().splitlines(), ["321"])
        finally:
            server.server_close()


if __name__ == "__main__":
    unittest.main()
