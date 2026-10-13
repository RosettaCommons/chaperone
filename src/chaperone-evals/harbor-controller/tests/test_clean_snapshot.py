#!/usr/bin/env python3
"""Regression tests for stripping Git metadata from agent workspaces."""

from __future__ import annotations

import errno
import json
import os
import pwd
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from environments.apptainer import ApptainerEnvironment
from task_paths import PRIVATE_TASKS_ENV
from scripts.snapshot_harness import (
    assert_no_evaluation_hints,
    assert_no_git_metadata,
    audit_model_instructions,
    build,
    validate_advertised_skills,
    validate_snapshot,
)


# Placeholders: the real fingerprints are held out and live in the private tree.
PLACEHOLDER_HINTS = {
    "secret-target": r"\bsecret-target\b",
    "SECRET-ID-7": r"\bsecret[-_ ]?id[-_ ]?7\b",
    "cutoff 0.123": r"(?<!\d)0\.123(?!\d)",
    "range Z1-9": r"\bZ1-9\b",
    "panel of 777": r"\b777\s+(?:natural|reference)",
}


class CleanSnapshotTest(unittest.TestCase):
    def setUp(self) -> None:
        private = tempfile.TemporaryDirectory()
        self.addCleanup(private.cleanup)
        (Path(private.name) / "audit_hints.json").write_text(
            json.dumps(
                [{"label": k, "pattern": v} for k, v in PLACEHOLDER_HINTS.items()]
            )
        )
        patcher = patch.dict(os.environ, {PRIVATE_TASKS_ENV: private.name})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_missing_artifact_staging_mount_is_recreated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            staging = Path(temporary) / "removed-staging"
            environment = object.__new__(ApptainerEnvironment)
            environment._staging_dir = staging

            environment._ensure_staging_dir()

            self.assertTrue(staging.is_dir())
            self.assertEqual(stat.S_IMODE(staging.stat().st_mode), 0o755)

    def test_runtime_identity_uses_real_workspace_local_home(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "runtime"
            workspace = root / "workspace"

            ApptainerEnvironment._prepare_runtime_identity(runtime, workspace)

            current = pwd.getpwuid(os.getuid())
            passwd_lines = (runtime / "passwd").read_text().splitlines()
            own_record = next(
                line.split(":")
                for line in passwd_lines
                if line.split(":", 1)[0] == current.pw_name
            )
            self.assertEqual(own_record[5], str(workspace.resolve() / ".home"))
            self.assertTrue((workspace / ".home").is_dir())

    def test_project_path_aliases_cover_both_cluster_spellings(self) -> None:
        self.assertEqual(
            ApptainerEnvironment._project_path_alias(Path("/mnt/projects/a/b")),
            Path("/projects/a/b"),
        )
        self.assertEqual(
            ApptainerEnvironment._project_path_alias(Path("/projects/a/b")),
            Path("/mnt/projects/a/b"),
        )
        self.assertEqual(
            ApptainerEnvironment._project_path_alias(Path("/mnt/net/a/b")),
            Path("/net/a/b"),
        )
        self.assertEqual(
            ApptainerEnvironment._project_path_alias(Path("/net/a/b")),
            Path("/mnt/net/a/b"),
        )
        self.assertIsNone(ApptainerEnvironment._project_path_alias(Path("/tmp/a")))

    def test_sbatch_broker_uses_short_private_socket_path(self) -> None:
        directory = ApptainerEnvironment._make_sbatch_broker_dir()
        try:
            self.assertLessEqual(len(os.fsencode(directory / "sbatch.sock")), 107)
            self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o700)
        finally:
            shutil.rmtree(directory)

    def test_outer_runtime_mounts_do_not_expose_raw_slurm_controls(self) -> None:
        mounted_sources = {
            str(mount["source"]) for mount in ApptainerEnvironment._runtime_mounts()
        }
        self.assertNotIn("/usr/bin/scancel", mounted_sources)
        self.assertNotIn("/usr/bin/scontrol", mounted_sources)
        self.assertNotIn("/usr/bin/srun", mounted_sources)

    def test_runtime_mounts_include_on_prem_mpnn_code(self) -> None:
        mounted_sources = {
            str(mount["source"]) for mount in ApptainerEnvironment._runtime_mounts()
        }
        required_sources = {
            "/projects/ml/modelhub",
            "/net/software/lab/fused_mpnn",
            "/net/tukwila/rohith/RFdiffusion2",
            "/net/databases/lab/aa-diffusion",
            "/net/software/containers/colabfold.sif",
            "/projects/ml/esm-weights",
            "/net/databases/mpnn",
        }
        self.assertLessEqual(required_sources, mounted_sources)
        for mount in ApptainerEnvironment._runtime_mounts():
            if str(mount["source"]) in required_sources:
                self.assertTrue(mount["read_only"])

    def test_worker_wrapper_uses_node_local_tmp_and_dns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            (workspace / "chaperone" / "src").mkdir(parents=True)
            image = root / "worker.sif"
            image.touch()
            wrapper = root / "slurm-python"
            ApptainerEnvironment._prepare_slurm_python(wrapper, workspace, image)
            script = wrapper.read_text()
            self.assertIn("SLURM_TMPDIR", script)
            self.assertIn("SLURM_ARRAY_TASK_ID", script)
            self.assertIn("runtime_bind_args", script)
            self.assertIn("/run/systemd/resolve/resolv.conf", script)
            self.assertIn("elif [[ -s /etc/resolv.conf ]]", script)
            self.assertIn(
                'if [[ -v "$name" ]]; then env_args+=(--env "$name=${!name}"); fi',
                script,
            )
            self.assertIn("HBPLUS_COPY", script)
            self.assertIn("PYROSETTA_PATH", script)
            self.assertIn("X3DNA_BIN_DIR", script)
            self.assertIn("X3DNA_DIR", script)
            self.assertIn("OPENBLAS_NUM_THREADS", script)
            self.assertIn("OMP_NUM_THREADS", script)
            self.assertIn("POLARS_MAX_THREADS", script)
            self.assertIn("TOKIO_WORKER_THREADS", script)
            self.assertIn("USALIGN_BIN", script)
            subprocess.run(["bash", "-n", str(wrapper)], check=True)
            for path in (
                "/projects/ml/modelhub",
                "/net/software/lab/fused_mpnn",
                "/net/tukwila/rohith/RFdiffusion2",
                "/net/databases/lab/aa-diffusion",
            ):
                self.assertIn(f"{path}:{path}:ro", script)

    def test_worker_wrapper_reasserts_site_environment_after_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            (workspace / "chaperone" / "src").mkdir(parents=True)
            image = root / "worker.sif"
            image.touch()
            wrapper = root / "slurm-python"
            cache = workspace / "score_run" / ".cache" / "af3_jax"

            ApptainerEnvironment._prepare_slurm_python(
                wrapper,
                workspace,
                image,
                site_environment={
                    "AF3_JAX_CACHE_DIR": str(cache),
                    "HBPLUS_COPY": "/trusted/hb plus",
                },
            )

            script = wrapper.read_text()
            source_position = script.index(".env.ipd")
            for export in ("export AF3_JAX_CACHE_DIR=", "export HBPLUS_COPY="):
                self.assertGreater(script.index(export), source_position)
            self.assertIn(str(cache), script)
            self.assertIn("/trusted/hb plus", script)
            subprocess.run(["bash", "-n", str(wrapper)], check=True)

    def test_worker_wrapper_without_site_environment_exports_nothing_extra(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            (workspace / "chaperone" / "src").mkdir(parents=True)
            image = root / "worker.sif"
            image.touch()
            wrapper = root / "slurm-python"

            ApptainerEnvironment._prepare_slurm_python(wrapper, workspace, image)

            self.assertNotIn("export AF3_JAX_CACHE_DIR=", wrapper.read_text())

    def test_worker_site_environment_rejects_unapproved_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            (workspace / "chaperone" / "src").mkdir(parents=True)
            image = root / "worker.sif"
            image.touch()

            for name in ("BAD-NAME", "OPENAI_API_KEY", "PATH"):
                with self.subTest(name=name):
                    with self.assertRaisesRegex(ValueError, "unapproved"):
                        ApptainerEnvironment._prepare_slurm_python(
                            root / "slurm-python",
                            workspace,
                            image,
                            site_environment={name: "value"},
                        )

    def test_worker_wrapper_mounts_chaperone_checkout_at_its_path_alias(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            chaperone_repo = root / "chaperone"
            (chaperone_repo / "src").mkdir(parents=True)
            image = root / "worker.sif"
            image.touch()
            wrapper = root / "slurm-python"
            alias = Path("/mnt/projects/ml/pipelines/evals/pipelines")
            original_alias = ApptainerEnvironment._project_path_alias

            def project_alias(path: Path) -> Path | None:
                if path == chaperone_repo:
                    return alias
                return original_alias(path)

            with patch.object(
                ApptainerEnvironment,
                "_project_path_alias",
                side_effect=project_alias,
            ):
                ApptainerEnvironment._prepare_slurm_python(
                    wrapper,
                    workspace,
                    image,
                    chaperone_repo=chaperone_repo,
                )

            self.assertIn(f"{chaperone_repo}:{alias}:ro", wrapper.read_text())

    def test_worker_wrapper_mounts_task_assets_at_both_project_spellings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            chaperone_repo = root / "chaperone"
            (chaperone_repo / "src").mkdir(parents=True)
            image = root / "worker.sif"
            image.touch()
            wrapper = root / "slurm-python"
            task = Path("/mnt/projects/ml/shared/evals_task/example_task")
            physical_task = Path("/projects/ml/shared/evals_task/example_task")
            original_exists = Path.exists

            def exists(path: Path) -> bool:
                if path == physical_task:
                    return True
                return original_exists(path)

            with patch.object(Path, "exists", new=exists):
                ApptainerEnvironment._prepare_slurm_python(
                    wrapper,
                    workspace,
                    image,
                    chaperone_repo=chaperone_repo,
                    worker_readonly_paths=(task,),
                )

            script = wrapper.read_text()
            self.assertIn(f"{physical_task}:{task}:ro", script)
            self.assertIn(f"{physical_task}:{physical_task}:ro", script)

    def test_worker_wrapper_can_be_regenerated_after_read_only_install(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            (workspace / "chaperone" / "src").mkdir(parents=True)
            image = root / "worker.sif"
            image.touch()
            wrapper = root / "slurm-python"

            ApptainerEnvironment._prepare_slurm_python(wrapper, workspace, image)
            self.assertEqual(stat.S_IMODE(wrapper.stat().st_mode), 0o500)
            ApptainerEnvironment._prepare_slurm_python(wrapper, workspace, image)

            self.assertEqual(stat.S_IMODE(wrapper.stat().st_mode), 0o500)
            subprocess.run(["bash", "-n", str(wrapper)], check=True)

    def test_unstatable_optional_runtime_path_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            (workspace / "chaperone" / "src").mkdir(parents=True)
            image = root / "worker.sif"
            image.touch()
            wrapper = root / "slurm-python"
            optional = Path("/net/optional/capability")
            original_exists = Path.exists

            def exists(path: Path) -> bool:
                if path == optional:
                    raise OSError(22, "Invalid argument", str(path))
                return original_exists(path)

            with (
                patch(
                    "environments.apptainer._runtime_bind_pairs",
                    return_value=[(str(optional), str(optional))],
                ),
                patch.object(Path, "exists", new=exists),
            ):
                self.assertEqual(ApptainerEnvironment._runtime_mounts(), [])
                ApptainerEnvironment._prepare_slurm_python(wrapper, workspace, image)

            self.assertNotIn(str(optional), wrapper.read_text())

    def test_missing_required_worker_image_still_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            (workspace / "chaperone" / "src").mkdir(parents=True)
            with self.assertRaisesRegex(
                FileNotFoundError, "Shared Chaperone worker SIF was not found"
            ):
                ApptainerEnvironment._prepare_slurm_python(
                    root / "slurm-python", workspace, root / "missing.sif"
                )

    def test_snapshot_validation_rejects_nested_git_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "library" / ".git").mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError, r"library/.git"):
                assert_no_git_metadata(root)

    def test_snapshot_rejects_eval_target_hints_in_allowlisted_files(self) -> None:
        hints = (
            "secret-target",
            "SECRET-ID-7",
            "cutoff 0.123",
            "range Z1-9",
            "777 natural references",
        )
        for index, hint in enumerate(hints):
            with self.subTest(hint=hint), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = self._minimal_source(root)
                leak = source / "src" / "chaperone" / "target_hint.py"
                leak.write_text(f'TARGET_HINT = "{hint}"\n')

                with self.assertRaisesRegex(
                    RuntimeError, "evaluation-specific hints.*target_hint.py"
                ):
                    build(source, root / f"bundles-{index}")

    def test_hints_are_found_through_simple_obfuscation_and_in_file_names(self) -> None:
        variants = {
            "zero_width.py": 'T = "secret\u200b-target"\n'.encode("utf-8"),
            "fullwidth.py": 'T = "\uff53ecret-target"\n'.encode("utf-8"),
            "utf16.py": 'T = "secret-target"\n'.encode("utf-16"),
        }
        for index, (name, payload) in enumerate(variants.items()):
            with self.subTest(variant=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = self._minimal_source(root)
                (source / "src" / "chaperone" / name).write_bytes(payload)
                with self.assertRaisesRegex(RuntimeError, "evaluation-specific hints"):
                    build(source, root / f"bundles-{index}")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._minimal_source(root)
            (source / "src" / "chaperone" / "secret-target-notes.py").write_text(
                "X = 1\n"
            )
            with self.assertRaisesRegex(RuntimeError, "in the file name"):
                build(source, root / "bundles-name")

    def test_forbidden_directories_match_case_insensitively(self) -> None:
        for directory in ("Examples", "CAMPAIGNS", "Tests"):
            with (
                self.subTest(directory=directory),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                source = self._minimal_source(root)
                folder = source / "src" / "chaperone" / directory
                folder.mkdir(parents=True)
                (folder / "run.py").write_text("X = 1\n")
                with self.assertRaisesRegex(
                    RuntimeError, "forbidden generated or evaluation"
                ):
                    build(source, root / "bundles")

    def test_bundle_fails_closed_without_audit_patterns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._minimal_source(root)
            (root / "empty").mkdir()
            with patch.dict(os.environ, {PRIVATE_TASKS_ENV: str(root / "empty")}):
                with self.assertRaisesRegex(RuntimeError, "audit patterns not found"):
                    build(source, root / "bundles")

    def test_snapshot_hint_audit_accepts_generic_rna_apis(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = self._minimal_source(Path(temporary))
            generic = source / "src" / "chaperone" / "rna_api.py"
            generic.write_text(
                'DESCRIPTION = "RNA motif base pairing and sequence identity"\n'
            )

            assert_no_evaluation_hints(source)

    @staticmethod
    def _minimal_source(root: Path) -> Path:
        source = root / "source"
        (source / "src" / "blocks").mkdir(parents=True)
        (source / "src" / "chaperone").mkdir(parents=True)
        (source / "src" / "chaperone_cli").mkdir(parents=True)
        (source / ".agents" / "skills" / "design").mkdir(parents=True)
        for name in ("AGENTS.md", "README.md", "SKILL.md"):
            (source / name).write_text(f"{name}\n")
        (source / "pyproject.toml").write_text("[project]\nname='chaperone'\n")
        (source / "src" / "blocks" / "__init__.py").write_text("BLOCKS = True\n")
        (source / "src" / "chaperone" / "__init__.py").write_text("CHAPERONE = True\n")
        (source / "src" / "chaperone_cli" / "job_wait.py").write_text(
            "def wait_for_completion(): pass\n"
            "def launch_and_wait(): pass\n"
            "def serve_mcp(): pass\n"
        )
        (source / "src" / "chaperone_cli" / "main.py").write_text(
            "from chaperone_cli.job_wait import serve_mcp\n"
        )
        (source / ".agents" / "skills" / "design" / "SKILL.md").write_text("# Design\n")
        return source

    def test_bundle_copies_only_raw_runtime_and_skill_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._minimal_source(root)
            (source / ".git" / "refs" / "remotes" / "origin").mkdir(parents=True)
            (source / ".git" / "refs" / "remotes" / "origin" / "secret").write_text(
                "commit\n"
            )
            (source / "docs" / "site" / "evals").mkdir(parents=True)
            (source / "docs" / "site" / "evals" / "index.html").write_text(
                "held-out gates\n"
            )
            (source / "docs" / "site" / "evals" / "score_per_design.csv").write_text(
                "sequence,score\n"
            )
            (source / "docs" / "site" / "blocks-reference.md").write_text(
                "# Generated block reference\n"
            )
            (source / "campaigns" / "old").mkdir(parents=True)
            (source / "campaigns" / "old" / "selection.py").write_text(
                "SECRET = True\n"
            )
            (source / "src" / "chaperone" / "fresh.py").write_text("FRESH = True\n")
            (source / "src" / "chaperone" / "evals.py").write_text(
                "EVALUATION_HELPER = True\n"
            )

            seed = build(source, root / "bundles")
            validate_snapshot(seed)

            self.assertTrue((seed / "src" / "chaperone" / "fresh.py").is_file())
            self.assertTrue(
                (seed / ".agents" / "skills" / "design" / "SKILL.md").is_file()
            )
            self.assertFalse((seed / ".git").exists())
            self.assertTrue((seed / "docs" / "site" / "blocks-reference.md").is_file())
            self.assertFalse((seed / "docs" / "site" / "evals").exists())
            self.assertFalse((seed / "campaigns").exists())
            self.assertFalse((seed / "src" / "chaperone" / "evals.py").exists())
            self.assertFalse((seed / "README.md").exists())
            self.assertFalse(any(seed.rglob("score_per_design.csv")))
            self.assertFalse(any(seed.rglob(".git")))

    def test_snapshot_preflight_rejects_dangling_agent_skill_route(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = self._minimal_source(Path(temporary))
            (source / "AGENTS.md").write_text("Use `/design` or `/missing-skill`.\n")

            with self.assertRaisesRegex(RuntimeError, "missing-skill.*AGENTS.md"):
                validate_advertised_skills(source)

    def test_snapshot_preflight_rejects_dangling_index_skill_route(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = self._minimal_source(Path(temporary))
            index = source / ".agents" / "skills" / "INDEX.md"
            index.write_text(
                "| Skill | Use |\n|---|---|\n"
                "| `design` | plan |\n| `missing-index` | unavailable |\n"
            )

            with self.assertRaisesRegex(RuntimeError, "missing-index.*INDEX.md"):
                build(source, Path(temporary) / "bundles")

    def test_bundle_never_includes_the_evals_tree_under_src(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._minimal_source(root)
            evals = source / "src" / "chaperone-evals"
            (evals / "harbor-controller" / "scripts").mkdir(parents=True)
            (evals / "harbor-controller" / "scripts" / "score.py").write_text(
                "HELD_OUT = True\n"
            )
            (evals / "evals_task" / "task" / "inputs").mkdir(parents=True)
            (evals / "evals_task" / "task" / "inputs" / "target.pdb").write_text(
                "ATOM\n"
            )
            (evals / "evals_task" / "task" / "score").mkdir()
            (evals / "evals_task" / "task" / "score" / "score.py").write_text(
                "GATES = 0.5\n"
            )
            (evals / "evals_task" / "task" / "example").mkdir()
            (evals / "evals_task" / "task" / "example" / "submission.csv").write_text(
                "sequence\nMKV\n"
            )
            (evals / "scoring").mkdir()
            (evals / "scoring" / "__init__.py").write_text("GATES = 0.5\n")
            # A bootstrapped controller carries a virtualenv and a symlink to it;
            # neither may be walked, let alone rejected as a bundle symlink.
            (evals / ".venv" / "bin").mkdir(parents=True)
            (evals / ".venv" / "bin" / "python").symlink_to("/usr/bin/python3")
            (evals / "harbor-controller" / ".venv").symlink_to("../.venv")
            (source / "src" / "stray.py").write_text("STRAY = True\n")

            seed = build(source, root / "bundles")

            bundled = {
                path.relative_to(seed).as_posix()
                for path in seed.rglob("*")
                if path.is_file()
            }
            self.assertTrue(bundled)
            self.assertFalse(
                [name for name in bundled if "chaperone-evals" in name], bundled
            )
            self.assertNotIn("src/stray.py", bundled)
            self.assertEqual(validate_snapshot(seed, seed.name), seed.name)

            leaked = seed / "src" / "chaperone-evals" / "scoring"
            leaked.mkdir(parents=True)
            (leaked / "__init__.py").write_text("GATES = 0.5\n")
            with self.assertRaisesRegex(RuntimeError, "non-allowlisted"):
                validate_snapshot(seed)

    def test_top_level_examples_never_change_the_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._minimal_source(root)
            without = build(source, root / "without").name
            (source / "examples" / "demo").mkdir(parents=True)
            (source / "examples" / "demo" / "run.py").write_text("EXAMPLE = 1\n")

            seed = build(source, root / "with")

            self.assertEqual(seed.name, without)
            self.assertFalse(
                any(
                    "examples" in part
                    for part in seed.rglob("*")
                    for part in part.relative_to(seed).parts
                )
            )

    def test_example_injected_into_a_bundle_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._minimal_source(root)
            seed = build(source, root / "bundles")
            (seed / "src" / "blocks" / "examples").mkdir(parents=True)
            (seed / "src" / "blocks" / "examples" / "run.py").write_text("X = 1\n")

            with self.assertRaises(RuntimeError):
                validate_snapshot(seed)

    def test_snapshot_requires_chaperone_monitor_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._minimal_source(root)
            (source / "src" / "chaperone_cli" / "job_wait.py").unlink()

            with self.assertRaisesRegex(RuntimeError, "chaperone_cli/job_wait.py"):
                build(source, root / "bundles")

    def test_concurrent_bundle_publisher_reuses_valid_winner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._minimal_source(root)
            output_root = root / "bundles"

            def publish_competitor(staging: Path, destination: Path):
                shutil.copytree(staging, destination)
                raise OSError(errno.ENOTEMPTY, "Directory not empty", destination)

            with patch.object(Path, "rename", new=publish_competitor):
                seed = build(source, output_root)

            validate_snapshot(seed, seed.name)
            self.assertFalse(any(output_root.glob(f".{seed.name}.*")))

    def test_bundle_rejects_symlink_in_allowlisted_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._minimal_source(root)
            outside = root / "outside.py"
            outside.write_text("SECRET = True\n")
            (source / "src" / "chaperone" / "escape.py").symlink_to(outside)
            with self.assertRaisesRegex(RuntimeError, "Refusing symlink"):
                build(source, root / "bundles")

    def test_content_addressed_bundle_rejects_post_build_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seed = build(self._minimal_source(root), root / "bundles")
            (seed / "src" / "chaperone" / "__init__.py").write_text("TAMPERED = True\n")
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                validate_snapshot(seed, seed.name)

    def test_generated_prompt_audit_rejects_mandatory_bookkeeping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._minimal_source(root)
            instruction = root / "instruction.md"
            instruction.write_text(
                "You must create terminal_report.json before finishing.\n"
            )
            snapshot = build(source, root / "bundles")

            with self.assertRaisesRegex(RuntimeError, "controller-owned bookkeeping"):
                audit_model_instructions([instruction], snapshot)

            instruction.write_text(
                "Only submit designs.csv; terminal_report.json is not required.\n"
            )
            audit_model_instructions([instruction], snapshot)

    def test_workspace_copy_has_no_git_repository_or_history(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._minimal_source(root)
            seed = build(source, root / "bundles")
            inputs = root / "inputs"
            workspace = root / "workspace"
            inputs.mkdir()
            (inputs / "ligand.pdb").write_text("END\n")

            ApptainerEnvironment._prepare_workspace(workspace, seed, inputs)

            chaperone = workspace / "chaperone"
            validate_snapshot(chaperone)
            self.assertFalse((chaperone / ".git").exists())
            self.assertNotEqual(
                subprocess.run(
                    ["git", "rev-parse", "--is-inside-work-tree"],
                    cwd=chaperone,
                    capture_output=True,
                    text=True,
                ).returncode,
                0,
            )


if __name__ == "__main__":
    unittest.main()


def test_sanitized_snapshot_resources_match_the_scheduler_guard(tmp_path):
    """The bundle has no dotenv files; the controller must supply its site map."""
    import json
    import os
    import subprocess
    import sys

    from environments.apptainer import SCIENTIFIC_ENV_DEFAULTS
    from scripts.slurm_gpu_guard import validate_request
    from scripts.snapshot_harness import build

    root = Path(__file__).resolve().parents[4]
    snapshot = build(root, tmp_path / "snapshots")
    assert not list(snapshot.glob(".env*"))
    env = {
        **os.environ,
        **SCIENTIFIC_ENV_DEFAULTS,
        "PYTHONPATH": str(snapshot / "src"),
    }
    code = """
import json
import chaperone
from blocks import Blocks
from chaperone.settings import ChaperoneSettings
from chaperone.engine.resolve import _resolve_job_args, RuntimeArgs
chaperone._global_settings = ChaperoneSettings(_env_file=())
result = {}
for name in ('boltz2', 'mpnn', 'rfd2'):
    args = _resolve_job_args(None, Blocks[name].model_construct(name=name),
                             RuntimeArgs(use_local_gpus=False))
    result[name] = [args.partition, args.gres]
print(json.dumps(result))
"""
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, cwd=snapshot,
        check=True, capture_output=True, text=True,
    )
    resources = json.loads(result.stdout)
    assert resources["mpnn"] == ["cpu", None]
    for name in ("boltz2", "rfd2"):
        assert resources[name] == ["gpu", "gpu:small:1"]
        assert validate_request([], [f"#SBATCH --gres={resources[name][1]}"], 5) == 1
