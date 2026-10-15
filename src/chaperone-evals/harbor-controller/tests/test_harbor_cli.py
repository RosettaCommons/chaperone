from __future__ import annotations

import errno
import importlib.util
import os
import shlex
import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from harbor.models.trial.config import TaskConfig, TrialConfig


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "controller_harbor_cli", ROOT / "scripts" / "harbor_cli.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ReadableTrialNameTests(unittest.TestCase):
    def setUp(self) -> None:
        MODULE._TRIAL_NAME_COUNTS.clear()

    def test_first_attempt_has_no_random_suffix(self) -> None:
        config = TrialConfig(
            task=TaskConfig(path=Path("/tmp/biotin_binder_design_test"))
        )
        self.assertEqual(
            MODULE._readable_trial_name(config), "biotin_binder_design_test"
        )

    def test_harbor_model_validator_uses_patched_name_generator(self) -> None:
        original = TrialConfig.generate_trial_name
        TrialConfig.generate_trial_name = MODULE._readable_trial_name
        try:
            config = TrialConfig(
                task=TaskConfig(path=Path("/tmp/biotin_binder_design_test"))
            )
        finally:
            TrialConfig.generate_trial_name = original
        self.assertEqual(config.trial_name, "biotin_binder_design_test")

    def test_later_attempts_get_numeric_suffixes(self) -> None:
        config = TrialConfig(
            task=TaskConfig(path=Path("/tmp/biotin_binder_design_test"))
        )
        self.assertEqual(
            MODULE._readable_trial_name(config), "biotin_binder_design_test"
        )
        self.assertEqual(
            MODULE._readable_trial_name(config), "biotin_binder_design_test__2"
        )

    def test_codex_config_uses_direct_monitor_and_default_compaction(self) -> None:
        codex = SimpleNamespace(
            mcp_servers=[
                SimpleNamespace(
                    name="campaign-monitor",
                    transport="stdio",
                    command="/opt/monitor.py",
                    args=["--server-name", "campaign-monitor"],
                )
            ]
        )
        with patch.dict(os.environ, {"HARBOR_AGENT_TIMEOUT_SEC": "21600"}):
            command = MODULE._codex_config_with_long_lived_mcp(codex)

        assert command is not None
        assert "model_auto_compact_token_limit" not in command
        assert 'direct_only_tool_namespaces = ["mcp__campaign_monitor"]' in command
        assert "tool_timeout_sec = 21600" in command
        assert "startup_timeout_sec = 60" in command
        assert "required = true" in command
        parsed = tomllib.loads(shlex.split(command)[1])
        assert "model_auto_compact_token_limit" not in parsed
        assert "model_auto_compact_token_limit_scope" not in parsed
        assert parsed["features"]["code_mode"]["enabled"] is True
        assert parsed["features"]["code_mode"]["direct_only_tool_namespaces"] == [
            "mcp__campaign_monitor"
        ]
        assert parsed["tool_output_token_limit"] == 2048
        assert parsed["mcp_servers"]["campaign-monitor"]["command"] == (
            "/opt/monitor.py"
        )
        assert parsed["mcp_servers"]["campaign-monitor"]["args"] == [
            "--server-name",
            "campaign-monitor",
        ]
        assert parsed["mcp_servers"]["campaign-monitor"]["env_vars"] == list(
            MODULE._CAMPAIGN_MONITOR_ENV_VARS
        )
        for required_name in (
            "HARBOR_HOST_WORKSPACE",
            "HARBOR_SBATCH_SOCKET",
            "CHAPERONE_SLURM_PYTHON",
            "CHAPERONE_SLURM_ARRAY_PARALLELISM",
            "CHAPERONE_DEFAULT_PARTITION",
            "CHAPERONE_GPU_PARTITION",
            "CHAPERONE_GRES_MAP",
            "FOUNDRY_CHECKPOINT_DIRS",
            "HBPLUS_COPY",
            "CCD_MIRROR_PATH",
            "PDB_MIRROR_PATH",
        ):
            assert (
                required_name in parsed["mcp_servers"]["campaign-monitor"]["env_vars"]
            )
        exposed = set(parsed["mcp_servers"]["campaign-monitor"]["env_vars"])
        assert not {"CHAPERONE_EVALS_PRIVATE", "OPENAI_API_KEY", "SLURM_JWT"} & exposed
        assert parsed["mcp_servers"]["campaign-monitor"]["tool_timeout_sec"] == 21600

    def test_codex_mcp_namespace_matches_exposed_tool_name(self) -> None:
        assert (
            MODULE._codex_mcp_tool_namespace("campaign-monitor")
            == "mcp__campaign_monitor"
        )


class FileHandlerCloseTests(unittest.TestCase):
    def test_estale_during_close_is_ignored(self) -> None:
        with patch.object(
            MODULE,
            "_ORIGINAL_FILE_HANDLER_CLOSE",
            side_effect=OSError(errno.ESTALE, "Stale file handle"),
        ):
            MODULE._close_file_handler_tolerating_estale(SimpleNamespace())

    def test_other_close_errors_still_escape(self) -> None:
        with patch.object(
            MODULE,
            "_ORIGINAL_FILE_HANDLER_CLOSE",
            side_effect=OSError(errno.EIO, "Input/output error"),
        ):
            with self.assertRaisesRegex(OSError, "Input/output error"):
                MODULE._close_file_handler_tolerating_estale(SimpleNamespace())


if __name__ == "__main__":
    unittest.main()
