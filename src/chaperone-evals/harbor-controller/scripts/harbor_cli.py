#!/usr/bin/env python3
"""Run Harbor with readable, deterministic trial directory names.

Harbor normally appends a random seven-character suffix to every trial name.
The controller already gives each job a unique timestamp (and Slurm job ID), so
the first trial can use the task directory name directly.  Additional attempts
within the same job receive a small numeric suffix to avoid collisions.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import re
import shlex
from collections import defaultdict

from harbor.agents.installed.codex import Codex
from harbor.models.trial.config import TrialConfig


_TRIAL_NAME_COUNTS: defaultdict[str, int] = defaultdict(int)
_ORIGINAL_FILE_HANDLER_CLOSE = logging.FileHandler.close

# Codex intentionally gives stdio MCP servers a filtered environment.  The
# campaign monitor is part of the trusted harness and needs the same
# non-secret runtime contract as the agent shell so Pipelines can submit
# guarded Slurm jobs and nested scientific containers without campaign code
# hard-coding trial-specific paths.
_CAMPAIGN_MONITOR_ENV_VARS = (
    "PATH",
    "HOME",
    "TMPDIR",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "MPLCONFIGDIR",
    "PYTHONPATH",
    "QHOP",
    "HARBOR_HOST_WORKSPACE",
    "HARBOR_SBATCH_SOCKET",
    "PIPELINES_SLURM_PYTHON",
    "PIPELINES_COMPUTE_BACKEND",
    "PIPELINES_CONTAINER_BACKEND",
    "PIPELINES_CONTAINER_BIND_RUNDIR",
    "PIPELINES_CONTAINER_BIND_WORKSPACE",
    "PIPELINES_SLURM_ARRAY_PARALLELISM",
    "PIPELINES_CONTAINER_DIRS",
    "HF_HUB_CACHE",
    "BOLTZ_CACHE_DIR",
    "AF3_JAX_CACHE_DIR",
    "ALPHAFOLD3_WEIGHTS_DIR",
    "OPENFOLD3_CACHE_DIR",
    "HBPLUS_COPY",
    "FOLDSEEK_EXEC",
    "FOUNDRY_CHECKPOINT_DIRS",
    "CCD_MIRROR_PATH",
    "PDB_MIRROR_PATH",
)


def _readable_trial_name(config: TrialConfig) -> str:
    task_name = config.task.get_task_id().get_name().split("/")[-1]
    base_name = task_name[:64].rstrip("_-") or "trial"
    _TRIAL_NAME_COUNTS[base_name] += 1
    attempt = _TRIAL_NAME_COUNTS[base_name]
    return base_name if attempt == 1 else f"{base_name}__{attempt}"


def _close_file_handler_tolerating_estale(self: logging.FileHandler) -> None:
    """Keep a transient shared-filesystem close failure from voiding a run.

    Harbor closes trial and job log handlers after it has recovered the trial
    artifacts. On the shared run filesystem, ``stream.close()`` can raise
    ``ESTALE`` even though all output has already been written. Letting that
    teardown-only error escape changes an otherwise completed Harbor job into
    a failed controller allocation. Other close errors remain fatal.
    """
    try:
        _ORIGINAL_FILE_HANDLER_CLOSE(self)
    except OSError as exc:
        if exc.errno != errno.ESTALE:
            raise


def _codex_mcp_tool_namespace(server_name: str) -> str:
    """Return the tool namespace Codex derives from an MCP server name."""
    normalized = re.sub(r"[^A-Za-z0-9_]+", "_", server_name).strip("_")
    if not normalized:
        raise ValueError("MCP server name must contain an alphanumeric character")
    return f"mcp__{normalized}"


def _codex_config_with_long_lived_mcp(self: Codex) -> str | None:
    """Expose the campaign monitor directly with a run-long tool timeout."""
    timeout = max(600, int(os.environ.get("HARBOR_AGENT_TIMEOUT_SEC", "43200")))
    direct_namespaces = [
        _codex_mcp_tool_namespace(server.name)
        for server in self.mcp_servers
        if server.name == "campaign-monitor"
    ]
    # Codex measures retained tool results in tokens. 2,048 tokens is the
    # closest stable configuration-level approximation to an 8 KiB text cap.
    lines = ["tool_output_token_limit = 2048", ""]
    if direct_namespaces:
        rendered = ", ".join(f'"{name}"' for name in direct_namespaces)
        lines.extend(
            [
                "[features.code_mode]",
                "enabled = true",
                f"direct_only_tool_namespaces = [{rendered}]",
                "",
            ]
        )
    for server in self.mcp_servers:
        lines.append(f"[mcp_servers.{server.name}]")
        if server.transport == "stdio":
            lines.append(f"command = {json.dumps(server.command)}")
            rendered_args = ", ".join(json.dumps(arg) for arg in server.args)
            lines.append(f"args = [{rendered_args}]")
            if server.name == "campaign-monitor":
                rendered_env_vars = ", ".join(
                    json.dumps(name) for name in _CAMPAIGN_MONITOR_ENV_VARS
                )
                lines.append(f"env_vars = [{rendered_env_vars}]")
        else:
            lines.append(f'url = "{server.url}"')
        lines.extend(
            [
                "startup_timeout_sec = 60",
                f"tool_timeout_sec = {timeout}",
                "required = true",
                "",
            ]
        )
    escaped_config = shlex.quote("\n".join(lines))
    return f'echo {escaped_config} >> "$CODEX_HOME/config.toml"'


def main() -> None:
    # Harbor 0.16 does not expose a job-level trial-name template. Patch only
    # this process before the CLI constructs TrialConfig instances.
    TrialConfig.generate_trial_name = _readable_trial_name  # type: ignore[method-assign]
    logging.FileHandler.close = (  # type: ignore[method-assign]
        _close_file_handler_tolerating_estale
    )
    Codex._build_register_mcp_servers_command = (  # type: ignore[method-assign]
        _codex_config_with_long_lived_mcp
    )

    from harbor.cli.main import app

    app()


if __name__ == "__main__":
    main()
