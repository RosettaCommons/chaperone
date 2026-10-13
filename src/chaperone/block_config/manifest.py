"""Standalone Manifest dataclass + executor for compiled-block subprocesses.

Lives in its own module (separate from compiled.py) so that lightweight
processes can import ``Manifest`` without dragging in the heavy
``CompiledBlockFunction`` import chain (which transitively pulls in
``chaperone.data.PipelineDataHandler`` → polars, ``BlockFunction`` →
Pydantic block registry, etc.).

In particular, ``runner_wait`` imports only from this module so the
long-running supervisor process during the apptainer wait can stay
around ~50 MiB instead of ~600 MiB.

For backwards compatibility ``compiled.Manifest`` is re-exported from
here, so existing imports of ``chaperone.block_config.compiled.Manifest``
continue to work unchanged.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import sys
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

# Operators a block may place in ``command`` as their own element to request
# shell semantics (see rfdiffusion2 and util_blocks/testing). Everything else is
# an argument *value* and is quoted, even when it happens to contain one of
# these characters — e.g. MPNN's ``--symmetry_residues A1,B71|A2,B72`` — which
# would otherwise be word-split by the shell into "command not found" noise and
# a silently truncated command line.
SHELL_OPERATORS = frozenset({"&&", "||", "|", ">", ">>", "<", "<<", ";"})


@dataclass
class Manifest:
    name: str
    command: List[str]
    cwd: str
    command_str: str = None

    def __post_init__(self):
        self.command = list(filter(None, self.command))  # Remove None entries
        self.command = [str(c) for c in self.command]  # Ensure all entries are strings
        if self.command_str is None:
            self.command_str = " ".join(
                part if part in SHELL_OPERATORS else shlex.quote(part)
                for part in self.command
            )

    @classmethod
    def read(cls, path: str | Path) -> "Manifest":
        path = Path(path)
        if path.is_dir():
            path = Path(path) / "manifest.json"
        return cls(**json.load(open(path)))

    def write(self, dir: str | Path) -> Path:
        path = Path(dir) / "manifest.json"
        # exist_ok re-checks with a stat after EEXIST; on NFS that can come back stale and
        # raise, so concurrent jobs creating one shared directory tolerate FileExistsError.
        with suppress(FileExistsError):
            path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)
        return path

    # Environment variables to remove to prevent Jupyter/notebook pollution
    _CLEAN_ENV_VARS = [
        "PYTHONPATH",
        "JUPYTER_PATH",
        "JPY_PARENT_PID",
        "PYDEVD_USE_FRAME_EVAL",
        "MPLBACKEND",
    ]

    @classmethod
    def exec(
        cls, path: str | Path, log_path: str | Path | None = None
    ) -> subprocess.CompletedProcess:
        """Execute manifest with real-time output streaming and logging.

        Args:
            path: Path to manifest file
            log_path: Optional path to write combined stdout/stderr log
        """
        manifest = cls.read(path)

        # Prepare clean environment (remove Jupyter/notebook pollution)
        env = os.environ.copy()
        for var in cls._CLEAN_ENV_VARS:
            env.pop(var, None)
        for var in list(env):
            if var.startswith(("APPTAINER_", "SINGULARITY_")):
                env.pop(var, None)

        # Ensure TMPDIR points to a writable directory.  SLURM sets TMPDIR
        # to a job-specific scratch path (e.g. /scratch/user/jobid/) that
        # does not exist when re-running locally via --local.  nvcc and
        # torch JIT both write intermediate files to TMPDIR, so a stale
        # value causes "Could not open output file" fatals.
        tmpdir = env.get("TMPDIR", "")
        if tmpdir and not os.path.isdir(tmpdir):
            logger.debug(
                "TMPDIR=%s does not exist; resetting to /tmp for subprocess", tmpdir
            )
            env["TMPDIR"] = "/tmp"

        # Open log file if specified
        log_file = None
        if log_path:
            with suppress(FileExistsError):
                Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            log_file = open(log_path, "w")

        # Use command_str with shell=True if it contains shell operators. This
        # is safe now that command_str quotes every non-operator argument: an
        # argument whose *value* contains one of these characters still routes
        # through the shell, but reaches the program intact.
        use_shell = manifest.command_str and any(
            op in manifest.command_str for op in ["&&", "||", "|", ">", "<", ";"]
        )
        cmd = manifest.command_str if use_shell else manifest.command

        try:
            # Stream output in real-time
            process = subprocess.Popen(
                cmd,
                cwd=manifest.cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                text=True,
                bufsize=1,  # Line buffered
                shell=use_shell,
                env=env,
            )

            stdout_lines = []

            # Use readline() instead of iterator — the iterator uses an internal
            # C-level buffer that blocks and swallows KeyboardInterrupt.
            #
            # Live output goes to plain sys.stdout (not rich.console) so this
            # module stays standalone-importable without dragging in
            # `chaperone.common` → `chaperone/__init__.py` → polars/the block
            # registry. The log_file already captures the same content
            # verbatim if a richer record is needed.
            try:
                while True:
                    line = process.stdout.readline()
                    if not line:
                        break

                    if log_file:
                        log_file.write(line)
                        log_file.flush()

                    sys.stdout.write(line)
                    sys.stdout.flush()
                    stdout_lines.append(line)
            except KeyboardInterrupt:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise

            # Wait for process to complete
            returncode = process.wait()

            # Create CompletedProcess-like result
            result = subprocess.CompletedProcess(
                args=cmd,
                returncode=returncode,
                stdout="".join(stdout_lines),
                stderr="",  # Already merged into stdout
            )

            return result

        finally:
            if log_file:
                log_file.close()
