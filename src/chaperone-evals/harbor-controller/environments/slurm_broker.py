"""Reusable host-side lifecycle for the guarded Slurm broker."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


from site_config import get_site

CONTROLLER_ROOT = Path(__file__).resolve().parents[1]
SBATCH_GUARD = CONTROLLER_ROOT / "scripts" / "slurm_gpu_guard.py"
SOCKET_TARGET = "/run/harbor-slurm/sbatch.sock"
_SOCKET_PATH_MAX_BYTES = 107

SLURM_RUNTIME_PATHS = (
    "/etc/slurm",
    "/run/munge/munge.socket.2",
    "/usr/lib/slurm",
    "/usr/lib/x86_64-linux-gnu/libmunge.so.2",
    "/usr/bin/sacct",
    "/usr/bin/sinfo",
    "/usr/bin/squeue",
    "/usr/bin/sstat",
)


class SlurmBrokerProcess:
    """Start, monitor, and stop one trial-scoped guarded Slurm broker."""

    def __init__(
        self,
        *,
        workspace: Path,
        runtime_dir: Path,
        policy: str,
        slurm_python: Path | None = None,
        generic_runner: Path | None = None,
        payload_dir: Path | None = None,
        workspace_alias: Path | None = None,
        client_path: Path | None = None,
        exclude_nodes: str | None = None,
    ) -> None:
        self.workspace = workspace.resolve()
        self.runtime_dir = runtime_dir.resolve()
        self.policy = policy
        self.slurm_python = slurm_python.resolve() if slurm_python else None
        self.generic_runner = generic_runner.resolve() if generic_runner else None
        self.payload_dir = payload_dir.resolve() if payload_dir else None
        self.workspace_alias = workspace_alias
        self.client_path = (client_path or SBATCH_GUARD).resolve()
        self.exclude_nodes = (
            exclude_nodes
            if exclude_nodes is not None
            else os.environ.get("HARBOR_EXCLUDED_NODES", "")
        ).strip()
        self.socket_dir = self._make_socket_dir()
        self.log_path = self.runtime_dir / "slurm-broker.log"
        self.submission_registry = self.runtime_dir / "submissions.jsonl"
        self.process: subprocess.Popen[bytes] | None = None
        self.log_handle = None

    @staticmethod
    def _make_socket_dir() -> Path:
        directory = Path(tempfile.mkdtemp(prefix="harbor-sb-", dir="/tmp"))
        directory.chmod(0o700)
        if len(os.fsencode(directory / "sbatch.sock")) > _SOCKET_PATH_MAX_BYTES:
            shutil.rmtree(directory, ignore_errors=True)
            raise RuntimeError("Guarded Slurm socket path exceeds the Unix limit")
        return directory

    def mounts(self) -> list[dict[str, object]]:
        brokered_commands = (
            ("sbatch", "scancel", "squeue")
            if self.policy == "generic"
            else ("sbatch", "scancel", "scontrol", "srun")
        )
        mounts = [
            {
                "type": "bind",
                "source": str(self.socket_dir),
                "target": "/run/harbor-slurm",
            },
            *[
                {
                    "type": "bind",
                    "source": str(self.client_path),
                    "target": f"/usr/bin/{command}",
                    "read_only": True,
                }
                for command in brokered_commands
            ],
        ]
        if self.policy != "generic":
            mounts.extend(
                {
                    "type": "bind",
                    "source": path,
                    "target": path,
                    "read_only": True,
                }
                for path in SLURM_RUNTIME_PATHS
                if Path(path).exists()
            )
        return mounts

    def start(self) -> None:
        if self.process is not None:
            return
        real_commands = {
            name: shutil.which(name)
            for name in ("sbatch", "scancel", "scontrol", "srun", "squeue")
        }
        missing = sorted(name for name, value in real_commands.items() if value is None)
        if missing:
            raise FileNotFoundError(
                "Host Slurm commands unavailable: " + ", ".join(missing)
            )
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        # The presence of an empty registry distinguishes a real zero-job
        # trial from missing controller bookkeeping during post-run audit.
        self.submission_registry.touch(exist_ok=True)
        self.submission_registry.chmod(0o600)
        site = get_site()
        command = [
            sys.executable,
            str(self.client_path),
            "serve",
            "--socket",
            str(self.socket_dir / "sbatch.sock"),
            "--real-sbatch",
            str(real_commands["sbatch"]),
            "--real-scancel",
            str(real_commands["scancel"]),
            "--real-scontrol",
            str(real_commands["scontrol"]),
            "--real-srun",
            str(real_commands["srun"]),
            "--real-squeue",
            str(real_commands["squeue"]),
            "--workspace",
            str(self.workspace),
            "--submission-registry",
            str(self.submission_registry),
            "--policy",
            self.policy,
            "--max-total-gpus",
            str(site["max_total_gpus"]),
            "--max-requested-gpus-per-submission",
            str(site["max_requested_gpus_per_submission"]),
            "--allowed-gpu-type",
            site["gpu_type"],
        ]
        for key in (
            "cpu_partition",
            "gpu_partition",
            "gpu_constraint",
            "account",
            "qos",
        ):
            command.append("--" + key.replace("_", "-") + "=" + site[key])
        if self.exclude_nodes:
            command.extend(("--exclude-nodes", self.exclude_nodes))
        if self.slurm_python is not None:
            command.extend(("--slurm-python", str(self.slurm_python)))
        if self.workspace_alias is not None:
            command.extend(("--workspace-alias", str(self.workspace_alias)))
        if self.generic_runner is not None:
            command.extend(("--generic-runner", str(self.generic_runner)))
        if self.payload_dir is not None:
            command.extend(("--payload-dir", str(self.payload_dir)))
        self.log_handle = self.log_path.open("ab", buffering=0)
        self.process = subprocess.Popen(
            command,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + 5
        socket_path = self.socket_dir / "sbatch.sock"
        while time.monotonic() < deadline:
            if socket_path.is_socket():
                return
            if self.process.poll() is not None:
                break
            time.sleep(0.05)
        self.stop()
        detail = (
            self.log_path.read_text(errors="replace") if self.log_path.exists() else ""
        )
        raise RuntimeError("Guarded Slurm broker failed to start: " + detail.strip())

    def stop(self) -> None:
        process = self.process
        self.process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None
        shutil.rmtree(self.socket_dir, ignore_errors=True)
