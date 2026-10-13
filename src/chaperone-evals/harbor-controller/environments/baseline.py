"""Strict clean-room environment for the baseline evaluation arm.

The baseline deliberately uses Harbor's stock Singularity lifecycle.  This
module only supplies policy that Harbor's generic environment cannot express:
pinning the audited base image, constraining mounts to trial-local plumbing,
and selecting a controller-created workspace as the working directory.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import override

from harbor.environments.singularity.singularity import SingularityEnvironment

from environments.slurm_broker import (
    SBATCH_GUARD,
    SOCKET_TARGET,
    SlurmBrokerProcess,
)
from environments.health_watchdog import ControllerHealthWatchdog


from site_config import get_site

_FORBIDDEN_ENV_NAMES = {
    "CHAPERONE_COMPUTE_BACKEND",
    "CHAPERONE_CONTAINER_BACKEND",
    "CHAPERONE_REPO",
    "CHAPERONE_SLURM_PYTHON",
    "PYTHONPATH",
    "QHOP",
}
_FORBIDDEN_KWARGS = {
    "apptainer_gpus",
    "apptainer_image",
    "baseline_image",
    "task_inputs",
    "workspace_seed",
}
# Real cluster paths (/projects/ml/pipelines, /net/software/containers/pipelines)
# and pre-rename images still carry the old project name, so a baseline value is
# project-specific if it contains either name.
_PROJECT_NAMES = ("chaperone", "pipelines")


def _names_project(value: str) -> bool:
    lowered = value.lower()
    return any(name in lowered for name in _PROJECT_NAMES)


_UNSAFE_APPTAINER_HOST_ENV = {
    "APPTAINER_BIND",
    "APPTAINER_BINDPATH",
    "APPTAINER_HOME",
    "APPTAINER_MOUNT",
    "SINGULARITY_BIND",
    "SINGULARITY_BINDPATH",
    "SINGULARITY_HOME",
    "SINGULARITY_MOUNT",
}

# This check runs inside the immutable image before Harbor installs its own
# control server and Codex into the per-trial writable tmpfs.  Basic operating
# system tools (including uv, Git, Node, and system Python) are allowed; a
# scientific environment, Chaperone installation, model asset, or prebuilt
# Python virtual environment is not.
_CLEAN_IMAGE_PROBE = r"""
set -eu
for path in \
  /opt/chaperone \
  /opt/pipelines \
  /opt/skills \
  /harbor/skills \
  /workspace/chaperone \
  /workspace/pipelines \
  /projects/ml/pipelines \
  /mnt/projects/ml/pipelines \
  /net/databases \
  /databases \
  /weights
do
  test ! -e "$path"
done
! command -v chaperone >/dev/null 2>&1
! command -v pipelines >/dev/null 2>&1
! command -v codex >/dev/null 2>&1
for command in \
  rfd3 rfdiffusion proteinmpnn ligandmpnn foundrympnn \
  alphafold alphafold3 boltz boltzgen chai_lab esmfold \
  rosetta_scripts pyrosetta colabfold_batch openfold mmseqs foldseek jackhmmer
do
  ! command -v "$command" >/dev/null 2>&1
done
for root in /opt /usr/local /models /checkpoints /weights; do
  test ! -e "$root" && continue
  test -z "$(find "$root" -xdev \
    \( -iname '*rfdiffusion*' -o -iname '*proteinmpnn*' \
       -o -iname '*ligandmpnn*' -o -iname '*foundrympnn*' \
       -o -iname '*alphafold*' -o -iname '*boltz*' -o -iname '*chai*' \
       -o -iname '*esmfold*' -o -iname '*rosetta*' \
       -o -iname '*colabfold*' -o -iname '*openfold*' \
       -o -iname '*mmseqs*' -o -iname '*foldseek*' \
       -o -name '*.a3m' -o -name '*.pt' -o -name '*.pth' -o -name '*.onnx' \) \
    -print -quit 2>/dev/null)"
done
test -z "$(find / -xdev -type f -name pyvenv.cfg -print -quit 2>/dev/null)"
test -z "$(find / -xdev -type d -name conda-meta -print -quit 2>/dev/null)"
test -z "$(find / -xdev -type f \( -name '*.ckpt' -o -name '*.safetensors' \) -print -quit 2>/dev/null)"
python3 - <<'PY'
for name in (
    "Bio", "alphafold", "boltz", "chai_lab", "esm", "jax", "numpy",
    "openfold", "pyrosetta", "rdkit", "scipy", "tensorflow", "torch",
):
    try:
        __import__(name)
    except ImportError:
        continue
    raise SystemExit(f"scientific Python package is preinstalled: {name}")
PY
"""


class BaselineSingularityEnvironment(SingularityEnvironment):
    """Harbor Singularity with a pinned clean image and minimal host plumbing.

    Harbor's stock command execution and container lifecycle are retained; thin
    start/stop wrappers manage only the host-side guarded Slurm broker. Harbor
    creates its control-plane dependencies in the disposable container overlay,
    installs the selected agent, and presents a fresh writable
    ``/workspace`` containing only the selected task's copied ``inputs/``.
    In particular, there is no host workspace seed, task setup script, skill,
    raw Slurm client/configuration, checkpoint, cache, scientific database,
    Chaperone source tree, or Chaperone virtualenv. The only compute commands
    are brokered sbatch/scancel/squeue clients, and workloads always re-enter
    this same clean image.
    """

    # Harbor's Singularity exec falls back to a hardcoded 600s HTTP timeout
    # whenever the caller passes timeout_sec=None:
    #
    #   _DEFAULT_HTTP_TIMEOUT = 600
    #   http_timeout = (timeout_sec + 10) if timeout_sec is not None else ...
    #
    # Its codex agent runs the whole session as one exec with no timeout_sec,
    # so every trial died at ~10min with "HTTP request timed out after 600
    # seconds", surfaced as AgentTimeoutError claiming the agent budget had
    # elapsed. Long agent commands are normal here — a single design sweep can
    # run for hours — so default to the agent budget instead of 10 minutes.
    DEFAULT_EXEC_TIMEOUT_SEC = 12 * 60 * 60

    @override
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ):
        if timeout_sec is None:
            timeout_sec = self._default_exec_timeout_sec
        return await super().exec(
            command, cwd=cwd, env=env, timeout_sec=timeout_sec, user=user
        )

    def __init__(self, *args, mounts=None, persistent_env=None, **kwargs) -> None:
        allowed_harbor_targets = {
            "/logs/agent",
            "/logs/verifier",
            "/logs/artifacts",
        }
        if mounts and any(
            mount.get("type") != "bind"
            or str(mount.get("target", "")) not in allowed_harbor_targets
            for mount in mounts
        ):
            raise ValueError(
                "Baseline trials do not permit caller-provided host bind mounts"
            )
        supplied_forbidden = sorted(
            name for name in _FORBIDDEN_KWARGS if kwargs.get(name) not in (None, "", 0)
        )
        if supplied_forbidden:
            raise ValueError(
                "Baseline trials do not accept harness resources: "
                + ", ".join(supplied_forbidden)
            )
        for name in _FORBIDDEN_KWARGS:
            kwargs.pop(name, None)

        configured_env = dict(persistent_env or {})
        forbidden_env = sorted(_FORBIDDEN_ENV_NAMES.intersection(configured_env))
        if forbidden_env:
            raise ValueError(
                "Baseline task configuration contains harness environment "
                "variables: " + ", ".join(forbidden_env)
            )
        if any(_names_project(value) for value in configured_env.values()):
            raise ValueError(
                "Baseline task environment must not reference Chaperone paths"
            )

        task_env_config = kwargs.get("task_env_config")
        if task_env_config is None and len(args) >= 5:
            task_env_config = args[4]
        if task_env_config is not None:
            if task_env_config.skills_dir:
                raise ValueError("Baseline trials must not configure skills")
            if task_env_config.mcp_servers:
                raise ValueError("Baseline trials must not configure MCP servers")
            resolved_env = {**task_env_config.env, **configured_env}
            forbidden_resolved_env = sorted(
                _FORBIDDEN_ENV_NAMES.intersection(resolved_env)
            )
            if forbidden_resolved_env or any(
                _names_project(value) for value in resolved_env.values()
            ):
                detail = ", ".join(forbidden_resolved_env) or "Chaperone path value"
                raise ValueError(
                    "Resolved baseline environment contains harness configuration: "
                    + detail
                )

        resolved_timeout = configured_env.get("HARBOR_AGENT_TIMEOUT_SEC")
        if resolved_timeout is None and task_env_config is not None:
            resolved_timeout = task_env_config.env.get("HARBOR_AGENT_TIMEOUT_SEC")
        try:
            self._default_exec_timeout_sec = int(
                resolved_timeout or self.DEFAULT_EXEC_TIMEOUT_SEC
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("HARBOR_AGENT_TIMEOUT_SEC must be an integer") from exc
        if self._default_exec_timeout_sec < 1:
            raise ValueError("HARBOR_AGENT_TIMEOUT_SEC must be positive")

        environment_dir = kwargs.get("environment_dir")
        if environment_dir is None and args:
            environment_dir = args[0]
        if environment_dir is not None:
            self._validate_environment_inputs(Path(environment_dir))

        trial_paths = kwargs.get("trial_paths")
        if trial_paths is None and len(args) >= 4:
            trial_paths = args[3]
        if trial_paths is None:
            raise ValueError("Baseline environment requires Harbor trial paths")
        trial_root = Path(trial_paths.trial_dir).resolve()
        harbor_mounts = self._validate_harbor_mounts(mounts or [], trial_root)
        workspace = Path(trial_paths.trial_dir) / "baseline-workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        if any(workspace.iterdir()):
            raise ValueError(
                f"Baseline workspace must be empty at startup: {workspace}"
            )

        runtime = trial_root / ".baseline-runtime"
        runtime.mkdir(mode=0o700, exist_ok=True)
        resolver = runtime / "resolv.conf"
        resolver.write_text(self._resolver_config().read_text())
        resolver.chmod(0o600)
        payload_dir = runtime / "payloads"
        payload_dir.mkdir(mode=0o700, exist_ok=True)
        client = runtime / "slurm-client.py"
        shutil.copy2(SBATCH_GUARD, client)
        client.chmod(0o500)
        runner = runtime / "slurm-worker"
        self._prepare_generic_worker(
            runner=runner,
            workspace=workspace,
            runtime=runtime,
            image=Path(get_site()["baseline_image"]).resolve(),
        )
        self._slurm_broker = SlurmBrokerProcess(
            workspace=workspace,
            runtime_dir=trial_root / ".runtime" / "slurm-broker",
            policy="generic",
            workspace_alias=Path("/workspace"),
            generic_runner=runner,
            payload_dir=payload_dir,
            client_path=client,
        )
        self._health_watchdog: ControllerHealthWatchdog | None = None
        # Harbor stock Singularity does not pass --cleanenv itself.  Each
        # baseline launcher process is arm-specific, so make clean inheritance
        # a process-wide invariant before Harbor starts its server.
        self._scrub_apptainer_host_env()
        os.environ["APPTAINER_CLEANENV"] = "1"
        os.environ["SINGULARITY_CLEANENV"] = "1"
        configured_env.update(
            {
                "HARBOR_MAX_TOTAL_GPUS": str(get_site()["max_total_gpus"]),
                "HARBOR_MAX_REQUESTED_GPUS_PER_SUBMISSION": str(
                    get_site()["max_requested_gpus_per_submission"]
                ),
                "HARBOR_ALLOWED_GPU_TYPE": get_site()["gpu_type"],
                "HARBOR_SBATCH_SOCKET": SOCKET_TARGET,
                "HARBOR_SLURM_POLICY": "generic-contained",
            }
        )
        super().__init__(
            *args,
            mounts=[
                *harbor_mounts,
                {
                    "type": "bind",
                    "source": str(workspace),
                    "target": "/workspace",
                },
                {
                    "type": "bind",
                    "source": str(resolver),
                    "target": "/etc/resolv.conf",
                },
                *self._slurm_broker.mounts(),
            ],
            persistent_env=configured_env,
            singularity_no_mount="home,tmp,bind-paths",
            **kwargs,
        )

    @override
    async def start(self, force_build: bool) -> None:
        await super().start(force_build)
        try:
            # A stable, pre-existing NFS directory avoids cross-node negative
            # dentry races while all actual log opens remain contained.
            (self._slurm_broker.workspace / "logs").mkdir(exist_ok=True)
            self._slurm_broker.start()

            async def probe() -> None:
                result = await self.exec(
                    "set -e; test -x /bin/bash; test -x /bin/sh; "
                    "test -x /usr/bin/python3; "
                    "test -S /run/harbor-slurm/sbatch.sock; "
                    "mkdir -p .harbor; probe=.harbor/.controller-health-probe; "
                    'printf healthy >"$probe"; '
                    'test "$(cat "$probe")" = healthy; rm -f "$probe"',
                    timeout_sec=45,
                )
                if result.return_code != 0:
                    raise RuntimeError(result.stdout or "controller probe failed")

            async def fail(_error: BaseException) -> None:
                self._slurm_broker.stop()

            self._health_watchdog = ControllerHealthWatchdog(
                report_path=self._slurm_broker.workspace
                / ".harbor"
                / "controller_health.json",
                probe=probe,
                on_failure=fail,
            )
            self._health_watchdog.start()
        except BaseException:
            await super().stop(delete=True)
            raise

    @override
    async def stop(self, delete: bool) -> None:
        try:
            if self._health_watchdog is not None:
                await self._health_watchdog.stop()
                self._health_watchdog = None
            self._slurm_broker.stop()
        finally:
            try:
                await super().stop(delete)
            finally:
                # Harbor retains completed asyncio subprocess/task objects on
                # the environment. If their finalizers run only after
                # asyncio.run() has closed its loop, CPython emits an ignored
                # ``RuntimeError: Event loop is closed`` during otherwise
                # successful shutdown. Drop the references while this loop is
                # still alive and give queued pipe-close callbacks one turn.
                self._stream_task = None
                self._memory_watchdog_task = None
                self._server_process = None
                await asyncio.sleep(0)

    @override
    def _validate_definition(self) -> None:
        super()._validate_definition()
        forbidden_env = sorted(_FORBIDDEN_ENV_NAMES.intersection(self._persistent_env))
        if forbidden_env or any(
            _names_project(value) for value in self._persistent_env.values()
        ):
            detail = ", ".join(forbidden_env) or "Chaperone path value"
            raise ValueError(
                "Resolved baseline environment contains harness configuration: "
                + detail
            )
        configured = Path(self._docker_image or "").expanduser().resolve()
        expected = Path(get_site()["baseline_image"]).resolve()
        if configured != expected:
            raise ValueError(
                "Baseline trials must use the audited clean image "
                f"{expected}, not {configured}"
            )
        completed = subprocess.run(
            [
                "apptainer",
                "exec",
                "--no-mount",
                "home,tmp,bind-paths",
                "--cleanenv",
                "--containall",
                str(configured),
                "bash",
                "-lc",
                _CLEAN_IMAGE_PROBE,
            ],
            text=True,
            capture_output=True,
            timeout=60,
        )
        if completed.returncode:
            detail = (completed.stdout + completed.stderr).strip()
            raise ValueError(
                "The pinned baseline image failed its clean-room audit"
                + (f": {detail}" if detail else "")
            )

    @override
    def _resolve_workdir(self) -> str:
        return "/workspace"

    @staticmethod
    def _prepare_generic_worker(
        *, runner: Path, workspace: Path, runtime: Path, image: Path
    ) -> None:
        """Create the only compute-node entry point approved by the broker."""
        script = "\n".join(
            (
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"workspace={shlex.quote(str(workspace))}",
                f"runtime={shlex.quote(str(runtime))}",
                f"image={shlex.quote(str(image))}",
                "payload=${1:?missing snapshotted payload}",
                "output=${2:?missing contained stdout path}",
                "error=${3:?missing contained stderr path}",
                'case "$payload" in "$runtime"/payloads/payload-*.sh) ;;',
                "  *) echo 'invalid guarded payload path' >&2; exit 2 ;;",
                "esac",
                'render_path() { local value=$1; value=${value//%%/__HARBOR_PERCENT__}; value=${value//%A/${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-0}}}; value=${value//%a/${SLURM_ARRAY_TASK_ID:-0}}; value=${value//%j/${SLURM_JOB_ID:-0}}; value=${value//%x/${SLURM_JOB_NAME:-job}}; value=${value//__HARBOR_PERCENT__/%}; printf %s "$value"; }',
                'output=$(render_path "$output")',
                'error=$(render_path "$error")',
                'case "$output" in /workspace/*) ;; *) echo "invalid stdout path" >&2; exit 2 ;; esac',
                'case "$error" in /workspace/*) ;; *) echo "invalid stderr path" >&2; exit 2 ;; esac',
                "gpu_args=()",
                'if [[ -n "${CUDA_VISIBLE_DEVICES:-}" || -n "${SLURM_JOB_GPUS:-}" ]]; then gpu_args=(--nv); fi',
                'worker_tmp=${SLURM_TMPDIR:-"$workspace/.tmp/slurm-${SLURM_JOB_ID:-manual}-${SLURM_ARRAY_TASK_ID:-0}"}',
                'mkdir -p "$worker_tmp" "$workspace/.home" "$workspace/.tmp"',
                "env_args=()",
                "for name in CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy; do",
                '  if [[ -v "$name" ]]; then env_args+=(--env "$name=${!name}"); fi',
                "done",
                'while IFS= read -r name; do env_args+=(--env "$name=${!name}"); done < <(compgen -A variable SLURM_)',
                'payload_name=$(basename "$payload")',
                'env_args+=(--env "HARBOR_LOG_OUTPUT=$output" --env "HARBOR_LOG_ERROR=$error")',
                'exec apptainer exec "${gpu_args[@]}" "${env_args[@]}" \\',
                "  --cleanenv --containall --no-mount home --no-mount tmp --no-mount bind-paths \\",
                "  --writable-tmpfs --pid --pwd /workspace \\",
                '  -B "$workspace:/workspace" \\',
                '  -B "$runtime:/run/harbor-baseline:ro" \\',
                '  -B "$worker_tmp:/tmp" \\',
                '  -B "$runtime/resolv.conf:/etc/resolv.conf:ro" \\',
                '  "$image" bash -lc \'export HOME=/workspace/.home TMPDIR=/tmp XDG_CACHE_HOME=/workspace/.home/.cache XDG_CONFIG_HOME=/workspace/.home/.config; cd /workspace; mkdir -p "$(dirname "$HARBOR_LOG_OUTPUT")" "$(dirname "$HARBOR_LOG_ERROR")"; exec bash "/run/harbor-baseline/payloads/$1" >"$HARBOR_LOG_OUTPUT" 2>"$HARBOR_LOG_ERROR"\' baseline "$payload_name"',
                "",
            )
        )
        runner.write_text(script)
        runner.chmod(0o500)

    @staticmethod
    def _validate_harbor_mounts(
        mounts: list[dict[str, object]], trial_root: Path
    ) -> list[dict[str, object]]:
        """Allow only Harbor's three bookkeeping mounts beneath this trial."""
        allowed_targets = {"/logs/agent", "/logs/verifier", "/logs/artifacts"}
        validated: list[dict[str, object]] = []
        seen: set[str] = set()
        for mount in mounts:
            target = str(mount.get("target", ""))
            source_text = str(mount.get("source", ""))
            try:
                source = Path(source_text).resolve(strict=True)
            except OSError as exc:
                raise ValueError(
                    f"Baseline Harbor mount source is unavailable: {source_text}"
                ) from exc
            if (
                mount.get("type") != "bind"
                or target not in allowed_targets
                or target in seen
                or not source.is_relative_to(trial_root)
            ):
                raise ValueError(
                    "Baseline trials do not permit caller-provided host bind "
                    f"mount: {source_text}:{target}"
                )
            seen.add(target)
            validated.append(dict(mount))
        if seen != allowed_targets:
            missing = ", ".join(sorted(allowed_targets - seen))
            raise ValueError(
                "Baseline requires exactly Harbor's trial-local bookkeeping "
                f"mounts; missing: {missing or 'none'}"
            )
        return validated

    @staticmethod
    def _validate_environment_inputs(environment_dir: Path) -> None:
        """Allow exactly the selected task's inert ``inputs/`` payload."""
        children = list(environment_dir.iterdir()) if environment_dir.is_dir() else []
        if any(child.name != "inputs" for child in children):
            raise ValueError(
                "Baseline environment may contain only the selected task inputs"
            )
        inputs = environment_dir / "inputs"
        if not inputs.exists():
            return
        if not inputs.is_dir() or inputs.is_symlink():
            raise ValueError("Baseline task inputs must be a real directory")
        forbidden_names = {
            ".git",
            "SKILL.md",
            "examples",
            "chaperone",
            "pipelines",
            "pyvenv.cfg",
            "weights",
        }
        forbidden_suffixes = {".ckpt", ".onnx", ".pt", ".pth", ".safetensors", ".sif"}
        for path in inputs.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"Baseline task inputs contain a symlink: {path}")
            if (
                path.name in forbidden_names
                or path.suffix.lower() in forbidden_suffixes
            ):
                raise ValueError(
                    f"Baseline task inputs contain a forbidden resource: {path}"
                )

    @staticmethod
    def _resolver_config() -> Path:
        """Return generic DNS plumbing needed for public package installs."""
        for candidate in (
            Path("/run/systemd/resolve/resolv.conf"),
            Path("/etc/resolv.conf"),
        ):
            try:
                if candidate.is_file() and candidate.stat().st_size:
                    return candidate.resolve(strict=True)
            except OSError:
                continue
        raise FileNotFoundError("No usable DNS resolver configuration was found")

    @staticmethod
    def _scrub_apptainer_host_env() -> None:
        """Remove ambient bind and container-environment injection controls."""
        for name in list(os.environ):
            if (
                name in _UNSAFE_APPTAINER_HOST_ENV
                or name.startswith("APPTAINERENV_")
                or name.startswith("SINGULARITYENV_")
            ):
                os.environ.pop(name, None)
