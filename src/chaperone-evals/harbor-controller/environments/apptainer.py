"""Contained Harbor environment backed by a prebuilt Apptainer image."""

from __future__ import annotations

import asyncio
import grp
import os
import pwd
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import override

from harbor.environments.base import ExecResult, OutputStream
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.environments.singularity.singularity import SingularityEnvironment
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

from environments.slurm_broker import (
    SlurmBrokerProcess,
)
from environments.health_watchdog import ControllerHealthWatchdog
from scripts.snapshot_harness import validate_snapshot
from site_config import get_site, ENV_NAMES


# Runtime dependencies and scientific assets explicitly assigned by the task.
# Missing paths are skipped so the controller remains portable.


def _runtime_bind_pairs() -> list[tuple[str, str]]:
    """(source, target) for each runtime path, expanding bare same-path entries."""
    return [
        tuple(entry) if isinstance(entry, (tuple, list)) else (entry, entry)
        for entry in get_site()["runtime_binds"]
    ]


def _optional_path_exists(path: Path) -> bool:
    """Best-effort probe for optional capability mounts.

    Some institutional automounts raise errors such as EINVAL during lookup
    instead of behaving like a missing path. Optional scientific capabilities
    must not prevent unrelated scorers from constructing their worker wrapper.
    Required paths are deliberately validated separately and still fail loud.
    """
    try:
        return path.exists()
    except OSError:
        return False


_CONTROLLER_ROOT = Path(__file__).resolve().parents[1]
_SBATCH_GUARD = _CONTROLLER_ROOT / "scripts" / "slurm_gpu_guard.py"
_UNIX_SOCKET_PATH_MAX_BYTES = 107
_BROKERED_SLURM_PATHS = {
    "/usr/bin/scancel",
    "/usr/bin/scontrol",
    "/usr/bin/srun",
}

_HOST_ENV_PASSTHROUGH = (
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "POLARS_MAX_THREADS",
    "RAYON_NUM_THREADS",
    "TOKIO_WORKER_THREADS",
)

# Dynamic scientific-tool locations assigned by the controller. These values
# must be inherited from the submitting process rather than given cluster-wide
# defaults, and are the only additional host variables admitted through the
# worker's ``apptainer exec --cleanenv`` boundary.
_ALLOWLISTED_WORKER_ENV_PASSTHROUGH = ("USALIGN_BIN",)


class ApptainerEnvironment(SingularityEnvironment):
    """Use a prebuilt SIF with a private workspace and optional ``--nv``.

    Commands run directly with ``apptainer exec`` instead of Harbor's stock
    in-container HTTP helper. This avoids installing server dependencies into
    the immutable scientific image and keeps every persistent write inside
    explicitly mounted per-trial directories.
    """

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        *args,
        mounts=None,
        workspace_seed: Path | str | None = None,
        task_inputs: Path | str | None = None,
        apptainer_gpus: int | str = 0,
        apptainer_image: Path | str | None = None,
        **kwargs,
    ) -> None:
        persistent_env = dict(kwargs.pop("persistent_env", {}) or {})
        self._host_workspace: Path | None = None
        self._apptainer_image = (
            str(Path(apptainer_image).expanduser().resolve())
            if apptainer_image
            else None
        )
        self._slurm_broker: SlurmBrokerProcess | None = None
        self._health_watchdog: ControllerHealthWatchdog | None = None
        self._slurm_python: Path | None = None
        self._slurm_worker_image = Path(
            get_site()["worker_image"]
            or apptainer_image
            or task_env_config.docker_image
            or ""
        ).resolve()
        try:
            gpu_count = int(apptainer_gpus)
        except (TypeError, ValueError) as exc:
            raise ValueError("apptainer_gpus must be a non-negative integer") from exc
        if gpu_count < 0:
            raise ValueError("apptainer_gpus must be a non-negative integer")

        self._gpu_count = gpu_count
        self._gpu_enabled = gpu_count > 0 and "__verifier__" not in session_id
        resolved_mounts = list(mounts or [])

        # ``--containall`` gives the container an empty /etc/resolv.conf on
        # systemd-resolved hosts.  Mount the upstream resolver configuration
        # explicitly so network-enabled agents can resolve API endpoints while
        # retaining the restricted bind-mount allowlist.
        if not any(
            mount.get("type") == "bind" and mount.get("target") == "/etc/resolv.conf"
            for mount in resolved_mounts
        ):
            resolved_mounts.append(
                {
                    "type": "bind",
                    "source": str(self._find_resolver_config()),
                    "target": "/etc/resolv.conf",
                    "read_only": True,
                }
            )

        if "__verifier__" not in session_id:
            if workspace_seed is None or task_inputs is None:
                raise ValueError(
                    "workspace_seed and task_inputs are required for agent trials"
                )
            workspace = trial_paths.trial_dir / "workspace"
            self._host_workspace = workspace.resolve()
            runtime_dir = trial_paths.trial_dir / ".runtime"
            self._prepare_workspace(
                workspace,
                Path(workspace_seed).resolve(),
                Path(task_inputs).resolve(),
            )
            image_cache_dir = workspace / ".harbor" / "singularity-cache"
            image_cache_dir.mkdir(parents=True, exist_ok=True)
            kwargs.setdefault("singularity_image_cache_dir", image_cache_dir)
            self._prepare_runtime_identity(runtime_dir, self._host_workspace)
            # Scientific assets are mounted read-only. Compilation caches
            # must instead live in the writable trial, even when the selected
            # site profile points AF3 at a shared weights/cache directory.
            # Use the same value when constructing blocks in the agent and
            # when executing their serialized commands in Slurm workers.
            af3_jax_cache = self._host_workspace / ".cache" / "af3_jax"
            af3_jax_cache.mkdir(parents=True, exist_ok=True)
            trial_site_environment = {
                **get_site()["environment"],
                "AF3_JAX_CACHE_DIR": str(af3_jax_cache),
            }
            slurm_python = runtime_dir / "slurm-python"
            self._prepare_slurm_python(
                slurm_python,
                self._host_workspace,
                self._slurm_worker_image,
                site_environment=trial_site_environment,
            )
            self._slurm_python = slurm_python.resolve()
            codex_vendor = self._find_codex_vendor()
            broker_client = runtime_dir / "slurm-client.py"
            shutil.copy2(_SBATCH_GUARD, broker_client)
            broker_client.chmod(0o500)
            self._slurm_broker = SlurmBrokerProcess(
                workspace=self._host_workspace,
                runtime_dir=runtime_dir / "slurm-broker",
                policy="submitit",
                slurm_python=self._slurm_python,
                client_path=broker_client,
            )
            broker_dir = self._slurm_broker.socket_dir
            persistent_env.update(
                {
                    "HARBOR_HOST_WORKSPACE": str(self._host_workspace),
                    "HOME": str(self._host_workspace / ".home"),
                    "TMPDIR": str(self._host_workspace / ".tmp"),
                    "XDG_CACHE_HOME": str(self._host_workspace / ".home/.cache"),
                    "XDG_CONFIG_HOME": str(self._host_workspace / ".home/.config"),
                    "MPLCONFIGDIR": str(
                        self._host_workspace / ".home/.config/matplotlib"
                    ),
                    "PYTHONPATH": str(self._host_workspace / "chaperone/src"),
                    "QHOP": "false",
                    "HARBOR_SBATCH_SOCKET": "/run/harbor-slurm/sbatch.sock",
                    "CHAPERONE_SLURM_PYTHON": str(slurm_python),
                    "CHAPERONE_SLURM_ARRAY_PARALLELISM": str(
                        get_site()["max_total_gpus"]
                    ),
                    "HARBOR_MAX_REQUESTED_GPUS_PER_SUBMISSION": str(
                        get_site()["max_requested_gpus_per_submission"]
                    ),
                    **trial_site_environment,
                }
            )
            resolved_mounts.extend(
                [
                    {"type": "bind", "source": str(workspace), "target": "/workspace"},
                    {
                        "type": "bind",
                        "source": str(self._host_workspace),
                        "target": str(self._host_workspace),
                    },
                    {
                        "type": "bind",
                        "source": str(workspace / ".tmp"),
                        "target": "/tmp",
                    },
                    {
                        "type": "bind",
                        "source": str(workspace / ".harbor"),
                        "target": "/harbor",
                    },
                    {
                        "type": "bind",
                        "source": str(runtime_dir / "passwd"),
                        "target": "/etc/passwd",
                        "read_only": True,
                    },
                    {
                        "type": "bind",
                        "source": str(runtime_dir / "group"),
                        "target": "/etc/group",
                        "read_only": True,
                    },
                    {
                        "type": "bind",
                        "source": str(broker_dir),
                        "target": "/run/harbor-slurm",
                    },
                    {
                        "type": "bind",
                        "source": str(broker_client),
                        "target": "/usr/bin/sbatch",
                        "read_only": True,
                    },
                    {
                        "type": "bind",
                        "source": str(slurm_python),
                        "target": str(slurm_python),
                        "read_only": True,
                    },
                ]
            )
            resolved_mounts.extend(self._runtime_mounts())
            # The outer agent sees scoped broker clients for mutating/control
            # commands. The generated compute-node wrapper still binds the
            # node's real Slurm binaries inside broker-approved allocations.
            for command in ("scancel", "scontrol", "srun"):
                resolved_mounts.append(
                    {
                        "type": "bind",
                        "source": str(broker_client),
                        "target": f"/usr/bin/{command}",
                        "read_only": True,
                    }
                )
            resolved_mounts.append(
                {
                    "type": "bind",
                    "source": str(codex_vendor),
                    "target": "/opt/harbor-codex",
                    "read_only": True,
                }
            )
        try:
            super().__init__(
                environment_dir=environment_dir,
                environment_name=environment_name,
                session_id=session_id,
                trial_paths=trial_paths,
                task_env_config=task_env_config,
                *args,
                mounts=resolved_mounts,
                persistent_env=persistent_env,
                **kwargs,
            )
        except BaseException:
            self._stop_sbatch_broker()
            raise

    @property
    def _docker_image(self) -> str | None:
        """Use a per-run SIF when the Slurm submission workflow built one."""
        return self._apptainer_image or super()._docker_image

    def _start_sbatch_broker(self) -> None:
        if self._slurm_broker is not None:
            self._slurm_broker.start()

    def _stop_sbatch_broker(self) -> None:
        if self._slurm_broker is not None:
            self._slurm_broker.stop()

    @staticmethod
    def _make_sbatch_broker_dir() -> Path:
        """Allocate a private directory whose Unix socket path stays short."""
        directory = Path(tempfile.mkdtemp(prefix="harbor-sb-", dir="/tmp"))
        directory.chmod(0o700)
        socket_path = directory / "sbatch.sock"
        if len(os.fsencode(socket_path)) > _UNIX_SOCKET_PATH_MAX_BYTES:
            shutil.rmtree(directory, ignore_errors=True)
            raise RuntimeError(f"Guarded sbatch socket path is too long: {socket_path}")
        return directory

    @staticmethod
    def _prepare_workspace(workspace: Path, seed: Path, inputs: Path) -> None:
        if not seed.is_dir():
            raise FileNotFoundError(f"Harness snapshot not found: {seed}")
        if not inputs.is_dir():
            raise FileNotFoundError(f"Task inputs not found: {inputs}")
        validate_snapshot(seed, seed.name)
        input_symlinks = sorted(
            path.relative_to(inputs) for path in inputs.rglob("*") if path.is_symlink()
        )
        if input_symlinks:
            raise RuntimeError(
                "Task inputs contain forbidden symlinks: "
                + ", ".join(str(path) for path in input_symlinks[:5])
            )

        workspace.mkdir(parents=True, exist_ok=True)
        chaperone = workspace / "chaperone"
        if not chaperone.exists():
            staging = workspace / ".chaperone.partial"
            if staging.exists():
                shutil.rmtree(staging)
            try:
                shutil.copytree(seed, staging)
                validate_snapshot(staging, seed.name)
                staging.rename(chaperone)
            except BaseException:
                shutil.rmtree(staging, ignore_errors=True)
                raise
        else:
            # The workspace directory is named "chaperone", so compare it to
            # the immutable source bundle's digest rather than its own name.
            validate_snapshot(chaperone, seed.name)

        input_copy = workspace / "inputs"
        if not input_copy.exists():
            shutil.copytree(inputs, input_copy)
        for directory in ("submission", ".home", ".tmp", ".harbor"):
            (workspace / directory).mkdir(exist_ok=True)

    @staticmethod
    def _project_path_alias(path: Path) -> Path | None:
        """Return the equivalent shared-filesystem spelling, if any."""
        text = str(path)
        for source, target in (
            ("/mnt/projects/", "/projects/"),
            ("/projects/", "/mnt/projects/"),
            ("/mnt/net/", "/net/"),
            ("/net/", "/mnt/net/"),
        ):
            if text.startswith(source):
                return Path(target + text.removeprefix(source))
        return None

    @staticmethod
    def _prepare_slurm_python(
        path: Path,
        workspace: Path,
        image: Path,
        chaperone_repo: Path | None = None,
        worker_readonly_paths: tuple[Path, ...] = (),
        site_environment: Mapping[str, str] | None = None,
    ) -> None:
        """Create the shared executable Submitit uses on compute nodes.

        ``site_environment`` holds controller-resolved values for scientific
        tool locations (names from ``site_config.ENV_NAMES`` only). They are
        exported inside the worker after the checkout's ``.env`` files are
        sourced, so they win over the checkout defaults: the host scorer and the
        ``--cleanenv`` Slurm worker then see the same setting.
        """
        if not image.is_file() or image.suffix != ".sif":
            raise FileNotFoundError(
                f"Shared Chaperone worker SIF was not found: {image}"
            )
        # Preserve the caller's logical shared-filesystem spelling. On DIGS
        # CPU nodes /projects contains the checkout's .venv while resolving it
        # to /mnt/projects produces a namespace without that environment.
        chaperone_repo = Path(
            os.path.abspath(chaperone_repo or workspace / "chaperone")
        )
        if not (chaperone_repo / "src").is_dir():
            raise FileNotFoundError(
                f"Chaperone source tree not found: {chaperone_repo}"
            )
        bind_paths: list[tuple[Path, Path, bool]] = [(workspace, workspace, False)]
        # The worker runs with --containall, so its synthetic /etc/passwd does
        # not know the submitting uid.  Use the minimal NSS files staged next
        # to slurm-python (current user plus Slurm service accounts) rather
        # than exposing the host's complete account database.
        for name in ("passwd", "group"):
            identity_file = path.parent / name
            if identity_file.is_file():
                bind_paths.append((identity_file, Path("/etc") / name, True))
        try:
            chaperone_repo.relative_to(workspace)
        except ValueError:
            bind_paths.append((chaperone_repo, chaperone_repo, True))
        chaperone_repo_alias = ApptainerEnvironment._project_path_alias(chaperone_repo)
        if chaperone_repo_alias is not None:
            # A host-side scorer launched from a CPU node can canonicalize
            # /projects to /mnt/projects before it serializes task assets into
            # a Submitit pickle. Mount the checkout at both spellings so held-
            # out inputs such as score/../inputs/target.pdb remain readable in
            # the scientific worker container.
            bind_paths.append((chaperone_repo, chaperone_repo_alias, True))
        for readonly_path in worker_readonly_paths:
            readonly_path = Path(os.path.abspath(readonly_path))
            readonly_alias = ApptainerEnvironment._project_path_alias(readonly_path)
            # Compute nodes use /projects as the physical shared-filesystem
            # mount, while CPU controllers can canonicalize the same path to
            # /mnt/projects. Always bind from the physical spelling and expose
            # the exact task asset directory at both spellings. This keeps
            # held-out templates and controller-staged inputs readable without
            # mounting the enclosing evaluation repository.
            if str(readonly_path).startswith("/mnt/projects/"):
                readonly_source = readonly_alias or readonly_path
            else:
                readonly_source = readonly_path
            bind_paths.append((readonly_source, readonly_path, True))
            if readonly_alias is not None:
                bind_paths.append((readonly_source, readonly_alias, True))
        workspace_alias = ApptainerEnvironment._project_path_alias(workspace)
        if workspace_alias is not None:
            # Chaperone's nested scientific containers bind /projects to
            # /mnt/projects. Expose the writable trial at both equivalent
            # spellings so input and output paths survive that extra boundary.
            bind_paths.append((workspace, workspace_alias, False))
        bind_paths.extend(
            (Path(source), Path(target), True)
            for source, target in _runtime_bind_pairs()
        )
        bind_args: list[str] = []
        seen: set[str] = set()
        for source, target, read_only in bind_paths:
            if not _optional_path_exists(source):
                continue
            specification = f"{source}:{target}"
            if read_only:
                specification += ":ro"
            if specification in seen:
                continue
            seen.add(specification)
            bind_args.extend(("-B", specification))
        site_environment = {**get_site()["environment"], **(site_environment or {})}
        unapproved = set(site_environment) - ENV_NAMES
        if unapproved:
            raise ValueError(
                "unapproved scientific worker environment names: "
                + ", ".join(sorted(unapproved))
            )
        if any("\x00" in str(value) for value in site_environment.values()):
            raise ValueError("worker environment values must not contain NUL")
        site_exports = tuple(
            f"export {name}={shlex.quote(str(value))}"
            for name, value in sorted(site_environment.items())
        )
        inner = "\n".join(
            (
                f"export HOME={shlex.quote(str(workspace / '.home'))}",
                "export TMPDIR=/tmp",
                f"export XDG_CACHE_HOME={shlex.quote(str(workspace / '.home/.cache'))}",
                f"export XDG_CONFIG_HOME={shlex.quote(str(workspace / '.home/.config'))}",
                f"export MPLCONFIGDIR={shlex.quote(str(workspace / '.home/.config/matplotlib'))}",
                # Some Chaperone scientific manifests include standard home
                # mount points even when they do not consume files from them.
                # --containall may omit /mnt/home, so create empty mount
                # sources in the worker's writable overlay before nesting.
                "mkdir -p /mnt/home /home",
                f"export PYTHONPATH={shlex.quote(str(chaperone_repo / 'src'))}${{PYTHONPATH:+:$PYTHONPATH}}",
                f"if [[ -f {shlex.quote(str(chaperone_repo / '.env'))} ]]; then",
                "  set -a",
                f"  source {shlex.quote(str(chaperone_repo / '.env'))}",
                "  set +a",
                "fi",
                # Reassert the controller's values after the env files so an
                # operator override or a scorer-specific path is not replaced
                # by the checkout default.
                *site_exports,
                "export QHOP=false",
                "export CHAPERONE_COMPUTE_BACKEND=auto",
                "export CHAPERONE_CONTAINER_BACKEND=apptainer",
                f'cd "${{HARBOR_WORKER_CWD:-{str(workspace)}}}"',
                f'exec {shlex.quote(get_site()["worker_python"])} "$@"',
            )
        )
        command = shlex.join(
            [
                "apptainer",
                "exec",
                "--cleanenv",
                "--containall",
                "--no-mount",
                "home",
                "--no-mount",
                "tmp",
                "--no-mount",
                "bind-paths",
                "--writable-tmpfs",
                "--pwd",
                str(workspace),
                *bind_args,
                str(image),
                "bash",
                "-lc",
                inner,
                "bash",
            ]
        )
        # Add --nv only when Slurm assigned GPUs to this worker.
        script = "\n".join(
            (
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "gpu_args=()",
                "env_args=()",
                "runtime_bind_args=()",
                f"worker_cwd=${{SLURM_SUBMIT_DIR:-{shlex.quote(str(workspace))}}}",
                f'case "$worker_cwd/" in {shlex.quote(str(workspace))}/*) ;; *) worker_cwd={shlex.quote(str(workspace))} ;; esac',
                f"worker_tmp=${{SLURM_TMPDIR:-{shlex.quote(str(workspace / '.tmp'))}/slurm-${{SLURM_JOB_ID:-manual}}-${{SLURM_ARRAY_TASK_ID:-0}}}}",
                'mkdir -p "$worker_tmp"',
                'runtime_bind_args+=(-B "$worker_tmp:/tmp")',
                "if [[ -s /run/systemd/resolve/resolv.conf ]]; then",
                "  worker_resolver=/run/systemd/resolve/resolv.conf",
                "elif [[ -s /etc/resolv.conf ]]; then",
                "  worker_resolver=/etc/resolv.conf",
                "else",
                "  echo 'No usable DNS resolver exists on this Slurm node' >&2",
                "  exit 2",
                "fi",
                'runtime_bind_args+=(-B "$worker_resolver:/etc/resolv.conf:ro")',
                'env_args+=(--env "HARBOR_WORKER_CWD=$worker_cwd")',
                'if [[ -n "${CUDA_VISIBLE_DEVICES:-}" || -n "${SLURM_JOB_GPUS:-}" ]]; then',
                "  gpu_args=(--nv)",
                "fi",
                "for name in "
                + " ".join(
                    sorted(
                        (
                            *_HOST_ENV_PASSTHROUGH,
                            *_ALLOWLISTED_WORKER_ENV_PASSTHROUGH,
                            "SUBMITIT_EXECUTOR",
                        )
                    )
                )
                + "; do",
                '  if declare -p "$name" >/dev/null 2>&1; then env_args+=(--env "$name=${!name}"); fi',
                "done",
                "for name in " + " ".join(sorted(ENV_NAMES)) + "; do",
                '  if declare -p "$name" >/dev/null 2>&1; then env_args+=(--env "$name=${!name}"); fi',
                "done",
                'while IFS= read -r name; do env_args+=(--env "$name=${!name}"); done < <(compgen -A variable SLURM_)',
                command.replace(
                    "apptainer exec",
                    'apptainer exec "${gpu_args[@]}" "${env_args[@]}" "${runtime_bind_args[@]}"',
                    1,
                )
                + ' "$@"',
                "",
            )
        )
        # The installed wrapper is deliberately read-only. Make a prior copy
        # writable before regenerating it during a scorer retry.
        if path.exists():
            path.chmod(0o700)
        path.write_text(script)
        path.chmod(0o500)

    @staticmethod
    def _prepare_runtime_identity(runtime_dir: Path, workspace: Path) -> None:
        """Create minimal NSS files needed by Apptainer and the Slurm client.

        Binding the host's complete passwd/group databases would reveal unrelated
        local account names. These files contain only the invoking user and the
        service identities named by the mounted Slurm configuration.
        """
        runtime_dir.mkdir(parents=True, exist_ok=True)
        worker_home = workspace.resolve() / ".home"
        worker_home.mkdir(parents=True, exist_ok=True)
        if any(character in str(worker_home) for character in (":", "\n", "\r")):
            raise ValueError(f"unsafe workspace home path: {worker_home}")

        passwd_records: dict[str, pwd.struct_passwd] = {}
        for name in ("root", "slurm", "munge", "nobody"):
            try:
                record = pwd.getpwnam(name)
            except KeyError:
                continue
            passwd_records[record.pw_name] = record
        current = pwd.getpwuid(os.getuid())
        passwd_records[current.pw_name] = current

        passwd_lines = []
        for record in passwd_records.values():
            home = str(worker_home) if record.pw_uid == os.getuid() else record.pw_dir
            shell = "/bin/bash" if record.pw_uid == os.getuid() else record.pw_shell
            passwd_lines.append(
                ":".join(
                    (
                        record.pw_name,
                        "x",
                        str(record.pw_uid),
                        str(record.pw_gid),
                        record.pw_gecos,
                        home,
                        shell,
                    )
                )
            )

        group_records: dict[str, grp.struct_group] = {}
        wanted_gids = {record.pw_gid for record in passwd_records.values()}
        for gid in wanted_gids:
            try:
                record = grp.getgrgid(gid)
            except KeyError:
                continue
            group_records[record.gr_name] = record
        for name in ("root", "slurm", "munge", "nogroup"):
            try:
                record = grp.getgrnam(name)
            except KeyError:
                continue
            group_records[record.gr_name] = record

        included_users = set(passwd_records)
        group_lines = []
        for record in group_records.values():
            members = ",".join(
                member for member in record.gr_mem if member in included_users
            )
            group_lines.append(f"{record.gr_name}:x:{record.gr_gid}:{members}")

        (runtime_dir / "passwd").write_text("\n".join(passwd_lines) + "\n")
        (runtime_dir / "group").write_text("\n".join(group_lines) + "\n")
        (runtime_dir / "passwd").chmod(0o644)
        (runtime_dir / "group").chmod(0o644)

    @staticmethod
    def _find_codex_vendor() -> Path:
        from environments.codex_vendor import find_codex_vendor

        return find_codex_vendor()

    @staticmethod
    def _find_resolver_config() -> Path:
        """Return a non-empty host resolver file suitable for the container."""
        candidates = (
            # Prefer the upstream DNS servers over systemd-resolved's loopback
            # stub, which may not be reachable from every Apptainer network
            # configuration.
            Path("/run/systemd/resolve/resolv.conf"),
            Path("/etc/resolv.conf"),
        )
        for candidate in candidates:
            try:
                if candidate.is_file() and candidate.stat().st_size > 0:
                    return candidate.resolve(strict=True)
            except OSError:
                continue
        raise FileNotFoundError(
            "No usable DNS resolver configuration was found on the host "
            "(/run/systemd/resolve/resolv.conf or /etc/resolv.conf)"
        )

    @staticmethod
    def _runtime_mounts() -> list[dict[str, object]]:
        return [
            {
                "type": "bind",
                "source": source,
                "target": target,
                "read_only": True,
            }
            for source, target in _runtime_bind_pairs()
            if target not in _BROKERED_SLURM_PATHS
            and _optional_path_exists(Path(source))
        ]

    @classmethod
    @override
    def preflight(cls) -> None:
        executable = shutil.which("apptainer")
        if executable is None:
            raise SystemExit(
                "Apptainer is not installed or not on PATH. "
                "Please load/install Apptainer and try again."
            )
        result = subprocess.run(
            [executable, "version"], capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise SystemExit(f"Apptainer preflight failed: {detail}")

    @property
    @override
    def capabilities(self) -> EnvironmentCapabilities:
        return super().capabilities.model_copy(update={"gpus": True})

    @override
    def _resolve_workdir(self) -> str:
        if self._host_workspace is not None:
            return str(self._host_workspace)
        return self.task_env_config.workdir or "/workspace"

    @override
    async def start(self, force_build: bool) -> None:
        del force_build
        if not self._is_sif_image:
            raise ValueError(
                "ApptainerEnvironment requires [environment].docker_image to "
                "reference an existing .sif file"
            )
        self._sif_path = Path(self._docker_image or "")
        self._staging_dir = Path(tempfile.mkdtemp(prefix="apptainer-staging-"))
        self._staging_dir.chmod(0o755)
        try:
            self._start_sbatch_broker()
            result = await self.exec(
                "codex --version"
                f" && {shlex.quote(get_site()['worker_python'])} -m chaperone_cli.main --help >/dev/null"
                " && sbatch --version >/dev/null"
                " && getent ahosts api.openai.com >/dev/null"
            )
            if result.return_code != 0:
                raise RuntimeError(
                    "Apptainer runtime validation failed: "
                    + (result.stdout or result.stderr or "unknown error")
                )
            if self._host_workspace is not None:

                async def probe() -> None:
                    health = await self.exec(
                        "set -e; test -x /bin/bash; test -x /bin/sh; "
                        f"test -x {shlex.quote(get_site()['worker_python'])}; "
                        "test -S /run/harbor-slurm/sbatch.sock; "
                        "probe=.harbor/.controller-health-probe; "
                        'printf healthy >"$probe"; test "$(cat "$probe")" = healthy; '
                        'rm -f "$probe"',
                        timeout_sec=45,
                    )
                    if health.return_code != 0:
                        raise RuntimeError(health.stdout or "controller probe failed")

                async def fail(_error: BaseException) -> None:
                    self._stop_sbatch_broker()

                self._health_watchdog = ControllerHealthWatchdog(
                    report_path=self._host_workspace
                    / ".harbor"
                    / "controller_health.json",
                    probe=probe,
                    on_failure=fail,
                )
                self._health_watchdog.start()
        except BaseException:
            self._stop_sbatch_broker()
            if self._staging_dir is not None:
                shutil.rmtree(self._staging_dir, ignore_errors=True)
                self._staging_dir = None
            raise

    @override
    async def stop(self, delete: bool) -> None:
        try:
            if self._health_watchdog is not None:
                await self._health_watchdog.stop()
                self._health_watchdog = None
            await super().stop(delete)
        finally:
            self._stop_sbatch_broker()
            if self._host_workspace is not None:
                cache_dir = self._host_workspace / ".harbor" / "singularity-cache"
                shutil.rmtree(cache_dir, ignore_errors=True)

    def _bind_args(self) -> list[str]:
        bind_args = ["-B", f"{self._staging_dir}:/staging"]
        for mount in self._mounts:
            if mount.get("type") != "bind":
                continue
            specification = f"{mount['source']}:{mount['target']}"
            if mount.get("read_only"):
                specification += ":ro"
            bind_args.extend(["-B", specification])
        task_files = self.environment_dir / "files"
        if task_files.is_dir():
            bind_args.extend(["-B", f"{task_files}:/staging/env_files:ro"])
        return bind_args

    def _ensure_staging_dir(self) -> None:
        """Restore Harbor's host staging mount if temporary cleanup removed it.

        Harbor's inherited artifact downloader copies through ``/staging``.
        Long-running trials can outlive host temporary-directory cleanup, and a
        missing bind source otherwise turns a valid submission into a failed
        trial during the final artifact download.  The directory contains only
        transient transfer files, so recreating an empty mount is safe.
        """
        if self._staging_dir is None:
            raise RuntimeError("Apptainer environment has not been started")
        self._staging_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
        self._staging_dir.chmod(0o755)

    @override
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        del user  # All commands run as the unprivileged invoking user.
        if self._sif_path is None or self._staging_dir is None:
            raise RuntimeError("Apptainer environment has not been started")
        self._ensure_staging_dir()

        disabled_mount_args: list[str] = []
        disabled_mounts = self._singularity_no_mount
        if disabled_mounts is None:
            disabled_mounts = "home,tmp,bind-paths"
        for mount_type in disabled_mounts.split(","):
            mount_type = mount_type.strip()
            if mount_type:
                disabled_mount_args.extend(["--no-mount", mount_type])

        setup = self.environment_dir / "files" / "setup.sh"
        shell_command = command
        if setup.is_file():
            shell_command = "source /staging/env_files/setup.sh && " + shell_command
        container_env = self._merge_env(env) or {}
        for name in _HOST_ENV_PASSTHROUGH:
            if name in os.environ and name not in container_env:
                container_env[name] = os.environ[name]
        for name, value in os.environ.items():
            if name.startswith("SLURM_") and name not in container_env:
                container_env[name] = value
        # Values travel to apptainer through its own environment
        # (APPTAINERENV_<NAME>, honoured under --cleanenv), never as --env
        # arguments: an API key on the command line is readable by other users of
        # the node through ps and /proc/<pid>/cmdline.
        container_env_vars = {
            f"APPTAINERENV_{name}": str(value)
            for name, value in container_env.items()
            if name != "HOME"
        }
        gpu_args = ["--nv"] if self._gpu_enabled else []
        apptainer_command = [
            shutil.which("apptainer") or "apptainer",
            "exec",
            *disabled_mount_args,
            *gpu_args,
            "--cleanenv",
            "--no-eval",
            "--pwd",
            cwd or self._workdir,
            "--writable-tmpfs",
            "--containall",
            "--pid",
            *self._bind_args(),
            str(self._sif_path),
            "bash",
            "-lc",
            shell_command,
        ]
        process_env = {
            name: value
            for name, value in os.environ.items()
            if not name.startswith(("APPTAINER_", "SINGULARITY_"))
        }
        process_env.update(container_env_vars)
        process = await asyncio.create_subprocess_exec(
            *apptainer_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=process_env,
        )
        try:
            stdout, _ = await asyncio.wait_for(
                process.communicate(), timeout=timeout_sec
            )
        except asyncio.CancelledError:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
            raise
        except asyncio.TimeoutError:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
            return ExecResult(
                stdout=f"Command timed out after {timeout_sec} seconds",
                stderr=None,
                return_code=124,
            )

        output = stdout.decode(errors="replace") if stdout else None
        callback = self._output_callback()
        if callback is not None and output:
            await callback(output, OutputStream.STDOUT)
        return ExecResult(
            stdout=output,
            stderr=None,
            return_code=process.returncode if process.returncode is not None else 1,
        )
