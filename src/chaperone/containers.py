"""Image: Simple builder for container images using HPCCM.

Example:
    E1_IMAGE = (
        Image("e1")
        .base_image("ubuntu:24.04")
        .packages("git", "python3")
        .run("pip install torch")
        .env(CUDA_HOME="/usr/local/cuda")
    )

    # Generate definitions
    print(E1_IMAGE.to_singularity())

    # Find built container
    sif_path = E1_IMAGE.get_image_path()
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import cache
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess


import hpccm
from hpccm.building_blocks import packages
from hpccm.primitives import baseimage, copy, environment, label, raw, shell


logger = logging.getLogger(__name__)

_UNSAFE_RUNDIR_BINDS = frozenset(
    Path(path)
    for path in (
        "/",
        "/home",
        "/mnt",
        "/mnt/net",
        "/mnt/net/scratch",
        "/mnt/projects",
        "/net",
        "/net/scratch",
        "/projects",
        "/tmp",
    )
)

_RESOLVER_PATHS = (
    # Prefer upstream nameservers over systemd-resolved's loopback stub.
    Path("/run/systemd/resolve/resolv.conf"),
    Path("/etc/resolv.conf"),
)


def _runtime_resolver_bind(binds: list[tuple[str, str]]) -> tuple[str, str] | None:
    """Keep DNS available in contained Apptainer tools with one read-only file.

    This also preserves the resolver already assigned to an outer evaluation
    container. It grants no directory or scientific-asset access. An explicit
    mount at the resolver (or an enclosing directory) owns that configuration.
    """
    target = Path("/etc/resolv.conf")
    for _, destination in binds:
        destination_path = Path(str(destination).split(":", 1)[0])
        if destination_path.is_absolute() and target.is_relative_to(destination_path):
            return None
    for candidate in _RESOLVER_PATHS:
        try:
            if candidate.is_file() and candidate.stat().st_size > 0:
                return str(candidate.resolve(strict=True)), str(target) + ":ro"
        except OSError:
            # Optional host paths can be missing, inaccessible or unmounted.
            # Offline tools still work without a resolver; do not bind a
            # nonexistent source or substitute an unrelated host directory.
            continue
    return None


@cache
def _supports_private_mount_namespace() -> bool:
    """Return whether the host permits the mount namespace wrapper.

    Some cluster security configurations disable unprivileged ``unshare``
    while continuing to support Apptainer directly. Probe once on the
    submission host so containerized blocks remain runnable after such a
    policy change instead of failing before Apptainer starts.
    """
    executable = shutil.which("unshare")
    if executable is None:
        return False
    try:
        result = subprocess.run(
            [
                executable,
                "--mount",
                "--propagation",
                "private",
                "--fork",
                "--",
                "true",
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def _checked_runtime_path(value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    path = path.absolute()
    if path in _UNSAFE_RUNDIR_BINDS or path.resolve() in _UNSAFE_RUNDIR_BINDS:
        raise ValueError(f"Refusing broad nested-container {label} bind: {path}")
    return path


_RUNTIME_BIND_SETTINGS: ContextVar[dict | None] = ContextVar(
    "container_runtime_bind_settings", default=None
)
_RUNTIME_GPU: ContextVar[bool] = ContextVar("container_runtime_gpu", default=False)


@contextmanager
def container_runtime_gpu(enabled: bool):
    """Use block resource requirements without changing process-wide state."""
    token = _RUNTIME_GPU.set(enabled)
    try:
        yield
    finally:
        _RUNTIME_GPU.reset(token)


def refresh_container_gpu_options(command: list[str], enabled: bool) -> list[str]:
    """Refresh only runtime GPU options in a frozen, generated image template.

    Submission-time mounts and lazy environment values must remain frozen.
    Explicit user templates are never passed to this function.
    """
    from chaperone.common import is_macos

    result = list(command)
    if result[:2] == ["docker", "run"]:
        index, docker = 2, True
    elif "apptainer" in result:
        index, docker = result.index("apptainer") + 2, False
    else:
        return result
    start = index
    options = []
    pairs = {
        "-v",
        "-e",
        "--gpus",
        "--bind",
        "--env",
        "--no-mount",
        "--pwd",
        "--cwd",
        "--home",
        "-w",
        "--workdir",
    }
    while index < len(result) and result[index].startswith("-"):
        option = result[index]
        width = 2 if option in pairs else 1
        value = result[index + 1] if width == 2 else ""
        if option not in {"--gpus", "--nv"} and not (
            option in {"-e", "--env"}
            and value.split("=", 1)[0]
            in {"CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"}
        ):
            options.extend(result[index : index + width])
        index += width
    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    if mask is not None:
        mask = mask.strip()
    enabled = enabled and mask not in ("", "-1", "none", "void")
    if docker:
        if enabled and not is_macos():
            options += ["--gpus", "all" if mask is None else f'"device={mask}"']
            if mask is not None:
                inner = ",".join(str(i) for i in range(len(mask.split(","))))
                options += ["-e", f"CUDA_VISIBLE_DEVICES={inner}"]
        else:
            options += [
                "-e",
                "NVIDIA_VISIBLE_DEVICES=void",
                "-e",
                "CUDA_VISIBLE_DEVICES=",
            ]
    else:
        if enabled:
            options += ["--nv"]
        if not enabled or mask is not None:
            options += ["--env", f"CUDA_VISIBLE_DEVICES={mask if enabled else ''}"]
    return result[:start] + options + result[index:]


def _runtime_setting(settings: object, name: str, default=None):
    context = _RUNTIME_BIND_SETTINGS.get()
    if context is not None and name in context:
        return context[name]
    return getattr(settings, name, default)


@contextmanager
def container_runtime_bind_settings(context: dict | None):
    """Restore submission paths while staging a block in a fresh Slurm worker."""
    token = _RUNTIME_BIND_SETTINGS.set(context)
    try:
        yield
    finally:
        _RUNTIME_BIND_SETTINGS.reset(token)


def capture_container_runtime_bind_settings(rundir: Path) -> dict:
    """Snapshot exact engine paths, independent of a worker's cwd or settings."""
    from chaperone import get_settings

    settings = get_settings()
    workspace = _runtime_workspace_bind(settings)
    return {
        "RUNDIR": rundir,
        "CHAPERONE_CONTAINER_BIND_RUNDIR": settings.CHAPERONE_CONTAINER_BIND_RUNDIR,
        "CHAPERONE_CONTAINER_BIND_WORKSPACE": workspace is not None,
        "CHAPERONE_CONTAINER_WORKSPACE": workspace[0] if workspace else None,
        "_HARBOR_HOST_WORKSPACE": os.environ.get("HARBOR_HOST_WORKSPACE"),
    }


def _runtime_rundir_bind(settings: object) -> tuple[str, str] | None:
    """Expose the exact active run directory once the engine has selected it."""
    if _runtime_setting(settings, "CHAPERONE_CONTAINER_BIND_RUNDIR", False) is not True:
        return None
    configured = _runtime_setting(settings, "RUNDIR", None)
    if configured is None:
        # Block instances commonly precede ExecutionEngine construction. Their
        # generated templates are refreshed when create_manifest is called.
        return None
    path = _checked_runtime_path(configured, label="RUNDIR")
    return str(path), str(path)


def _runtime_workspace_bind(settings: object) -> tuple[str, str] | None:
    """Use the trusted eval workspace, or an ordinary user's exact work folder."""
    if (
        _runtime_setting(settings, "CHAPERONE_CONTAINER_BIND_WORKSPACE", False)
        is not True
    ):
        return None
    configured = (
        os.environ.get("HARBOR_HOST_WORKSPACE")
        or _runtime_setting(settings, "_HARBOR_HOST_WORKSPACE")
        or _runtime_setting(settings, "CHAPERONE_CONTAINER_WORKSPACE")
    )
    if configured is None:
        configured = Path.cwd()
        if (
            configured in _UNSAFE_RUNDIR_BINDS
            or configured.resolve() in _UNSAFE_RUNDIR_BINDS
            or configured.resolve() == Path.home().resolve()
        ):
            return None
    path = _checked_runtime_path(configured, label="workspace")
    return str(path), str(path)


def _runtime_bind_aliases(bind: tuple[str, str]) -> list[tuple[str, str]]:
    """Keep both /net and /mnt/net (and other resolved spellings) usable."""
    source, destination = bind
    resolved = str(Path(source).resolve())
    return list(dict.fromkeys([bind, (resolved, resolved)]))


def _runtime_script_binds(
    settings: object, args: tuple[str, ...]
) -> list[tuple[str, str]]:
    """Expose installed wrapper code without mounting its enclosing environment."""
    if (
        _runtime_setting(settings, "CHAPERONE_CONTAINER_BIND_WORKSPACE", False)
        is not True
    ):
        return []
    # Restricted workers receive only their assigned workspace and
    # read-only assets. Never infer additional source mounts in that mode.
    if os.environ.get("HARBOR_HOST_WORKSPACE") or _runtime_setting(
        settings, "_HARBOR_HOST_WORKSPACE"
    ):
        return []
    covered = [
        Path(destination)
        for bind in (_runtime_rundir_bind(settings), _runtime_workspace_bind(settings))
        if bind is not None
        for _, destination in _runtime_bind_aliases(bind)
    ]
    mounts = []
    for argument in args:
        script = Path(argument)
        if (
            script.is_absolute()
            and script.suffix in {".py", ".sh"}
            and script.is_file()
        ):
            directory = _checked_runtime_path(script.parent, label="script directory")
            if any(
                directory.resolve().is_relative_to(root.resolve()) for root in covered
            ):
                if not any(directory.is_relative_to(root) for root in covered):
                    # The workspace may have been captured from a resolved cwd
                    # (/mnt/net) while the wrapper retains /net. Expose that
                    # exact alias with the workspace's existing writable access.
                    mounts.append((str(directory), str(directory)))
                # Never add a read-only resolved alias over a writable workspace.
                continue
            for source, destination in _runtime_bind_aliases(
                (str(directory), str(directory))
            ):
                mounts.append((source, destination + ":ro"))
    return list(dict.fromkeys(mounts))


def get_container_dirs() -> list[Path]:
    """Get container search directories in priority order."""
    dirs: list[Path] = []

    # Environment variable (highest priority)
    from chaperone import get_setting_field

    env_dirs = get_setting_field("CHAPERONE_CONTAINER_DIRS", default="")
    if env_dirs:
        dirs.extend(Path(p.strip()) for p in env_dirs.split(":") if p.strip())

    # User directory (fallback)
    dirs.append(Path.home() / ".chaperone" / "containers")

    return list(dict.fromkeys(dirs))  # Dedupe preserving order


@dataclass
class Image:
    """Builder for container images using HPCCM."""

    name: str
    _base_image: str | None = None
    _commands: list[list[str]] = field(default_factory=list)
    _env_vars: dict[str, str] = field(default_factory=dict)
    _labels: dict[str, str] = field(default_factory=dict)
    _apt_packages: list[str] = field(default_factory=list)
    _copies: list[tuple[str, str]] = field(default_factory=list)
    _git_clones: list[tuple[str, str, str | None]] = field(default_factory=list)
    _runscript: str | None = None
    _existing_path: Path | None = None
    _command_config: dict | None = None

    def _clone(self) -> Image:
        # Bypass __init__/__post_init__ to avoid re-registering intermediate clones
        new = object.__new__(Image)
        new.name = self.name
        new._base_image = self._base_image
        new._commands = [list(c) for c in self._commands]
        new._env_vars = dict(self._env_vars)
        new._labels = dict(self._labels)
        new._apt_packages = list(self._apt_packages)
        new._copies = list(self._copies)
        new._git_clones = list(self._git_clones)
        new._runscript = self._runscript
        new._existing_path = self._existing_path
        new._command_config = (
            dict(self._command_config) if self._command_config else None
        )
        return new

    def base_image(self, image: str) -> Image:
        """Set the base image for the container."""
        new = self._clone()
        new._base_image = image
        return new

    def packages(self, *pkgs: str) -> Image:
        """Install apt packages."""
        new = self._clone()
        new._apt_packages.extend(pkgs)
        return new

    def run(self, *commands: str) -> Image:
        """Run shell commands."""
        new = self._clone()
        new._commands.append(list(commands))
        return new

    def env(self, **variables: str) -> Image:
        """Set environment variables."""
        new = self._clone()
        new._env_vars.update(variables)
        return new

    def label(self, **labels: str) -> Image:
        """Add labels/metadata."""
        new = self._clone()
        new._labels.update(labels)
        return new

    def copy(self, src: str, dst: str) -> Image:
        """Copy files into container."""
        new = self._clone()
        new._copies.append((src, dst))
        return new

    def git_clone(self, url: str, dest: str, branch: str | None = None) -> Image:
        """Clone a git repo into the container.

        The repo is cloned on the host before the container build and injected
        via %files (apptainer) or COPY (docker). This avoids needing network
        access to GitHub during the container build itself.

        Args:
            url: Git repository URL.
            dest: Destination path inside the container (e.g. "/opt/repo").
            branch: Optional branch/tag to checkout.
        """
        new = self._clone()
        new._git_clones.append((url, dest, branch))
        return new

    def prepare_git_clones(self, workdir: Path) -> Image:
        """Clone git repos on the host and convert to %files copies.

        Called by the build system before generating the def file. Returns a
        new Image with _git_clones resolved into _copies.

        Args:
            workdir: Host directory to clone repos into.
        """
        import subprocess

        if not self._git_clones:
            return self

        new = self._clone()
        workdir.mkdir(parents=True, exist_ok=True)

        resolved = []
        for url, dest, branch in new._git_clones:
            repo_name = url.rstrip("/").split("/")[-1].replace(".git", "")
            host_path = workdir / repo_name

            cmd = ["git", "clone"]
            if branch:
                cmd.extend(["-b", branch])
            cmd.extend([url, str(host_path)])

            print(f"  Pre-cloning {url} -> {host_path}")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as exc:
                requirement = self._labels.get(
                    "BuildAccess", "Check repository access and network connectivity."
                )
                raise RuntimeError(
                    f"Cannot fetch container source {url}. {requirement}"
                ) from exc
            resolved.append((str(host_path), dest))

        new._copies = resolved + new._copies
        new._git_clones = []
        return new

    def runscript(self, cmd: str) -> Image:
        """Set the container's runscript/entrypoint."""
        new = self._clone()
        new._runscript = cmd
        return new

    def template_command(
        self,
        exec_mode: str = "run",
        binds: list[tuple[str, str]]
        | Callable[[], list[tuple[str, str]]]
        | None = None,
        env: dict[str, str | Callable[[], str]] | None = None,
    ) -> Image:
        """Configure default container command parameters for this container.

        Env values and binds may be zero-arg callables, evaluated at command creation.
        Missing default mounts on buildable recipes are optional site paths;
        explicit per-call mounts and existing-image mounts remain required.
        """
        new = self._clone()
        new._command_config = {
            "exec_mode": exec_mode,
            "binds": binds or [],
            "env": env or {},
        }
        return new

    def build_command(
        self,
        *args: str,
        exec_mode: str | None = None,
        binds: list[tuple[str, str]] | None = None,
        env: dict[str, str] | None = None,
        gpu: bool | None = None,
    ) -> list[str]:
        """Build container command for this container.

        Uses template_command() defaults if set, with per-call overrides.
        Generates apptainer or docker commands based on CONTAINER_BACKEND setting.
        """
        from chaperone import get_settings

        from chaperone.orchestration.modal_backend import current_command_context

        cloud_context = current_command_context()
        if cloud_context is not None:
            return cloud_context.build_command(self, args, env, binds, exec_mode)

        config = self._command_config or {}
        eff_exec_mode = exec_mode or config.get("exec_mode", "run")
        default_binds = config.get("binds", [])
        if callable(default_binds):
            default_binds = default_binds()
        if not self.is_existing:
            default_binds = [
                (src, dst) for src, dst in default_binds if Path(src).exists()
            ]
        all_binds = default_binds + (binds or [])
        all_env = {**config.get("env", {}), **(env or {})}
        resolved = {}
        for k, v in all_env.items():
            if callable(v):
                try:
                    resolved[k] = v()
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to resolve env var '{k}' (lazy callable): {e}"
                    ) from e
            else:
                resolved[k] = v
        all_env = resolved
        wants_gpu = _RUNTIME_GPU.get() if gpu is None else gpu
        if all_env.get("CUDA_VISIBLE_DEVICES") in ("", "-1"):
            wants_gpu = False

        # Resolve image reference
        if self._existing_path:
            image_ref = str(self._existing_path)
        else:
            image_ref = f"${{CONTAINER:{self.name}}}"

        settings = get_settings()
        for runtime_bind in (
            _runtime_rundir_bind(settings),
            _runtime_workspace_bind(settings),
        ):
            if runtime_bind is not None:
                for alias in _runtime_bind_aliases(runtime_bind):
                    if alias not in all_binds:
                        all_binds.append(alias)
        for script_bind in _runtime_script_binds(settings, args):
            if script_bind not in all_binds:
                all_binds.append(script_bind)

        backend = settings.CHAPERONE_CONTAINER_BACKEND

        if backend == "docker":
            cmd = ["docker", "run", "--rm"]
            for src, dst in all_binds:
                cmd.extend(["-v", f"{src}:{dst}"])
            for key, val in all_env.items():
                cmd.extend(["-e", f"{key}={val}"])
            cmd.append(image_ref)
        else:
            resolver_bind = _runtime_resolver_bind(all_binds)
            if resolver_bind is not None:
                all_binds.append(resolver_bind)
            private_namespace = _supports_private_mount_namespace()
            apptainer_cmd = [
                "apptainer",
                eff_exec_mode,
                "--cleanenv",
                "--containall",
                "--no-mount",
                "home,tmp,bind-paths",
                "--writable-tmpfs",
                "--pid",
            ]
            if private_namespace:
                cmd = [
                    "unshare",
                    "--mount",
                    "--propagation",
                    "private",
                    "--fork",
                    "--",
                    *apptainer_cmd,
                ]
            else:
                logger.debug(
                    "Private mount namespaces are unavailable; launching "
                    "Apptainer directly"
                )
                cmd = apptainer_cmd
                # ``--containall --no-mount bind-paths`` deliberately removes
                # Apptainer's implicit host mounts.  Cluster sandboxes and
                # most tool images live under /projects, so direct fallback
                # must restore both path spellings used by existing tools.
                # The private-namespace wrapper provides these mounts itself.
                destinations = {dst for _, dst in all_binds}
                if Path("/projects").is_dir() and "/projects" not in destinations:
                    all_binds.append(("/projects", "/projects"))
                if Path("/projects").is_dir() and "/mnt/projects" not in destinations:
                    all_binds.append(("/projects", "/mnt/projects"))
            for src, dst in all_binds:
                cmd.extend(["--bind", f"{src}:{dst}"])
            for key, val in all_env.items():
                cmd.extend(["--env", f"{key}={val}"])
            cmd.append(image_ref)

        cmd = refresh_container_gpu_options(cmd, wants_gpu)
        cmd.extend(args)
        return cmd

    @classmethod
    def from_existing(cls, name: str, path: str | Path) -> "Image":
        """Reference an existing container without rebuilding."""
        img = cls(name=name)
        img._existing_path = Path(path)
        return img

    @property
    def is_existing(self) -> bool:
        return self._existing_path is not None

    def get_image_path(self) -> Path | str:
        """Find the built container image.

        Returns:
            Path to the .sif file (apptainer) or image tag string (docker)

        Raises:
            FileNotFoundError: If container not found in any search directory
        """
        from chaperone import get_settings

        if self._existing_path:
            if self._existing_path.exists():
                return self._existing_path
            raise FileNotFoundError(f"Container not found: {self._existing_path}")

        backend = get_settings().CHAPERONE_CONTAINER_BACKEND

        if backend == "docker":
            import subprocess

            tag = f"chaperone/{self.name}:latest"
            result = subprocess.run(
                ["docker", "image", "inspect", tag],
                capture_output=True,
            )
            if result.returncode == 0:
                return tag
            raise FileNotFoundError(
                f"Docker image '{tag}' not found.\n"
                f"Run 'chaperone install {self.name}' to build it."
            )

        for d in get_container_dirs():
            sif = d / f"{self.name}.sif"
            if sif.exists():
                return sif

        search_paths = "\n  ".join(str(d) for d in get_container_dirs())
        raise FileNotFoundError(
            f"Container '{self.name}' not found. Searched:\n  {search_paths}\n"
            f"Run 'chaperone install {self.name}' to build it."
        )

    def _build_stage(self, *, docker: bool = False) -> hpccm.Stage:
        """Build HPCCM Stage from this spec."""
        if not self._base_image:
            raise ValueError(
                "Base image not set. Use .base_image('ubuntu:24.04') when building the image."
            )
        stage = hpccm.Stage()
        stage += baseimage(image=self._base_image)

        if self._apt_packages:
            stage += packages(apt=self._apt_packages)

        for src, dst in self._copies:
            stage += copy(src=src, dest=dst)

        # Emit RUN git clone for unresolved git_clones (not yet converted
        # to COPY via prepare_git_clones).
        for url, dest, branch in self._git_clones:
            clone_cmd = "git clone"
            if branch:
                clone_cmd += f" -b {branch}"
            clone_cmd += f" {url} {dest}"
            stage += shell(commands=[clone_cmd])

        for cmd_group in self._commands:
            if docker and any("\n" in command for command in cmd_group):
                # A quoted Python/shell program may contain literal newlines.
                # Docker still treats those as instruction boundaries in the
                # shell form; JSON form preserves them inside a single RUN.
                stage += raw(
                    docker="RUN "
                    + json.dumps(["/bin/sh", "-c", " && ".join(cmd_group)])
                )
            else:
                stage += shell(commands=cmd_group)

        if self._env_vars:
            stage += environment(variables=self._env_vars)

        if self._labels:
            # HPCCM emits label values verbatim. Docker needs quoting for
            # spaces/quotes and literal dollars; Apptainer keeps its text form.
            metadata = {
                key: json.dumps(str(value)).replace("$", r"\$") if docker else value
                for key, value in self._labels.items()
            }
            stage += label(metadata=metadata)

        if self._runscript:
            stage += raw(
                singularity=f"%runscript\n    exec {self._runscript}", docker=""
            )

        return stage

    def to_dockerfile(self) -> str:
        """Generate Dockerfile."""
        if self._existing_path:
            raise ValueError("Cannot generate Dockerfile for existing container")
        stage = self._build_stage(docker=True)
        hpccm.config.set_container_format("docker")
        return str(stage)

    def to_singularity(self) -> str:
        """Generate Singularity/Apptainer definition."""
        if self._existing_path:
            raise ValueError("Cannot generate definition for existing container")
        stage = self._build_stage()
        hpccm.config.set_container_format("singularity")
        return str(stage)

    def __repr__(self) -> str:
        if self._existing_path:
            return f"Image({self.name!r}, existing={self._existing_path})"
        return f"Image({self.name!r}, base={self._base_image}, commands={len(self._commands)})"
