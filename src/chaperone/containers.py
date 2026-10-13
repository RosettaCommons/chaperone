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
from dataclasses import dataclass, field
from functools import cache
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
        "/net",
        "/net/scratch",
        "/projects",
        "/tmp",
    )
)


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


def _runtime_rundir_bind(settings: object) -> tuple[str, str] | None:
    """Return the opt-in, exact run-directory bind for contained tools."""
    if getattr(settings, "CHAPERONE_CONTAINER_BIND_RUNDIR", False) is not True:
        return None
    configured = getattr(settings, "RUNDIR", None)
    if configured is None:
        raise ValueError(
            "CHAPERONE_CONTAINER_BIND_RUNDIR requires an active absolute RUNDIR"
        )
    configured_path = Path(configured).expanduser()
    if not configured_path.is_absolute():
        raise ValueError(
            "CHAPERONE_CONTAINER_BIND_RUNDIR requires an active absolute RUNDIR"
        )
    rundir = configured_path.absolute()
    if rundir in _UNSAFE_RUNDIR_BINDS:
        raise ValueError(f"Refusing broad nested-container RUNDIR bind: {rundir}")
    spelling = str(rundir)
    return spelling, spelling


def _runtime_workspace_bind(settings: object) -> tuple[str, str] | None:
    """Return the opt-in, exact workspace bind provided by the runtime."""
    if getattr(settings, "CHAPERONE_CONTAINER_BIND_WORKSPACE", False) is not True:
        return None
    configured = os.environ.get("HARBOR_HOST_WORKSPACE")
    if not configured:
        raise ValueError(
            "CHAPERONE_CONTAINER_BIND_WORKSPACE requires HARBOR_HOST_WORKSPACE"
        )
    workspace = Path(configured).expanduser()
    if not workspace.is_absolute():
        raise ValueError("HARBOR_HOST_WORKSPACE must be absolute")
    workspace = workspace.absolute()
    if workspace in _UNSAFE_RUNDIR_BINDS:
        raise ValueError(f"Refusing broad nested-container workspace bind: {workspace}")
    spelling = str(workspace)
    return spelling, spelling


def get_container_dirs() -> list[Path]:
    """Get container search directories in priority order."""
    dirs: list[Path] = []

    # Environment variable (highest priority)
    from chaperone import get_setting_field

    env_dirs = get_setting_field("CHAPERONE_CONTAINER_DIRS", default="")
    if env_dirs:
        dirs.extend(Path(p.strip()) for p in env_dirs.split(":") if p.strip())

    # Shared location
    shared = Path("/projects/ml/pipelines/containers")
    if shared.exists():
        dirs.append(shared)

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
            subprocess.run(cmd, check=True)
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
        binds: list[tuple[str, str]] | None = None,
        env: dict[str, str | Callable[[], str]] | None = None,
    ) -> Image:
        """Configure default container command parameters for this container.

        Env values may be strings or zero-arg callables (evaluated lazily in build_command).
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
    ) -> list[str]:
        """Build container command for this container.

        Uses template_command() defaults if set, with per-call overrides.
        Generates apptainer or docker commands based on CONTAINER_BACKEND setting.
        """
        from chaperone import get_settings

        config = self._command_config or {}
        eff_exec_mode = exec_mode or config.get("exec_mode", "run")
        all_binds = config.get("binds", []) + (binds or [])
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

        # On macOS dev boxes, filter out binds whose source path doesn't
        # exist on the host (e.g. /mnt/home, /projects). Linux production
        # paths must always be passed through unchanged so unit tests that
        # use synthetic /tmp paths still see -v flags in the command.
        from chaperone.common import is_macos

        if is_macos():
            valid_binds = []
            for src, dst in all_binds:
                if Path(src).exists():
                    valid_binds.append((src, dst))
                else:
                    import logging

                    logging.getLogger(__name__).debug(
                        "Skipping bind %s:%s (source does not exist)", src, dst
                    )
            all_binds = valid_binds

        # Resolve image reference
        if self._existing_path:
            image_ref = str(self._existing_path)
        else:
            image_ref = f"${{CONTAINER:{self.name}}}"

        settings = get_settings()
        runtime_bind = _runtime_rundir_bind(settings)
        if runtime_bind is not None and runtime_bind not in all_binds:
            # --containall deliberately hides ambient host mounts. Controllers
            # opt in to exposing only their exact run directory, not the
            # enclosing workspace or all of /net/scratch.
            all_binds.append(runtime_bind)
        workspace_bind = _runtime_workspace_bind(settings)
        if workspace_bind is not None and workspace_bind not in all_binds:
            all_binds.append(workspace_bind)

        backend = settings.CHAPERONE_CONTAINER_BACKEND

        if backend == "docker":
            cmd = ["docker", "run"]
            # --gpus requires NVIDIA Container Toolkit (Linux only)
            if not is_macos():
                cmd.extend(["--gpus", "all"])
            for src, dst in all_binds:
                cmd.extend(["-v", f"{src}:{dst}"])
            for key, val in all_env.items():
                cmd.extend(["-e", f"{key}={val}"])
            cmd.append(image_ref)
        else:
            private_namespace = _supports_private_mount_namespace()
            apptainer_cmd = [
                "apptainer",
                eff_exec_mode,
                "--nv",
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

    def _build_stage(self) -> hpccm.Stage:
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
            stage += shell(commands=cmd_group)

        if self._env_vars:
            stage += environment(variables=self._env_vars)

        if self._labels:
            stage += label(metadata=self._labels)

        if self._runscript:
            stage += raw(
                singularity=f"%runscript\n    exec {self._runscript}", docker=""
            )

        return stage

    def to_dockerfile(self) -> str:
        """Generate Dockerfile."""
        if self._existing_path:
            raise ValueError("Cannot generate Dockerfile for existing container")
        stage = self._build_stage()
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
