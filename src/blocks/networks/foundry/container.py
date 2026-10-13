"""Foundry container shared by RFD3, RF3, and FoundryMPNN blocks.

Auto-selects between CUDA (Linux/GPU) and CPU-only (Apple Silicon) variants.
On ARM64 Macs, the CUDA container can't be built so the public container is
used instead. RFD3 runs on MPS; MPNN is forced to CPU; RF3 is not yet
supported on Apple Silicon (MPS MLIR bug).
"""

from pathlib import Path

from chaperone import get_setting_field
from chaperone.common import is_apple_silicon
from chaperone.containers import Image
from blocks.constants import CUDA_INSTALL


def _foundry_binds() -> list[tuple[str, str]]:
    """Bind list for the foundry container.

    foundry.sif hardcodes /mnt/projects for its CCD mirror, so on hosts where
    that path does not exist we alias /projects onto it. But when the host has
    a real /mnt/projects — as managed workers do, with workspaces living
    under it — that alias shadows the workspace: a path inside the container
    resolves to a different inode than the same path outside, and blocks fail
    with FileNotFoundError on files that plainly exist. So bind the real
    directory through instead, which satisfies the CCD mirror either way.
    """
    binds = [("/mnt/home", "/mnt/home"), ("/projects", "/projects")]
    # Active run and runtime-workspace paths are injected centrally by
    # Image.build_command when the controller enables the exact-path options.
    # Never restore broad /net or /mnt/net mounts here.
    alias, target = Path("/mnt/projects"), Path("/projects")
    # Compare device+inode rather than testing for a symlink: a bind-mounted
    # /mnt/projects is a real directory too, and would still shadow.
    try:
        distinct = alias.is_dir() and alias.stat()[:2] != target.stat()[:2]
    except OSError:
        distinct = False
    binds.append(
        ("/mnt/projects", "/mnt/projects")
        if distinct
        else ("/projects", "/mnt/projects")
    )
    return binds


# ── CUDA variant (default for Linux / GPU clusters) ─────────────────────

_FOUNDRY_CUDA_CONTAINER = (
    Image("foundry")
    .base_image("ubuntu:24.04")
    .packages(
        "build-essential",
        "git",
        "wget",
        "curl",
        "ca-certificates",
        "gnupg",
        "python3.12",
        "python3.12-dev",
        "python3.12-venv",
        "python3-pip",
        "libaio-dev",
    )
    .run(*CUDA_INSTALL)
    .run(
        "python3.12 -m venv /opt/foundry-env",
        ". /opt/foundry-env/bin/activate && pip install --upgrade pip setuptools wheel",
    )
    .run(
        "cd /opt && git clone -b production https://github.com/RosettaCommons/foundry.git",
        ". /opt/foundry-env/bin/activate && cd /opt/foundry && pip install -e '.[all]'",
        ". /opt/foundry-env/bin/activate && rfd3na --help >/dev/null",
    )
    .run(
        "cd /opt && git clone https://github.com/NVIDIA/apex.git",
        ". /opt/foundry-env/bin/activate && cd /opt/apex && "
        'TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0" '
        "pip install -v --no-cache-dir --no-build-isolation "
        '--config-settings "--build-option=--cpp_ext" '
        '--config-settings "--build-option=--cuda_ext" .',
    )
    .run("apt-get clean && rm -rf /var/lib/apt/lists/*")
    .env(
        CUDA_HOME="/usr/local/cuda",
        PATH="/opt/foundry-env/bin:/usr/local/cuda/bin:$PATH",
        LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH",
        VIRTUAL_ENV="/opt/foundry-env",
    )
    .label(Version="v1.0.0", CUDA="12.6")
    .runscript('"$@"')
    .template_command(
        exec_mode="exec",
        binds=_foundry_binds(),
        env={
            "FOUNDRY_CHECKPOINT_DIRS": lambda: get_setting_field(
                "FOUNDRY_CHECKPOINT_DIRS", ""
            )
        },
    )
)


# ── CPU/MPS variant (Apple Silicon / non-CUDA environments) ─────────────

_FOUNDRY_PUBLIC_CONTAINER = (
    Image("foundry_public")
    .base_image("ubuntu:24.04")
    .packages(
        "build-essential",
        "git",
        "wget",
        "curl",
        "ca-certificates",
        "python3.12",
        "python3.12-dev",
        "python3.12-venv",
        "python3-pip",
    )
    .run(
        "python3.12 -m venv /opt/foundry-env",
        ". /opt/foundry-env/bin/activate && pip install --upgrade pip setuptools wheel",
    )
    .run(
        "cd /opt && git clone -b production https://github.com/RosettaCommons/foundry.git",
        ". /opt/foundry-env/bin/activate && cd /opt/foundry && pip install -e '.[all]'",
        ". /opt/foundry-env/bin/activate && rfd3na --help >/dev/null",
    )
    .run("apt-get clean && rm -rf /var/lib/apt/lists/*")
    .env(
        PATH="/opt/foundry-env/bin:$PATH",
        VIRTUAL_ENV="/opt/foundry-env",
    )
    .label(Version="v1.0.0", Platform="cpu")
    .runscript('"$@"')
    .template_command(
        exec_mode="exec",
        env={
            "FOUNDRY_CHECKPOINT_DIRS": lambda: get_setting_field(
                "FOUNDRY_CHECKPOINT_DIRS", ""
            )
        },
    )
)


# ── Auto-select ─────────────────────────────────────────────────────────
# On Apple Silicon use the public CPU variant; everywhere else use CUDA.

FOUNDRY_CONTAINER = (
    _FOUNDRY_CUDA_CONTAINER
    if not is_apple_silicon()
    else _FOUNDRY_PUBLIC_CONTAINER
)
FOUNDRY_PUBLIC_CONTAINER = _FOUNDRY_PUBLIC_CONTAINER
