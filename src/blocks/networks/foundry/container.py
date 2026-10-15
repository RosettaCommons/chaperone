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


_FOUNDRY_REVISION = "54d188b3c0c27654a01c875b173148a508fcd858"
# Pin the upstream AtomWorks 3 port together with its supported Biotite version.
_FOUNDRY_DEPENDENCIES = (
    "'atomworks[ml]==3.0.1' 'biotite==1.6.0' 'lightning==2.6.1' 'numpy==2.4.3'"
)
_FOUNDRY_IMPORT_CHECK = (
    "/opt/foundry-env/bin/python -c 'from importlib.metadata import version; "
    'assert version("rc-foundry") == "1.0.0"; '
    'assert version("atomworks") == "3.0.1"; '
    'assert version("biotite") == "1.6.0"; '
    "import rfd3.engine; import rfd3na.engine; "
    "import rf3.inference_engines.rf3; import mpnn.inference_engines.mpnn'"
)

# Exercise native upstream inference contracts without modifying installed source.
_FOUNDRY_INFERENCE_CHECK = (
    "/opt/foundry-env/bin/python -c '"
    "from rfd3na.inference.input_parsing import DesignInputSpecification; "
    "from biotite.structure import get_residue_starts; "
    "from rf3.inference_engines.rf3 import RF3InferenceEngine; "
    'atoms = DesignInputSpecification(contig="2,/0,3R,/0,4D", length="9").build(); '
    "assert len(get_residue_starts(atoms)) == 9; "
    'assert RF3InferenceEngine(ckpt_path="unused.ckpt").transform_overrides["residue_cache_dir"] is None'
    "'"
)


def default_checkpoint(model: str, filename: str) -> str:
    """Use a configured installed checkpoint or Foundry's public model registry."""
    directories = get_setting_field("FOUNDRY_CHECKPOINT_DIRS", "") or ""
    for directory in directories.split(":"):
        if directory and (path := Path(directory).expanduser() / filename).is_file():
            return str(path.resolve())
    return model


def _checkpoint_binds() -> list[tuple[str, str]]:
    directories = get_setting_field("FOUNDRY_CHECKPOINT_DIRS", "") or ""
    paths = [Path(p).expanduser().resolve() for p in directories.split(":") if p]
    paths.append(Path.home() / ".foundry")
    return [(str(p), str(p)) for p in dict.fromkeys(paths) if p.is_dir()]


def _foundry_binds() -> list[tuple[str, str]]:
    """Mount configured checkpoints; the engine supplies workspace/run mounts."""
    return _checkpoint_binds()


# ── CUDA variant (default for Linux / GPU clusters) ─────────────────────

_FOUNDRY_CUDA_CONTAINER = (
    Image("foundry")
    .base_image("ubuntu:24.04")
    .packages(
        "build-essential",
        "ninja-build",
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
        # Match CUDA12 RF3 operators and the toolkit used to compile Apex.
        ". /opt/foundry-env/bin/activate && pip install torch==2.11.0 "
        "--index-url https://download.pytorch.org/whl/cu128",
    )
    .run(
        "cd /opt && git clone -b production https://github.com/RosettaCommons/foundry.git",
        "git -C /opt/foundry fetch origin refs/pull/404/head",
        f"git -C /opt/foundry checkout --detach {_FOUNDRY_REVISION}",
        ". /opt/foundry-env/bin/activate && cd /opt/foundry && pip install -e '.[all]' "
        + _FOUNDRY_DEPENDENCIES,
        ". /opt/foundry-env/bin/activate && rfd3na --help >/dev/null",
    )
    .run(
        "cd /opt && git clone https://github.com/NVIDIA/apex.git",
        ". /opt/foundry-env/bin/activate && cd /opt/apex && "
        'MAX_JOBS=4 TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;12.0+PTX" '
        "pip install -v --no-cache-dir --no-build-isolation "
        '--config-settings "--build-option=--cpp_ext" '
        '--config-settings "--build-option=--cuda_ext" .',
    )
    .run(
        ". /opt/foundry-env/bin/activate && pip check",
        "/opt/foundry-env/bin/python -c 'import torch; import apex; "
        'import fused_layer_norm_cuda; assert torch.version.cuda == "12.8"; '
        "print(torch.__version__, torch.version.cuda)'",
        _FOUNDRY_IMPORT_CHECK,
        _FOUNDRY_INFERENCE_CHECK,
    )
    .run("apt-get clean && rm -rf /var/lib/apt/lists/*")
    .env(
        CUDA_HOME="/usr/local/cuda",
        PATH="/opt/foundry-env/bin:/usr/local/cuda/bin:$PATH",
        LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH",
        VIRTUAL_ENV="/opt/foundry-env",
    )
    .label(
        RF3ResidueCache="disabled-for-inference",
        Foundry="1.0.0",
        Version=_FOUNDRY_REVISION,
        CUDA="12.8",
        PyTorch="2.11.0",
        AtomWorks="3.0.1",
    )
    .runscript('"$@"')
    .template_command(
        exec_mode="exec",
        binds=_foundry_binds,
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
        "ninja-build",
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
        ". /opt/foundry-env/bin/activate && pip install torch==2.11.0 "
        "--index-url https://download.pytorch.org/whl/cpu",
    )
    .run(
        "cd /opt && git clone -b production https://github.com/RosettaCommons/foundry.git",
        "git -C /opt/foundry fetch origin refs/pull/404/head",
        f"git -C /opt/foundry checkout --detach {_FOUNDRY_REVISION}",
        ". /opt/foundry-env/bin/activate && cd /opt/foundry && pip install -e '.[all]' "
        + _FOUNDRY_DEPENDENCIES,
        ". /opt/foundry-env/bin/activate && rfd3na --help >/dev/null",
    )
    .run(
        ". /opt/foundry-env/bin/activate && pip check",
        _FOUNDRY_IMPORT_CHECK,
        _FOUNDRY_INFERENCE_CHECK,
    )
    .run("apt-get clean && rm -rf /var/lib/apt/lists/*")
    .env(
        PATH="/opt/foundry-env/bin:$PATH",
        VIRTUAL_ENV="/opt/foundry-env",
    )
    .label(
        RF3ResidueCache="disabled-for-inference",
        Foundry="1.0.0",
        Version=_FOUNDRY_REVISION,
        Platform="cpu",
        PyTorch="2.11.0",
        AtomWorks="3.0.1",
    )
    .runscript('"$@"')
    .template_command(
        exec_mode="exec",
        binds=_checkpoint_binds,
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
    _FOUNDRY_CUDA_CONTAINER if not is_apple_silicon() else _FOUNDRY_PUBLIC_CONTAINER
)
FOUNDRY_PUBLIC_CONTAINER = _FOUNDRY_PUBLIC_CONTAINER
