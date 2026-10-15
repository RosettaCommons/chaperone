"""Pinned, portable RFO runtime, isolated from the shared Foundry image."""

import shlex
from pathlib import Path

from chaperone.containers import Image
from blocks.networks.mpnn import (
    _MPNN_CHECKPOINTS,
    _MPNN_REVISION,
    _MPNN_NUMPY_COMPATIBILITY_PATCH,
)

from blocks.networks.foundry.container import _FOUNDRY_REVISION as FOUNDRY_REVISION

# RFO needs independent environments: Boltz's NumPy constraint conflicts with
# current Foundry/AtomWorks. No external installations or nested containers.
RFOPTIMIZATION_CONTAINER = (
    Image("rfoptimization")
    .base_image("ubuntu:24.04")
    .copy(
        str(Path(__file__).parents[1] / "_aw3_compat.py"),
        "/opt/chaperone-compat/_aw3_compat.py",
    )
    .copy(
        str(Path(__file__).parents[1] / "atomworks3.patch"),
        "/opt/chaperone-compat/atomworks3.patch",
    )
    .packages(
        "build-essential",
        "git",
        "wget",
        "ca-certificates",
        "python3.12",
        "python3.12-dev",
        "python3.12-venv",
    )
    .run(
        "git clone https://github.com/RosettaCommons/foundry.git /opt/foundry",
        "git -C /opt/foundry fetch origin refs/pull/404/head",
        f"git -C /opt/foundry checkout --detach {FOUNDRY_REVISION}",
        "python3.12 -m venv /opt/rfo-env",
        "/opt/rfo-env/bin/pip install --upgrade pip setuptools wheel",
        "/opt/rfo-env/bin/pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128",
        "/opt/rfo-env/bin/pip install -e '/opt/foundry[rf3,rfo]' 'atomworks[ml]==3.0.1' biotite==1.6.0 numpy==2.4.3 lightning==2.6.1 torch==2.11.0",
        "python3.12 -m venv /opt/boltz-env",
        "/opt/boltz-env/bin/pip install --upgrade pip setuptools wheel",
        "/opt/boltz-env/bin/pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128",
        "/opt/boltz-env/bin/pip install 'boltz[cuda]==2.2.1' numpy==1.26.4 torch==2.11.0",
        "python3.12 -m venv /opt/mpnn-env",
        "/opt/mpnn-env/bin/pip install --upgrade pip setuptools wheel",
        "/opt/mpnn-env/bin/pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128",
        "/opt/mpnn-env/bin/pip install numpy==1.26.4 prody==2.6.1 scipy biopython ml-collections dm-tree",
        "git clone https://github.com/dauparas/LigandMPNN.git /opt/ligandmpnn",
        f"git -C /opt/ligandmpnn checkout --detach {_MPNN_REVISION}",
        "/opt/mpnn-env/bin/python -c " + shlex.quote(_MPNN_NUMPY_COMPATIBILITY_PATCH),
        "mkdir -p /opt/ligandmpnn/model_params",
        *[
            "wget --timeout=30 --tries=3 -q "
            "https://huggingface.co/zcpbx/ligandmpnn-weights/resolve/"
            f"717f35008c45f6dd384a5e6cb0caab3ce99f28fc/model_params/{filename} "
            f"-O /opt/ligandmpnn/model_params/{filename} && "
            f"echo '{digest}  /opt/ligandmpnn/model_params/{filename}' | sha256sum -c -"
            for filename, digest in (
                _MPNN_CHECKPOINTS["protein_mpnn"][0.20],
                _MPNN_CHECKPOINTS["ligand_mpnn"][0.10],
            )
        ],
        "/opt/rfo-env/bin/pip check",
        "/opt/boltz-env/bin/pip check",
        "/opt/mpnn-env/bin/pip check",
        "/opt/rfo-env/bin/python /opt/chaperone-compat/_aw3_compat.py /opt/foundry /opt/chaperone-compat/atomworks3.patch --check",
        "/opt/rfo-env/bin/rfo --help >/dev/null",
        "/opt/rfo-env/bin/python -c "
        + shlex.quote(
            "from importlib.metadata import version; "
            'assert version("rc-foundry") == "1.0.0"; '
            'assert version("atomworks") == "3.0.1"; '
            'assert version("biotite") == "1.6.0"'
        ),
        "/opt/rfo-env/bin/python -c 'import rfo.backprop.optimize; import rf3.inference_engines.rf3; from rfo.backprop.tools import af3_auto_detect_breaking, af3_no_breaking_point'",
        "/opt/boltz-env/bin/boltz predict --help >/dev/null",
        "/opt/mpnn-env/bin/python /opt/ligandmpnn/run.py --help >/dev/null",
        "apt-get clean && rm -rf /var/lib/apt/lists/* /root/.cache/pip",
    )
    .env(PATH="/opt/rfo-env/bin:$PATH", VIRTUAL_ENV="/opt/rfo-env")
    .label(
        Foundry=FOUNDRY_REVISION,
        AtomWorks="3.0.1",
        Boltz="2.2.1",
        LigandMPNN=_MPNN_REVISION,
    )
    .runscript('"$@"')
    # RF3's gradient path hits an incompatible cuEquivariance triangle kernel
    # with the public checkpoint. Use Foundry's supported PyTorch fallback.
    .template_command(
        exec_mode="exec",
        env={"DISABLE_CUEQUIVARIANCE": "1", "PYTHON_DOTENV_DISABLED": "1"},
    )
)
