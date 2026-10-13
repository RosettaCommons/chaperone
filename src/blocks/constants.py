"""
Useful if you want a fixed place for constant names
e.g. sharing column names across functions in the repo.
"""

import logging

from atomworks.constants import (
    STANDARD_AA,
    STANDARD_DNA,
    STANDARD_RNA,
)

logger = logging.getLogger(__name__)

###########################################################################
# Container constants
###########################################################################


# Common CUDA installation commands for Ubuntu 24.04.
# 12-8 is the default so sm_120 (Blackwell, e.g. RTX Pro b4000) is supported;
# lower versions stay callable via `make_cuda_install("12-6")`.
def make_cuda_install(version: str = "12-8") -> list[str]:
    """HPCCM run-steps that install ``cuda-toolkit-<version>`` on Ubuntu 24.04."""
    return [
        "mkdir -p /etc/apt/keyrings",
        "curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub | gpg --dearmor -o /etc/apt/keyrings/nvidia-cuda.gpg",
        'echo "deb [signed-by=/etc/apt/keyrings/nvidia-cuda.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64 /" > /etc/apt/sources.list.d/cuda.list',
        f"apt-get update && apt-get install -y cuda-toolkit-{version}",
    ]


CUDA_INSTALL = make_cuda_install("12-8")


# Default apt packages shared by the CUDA + venv python image prelude
# (the python interpreter, its ``-venv`` package and ``python3-pip`` are
# appended after these by :func:`cuda_python_base`).
CUDA_PYTHON_BASE_PACKAGES: tuple[str, ...] = (
    "build-essential",
    "git",
    "curl",
    "ca-certificates",
    "gnupg",
)


def cuda_python_base(
    image,
    env_name: str,
    *,
    python: str = "python3.12",
    packages: tuple[str, ...] | list[str] | None = None,
    create_venv: bool = True,
    base_image: str = "ubuntu:24.04",
    cuda_install: list[str] | None = None,
):
    """Apply the shared ubuntu -> packages -> CUDA -> venv -> env prelude.

    Collapses the repeated container-builder skeleton used by the CUDA python
    blocks (esm2, e1, boltzgen, disco, protein_hunter, ...). Given a freshly
    constructed ``Image("name")``, returns an ``Image`` with:

    - ``base_image(base_image)``
    - ``packages(*packages)`` (defaults to :data:`CUDA_PYTHON_BASE_PACKAGES`
      plus the requested ``python`` and its ``-venv`` package)
    - ``run(*cuda_install)`` (defaults to :data:`CUDA_INSTALL`)
    - if ``create_venv``: ``run(<python> -m venv /opt/<env_name>-env``,
      pip upgrade) and the canonical CUDA/PATH/LD_LIBRARY_PATH/VIRTUAL_ENV env

    Callers chain their tool-specific ``run``/``git_clone``/extra ``env``/
    ``label``/``runscript``/``template_command`` steps onto the result. The
    produced definition is equivalent to the previous inline builders.

    Args:
        image: A constructed ``Image`` instance (e.g. ``Image("esm2")``).
        env_name: Logical env name; the venv is created at ``/opt/<env_name>-env``.
        python: Python interpreter package/binary (e.g. ``"python3.12"``).
        packages: Override the apt package list entirely. When ``None`` the
            shared base packages plus ``python`` and ``<python>-venv`` are used.
        create_venv: When ``False``, skip venv creation and the env block so the
            caller can manage its own environment (e.g. a uv-managed venv).
        base_image: Base image string.
        cuda_install: CUDA install run-steps (defaults to :data:`CUDA_INSTALL`).
    """
    if packages is None:
        packages = list(CUDA_PYTHON_BASE_PACKAGES) + [
            python,
            f"{python}-venv",
            "python3-pip",
        ]

    img = image.base_image(base_image).packages(*packages)
    steps = cuda_install if cuda_install is not None else CUDA_INSTALL
    if steps:
        img = img.run(*steps)

    if not create_venv:
        return img

    venv = f"/opt/{env_name}-env"
    img = img.run(
        f"{python} -m venv {venv}",
        f". {venv}/bin/activate && pip install --upgrade pip setuptools wheel",
    )
    return cuda_venv_env(img, venv)


def cuda_venv_env(image, venv: str):
    """Apply the canonical CUDA + venv environment variables to ``image``.

    Shared tail of the CUDA python image prelude: ``CUDA_HOME``, ``PATH``
    (venv bin + cuda bin), ``LD_LIBRARY_PATH`` (cuda lib64) and ``VIRTUAL_ENV``.

    Args:
        image: A constructed ``Image`` instance.
        venv: Absolute path to the venv (e.g. ``/opt/foo-env``).
    """
    return image.env(
        CUDA_HOME="/usr/local/cuda",
        PATH=f"{venv}/bin:/usr/local/cuda/bin:$PATH",
        LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH",
        VIRTUAL_ENV=venv,
    )


###########################################################################
# Chaperone constants
###########################################################################

TASK_COLUMN_NAME = "task"
DNA_TWO_TO_ONE = {
    "DA": "A",
    "DT": "T",
    "DG": "G",
    "DC": "C",
}

STANDARD_CA_DIST = 3.8
"""Standard distance between CA atoms of adjacent residues. Used for detecting chainbreaks."""

TIP_BY_RESTYPE = {
    "TRP": [
        "CG",
        "CD1",
        "CD2",
        "NE1",
        "CE2",
        "CE3",
        "CZ2",
        "CZ3",
        "CH2",
    ],  # fix both rings
    "HIS": ["CG", "ND1", "CD2", "CE1", "NE2"],  # fixed ring
    "TYR": ["CZ", "OH"],  # keeps ring dihedral flexible
    "PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "ASN": ["CB", "CG", "OD1", "ND2"],
    "ASP": ["CB", "CG", "OD1", "OD2"],
    "GLN": ["CG", "CD", "OE1", "NE2"],
    "GLU": ["CG", "CD", "OE1", "OE2"],
    "CYS": ["CB", "SG"],
    "SER": ["CB", "OG"],
    "THR": ["CB", "OG1"],
    "LEU": ["CB", "CG", "CD1", "CD2"],
    "VAL": ["CG1", "CG2"],
    "ILE": ["CB", "CG2"],
    "MET": ["SD", "CE"],
    "LYS": ["CE", "NZ"],
    "ARG": ["CD", "NE", "CZ", "NH1", "NH2"],
    "PRO": None,
    "ALA": None,
    "GLY": None,
    "UNK": None,
    "MSK": None,
}

AROMATIC_RESIDUES = ["PHE", "TYR", "TRP", "HIS"]

NON_LIGAND_NAMES = (
    list(STANDARD_AA) + list(STANDARD_RNA) + list(STANDARD_DNA) + ["UNK"]
)  # + list(AF3_EXCLUDED_LIGANDS) + list(CRYSTALLIZATION_AIDS)

AF3_RELEASE_LINES = [
    "# \n",
    "loop_\n",
    "_pdbx_audit_revision_history.ordinal \n",
    "_pdbx_audit_revision_history.data_content_type \n",
    "_pdbx_audit_revision_history.major_revision \n",
    "_pdbx_audit_revision_history.minor_revision \n",
    "_pdbx_audit_revision_history.revision_date \n",
    "1 'Structure model' 1 0 2004-07-13 \n",
    "2 'Structure model' 1 1 2011-06-14 \n",
    "3 'Structure model' 1 2 2011-07-13 \n",
    "4 'Structure model' 1 3 2011-07-27 \n",
    "5 'Structure model' 1 4 2012-12-12 \n",
    "6 'Structure model' 2 0 2023-08-23 \n",
    "# \n",
    "_diffrn_detector.diffrn_id              1 \n",
    "_diffrn_detector.detector               CCD \n",
    "_diffrn_detector.type                   'ADSC QUANTUM 4' \n",
    "_diffrn_detector.pdbx_collection_date   2003-05-05 \n",
    "_diffrn_detector.details                ? \n",
    "# \n",
    "_pdbx_database_status.status_code                     REL \n",
    "_pdbx_database_status.entry_id                        1T7D \n",
    "_pdbx_database_status.recvd_initial_deposition_date   2004-05-09 \n",
    "_pdbx_database_status.deposit_site                    RCSB \n",
    "_pdbx_database_status.process_site                    RCSB \n",
    "_pdbx_database_status.status_code_sf                  REL \n",
    "_pdbx_database_status.SG_entry                        . \n",
    "_pdbx_database_status.status_code_mr                  ? \n",
    "_pdbx_database_status.status_code_cs                  ? \n",
    "_pdbx_database_status.pdb_format_compatible           Y \n",
    "_pdbx_database_status.status_code_nmr_data            ? \n",
    "_pdbx_database_status.methods_development_category    ? \n",
    "# ",
]
