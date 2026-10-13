"""RFdiffusion2 (rf_diffusion) block for atomic-level protein design.

RFdiffusion2, also known as RFdiffusion-AA or rf_diffusion, provides atomic-level
protein design capabilities including:
- Enzyme design from atomic motifs and small molecules
- Small-molecule binder design with RASA conditioning
- Partial sequence design with unknown sequence positions
"""

import getpass
import json
import logging
from pathlib import Path
import os
import sys
import types

import numpy as np
from pydantic import Field

from chaperone import (
    CompiledBlockFunction,
    PipelineDataHandler,
    JobArgs,
)
from chaperone.block_config import TIER_PRIMARY
from chaperone.containers import Image
from chaperone.common import glob_designs_in_dir, tail_block_log
from chaperone.data import make_row
from typing import List
from chaperone.data.schema import DesignMetadata

logger = logging.getLogger(__name__)

# Two variants of the RFdiffusion2 container exist:
#
#   1. The prebuilt on-prem SIF. IPD jobs use this one; the RFdiffusion2 code
#      and checkpoints live on site storage and are bound in.
#
#   2. A portable buildable variant (`chaperone install rfdiffusion2_portable`)
#      that follows the upstream spec (RosettaCommons/RFdiffusion2,
#      rf_diffusion/exec/rf_diffusion_aa.spec) and clones the public repo into
#      /opt/RFdiffusion2 at build time, along with the RFD_173.pt checkpoint
#      (downloaded and sha256-checked during the build).
#
# We auto-select at import time: if the on-prem SIF exists on disk, prefer it
# (so IPD runs stay byte-identical to the past); otherwise fall back to the
# buildable variant.

_ONPREM_RFD2_SIF = (
    "/net/software/containers/versions/rf_diffusion_aa/24-05-21/rf_diffusion_aa.sif"
)
_ONPREM_RFD2_CODE = "/net/tukwila/rohith/RFdiffusion2"
_ONPREM_RFD2_CKPT = "/net/databases/lab/aa-diffusion/models/RFD_140_ORI_tipatom.pt"

_RFD2_SIF_CONTAINER = Image.from_existing(
    "rfdiffusion2", _ONPREM_RFD2_SIF
).template_command(
    binds=[
        (_ONPREM_RFD2_CODE, _ONPREM_RFD2_CODE),
        ("/net/databases/lab/aa-diffusion", "/net/databases/lab/aa-diffusion"),
    ]
)

# Upstream commit the portable recipe is pinned to.
RFD2_REPO_URL = "https://github.com/RosettaCommons/RFdiffusion2"
RFD2_COMMIT = "d365cbf4db3958814a9f8e4f6f94fa309dfebc2b"
_RFD2_CODE_DIR = "/opt/RFdiffusion2"
# The checkpoint upstream recommends (RosettaCommons/RFdiffusion2 docs, usage.rst),
# baked into the image. Upstream publishes no checksum; this one was computed
# from the served file.
_RFD2_CKPT_NAME = "RFD_173.pt"
_RFD2_CKPT_URL = (
    f"https://files.ipd.uw.edu/pub/rfdiffusion2/model_weights/{_RFD2_CKPT_NAME}"
)
_RFD2_CKPT_SHA256 = "590e126057f780afc1249d29545f0f90635562b1a3df5aff013afcfc39c3d3c3"
_RFD2_BAKED_CKPT = f"/opt/rfd2-weights/{_RFD2_CKPT_NAME}"

_RFD2_PORTABLE_CONTAINER = (
    Image("rfdiffusion2_portable")
    # Ubuntu 24.04 is required by this cluster's root-mapped Apptainer fakeroot
    # helper (built against glibc >=2.38); the conda env carries the pinned
    # CUDA 12.4 / PyTorch 2.4 stack, so the OS release does not matter to it.
    .base_image("ubuntu:24.04")
    .packages(
        "build-essential",
        "bzip2",
        "ca-certificates",
        "curl",
        "git",
        "libaio-dev",
        "libjpeg-dev",
        "libpng-dev",
        "libx11-6",
        "libxau6",
        "libxext6",
        "libxrender1",
        "wget",
    )
    .run(
        # Upstream's spec installs into Miniconda's base env. Miniforge avoids
        # the Anaconda terms-of-service prompt, and its base env pins a newer
        # Python than the stack needs, so the stack gets its own env.
        "wget -q -O /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh",
        "bash /tmp/miniforge.sh -b -p /opt/conda && rm /tmp/miniforge.sh",
        "/opt/conda/bin/conda create --yes -p /opt/rfd2-env"
        " -c nvidia/label/cuda-12.4.0 -c https://conda.rosettacommons.org"
        " -c pytorch -c dglteam/label/th24_cu124"
        " python==3.11 pip 'numpy<2' matplotlib dgl conda-forge::openbabel==3.1.1"
        " cuda pytorch==2.4 pytorch-cuda==12.4 pyrosetta",
    )
    .run(
        # Versions copied from upstream's rf_diffusion_aa.spec.
        "export PATH=/opt/rfd2-env/bin:$PATH && pip install --no-cache-dir"
        " hydra-core==1.3.1 ml-collections==0.1.1 addict==2.4.0 assertpy==1.1.0"
        " biopython==1.83 colorlog compact-json cython==3.0.0 cytoolz==0.12.3"
        " deepdiff==6.3.0 dm-tree==0.1.8 e3nn==0.5.1 einops==0.7.0 executing==2.0.0"
        " fastparquet==2024.5.0 fire==0.6.0 GPUtil==1.4.0 icecream==2.1.3"
        " mdtraj==1.10.0 numba omegaconf==2.3.0 opt_einsum==3.3.0 pandas==1.5.0"
        " plotly==5.16.1 py3Dmol==2.2.1 pyarrow==17.0.0 pydantic pyrsistent==0.19.3"
        " rdkit==2024.3.5 RestrictedPython scipy==1.13.1 seaborn==0.13.2"
        " submitit sympy==1.13.2 tmtools tqdm==4.65.0 typer==0.12.5 wandb==0.13.10"
        " biotite torchdata==0.9.0",
        "export PATH=/opt/rfd2-env/bin:$PATH && pip install --no-cache-dir"
        " pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv"
        " -f https://data.pyg.org/whl/torch-2.4.0+cu124.html",
        'export PATH=/opt/rfd2-env/bin:$PATH && pip install --no-cache-dir -U -i https://pypi.anaconda.org/rapidsai-wheels-nightly/simple "pylibcugraphops-cu12>=24.6.0a24"',
        "export PATH=/opt/rfd2-env/bin:$PATH && pip install --no-cache-dir deepspeed==0.15.1",
        # Installing ipd pulls in its dependencies; the repo ships its own
        # copy of the package, so the installed one is removed again.
        "export PATH=/opt/rfd2-env/bin:$PATH && pip install --no-cache-dir"
        " git+https://github.com/RalphMao/PyTimer.git"
        " git+https://github.com/baker-laboratory/ipd.git",
        "export PATH=/opt/rfd2-env/bin:$PATH && pip uninstall -y ipd",
        "git clone --depth 1 -b v3.5.1 https://github.com/NVIDIA/cutlass /opt/cutlass",
    )
    .git_clone(RFD2_REPO_URL, _RFD2_CODE_DIR)
    .run(
        # Fails the build if the download does not match the pinned hash.
        f"mkdir -p /opt/rfd2-weights && curl -fL --retry 5 -o {_RFD2_BAKED_CKPT} {_RFD2_CKPT_URL}"
        f" && echo '{_RFD2_CKPT_SHA256}  {_RFD2_BAKED_CKPT}' | sha256sum -c -",
        f"cd {_RFD2_CODE_DIR} && git checkout {RFD2_COMMIT}",
        "/opt/conda/bin/conda clean -a -y",
        "apt-get clean && rm -rf /var/lib/apt/lists/*",
    )
    .env(
        PATH="/opt/rfd2-env/bin:$PATH",
        CUTLASS_PATH="/opt/cutlass/",
        MKL_SERVICE_FORCE_INTEL="1",
        PYTHONPATH=_RFD2_CODE_DIR,
    )
    .label(Version=RFD2_COMMIT[:7], CUDA="12.4", Source="RosettaCommons/RFdiffusion2")
    # Same entrypoint as upstream's spec: the first argument is a python file.
    .runscript('python "$@"')
    .template_command()
)


def _onprem_rfd2_sif_exists() -> bool:
    return Path(_ONPREM_RFD2_SIF).is_file()


def _pick_rfd2_container() -> Image:
    """On-prem: use the prebuilt SIF. Elsewhere: build from the public repo."""
    if _onprem_rfd2_sif_exists():
        return _RFD2_SIF_CONTAINER
    return _RFD2_PORTABLE_CONTAINER


RFD2_CONTAINER = _pick_rfd2_container()
RFD2_PORTABLE_CONTAINER = _RFD2_PORTABLE_CONTAINER


def _default_ckpt_path() -> str:
    if RFD2_CONTAINER is not _RFD2_PORTABLE_CONTAINER:
        return _ONPREM_RFD2_CKPT
    return _RFD2_BAKED_CKPT


###########################################################################
# TRB File Handling
###########################################################################

_rfd_modules_spoofed = False


def _spoof_rfd_modules():
    """Spoof rf_diffusion modules to allow unpickling TRB files.

    TRB files contain pickled numpy arrays that reference rf_diffusion classes.
    This creates dummy modules so numpy.load can unpickle them without
    the actual rf_diffusion package installed.
    """
    global _rfd_modules_spoofed
    if _rfd_modules_spoofed:
        return

    # Dummy classes for spoofing
    class AtomizerSpec:
        def __init__(self, *args, **kwargs):
            pass

    class AtomizedLabel:
        def __init__(self, *args, **kwargs):
            pass

    # Create mock module hierarchy
    rf_diffusion = types.ModuleType("rf_diffusion")
    atomization_primitives = types.ModuleType("rf_diffusion.atomization_primitives")

    # Inject dummy classes
    atomization_primitives.AtomizerSpec = AtomizerSpec
    atomization_primitives.AtomizedLabel = AtomizedLabel

    # Register modules
    sys.modules["rf_diffusion"] = rf_diffusion
    sys.modules["rf_diffusion.atomization_primitives"] = atomization_primitives

    _rfd_modules_spoofed = True
    logger.debug("Spoofed rf_diffusion modules for TRB file parsing")


def _extract_trb_info(pdb_path: str, raise_on_missing: bool = True) -> dict:
    """Extract motif information from a TRB file.

    Args:
        pdb_path: Path to the PDB file (TRB file is derived from this)
        raise_on_missing: If True, raise error when TRB file is missing

    Returns:
        Dictionary with design_metadata fields:
        - motif: List of PDB indices, e.g. ['A22', 'A75', 'A124'] (for MPNN compatibility)
        - motif_idx0: List of 0-indexed motif positions, e.g. [21, 74, 123]
        - motif_pdb_idx: Comma-separated PDB indices, e.g. 'A22,A75,A124'
        - motif_atom_contigs: Dict mapping PDB idx to atom names, e.g.
            {'A22': ['NH2', 'NH1', 'CZ'], 'A75': ['NZ', 'CE']}


    e.g.:
            True,  True])},
            'con_ref_pdb_idx': [('A', np.int64(18)), ('A', np.int64(47)), ('A', np.int64(102)), ('A', np.int64(167)), ('A', np.int64(201)), ('A', np.int64(238))],
            'con_hal_pdb_idx': [('A', np.int64(74)), ('A', np.int64(73)), ('A', np.int64(37)), ('A', np.int64(11)), ('A', np.int64(165)), ('A', np.int64(101))],
            'con_ref_idx0': array([0, 1, 2, 3, 4, 5]), 'con_hal_idx0': array([ 73,  72,  36,  10, 164, 100])
    """
    _spoof_rfd_modules()

    def join(x):
        return "{}{}".format(x[0], int(x[1]))

    trb_path = pdb_path.replace(".pdb", ".trb")
    result = {}

    if not os.path.exists(trb_path):
        if raise_on_missing:
            raise ValueError(f"TRB file {trb_path} not found for PDB {pdb_path}")
        logger.warning(f"TRB file {trb_path} not found for PDB {pdb_path}")
        return result
    try:
        trb = np.load(trb_path, allow_pickle=True)

        # # Extract motif indices (0-indexed)
        # if 'con_hal_idx0' in trb:
        #     result['motif_idx0'] = np.array(trb['con_hal_idx0']).astype(int).tolist()

        # Extract PDB indices (chain + residue number)
        if "con_hal_pdb_idx" in trb:
            result[DesignMetadata.MOTIF] = [join(idx) for idx in trb["con_hal_pdb_idx"]]

            result[DesignMetadata.DIFFUSED_INDEX_MAP] = json.dumps(
                {
                    join(ref_idx): join(hal_idx)
                    for ref_idx, hal_idx in zip(
                        trb["con_ref_pdb_idx"], trb["con_hal_pdb_idx"]
                    )
                }
            )

        if "config" in trb:
            inference_cfg = trb["config"].get("inference", {})
            if "ligand" in inference_cfg:
                result[DesignMetadata.LIGAND] = inference_cfg["ligand"]
            result[DesignMetadata.ORIGINAL_INPUT] = inference_cfg.get("input_pdb", "")

            path = trb["config"]["inference"]["output_prefix"].split("/")[-1]
            task = path.split("_chunk")[0]
            result[DesignMetadata.TASK] = task

    except Exception as e:
        logger.error(f"Error parsing TRB file {trb_path}: {e}")
        if raise_on_missing:
            raise

    return result


###########################################################################
# RFdiffusion2 Block
###########################################################################


class RFD2Block(CompiledBlockFunction):
    """RFdiffusion2 (rf_diffusion) block for atomic-level protein design.

    This block runs RFdiffusion2 (also known as RFdiffusion-AA or rf_diffusion),
    which provides atomic-level protein design capabilities.
    """

    _name = "rfd2"
    _display_name = "RFdiffusion"
    _description = "RFdiffusion2 (rf_diffusion) block for atomic-level protein design"
    _image = RFD2_CONTAINER
    _resources = JobArgs(
        partition="gpu",
        gpu_class="small",
        mem_gb=32,
        rows_per_job=32,
    )
    _requires_input = False
    _inline = False
    _chunked = True

    @classmethod
    def _get_image(cls) -> Image:
        # Resolved from the module global so the on-prem/portable choice made
        # at import time is the single source of truth.
        return RFD2_CONTAINER

    def get_command_template(self) -> List[str]:
        image = self._get_image()
        if image is _RFD2_PORTABLE_CONTAINER:
            # Code and the default checkpoint are baked into the image; a custom
            # ckpt_path is bound in from the host.
            binds = []
            if self.ckpt_path != _RFD2_BAKED_CKPT:
                ckpt_dir = os.path.dirname(os.path.abspath(self.ckpt_path))
                binds.append((ckpt_dir, ckpt_dir))
            # The checkpoint's embedded training config interpolates $USER,
            # which --cleanenv would otherwise leave unset.
            return image.build_command(
                f"{_RFD2_CODE_DIR}/rf_diffusion/run_inference.py",
                env={"PYTHONPATH": _RFD2_CODE_DIR, "USER": getpass.getuser()},
                binds=binds,
            )
        return image.build_command(
            f"{_ONPREM_RFD2_CODE}/rf_diffusion/run_inference.py",
            env={"PYTHONPATH": f"{_ONPREM_RFD2_CODE}/"},
        )

    inputs: str | List[str] = Field(
        ...,
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Hydra config name to use (references a yaml/json file in benchmark configs)",
    )
    num_designs: int = Field(
        1,
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Number of designs to generate per benchmark case",
    )
    ckpt_path: str = Field(
        default_factory=_default_ckpt_path,
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Path to RFdiffusion2 checkpoint file. The portable image "
        "defaults to the RFD_173.pt baked into it.",
    )
    raise_on_missing_trbs: bool = Field(
        True, description="Raise error if TRB file is missing for a design"
    )

    def init_handler(self):
        from chaperone.data.schema import Column

        return PipelineDataHandler(
            [
                {Column.STRUCTURES: [], Column.METRICS: "{}"}
                for _ in range(self.num_designs)
            ]
        )

    def create_command(self, handler, sandbox_dir, chunk_id):
        if (
            self._get_image() is _RFD2_PORTABLE_CONTAINER
            and self.ckpt_path != _RFD2_BAKED_CKPT
            and not os.path.isfile(self.ckpt_path)
        ):
            raise FileNotFoundError(
                f"RFdiffusion2 checkpoint not found: {self.ckpt_path}"
            )
        cfgs = self.inputs if isinstance(self.inputs, list) else self.inputs.split(",")
        cmds = []
        for cfg in cfgs:
            cfg = Path(cfg).absolute()
            # Materialize archived structures; ordinary paths pass through.
            cfg = self.stage_file(cfg, sandbox_dir, subdir="configs")

            # Built as hydra-style cmd
            # --config-dir adds RFdiffusion2 base configs to search path for defaults (e.g., 'aa')
            # RFdiffusion2 uses inference.output_prefix and inference.num_designs
            cmd = (
                self.command_template
                + [
                    f"--config-dir={cfg.parent}",
                    f"--config-name={cfg.name.replace(cfg.suffix, '')}",
                    f"inference.output_prefix={sandbox_dir}/{cfg.name.replace(cfg.suffix, '')}_chunk{chunk_id}",
                    f"inference.num_designs={len(handler)}",
                    f"inference.ckpt_path={self.ckpt_path}",
                    f"inference.seed_offset={int(chunk_id) * self.num_designs}",
                ]
                + ["&&"]
            )
            cmds.extend(cmd)

        return cmds[:-1]  # remove last &&

    def compile(self, sandbox_dir):
        """Compile RFD2 outputs into a PipelineDataHandler.

        Extracts motif information from TRB files produced by RFdiffusion2:
        - motif_idx0: List of 0-indexed motif positions
        - motif_pdb_idx: Comma-separated PDB indices (e.g., 'A22,A75,A124')
        - motif_atom_contigs: Dict mapping PDB idx to atom names (for atomized designs)
        """
        paths = glob_designs_in_dir(dir=sandbox_dir)

        if len(paths) == 0:
            raise RuntimeError(
                f"RFD2 execution completed but no output files were generated in {sandbox_dir}. "
                f"This may indicate:\n"
                f"  - Invalid configuration or benchmark specification\n"
                f"  - Execution errors (check sandbox directory for logs)\n"
                f"  - Output files written to different location\n"
                + tail_block_log(sandbox_dir)
            )

        # Process outputs and create rows
        rows = []
        for path in paths:
            # Extract motif info from TRB file
            trb_info = _extract_trb_info(
                pdb_path=path, raise_on_missing=self.raise_on_missing_trbs
            )

            rows.append(
                make_row(
                    name=self.name,
                    path=path,
                    metadata=trb_info,
                    # metrics={self.name: trb_info.get("metrics", {})}  # Nested under block name,
                )
            )

        handler = PipelineDataHandler(rows)
        return handler
