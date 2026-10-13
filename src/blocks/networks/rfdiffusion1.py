"""RFdiffusion v1 block for backbone-level protein design.

RFdiffusion v1 (original, from RosettaCommons/RFdiffusion) provides backbone-level
protein design capabilities including:
- Unconditional monomer generation
- Motif scaffolding (fixed backbone motifs)
- Symmetric oligomer design (cyclic, dihedral, tetrahedral)
- Binder design with hotspot residues
- Partial diffusion for design diversification
- Fold-conditioned design with secondary structure/adjacency constraints

Reference: Watson et al., "De novo design of protein structure and function with RFdiffusion"
Nature (2023). https://doi.org/10.1038/s41586-023-06415-8

Contig Syntax
=============
RFdiffusion uses a contig string to describe the protein topology. The contig
is a **space-separated** list of chain specifications, where each chain is a
**slash-separated** list of segments:

Segments:
  - ``A1-7``   — motif: keep residues 1-7 from chain A of the input PDB
  - ``150``    — diffuse: generate 150 new residues
  - ``10-40``  — diffuse: random length between 10 and 40
  - ``0``      — chain break: marks the end of a chain (receptor/fixed)

Chain types:
  - Inpaint chain: has at least one diffused segment (e.g. ``10-40/A163-181/10-40``)
  - Receptor/fixed chain: ALL segments are motif refs AND ends with ``/0``
    (e.g. ``A1-7/B1-7/C1-7/0``)

Examples:
  - Unconditional monomer:           ``['100-200']``
  - Motif scaffolding:               ``['10-40/A163-181/10-40']``
  - C3 symmetric (no motif):         ``['480-480']``  (total = 3 × 160)
  - C4 + motif (nickel example):     ``['50/A2-4/50/0 50/A7-9/50/0 50/A12-14/50/0 50/A17-19/50/0']``
  - C3 + motif (nanoparticle):       ``['A1-7/150/0 B1-7/150/0 C1-7/150/0']``
  - Binder design:                   ``['A1-150/0 70-100']``  (target chain + binder)

Important:
  - For symmetric designs, the total sequence length (sum of all diffused + motif
    residues) must be divisible by the symmetry order (e.g. 3 for C3).
  - Segments within a chain are separated by ``/``, NOT commas.
  - Chains are separated by spaces.
  - The Hydra CLI expects the contig as: ``'contigmap.contigs=[A1-7/150/0 B1-7/150/0 C1-7/150/0]'``
    (single-quoted on shell, brackets for Hydra list).

Container
=========
The container mounts model weights at runtime from ``RFDIFFUSION_WEIGHTS_DIR``
(see ``.env.ipd``). The container itself contains the RFdiffusion code and
dependencies (PyTorch, DGL, SE3-Transformer) but NOT the ~2.3 GB of model weights.

The ``--writable-tmpfs`` apptainer flag is required because RFdiffusion writes
noise schedules to ``/opt/RFdiffusion/schedules/`` at runtime.
"""

import logging
import os
from typing import List, Optional

from pydantic import Field

from chaperone import CompiledBlockFunction, JobArgs, PipelineDataHandler
from chaperone.block_config import TIER_PRIMARY
from chaperone.block_config.compiled import ScriptSpec
from chaperone.common import glob_designs_in_dir, tail_block_log
from chaperone.containers import Image
from chaperone.data import make_row
from blocks.constants import CUDA_INSTALL

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# Container: builds from public RosettaCommons/RFdiffusion
# ──────────────────────────────────────────────────────────

# Model weight URLs from IPD file server
_RFD1_WEIGHTS = {
    "Base_ckpt.pt": "http://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt",
    "Complex_base_ckpt.pt": "http://files.ipd.uw.edu/pub/RFdiffusion/e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt",
    "Complex_Fold_base_ckpt.pt": "http://files.ipd.uw.edu/pub/RFdiffusion/60f09a193fb5e5ccdc4980417708dbab/Complex_Fold_base_ckpt.pt",
    "InpaintSeq_ckpt.pt": "http://files.ipd.uw.edu/pub/RFdiffusion/74f51cfb8b440f50d70878e05361d8f0/InpaintSeq_ckpt.pt",
    "ActiveSite_ckpt.pt": "http://files.ipd.uw.edu/pub/RFdiffusion/5532d2e1f3a4738decd58b19d633b3c3/ActiveSite_ckpt.pt",
}

_WEIGHT_DOWNLOADS = " && ".join(
    f"wget -q -O /opt/RFdiffusion/models/{name} {url}"
    for name, url in _RFD1_WEIGHTS.items()
)

RFD1_CONTAINER = (
    Image("rfdiffusion1")
    .base_image("ubuntu:24.04")
    .packages(
        "build-essential",
        "git",
        "wget",
        "curl",
        "ca-certificates",
        "gnupg",
        "software-properties-common",
        "python3.12",
        "python3.12-dev",
        "python3.12-venv",
        "python3-pip",
    )
    .run(*CUDA_INSTALL)
    .run(
        # Create venv
        "python3.12 -m venv /opt/rfd1-env",
        ". /opt/rfd1-env/bin/activate && pip install --upgrade pip setuptools wheel",
    )
    .git_clone(
        "https://github.com/RosettaCommons/RFdiffusion.git",
        "/opt/RFdiffusion",
    )
    .run(
        # Install PyTorch 2.4 + DGL (pinned for compatibility)
        ". /opt/rfd1-env/bin/activate && "
        "pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124 && "
        "pip install --no-cache-dir dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html",
    )
    .run(
        # Install SE3-Transformer: install requirements first (skip pinned e3nn
        # to avoid conflicts), then install the package itself
        ". /opt/rfd1-env/bin/activate && "
        "pip install --no-cache-dir wandb pynvml 'git+https://github.com/NVIDIA/dllogger#egg=dllogger' decorator && "
        "cd /opt/RFdiffusion/env/SE3Transformer && "
        "pip install --no-build-isolation --no-cache-dir .",
    )
    .run(
        # Install RFdiffusion dependencies (e3nn after SE3Transformer to get newer version)
        ". /opt/rfd1-env/bin/activate && "
        "pip install --no-cache-dir hydra-core pyrsistent icecream opt-einsum e3nn",
    )
    .run(
        # Create model directory (weights mounted at runtime via RFDIFFUSION_WEIGHTS_DIR)
        # and schedules directory (RFdiffusion writes noise schedules here)
        "mkdir -p /opt/RFdiffusion/models",
        "mkdir -p /opt/RFdiffusion/schedules",
    )
    .run("apt-get clean && rm -rf /var/lib/apt/lists/*")
    .env(
        CUDA_HOME="/usr/local/cuda",
        PATH="/opt/rfd1-env/bin:/usr/local/cuda/bin:$PATH",
        LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH",
        VIRTUAL_ENV="/opt/rfd1-env",
        DGLBACKEND="pytorch",
        PYTHONPATH="/opt/RFdiffusion:$PYTHONPATH",
    )
    .label(
        Version="v1.1.0",
        CUDA="12.6",
        Source="https://github.com/RosettaCommons/RFdiffusion",
    )
    .runscript('"$@"')
    .template_command(
        exec_mode="exec",
        binds=[
            ("/mnt/home", "/mnt/home"),
        ],
        env={},
    )
)


# ──────────────────────────────────────────────────────────
# TRB File Handling (shared with rfd2)
# ──────────────────────────────────────────────────────────


def _extract_trb_info(pdb_path: str, raise_on_missing: bool = True) -> dict:
    """Extract motif information from a TRB file produced by RFdiffusion v1.

    Returns dict with motif, diffused_index_map, etc.
    """
    from blocks.networks.rfdiffusion2 import _extract_trb_info as _rfd2_extract_trb

    return _rfd2_extract_trb(pdb_path, raise_on_missing=raise_on_missing)


# ──────────────────────────────────────────────────────────
# RFdiffusion v1 Block
# ──────────────────────────────────────────────────────────

# Script is baked in; model weights are mounted at runtime
_RFD1_SCRIPT = "/opt/RFdiffusion/scripts/run_inference.py"
_RFD1_MODELS_DIR = "/opt/RFdiffusion/models"


def _get_rfd1_weights_dir() -> str:
    """Get the RFdiffusion weights directory path."""
    from chaperone import get_setting_field

    return get_setting_field("RFDIFFUSION_WEIGHTS_DIR", default="")


class RFD1Block(CompiledBlockFunction):
    """RFdiffusion v1 block for backbone-level protein design.

    Runs the original RFdiffusion (RosettaCommons/RFdiffusion) for backbone
    generation, motif scaffolding, symmetric oligomer design, and binder design.
    Uses Hydra-style CLI configuration.

    Model weights are mounted at runtime from RFDIFFUSION_WEIGHTS_DIR setting.
    """

    _name = "rfd1"
    _description = "RFdiffusion v1 backbone-level protein design"
    _script_spec = ScriptSpec(
        command=["python", _RFD1_SCRIPT],
        image=RFD1_CONTAINER,
        source_url="https://github.com/RosettaCommons/RFdiffusion",
    )
    _resources = JobArgs(
        partition="gpu",
        gpu_class="small",
        mem_gb=16,
        rows_per_job=32,
    )
    _requires_input = False
    _inline = False
    _chunked = True

    # ── Primary parameters ─────────────────────────────────

    contigs: str = Field(
        ...,
        json_schema_extra={"tier": TIER_PRIMARY},
        description=(
            "Contig specification. Segments separated by /, chains by spaces. "
            "E.g. '150-150' for unconditional, "
            "'5-15/A10-25/30-40' for scaffolding, "
            "'A1-7/150/0 B1-7/150/0 C1-7/150/0' for symmetric with motif"
        ),
    )
    num_designs: int = Field(
        10,
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Number of designs to generate per chunk",
    )
    input_pdb: Optional[str] = Field(
        None,
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Input PDB for motif scaffolding or binder design",
    )
    config_name: str = Field(
        "base",
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Hydra config name (e.g. 'base', 'symmetry')",
    )

    # ── Symmetry ───────────────────────────────────────────

    symmetry: Optional[str] = Field(
        None,
        description="Symmetry type for oligomer design (e.g. 'c3', 'c5', 'd3', 'tetrahedral'). Sets config_name='symmetry' automatically.",
    )

    # ── Diffusion control ──────────────────────────────────

    diffuser_T: int = Field(
        50,
        description="Number of denoising timesteps",
    )
    partial_T: Optional[int] = Field(
        None,
        description="Partial diffusion timesteps (for design diversification)",
    )
    noise_scale_ca: Optional[float] = Field(
        None,
        description="Translation noise scaling (denoiser.noise_scale_ca)",
    )
    noise_scale_frame: Optional[float] = Field(
        None,
        description="Rotation noise scaling (denoiser.noise_scale_frame)",
    )

    # ── Checkpoints ────────────────────────────────────────

    ckpt_override_path: Optional[str] = Field(
        None,
        description=(
            "Custom checkpoint path. Defaults to container's Base_ckpt.pt. "
            "Common options: Complex_base_ckpt.pt, Complex_Fold_base_ckpt.pt, "
            "ActiveSite_ckpt.pt, InpaintSeq_ckpt.pt"
        ),
    )

    # ── Binder design ──────────────────────────────────────

    hotspot_res: Optional[List[str]] = Field(
        None,
        description="Hotspot residues for binder design, e.g. ['A30', 'A33', 'A34']",
    )

    # ── Scaffolding ────────────────────────────────────────

    inpaint_seq: Optional[List[str]] = Field(
        None,
        description="Inpaint sequence positions, e.g. ['A1', 'A30-40']",
    )
    inpaint_str: Optional[List[str]] = Field(
        None,
        description="Inpaint structure positions (flexible regions), e.g. ['B165-178']",
    )

    # ── Scaffold-guided design ─────────────────────────────

    scaffold_guided: bool = Field(
        False,
        description="Enable fold-conditioned design with ss/adj constraints",
    )
    target_pdb: Optional[str] = Field(
        None,
        description="Target PDB for scaffold-guided binder design",
    )
    scaffold_dir: Optional[str] = Field(
        None,
        description="Directory of _ss.pt and _adj.pt files for scaffold guidance",
    )

    # ── Extra Hydra overrides ──────────────────────────────

    extra_hydra_args: Optional[List[str]] = Field(
        None,
        description="Additional Hydra CLI overrides passed verbatim, e.g. ['diffuser.chi_type=interp']",
    )

    # ── TRB parsing ────────────────────────────────────────

    raise_on_missing_trbs: bool = Field(
        True,
        description="Raise error if TRB file is missing for a design",
    )

    def get_command_template(self) -> List[str]:
        """Override to add weights bind mount and writable tmpfs overlay.

        RFdiffusion writes noise schedules to /opt/RFdiffusion/schedules/
        which requires a writable overlay on the read-only SIF.
        """
        weights_dir = _get_rfd1_weights_dir()
        extra_binds = []
        if weights_dir:
            extra_binds = [(weights_dir, _RFD1_MODELS_DIR)]
        cmd = RFD1_CONTAINER.build_command(
            "python",
            _RFD1_SCRIPT,
            binds=extra_binds,
        )
        # Insert --writable-tmpfs before the image reference for apptainer
        # so RFdiffusion can write noise schedules
        from chaperone import get_settings

        if get_settings().CHAPERONE_CONTAINER_BACKEND != "docker":
            # Find image ref position and insert before it
            for i, arg in enumerate(cmd):
                if arg.startswith("${CONTAINER:") or arg.endswith(".sif"):
                    cmd.insert(i, "--writable-tmpfs")
                    break
        return cmd

    def init_handler(self):
        from chaperone.data.schema import Column

        return PipelineDataHandler(
            [
                {Column.STRUCTURES: [], Column.METRICS: "{}"}
                for _ in range(self.num_designs)
            ]
        )

    def create_command(self, handler, sandbox_dir, chunk_id):
        sandbox_dir = os.path.abspath(sandbox_dir)

        # Resolve config name — auto-switch to 'symmetry' if symmetry is set
        cfg_name = self.config_name
        if self.symmetry and cfg_name == "base":
            cfg_name = "symmetry"

        cmd = self.command_template + [
            f"--config-name={cfg_name}",
            f"inference.output_prefix={sandbox_dir}/design_chunk{chunk_id}",
            f"inference.num_designs={len(handler)}",
            # Use + prefix for keys not in the default config (e.g. symmetry config)
            f"+inference.seed_offset={int(chunk_id) * self.num_designs}",
            f"diffuser.T={self.diffuser_T}",
        ]

        # Contigs — pass as a single Hydra override string.
        cmd.append(f"contigmap.contigs=[{self.contigs}]")

        # Input PDB
        if self.input_pdb:
            cmd.append(f"inference.input_pdb={os.path.abspath(self.input_pdb)}")

        # Symmetry
        if self.symmetry:
            cmd.append(f"inference.symmetry={self.symmetry}")

        # Checkpoint — default to container's Base_ckpt.pt
        if self.ckpt_override_path:
            cmd.append(f"inference.ckpt_override_path={self.ckpt_override_path}")
        else:
            cmd.append(f"inference.ckpt_override_path={_RFD1_MODELS_DIR}/Base_ckpt.pt")

        # Diffusion control
        if self.partial_T is not None:
            cmd.append(f"diffuser.partial_T={self.partial_T}")
        if self.noise_scale_ca is not None:
            cmd.append(f"denoiser.noise_scale_ca={self.noise_scale_ca}")
        if self.noise_scale_frame is not None:
            cmd.append(f"denoiser.noise_scale_frame={self.noise_scale_frame}")

        # Binder design
        if self.hotspot_res:
            cmd.append(f"ppi.hotspot_res={self.hotspot_res}")

        # Inpainting
        if self.inpaint_seq:
            cmd.append(f"contigmap.inpaint_seq={self.inpaint_seq}")
        if self.inpaint_str:
            cmd.append(f"contigmap.inpaint_str={self.inpaint_str}")

        # Scaffold-guided design
        if self.scaffold_guided:
            cmd.append("scaffoldguided.scaffoldguided=True")
            if self.target_pdb:
                cmd.append("scaffoldguided.target_pdb=True")
                cmd.append(
                    f"scaffoldguided.target_path={os.path.abspath(self.target_pdb)}"
                )
            if self.scaffold_dir:
                cmd.append(
                    f"scaffoldguided.scaffold_dir={os.path.abspath(self.scaffold_dir)}"
                )

        # Extra hydra overrides
        if self.extra_hydra_args:
            cmd.extend(self.extra_hydra_args)

        return cmd

    def compile(self, sandbox_dir):
        """Compile RFD1 outputs into a PipelineDataHandler.

        Reads PDB files and extracts motif info from TRB files.
        """
        paths = glob_designs_in_dir(dir=sandbox_dir)

        if len(paths) == 0:
            raise RuntimeError(
                f"RFD1 execution completed but no output files were generated in {sandbox_dir}. "
                f"Check sandbox directory for logs."
                + tail_block_log(sandbox_dir)
            )

        rows = []
        for path in paths:
            trb_info = _extract_trb_info(
                pdb_path=path, raise_on_missing=self.raise_on_missing_trbs
            )
            rows.append(
                make_row(
                    name=self.name,
                    path=path,
                    metadata=trb_info,
                )
            )

        return PipelineDataHandler(rows)
