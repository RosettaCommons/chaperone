import os
import json
import glob
import shutil
import logging
import shlex
from pathlib import Path

from typing import Optional, Literal

from pydantic import Field, field_validator

from chaperone import (
    CompiledBlockFunction,
    PipelineDataHandler,
    JobArgs,
    get_setting_field,
)
from chaperone.block_config import TIER_PRIMARY, ScriptSpec
from chaperone.containers import Image
from chaperone.data import make_row
from chaperone.data.schema import Column
from blocks.constants import cuda_python_base

logger = logging.getLogger(__name__)

# ── Container definition ─────────────────────────────────────────────────

# Upstream does not publish a uv.lock. Preserve the 4.x ESM checkpoint layout:
# Transformers 5 moves its per-layer rotary buffers, breaking strict DISCO loads.
_DISCO_UV_CONFIG = """
[tool.uv]
constraint-dependencies = ["torch==2.11.0", "deepspeed==0.19.3", "lightning==2.6.5", "transformers==4.57.6", "numpy==1.26.4"]
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
[tool.uv.sources.torch]
index = "pytorch-cu128"
"""
_DISCO_ESM_COMPATIBILITY_CHECK = """
from transformers import EsmConfig, EsmForMaskedLM
config = EsmConfig(vocab_size=33, hidden_size=16, num_hidden_layers=2, num_attention_heads=2, intermediate_size=32, position_embedding_type='rotary', pad_token_id=1, mask_token_id=32)
model = EsmForMaskedLM(config)
expected = {f'esm.encoder.layer.{i}.attention.self.rotary_embeddings.inv_freq' for i in range(2)}
observed = {key for key in model.state_dict() if key.endswith('rotary_embeddings.inv_freq')}
if observed != expected:
    raise RuntimeError(f'ESM rotary buffer layout is incompatible with the DISCO checkpoint: {observed}')
model.load_state_dict(model.state_dict(), strict=True)
"""

DISCO_CONTAINER = (
    cuda_python_base(
        Image("disco"),
        "disco",
        # DISCO manages its own environment via uv (see below), so skip the
        # shared venv/env block.
        create_venv=False,
        packages=(
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
        ),
    )
    .run(
        "python3.12 -m venv /opt/uv-env && /opt/uv-env/bin/pip install uv==0.9.24",
    )
    .git_clone(
        "https://github.com/DISCO-design/DISCO.git",
        "/opt/disco",
        branch="main",
    )
    .run(
        "git -C /opt/disco checkout --detach 398098c57e3f1dd6dd6205c059f7792dcc9b9147",
        "python3.12 -c " + shlex.quote(
            "from pathlib import Path; p = Path('/opt/disco/pyproject.toml'); "
            f"p.write_text(p.read_text() + {_DISCO_UV_CONFIG!r})"
        ),
        "cd /opt/disco && /opt/uv-env/bin/uv sync --python python3.12",
        "/opt/uv-env/bin/uv pip check --python /opt/disco/.venv/bin/python",
    )
    .run(
        # Clone CUTLASS for DeepSpeed EvoformerAttention
        "git init /opt/cutlass && "
        "git -C /opt/cutlass remote add origin https://github.com/NVIDIA/cutlass.git && "
        "git -C /opt/cutlass fetch --depth 1 origin bd01dd3651330accacb8083aedfac3bf522f25e5 && "
        "git -C /opt/cutlass checkout --detach FETCH_HEAD",
    )
    .run(
        "/opt/disco/.venv/bin/python -c " + shlex.quote(f"exec({_DISCO_ESM_COMPATIBILITY_CHECK!r})"),
        "cd /opt/disco && CUDA_HOME=/usr/local/cuda CUTLASS_PATH=/opt/cutlass "
        "/opt/disco/.venv/bin/python -c "
        "'import torch; import runner.inference; assert torch.version.cuda == \"12.8\"; print(torch.__version__)'",
    )
    .run("apt-get clean && rm -rf /var/lib/apt/lists/*")
    .env(
        CUDA_HOME="/usr/local/cuda",
        PATH="/opt/disco/.venv/bin:/usr/local/cuda/bin:$PATH",
        LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH",
        VIRTUAL_ENV="/opt/disco/.venv",
        CUTLASS_PATH="/opt/cutlass",
    )
    .label(Version="398098c57e3f1dd6dd6205c059f7792dcc9b9147", CUDA="12.8", PyTorch="2.11.0")
    .runscript('"$@"')
    .template_command(
        exec_mode="exec",
        binds=[("/mnt/home", "/mnt/home")],
        env={"HF_HUB_CACHE": lambda: get_setting_field("HF_HUB_CACHE", "")},
    )
)

# ── Block implementation ─────────────────────────────────────────────────


class DISCOBlock(CompiledBlockFunction):
    """DISCO: Diffusion for Sequence-Structure Co-design.

    Simultaneously co-designs protein sequences and 3D structures,
    conditioned on arbitrary biomolecules (ligands, DNA, RNA).
    Uses a Hydra-based runner with JSON input configs following the
    AlphaFold Server format.
    """

    _name = "disco"
    _display_name = "DISCO"
    _description = "Multimodal protein sequence-structure co-design"
    _script_spec = ScriptSpec(
        command=["python", "/opt/disco/runner/inference.py"],
        image=DISCO_CONTAINER,
        source_url="https://github.com/DISCO-design/DISCO",
    )
    _resources = JobArgs(
        partition="gpu",
        gpu_class="small",
        mem_gb=32,
        rows_per_job=1,
    )
    _requires_input = False
    _inline = False
    _chunked = True

    # ── Primary parameters ────────────────────────────────────────────

    inputs: str = Field(
        ...,
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Path to DISCO input JSON file (AF3 Server-like format)",
    )
    num_seeds: int = Field(
        5,
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Number of random seeds (samples per job in the input JSON)",
    )
    experiment: Literal["designable", "diverse"] = Field(
        "designable",
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Experiment preset: 'designable' (higher designability) or 'diverse' (greater structural variety)",
    )
    effort: Literal["fast", "max"] = Field(
        "fast",
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Compute preset: 'fast' (~4x faster, default) or 'max' (full quality). Use 'max' for conditional generation.",
    )

    # ── Advanced parameters ───────────────────────────────────────────

    n_batches: int = Field(
        1,
        description="Number of parallel chunks to split seeds across",
    )
    use_deepspeed_evo_attention: bool = Field(
        True,
        description="Use DeepSpeed EvoformerAttention (requires CUTLASS + Ampere+ GPU)",
    )
    n_diffusion_steps: Optional[int] = Field(
        None,
        description="Override number of diffusion steps (default: set by effort preset)",
    )
    n_cycles: Optional[int] = Field(
        None,
        description="Override number of recycling cycles (default: set by effort preset)",
    )
    noisy_guidance: Optional[bool] = Field(
        None,
        description="Enable/disable noisy guidance (default: set by experiment preset)",
    )
    ckpt_path: Optional[str] = Field(
        None,
        description="Path to custom model checkpoint (default: auto-download from HuggingFace)",
    )
    cache_dir: str = Field(
        default_factory=lambda: str(
            get_setting_field("HF_HUB_CACHE", None)
            or Path.home() / ".cache" / "huggingface" / "hub"
        ),
        description="Hugging Face model cache mounted into the container",
    )

    @field_validator("inputs")
    def ensure_inputs_exist(cls, v):
        if not os.path.exists(v):
            raise ValueError(f"Input JSON file {v} does not exist")
        return os.path.abspath(v)

    @field_validator("ckpt_path")
    def ensure_ckpt_exists(cls, v):
        if v is not None and not os.path.exists(v):
            raise ValueError(f"Checkpoint path {v} does not exist")
        return str(Path(v).resolve()) if v is not None else None

    def get_command_template(self) -> list[str]:
        self.cache_dir = str(Path(self.cache_dir).expanduser().resolve())
        binds = [(self.cache_dir, self.cache_dir)]
        if self.ckpt_path:
            binds.append((self.ckpt_path, self.ckpt_path))
        return self._script_spec.image.build_command(
            *self._script_spec.command,
            env={"HF_HUB_CACHE": self.cache_dir},
            binds=binds,
        )

    def init_handler(self):
        rows = []
        for _ in range(self.n_batches):
            rows.append({Column.STRUCTURES: [], Column.METRICS: "{}"})
        return PipelineDataHandler(rows)

    def create_command(self, handler, sandbox_dir, chunk_id):
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        sandbox_dir = os.path.abspath(sandbox_dir)
        os.makedirs(sandbox_dir, exist_ok=True)

        # Copy input JSON into sandbox for reproducibility
        input_dest = os.path.join(sandbox_dir, "input_config.json")
        shutil.copy2(self.inputs, input_dest)

        # Copy any referenced SDF/MOL files into sandbox
        self._stage_ligand_files(input_dest, sandbox_dir)

        # Split seeds across chunks
        all_seeds = list(range(self.num_seeds))
        chunk_seeds = _split_seeds(all_seeds, self.n_batches, chunk_id)

        if not chunk_seeds:
            return []

        seeds_str = "[" + ",".join(str(s) for s in chunk_seeds) + "]"

        # Output directory per chunk
        dump_dir = os.path.join(sandbox_dir, "output")

        # Build Hydra-style command
        cmd = self.command_template + [
            f"input_json_path={input_dest}",
            f"experiment={self.experiment}",
            f"effort={self.effort}",
            f"seeds={seeds_str}",
            f"dump_dir={dump_dir}",
            f"use_deepspeed_evo_attention={str(self.use_deepspeed_evo_attention).lower()}",
        ]

        if self.n_diffusion_steps is not None:
            cmd.append(f"sample_diffusion.N_step={self.n_diffusion_steps}")
        if self.n_cycles is not None:
            cmd.append(f"model.N_cycle={self.n_cycles}")
        if self.noisy_guidance is not None:
            cmd.append(
                f"sample_diffusion.noisy_guidance.enabled={str(self.noisy_guidance).lower()}"
            )
        if self.ckpt_path is not None:
            cmd.append(f"load_checkpoint_path={self.ckpt_path}")

        return cmd

    def _stage_ligand_files(self, input_json_path, sandbox_dir):
        """Copy ligand files referenced with FILE_ prefix into sandbox and update paths."""
        with open(input_json_path) as f:
            data = json.load(f)

        modified = False
        for job in data:
            for seq_entry in job.get("sequences", []):
                for key in ("ligand",):
                    if key in seq_entry and isinstance(seq_entry[key], dict):
                        lig_spec = seq_entry[key].get("ligand", "")
                        if isinstance(lig_spec, str) and lig_spec.startswith("FILE_"):
                            src_path = lig_spec[5:]  # Strip FILE_ prefix
                            if not os.path.isabs(src_path):
                                # Resolve relative to original inputs directory
                                src_path = os.path.join(
                                    os.path.dirname(self.inputs), src_path
                                )
                            if os.path.exists(src_path):
                                ligand_dir = os.path.join(sandbox_dir, "ligands")
                                os.makedirs(ligand_dir, exist_ok=True)
                                dest = os.path.join(
                                    ligand_dir, os.path.basename(src_path)
                                )
                                shutil.copy2(src_path, dest)
                                seq_entry[key]["ligand"] = f"FILE_{dest}"
                                modified = True

        if modified:
            with open(input_json_path, "w") as f:
                json.dump(data, f, indent=2)

    def compile(self, sandbox_dir):
        return compile_disco(sandbox_dir, name=self.name)


# ── Output compilation ───────────────────────────────────────────────────


def compile_disco(sandbox_dir, name="disco"):
    """Compile DISCO outputs into a PipelineDataHandler."""
    sandbox_dir = str(sandbox_dir)

    # DISCO outputs to dump_dir/pdbs/ and dump_dir/sequences/
    output_dir = os.path.join(sandbox_dir, "output")
    pdbs_dir = os.path.join(output_dir, "pdbs")
    seq_dir = os.path.join(output_dir, "sequences")

    # Find all output structure files (PDB and CIF)
    paths = []
    if os.path.isdir(pdbs_dir):
        paths = sorted(
            glob.glob(os.path.join(pdbs_dir, "*.pdb"))
            + glob.glob(os.path.join(pdbs_dir, "*.cif"))
        )

    if not paths:
        raise RuntimeError(
            f"DISCO completed but no output files found in {pdbs_dir}. "
            f"Check {os.path.join(output_dir, 'ERR')} for error logs."
        )

    # Load per-design sequence info if available
    seq_lookup = {}
    if os.path.isdir(seq_dir):
        for seq_file in glob.glob(os.path.join(seq_dir, "*.txt")):
            basename = os.path.splitext(os.path.basename(seq_file))[0]
            seq_lookup[basename] = _parse_disco_sequence_file(seq_file)

    # Load ligand info from sidecar files
    ligand_lookup = {}
    if os.path.isdir(pdbs_dir):
        for lig_file in glob.glob(os.path.join(pdbs_dir, "*_ligands.txt")):
            # e.g. length_200_heme_b_sample_0_ligands.txt -> length_200_heme_b_sample_0
            basename = os.path.basename(lig_file).replace("_ligands.txt", "")
            ligand_lookup[basename] = _parse_disco_ligand_file(lig_file)

    rows = []
    for path in paths:
        basename = os.path.splitext(os.path.basename(path))[0]

        metrics = {}
        metadata = {}

        # Add sequence info
        if basename in seq_lookup:
            seq_info = seq_lookup[basename]
            if seq_info.get("sequence"):
                metrics["sequence"] = seq_info["sequence"]
            if seq_info.get("ligand_smiles"):
                metadata["ligand_smiles"] = seq_info["ligand_smiles"]

        # Add ligand info
        if basename in ligand_lookup:
            metadata["ligand_smiles"] = ligand_lookup[basename]

        # Parse seed from filename: <name>_sample_<seed>.pdb
        parts = basename.rsplit("_sample_", 1)
        if len(parts) == 2:
            metadata["job_name"] = parts[0]
            metadata["seed"] = int(parts[1])

        rows.append(
            make_row(
                name=name,
                path=path,
                metadata=metadata,
                metrics={name: metrics} if metrics else {},
            )
        )

    logger.info(f"DISCO compiled {len(rows)} designs from {pdbs_dir}")
    return PipelineDataHandler(rows)


def _parse_disco_sequence_file(seq_file):
    """Parse a DISCO sequence output .txt file."""
    result = {"sequence": None, "ligand_smiles": []}
    try:
        with open(seq_file) as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    continue
                if line.startswith("ligand_smiles "):
                    result["ligand_smiles"].append(line.split(" ", 1)[1])
                elif line and not line.startswith("#"):
                    # First non-header, non-ligand line is the sequence
                    if result["sequence"] is None:
                        result["sequence"] = line
    except Exception as e:
        logger.warning(f"Failed to parse sequence file {seq_file}: {e}")
    return result


def _parse_disco_ligand_file(lig_file):
    """Parse a DISCO ligands.txt sidecar file."""
    smiles = []
    try:
        with open(lig_file) as f:
            for line in f:
                line = line.strip()
                if line.startswith("ligand_smiles "):
                    smiles.append(line.split(" ", 1)[1])
    except Exception as e:
        logger.warning(f"Failed to parse ligand file {lig_file}: {e}")
    return smiles


def _split_seeds(all_seeds, n_chunks, chunk_id):
    """Split seeds evenly across chunks."""
    if chunk_id >= n_chunks:
        return []
    chunk_size = len(all_seeds) // n_chunks
    remainder = len(all_seeds) % n_chunks
    start = chunk_id * chunk_size + min(chunk_id, remainder)
    end = start + chunk_size + (1 if chunk_id < remainder else 0)
    return all_seeds[start:end]
