"""Protenix structure prediction block."""

import json
import logging
import os
import shlex
from pathlib import Path
from typing import Optional

from pydantic import Field, model_validator

from blocks.constants import CUDA_INSTALL
from blocks.networks.prediction.helpers import (
    create_protenix_json_from_design,
    fill_protenix_json_from_template,
    is_pdb_file,
    pdb_assign_ligand_bonds,
)
from chaperone import PipelineDataHandler, JobArgs, get_setting_field
from chaperone.block_config import AF3LikePredictionBlock, TIER_PRIMARY, ScriptSpec
from chaperone.block_config.prediction import select_best_models
from chaperone.common import glob_designs_in_dir
from chaperone.containers import Image
from chaperone.data import make_row

logger = logging.getLogger(__name__)

###########################################################################
# Container
###########################################################################

PROTENIX_CONTAINER = (
    Image("protenix")
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
        "kalign",
    )
    .run(*CUDA_INSTALL)
    # Build HMMER 3.4 from source (needed for MSA when use_msa=True)
    .run(
        "mkdir -p /opt/hmmer_build /usr/local/hmmer",
        "wget http://eddylab.org/software/hmmer/hmmer-3.4.tar.gz -P /opt/hmmer_build",
        "cd /opt/hmmer_build && echo "
        "'ca70d94fd0cf271bd7063423aabb116d42de533117343a9b27a65c17ff06fbf3 hmmer-3.4.tar.gz' "
        "| sha256sum --check",
        "cd /opt/hmmer_build && tar zxf hmmer-3.4.tar.gz && rm hmmer-3.4.tar.gz",
        "cd /opt/hmmer_build/hmmer-3.4 && ./configure --prefix /usr/local/hmmer && make -j4 && make install",
        "cd /opt/hmmer_build/hmmer-3.4/easel && make install",
        "rm -rf /opt/hmmer_build",
    )
    .run(
        "python3.12 -m venv /opt/protenix-env",
        ". /opt/protenix-env/bin/activate && pip install --upgrade 'pip>=26.2.2' setuptools wheel",
    )
    # Keep Protenix's declared versions together, selecting CUDA 12.8 wheels
    # explicitly. Blackwell inference still needs verification on that hardware.
    .run(
        ". /opt/protenix-env/bin/activate && "
        "pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 "
        "--index-url https://download.pytorch.org/whl/cu128",
        ". /opt/protenix-env/bin/activate && pip install protenix==2.0.0 && pip check",
    )
    .run(
        # Upstream 2.0.0 omits sm_120 from its explicit JIT architecture list.
        # Fail if this exact pinned-source patch no longer matches.
        "/opt/protenix-env/bin/python -c " + shlex.quote(
            "from pathlib import Path; "
            "p = Path('/opt/protenix-env/lib/python3.12/site-packages/protenix/model/layer_norm/torch_ext_compile.py'); "
            "s = p.read_text(); old = '(\"100\", \"100\"),'; "
            "assert s.count(old) == 1, 'Protenix CUDA architecture patch no longer matches'; "
            "p.write_text(s.replace(old, old + '\\n        (\"120\", \"120\"),'))"
        ),
        # Import the real runner. This also compiles and bakes its CUDA layer
        # norm extension, with bounded CPU parallelism and no GPU inference.
        "CUDA_HOME=/usr/local/cuda MAX_JOBS=2 "
        "/opt/protenix-env/bin/python -c "
        "'import torch; import runner.batch_inference; "
        "assert torch.__version__ == \"2.7.1+cu128\"; assert torch.version.cuda == \"12.8\"'",
    )
    .run("apt-get clean && rm -rf /var/lib/apt/lists/*")
    .env(
        CUDA_HOME="/usr/local/cuda",
        PATH="/opt/protenix-env/bin:/usr/local/cuda/bin:/usr/local/hmmer/bin:$PATH",
        LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH",
        VIRTUAL_ENV="/opt/protenix-env",
    )
    .label(Version="2.0.0", CUDA="12.8", PyTorch="2.7.1")
    .runscript('"$@"')
    .template_command(
        exec_mode="exec",
        binds=[],
        env={},
    )
)


###########################################################################
# Block
###########################################################################


class ProtenixBlock(AF3LikePredictionBlock):
    """Protenix biomolecular structure prediction (v2).

    Predicts structures for protein complexes, protein-ligand,
    protein-nucleic acid, and other biomolecular systems using ByteDance's
    Protenix model. Supports multiple model variants from v2 (464M) to
    tiny (109M) for different accuracy/speed tradeoffs.

    v2 adds Training-Free Guidance (TFG) for geometric constraint
    enforcement during diffusion sampling without retraining.

    Basic usage builds Protenix input JSON automatically from the upstream
    design's atom array. For custom inputs (enzyme design, constraints,
    covalent bonds), use ``template_json`` to supply a hand-crafted
    Protenix JSON template and the block will fill in designed sequences.
    """

    _name = "protenix"
    _description = "Protenix structure prediction"
    _script_spec = ScriptSpec(
        command=["protenix", "pred"],
        image=PROTENIX_CONTAINER,
        source_url="https://github.com/bytedance/Protenix",
    )
    _resources = JobArgs(
        partition="gpu",
        gpu_class="small",
        mem_gb=32,
        rows_per_job=200,
    )

    # --- Protenix-specific fields ---

    model_name: str = Field(
        "protenix_base_20250630_v1.0.0",
        json_schema_extra={"tier": TIER_PRIMARY},
        description=(
            "Model checkpoint name. Options: "
            "protenix-v2 (464M, best accuracy, TFG support), "
            "protenix_base_20250630_v1.0.0 (368M, default, June 2025 data cutoff), "
            "protenix_base_default_v1.0.0 (368M, Sept 2021 data cutoff), "
            "protenix_mini_esm_v0.5.0 (~135M, fast + ESM), "
            "protenix_mini_ism_v0.5.0 (~135M, fast + ISM), "
            "protenix_mini_default_v0.5.0 (~134M, fast), "
            "protenix_tiny_default_v0.5.0 (~109M, ultra-light)."
        ),
    )
    seeds: str = Field(
        "101",
        json_schema_extra={"tier": TIER_PRIMARY},
        description=(
            "Comma-separated random seeds for independent runs. "
            "Each seed runs a completely independent prediction. "
            "Example: '101,102,103' for 3 independent runs."
        ),
    )
    dtype: str = Field(
        "bf16",
        description="Precision: 'bf16' (faster, default) or 'fp32' (more precise).",
    )

    # MSA / templates
    use_msa: bool = Field(
        False,
        description=(
            "Enable multiple sequence alignment features. "
            "Requires HMMER databases accessible via /databases bind mount. "
            "Default False for design workflows where MSA is unnecessary."
        ),
    )
    use_template: bool = Field(
        False,
        description="Use template structures for guided prediction.",
    )
    use_tfg_guidance: bool = Field(
        False,
        description=(
            "Enable Training-Free Guidance (TFG) during diffusion sampling. "
            "TFG enforces geometric constraints (steric, torsion, bond, "
            "pairwise distance) without model retraining. Requires v2 model."
        ),
    )

    # Confidence
    need_atom_confidence: bool = Field(
        False,
        description="Output per-atom confidence scores (generates full_data JSON).",
    )

    # Performance tuning
    enable_cache: bool = Field(
        True,
        description="Cache shareable diffusion variables for faster inference.",
    )
    enable_fusion: bool = Field(
        True,
        description="Enable kernel fusion optimizations.",
    )
    enable_tf32: bool = Field(
        True,
        description="Enable TF32 for FP32 matrix multiplications (faster on Ampere+).",
    )
    trimul_kernel: str = Field(
        "cuequivariance",
        description="Triangle multiplication kernel: 'cuequivariance' or 'torch'.",
    )
    triatt_kernel: str = Field(
        "cuequivariance",
        description=(
            "Triangle attention kernel: 'cuequivariance', 'deepspeed', 'triattention', or 'torch'."
        ),
    )

    # Cache/weights
    cache_dir: str = Field(
        default_factory=lambda: get_setting_field(
            "PROTENIX_CACHE_DIR", os.path.expanduser("~/.protenix")
        ),
        description="Directory for Protenix model weights (auto-downloaded on first run).",
    )

    # Template JSON input
    template_json: Optional[str] = Field(
        None,
        description=(
            "Path to a template Protenix JSON file. If provided, protein "
            "sequences from upstream designs are filled into this template "
            "while preserving ligands, constraints, and covalent bonds."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _remap_legacy_fields(cls, data):
        data = cls._remap_legacy_samples_field(data, "sample")
        data = cls._remap_legacy_field(data, "step", "num_diffusion_steps")
        data = cls._remap_legacy_field(data, "cycle", "n_recycles")
        return data

    @model_validator(mode="after")
    def _check_unsupported(self):
        self._assert_unsupported_fields_default(ckpt_path=None)
        return self

    @model_validator(mode="after")
    def _validate_template_json(self):
        if self.template_json is not None and not os.path.exists(self.template_json):
            raise ValueError(f"template_json file does not exist: {self.template_json}")
        return self

    def get_command_template(self) -> list[str]:
        self.cache_dir = str(Path(self.cache_dir).expanduser().resolve())
        env = self._jit_safe_env()
        env.update(
            PROTENIX_ROOT_DIR=self.cache_dir,
            CUEQ_TRITON_CACHE_DIR="/tmp/cuequivariance-triton",
            TRITON_CACHE_DIR="/tmp/triton",
            TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor",
            MAX_JOBS="2",
        )
        return self._script_spec.image.build_command(
            *self._script_spec.command, env=env,
            binds=[(self.cache_dir, self.cache_dir)],
        )

    def create_command(self, handler, sandbox_dir, chunk_id) -> list[str]:
        try:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise OSError(f"Cannot prepare PROTENIX_ROOT_DIR cache directory: {self.cache_dir}") from error
        sandbox_dir, inputdir, outputdir = self._setup_io_dirs(sandbox_dir)

        # Write JSON inputs for each design in the chunk
        for loaded in handler.get_prediction_inputs(
            which=self.which, load_context=True
        ):
            json_file = os.path.join(inputdir, f"{loaded.basename}.json")
            if os.path.exists(json_file):
                logger.info(f"Skipping existing input: {json_file}")
                continue

            atom_array = loaded.atom_array

            # Fix missing ligand bonds for PDB inputs (same as RF3 pattern)
            if self.specified_smiles and is_pdb_file(loaded.path):
                atom_array = pdb_assign_ligand_bonds(atom_array, self.specified_smiles)

            if self.template_json:
                json_data = fill_protenix_json_from_template(
                    template_json=self.template_json,
                    atom_array=atom_array,
                    name=loaded.basename,
                    protein_sequences=loaded.sequences,
                )
            else:
                json_data = create_protenix_json_from_design(
                    atom_array=atom_array,
                    name=loaded.basename,
                    remove_ligands=self.remove_ligands,
                    specified_smiles=self.specified_smiles,
                    specified_ccd=self.specified_ccd,
                    disable_ccd_detection=self.disable_ccd_detection,
                    protein_sequences=loaded.sequences,
                )

            with open(json_file, "w") as f:
                json.dump(json_data, f, indent=4)

        # Build command
        cmd = self.command_template + [
            "--input",
            inputdir,
            "--out_dir",
            outputdir,
            "--model_name",
            self.model_name,
            "--seeds",
            self.seeds,
            "--step",
            str(self.num_diffusion_steps),
            "--sample",
            str(self.diffusion_batch_size),
            "--dtype",
            self.dtype,
            "--trimul_kernel",
            self.trimul_kernel,
            "--triatt_kernel",
            self.triatt_kernel,
        ]

        if self.n_recycles is not None:
            cmd += ["--cycle", str(self.n_recycles)]

        if not self.use_msa:
            cmd += ["--use_msa", "False"]

        if self.use_template:
            cmd += ["--use_template", "True"]

        if self.use_tfg_guidance:
            cmd += ["--use_tfg_guidance", "True"]

        if self.need_atom_confidence:
            cmd += ["--need_atom_confidence", "True"]

        if not self.enable_cache:
            cmd += ["--enable_cache", "False"]

        if not self.enable_fusion:
            cmd += ["--enable_fusion", "False"]

        if not self.enable_tf32:
            cmd += ["--enable_tf32", "False"]

        return cmd

    def compile(self, sandbox_dir) -> PipelineDataHandler:
        sandbox_dir = os.path.abspath(sandbox_dir)
        outputdir = os.path.join(sandbox_dir, "outputs")

        # Protenix outputs: outputs/<dataset>/<design>/seed_<N>/predictions/<name>_sample_<rank>.cif
        paths = glob_designs_in_dir(outputdir, recursive=True, extensions=("*.cif",))

        self._guard_outputs(paths, outputdir)

        if self.read_most_confident_model:
            paths = _select_best_protenix_models(paths)

        rows = []
        for path in sorted(paths):
            metrics = _load_protenix_confidence(path)

            rows.append(
                make_row(
                    name=self.name,
                    path=str(path),
                    metadata={},
                    metrics={self.name: metrics},
                    as_metric=self.compile_as_metric,
                )
            )

        return PipelineDataHandler(rows)


###########################################################################
# Output helpers
###########################################################################


def _load_protenix_confidence(cif_path: str) -> dict:
    """Load confidence metrics from the companion summary JSON for a CIF file.

    Protenix naming: ``{name}_sample_{rank}.cif`` pairs with
    ``{name}_summary_confidence_sample_{rank}.json``.
    """
    metrics = {}
    basename = os.path.splitext(os.path.basename(cif_path))[0]
    # Replace only the LAST occurrence of _sample_ — design names from
    # earlier pipeline stages may themselves contain _sample_ tokens
    # (e.g. "design_sample_0__5_1_sample_3.cif").  Using str.replace()
    # would corrupt all of them; rsplit + rejoin targets only the suffix
    # that Protenix appended.
    parts = basename.rsplit("_sample_", 1)
    if len(parts) == 2:
        conf_name = parts[0] + "_summary_confidence_sample_" + parts[1]
    else:
        conf_name = basename.replace("_sample_", "_summary_confidence_sample_")
    conf_path = os.path.join(os.path.dirname(cif_path), f"{conf_name}.json")

    if os.path.exists(conf_path):
        try:
            with open(conf_path) as f:
                conf_data = json.load(f)
            if "ranking_score" in conf_data:
                metrics["ranking_score"] = conf_data["ranking_score"]
            # Extract summary_metrics if present
            summary = conf_data.get("summary_metrics", conf_data)
            for key in [
                "ptm",
                "iptm",
                "plddt",
                "protein_iptm",
                "ligand_iptm",
                "protein_plddt",
                "ligand_plddt",
                "complex_plddt",
                "has_clash",
            ]:
                if key in summary:
                    metrics[key] = summary[key]
        except Exception as e:
            logger.warning(f"Failed to load confidence from {conf_path}: {e}")

    return metrics


def _select_best_protenix_models(paths: list[str]) -> list[str]:
    """Select the highest-ranking-score model per design across all seeds/samples.

    Groups CIF files by design name (everything before ``_sample_``) and
    picks the one with the best ``ranking_score`` from the companion JSON.
    """

    def _design_name(path: str):
        basename = os.path.splitext(os.path.basename(path))[0]
        parts = basename.rsplit("_sample_", 1)
        return parts[0] if len(parts) == 2 else basename

    def _score(path: str):
        return _load_protenix_confidence(path).get("ranking_score")

    return select_best_models(paths, group_fn=_design_name, score_fn=_score)
