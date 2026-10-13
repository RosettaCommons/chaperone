"""Boltz-2 structure prediction block."""

import glob
import json
import logging
import os
from typing import Optional

import yaml
from pydantic import Field, model_validator

from blocks.constants import CUDA_INSTALL
from blocks.networks.prediction.helpers import (
    create_boltz_yaml_from_design,
    fill_boltz_yaml_from_template,
)
from chaperone import PipelineDataHandler, JobArgs, get_setting_field
from chaperone.block_config import AF3LikePredictionBlock, TIER_PRIMARY, ScriptSpec
from chaperone.block_config.prediction import (
    MIN_INTERFACE_IPTM_KEY,
    select_best_models,
)
from chaperone.common import glob_designs_in_dir
from chaperone.containers import Image
from chaperone.data import make_row

logger = logging.getLogger(__name__)

###########################################################################
# Container
###########################################################################

BOLTZ2_CONTAINER = (
    Image("boltz2")
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
    )
    .run(*CUDA_INSTALL)
    .run(
        "python3.12 -m venv /opt/boltz2-env",
        ". /opt/boltz2-env/bin/activate && pip install --upgrade pip setuptools wheel",
    )
    .run(
        ". /opt/boltz2-env/bin/activate && "
        "pip install torch --index-url https://download.pytorch.org/whl/cu126",
    )
    .run(
        ". /opt/boltz2-env/bin/activate && pip install 'boltz[cuda]'",
    )
    .run("apt-get clean && rm -rf /var/lib/apt/lists/*")
    .env(
        CUDA_HOME="/usr/local/cuda",
        PATH="/opt/boltz2-env/bin:/usr/local/cuda/bin:$PATH",
        LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH",
        VIRTUAL_ENV="/opt/boltz2-env",
    )
    .label(Version="v2.2.1", CUDA="12.6")
    .runscript('"$@"')
    .template_command(
        exec_mode="exec",
        binds=[("/mnt/home", "/mnt/home"), ("/projects", "/projects")],
        env={"BOLTZ_CACHE_DIR": lambda: get_setting_field("BOLTZ_CACHE_DIR", "")},
    )
)


###########################################################################
# Block
###########################################################################


class Boltz2Block(AF3LikePredictionBlock):
    """Boltz-2 biomolecular structure prediction.

    Predicts structures for protein complexes, protein-ligand,
    protein-nucleic acid, and other biomolecular systems. Also supports
    binding affinity prediction for protein-ligand complexes.
    """

    _name = "boltz2"
    _display_name = "Boltz-2"
    _description = "Boltz-2 structure prediction"
    _script_spec = ScriptSpec(
        command=["boltz", "predict"],
        image=BOLTZ2_CONTAINER,
        source_url="https://github.com/jwohlwend/boltz",
    )
    _resources = JobArgs(
        partition="gpu",
        gpu_class="small",
        mem_gb=32,
        rows_per_job=200,
    )

    # n_recycles and num_diffusion_steps inherited from AF3LikePredictionBlock
    override: bool = Field(
        True,
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Override existing predictions if found.",
    )
    max_parallel_samples: int = Field(
        1, description="Max samples to predict in parallel."
    )
    step_scale: float = Field(
        1.638,
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Diffusion step scale. Lower = more diversity (recommended 1-2).",
    )
    output_format: str = Field("mmcif", description="Output format: 'mmcif' or 'pdb'.")

    # MSA
    use_msa_server: bool = Field(
        False, description="Auto-generate MSAs via mmseqs2 server."
    )
    msa_server_url: Optional[str] = Field(
        None, description="MSA server URL (default: ColabFold)."
    )

    # Potentials
    use_potentials: bool = Field(
        False,
        description="Use inference-time potentials for better physical plausibility.",
    )

    # Affinity
    predict_affinity: bool = Field(
        False, description="Enable affinity prediction for protein-ligand complexes."
    )

    # Model/cache paths
    cache_dir: str = Field(
        default_factory=lambda: get_setting_field(
            "BOLTZ_CACHE_DIR", os.path.expanduser("~/.boltz")
        ),
        description="Directory containing Boltz model weights and CCD data.",
    )

    # Input conversion options
    template_yaml: Optional[str] = Field(
        None,
        description="Path to a template YAML file. If provided, sequences from "
        "upstream designs are filled into this template.",
    )
    no_kernels: bool = Field(
        False, description="Disable cuEquivariance kernels (for older GPUs)."
    )

    @model_validator(mode="before")
    @classmethod
    def _remap_legacy_fields(cls, data):
        data = cls._remap_legacy_samples_field(data, "diffusion_samples")
        data = cls._remap_legacy_field(data, "sampling_steps", "num_diffusion_steps")
        data = cls._remap_legacy_field(data, "recycling_steps", "n_recycles")
        return data

    @model_validator(mode="after")
    def _check_unsupported(self):
        self._assert_unsupported_fields_default(
            ckpt_path=None,
        )
        return self

    def create_command(self, handler, sandbox_dir, chunk_id) -> list[str]:
        sandbox_dir = os.path.abspath(sandbox_dir)
        inputdir = os.path.join(sandbox_dir, "inputs")
        os.makedirs(inputdir, exist_ok=True)

        # Write YAML inputs for each design in the chunk
        for loaded in handler.get_prediction_inputs(
            which=self.which, load_context=True
        ):
            yaml_file = os.path.join(inputdir, f"{loaded.basename}.yaml")
            if os.path.exists(yaml_file):
                logger.info(f"Skipping existing input: {yaml_file}")
                continue

            if self.template_yaml:
                yaml_data = fill_boltz_yaml_from_template(
                    template_yaml=self.template_yaml,
                    atom_array=loaded.atom_array,
                    protein_sequences=loaded.sequences,
                )
            else:
                yaml_data = create_boltz_yaml_from_design(
                    atom_array=loaded.atom_array,
                    remove_ligands=self.remove_ligands,
                    specified_smiles=self.specified_smiles,
                    specified_ccd=self.specified_ccd,
                    disable_ccd_detection=self.disable_ccd_detection,
                    predict_affinity=self.predict_affinity,
                    protein_sequences=loaded.sequences,
                )

            with open(yaml_file, "w") as f:
                yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

        # Build command
        cache_dir = self.cache_dir
        cmd = self.command_template + [
            inputdir,
            "--out_dir",
            sandbox_dir,
            "--cache",
            cache_dir,
            "--sampling_steps",
            str(self.num_diffusion_steps),
            "--diffusion_samples",
            str(self.diffusion_batch_size),
            "--max_parallel_samples",
            str(self.max_parallel_samples),
            "--step_scale",
            str(self.step_scale),
            "--output_format",
            self.output_format,
        ]

        if self.n_recycles is not None:
            cmd += ["--recycling_steps", str(self.n_recycles)]

        if self.use_msa_server:
            cmd.append("--use_msa_server")
            if self.msa_server_url:
                cmd.extend(["--msa_server_url", self.msa_server_url])

        if self.use_potentials:
            cmd.append("--use_potentials")

        if self.override:
            cmd.append("--override")

        if self.no_kernels:
            cmd.append("--no_kernels")

        return cmd

    def compile(self, sandbox_dir) -> PipelineDataHandler:
        sandbox_dir = os.path.abspath(sandbox_dir)

        # Boltz wraps output in boltz_results_<input_dir_stem>/ subdirectory
        # e.g. sandbox_dir/boltz_results_inputs/predictions/<name>/<name>_model_0.cif
        results_dirs = glob.glob(os.path.join(sandbox_dir, "boltz_results_*"))
        if results_dirs:
            predictions_dir = os.path.join(results_dirs[0], "predictions")
        else:
            predictions_dir = os.path.join(sandbox_dir, "predictions")

        # Collect all structure files
        exts = ("*.cif",) if self.output_format == "mmcif" else ("*.pdb",)
        paths = glob_designs_in_dir(predictions_dir, recursive=True, extensions=exts)

        self._guard_outputs(paths, predictions_dir)

        if self.read_most_confident_model:
            paths = _select_best_boltz2_models(paths)

        rows = []
        for path in sorted(paths):
            basename = os.path.splitext(os.path.basename(path))[0]
            # e.g. "design_name_model_0" -> parent dir is the design name
            parent_dir = os.path.dirname(path)

            # Load confidence metrics
            metrics = {}
            conf_files = glob.glob(
                os.path.join(parent_dir, f"confidence_*{basename}*.json")
            )
            if not conf_files:
                # Try matching by model name pattern
                conf_files = glob.glob(os.path.join(parent_dir, "confidence_*.json"))
            for conf_file in conf_files:
                try:
                    with open(conf_file) as f:
                        conf_data = json.load(f)
                    metrics.update(
                        {
                            k: conf_data[k]
                            for k in [
                                "confidence_score",
                                "ptm",
                                "iptm",
                                "ligand_iptm",
                                "protein_iptm",
                                "complex_plddt",
                                "complex_iplddt",
                                "complex_pde",
                                "complex_ipde",
                            ]
                            if k in conf_data
                        }
                    )
                    # Compute min_pae_interaction from pair_chains_iptm
                    pair_iptm = conf_data.get("pair_chains_iptm", {})
                    if len(pair_iptm) > 1:
                        off_diag = [
                            pair_iptm[i][j]
                            for i in pair_iptm
                            for j in pair_iptm[i]
                            if i != j and pair_iptm[i][j] is not None
                        ]
                        if off_diag:
                            # ipTM-derived interface metric (NOT PAE — Boltz2's
                            # confidence output has no chain-pair PAE matrix).
                            metrics[MIN_INTERFACE_IPTM_KEY] = min(off_diag)
                except Exception as e:
                    logger.warning(f"Failed to load confidence from {conf_file}: {e}")

            # Load affinity metrics if present
            aff_files = glob.glob(os.path.join(parent_dir, "affinity_*.json"))
            for aff_file in aff_files:
                try:
                    with open(aff_file) as f:
                        aff_data = json.load(f)
                    metrics.update(
                        {
                            k: aff_data[k]
                            for k in [
                                "affinity_pred_value",
                                "affinity_probability_binary",
                            ]
                            if k in aff_data
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to load affinity from {aff_file}: {e}")

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


def _select_best_boltz2_models(paths: list[str]) -> list[str]:
    """Select the highest-confidence model per design.

    Groups CIF/PDB files by their parent directory (one dir per design)
    and picks the model with the best ``confidence_score`` from the
    companion JSON.
    """

    def _score(path: str):
        basename = os.path.splitext(os.path.basename(path))[0]
        conf_file = os.path.join(os.path.dirname(path), f"confidence_{basename}.json")
        if os.path.exists(conf_file):
            try:
                with open(conf_file) as f:
                    return json.load(f).get("confidence_score")
            except Exception:
                return None
        return None

    return select_best_models(paths, group_fn=os.path.dirname, score_fn=_score)


if __name__ == "__main__":
    from pathlib import Path

    REPO_ROOT = Path(__file__).resolve().parents[4]
    TEST_DATA = REPO_ROOT / "tests" / "test_data" / "unittests" / "boltz2"
    SANDBOX = REPO_ROOT / "unittest_outs" / "boltz2"
    SANDBOX.mkdir(parents=True, exist_ok=True)

    block = Boltz2Block(num_diffusion_samples=1, diffusion_batch_size=1)
    handler = PipelineDataHandler.from_paths(
        [str(TEST_DATA / "design_0.pdb")], name="mpnn"
    )
    handler = block(handler, sandbox_dir=SANDBOX, chunk_id=0)
    print(handler.unpack())
