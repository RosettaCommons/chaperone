"""OpenFold 3 structure prediction block."""

import glob
import json
import logging
import os
from pathlib import Path
from typing import Optional

from pydantic import Field, model_validator

from blocks.constants import CUDA_INSTALL
from blocks.networks.prediction.helpers import (
    create_openfold3_json_from_design,
    fill_openfold3_json_from_template,
)
from chaperone import JobArgs, PipelineDataHandler, get_setting_field
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

OPENFOLD3_CONTAINER = (
    Image("openfold3")
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
        "python3.12 -m venv /opt/openfold3-env",
        ". /opt/openfold3-env/bin/activate && pip install --upgrade pip setuptools wheel",
    )
    .run(
        ". /opt/openfold3-env/bin/activate && "
        "pip install torch --index-url https://download.pytorch.org/whl/cu126",
    )
    .git_clone("https://github.com/aqlaboratory/openfold-3.git", "/opt/openfold3")
    .run(
        ". /opt/openfold3-env/bin/activate && cd /opt/openfold3 && pip install .",
    )
    .run("apt-get clean && rm -rf /var/lib/apt/lists/*")
    .env(
        CUDA_HOME="/usr/local/cuda",
        PATH="/opt/openfold3-env/bin:/usr/local/cuda/bin:$PATH",
        LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH",
        VIRTUAL_ENV="/opt/openfold3-env",
    )
    .label(Version="0.4.0", CUDA="12.6")
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


class OpenFold3Block(AF3LikePredictionBlock):
    """OpenFold 3 biomolecular structure prediction.

    Open-source structure prediction model (Apache 2.0) from the
    AlQuraishi Laboratory. Predicts structures for protein complexes,
    protein-ligand, protein-nucleic acid, and other biomolecular systems.

    Uses the ``run_openfold predict`` CLI with JSON-based input.
    Model weights are downloaded via ``setup_openfold`` to the cache_dir.
    """

    _name = "openfold3"
    _description = "OpenFold 3 structure prediction"
    _script_spec = ScriptSpec(
        command=["run_openfold", "predict"],
        image=OPENFOLD3_CONTAINER,
        source_url="https://github.com/aqlaboratory/openfold-3",
    )
    _resources = JobArgs(
        partition="gpu",
        gpu_class="small",
        mem_gb=64,
        rows_per_job=200,
    )

    # --- OpenFold3-specific fields ---

    num_model_seeds: int = Field(
        1,
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Number of model seeds to use for each query.",
    )
    use_msa_server: bool = Field(
        False,
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Use ColabFold MSA server for alignments. False for single-sequence prediction.",
    )
    use_templates: bool = Field(
        False,
        description="Use ColabFold MSA server for template alignments.",
    )

    template_json: Optional[str] = Field(
        None,
        description="Path to template JSON file for OpenFold3 input. If provided, "
        "sequences from upstream designs are filled into this template.",
    )

    cache_dir: str = Field(
        default_factory=lambda: get_setting_field(
            "OPENFOLD3_CACHE_DIR", os.path.expanduser("~/.openfold3")
        ),
        description="Directory containing OpenFold3 model weights.",
    )

    runner_yaml: Optional[str] = Field(
        None,
        description="Path to a runner YAML for advanced model/dataset configuration.",
    )

    def get_command_template(self) -> list[str]:
        env = self._jit_safe_env()
        # DeepSpeed JIT-compiles evoformer CUDA kernels; needs CUTLASS headers
        env["CUTLASS_PATH"] = "DS_USE_CUTLASS_PYTHON_BINDINGS"
        if self.cache_dir:
            self.cache_dir = str(Path(self.cache_dir).expanduser().resolve())
            env["OPENFOLD_CACHE"] = self.cache_dir
        return self._script_spec.image.build_command(
            *self._script_spec.command, env=env or None,
            binds=[(self.cache_dir, self.cache_dir)] if self.cache_dir else [],
        )

    @model_validator(mode="before")
    @classmethod
    def _remap_legacy_fields(cls, data):
        data = cls._remap_legacy_samples_field(data, "num_diffusion_samples")
        return data

    @model_validator(mode="after")
    def _check_unsupported(self):
        self._assert_unsupported_fields_default(
            n_recycles=None,
            num_diffusion_steps=200,
        )
        return self

    @model_validator(mode="after")
    def validate_template_json_exists(self):
        if self.template_json is not None and not os.path.exists(self.template_json):
            raise ValueError(f"template_json file does not exist: {self.template_json}")
        return self

    def create_command(self, handler, sandbox_dir, chunk_id) -> list[str]:
        sandbox_dir, inputdir, outputdir = self._setup_io_dirs(sandbox_dir)
        if self.cache_dir:
            Path(self.cache_dir).expanduser().resolve().mkdir(parents=True, exist_ok=True)

        # Build the queries dict with one entry per design
        queries = {}
        for loaded in handler.get_prediction_inputs(
            which=self.which, load_context=True
        ):
            if self.template_json:
                json_data = fill_openfold3_json_from_template(
                    template_json=self.template_json,
                    atom_array=loaded.atom_array,
                    name=loaded.basename,
                    protein_sequences=loaded.sequences,
                )
                # Template returns full InferenceQuerySet structure
                queries.update(json_data.get("queries", {loaded.basename: json_data}))
            else:
                query_data = create_openfold3_json_from_design(
                    atom_array=loaded.atom_array,
                    name=loaded.basename,
                    remove_ligands=self.remove_ligands,
                    specified_smiles=self.specified_smiles,
                    specified_ccd=self.specified_ccd,
                    disable_ccd_detection=self.disable_ccd_detection,
                    protein_sequences=loaded.sequences,
                )
                queries.update(query_data)

        # Write combined InferenceQuerySet JSON
        query_set = {"seeds": [42], "queries": queries}
        json_path = os.path.join(inputdir, "openfold3_input.json")
        with open(json_path, "w") as f:
            json.dump(query_set, f, indent=4)

        # Build command
        cmd = self.command_template + [
            f"--query_json={json_path}",
            f"--output_dir={outputdir}",
        ]

        if self.diffusion_batch_size != 5:
            cmd.append(f"--num_diffusion_samples={self.diffusion_batch_size}")

        if self.num_model_seeds != 1:
            cmd.append(f"--num_model_seeds={self.num_model_seeds}")

        cmd.append(f"--use_msa_server={self.use_msa_server}")
        cmd.append(f"--use_templates={self.use_templates}")

        if self.ckpt_path:
            cmd.append(f"--inference_ckpt_path={self.ckpt_path}")

        if self.runner_yaml:
            cmd.append(f"--runner_yaml={self.runner_yaml}")

        return cmd

    def compile(self, sandbox_dir) -> PipelineDataHandler:
        sandbox_dir = os.path.abspath(sandbox_dir)
        outputdir = os.path.join(sandbox_dir, "outputs")

        # Output structure: {outputdir}/{query_id}/seed_{N}/{query_id}_seed_{N}_sample_{M}_model.pdb
        paths = glob_designs_in_dir(outputdir, recursive=True)

        self._guard_outputs(paths, outputdir)

        if self.read_most_confident_model:
            paths = _select_best_openfold3_models(paths, outputdir)

        rows = []
        for path in sorted(paths):
            metrics = _load_openfold3_confidence(path)

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


def _openfold3_stem(path: str) -> str:
    """Strip extension and ``_model`` suffix from an OpenFold3 output filename."""
    stem = os.path.basename(path)
    for ext in (".cif", ".pdb"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    if stem.endswith("_model"):
        stem = stem[: -len("_model")]
    return stem


def _load_openfold3_confidence(path: str) -> dict:
    """Load confidence metrics from the companion aggregated JSON for a structure file."""
    metrics = {}
    path_dir = os.path.dirname(path)
    stem = _openfold3_stem(path)

    conf_path = os.path.join(path_dir, f"{stem}_confidences_aggregated.json")
    if not os.path.exists(conf_path):
        # Fallback: glob for any aggregated confidence file
        candidates = glob.glob(os.path.join(path_dir, "*confidences_aggregated*.json"))
        if candidates:
            conf_path = candidates[0]

    if os.path.exists(conf_path):
        try:
            with open(conf_path) as f:
                data = json.load(f)

            # Extract scalar metrics
            for key in [
                "avg_plddt",
                "gpde",
                "ptm",
                "iptm",
                "disorder",
                "has_clash",
                "sample_ranking_score",
            ]:
                if key in data:
                    metrics[key] = data[key]

            # Rename avg_plddt → plddt for consistency with other blocks
            if "avg_plddt" in metrics:
                metrics["plddt"] = metrics.pop("avg_plddt")

            # Rename sample_ranking_score → ranking_score for consistency
            if "sample_ranking_score" in metrics:
                metrics["ranking_score"] = metrics.pop("sample_ranking_score")

            # ipTM-derived interface metric (NOT PAE — OpenFold3's confidence
            # output exposes only a chain-pair ipTM matrix, no PAE matrix).
            chain_pair_iptm = data.get("chain_pair_iptm", {})
            if len(chain_pair_iptm) > 1:
                off_diag = [v for k, v in chain_pair_iptm.items() if v is not None]
                if off_diag:
                    metrics[MIN_INTERFACE_IPTM_KEY] = min(off_diag)

        except Exception as e:
            logger.warning(
                f"Failed to load OpenFold3 confidences from {conf_path}: {e}"
            )

    return metrics


def _select_best_openfold3_models(paths, outputdir):
    """Select the best sample per query based on ranking score."""

    def _query_id(p: str):
        return os.path.relpath(p, outputdir).split(os.sep)[0]

    def _score(p: str):
        stem = _openfold3_stem(p)
        conf_path = os.path.join(
            os.path.dirname(p), f"{stem}_confidences_aggregated.json"
        )
        if os.path.exists(conf_path):
            try:
                with open(conf_path) as f:
                    return json.load(f).get("sample_ranking_score")
            except Exception:
                return None
        return None

    return select_best_models(paths, group_fn=_query_id, score_fn=_score)
