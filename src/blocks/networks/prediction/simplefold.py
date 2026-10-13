"""Apple SimpleFold sequence-only protein structure prediction block."""

from __future__ import annotations

import glob
import logging
import math
import os
import re
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from blocks.constants import make_cuda_install
from blocks.networks.prediction.monomer_reference import (
    load_monomer_references,
    stage_monomer_manifest,
)
from chaperone import JobArgs, PipelineDataHandler, get_setting_field
from chaperone.block_config import AF3LikePredictionBlock, ScriptSpec
from chaperone.block_config.base import TIER_PRIMARY, TIER_SECONDARY
from chaperone.containers import Image
from chaperone.data import make_row
from chaperone.io import parse_to_atom_array

logger = logging.getLogger(__name__)

SIMPLEFOLD_COMMIT = "c7a5570a6be9f5c695126e27c804e77567209934"
SIMPLEFOLD_RUNNER_PATH = (
    Path(__file__).parent.parent.parent / "scripts" / "run_simplefold.py"
)
SIMPLEFOLD_RUNNER_DEST = "/opt/simplefold/run_simplefold.py"
SAMPLE_SUFFIX = re.compile(r"^(?P<target>.+)_sampled_(?P<sample>\d+)$")


def _write_fasta(path: Path, target: str, sequence: str) -> None:
    """Write one unwrapped FASTA record for SimpleFold's directory loader."""
    path.write_text(f">{target}\n{sequence}\n")


###########################################################################
# Container
###########################################################################

SIMPLEFOLD_CONTAINER = (
    Image("simplefold")
    .base_image("ubuntu:22.04")
    .packages(
        "build-essential",
        "ca-certificates",
        "curl",
        "git",
        "gnupg",
        "python3.10",
        "python3.10-dev",
        "python3.10-venv",
        "python3-pip",
    )
    .run(*make_cuda_install("12-6", ubuntu="2204"))
    .run(
        "python3.10 -m venv /opt/simplefold-env",
        ". /opt/simplefold-env/bin/activate && pip install --upgrade pip setuptools wheel",
    )
    .run(
        ". /opt/simplefold-env/bin/activate && "
        "pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126 && "
        "pip install fair-esm==2.0.0"
    )
    .git_clone(
        "https://github.com/apple/ml-simplefold.git", "/opt/simplefold", branch="main"
    )
    .run(
        f"cd /opt/simplefold && git checkout {SIMPLEFOLD_COMMIT}",
        ". /opt/simplefold-env/bin/activate && cd /opt/simplefold && pip install .",
    )
    .copy(str(SIMPLEFOLD_RUNNER_PATH), SIMPLEFOLD_RUNNER_DEST)
    .run("apt-get clean && rm -rf /var/lib/apt/lists/*")
    .env(
        CUDA_HOME="/usr/local/cuda",
        PATH="/opt/simplefold-env/bin:/usr/local/cuda/bin:$PATH",
        LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH",
        VIRTUAL_ENV="/opt/simplefold-env",
    )
    .label(Version=SIMPLEFOLD_COMMIT, CUDA="12.6", Source="apple/ml-simplefold")
    .runscript('"$@"')
    .template_command(
        exec_mode="exec",
        binds=[
            ("/home", "/home"),
            ("/mnt/home", "/mnt/home"),
            ("/net", "/net"),
            ("/mnt/net", "/mnt/net"),
            ("/projects", "/projects"),
        ],
        env={},
    )
)


###########################################################################
# Block
###########################################################################


class SimpleFoldBlock(AF3LikePredictionBlock):
    """Apple SimpleFold single-sequence, protein-monomer structure prediction.

    The released model has no ligand, nucleic-acid, multimer, template, or MSA
    input. Non-protein atoms are removed and multi-chain protein inputs fail
    explicitly. Compile canonicalizes raw outputs to CIF through AtomArrays;
    scoring lives in the ``score_val_structures`` block, not here.
    """

    _name = "simplefold"
    _display_name = "SimpleFold"
    _description = "Apple SimpleFold sequence-only protein monomer prediction"
    _script_spec = ScriptSpec(
        command=["python", SIMPLEFOLD_RUNNER_DEST],
        image=SIMPLEFOLD_CONTAINER,
        source_url="https://github.com/apple/ml-simplefold",
    )
    _resources = JobArgs(
        partition="gpu",
        gpu_class="large",
        mem_gb=96,
        time_limit="24:00:00",
        rows_per_job=50,
        cpus_per_task=8,
    )

    diffusion_batch_size: int = Field(
        1,
        ge=1,
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Number of independently sampled conformers per protein.",
    )
    num_diffusion_steps: int = Field(
        500,
        ge=1,
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Number of Euler-Maruyama flow-sampling steps.",
    )
    model: Literal[
        "simplefold_100M",
        "simplefold_360M",
        "simplefold_700M",
        "simplefold_1.1B",
        "simplefold_1.6B",
        "simplefold_3B",
    ] = Field(
        "simplefold_100M",
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Released SimpleFold folding checkpoint.",
    )
    tau: float = Field(
        0.01,
        gt=0,
        json_schema_extra={"tier": TIER_SECONDARY},
        description="Diffusion coefficient scaling factor.",
    )
    seed: int = Field(
        42,
        json_schema_extra={"tier": TIER_SECONDARY},
        description="Random seed; chunk id is added to keep chunks independent.",
    )
    predict_plddt: bool = Field(
        False,
        json_schema_extra={"tier": TIER_SECONDARY},
        description=(
            "Run the optional pLDDT head. This additionally loads a fixed 1.6B "
            "folding model and is substantially more memory intensive."
        ),
    )
    raw_output_format: Literal["pdb", "mmcif"] = Field(
        "pdb",
        description="SimpleFold's raw output format before compile-time conversion.",
    )
    convert_outputs_to_cif: bool = Field(
        True,
        description="Parse raw predictions into AtomArrays and write canonical CIF outputs.",
    )
    weights_dir: str = Field(
        default_factory=lambda: get_setting_field(
            "SIMPLEFOLD_WEIGHTS_DIR", str(Path.home() / ".cache" / "simplefold")
        ),
        description="Shared directory for folding, ESM2, and CCD downloads.",
    )

    @model_validator(mode="after")
    def _check_supported_fields(self):
        self._assert_unsupported_fields_default(
            n_recycles=None,
            ckpt_path=None,
            remove_ligands=False,
            specified_smiles={},
            specified_ccd={},
            disable_ccd_detection=False,
        )
        if not os.path.isabs(self.weights_dir):
            raise ValueError(f"weights_dir must be absolute: {self.weights_dir}")
        if (
            self.read_most_confident_model
            and self.diffusion_batch_size > 1
            and not self.predict_plddt
        ):
            logger.warning(
                "SimpleFold cannot rank multiple samples without predict_plddt; "
                "compile will keep sample 0"
            )
        return self

    def _manifest_dir(self, sandbox_dir) -> Path:
        return Path(sandbox_dir) / f"{self.name}_manifest"

    def get_command_template(self) -> list[str]:
        env = self._jit_safe_env()
        env["TORCH_HOME"] = os.path.join(self.weights_dir, "torch")
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        return self._script_spec.image.build_command(
            *self._script_spec.command, env=env
        )

    def create_command(self, handler, sandbox_dir, chunk_id) -> list[str]:
        _, inputdir, outputdir = self._setup_io_dirs(sandbox_dir)

        for loaded in handler.get_data(which=self.which, load=True):
            manifest = stage_monomer_manifest(
                loaded,
                manifest_dir=self._manifest_dir(sandbox_dir),
            )
            _write_fasta(
                Path(inputdir) / f"{manifest['target']}.fasta",
                manifest["target"],
                manifest["sequence"],
            )

        cmd = self.command_template + [
            "--simplefold-model",
            self.model,
            "--weights-dir",
            self.weights_dir,
            "--output-dir",
            outputdir,
            "--num-steps",
            str(self.num_diffusion_steps),
            "--tau",
            str(self.tau),
            "--fasta-path",
            inputdir,
            "--num-samples",
            str(self.diffusion_batch_size),
            "--output-format",
            self.raw_output_format,
            "--seed",
            str(self.seed + int(chunk_id)),
        ]
        if self.predict_plddt:
            cmd.append("--predict-plddt")
        return cmd

    def compile(self, sandbox_dir) -> PipelineDataHandler:
        sandbox = Path(sandbox_dir).resolve()
        prediction_dir = sandbox / "outputs" / f"predictions_{self.model}"
        raw_paths = sorted(
            glob.glob(str(prediction_dir / "*.pdb"))
            + glob.glob(str(prediction_dir / "*.cif"))
        )
        self._guard_outputs(raw_paths, str(prediction_dir))

        manifests = load_monomer_references(self._manifest_dir(sandbox))
        parsed = []
        for raw_path in raw_paths:
            raw = Path(raw_path)
            match = SAMPLE_SUFFIX.fullmatch(raw.stem)
            if match is None:
                raise ValueError(f"Unexpected SimpleFold output name: {raw.name}")
            target = match.group("target")
            sample_index = int(match.group("sample"))
            if target not in manifests:
                raise KeyError(f"No SimpleFold manifest for output target {target!r}")
            predicted = parse_to_atom_array(raw, assign_types=True, remove_ccds=[])
            parsed.append((target, sample_index, raw, predicted, manifests[target]))

        if self.read_most_confident_model:
            selected = {}
            for item in parsed:
                target, sample_index, _, predicted, _ = item
                score = (
                    float(np.mean(predicted.b_factor))
                    if self.predict_plddt
                    and "b_factor" in predicted.get_annotation_categories()
                    else -float(sample_index)
                )
                if target not in selected or score > selected[target][0]:
                    selected[target] = (score, item)
            parsed = [value[1] for value in selected.values()]

        compiled_dir = sandbox / "outputs" / "compiled_cif"
        if self.convert_outputs_to_cif:
            compiled_dir.mkdir(parents=True, exist_ok=True)
            from atomworks.io.utils.io_utils import to_cif_file

            from chaperone.io import ensure_bonds

        rows = []
        for target, sample_index, raw, predicted, manifest in sorted(
            parsed, key=lambda item: (item[0], item[1])
        ):
            output_path = raw
            if self.convert_outputs_to_cif:
                output_path = compiled_dir / f"{raw.stem}.cif"
                to_cif_file(ensure_bonds(predicted), output_path)

            metrics: dict[str, object] = {
                "model": self.model,
                "sample_index": sample_index,
                "num_diffusion_steps": self.num_diffusion_steps,
                "tau": self.tau,
                "dropped_nonprotein_atoms": manifest["dropped_nonprotein_atoms"],
            }
            if (
                self.predict_plddt
                and "b_factor" in predicted.get_annotation_categories()
            ):
                plddt = float(np.mean(predicted.b_factor))
                if not math.isfinite(plddt):
                    raise ValueError(f"Non-finite SimpleFold pLDDT for {raw}")
                metrics["plddt"] = plddt
            rows.append(
                make_row(
                    name=self.name,
                    path=output_path,
                    path_handle=manifest["source_path"],
                    metadata={},
                    metrics={self.name: metrics},
                    as_metric=self.compile_as_metric,
                )
            )
        return PipelineDataHandler(rows)
