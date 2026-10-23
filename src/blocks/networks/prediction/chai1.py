"""Chai-1 structure prediction block."""

import glob
import logging
import os
import shlex
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import Field, model_validator

from blocks.constants import CUDA_INSTALL

from blocks.networks.prediction.helpers import create_chai_fasta_from_design
from chaperone import PipelineDataHandler, JobArgs, get_setting_field
from chaperone.block_config import AF3LikePredictionBlock, ScriptSpec
from chaperone.block_config.prediction import select_best_models
from chaperone.containers import Image
from chaperone.data import make_row

logger = logging.getLogger(__name__)

###########################################################################
# Container
###########################################################################

# Chai 0.6.1 caps Torch below 2.7, whose CUDA wheels lack Blackwell kernels.
# The site's working stack uses 2.7.1/cu128. Preserve Chai's metadata and check
# every other dependency normally; do not hide unrelated resolver conflicts.
_CHAI_INSTALL_DEPENDENCIES = """
import importlib.metadata as metadata
import subprocess
import sys
from packaging.requirements import Requirement
requirements = [value for value in metadata.requires('chai_lab') if Requirement(value).name != 'torch']
subprocess.run([sys.executable, '-m', 'pip', 'install', *requirements, 'numpy==1.26.4', 'torch==2.7.1'], check=True)
"""
_CHAI_DEPENDENCY_CHECK = """
import subprocess
import sys
result = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True)
print(result.stdout, end='')
print(result.stderr, end='', file=sys.stderr)
expected = 'chai-lab 0.6.1 has requirement torch<2.7,>=2.3.1, but you have torch 2.7.1+cu128.'
if result.returncode != 1 or result.stdout.strip().splitlines() != [expected]:
    raise RuntimeError('Unexpected Chai dependency conflict; only the documented Torch 2.7.1/cu128 override is allowed')
print('Accepted documented Chai 0.6.1 Torch override for Blackwell; all other dependencies checked')
"""

CHAI1_CONTAINER = (
    Image("chai1")
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
        "python3.12 -m venv /opt/chai1-env",
        ". /opt/chai1-env/bin/activate && pip install --upgrade 'pip>=26.2.2' setuptools wheel",
    )
    .run(
        ". /opt/chai1-env/bin/activate && "
        "pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128",
    )
    .run(
        ". /opt/chai1-env/bin/activate && pip install --no-deps 'chai_lab==0.6.1' && pip install packaging",
        "/opt/chai1-env/bin/python -c " + shlex.quote(f"exec({_CHAI_INSTALL_DEPENDENCIES!r})"),
    )
    .run(
        "/opt/chai1-env/bin/python -c " + shlex.quote(f"exec({_CHAI_DEPENDENCY_CHECK!r})"),
        '/opt/chai1-env/bin/python -c \'import torch; from chai_lab.chai1 import run_inference; assert torch.version.cuda == "12.8"\'',
    )
    .run("apt-get clean && rm -rf /var/lib/apt/lists/*")
    .env(
        CUDA_HOME="/usr/local/cuda",
        PATH="/opt/chai1-env/bin:/usr/local/cuda/bin:$PATH",
        LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH",
        VIRTUAL_ENV="/opt/chai1-env",
    )
    .label(
        Version="0.6.1", CUDA="12.8", PyTorch="2.7.1",
        DependencyOverride="chai_lab-0.6.1_torch-lt2.7_override-2.7.1-cu128",
    )
    .runscript('"$@"')
    .template_command(
        exec_mode="exec",
        binds=[],
        env={"CHAI_DOWNLOADS_DIR": lambda: get_setting_field("CHAI_DOWNLOADS_DIR", "")},
    )
)


###########################################################################
# Block
###########################################################################


class Chai1Block(AF3LikePredictionBlock):
    """Chai-1 biomolecular structure prediction.

    Predicts structures for protein complexes, protein-ligand,
    protein-nucleic acid, and other biomolecular systems using Chai-1.
    """

    _name = "chai1"
    _display_name = "Chai-1"
    _description = "Chai-1 structure prediction"
    _script_spec = ScriptSpec(
        command=["chai-lab", "fold"],
        image=CHAI1_CONTAINER,
        source_url="https://github.com/chaidiscovery/chai-lab",
    )
    _resources = JobArgs(
        partition="gpu",
        gpu_class="small",
        mem_gb=32,
        rows_per_job=200,
    )

    # n_recycles and num_diffusion_steps inherited from AF3LikePredictionBlock

    seed: Optional[int] = Field(
        None,
        description="Random seed for reproducibility. None = random.",
    )
    use_esm_embeddings: bool = Field(
        True, description="Use ESM embeddings for proteins."
    )
    use_msa_server: bool = Field(
        False, description="Auto-generate MSAs via ColabFold mmseqs2 server."
    )
    msa_server_url: Optional[str] = Field(
        None, description="MSA server URL (default: ColabFold)."
    )
    low_memory: bool = Field(True, description="Use memory-efficient inference mode.")

    # Model/cache paths
    cache_dir: Optional[str] = Field(
        default_factory=lambda: get_setting_field(
            "CHAI_DOWNLOADS_DIR", None
        ) or os.path.expanduser("~/.chai"),
        description="Directory for Chai-1 model weights. Defaults to CHAI_DOWNLOADS_DIR setting.",
    )

    @model_validator(mode="before")
    @classmethod
    def _remap_legacy_fields(cls, data):
        data = cls._remap_legacy_samples_field(data, "num_diffn_samples")
        data = cls._remap_legacy_field(
            data, "num_diffn_timesteps", "num_diffusion_steps"
        )
        data = cls._remap_legacy_field(data, "num_trunk_recycles", "n_recycles")
        return data

    @model_validator(mode="after")
    def _check_unsupported(self):
        self._assert_unsupported_fields_default(ckpt_path=None)
        return self

    def get_command_template(self) -> list[str]:
        self.cache_dir = str(Path(self.cache_dir or "~/.chai").expanduser().resolve())
        env = {**self._jit_safe_env(), "CHAI_DOWNLOADS_DIR": self.cache_dir}
        return self._script_spec.image.build_command(
            *self._script_spec.command,
            env=env,
            binds=[(self.cache_dir, self.cache_dir)],
        )

    def create_command(self, handler, sandbox_dir, chunk_id) -> list[str]:
        try:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise OSError(f"Cannot prepare CHAI_DOWNLOADS_DIR cache directory: {self.cache_dir}") from error
        sandbox_dir, inputdir, outputdir = self._setup_io_dirs(sandbox_dir)

        # Write FASTA inputs for each design in the chunk
        for loaded in handler.get_prediction_inputs(
            which=self.which, load_context=True
        ):
            fasta_file = os.path.join(inputdir, f"{loaded.basename}.fasta")
            if os.path.exists(fasta_file):
                logger.info(f"Skipping existing input: {fasta_file}")
                continue

            fasta_content = create_chai_fasta_from_design(
                atom_array=loaded.atom_array,
                name=loaded.basename,
                remove_ligands=self.remove_ligands,
                specified_smiles=self.specified_smiles,
                specified_ccd=self.specified_ccd,
                disable_ccd_detection=self.disable_ccd_detection,
                protein_sequences=loaded.sequences,
            )

            with open(fasta_file, "w") as f:
                f.write(fasta_content)

        # Build per-input commands
        # Chai-1 CLI takes a single fasta + output_dir per invocation
        fasta_files = sorted(glob.glob(os.path.join(inputdir, "*.fasta")))
        subcmds = []
        for fasta_file in fasta_files:
            basename = os.path.splitext(os.path.basename(fasta_file))[0]
            design_outdir = os.path.join(outputdir, basename)
            os.makedirs(design_outdir, exist_ok=True)

            subcmd = self.command_template + [
                fasta_file,
                design_outdir,
                "--num-diffn-samples",
                str(self.diffusion_batch_size),
                "--num-diffn-timesteps",
                str(self.num_diffusion_steps),
            ]

            if self.n_recycles is not None:
                subcmd.extend(["--num-trunk-recycles", str(self.n_recycles)])

            if self.seed is not None:
                subcmd.extend(["--seed", str(self.seed)])

            if not self.use_esm_embeddings:
                subcmd.append("--no-use-esm-embeddings")

            if self.use_msa_server:
                subcmd.append("--use-msa-server")
                if self.msa_server_url:
                    subcmd.extend(["--msa-server-url", self.msa_server_url])

            if not self.low_memory:
                subcmd.append("--no-low-memory")

            subcmds.append(subcmd)

        # Single design: return flat token list (shell=False).
        # Multiple designs: join with literal && tokens in a flat list.
        # NOTE: do NOT use ["bash", "-c", compound_string] — Manifest
        # builds command_str via " ".join(command) which loses quoting
        # around the -c argument, causing the shell to misparse it.
        if len(subcmds) == 1:
            return subcmds[0]
        flat_cmd: list[str] = []
        for i, subcmd in enumerate(subcmds):
            flat_cmd.extend(subcmd)
            if i < len(subcmds) - 1:
                flat_cmd.append("&&")
        return flat_cmd

    def compile(self, sandbox_dir) -> PipelineDataHandler:
        sandbox_dir = os.path.abspath(sandbox_dir)
        outputdir = os.path.join(sandbox_dir, "outputs")

        # Chai-1 outputs: outputs/<design>/pred.model_idx_N.cif
        paths = glob.glob(os.path.join(outputdir, "*", "pred.model_idx_*.cif"))

        self._guard_outputs(paths, outputdir)

        if self.read_most_confident_model:
            paths = select_best_chai_models(paths)

        rows = []
        for path in sorted(paths):
            metrics = {}

            # Design name is the parent directory (outputs/<design>/pred.model_idx_N.cif)
            design_name = os.path.basename(os.path.dirname(path))
            model_idx = os.path.splitext(os.path.basename(path))[0]  # pred.model_idx_N

            # Symlink output CIF with design name so merge can match
            # pred.model_idx_0.cif -> <design>_model.cif (or <design>_model_idx_N.cif)
            suffix = "_model" if self.read_most_confident_model else f"_{model_idx}"
            linked_name = f"{design_name}{suffix}.cif"
            linked_path = os.path.join(os.path.dirname(path), linked_name)
            if not os.path.exists(linked_path):
                os.symlink(os.path.basename(path), linked_path)

            # Load scores from companion .npz file
            # pred.model_idx_0.cif -> scores.model_idx_0.npz
            score_path = path.replace("pred.model_idx_", "scores.model_idx_").replace(
                ".cif", ".npz"
            )
            if os.path.exists(score_path):
                try:
                    scores = np.load(score_path, allow_pickle=True)
                    # Chai-1 .npz stores all values with a leading batch
                    # dimension of 1 (e.g. shape (1,) not scalar).  Squeeze
                    # it before converting so scalar metrics are captured.
                    for key in [
                        "aggregate_score",
                        "ptm",
                        "iptm",
                        "per_chain_ptm",
                        "per_chain_pair_iptm",
                    ]:
                        if key in scores:
                            val = np.squeeze(scores[key])
                            if val.ndim == 0:
                                metrics[key] = float(val)
                            elif key in ("per_chain_ptm",):
                                metrics[key] = float(np.mean(val))
                    # pLDDT from CIF B-factors is more reliable, but
                    # aggregate_score is the primary ranking metric
                    if "aggregate_score" in metrics:
                        metrics["ranking_score"] = metrics["aggregate_score"]
                except Exception as e:
                    logger.warning(f"Failed to load scores from {score_path}: {e}")

            rows.append(
                make_row(
                    name=self.name,
                    path=str(linked_path),
                    metadata={},
                    metrics={self.name: metrics},
                    as_metric=self.compile_as_metric,
                )
            )

        return PipelineDataHandler(rows)


def select_best_chai_models(paths):
    """Select the highest-scoring model per design directory (Chai-1)."""

    def _score(path: str):
        score_path = path.replace("pred.model_idx_", "scores.model_idx_").replace(
            ".cif", ".npz"
        )
        if os.path.exists(score_path):
            try:
                scores = np.load(score_path, allow_pickle=True)
                if "aggregate_score" in scores:
                    return float(np.squeeze(scores["aggregate_score"]))
            except Exception:
                return None
        return None

    return select_best_models(paths, group_fn=os.path.dirname, score_fn=_score)


if __name__ == "__main__":
    from pathlib import Path

    REPO_ROOT = Path(__file__).resolve().parents[4]
    TEST_DATA = REPO_ROOT / "tests" / "test_data" / "unittests" / "chai1"
    SANDBOX = REPO_ROOT / "unittest_outs" / "chai1"
    SANDBOX.mkdir(parents=True, exist_ok=True)

    block = Chai1Block(num_diffusion_samples=1, diffusion_batch_size=1)
    handler = PipelineDataHandler.from_paths(
        [str(TEST_DATA / "design_0.pdb")], name="mpnn"
    )
    handler = block(handler, sandbox_dir=SANDBOX, chunk_id=0)
    print(handler.unpack())
