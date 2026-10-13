"""ESMFold2 structure prediction block.

ESMFold2 (Biohub) is a protein folding oracle built on an ESMC trunk with a
diffusion structure decoder. It supports single-sequence inference and
optional MSA conditioning, co-folds proteins with optional ligands, and
reports per-residue pLDDT plus pTM / ipTM.

Both the model and its ``transformers.models.esmfold2`` implementation live in
Biohub forks (upstream ``transformers`` has no ESMFold2), so the container
installs ``git+https://github.com/Biohub/transformers`` and
``git+https://github.com/Biohub/esm``. Weights are not baked into the image —
they are read from ``HF_HUB_CACHE`` (default ``/projects/ml/esm-weights``).
"""

import csv
import glob
import logging
import os
from pathlib import Path
from typing import Optional

from pydantic import Field, model_validator

from blocks.constants import make_cuda_install
from blocks.networks.prediction.helpers import create_esmfold2_spec_from_design
from blocks.networks.prediction.monomer_reference import (
    default_msa_dirs,
    normalize_msa_dirs,
    resolve_msa_path,
    stage_query_a3m,
)
from pipelines import JobArgs, PipelineDataHandler, get_setting_field
from pipelines.block_config import AF3LikePredictionBlock, ScriptSpec
from pipelines.block_config.base import TIER_PRIMARY, TIER_SECONDARY
from pipelines.containers import Image
from pipelines.data import make_row

logger = logging.getLogger(__name__)

ESMFOLD2_RUNNER_PATH = (
    Path(__file__).parent.parent.parent / "scripts" / "run_esmfold2_runner.py"
)
ESMFOLD2_RUNNER_DEST = "/opt/esmfold2/run_esmfold2_runner.py"

###########################################################################
# Container
###########################################################################

# torch 2.12 / CUDA 13.0 (matches the working esmc install); the ESMFold2 model
# lives only in the Biohub transformers fork, installed before the esm fork.
ESMFOLD2_CONTAINER = (
    Image("esmfold2")
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
    .run(*make_cuda_install("13-0"))
    .run(
        "python3.12 -m venv /opt/esmfold2-env",
        ". /opt/esmfold2-env/bin/activate && pip install --upgrade pip setuptools wheel",
    )
    .run(
        ". /opt/esmfold2-env/bin/activate && "
        "pip install torch --index-url https://download.pytorch.org/whl/cu130",
    )
    .run(
        # transformers fork first — it provides transformers.models.esmfold2,
        # which the esm fork imports at runtime.
        ". /opt/esmfold2-env/bin/activate && "
        'pip install "git+https://github.com/Biohub/transformers.git@main" && '
        'pip install "git+https://github.com/Biohub/esm.git@main"',
    )
    .run("mkdir -p /opt/esmfold2")
    .copy(str(ESMFOLD2_RUNNER_PATH), ESMFOLD2_RUNNER_DEST)
    .run("apt-get clean && rm -rf /var/lib/apt/lists/*")
    .env(
        CUDA_HOME="/usr/local/cuda",
        PATH="/opt/esmfold2-env/bin:/usr/local/cuda/bin:$PATH",
        LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH",
        VIRTUAL_ENV="/opt/esmfold2-env",
    )
    .label(Version="esm-3.3.0/tfm-4.57.6", CUDA="13.0")
    .runscript('"$@"')
    .template_command(
        exec_mode="exec",
        binds=[
            ("/home", "/home"),
            ("/mnt/home", "/mnt/home"),
            ("/projects", "/projects"),
        ],
        # The weights cache is supplied per run (ESM_WEIGHTS_DIR), not baked in.
        env={"HF_HUB_CACHE": lambda: get_setting_field("ESM_WEIGHTS_DIR", "")},
    )
)


###########################################################################
# Block
###########################################################################


class ESMFold2Block(AF3LikePredictionBlock):
    """ESMFold2 structure prediction with optional MSA conditioning.

    Folds protein chains (optionally with ligands) using the Biohub ESMFold2
    model. Inherited diffusion fields map onto the ESMFold2 API:
    ``diffusion_batch_size`` → ``num_diffusion_samples`` (samples batched in one
    forward pass) and ``num_diffusion_steps`` → ``num_sampling_steps``.
    ``num_loops`` is the trunk refinement count (API default 3; the reference
    driver used 20 for heavier refinement).
    """

    _name = "esmfold2"
    _display_name = "ESMFold2"
    _description = "ESMFold2 structure prediction with optional MSA conditioning"
    _script_spec = ScriptSpec(
        command=["python", str(ESMFOLD2_RUNNER_PATH)],
        image=ESMFOLD2_CONTAINER,
        source_url="https://github.com/Biohub/esm",
    )
    _resources = JobArgs(
        partition="gpu",
        gres="gpu:small:1",
        mem_gb=48,
        rows_per_job=200,
    )

    # diffusion_batch_size (→ num_diffusion_samples) and num_diffusion_steps
    # (→ num_sampling_steps) inherited from AF3LikePredictionBlock.

    num_loops: int = Field(
        3,
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Trunk refinement loops (API default 3; reference driver used 20).",
    )
    seed: Optional[int] = Field(
        0,
        json_schema_extra={"tier": TIER_SECONDARY},
        description="Random seed for input prep + diffusion sampling.",
    )
    model: str = Field(
        "biohub/ESMFold2",
        json_schema_extra={"tier": TIER_SECONDARY},
        description="HuggingFace model id (or local snapshot path) for ESMFold2.",
    )
    use_local_msas: bool = Field(
        False,
        description=(
            "Condition each protein chain on its precomputed A3M from the "
            "on-disk MSA stores. False runs ESMFold2 in single-sequence mode."
        ),
    )
    msa_dirs: Optional[list] = Field(
        None,
        description=(
            "Sharded MSA roots searched in order, as paths or "
            "{dir, extension, directory_depth} dicts. None uses "
            "PROTEIN_MSA_DIRS."
        ),
    )
    max_msa_sequences: Optional[int] = Field(
        16_384,
        ge=1,
        le=16_384,
        description="Maximum A3M depth supplied to ESMFold2's MSA encoder.",
    )
    raise_if_missing_msa_for_protein_of_length_n: Optional[int] = Field(
        None,
        ge=0,
        description=(
            "Fail when a protein chain at least this many residues long has no "
            "on-disk MSA instead of silently using single-sequence mode."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _remap_legacy_fields(cls, data):
        data = cls._remap_legacy_samples_field(data, "num_diffusion_samples")
        data = cls._remap_legacy_field(
            data, "num_sampling_steps", "num_diffusion_steps"
        )
        return data

    @model_validator(mode="after")
    def _check_unsupported(self):
        # ESMFold2 has no trunk-recycle or external checkpoint knob.
        self._assert_unsupported_fields_default(n_recycles=None, ckpt_path=None)
        if self.use_local_msas:
            if self._resolved_msa_dirs() is None:
                raise ValueError(
                    "use_local_msas requires msa_dirs or the PROTEIN_MSA_DIRS setting"
                )
        elif self.msa_dirs is not None:
            raise ValueError("msa_dirs has no effect unless use_local_msas is True")
        if (
            self.raise_if_missing_msa_for_protein_of_length_n is not None
            and not self.use_local_msas
        ):
            raise ValueError(
                "raise_if_missing_msa_for_protein_of_length_n requires "
                "use_local_msas"
            )
        return self

    def _resolved_msa_dirs(self) -> Optional[list]:
        if self.msa_dirs is not None:
            return normalize_msa_dirs(self.msa_dirs)
        return default_msa_dirs()

    def _stage_msas(self, spec: dict, inputdir: str) -> None:
        """Resolve, verify, and stage one A3M per protein chain in ``spec``."""
        if not self.use_local_msas:
            return

        msa_dirs = self._resolved_msa_dirs()
        if not msa_dirs:
            raise ValueError(
                "use_local_msas=True but no MSA roots are configured; set msa_dirs "
                "or PROTEIN_MSA_DIRS"
            )

        for chain_index, chain in enumerate(spec["chains"]):
            sequence = chain["sequence"]
            msa_path = resolve_msa_path(sequence, msa_dirs)
            if msa_path is None:
                threshold = self.raise_if_missing_msa_for_protein_of_length_n
                if threshold is not None and len(sequence) >= threshold:
                    raise ValueError(
                        f"No on-disk MSA for {spec['name']} chain {chain['id']} "
                        f"({len(sequence)} residues)"
                    )
                logger.warning(
                    "No on-disk MSA for %s chain %s (%d residues); ESMFold2 will "
                    "run that chain single-sequence",
                    spec["name"],
                    chain["id"],
                    len(sequence),
                )
                chain["msa_depth"] = 1
                continue

            staged = Path(inputdir) / f"{spec['name']}.chain_{chain_index}.a3m"
            depth = stage_query_a3m(
                msa_path,
                staged,
                sequence=sequence,
                max_sequences=self.max_msa_sequences,
            )
            chain["msa_path"] = str(staged)
            chain["msa_depth"] = depth

    def get_command_template(self) -> list[str]:
        env = self._jit_safe_env()
        # The runner folds every chunked design in one process; without this
        # the caching allocator fragments across iterations and OOMs partway
        # through a chunk on smaller GPUs.
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        weights_dir = get_setting_field("ESM_WEIGHTS_DIR", "")
        if weights_dir:
            env["HF_HUB_CACHE"] = weights_dir
        # Weights are pre-staged in that cache by design, but huggingface_hub
        # still resolves the revision on load and writes a refs/ pointer back
        # into the cache directory. On a shared, full filesystem that 40-byte
        # write is what fails, and it surfaces as
        #   OSError: [Errno 28] No space left on device
        # from deep inside transformers — nothing about it suggests the model
        # was present and readable the whole time. Offline mode skips the
        # revision lookup and the write; if a model is genuinely absent it now
        # says so instead of blaming the disk.
        env.setdefault("HF_HUB_OFFLINE", "1")
        return self._script_spec.image.build_command(
            *self._script_spec.command, env=env
        )

    def create_command(self, handler, sandbox_dir, chunk_id) -> list[str]:
        sandbox_dir, inputdir, outputdir = self._setup_io_dirs(sandbox_dir)

        import json

        for loaded in handler.get_prediction_inputs(
            which=self.which, load_context=True
        ):
            spec_file = os.path.join(inputdir, f"{loaded.basename}.json")
            if os.path.exists(spec_file):
                logger.info(f"Skipping existing input: {spec_file}")
                continue
            spec = create_esmfold2_spec_from_design(
                atom_array=loaded.atom_array,
                name=loaded.basename,
                remove_ligands=self.remove_ligands,
                specified_smiles=self.specified_smiles,
                specified_ccd=self.specified_ccd,
                disable_ccd_detection=self.disable_ccd_detection,
                protein_sequences=loaded.sequences,
            )
            self._stage_msas(spec, inputdir)
            with open(spec_file, "w") as f:
                json.dump(spec, f)

        cmd = self.command_template + [
            "--inputs",
            inputdir,
            "--outdir",
            outputdir,
            "--model",
            self.model,
            "--num-loops",
            str(self.num_loops),
            "--num-sampling-steps",
            str(self.num_diffusion_steps),
            "--num-diffusion-samples",
            str(self.diffusion_batch_size),
        ]
        if self.seed is not None:
            cmd += ["--seed", str(self.seed)]
        if not self.read_most_confident_model:
            cmd.append("--all-samples")
        return cmd

    def compile(self, sandbox_dir) -> PipelineDataHandler:
        sandbox_dir = os.path.abspath(sandbox_dir)
        outputdir = os.path.join(sandbox_dir, "outputs")

        paths = glob.glob(os.path.join(outputdir, "*.cif"))
        self._guard_outputs(paths, outputdir)

        # Metrics keyed by output stem from summary.csv.
        metrics_by_name = {}
        summary = os.path.join(outputdir, "summary.csv")
        if os.path.exists(summary):
            with open(summary, newline="") as f:
                for row in csv.DictReader(f):
                    name = row.get("name")
                    if not name:
                        continue

                    def _f(key):
                        try:
                            return float(row[key])
                        except (KeyError, TypeError, ValueError):
                            return None

                    metrics_by_name[name] = {
                        "plddt": _f("mean_plddt"),
                        "ptm": _f("ptm"),
                        "iptm": _f("iptm"),
                        "msa_depth": _f("msa_depth"),
                    }

        rows = []
        for path in sorted(paths):
            stem = os.path.splitext(os.path.basename(path))[0]
            metrics = {
                k: v for k, v in metrics_by_name.get(stem, {}).items() if v is not None
            }
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


if __name__ == "__main__":
    REPO_ROOT = Path(__file__).resolve().parents[4]
    TEST_DATA = REPO_ROOT / "tests" / "test_data" / "unittests" / "mpnn"
    SANDBOX = REPO_ROOT / "unittest_outs" / "esmfold2"
    SANDBOX.mkdir(parents=True, exist_ok=True)

    block = ESMFold2Block(diffusion_batch_size=1, num_loops=3)
    handler = PipelineDataHandler.from_paths(
        [str(TEST_DATA / "design_0.pdb")], name="mpnn"
    )
    handler = block(handler, sandbox_dir=SANDBOX, chunk_id=0)
    print(handler.unpack())
