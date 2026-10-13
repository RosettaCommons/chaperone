import os
import glob
import csv
import shutil
import logging

from pydantic import Field, field_validator
from typing import Optional

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

BOLTZGEN_CONTAINER = (
    cuda_python_base(
        Image("boltzgen"),
        "boltzgen",
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
        ". /opt/boltzgen-env/bin/activate && "
        "pip install torch --index-url https://download.pytorch.org/whl/cu126 && "
        "pip install boltzgen"
    )
    .run("apt-get clean && rm -rf /var/lib/apt/lists/*")
    .label(Version="v0.3.0", CUDA="12.6")
    .runscript('"$@"')
    .template_command(
        exec_mode="exec",
        binds=[("/mnt/home", "/mnt/home"), ("/projects", "/projects")],
        env={
            "HF_HUB_CACHE": lambda: get_setting_field("HF_HUB_CACHE", ""),
            # Site-provided job scratch can disappear between the outer Slurm
            # task and BoltzGen's multiprocessing workers.  Keep worker sockets
            # in stable, node-local storage instead.
            "TMPDIR": "/tmp",
        },
    )
)


class BoltzGenBlock(CompiledBlockFunction):
    """BoltzGen block for protein design.

    Runs the BoltzGen pipeline (design -> inverse folding -> folding -> analysis -> filtering).
    Takes a design specification YAML as input.
    """

    _name = "boltzgen"
    _display_name = "BoltzGen"
    _description = "BoltzGen protein design block"
    _script_spec = ScriptSpec(
        command=["boltzgen"],
        image=BOLTZGEN_CONTAINER,
        source_url="https://github.com/baker-laboratory/BoltzGen",
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

    # Primary input - design specification YAML
    config: str = Field(
        ...,
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Path to BoltzGen design specification YAML file",
    )

    # Pipeline configuration
    protocol: str = Field(
        "protein-anything",
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Design protocol (protein-anything, peptide-anything, protein-small_molecule, nanobody-anything, antibody-anything, protein-redesign)",
    )
    num_designs: int = Field(
        100,
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Total number of designs to generate",
    )
    budget: int = Field(
        30,
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Number of final designs after filtering",
    )
    steps: Optional[list[str]] = Field(
        None,
        description="Pipeline steps to run (default: all). Options: design, inverse_folding, design_folding, folding, affinity, analysis, filtering",
    )

    # Design parameters
    diffusion_batch_size: Optional[int] = Field(
        None,
        description="Designs per trunk run (default: auto)",
    )
    step_scale: Optional[float] = Field(
        None,
        description="Fixed diffusion step scale",
    )
    noise_scale: Optional[float] = Field(
        None,
        description="Fixed noise scale",
    )

    # Inverse folding
    skip_inverse_folding: bool = Field(
        False,
        description="Skip the inverse folding step",
    )
    inverse_fold_num_sequences: int = Field(
        1,
        description="Number of sequences per backbone",
    )

    # Filtering
    alpha: Optional[float] = Field(
        None,
        description="Quality vs diversity (0.0=quality, 1.0=diversity)",
    )

    # Execution
    devices: Optional[int] = Field(
        None,
        description="Number of GPUs (default: all available)",
    )
    reuse: bool = Field(
        False,
        description="Reuse existing results from previous runs",
    )
    n_batches: int = Field(
        1,
        description="Number of independent BoltzGen runs",
    )

    @field_validator("config")
    def ensure_config_exists(cls, v):
        if not os.path.exists(v):
            raise ValueError(f"Config file {v} does not exist")
        return os.path.abspath(v)

    def init_handler(self):
        rows = []
        for _ in range(self.n_batches):
            rows.append({Column.STRUCTURES: [], Column.METRICS: "{}"})
        return PipelineDataHandler(rows)

    def create_command(self, handler, sandbox_dir, chunk_id):
        sandbox_dir = os.path.abspath(sandbox_dir)
        os.makedirs(sandbox_dir, exist_ok=True)

        # Copy config into sandbox for reproducibility
        config_dest = os.path.join(sandbox_dir, "design_spec.yaml")
        shutil.copy2(self.config, config_dest)

        # Split designs across chunks
        designs_per_chunk = self.num_designs // self.n_batches
        if chunk_id < (self.num_designs % self.n_batches):
            designs_per_chunk += 1

        cmd = self.command_template + [
            "run",
            config_dest,
            "--output",
            sandbox_dir,
            "--protocol",
            self.protocol,
            "--num_designs",
            str(designs_per_chunk),
            "--budget",
            str(self.budget),
        ]

        if self.diffusion_batch_size is not None:
            cmd += ["--diffusion_batch_size", str(self.diffusion_batch_size)]
        if self.step_scale is not None:
            cmd += ["--step_scale", str(self.step_scale)]
        if self.noise_scale is not None:
            cmd += ["--noise_scale", str(self.noise_scale)]
        if self.skip_inverse_folding:
            cmd += ["--skip_inverse_folding"]
        if self.inverse_fold_num_sequences != 1:
            cmd += [
                "--inverse_fold_num_sequences",
                str(self.inverse_fold_num_sequences),
            ]
        if self.alpha is not None:
            cmd += ["--alpha", str(self.alpha)]
        if self.devices is not None:
            cmd += ["--devices", str(self.devices)]
        if self.reuse:
            cmd += ["--reuse"]
        if self.steps is not None:
            cmd += ["--steps"] + self.steps

        return cmd

    def compile(self, sandbox_dir):
        return compile_boltzgen(sandbox_dir, name=self.name, budget=self.budget)


def compile_boltzgen(sandbox_dir, name="boltzgen", budget=30):
    """Compile BoltzGen outputs into a PipelineDataHandler."""
    sandbox_dir = str(sandbox_dir)

    # Search output directories in preference order:
    # 1. Final ranked designs (filtered + diverse)
    # 2. Refolded complexes
    # 3. Inverse-folded designs
    # 4. Raw intermediate designs
    search_dirs = [
        os.path.join(sandbox_dir, "final_ranked_designs", f"final_{budget}_designs"),
        os.path.join(sandbox_dir, "intermediate_designs_inverse_folded", "refold_cif"),
        os.path.join(sandbox_dir, "intermediate_designs_inverse_folded"),
        os.path.join(sandbox_dir, "intermediate_designs"),
    ]

    paths = []
    for search_dir in search_dirs:
        if os.path.isdir(search_dir):
            paths = glob.glob(os.path.join(search_dir, "*.cif"))
            if paths:
                break

    if not paths:
        raise RuntimeError(
            f"BoltzGen completed but no output CIF files found in {sandbox_dir}. "
            f"Check the output directory for logs or errors."
        )

    # Find associated metrics CSV
    metrics_csv = None
    final_csv = os.path.join(
        sandbox_dir, "final_ranked_designs", f"final_designs_metrics_{budget}.csv"
    )
    agg_csv = os.path.join(
        sandbox_dir,
        "intermediate_designs_inverse_folded",
        "aggregate_metrics_analyze.csv",
    )
    if os.path.exists(final_csv):
        metrics_csv = final_csv
    elif os.path.exists(agg_csv):
        metrics_csv = agg_csv

    # Load per-design metrics
    per_design_metrics = {}
    if metrics_csv:
        per_design_metrics = _load_boltzgen_metrics(metrics_csv)

    rows = []
    for path in sorted(paths):
        basename = os.path.splitext(os.path.basename(path))[0]
        design_metrics = per_design_metrics.get(basename, {})

        rows.append(
            make_row(
                name=name,
                path=path,
                metrics={name: design_metrics} if design_metrics else {},
            )
        )

    return PipelineDataHandler(rows)


def _load_boltzgen_metrics(csv_path):
    """Load per-design metrics from a BoltzGen CSV."""
    metrics = {}
    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                design_name = row.get("design_name") or row.get("name")
                if design_name:
                    design_name = os.path.splitext(os.path.basename(design_name))[0]
                    metrics[design_name] = {
                        k: v for k, v in row.items() if k not in ("design_name", "name")
                    }
    except Exception as e:
        logger.warning(f"Failed to load BoltzGen metrics from {csv_path}: {e}")
    return metrics


if __name__ == "__main__":
    from pathlib import Path

    REPO_ROOT = Path(__file__).resolve().parents[4]
    TEST_DATA = REPO_ROOT / "tests" / "test_data" / "unittests" / "boltzgen"
    SANDBOX = REPO_ROOT / "unittest_outs" / "boltzgen"
    SANDBOX.mkdir(parents=True, exist_ok=True)

    block = BoltzGenBlock(
        config=str(TEST_DATA / "design_spec.yaml"),
        num_designs=1,
        budget=1,
        n_batches=1,
    )
    handler = block.init_handler()
    handler = block(handler, sandbox_dir=SANDBOX, chunk_id=0)
    print(handler.unpack())
