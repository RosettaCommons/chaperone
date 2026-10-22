"""BindCraft block for AlphaFold2-guided de novo binder design.

BindCraft jointly hallucinates a binder against a protein target, redesigns it
with ProteinMPNN, validates the complexes with AlphaFold2, and filters/ranks
accepted designs with PyRosetta-derived interface metrics.

The block accepts target structures from a :class:`PipelineDataHandler`; users
do not need to author BindCraft's target-settings JSON.  One target is run per
chunk because each BindCraft process owns a long-running trajectory loop and a
single output directory.

The Apptainer image is pinned to BindCraft commit
``efb5bfeb8b4b1a5944256f979c34e0c8e6a82d9d`` and a CUDA 12 JAX stack.  It
includes the public AlphaFold2 parameters.  PyRosetta is suitable for academic,
non-commercial use by default; other uses require an appropriate license.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator

from chaperone import CompiledBlockFunction, JobArgs, PipelineDataHandler
from chaperone.block_config import TIER_PRIMARY, ScriptSpec
from chaperone.common import tail_block_log, try_float
from chaperone.containers import Image
from chaperone.data import make_row
from chaperone.data.schema import DesignMetadata

logger = logging.getLogger(__name__)

_BINDCRAFT_COMMIT = "efb5bfeb8b4b1a5944256f979c34e0c8e6a82d9d"
_COLABDESIGN_COMMIT = "e31a56fe1d9b4de25c8697f3a28b75892941cc72"
_PDBFIXER_COMMIT = "94cfa4c0ca551cdc5f13320f9a658efd59f2b881"
_PYROSETTA_VERSION = "2026.29+releasequarterly.80a0635615"
_JAX_VERSION = "0.4.35"
_CUDA_ROOT = "/opt/bindcraft-env/lib/python3.10/site-packages/nvidia/cuda_nvcc"
_RUNNER_SOURCE = Path(__file__).parent.parent.parent / "scripts" / "run_bindcraft.py"
_RUNNER_DEST = "/opt/BindCraft/chaperone_run_bindcraft.py"

_PYTHON_PACKAGES = (
    "numpy==1.26.4 pandas==2.2.3 matplotlib==3.9.4 biopython==1.85 "
    "scipy==1.15.2 seaborn==0.13.2 tqdm==4.67.1 fsspec==2025.3.2 "
    "py3Dmol==2.4.2 chex==0.1.87 dm-haiku==0.0.13 flax==0.8.5 "
    "dm-tree==0.1.9 joblib==1.4.2 ml-collections==1.0.0 "
    "immutabledict==4.2.1 optax==0.2.4 orbax-checkpoint==0.6.4"
)

BINDCRAFT_CONTAINER = (
    Image("bindcraft")
    # Ubuntu 24.04 is required by this cluster's root-mapped Apptainer
    # fakeroot helper (built against glibc >=2.38). CUDA/cuDNN are supplied by
    # the pinned JAX CUDA 12 wheels, while --nv injects the host driver.
    .base_image("ubuntu:24.04")
    .packages(
        "ca-certificates",
        "git",
        "wget",
        "tar",
        "bzip2",
        "build-essential",
        "libgmp10",
        "ffmpeg",
        "python3-pip",
        "software-properties-common",
        "gnupg",
    )
    .git_clone("https://github.com/martinpacesa/BindCraft.git", "/opt/BindCraft")
    .copy(str(_RUNNER_SOURCE), _RUNNER_DEST)
    .run(
        f"git -C /opt/BindCraft checkout {_BINDCRAFT_COMMIT} && "
        f'test "$(git -C /opt/BindCraft rev-parse HEAD)" = "{_BINDCRAFT_COMMIT}"'
    )
    .run(
        "add-apt-repository -y ppa:deadsnakes/ppa && apt-get update && "
        "apt-get install -y --no-install-recommends "
        "python3.10 python3.10-dev python3.10-venv",
    )
    .run(
        "python3.10 -m venv /opt/bindcraft-env",
        "/opt/bindcraft-env/bin/pip install --no-cache-dir "
        "--upgrade 'pip>=26.2.2' setuptools wheel",
        "/opt/bindcraft-env/bin/pip install --no-cache-dir "
        f"'jax[cuda12]=={_JAX_VERSION}' {_PYTHON_PACKAGES}",
        "/opt/bindcraft-env/bin/pip install --no-cache-dir "
        f"'git+https://github.com/openmm/pdbfixer.git@{_PDBFIXER_COMMIT}'",
    )
    .run(
        "/opt/bindcraft-env/bin/pip install --no-cache-dir --no-deps "
        f"'git+https://github.com/sokrypton/ColabDesign.git@{_COLABDESIGN_COMMIT}'",
        "/opt/bindcraft-env/bin/pip install --no-cache-dir "
        f"'pyrosetta=={_PYROSETTA_VERSION}' --find-links "
        "https://west.rosettacommons.org/pyrosetta/quarterly/"
        "release.cxx11thread.serialization/",
    )
    .run(
        "mkdir -p /opt/BindCraft/params",
        "wget --tries=5 -O /tmp/alphafold_params.tar "
        "https://storage.googleapis.com/alphafold/"
        "alphafold_params_2022-12-06.tar",
        'test "$(stat -c %s /tmp/alphafold_params.tar)" = "5587968000"',
        "tar -tf /tmp/alphafold_params.tar >/dev/null",
        "tar -xf /tmp/alphafold_params.tar -C /opt/BindCraft/params",
        "test -s /opt/BindCraft/params/params_model_5_ptm.npz",
        "test -s /opt/BindCraft/params/params_model_5_multimer_v3.npz",
        "rm /tmp/alphafold_params.tar",
    )
    .run(
        "chmod 0755 /opt/BindCraft/functions/dssp "
        "/opt/BindCraft/functions/DAlphaBall.gcc "
        f"{_RUNNER_DEST}",
        f"CUDA_ROOT={_CUDA_ROOT} /opt/bindcraft-env/bin/python -c "
        "'import colabdesign, jax, jaxlib, pyrosetta'",
        "mkdir -p /opt/BindCraft/manifest",
        "/opt/bindcraft-env/bin/pip freeze > /opt/BindCraft/manifest/pip-freeze.txt",
        "rm -rf /var/lib/apt/lists/*",
    )
    .env(
        BINDCRAFT_HOME="/opt/BindCraft",
        CUDA_ROOT=_CUDA_ROOT,
        PATH="/opt/bindcraft-env/bin:$PATH",
        VIRTUAL_ENV="/opt/bindcraft-env",
        PYTHONNOUSERSITE="1",
        PYTHONUNBUFFERED="1",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        TMPDIR="/tmp",
    )
    .label(
        Version=f"main-{_BINDCRAFT_COMMIT[:12]}",
        Revision=_BINDCRAFT_COMMIT,
        CUDA="12",
        Source="https://github.com/martinpacesa/BindCraft",
        License="MIT-and-third-party",
    )
    .runscript(f'python -u {_RUNNER_DEST} "$@"')
    .template_command(
        exec_mode="exec",
        binds=[],
        env={"XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
    )
)


class BindCraftBlock(CompiledBlockFunction):
    """Design protein binders against each input target structure.

    ``chains`` selects the target chain(s).  ``target_hotspot_residues`` uses
    BindCraft syntax such as ``"A56,A60-64"``; ``None`` lets AlphaFold2 select
    a site.  Presets refer to JSON basenames bundled by BindCraft.  A custom
    JSON path can be supplied instead, and shallow override dictionaries are
    applied last.
    """

    _name = "bindcraft"
    _display_name = "BindCraft"
    _description = "AlphaFold2-guided de novo protein binder design"
    _script_spec = ScriptSpec(
        command=["python", "-u", _RUNNER_DEST],
        image=BINDCRAFT_CONTAINER,
        source_url="https://github.com/martinpacesa/BindCraft",
    )
    _resources = JobArgs(
        partition="gpu",
        gpu_class="large",
        mem_gb=48,
        time_limit="24:00:00",
        rows_per_job=1,
    )
    _requires_input = True
    _inline = False
    _chunked = True

    chains: str = Field(
        ...,
        min_length=1,
        json_schema_extra={"tier": TIER_PRIMARY},
        description="BindCraft target-chain selector, e.g. 'A' or 'A,B'.",
    )
    target_hotspot_residues: str | None = Field(
        None,
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Hotspot selection such as 'A56,A60-64'; None auto-selects a site.",
    )
    lengths: tuple[int, int] = Field(
        (65, 150),
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Inclusive minimum and maximum binder length.",
    )
    number_of_final_designs: int = Field(
        1,
        ge=1,
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Accepted designs requested per input target.",
    )
    max_trajectories: int | None = Field(
        100,
        ge=1,
        description=(
            "Maximum hallucination trajectories per target. None honors the advanced "
            "preset and may run until the accepted-design target is reached."
        ),
    )
    binder_name: str = Field("binder", description="Prefix for generated designs.")

    filter_preset: str = Field(
        "default_filters",
        description="Bundled BindCraft filter preset basename.",
    )
    advanced_preset: str = Field(
        "default_4stage_multimer",
        description="Bundled BindCraft advanced-settings preset basename.",
    )
    filters_path: str | None = Field(
        None, description="Optional custom filters JSON; overrides filter_preset."
    )
    advanced_path: str | None = Field(
        None, description="Optional custom advanced JSON; overrides advanced_preset."
    )
    filter_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="Shallow overrides applied to the selected filter JSON.",
    )
    advanced_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="Shallow overrides applied to the selected advanced JSON.",
    )

    @field_validator("lengths")
    @classmethod
    def _validate_lengths(cls, value: tuple[int, int]) -> tuple[int, int]:
        if value[0] < 1 or value[1] < value[0]:
            raise ValueError(
                "lengths must be (minimum, maximum) with 1 <= minimum <= maximum"
            )
        return value

    @field_validator("binder_name")
    @classmethod
    def _validate_binder_name(cls, value: str) -> str:
        if not value or re.fullmatch(r"[A-Za-z0-9_.-]+", value) is None:
            raise ValueError(
                "binder_name may contain only letters, numbers, underscore, dot, and dash"
            )
        return value

    @field_validator("filter_preset", "advanced_preset")
    @classmethod
    def _validate_preset_name(cls, value: str) -> str:
        if Path(value).name != value or value in {"", ".", ".."}:
            raise ValueError("preset names must be simple basenames")
        return value.removesuffix(".json")

    @staticmethod
    def _copy_config(path: str, destination: Path) -> str:
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise ValueError(f"BindCraft configuration does not exist: {source}")
        shutil.copy2(source, destination)
        return str(destination)

    def create_command(
        self, handler: PipelineDataHandler, sandbox_dir: Path, chunk_id: int
    ) -> list[str]:
        sandbox = Path(sandbox_dir).resolve()
        inputs_dir = sandbox / "inputs"
        config_dir = sandbox / "configs"
        design_dir = sandbox / "designs"
        for directory in (inputs_dir, config_dir, design_dir):
            directory.mkdir(parents=True, exist_ok=True)

        targets = list(handler.get_data(which="last", load=True))
        if len(targets) != 1:
            raise ValueError(
                "BindCraft requires exactly one target per chunk; keep rows_per_job=1"
            )
        target = targets[0]

        # BindCraft's parser expects PDB.  Normalize CIF, CatCIF, and PDB inputs
        # through the already-loaded atom array so chain identifiers are kept.
        from atomworks.io.utils.io_utils import to_pdb_string

        safe_target_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", target.basename)
        target_pdb = inputs_dir / f"{Path(safe_target_name).stem}.pdb"
        target_pdb.write_text(to_pdb_string(target.atom_array))

        settings = {
            "design_path": str(design_dir),
            "binder_name": f"{self.binder_name}_chunk{chunk_id}",
            "starting_pdb": str(target_pdb),
            "chains": self.chains,
            "target_hotspot_residues": self.target_hotspot_residues,
            "lengths": list(self.lengths),
            "number_of_final_designs": self.number_of_final_designs,
        }
        settings_path = config_dir / "target.json"
        settings_path.write_text(json.dumps(settings, indent=2) + "\n")

        advanced_overrides = dict(self.advanced_overrides)
        if self.max_trajectories is not None:
            advanced_overrides["max_trajectories"] = self.max_trajectories
        advanced_overrides_path = config_dir / "advanced_overrides.json"
        advanced_overrides_path.write_text(
            json.dumps(advanced_overrides, indent=2, sort_keys=True) + "\n"
        )
        filter_overrides_path = config_dir / "filter_overrides.json"
        filter_overrides_path.write_text(
            json.dumps(self.filter_overrides, indent=2, sort_keys=True) + "\n"
        )

        filters = self.filter_preset
        if self.filters_path:
            filters = self._copy_config(
                self.filters_path, config_dir / "custom_filters.json"
            )
        advanced = self.advanced_preset
        if self.advanced_path:
            advanced = self._copy_config(
                self.advanced_path, config_dir / "custom_advanced.json"
            )

        parent_id = target.design_id or target.basename
        manifest = {
            "parent_design_id": parent_id,
            "source_path": target.path,
            "target_pdb": str(target_pdb),
            "chains": self.chains,
            "target_hotspot_residues": self.target_hotspot_residues,
        }
        (sandbox / "bindcraft_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )

        cmd = self._script_spec.image.build_command(
            *self._script_spec.command,
            binds=[(str(sandbox), str(sandbox))],
        )
        return cmd + [
            "--settings",
            str(settings_path),
            "--filters",
            filters,
            "--advanced",
            advanced,
            "--filter-overrides",
            str(filter_overrides_path),
            "--advanced-overrides",
            str(advanced_overrides_path),
            "--config-dir",
            str(config_dir / "resolved"),
        ]

    @staticmethod
    def _design_key(path: Path) -> tuple[str, int | None]:
        ranked = re.fullmatch(r"(\d+)_(.+)_model\d+", path.stem)
        if ranked:
            return ranked.group(2), int(ranked.group(1))
        accepted = re.fullmatch(r"(.+)_model\d+", path.stem)
        if accepted:
            return accepted.group(1), None
        return path.stem, None

    def compile(self, sandbox_dir: Path) -> PipelineDataHandler:
        sandbox = Path(sandbox_dir).resolve()
        design_dir = sandbox / "designs"
        ranked = list((design_dir / "Accepted" / "Ranked").glob("*.pdb"))
        paths = ranked or list((design_dir / "Accepted").glob("*.pdb"))
        if not paths:
            raise RuntimeError(
                f"BindCraft produced no accepted PDB files in {design_dir}. "
                "Increase max_trajectories, relax filters, or inspect the block log."
                + tail_block_log(sandbox)
            )

        metrics_by_design: dict[str, dict[str, Any]] = {}
        final_csv = design_dir / "final_design_stats.csv"
        if final_csv.is_file():
            with final_csv.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    design = row.get("Design")
                    if design:
                        metrics_by_design[design] = {
                            key: try_float(value) for key, value in row.items()
                        }

        manifest_path = sandbox / "bindcraft_manifest.json"
        manifest = (
            json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        )
        parent_id = manifest.get("parent_design_id")
        source_path = manifest.get("source_path")

        def sort_key(path: Path) -> tuple[int, str]:
            _, rank = self._design_key(path)
            return (rank if rank is not None else 10**9, path.name)

        rows = []
        for path in sorted(paths, key=sort_key):
            design, filename_rank = self._design_key(path)
            block_metrics = dict(metrics_by_design.get(design, {}))
            rank = block_metrics.get("Rank", filename_rank)
            if rank is None:
                rank = filename_rank
            block_metrics.update(
                {
                    "accepted": True,
                    "rank": rank,
                    "target_chains": manifest.get("chains", self.chains),
                    "target_hotspot_residues": manifest.get(
                        "target_hotspot_residues", self.target_hotspot_residues
                    ),
                }
            )
            logical_parent = parent_id or Path(source_path or "target").stem
            rows.append(
                make_row(
                    name=self.name,
                    path=path,
                    path_handle=source_path,
                    design_id=f"{logical_parent}__bindcraft_{path.stem}",
                    parent_design_id=parent_id,
                    metadata={
                        DesignMetadata.TASK: "binder_design",
                        DesignMetadata.ORIGINAL_INPUT: source_path,
                    },
                    metrics={self.name: block_metrics},
                )
            )

        return PipelineDataHandler(rows)
