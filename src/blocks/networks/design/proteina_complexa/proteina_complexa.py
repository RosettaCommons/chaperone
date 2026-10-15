"""Protein binder generation with NVIDIA's partially latent flow model."""

from pathlib import Path
import json
import shutil
import re

import numpy as np
from biotite.structure.io.pdb import PDBFile
from biotite.structure import rmsd, superimpose

from pydantic import Field, field_validator, model_validator
import yaml

from chaperone import CompiledBlockFunction, PipelineDataHandler, JobArgs
from chaperone.block_config import ScriptSpec, TIER_PRIMARY
from chaperone.data import make_row
from chaperone.data.schema import Column
from .container import COMPLEXA_CONTAINER, COMPLEXA_REVISION


class ProteinaComplexaBlock(CompiledBlockFunction):
    """Co-design all-atom protein binders around a fixed protein target.

    Runs the upstream single-pass generator. Outputs contain target and binder,
    including designed side chains and sequence. Refolding/quality evaluation is
    a separate downstream block; generation does not certify a successful binder.
    """

    _name = "proteina_complexa"
    _display_name = "Proteina-Complexa"
    _description = "All-atom protein binder sequence/structure co-design"
    _script_spec = ScriptSpec(
        command=["python", "-m", "proteinfoundation.generate"],
        image=COMPLEXA_CONTAINER,
        source_url="https://github.com/NVIDIA-BioNeMo/Proteina-Complexa",
    )
    _resources = JobArgs(partition="gpu", gpu_class="large", mem_gb=48, rows_per_job=1)
    _requires_input = False
    _inline = False
    _chunked = True

    target_pdb: str = Field(
        ...,
        description="Target PDB; copied into each job sandbox",
        json_schema_extra={"tier": TIER_PRIMARY},
    )
    target_selection: str = Field(
        ...,
        description="Upstream target input specification, e.g. A1-115",
        json_schema_extra={"tier": TIER_PRIMARY},
    )
    hotspots: list[str] = Field(
        ...,
        min_length=1,
        description="Target residue identifiers, e.g. ['A37', 'A39']",
        json_schema_extra={"tier": TIER_PRIMARY},
    )
    binder_length: tuple[int, int] = Field(
        (80, 120), description="Inclusive minimum and maximum binder residue count"
    )
    num_designs: int = Field(1, ge=1, description="Designs generated per batch")
    n_batches: int = Field(
        1, ge=1, description="Independent engine jobs, each with a distinct seed"
    )
    batch_size: int = Field(1, ge=1, description="Simultaneous samples on one GPU")
    num_steps: int = Field(
        400, ge=1, description="Flow integration steps; upstream default is 400"
    )
    seed: int = Field(0, ge=0, description="Base seed, incremented by engine chunk ID")
    checkpoint_dir: str = Field(
        ...,
        description="Directory containing public complexa.ckpt and complexa_ae.ckpt checkpoints",
    )

    @field_validator("target_pdb")
    @classmethod
    def validate_target(cls, value):
        path = Path(value).expanduser().resolve()
        if not path.is_file() or path.suffix.lower() != ".pdb":
            raise ValueError("target_pdb must be an existing .pdb file")
        return str(path)

    @field_validator("checkpoint_dir")
    @classmethod
    def validate_checkpoints(cls, value):
        path = Path(value).expanduser().resolve()
        for filename in ("complexa.ckpt", "complexa_ae.ckpt"):
            if not (path / filename).is_file():
                raise ValueError(
                    f"Missing Proteina-Complexa checkpoint: {path / filename}"
                )
        return str(path)

    @model_validator(mode="after")
    def validate_lengths(self):
        if not 1 <= self.binder_length[0] <= self.binder_length[1]:
            raise ValueError("binder_length requires 1 <= minimum <= maximum")
        _selected_target_ca(self.target_pdb, self.target_selection, self.hotspots)
        return self

    def init_handler(self):
        return PipelineDataHandler(
            [
                {Column.STRUCTURES: [], Column.METRICS: "{}"}
                for _ in range(self.n_batches)
            ]
        )

    def create_command(self, handler, sandbox_dir, chunk_id):
        if len(handler) != 1:
            raise ValueError(
                "Proteina-Complexa requires one batch per chunk; keep rows_per_job=1 and do not merge batches with num_chunks"
            )
        sandbox = Path(sandbox_dir).resolve()
        sandbox.mkdir(parents=True, exist_ok=True)
        selected_ca = _selected_target_ca(
            self.target_pdb, self.target_selection, self.hotspots
        )
        matplotlib_cache = sandbox / ".cache" / "matplotlib"
        matplotlib_cache.mkdir(parents=True, exist_ok=True)
        staged_target = sandbox / "target.pdb"
        shutil.copy2(self.target_pdb, staged_target)
        # Keep only generation defaults: evaluation config would require unrelated
        # external models and databases. Hydra reads upstream defaults from image.
        config = {
            "defaults": ["/pipeline/binder/binder_generate@generation", "_self_"],
            "hydra": {
                "searchpath": ["file:///opt/complexa/configs"],
                "run": {"dir": str(sandbox)},
                "job": {"chdir": True},
            },
            "ckpt_path": self.checkpoint_dir,
            "ckpt_name": "complexa.ckpt",
            "autoencoder_ckpt_path": str(
                Path(self.checkpoint_dir) / "complexa_ae.ckpt"
            ),
            "seed": self.seed + int(chunk_id),
            "gen_njobs": 1,
            "generation": {
                "task_name": "target",
                "target_dict_cfg": {
                    "target": {
                        "target_path": str(staged_target),
                        "source": "user",
                        "target_filename": "target",
                        "target_input": self.target_selection,
                        "hotspot_residues": self.hotspots,
                        "binder_length": list(self.binder_length),
                        "pdb_id": None,
                    }
                },
                "dataloader": {
                    "batch_size": self.batch_size,
                    "dataset": {"nres": {"nsamples": self.num_designs}},
                },
                "args": {"nsteps": self.num_steps},
                "search": {"algorithm": "single-pass"},
                "reward_model": None,
                "refinement": {"algorithm": None},
            },
        }
        (sandbox / "job.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        (sandbox / "input_metadata.json").write_text(
            json.dumps(
                {
                    "target_pdb": self.target_pdb,
                    "target_selection": self.target_selection,
                    "hotspots": self.hotspots,
                    "seed": self.seed + int(chunk_id),
                    "upstream_revision": COMPLEXA_REVISION,
                    "target_residue_names": selected_ca.res_name.tolist(),
                    "target_ca_coordinates": selected_ca.coord.tolist(),
                }
            )
        )
        return self._script_spec.image.build_command(
            *self._script_spec.command,
            binds=[
                (str(sandbox), str(sandbox)),
                (self.checkpoint_dir, self.checkpoint_dir),
            ],
            env={"DATA_PATH": str(sandbox), "MPLCONFIGDIR": str(matplotlib_cache)},
        ) + [f"--config-path={sandbox}", "--config-name=job"]

    def compile(self, sandbox_dir):
        sandbox = Path(sandbox_dir).resolve()
        output = sandbox / "inference" / "job_target"
        paths = sorted(output.glob("job_*/*.pdb"))
        if len(paths) != self.num_designs:
            raise RuntimeError(
                f"Proteina-Complexa produced {len(paths)} structures; expected {self.num_designs}. Check {sandbox} logs."
            )
        metadata = json.loads((sandbox / "input_metadata.json").read_text())
        rows = []
        for path in paths:
            structure = PDBFile.read(path).get_structure(model=1)
            if not len(structure) or not np.isfinite(structure.coord).all():
                raise RuntimeError(f"Invalid or non-finite generated structure: {path}")
            ca = structure[structure.atom_name == "CA"]
            chains = list(dict.fromkeys(ca.chain_id))
            if len(chains) < 2:
                raise RuntimeError(f"Missing target or binder chain in {path}")
            binder = ca[ca.chain_id == chains[-1]]
            target = ca[ca.chain_id != chains[-1]]
            if not self.binder_length[0] <= len(binder) <= self.binder_length[1]:
                raise RuntimeError(f"Unexpected binder length {len(binder)} in {path}")
            if target.res_name.tolist() != metadata["target_residue_names"]:
                raise RuntimeError(f"Target sequence/crop changed in {path}")
            expected = np.asarray(metadata["target_ca_coordinates"], dtype=np.float32)
            aligned, _ = superimpose(expected, target.coord)
            target_rmsd = float(rmsd(expected, aligned))
            if target_rmsd > 0.02:
                raise RuntimeError(
                    f"Target coordinates changed (CA RMSD {target_rmsd:.4f} A) in {path}"
                )
            sample_metadata = {
                **{
                    key: value
                    for key, value in metadata.items()
                    if key != "target_ca_coordinates"
                },
                "original_input": metadata["target_pdb"],
                "motif": [f"{atom.chain_id}{atom.res_id}" for atom in target],
                "binder_chain": chains[-1],
                "target_chains": chains[:-1],
            }
            # Upstream restarts job/sample numbering in every engine chunk.
            # Give both row IDs and structure basenames a seed-qualified name:
            # legacy downstream tools still key work by the file stem.
            design_id = f"{self._name}_seed{metadata['seed']}_{path.stem}"
            published = sandbox / "complexes" / f"{design_id}.pdb"
            published.parent.mkdir(exist_ok=True)
            shutil.copy2(path, published)
            rows.append(
                make_row(
                    name=self.name,
                    path=str(published),
                    design_id=design_id,
                    metadata=sample_metadata,
                    metrics={self.name: {"target_ca_rmsd": target_rmsd}},
                )
            )
        return PipelineDataHandler(rows)


def _selected_target_ca(path, selection, hotspots):
    """Reject empty crops and silently ignored hotspots before allocating a GPU."""
    structure = PDBFile.read(path).get_structure(model=1)
    ca = structure[structure.atom_name == "CA"]
    mask = np.zeros(len(ca), dtype=bool)
    for segment in selection.split(","):
        match = re.fullmatch(r"([A-Za-z])(\d+)-(\d+)", segment.strip())
        if match is None:
            raise ValueError(
                "target_selection must contain comma-separated PDB residue ranges such as A1-115,B4-20"
            )
        chain, start, end = match.groups()
        start, end = int(start), int(end or start)
        if end < start:
            raise ValueError(f"Reversed target residue range: {segment}")
        mask |= (ca.chain_id == chain) & (ca.res_id >= start) & (ca.res_id <= end)
    selected = ca[mask]
    if not len(selected):
        raise ValueError("target_selection selects no protein CA atoms")
    available = {f"{atom.chain_id}{atom.res_id}" for atom in selected}
    if len(available) != len(selected) or np.any(selected.ins_code != ""):
        raise ValueError(
            "Target crop contains insertion codes or duplicate residue identifiers; normalize the PDB first"
        )
    missing = set(hotspots) - available
    if missing:
        raise ValueError(
            f"Hotspots outside target crop or absent from PDB: {sorted(missing)}"
        )
    return selected
