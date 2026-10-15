"""Bounded, single-target BindCraft2 binder design with accepted complexes out."""

import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

import numpy as np
from biotite.structure import get_residue_starts
from biotite.structure.io.pdbx import CIFFile, get_structure
from pydantic import Field, model_validator

from chaperone import CompiledBlockFunction, JobArgs, PipelineDataHandler
from chaperone.block_config import ScriptSpec
from chaperone.data import make_row
from chaperone.io import RES_NAME_TO_ONE_LETTER, write_pdb
from .container import BINDCRAFT2_CONTAINER, BINDCRAFT2_REVISION

# A deliberately bounded interface: arbitrary upstream campaign settings can
# bypass scheduling, staging, or accepted-only output semantics.
_OVERRIDES = (
    {
        "autotune",
        "desperation",
        "design_models",
        "validation_models",
        "design_recycles",
        "validation_recycles",
        "design_dropout",
        "sequence_candidates",
        "kept_sequences",
        "enough_passing_sequences",
        "mpnn_variant",
        "mpnn_model",
        "filters",
        "aa_bias",
        "min_monomer_plddt_final",
        "betasheet_reopt_trigger",
        "betasheet_reopt_extra_refine_steps",
        "betasheet_reopt_extra_anneal_steps",
    }
    | {f"{stage}_steps" for stage in ("screen", "refine", "anneal", "harden", "mutate")}
    | {
        f"min_{metric}_{stage}"
        for metric in ("plddt", "iptm")
        for stage in ("screen", "refine", "anneal", "harden", "mutate", "final")
    }
)


def _sequence(atoms):
    residues = atoms[get_residue_starts(atoms)]
    return "".join(RES_NAME_TO_ONE_LETTER.get(name, "X") for name in residues.res_name)


class BindCraft2Block(CompiledBlockFunction):
    """Design miniprotein binders to one protein chain and return accepted refolds.

    Upstream default quality filters remain active unless explicitly overridden.
    Exhausting the trajectory budget without the requested count fails the step.
    """

    _name = "bindcraft2"
    _display_name = "BindCraft2"
    _description = (
        "JAX binder optimization, ProteinMPNN redesign, and AlphaFold2 validation"
    )
    _script_spec = ScriptSpec(
        command=["bindcraft", "design"],
        image=BINDCRAFT2_CONTAINER,
        source_url="https://github.com/PacesaLab/BindCraft2",
    )
    _resources = JobArgs(
        partition="gpu",
        gpu_class="large",
        mem_gb=64,
        rows_per_job=1,
        time_limit="24:00:00",
    )
    _pipeline_inputs = ["last"]
    _inline = False
    _chunked = True

    chain: str = Field(
        ...,
        pattern=r"^[A-Za-z0-9]$",
        description="Single target protein chain in the input structure",
    )
    hotspots: str | None = Field(
        None, description="Target residue numbers/ranges, e.g. 54,56,66-70"
    )
    lengths: tuple[int, int] = Field(
        (60, 180), description="Inclusive binder length bounds"
    )
    number_of_final_designs: int = Field(1, ge=1)
    max_trajectories: int = Field(
        10,
        ge=1,
        description="Finite per-input optimization budget; failure if insufficient designs pass",
    )
    seed: int = Field(42, ge=0, le=4294967295)
    af2_params_dir: str | None = Field(
        None,
        description="Directory containing all five multimer-v3 and two PTM AF2 checkpoints; alternatively BINDCRAFT_AF2_PARAMS",
    )
    weights_dir: str = Field(
        default_factory=lambda: str(Path.home() / ".cache/bindcraft2"),
        description="Writable upstream weight download cache when no AF2 directory is supplied",
    )
    overrides: dict = Field(
        default_factory=dict,
        description="Selected upstream scientific settings; changing filters changes the acceptance criterion",
    )

    @model_validator(mode="after")
    def validate_settings(self):
        if not 1 <= self.lengths[0] <= self.lengths[1]:
            raise ValueError("lengths requires 1 <= minimum <= maximum")
        unknown = set(self.overrides) - _OVERRIDES
        if unknown:
            raise ValueError(f"Unsupported BindCraft2 overrides: {sorted(unknown)}")
        json.dumps(self.overrides, allow_nan=False)
        return self

    def create_command(self, handler, sandbox_dir, chunk_id):
        items = list(handler.get_data(which="last", load=True))
        if len(items) != 1:
            raise ValueError(
                "bindcraft2 requires one target per chunk; keep rows_per_job=1"
            )
        item = items[0]
        if item.metadata.get("motif"):
            raise ValueError("BindCraft2 does not enforce Chaperone motif masks")
        atoms = item.atom_array
        target = atoms[
            (atoms.chain_id == self.chain) & ~np.isin(atoms.element, ["H", "D"])
        ]
        if not len(target) or not np.isfinite(target.coord).all():
            raise ValueError(
                "Selected target chain is absent or has non-finite coordinates"
            )
        residues = target[get_residue_starts(target)]
        if (
            np.any(target.ins_code != "")
            or len(set(residues.res_id)) != len(residues)
            or any(name not in RES_NAME_TO_ONE_LETTER for name in residues.res_name)
            or np.any(target.hetero)
        ):
            raise ValueError(
                "Target must contain standard protein residues without ligands, insertion codes, or duplicate residue IDs"
            )
        for start, end in zip(
            get_residue_starts(target),
            get_residue_starts(target, add_exclusive_stop=True)[1:],
        ):
            if not {"N", "CA", "C", "O"} <= set(target.atom_name[start:end]):
                raise ValueError(
                    "Every target residue requires N, CA, C, O backbone atoms"
                )
        if self.hotspots is not None:
            available = set(residues.res_id)
            for segment in self.hotspots.split(","):
                match = re.fullmatch(r"(\d+)(?:-(\d+))?", segment.strip())
                if not match:
                    raise ValueError(
                        "hotspots must contain residue numbers or ranges, e.g. 54,66-70"
                    )
                first, last = int(match[1]), int(match[2] or match[1])
                if last < first or not set(range(first, last + 1)) <= available:
                    raise ValueError(f"Hotspot outside selected target: {segment}")
        root = Path(sandbox_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        write_pdb(target, root / "target.pdb")
        config = {
            **self.overrides,
            "targets": [
                {
                    "name": "target",
                    "target_path": str(root / "target.pdb"),
                    "chains": self.chain,
                    **({"hotspots": self.hotspots} if self.hotspots else {}),
                }
            ],
            "modality": "binder",
            "binder_lengths": list(self.lengths),
            "project_folder": str(root / "designs"),
            "campaign_name": "chaperone",
            "campaign_seed": (self.seed + int(chunk_id)) % 2**32,
            "number_of_final_designs": self.number_of_final_designs,
            "max_trajectories": self.max_trajectories,
            "auto_multi_gpu": False,
            "design_workers": 1,
            "workers_per_gpu": 1,
            "compile_next_length": False,
            "resume": False,
            "trajectory_only": False,
            "relax_accepted_designs": False,
        }
        (root / "job.json").write_text(json.dumps(config, indent=2, allow_nan=False))
        (root / "input_metadata.json").write_text(
            json.dumps(
                {
                    "path_handle": item.path,
                    "parent_design_id": item.design_id or item.basename,
                    "target_sequence": _sequence(target),
                    "target_chain": self.chain,
                    "seed": config["campaign_seed"],
                    "upstream_revision": BINDCRAFT2_REVISION,
                }
            )
        )
        env = {}
        binds = [(str(root), str(root))]
        params = self.af2_params_dir or os.environ.get("BINDCRAFT_AF2_PARAMS")
        if params:
            params = Path(params).expanduser().resolve()
            required = [f"params_model_{i}_multimer_v3.npz" for i in range(1, 6)] + [
                f"params_model_{i}_ptm.npz" for i in (1, 2)
            ]
            if not all((params / name).is_file() for name in required):
                if all((params / "params" / name).is_file() for name in required):
                    params /= "params"
                else:
                    raise ValueError(
                        "AF2 parameter directory is missing required multimer-v3/PTM checkpoints"
                    )
            env["BINDCRAFT_AF2_PARAMS"] = str(params)
            binds.append((str(params), str(params)))
        else:
            cache = Path(self.weights_dir).expanduser().resolve()
            cache.mkdir(parents=True, exist_ok=True)
            env["BINDCRAFT_WEIGHTS"] = str(cache)
            binds.append((str(cache), str(cache)))
        for variable, folder in [
            ("JAX_COMPILATION_CACHE_DIR", "jax"),
            ("MPLCONFIGDIR", "matplotlib"),
        ]:
            cache = root / ".cache" / folder
            cache.mkdir(parents=True, exist_ok=True)
            env[variable] = str(cache)
        return self._script_spec.image.build_command(
            *self._script_spec.command, binds=binds, env=env
        ) + [str(root / "job.json")]

    def compile(self, sandbox_dir):
        root = Path(sandbox_dir).resolve()
        metadata = json.loads((root / "input_metadata.json").read_text())
        ranked = root / "designs/3_Ranked"
        table = ranked / "!_Ranked.csv"
        if not table.is_file():
            raise RuntimeError(
                "BindCraft2 produced no ranked accepted designs; inspect trajectory/refold logs"
            )
        with table.open(newline="") as stream:
            records = list(csv.DictReader(stream))
        if len(records) < self.number_of_final_designs:
            raise RuntimeError(
                f"BindCraft2 accepted {len(records)} designs; requested {self.number_of_final_designs}"
            )
        rows, seen = [], set()
        for record in records:
            name = record.get("design", "")
            if (
                not re.fullmatch(r"[A-Za-z0-9_.-]+", name)
                or name in {".", ".."}
                or name in seen
            ):
                raise RuntimeError("Invalid or duplicate design ID in ranked table")
            seen.add(name)
            path = ranked / f"{name}.cif"
            if not path.is_file():
                raise RuntimeError(f"Missing ranked complex: {path}")
            atoms = get_structure(CIFFile.read(path), model=1)
            if not len(atoms) or not np.isfinite(atoms.coord).all():
                raise RuntimeError(f"Invalid ranked coordinates: {path}")
            if set(atoms.chain_id) != {"A", "B"}:
                raise RuntimeError("Ranked complex must contain target A and binder B")
            if _sequence(atoms[atoms.chain_id == "A"]) != metadata["target_sequence"]:
                raise RuntimeError("Ranked target sequence changed")
            sequence = _sequence(atoms[atoms.chain_id == "B"])
            if (
                sequence != record.get("Binder_Sequence")
                or not self.lengths[0] <= len(sequence) <= self.lengths[1]
            ):
                raise RuntimeError(
                    "Ranked binder sequence/length disagrees with configuration or scores"
                )
            metrics = {}
            for key, value in record.items():
                try:
                    number = float(value)
                    metrics[key] = number if np.isfinite(number) else None
                except (ValueError, TypeError):
                    metrics[key] = value
            identity = f"{metadata['parent_design_id']}:{metadata['seed']}:{name}"
            design_id = (
                "bindcraft2_" + hashlib.sha256(identity.encode()).hexdigest()[:20]
            )
            published = root / "complexes" / f"{design_id}.cif"
            published.parent.mkdir(exist_ok=True)
            shutil.copy2(path, published)
            rows.append(
                make_row(
                    name=self.name,
                    path=str(published),
                    path_handle=metadata["path_handle"],
                    design_id=design_id,
                    parent_design_id=metadata["parent_design_id"],
                    metadata={
                        "task": "binder",
                        "motif": [
                            f"A{residue.res_id}"
                            for residue in atoms[get_residue_starts(atoms)]
                            if residue.chain_id == "A"
                        ],
                        "original_input": metadata["path_handle"],
                        # Input annotations refer to a different chain/frame.
                        "diffused_index_map": "{}",
                        "ligand": [],
                        "interface": [],
                        "hbond": [],
                        "aromatics_near_ligand": [],
                    },
                    sequences={"A": metadata["target_sequence"], "B": sequence},
                    metrics={
                        self.name: {
                            **metrics,
                            # Custom metadata is not retained by the engine's fixed
                            # structure schema; keep provenance in its JSON metrics.
                            "provenance": {
                                **metadata,
                                "binder_chain": "B",
                                "target_chains": ["A"],
                                "upstream_design": name,
                                "settings_path": str(root / "job.json"),
                            },
                        }
                    },
                )
            )
        return PipelineDataHandler(rows)
