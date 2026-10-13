"""Enumerate explicit side-chain rotamers while preserving the input backbone."""

from __future__ import annotations

from itertools import product
from pathlib import Path
import re

import numpy as np
from pydantic import Field, model_validator

from pipelines import InlineBlockFunction, JobArgs, PipelineDataHandler
from pipelines.data.schema import make_row


_POSITION = re.compile(r"^(?P<chain>[^:\d]+):?(?P<resid>-?\d+)$")


def enumerate_rotamers(
    rotamers: dict[str, list[list[float]]],
) -> list[dict[str, list[float]]]:
    """Return the deterministic Cartesian product of per-position chi vectors."""

    positions = list(rotamers)
    return [
        dict(zip(positions, choices))
        for choices in product(*(rotamers[position] for position in positions))
    ]


def uniform_chi_grid(steps: list[float]) -> list[list[float]]:
    """Enumerate a full ``[-180, 180)`` grid for each requested chi angle."""

    if not steps or any(step <= 0 or step > 360 for step in steps):
        raise ValueError("chi steps must be between 0 and 360 degrees")
    axes = [np.arange(-180.0, 180.0, step).tolist() for step in steps]
    return [list(values) for values in product(*axes)]


def rotamer_design_id(
    source_design_id: str | None,
    source_path: str | Path,
    block_name: str,
    combination_index: int,
) -> str:
    """Return a stable child identity for one enumerated rotamer.

    A rotamer sweep is a one-to-many transform.  Reusing the parent's design
    identity makes the resulting handler ambiguous and prevents metric blocks
    from backlinking each child independently.  Include the block name so
    matched mutation/position arms can be concatenated without collisions.
    """

    source = source_design_id or Path(source_path).stem
    return f"{source}__{block_name}_{combination_index:04d}"


class SidechainRotamerSweepBlock(InlineBlockFunction):
    """Mutate named residues and enumerate explicit chi-angle combinations.

    This block is intended for small, hypothesis-driven functional-site sweeps,
    not whole-protein packing. The protein backbone and all non-target residue
    coordinates remain fixed; hetero atoms are copied back unchanged.
    """

    _name = "rotamer_sweep"
    _display_name = "Side-chain Rotamer Sweep"
    _description = "Enumerate explicit functional side-chain chi combinations"
    _inline = True
    _chunked = True
    _creates_new_step = True
    _resources = JobArgs(partition="cpu", mem_gb=8, rows_per_job=10)

    mutations: dict[str, str] = Field(
        ...,
        description="Residue identities to install, e.g. {'A12': 'PHE'}.",
    )
    rotamers: dict[str, list[list[float]]] = Field(
        ...,
        description=(
            "Per-position chi vectors in degrees. Every Cartesian combination "
            "is emitted as a separate structure."
        ),
    )
    max_combinations: int = Field(
        1000,
        ge=1,
        description="Safety cap on emitted combinations per input structure.",
    )

    @model_validator(mode="after")
    def _validate_sweep(self):
        if set(self.mutations) != set(self.rotamers):
            raise ValueError("mutations and rotamers must name identical positions")
        if any(not choices for choices in self.rotamers.values()):
            raise ValueError("every mutation position requires at least one rotamer")
        count = int(np.prod([len(values) for values in self.rotamers.values()]))
        if count > self.max_combinations:
            raise ValueError(
                f"rotamer sweep has {count} combinations, exceeding "
                f"max_combinations={self.max_combinations}"
            )
        return self

    def execute(
        self,
        handler: PipelineDataHandler,
        sandbox_dir: Path,
        chunk_id: int = 0,
    ) -> PipelineDataHandler:
        from atomworks.io.utils.io_utils import to_cif_file
        from pipelines.io import ensure_bonds
        from biotite.structure import concatenate
        from biotite.structure.io import load_structure

        from blocks.utils.pyrosetta import atom_array_to_pose, init_pyrosetta
        import pyrosetta as pyr

        init_pyrosetta()
        sandbox_dir = Path(sandbox_dir)
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        combinations = enumerate_rotamers(self.rotamers)
        rows = []

        for source_index, loaded in enumerate(
            handler.get_data(which="last", load=True)
        ):
            source = loaded.atom_array
            protein = source[~np.asarray(source.hetero, dtype=bool)]
            hetero = source[np.asarray(source.hetero, dtype=bool)]
            base_pose = atom_array_to_pose(protein)
            pose_positions = {}
            for position, residue_name in self.mutations.items():
                match = _POSITION.fullmatch(position)
                if match is None:
                    raise ValueError(
                        f"Invalid residue position {position!r}; expected A42 or A:42"
                    )
                pose_index = base_pose.pdb_info().pdb2pose(
                    match.group("chain"), int(match.group("resid"))
                )
                if pose_index == 0:
                    raise KeyError(f"Protein residue {position!r} was not found")
                mover = pyr.rosetta.protocols.simple_moves.MutateResidue(
                    pose_index, residue_name.upper()
                )
                mover.apply(base_pose)
                pose_positions[position] = pose_index

            temp_pdb = sandbox_dir / f".rotamer_chunk{chunk_id}.pdb"
            for combination_index, selected in enumerate(combinations):
                pose = base_pose.clone()
                for position, chis in selected.items():
                    pose_index = pose_positions[position]
                    if len(chis) > pose.residue(pose_index).nchi():
                        raise ValueError(
                            f"{position} {self.mutations[position]} has "
                            f"{pose.residue(pose_index).nchi()} chis, got {len(chis)}"
                        )
                    for chi_index, angle in enumerate(chis, start=1):
                        pose.set_chi(chi_index, pose_index, float(angle))

                pose.dump_pdb(str(temp_pdb))
                swept_protein = load_structure(temp_pdb, model=1)
                combined = (
                    concatenate([swept_protein, hetero])
                    if len(hetero)
                    else swept_protein
                )
                output_path = sandbox_dir / (
                    f"{Path(loaded.path).stem}__rotamer_{combination_index:04d}.cif"
                )
                to_cif_file(ensure_bonds(combined), str(output_path))
                rows.append(
                    make_row(
                        name=self.name,
                        path=str(output_path),
                        design_id=rotamer_design_id(
                            loaded.design_id,
                            loaded.path,
                            self.name,
                            combination_index,
                        ),
                        parent_design_id=loaded.design_id,
                        metadata={
                            "source_path": loaded.path,
                            "source_index": source_index,
                            "chunk_id": chunk_id,
                            "rotamer_index": combination_index,
                            "mutations": self.mutations,
                            "chi_angles": selected,
                        },
                    )
                )
            temp_pdb.unlink(missing_ok=True)

        return PipelineDataHandler(rows)
