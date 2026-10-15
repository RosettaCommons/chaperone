"""Repack a bounded set of protein sidechains while locking everything else."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
from pydantic import Field, field_validator, model_validator
from scipy.spatial import cKDTree

from chaperone import InlineBlockFunction, JobArgs, PipelineDataHandler
from chaperone.data.schema import make_row


_POSITION = re.compile(r"^(?P<chain>[^:\d]+):?(?P<resid>-?\d+)$")


def _parse_position(value: str) -> tuple[str, int]:
    match = _POSITION.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"Invalid residue position {value!r}; expected A42 or A:42")
    return match.group("chain"), int(match.group("resid"))


def protein_positions_in_clashes(
    atom_array,
    *,
    distance: float = 2.0,
    sequence_separation: int = 1,
) -> list[str]:
    """Return residues participating in nonbonded protein-heavy-atom clashes.

    Same-residue and nearby same-chain pairs are excluded exactly as in the
    all-heavy clash gate. The result is suitable for bounded fixed-identity
    rotamer repacking; it cannot repair a backbone-backbone collision.
    """

    from biotite.structure import filter_amino_acids

    if distance <= 0:
        raise ValueError("clash distance must be positive")
    if sequence_separation < 0:
        raise ValueError("sequence separation must be non-negative")

    keep = filter_amino_acids(atom_array) & (np.asarray(atom_array.element) != "H")
    if hasattr(atom_array, "hetero"):
        keep &= ~np.asarray(atom_array.hetero, dtype=bool)
    protein = atom_array[keep]
    if len(protein) < 2:
        return []

    pairs = np.asarray(
        sorted(cKDTree(protein.coord.astype(np.float64)).query_pairs(r=distance)),
        dtype=int,
    )
    if pairs.size == 0:
        return []
    left, right = pairs[:, 0], pairs[:, 1]
    same_chain = protein.chain_id[left] == protein.chain_id[right]
    nearby = same_chain & (
        np.abs(protein.res_id[left].astype(int) - protein.res_id[right].astype(int))
        <= sequence_separation
    )
    pairs = pairs[~nearby]
    return sorted(
        {
            f"{protein.chain_id[index]}{int(protein.res_id[index])}"
            for index in pairs.ravel()
        }
    )


def repack_selected_sidechains(
    pose,
    positions: list[str],
    *,
    scorefxn_name: str = "beta_nov16",
    extra_rotamers: bool = True,
) -> tuple[float, float]:
    """Repack selected pose residues in place without redesigning identities."""

    import pyrosetta as pyr

    pose_indices = []
    for position in positions:
        chain, residue = _parse_position(position)
        pose_index = pose.pdb_info().pdb2pose(chain, residue)
        if pose_index == 0:
            raise KeyError(f"Protein residue {position!r} was not found")
        pose_indices.append(pose_index)
    selected = set(pose_indices)

    task = pyr.standard_packer_task(pose)
    task.restrict_to_repacking()
    task.or_include_current(True)
    for pose_index in range(1, pose.total_residue() + 1):
        residue_task = task.nonconst_residue_task(pose_index)
        if pose_index not in selected:
            residue_task.prevent_repacking()
            continue
        residue_task.restrict_to_repacking()
        if extra_rotamers:
            residue_task.or_ex1(True)
            residue_task.or_ex2(True)
            residue_task.or_ex3(True)
            residue_task.or_ex4(True)

    scorefxn = pyr.rosetta.core.scoring.ScoreFunctionFactory.create_score_function(
        scorefxn_name
    )
    score_before = float(scorefxn(pose))
    packer = pyr.rosetta.protocols.minimization_packing.PackRotamersMover()
    packer.score_function(scorefxn)
    packer.task(task)
    packer.apply(pose)
    return score_before, float(scorefxn(pose))


class SelectiveSidechainRepackBlock(InlineBlockFunction):
    """Resolve local protein clashes without moving the backbone or ligand.

    Only the listed sidechains may select a new rotamer. Their amino-acid
    identities remain fixed; all other protein coordinates are retained and
    hetero atoms are copied back unchanged.
    """

    _name = "selective_sidechain_repack"
    _display_name = "Selective Sidechain Repack"
    _description = "Repack only explicitly selected protein sidechains"
    _inline = True
    _chunked = True
    _creates_new_step = True
    _resources = JobArgs(partition="cpu", mem_gb=8, rows_per_job=50)

    positions: list[str] = Field(
        default_factory=list,
        description="Protein residue IDs to repack, such as ['A42', 'A:73'].",
    )
    clash_distance: float | None = Field(
        None,
        gt=0,
        description=(
            "When set, also detect and repack each structure's residues in "
            "nonbonded protein-heavy-atom contacts below this distance."
        ),
    )
    sequence_separation: int = Field(
        1,
        ge=0,
        description="Same-chain residue separation excluded from clash detection.",
    )
    scorefxn_name: str = Field(
        "beta_nov16",
        description="Rosetta score function used to select fixed-identity rotamers.",
    )
    extra_rotamers: bool = Field(
        True,
        description="Include extra chi1-chi4 sampling at the selected positions.",
    )

    @field_validator("positions")
    @classmethod
    def _validate_positions(cls, positions: list[str]) -> list[str]:
        for position in positions:
            _parse_position(position)
        if len(set(positions)) != len(positions):
            raise ValueError("sidechain-repack positions must be unique")
        return positions

    @model_validator(mode="after")
    def _validate_selection_mode(self):
        if not self.positions and self.clash_distance is None:
            raise ValueError("provide positions or clash_distance")
        return self

    def execute(
        self,
        handler: PipelineDataHandler,
        sandbox_dir: Path,
        chunk_id: int = 0,
    ) -> PipelineDataHandler:
        from atomworks.io.utils.io_utils import to_cif_file
        from chaperone.io import ensure_bonds
        from biotite.structure import concatenate
        from biotite.structure.io import load_structure

        from blocks.utils.pyrosetta import atom_array_to_pose, init_pyrosetta

        init_pyrosetta()
        sandbox_dir = Path(sandbox_dir)
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        temporary_pdb = sandbox_dir / f".repack_chunk{chunk_id}.pdb"

        for loaded in handler.get_data(which="last", load=True):
            source = loaded.atom_array
            automatic_positions = (
                protein_positions_in_clashes(
                    source,
                    distance=self.clash_distance,
                    sequence_separation=self.sequence_separation,
                )
                if self.clash_distance is not None
                else []
            )
            positions = list(dict.fromkeys([*self.positions, *automatic_positions]))
            hetero_mask = np.asarray(source.hetero, dtype=bool)
            protein = source[~hetero_mask]
            hetero = source[hetero_mask]
            pose = atom_array_to_pose(protein)
            score_before, score_after = repack_selected_sidechains(
                pose,
                positions,
                scorefxn_name=self.scorefxn_name,
                extra_rotamers=self.extra_rotamers,
            )
            pose.dump_pdb(str(temporary_pdb))
            repacked_protein = load_structure(temporary_pdb, model=1)
            combined = (
                concatenate([repacked_protein, hetero]) if len(hetero) else repacked_protein
            )
            output_path = sandbox_dir / f"{Path(loaded.path).stem}.repacked.cif"
            to_cif_file(ensure_bonds(combined), str(output_path))
            rows.append(
                make_row(
                    name=self.name,
                    path=str(output_path),
                    metadata={"source_path": loaded.path, "positions": positions},
                    metrics={
                        "score_before": score_before,
                        "score_after": score_after,
                        "score_delta": score_after - score_before,
                        "repacked_positions": positions,
                        "automatic_clash_positions": automatic_positions,
                    },
                )
            )

        temporary_pdb.unlink(missing_ok=True)
        return PipelineDataHandler(rows)
