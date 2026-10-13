"""Restore selected side-chain conformers from an upstream structure.

Sequence models commonly keep motif residue identities fixed while still
repacking their side chains.  For ligand pockets this can erase the very
rotamers that define the motif.  This block copies only selected side-chain
atom coordinates from an upstream reference into the current structure and
leaves the designed sequence, backbone, ligand, and all other side chains
unchanged.  Downstream clash and interaction metrics must still validate the
result.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from pydantic import Field

from pipelines import InlineBlockFunction, JobArgs, PipelineDataHandler
from pipelines.data.schema import make_row


BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O"})
BACKBONE_ALIGNMENT_ATOMS = ("N", "CA", "C", "O")
_POSITION = re.compile(r"^(?P<chain>[^:\d]+):?(?P<resid>-?\d+)$")
_MPNN_NO_FIXED_POSITIONS = "_suppress_cif_motif_inference"


def restorable_positions(values: list[str]) -> list[str]:
    """Remove LigandMPNN's internal no-fixed-position sentinel.

    LigandMPNN writes the sentinel into its JSON file to suppress automatic CIF
    motif inference when every protein identity remains designable.  It is
    control metadata, not a residue identifier, so downstream restoration must
    treat it as an empty position list.
    """

    return [value for value in values if value != _MPNN_NO_FIXED_POSITIONS]


def protein_positions_near_ligand(
    atom_array,
    ligand_resnames: list[str],
    *,
    distance: float = 5.0,
) -> list[str]:
    """Return protein residue IDs with a heavy atom near the selected ligand.

    This produces the same geometric shell definition used by the ligand-pocket
    scorer, but returns compact ``A42`` identifiers suitable for selective
    sidechain restoration. It intentionally does not include remote fixed
    sequence positions.
    """

    if distance <= 0:
        raise ValueError("Ligand-contact distance must be positive")
    heavy = np.asarray(atom_array.element) != "H"
    hetero = np.asarray(atom_array.hetero, dtype=bool)
    ligand_mask = np.isin(atom_array.res_name, ligand_resnames) & heavy
    protein_mask = ~hetero & heavy
    ligand = atom_array[ligand_mask]
    protein = atom_array[protein_mask]
    if len(ligand) == 0:
        raise ValueError(f"No selected ligand atoms found: {sorted(ligand_resnames)}")
    if len(protein) == 0:
        raise ValueError("No protein heavy atoms found")

    distances = np.linalg.norm(
        protein.coord[:, None, :] - ligand.coord[None, :, :], axis=2
    )
    close = np.flatnonzero(distances.min(axis=1) <= distance)
    return sorted(
        {
            f"{protein.chain_id[index]}{int(protein.res_id[index])}"
            for index in close
        }
    )


def _parse_position(value: str) -> tuple[str, int]:
    """Parse ``A42`` or ``A:42`` into a chain/residue key."""

    match = _POSITION.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"Invalid residue position {value!r}; expected A42 or A:42")
    return match.group("chain"), int(match.group("resid"))


def restore_selected_sidechains(
    reference,
    designed,
    positions: list[str],
    *,
    align_backbone: bool = False,
):
    """Return a designed-array copy with selected reference rotamers restored.

    A residue is restored only when its residue name is identical in both
    structures.  Atom names are matched explicitly, so differing atom order is
    harmless and backbone coordinates are never copied.
    """

    output = designed.copy()
    restored_residues = 0
    restored_atoms = 0
    skipped: list[str] = []

    for position in positions:
        chain, resid = _parse_position(position)
        ref_mask = (reference.chain_id == chain) & (reference.res_id == resid)
        out_mask = (output.chain_id == chain) & (output.res_id == resid)
        ref_names = np.unique(reference.res_name[ref_mask])
        out_names = np.unique(output.res_name[out_mask])
        if len(ref_names) != 1 or len(out_names) != 1 or ref_names[0] != out_names[0]:
            skipped.append(position)
            continue

        transformation = None
        if align_backbone:
            from biotite.structure import superimpose

            reference_backbone = []
            output_backbone = []
            for atom_name in BACKBONE_ALIGNMENT_ATOMS:
                ref_atom = np.flatnonzero(ref_mask & (reference.atom_name == atom_name))
                out_atom = np.flatnonzero(out_mask & (output.atom_name == atom_name))
                if len(ref_atom) == 1 and len(out_atom) == 1:
                    reference_backbone.append(reference.coord[ref_atom[0]])
                    output_backbone.append(output.coord[out_atom[0]])
            if len(reference_backbone) < 3:
                skipped.append(position)
                continue
            _, transformation = superimpose(
                np.asarray(output_backbone),
                np.asarray(reference_backbone),
            )

        copied = 0
        ref_indices = np.flatnonzero(ref_mask)
        for ref_index in ref_indices:
            atom_name = str(reference.atom_name[ref_index])
            if atom_name in BACKBONE_ATOMS or str(reference.element[ref_index]) == "H":
                continue
            target = np.flatnonzero(out_mask & (output.atom_name == atom_name))
            if len(target) != 1:
                continue
            coordinate = reference.coord[ref_index]
            if transformation is not None:
                coordinate = transformation.apply(coordinate[np.newaxis, :])[0]
            output.coord[target[0]] = coordinate
            copied += 1

        if copied:
            restored_residues += 1
            restored_atoms += copied

    return output, restored_residues, restored_atoms, skipped


class RestoreSidechainsBlock(InlineBlockFunction):
    """Restore explicitly selected side-chain conformers from an earlier step.

    ``positions_json`` uses the same shape as LigandMPNN's fixed-position file:
    structure paths map to lists such as ``["A20", "A68"]``.  Matching is by
    basename so staged copies and their original paths resolve identically.
    """

    _name = "restore_sidechains"
    _display_name = "Restore Sidechains"
    _description = "Restore selected motif rotamers from an upstream structure"
    _inline = True
    _chunked = True
    _creates_new_step = True
    _resources = JobArgs(partition="cpu", mem_gb=8, rows_per_job=500)

    positions_json: str = Field(
        ...,
        description="JSON mapping reference structure paths to residue positions.",
    )
    reference_index: int = Field(
        0,
        description="Index of the reference structure in each lineage row.",
    )
    align_backbone: bool = Field(
        False,
        description=(
            "Rigidly align each reference residue's backbone onto the current "
            "backbone before copying its side chain. Use for predicted or "
            "relaxed structures whose backbone moved from the reference."
        ),
    )

    def execute(
        self,
        handler: PipelineDataHandler,
        sandbox_dir: Path,
        chunk_id: int = 0,
    ) -> PipelineDataHandler:
        from atomworks.io.utils.io_utils import to_cif_file
        from pipelines.io import ensure_bonds

        sandbox_dir = Path(sandbox_dir)
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        mapping = json.loads(Path(self.positions_json).read_text())
        by_basename = {Path(path).name: positions for path, positions in mapping.items()}

        rows = []
        which = [self.reference_index, -1]
        for index, (reference, designed) in enumerate(
            handler.get_data(which=which, load=True)
        ):
            positions = by_basename.get(Path(reference.path).name)
            if positions is None:
                raise KeyError(
                    f"No side-chain positions for reference {Path(reference.path).name!r}"
                )
            positions = restorable_positions(positions)
            restored, n_residues, n_atoms, skipped = restore_selected_sidechains(
                reference.atom_array,
                designed.atom_array,
                positions,
                align_backbone=self.align_backbone,
            )
            output_path = sandbox_dir / f"{Path(designed.path).stem}.restored.cif"
            to_cif_file(ensure_bonds(restored), str(output_path))
            rows.append(
                make_row(
                    name=self.name,
                    path=str(output_path),
                    metadata={
                        "reference_path": reference.path,
                        "source_path": designed.path,
                    },
                    metrics={
                        "restored_residues": n_residues,
                        "restored_atoms": n_atoms,
                        "skipped_positions": skipped,
                    },
                )
            )

        return PipelineDataHandler(rows)
