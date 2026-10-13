"""Fast protein-ligand heavy-atom clash metrics."""

from __future__ import annotations

import numpy as np
from pydantic import Field

from pipelines import InlineMetricBlock


def compute_ligand_clash_metrics(
    atom_array,
    *,
    ligand_resnames: list[str],
    cutoff: float = 2.0,
) -> dict:
    """Measure severe noncovalent protein-ligand penetrations."""

    ligand_mask = np.isin(atom_array.res_name, ligand_resnames)
    hetero = np.asarray(atom_array.hetero, dtype=bool)
    heavy = np.asarray(atom_array.element != "H")
    ligand = atom_array[ligand_mask & heavy]
    protein = atom_array[(~hetero) & (~ligand_mask) & heavy]
    if len(ligand) == 0:
        return {"valid": False, "reason": "selected ligand atoms not found"}
    if len(protein) == 0:
        return {"valid": False, "reason": "protein heavy atoms not found"}

    distances = np.linalg.norm(
        protein.coord[:, None, :] - ligand.coord[None, :, :], axis=2
    )
    protein_index, ligand_index = np.unravel_index(
        int(np.argmin(distances)), distances.shape
    )
    return {
        "valid": True,
        "n_clashes": int((distances < cutoff).sum()),
        "minimum_distance": float(distances[protein_index, ligand_index]),
        "closest_pair": {
            "protein_residue": (
                f"{protein.chain_id[protein_index]}"
                f"{int(protein.res_id[protein_index])}:{protein.res_name[protein_index]}"
            ),
            "protein_atom": str(protein.atom_name[protein_index]),
            "ligand_residue": (
                f"{ligand.chain_id[ligand_index]}"
                f"{int(ligand.res_id[ligand_index])}:{ligand.res_name[ligand_index]}"
            ),
            "ligand_atom": str(ligand.atom_name[ligand_index]),
        },
    }


class LigandClashMetricsBlock(InlineMetricBlock):
    """Reject severe protein-ligand heavy-atom penetrations cheaply."""

    _name = "ligand_clash"
    _display_name = "Ligand Clash Metrics"
    _description = "Measure the closest protein-ligand heavy-atom contact"
    _creates_new_step = False

    ligand_resnames: list[str] = Field(
        ...,
        description="Ligand residue names to score, e.g. ['ADN'].",
    )
    cutoff: float = Field(
        2.0,
        gt=0,
        description="Distance (A) below which a severe penetration is counted.",
    )

    def compute_metrics(self, loaded, *, context=None, index=0):
        return compute_ligand_clash_metrics(
            loaded.atom_array,
            ligand_resnames=self.ligand_resnames,
            cutoff=self.cutoff,
        )
