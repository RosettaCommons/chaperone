"""Aromatic neighbors scoring block for counting aromatic residues near ligands."""

import logging

from pydantic import Field

from pipelines import InlineMetricBlock

logger = logging.getLogger(__name__)


class AromaticNeighborsBlock(InlineMetricBlock):
    """Count aromatic residues near ligand atoms.

    Finds aromatic residues (PHE, TYR, TRP, HIS) within a distance threshold
    of ligand atoms in the last structure of each row.
    """

    _name = "aromatic_neighbors"
    _display_name = "Aromatic Neighbors"
    _description = "Count aromatic residues near ligand atoms"
    _creates_new_step = False

    distance_threshold: float = Field(
        default=6,
        description="Distance threshold in Angstroms for finding aromatic neighbors",
    )
    ligand_atom_names: list[str] | None = Field(
        default=None,
        description=(
            "Optional canonical ligand atom names to use as the distance query. "
            "None counts neighbors around the complete ligand."
        ),
    )

    def compute_metrics(self, loaded, *, context=None, index=0):
        from blocks.util_blocks.annotate import find_aromatic_residues_near_ligand

        try:
            aromatic_residues = find_aromatic_residues_near_ligand(
                loaded.atom_array,
                distance_threshold=self.distance_threshold,
                ligand_atom_names=self.ligand_atom_names,
            )
            metrics = {"num_aromatic_neighbors": len(aromatic_residues)}
            logger.info(
                "Computed aromatic neighbors for %s: num_aromatic_neighbors=%d",
                loaded.path,
                metrics["num_aromatic_neighbors"],
            )
        except Exception as e:
            # One unusual structure should not take down a whole campaign's
            # metric stage. Report the count as missing rather than zero: zero
            # is a real measurement meaning "no aromatics nearby", and silently
            # conflating the two would let failures pass a >= 1 filter as
            # genuine negatives.
            logger.error(
                "Failed to compute aromatic neighbors for %s: %s",
                loaded.path,
                e,
                exc_info=True,
            )
            metrics = {"num_aromatic_neighbors": None}

        return metrics
