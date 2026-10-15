"""RASA scoring block for computing solvent accessible surface area for ligand atoms."""

import logging
from typing import Optional

from pydantic import Field

from chaperone import InlineMetricBlock

logger = logging.getLogger(__name__)


class RASABlock(InlineMetricBlock):
    """Compute solvent accessible surface area (SASA/RASA) for ligand atoms.

    Calculates the absolute and relative solvent accessible surface area
    for ligand atoms in the last structure of each row.
    """

    _name = "rasa"
    _display_name = "RASA"
    _description = "Compute solvent accessible surface area for ligand atoms"
    _creates_new_step = False

    atom_groups: Optional[dict[str, list[str]]] = Field(
        default=None,
        description="Mapping of group name to list of ligand atom names. "
        "When provided, per-group RASA aggregates are included in the output.",
    )

    probe_radius: float = Field(
        default=1.4,
        description="Radius of the solvent probe in Angstroms.",
    )

    backbone_only: bool = Field(
        default=False,
        description="If True, strip protein sidechains before SASA so ligand "
        "accessibility is measured against backbone atoms (N, CA, C, O) only.",
    )

    def compute_metrics(self, loaded, *, context=None, index=0):
        from blocks.utils.rasa import get_ligand_rasa

        try:
            metrics = get_ligand_rasa(
                loaded.atom_array,
                atom_groups=self.atom_groups,
                probe_radius=self.probe_radius,
                backbone_only=self.backbone_only,
            )
            logger.info(
                "Computed RASA for %s: ligand_rasa=%s",
                loaded.path,
                metrics.get("ligand_rasa", "N/A"),
            )
        except Exception as e:
            logger.error(
                "Failed to compute RASA for %s: %s",
                loaded.path,
                e,
                exc_info=True,
            )
            raise

        return metrics
