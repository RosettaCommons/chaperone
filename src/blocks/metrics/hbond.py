"""HBond scoring block for computing hydrogen bond metrics for ligand interactions."""

import logging
import os
from typing import Optional

import numpy as np
from pydantic import Field

from chaperone import InlineMetricBlock

logger = logging.getLogger(__name__)


class HBondBlock(InlineMetricBlock):
    """Compute hydrogen bond metrics between ligand and protein.

    Uses HBPLUS to identify hydrogen bonds between motif (ligand) and
    non-motif atoms in the last structure of each row.
    """

    _name = "hbond_metrics"
    _display_name = "H-Bond Metrics"
    _description = "Compute hydrogen bond metrics for ligand interactions"
    _creates_new_step = False

    atom_groups: Optional[dict[str, list[str]]] = Field(
        default=None,
        description="Mapping of group name to list of ligand atom names. "
        "When provided, per-group h-bond counts are included in the output.",
    )
    verbose: bool = Field(
        default=False,
        description="When True, output also includes a `pairs` list with "
        "per-hbond atom-level detail (protein_residue, protein_atom, "
        "ligand_atom, distance, role).",
    )
    hbplus_path: Optional[str] = Field(
        default=None,
        description=(
            "Explicit HBPLUS executable. Resolved once during setup and "
            "propagated to workers so they do not depend on worker-local settings."
        ),
    )

    def setup(self, handler, sandbox_dir, chunk_id):
        from chaperone import get_setting_field

        hbplus_path = self.hbplus_path or get_setting_field(
            "HBPLUS_COPY", check_bin=True
        )
        if not os.path.isfile(hbplus_path) or not os.access(hbplus_path, os.X_OK):
            raise FileNotFoundError(
                f"HBPLUS capability is unavailable or not executable: {hbplus_path}"
            )
        return {
            "hbplus_path": hbplus_path,
            "hbplus_available": True,
        }

    def compute_metrics(self, loaded, *, context=None, index=0):
        from blocks.utils.hbonds import calculate_hbonds

        atom_array = loaded.atom_array

        if context["hbplus_available"]:
            try:
                annotated_array, motif_hbonds, group_counts = calculate_hbonds(
                    atom_array,
                    atom_groups=self.atom_groups,
                    hbplus_path=context["hbplus_path"],
                )
                metrics = {"num_hbonds": len(motif_hbonds)}
                if group_counts:
                    metrics.update(group_counts)
                if self.verbose:
                    metrics["pairs"] = _build_pair_records(
                        annotated_array, motif_hbonds
                    )
                logger.info(
                    "Computed hbond metrics for %s: num_hbonds=%d",
                    loaded.path,
                    metrics["num_hbonds"],
                )
            except Exception as e:
                logger.error(
                    "Failed to compute hbond metrics for %s: %s",
                    loaded.path,
                    e,
                    exc_info=True,
                )
                raise

        return metrics


def _build_pair_records(atom_array, motif_hbonds: list[dict]) -> list[dict]:
    """Convert HBPLUS motif_hbonds into atom-level pair records.

    For each H-bond, determines which side is the ligand using
    `atom_array.is_ligand` and emits one dict with both protein and ligand
    atom names plus distance and role.
    """
    pairs: list[dict] = []
    for hb in motif_hbonds:
        d_mask = (
            (atom_array.chain_iid == hb["d_chain"])
            & (atom_array.res_id == float(hb["d_resi"]))
            & (atom_array.atom_name == hb["d_atom"])
        )
        d_hits = np.where(d_mask)[0]
        if len(d_hits) != 1:
            continue
        ligand_is_donor = bool(atom_array.is_ligand[d_hits[0]])
        if ligand_is_donor:
            prot_chain = hb["a_chain"]
            prot_resi = hb["a_resi"]
            prot_resn = hb["a_resn"]
            prot_atom = hb["a_atom"]
            lig_resn = hb["d_resn"]
            lig_atom = hb["d_atom"]
            role = "ligand_donor"
        else:
            prot_chain = hb["d_chain"]
            prot_resi = hb["d_resi"]
            prot_resn = hb["d_resn"]
            prot_atom = hb["d_atom"]
            lig_resn = hb["a_resn"]
            lig_atom = hb["a_atom"]
            role = "protein_donor"
        pairs.append(
            {
                "protein_residue": f"{prot_chain}{prot_resi}",
                "protein_resname": prot_resn,
                "protein_atom": prot_atom,
                "ligand_resname": lig_resn,
                "ligand_atom": lig_atom,
                "distance": float(hb["dist"]),
                "role": role,
            }
        )
    return pairs
