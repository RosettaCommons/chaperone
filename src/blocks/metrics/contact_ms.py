"""Contact Molecular Surface (CMS) scoring block."""

import logging
from typing import ClassVar, List, Optional

import numpy as np
from pydantic import Field

from chaperone import InlineMetricBlock, JobArgs

logger = logging.getLogger(__name__)


class ContactMSBlock(InlineMetricBlock):
    """Compute contact molecular surface between binder and target chains.

    Uses the py_contact_ms implementation to calculate distance-weighted
    surface area (A^2) of the target buried by the binder.
    """

    _name = "contact_ms"
    _display_name = "Contact Molecular Surface"
    _description = "Compute contact molecular surface between binder and target chains"
    _creates_new_step = False
    _resources: ClassVar[JobArgs] = JobArgs(partition="cpu")

    binder_chains: List[str] = Field(..., description="Chain IDs for the binder")
    target_chains: List[str] = Field(..., description="Chain IDs for the target")
    target_atom_names: Optional[List[str]] = Field(
        None,
        description=(
            "Restrict CMS aggregation to these target atom names (e.g. ['C12', 'O3']). "
            "The molecular surface is still generated from the FULL target so occlusion "
            "is physically correct — only the final per-atom sum is masked."
        ),
    )
    target_residues: Optional[List[str]] = Field(
        None,
        description=(
            "Restrict CMS aggregation to these target residues, as chain+number "
            "identifiers (e.g. ['B56', 'B121']). Use for epitope-directed "
            "designs, where a binder can satisfy every global metric while "
            "packing against the wrong face. As with target_atom_names the "
            "surface comes from the FULL target, so occlusion stays correct "
            "and only the final sum is masked."
        ),
    )

    def compute_metrics(self, loaded, *, context=None, index=0):
        from blocks.metrics.py_contact_ms import (
            calculate_contact_ms,
            get_radii_from_names,
        )

        atom_array = loaded.atom_array
        all_chain_ids = set(np.unique(atom_array.chain_id))

        missing_binder = set(self.binder_chains) - all_chain_ids
        missing_target = set(self.target_chains) - all_chain_ids
        if missing_binder or missing_target:
            logger.warning(
                "contact_ms: chain(s) not found in %s (missing binder=%s, target=%s, available=%s) "
                "— setting cms to 0.0",
                loaded.path,
                missing_binder or "{}",
                missing_target or "{}",
                sorted(all_chain_ids),
            )
            return 0.0

        binder_mask = np.isin(atom_array.chain_id, self.binder_chains)
        target_mask = np.isin(atom_array.chain_id, self.target_chains)

        binder_atoms = atom_array[binder_mask]
        target_atoms = atom_array[target_mask]

        binder_res = [s.strip() for s in binder_atoms.res_name]
        binder_anames = [s.strip() for s in binder_atoms.atom_name]
        target_res = [s.strip() for s in target_atoms.res_name]
        target_anames = [s.strip() for s in target_atoms.atom_name]

        binder_radii = get_radii_from_names(binder_res, binder_anames)
        target_radii = get_radii_from_names(target_res, target_anames)

        # Filter out atoms with no matching radius
        binder_valid = binder_radii > 0
        target_valid = target_radii > 0

        if not np.any(binder_valid) or not np.any(target_valid):
            logger.warning("All radii are zero for %s — returning zeros", loaded.path)
            return 0.0

        binder_xyz = binder_atoms.coord[binder_valid]
        binder_r = binder_radii[binder_valid]
        target_xyz = target_atoms.coord[target_valid]
        target_r = target_radii[target_valid]

        # Drop binder atoms beyond 8Å of any target atom, as it helps us save compute time
        from scipy.spatial.distance import cdist

        near = cdist(binder_xyz, target_xyz).min(axis=1) < 8.5
        binder_xyz, binder_r = binder_xyz[near], binder_r[near]

        try:
            cms, per_atom_target_cms = calculate_contact_ms(
                binder_xyz, binder_r, target_xyz, target_r
            )
        except Exception as e:
            # Safety net for any remaining geometry failure
            logger.warning("CMS failed for %s (%s) — cms=0", loaded.path, e)
            return 0.0

        if self.target_residues is not None:
            wanted = {r.strip() for r in self.target_residues}
            ids = np.asarray(
                [
                    f"{c.strip()}{r}"
                    for c, r in zip(target_atoms.chain_id, target_atoms.res_id)
                ]
            )[target_valid]
            sel = np.isin(ids, list(wanted))
            missing = wanted - set(ids[sel])
            if missing:
                # Silence here would report a plausible number computed over
                # nothing, so say which residues were not found.
                logger.warning(
                    "contact_ms: requested target residues absent from %s: %s",
                    loaded.path,
                    sorted(missing),
                )
            cms = float(per_atom_target_cms[sel].sum())

        elif self.target_atom_names is not None:
            # per_atom_target_cms aligns with target_atoms[target_valid], so index
            # the stripped name list with the same mask before building the selection.
            valid_names = np.asarray(target_anames)[target_valid]
            sel = np.isin(valid_names, self.target_atom_names)
            missing = set(self.target_atom_names) - set(valid_names[sel])
            if missing:
                logger.warning(
                    "contact_ms: requested target atoms not present (or radius-filtered) "
                    "in %s: %s",
                    loaded.path,
                    sorted(missing),
                )
            cms = float(per_atom_target_cms[sel].sum())

        logger.info("Computed CMS for %s: cms=%.2f", loaded.path, cms)

        return cms
