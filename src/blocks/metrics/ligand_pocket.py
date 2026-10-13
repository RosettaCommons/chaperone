"""Distributed ligand-contact, enclosure, and pocket-shell metrics."""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
from pydantic import Field

from pipelines import InlineMetricBlock


logger = logging.getLogger(__name__)


def _normalized_entropy(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[values > 0]
    if len(values) <= 1:
        return float(len(values))
    probabilities = values / values.sum()
    return float(-(probabilities * np.log(probabilities)).sum() / np.log(len(values)))


def _residue_key(array, index: int) -> str:
    return (
        f"{array.chain_id[index]}{int(array.res_id[index])}:"
        f"{array.res_name[index]}"
    )


def compute_ligand_pocket_metrics(
    atom_array,
    *,
    ligand_resnames: list[str],
    core_atom_names: list[str],
    tail_polar_atom_names: list[str],
    contact_distance: float = 5.0,
    polar_distance: float = 3.6,
    clash_distance: float = 2.0,
) -> dict:
    """Measure contact area and how evenly a protein encloses one ligand.

    Contact molecular surface (CMS) is computed on the complete selected
    ligand. Geometric coverage and the shell use protein heavy atoms within
    ``contact_distance``. Polar support counts protein N/O/S atoms within
    ``polar_distance`` of each requested tail atom.
    """

    # Some structures arrive with bond tables produced by a different Biotite
    # build. Bond-free metric calculations avoid slicing those incompatible
    # tables while leaving coordinates and annotations unchanged.
    working = atom_array.copy()
    working.bonds = None
    protein_mask = ~np.asarray(working.hetero, dtype=bool)
    ligand_mask = np.isin(working.res_name, ligand_resnames)
    protein = working[protein_mask & (working.element != "H")]
    ligand = working[ligand_mask & (working.element != "H")]
    if len(ligand) == 0:
        return {"valid": False, "reason": "selected ligand atoms not found"}
    if len(protein) == 0:
        return {"valid": False, "reason": "protein heavy atoms not found"}

    atom_names = [str(name) for name in ligand.atom_name]
    core_present = [name for name in core_atom_names if name in atom_names]
    tail_present = [name for name in tail_polar_atom_names if name in atom_names]

    distances = np.linalg.norm(
        protein.coord[:, None, :] - ligand.coord[None, :, :], axis=2
    )
    minimum_protein_ligand_distance = float(distances.min())
    protein_ligand_clash_count = int((distances < clash_distance).sum())
    per_atom_heavy_contacts = (distances <= contact_distance).sum(axis=0)
    covered = per_atom_heavy_contacts > 0

    shell_atom_mask = distances.min(axis=1) <= contact_distance
    shell_indices = np.flatnonzero(shell_atom_mask)
    shell_residues = sorted({_residue_key(protein, int(i)) for i in shell_indices})

    polar_protein_mask = np.isin(protein.element, ("N", "O", "S"))
    polar_distances = distances[polar_protein_mask]
    polar_protein = protein[polar_protein_mask]
    tail_support_counts: dict[str, int] = {}
    tail_support_residues: dict[str, list[str]] = {}
    for tail_name in tail_polar_atom_names:
        ligand_indices = np.flatnonzero(ligand.atom_name == tail_name)
        if not len(ligand_indices):
            tail_support_counts[tail_name] = 0
            tail_support_residues[tail_name] = []
            continue
        close = np.flatnonzero(
            polar_distances[:, ligand_indices].min(axis=1) <= polar_distance
        )
        residues = sorted({_residue_key(polar_protein, int(i)) for i in close})
        tail_support_counts[tail_name] = len(residues)
        tail_support_residues[tail_name] = residues

    from blocks.metrics.py_contact_ms import (
        calculate_contact_ms,
        get_radii_from_names,
    )

    protein_radii = get_radii_from_names(
        [str(name).strip() for name in protein.res_name],
        [str(name).strip() for name in protein.atom_name],
    )
    ligand_radii = get_radii_from_names(
        [str(name).strip() for name in ligand.res_name],
        [str(name).strip() for name in ligand.atom_name],
    )
    protein_valid = protein_radii > 0
    ligand_valid = ligand_radii > 0
    nearby = distances[:, ligand_valid].min(axis=1) < 8.5
    protein_valid &= nearby
    contact_error = None
    try:
        contact_ms, per_atom_contact_ms_valid = calculate_contact_ms(
            protein.coord[protein_valid],
            protein_radii[protein_valid],
            ligand.coord[ligand_valid],
            ligand_radii[ligand_valid],
        )
    except RuntimeError as exc:
        # The surface implementation rejects physically malformed inputs in
        # which one atomic sphere contains another (typically collapsed atoms
        # in a generated structure). One bad design must fail its metric gate,
        # not abort scoring for an otherwise valid campaign batch.
        contact_error = str(exc)
        contact_ms = 0.0
        per_atom_contact_ms_valid = np.zeros(ligand_valid.sum(), dtype=float)
        logger.warning("Contact molecular surface rejected structure: %s", exc)
    per_atom_contact_ms = np.zeros(len(ligand), dtype=float)
    per_atom_contact_ms[ligand_valid] = per_atom_contact_ms_valid

    # Aggregate duplicate atom names defensively (a selected ligand should
    # normally be one residue, but the block should not silently overwrite).
    contact_ms_by_name: dict[str, float] = defaultdict(float)
    heavy_contacts_by_name: dict[str, int] = defaultdict(int)
    for name, area, count in zip(
        atom_names, per_atom_contact_ms, per_atom_heavy_contacts
    ):
        contact_ms_by_name[name] += float(area)
        heavy_contacts_by_name[name] += int(count)

    from blocks.utils.rasa import get_ligand_rasa

    selected = working[protein_mask | ligand_mask]
    rasa = get_ligand_rasa(selected)
    enclosure = 1.0 - float(rasa["mean_ligand_atom_rasa"])

    core_mask = np.isin(ligand.atom_name, core_atom_names)
    tail_mask = np.isin(ligand.atom_name, tail_polar_atom_names)
    supported_tail = [
        name for name in tail_polar_atom_names if tail_support_counts.get(name, 0) > 0
    ]

    positive_area = per_atom_contact_ms[per_atom_contact_ms > 0]
    total_area = float(contact_ms)
    max_area_fraction = (
        float(positive_area.max() / total_area)
        if len(positive_area) and total_area > 0
        else 0.0
    )
    return {
        "valid": contact_error is None,
        "reason": (
            None
            if contact_error is None
            else f"contact molecular surface failed: {contact_error}"
        ),
        "ligand_contact_ms": total_area,
        "minimum_protein_ligand_distance": minimum_protein_ligand_distance,
        "protein_ligand_clash_count": protein_ligand_clash_count,
        "enclosure": enclosure,
        "shell_residue_count": len(shell_residues),
        "shell_residues": shell_residues,
        "ligand_heavy_atom_count": len(ligand),
        "ligand_atom_coverage_fraction": float(covered.mean()),
        "covered_ligand_atoms": sorted(
            name for name, is_covered in zip(atom_names, covered) if is_covered
        ),
        "uncovered_ligand_atoms": sorted(
            name for name, is_covered in zip(atom_names, covered) if not is_covered
        ),
        "core_atom_count": len(core_present),
        "core_atom_coverage_fraction": (
            float(covered[core_mask].mean()) if np.any(core_mask) else 0.0
        ),
        "tail_atom_count": len(tail_present),
        "tail_atom_coverage_fraction": (
            float(covered[tail_mask].mean()) if np.any(tail_mask) else 0.0
        ),
        "tail_polar_support_fraction": (
            len(supported_tail) / len(tail_polar_atom_names)
            if tail_polar_atom_names
            else 1.0
        ),
        "tail_polar_supported_atoms": supported_tail,
        "tail_polar_support_counts": tail_support_counts,
        "tail_polar_support_residues": tail_support_residues,
        "per_atom_heavy_contact_counts": dict(heavy_contacts_by_name),
        "minimum_atom_heavy_contacts": int(per_atom_heavy_contacts.min()),
        "per_atom_contact_ms": dict(contact_ms_by_name),
        "contact_ms_evenness": _normalized_entropy(per_atom_contact_ms),
        "maximum_atom_contact_ms_fraction": max_area_fraction,
    }


class LigandPocketMetricsBlock(InlineMetricBlock):
    """Quantify ligand CMS, enclosure, shell depth, and contact distribution."""

    _name = "ligand_pocket"
    _display_name = "Ligand Pocket"
    _description = "Distributed ligand contact, enclosure, and shell metrics"
    _creates_new_step = False

    ligand_resnames: list[str] = Field(
        ...,
        description="Ligand residue names to measure, e.g. ['ADN'].",
    )
    core_atom_names: list[str] = Field(
        default_factory=list,
        description="Ligand core atoms that must all receive protein contact.",
    )
    tail_polar_atom_names: list[str] = Field(
        default_factory=list,
        description="Ligand tail atoms requiring explicit polar support.",
    )
    contact_distance: float = Field(
        5.0,
        gt=0,
        description="Protein-heavy-atom distance defining coverage and shell.",
    )
    polar_distance: float = Field(
        3.6,
        gt=0,
        description="Maximum protein N/O/S distance defining polar support.",
    )
    clash_distance: float = Field(
        2.0,
        gt=0,
        description=(
            "Protein-ligand heavy-atom distance below which a severe clash is counted."
        ),
    )

    def compute_metrics(self, loaded, *, context=None, index=0):
        metrics = compute_ligand_pocket_metrics(
            loaded.atom_array,
            ligand_resnames=self.ligand_resnames,
            core_atom_names=self.core_atom_names,
            tail_polar_atom_names=self.tail_polar_atom_names,
            contact_distance=self.contact_distance,
            polar_distance=self.polar_distance,
            clash_distance=self.clash_distance,
        )
        logger.info(
            "Computed ligand pocket for %s: cms=%s shell=%s enclosure=%s",
            loaded.path,
            metrics.get("ligand_contact_ms"),
            metrics.get("shell_residue_count"),
            metrics.get("enclosure"),
        )
        return metrics
