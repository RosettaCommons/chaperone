"""
Relative Accessible Surface Area (RASA) calculation utilities.

This module provides functionality to calculate RASA for ligand atoms in protein structures.
"""

import numpy as np
import biotite.structure as struc
import biotite.structure.info as info
from biotite.structure import sasa


def _mixed_vdw_radii(arr):
    """
    Compute VdW radii for atoms using mixed strategy.

    For protein/nucleic acid atoms, use protor radii (implicit H).
    For ligands and other hetero atoms, use element-based radii.

    Args:
        arr: AtomArray with element annotations.

    Returns:
        Array of VdW radii for each atom.
    """
    if getattr(arr, "element", None) is None or np.any(arr.element == ""):
        arr.element = struc.infer_elements(arr)

    aa = set(info.amino_acid_names())
    na = set(info.nucleotide_names())
    radii = np.empty(arr.array_length(), dtype=float)

    for i, (res, name, elem, hetero) in enumerate(
        zip(arr.res_name, arr.atom_name, arr.element, arr.hetero)
    ):
        try:
            if (not hetero) and (res in aa or res in na):
                radii[i] = info.vdw_radius_protor(res, name)
            else:
                raise KeyError
        except KeyError:
            r = info.vdw_radius_single(elem)
            if r is None:
                raise KeyError(f"No VdW radius for element '{elem}' at atom index {i}")
            radii[i] = r
    return radii


def _compute_rasa_from_asa(ligand_atoms, ligand_asa_vals, max_asa_lookup):
    """Compute per-atom RASA values from ASA values and a max-ASA lookup table.

    Returns:
        Tuple of (rasa_vals array, atom_names list)
    """
    rasa_vals = []
    atom_names = []
    for atom, abs_asa in zip(ligand_atoms, ligand_asa_vals):
        elem = atom.element
        atom_names.append(atom.atom_name)
        max_asa = max_asa_lookup.get(elem, np.nan)
        if np.isnan(max_asa) or max_asa <= 0:
            rasa_vals.append(np.nan)
        else:
            rasa_vals.append(abs_asa / max_asa)
    return np.array(rasa_vals, dtype=float), atom_names


def _aggregate_rasa(rasa_vals, atom_names):
    """Aggregate per-atom RASA into summary and named-atom metrics."""
    out = {"num_ligand_atoms": len(rasa_vals)}
    if len(rasa_vals) == 0:
        out["ligand_rasa"] = 0.0
        out["mean_ligand_atom_rasa"] = 0.0
        return out
    out["ligand_rasa"] = float(np.nansum(rasa_vals))
    out["mean_ligand_atom_rasa"] = float(np.nanmean(rasa_vals))
    for name, val in zip(atom_names, rasa_vals):
        out[f"ligand_atom_rasa.{name}"] = float(val)
    return out


def get_ligand_rasa(
    atom_array,
    max_asa_lookup: dict[str, float] | None = None,
    probe_radius: float = 1.4,
    atom_groups: dict[str, list[str]] | None = None,
    backbone_only: bool = False,
) -> dict[str, float]:
    """
    Compute absolute (ASA) and relative (RASA) solvent-accessible surface area for ligand atoms.

    Args:
        atom_array: The full molecular structure with `is_ligand` annotation.
        max_asa_lookup: Mapping from atom element to its reference maximum ASA in Angstroms^2.
            If None, a default table is used.
        probe_radius: Radius of the solvent probe in Angstroms (default 1.4).
        atom_groups: Optional mapping of group name to list of atom names.
            When provided, returns per-group aggregates nested under group keys,
            in addition to top-level totals.
        backbone_only: If True, strip protein sidechains before SASA so ligand
            accessibility is measured against backbone atoms (N, CA, C, O) only.
        (Very useful for broad accessibility)

    Returns:
        Dictionary containing:
            - num_ligand_atoms: int
            - ligand_rasa: float (total RASA, sum of fractions)
            - mean_ligand_atom_rasa: float (mean RASA per atom, 0-1)
            - ligand_atom_rasa.{atom_name}: float (per-atom RASA)
        When atom_groups is provided, additionally contains:
            - {group_name}: dict with the same keys scoped to that group's atoms
    """
    if max_asa_lookup is None:
        max_asa_lookup = {
            "C": 24.0,
            "N": 15.0,
            "O": 18.0,
            "S": 24.0,
            "P": 24.0,
            "H": 8.0,
        }

    from blocks.utils.geometry import get_backbone_mask, get_ligand_mask

    ligand_mask = get_ligand_mask(atom_array)

    if backbone_only:
        bkbn_mask = get_backbone_mask(atom_array) & ~atom_array.hetero
        keep = bkbn_mask | ligand_mask
        atom_array = atom_array[keep]
        ligand_mask = get_ligand_mask(atom_array)

    ligand_atoms = atom_array[ligand_mask]
    n_lig = len(ligand_atoms)

    out = {"num_ligand_atoms": n_lig}
    if n_lig == 0:
        out.update({"ligand_rasa": 0.0, "mean_ligand_atom_rasa": 0.0})
        if atom_groups:
            for group_name in atom_groups:
                out[group_name] = 0.0
        return out

    radii = _mixed_vdw_radii(atom_array)
    full_sasa = sasa(atom_array, probe_radius=probe_radius, vdw_radii=radii)

    ligand_asa_vals = full_sasa[ligand_mask]

    rasa_vals, atom_names = _compute_rasa_from_asa(
        ligand_atoms, ligand_asa_vals, max_asa_lookup
    )

    # Top-level aggregates (always present for backward compat)
    out.update(_aggregate_rasa(rasa_vals, atom_names))

    if atom_groups:
        for group_name, group_atom_names in atom_groups.items():
            group_set = set(group_atom_names)
            group_indices = [
                i for i, name in enumerate(atom_names) if name in group_set
            ]
            group_rasa = (
                rasa_vals[group_indices] if group_indices else np.array([], dtype=float)
            )
            if len(group_rasa) == 0:
                out[group_name] = 0.0
            else:
                out[group_name] = float(np.nanmedian(group_rasa))

    return out
