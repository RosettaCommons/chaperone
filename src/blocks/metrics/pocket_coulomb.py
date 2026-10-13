"""Pocket Coulomb Score block.

Calculates a continuous electrostatic complementarity score between
a small molecule's charged atoms and the charged residues (ASP, GLU, LYS, ARG)
in an automatically detected protein binding pocket.

Score convention:
  negative = favorable (opposite charges nearby)
  positive = unfavorable (same charges nearby)
"""

import logging
from typing import Optional

import numpy as np
from pydantic import Field

from chaperone import InlineMetricBlock

logger = logging.getLogger(__name__)

# Distribute formal charge across equivalent atoms
RESIDUE_CHARGE_MAP = {
    # Negative residues (-1 total)
    ("ASP", "OD1"): -0.5,
    ("ASP", "OD2"): -0.5,
    ("GLU", "OE1"): -0.5,
    ("GLU", "OE2"): -0.5,
    # Positive residues (+1 total)
    ("LYS", "NZ"): +1.0,
    ("ARG", "NH1"): +1.0 / 3,
    ("ARG", "NH2"): +1.0 / 3,
    ("ARG", "NE"): +1.0 / 3,
}

CHARGED_RESIDUES = {"ASP", "GLU", "LYS", "ARG"}


class PocketCoulombBlock(InlineMetricBlock):
    """Electrostatic complementarity between charged ligand atoms and pocket residues.

    Automatically detects the binding pocket (protein residues nearest the ligand)
    then computes a Coulomb-like score between charged atoms on both sides.
    """

    _name = "pocket_coulomb"
    _display_name = "Pocket Coulomb Score"
    _description = (
        "Electrostatic complementarity between charged ligand atoms "
        "and charged pocket residues"
    )
    _creates_new_step = False

    pocket_radius: float = Field(
        default=8.0,
        description="CB distance cutoff (A) for defining pocket residues around the ligand",
    )
    distance_floor: float = Field(
        default=1.5,
        description="Minimum distance (A) to prevent singularities from clashes",
    )
    distance_cutoff: float = Field(
        default=12.0,
        description="Ignore charged protein atoms beyond this distance (A) from any charged ligand atom",
    )
    power: float = Field(
        default=2.0,
        description="Distance exponent for coulomb-like decay (2.0 = implicit solvent screening)",
    )
    ref_smiles: Optional[str] = Field(
        default=None,
        description="Reference SMILES for rebuilding ligand bond graph when atom names "
        "don't follow CCD convention (diffusion model outputs)",
    )
    per_residue: bool = Field(
        default=False,
        description="Include per-residue score breakdown in output metrics",
    )

    def compute_metrics(self, loaded, *, context=None, index=0):
        try:
            metrics = _compute_pocket_coulomb(
                loaded.atom_array,
                pocket_radius=self.pocket_radius,
                distance_floor=self.distance_floor,
                distance_cutoff=self.distance_cutoff,
                power=self.power,
                ref_smiles=self.ref_smiles,
                per_residue=self.per_residue,
            )
            logger.info(
                "Computed pocket coulomb for %s: total_score=%.4f, "
                "n_charged_lig=%d, n_charged_res=%d",
                loaded.path,
                metrics["total_score"],
                metrics["n_charged_ligand_atoms"],
                metrics["n_charged_residues"],
            )
        except Exception as e:
            logger.error(
                "Failed to compute pocket coulomb for %s: %s",
                loaded.path,
                e,
                exc_info=True,
            )
            raise

        return metrics


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_ZERO_RESULT = {
    "total_score": 0.0,
    "n_charged_ligand_atoms": 0,
    "n_charged_residues": 0,
}


def _get_ligand_formal_charges(
    ligand_array, ref_smiles: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Extract formal charges from ligand atoms via RDKit.

    Returns (indices, charges) where indices are into heavy-atom subset
    of ligand_array, and charges are the corresponding formal charges.
    Only atoms with nonzero formal charge are returned.
    """
    from atomworks.io.tools.rdkit import atom_array_to_rdkit
    from rdkit import Chem

    from blocks.utils.geometry import get_heavy_mask
    from blocks.utils.ligand import _get_ref_smiles, _rebuild_mol_from_geometry

    heavy_mask = get_heavy_mask(ligand_array)
    heavy_array = ligand_array[heavy_mask]

    if len(heavy_array) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    mol = atom_array_to_rdkit(heavy_array, hydrogen_policy="remove")

    # Check for disconnected graph (same pattern as protonate_ligand)
    smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
    if "." in smi or mol.GetNumBonds() == 0:
        resolved_ref = _get_ref_smiles(str(heavy_array.res_name[0]), ref_smiles)
        if resolved_ref is None:
            logger.warning(
                "Ligand %s bond graph disconnected and no ref SMILES available; "
                "treating all ligand atoms as uncharged",
                heavy_array.res_name[0],
            )
            return np.array([], dtype=int), np.array([], dtype=float)
        logger.info(
            "Ligand %s bond graph disconnected; rebuilding from geometry + ref SMILES",
            heavy_array.res_name[0],
        )
        mol = _rebuild_mol_from_geometry(heavy_array, resolved_ref)

    indices = []
    charges = []
    for atom in mol.GetAtoms():
        fc = atom.GetFormalCharge()
        if fc != 0:
            indices.append(atom.GetIdx())
            charges.append(float(fc))

    return np.array(indices, dtype=int), np.array(charges, dtype=float)


def _compute_pocket_coulomb(
    atom_array,
    *,
    pocket_radius: float = 8.0,
    distance_floor: float = 1.5,
    distance_cutoff: float = 12.0,
    power: float = 2.0,
    ref_smiles: str | None = None,
    per_residue: bool = False,
) -> dict:
    """Compute pocket coulomb score for a protein-ligand structure.

    1. Detect pocket residues (protein CBs within pocket_radius of ligand)
    2. Find charged protein atoms in pocket via RESIDUE_CHARGE_MAP
    3. Get ligand formal charges via RDKit
    4. Compute pairwise Coulomb-like score
    """
    from scipy.spatial.distance import cdist

    from blocks.utils.geometry import get_cb_mask, get_heavy_mask, get_ligand_mask

    # 1. Find ligand atoms
    ligand_mask = get_ligand_mask(atom_array)
    if ligand_mask is None:
        return dict(_ZERO_RESULT)

    if not np.any(ligand_mask):
        return dict(_ZERO_RESULT)

    ligand_array = atom_array[ligand_mask]
    ligand_heavy_mask = get_heavy_mask(ligand_array)
    ligand_heavy_coords = ligand_array.coord[ligand_heavy_mask]

    if len(ligand_heavy_coords) == 0:
        return dict(_ZERO_RESULT)

    # 2. Pocket detection: find protein CBs within pocket_radius of any ligand atom
    protein_mask = ~ligand_mask
    cb_mask = protein_mask & get_cb_mask(atom_array)

    if not np.any(cb_mask):
        return dict(_ZERO_RESULT)

    cb_coords = atom_array.coord[cb_mask]
    cb_indices = np.where(cb_mask)[0]

    dists_cb_lig = cdist(cb_coords, ligand_heavy_coords)
    min_dists_to_lig = np.min(dists_cb_lig, axis=1)
    pocket_cb_local = min_dists_to_lig <= pocket_radius
    pocket_cb_indices = cb_indices[pocket_cb_local]

    if len(pocket_cb_indices) == 0:
        return dict(_ZERO_RESULT)

    # Expand pocket CBs to all atoms of those residues
    pocket_res_ids = set()
    for idx in pocket_cb_indices:
        pocket_res_ids.add((atom_array.chain_id[idx], atom_array.res_id[idx]))

    # 3. Find charged protein atoms in pocket
    prot_charged_coords = []
    prot_charged_charges = []
    prot_charged_residues = []  # (chain, resid, resname) for per-residue grouping

    for i in range(len(atom_array)):
        if ligand_mask[i]:
            continue
        key = (atom_array.res_name[i], atom_array.atom_name[i])
        if key not in RESIDUE_CHARGE_MAP:
            continue
        res_key = (atom_array.chain_id[i], int(atom_array.res_id[i]))
        if res_key not in pocket_res_ids:
            continue
        prot_charged_coords.append(atom_array.coord[i])
        prot_charged_charges.append(RESIDUE_CHARGE_MAP[key])
        prot_charged_residues.append(
            (atom_array.chain_id[i], int(atom_array.res_id[i]), atom_array.res_name[i])
        )

    if not prot_charged_coords:
        return dict(_ZERO_RESULT)

    prot_coords = np.array(prot_charged_coords)
    prot_charges = np.array(prot_charged_charges)

    # 4. Get ligand formal charges
    lig_indices, lig_charges = _get_ligand_formal_charges(ligand_array, ref_smiles)

    if len(lig_indices) == 0:
        # Count charged residues even if ligand has no formal charges
        unique_res = set(prot_charged_residues)
        return {
            "total_score": 0.0,
            "n_charged_ligand_atoms": 0,
            "n_charged_residues": len(unique_res),
        }

    # Coordinates of charged ligand atoms (index into heavy-atom subset)
    ligand_heavy = ligand_array[get_heavy_mask(ligand_array)]
    lig_charged_coords = ligand_heavy.coord[lig_indices]

    # 5. Vectorized Coulomb calculation
    dists = cdist(prot_coords, lig_charged_coords)
    dists = np.clip(dists, distance_floor, None)

    mask = dists <= distance_cutoff
    charge_products = prot_charges[:, None] * lig_charges[None, :]
    terms = np.where(mask, charge_products / (dists**power), 0.0)

    unique_res = set(prot_charged_residues)
    result = {
        "total_score": float(np.sum(terms)),
        "n_charged_ligand_atoms": len(lig_indices),
        "n_charged_residues": len(unique_res),
    }

    # 6. Per-residue breakdown
    if per_residue:
        res_contributions: dict[str, float] = {}
        for prot_idx, res_tuple in enumerate(prot_charged_residues):
            chain, resid, resname = res_tuple
            key = f"per_res.{chain}{resid}_{resname}"
            row_sum = float(np.sum(terms[prot_idx, :]))
            res_contributions[key] = res_contributions.get(key, 0.0) + row_sum
        result.update(res_contributions)

    return result
