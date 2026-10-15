"""
Module for calculating dihedral angles and rotamer probabilities.
Uses backbone-dependent Dunbrack library for calculation.

Author(s): Jie Chen, Jasper Butcher
"""

import functools
import logging

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import logsumexp

from chaperone import get_setting_field

logger = logging.getLogger(__name__)

# From openfold: residue_constants.py
# Format: The list for each AA type contains chi1, chi2, chi3, chi4 in
# this order (or a relevant subset from chi1 onwards). ALA and GLY don't have
# chi angles so their chi angle lists are empty.
chi_angles_atoms = {
    "ALA": [],
    # Chi5 in arginine is always 0 +- 5 degrees, so ignore it.
    "ARG": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "NE"],
        ["CG", "CD", "NE", "CZ"],
    ],
    "ASN": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "ASP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "CYS": [["N", "CA", "CB", "SG"]],
    "GLN": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "OE1"],
    ],
    "GLU": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "OE1"],
    ],
    "GLY": [],
    "HIS": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "ND1"]],
    "ILE": [["N", "CA", "CB", "CG1"], ["CA", "CB", "CG1", "CD1"]],
    "LEU": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "LYS": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "CE"],
        ["CG", "CD", "CE", "NZ"],
    ],
    "MET": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "SD"],
        ["CB", "CG", "SD", "CE"],
    ],
    "PHE": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "PRO": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"]],
    "SER": [["N", "CA", "CB", "OG"]],
    "THR": [["N", "CA", "CB", "OG1"]],
    "TRP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "TYR": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "VAL": [["N", "CA", "CB", "CG1"]],
}

symmetric_chi_angles = {
    "ASP": [2],
    "ASN": [2],
    "PHE": [2],
    "TYR": [2],
    "HIS": [2],
    "TRP": [2],
    "GLU": [3],
    "GLN": [3],
    "ARG": [4],
    "LYS": [4],
    "MET": [3],
}

# Thresholds for normalizing rotamer probabilities to scores, where >1
ROTAMER_PROB_SCORE_NORMALIZATIONS = {
    "THR": -4.964161092981123,
    "SER": -4.976023922825927,
    "HIS": -6.813713459950055,
    "ARG": -12.52241880442645,
    "GLN": -9.175096415074679,
    "ASN": -5.571420605307308,
    "VAL": -13.673717051354595,
    "MET": -11.94377910612243,
    "ILE": -8.342745669087149,
    "LEU": -12.530387717684619,
    "GLU": -28.33234053897388,
    "ASP": -34.792404890156256,
    "TYR": -5.605853226806332,
    "LYS": -15.260760556925096,
    "TRP": -5.7102165539264105,
    "PHE": -6.43329732594977,
    "CYS": -2.9821400946080394,
}
RESIDUES_WITHOUT_ROTAMERS = ["GLY", "ALA", "PRO"]


@functools.lru_cache(maxsize=20)
def cached_library_file(res_name: str):
    """Cache and load the Dunbrack rotamer library file for a given residue name."""
    dunbrack_path = get_setting_field("DUNBRACK_LIBRARY_COPY")
    assert Path(dunbrack_path).exists(), (
        f"DUNBRACK_LIBRARY_COPY path {dunbrack_path} does not exist."
    )
    file_path = f"{dunbrack_path}/{res_name.lower()}.bbdep.rotamers.lib"

    if not Path(file_path).exists():
        raise FileNotFoundError(
            f"Rotamer library file for {res_name} not found. Expected at {file_path}"
        )

    df = pd.read_csv(file_path, skiprows=38, header=None, sep=r"\s+")
    df.columns = [
        "T",
        "Phi",
        "Psi",
        "Count",
        "r1",
        "r2",
        "r3",
        "r4",
        "Probabil",
        "chi1Val",
        "chi2Val",
        "chi3Val",
        "chi4Val",
        "chi1Sig",
        "chi2Sig",
        "chi3Sig",
        "chi4Sig",
    ]

    logger.info("Loaded %s rotamer library into cache", res_name)
    return df


def subset_library(rotamer_df, phi, psi):
    """Subset the rotamer library based on phi and psi angles."""
    phi = int(np.round(phi / 10.0) * 10)
    psi = int(np.round(psi / 10.0) * 10)
    rotamer_df = rotamer_df[(rotamer_df["Phi"] == phi) & (rotamer_df["Psi"] == psi)]
    if rotamer_df.empty:
        logger.warning(
            "Backbone not found in rotamer library for phi=%d, psi=%d", phi, psi
        )
        return None
    return rotamer_df


def get_rotamer_probability(
    residue_xyz: np.ndarray,
    atom_names: np.ndarray,
    res_name: str,
    phi: float,
    psi: float,
    return_best_rotamer: bool = False,
    return_score: bool = False,
) -> float:
    """
    Get the probability of a rotamer given the residue coordinates, residue name,
    phi, and psi angles using the Dunbrack backbone-dependent library.

    Parameters:
        residue_xyz: Coordinates of the residue atoms [N, 3].
        atom_names: Names corresponding to each coordinate [N,].
        res_name: Residue name (uppercased).
        phi: Phi dihedral angle in degrees.
        psi: Psi dihedral angle in degrees.
        return_best_rotamer: If True, also return the best matching rotamer entry.
        return_score: If True, also return a normalized score.

    Returns:
        Estimated (Gaussian-mixture model) probability of the rotamer,
        optionally with score or best rotamer.
    """
    if res_name in RESIDUES_WITHOUT_ROTAMERS:
        return None

    rotamer_df = cached_library_file(res_name)

    # Narrow down the rotamers based on phi and psi angles
    rotamer_df = subset_library(rotamer_df, phi, psi)
    if rotamer_df is None:
        return None

    # Get each atom position of the residue and store it with corresponding chi angle
    atom_info = {
        atom_name.strip(): coord
        for atom_name, coord in zip(atom_names, residue_xyz)
        if atom_name
    }
    chi_angles = chi_angles_atoms[res_name]
    dihedral_angles = {}
    for chi_num, chi in enumerate(chi_angles):
        dihedral_coords = []
        for atom in chi:
            if atom in atom_info:
                dihedral_coords.append(atom_info[atom])
            else:
                logger.warning("Atom %s not found in residue %s", atom, res_name)
                return None

        dihedral_coords = np.array(dihedral_coords, dtype=float)
        angle = dihedral_calculation(dihedral_coords)

        # Deal with residues with symmetric chi angles
        if (chi_num + 1) in symmetric_chi_angles.get(res_name, []):
            angle = abs(angle) % 180
        else:
            angle = angle % 360
        dihedral_angles[chi_num] = angle

    total_log_prob = []
    for _, entry in rotamer_df.iterrows():
        log_prob = np.log(entry["Probabil"]) if entry["Probabil"] != 0 else 0
        for i in range(len(chi_angles)):
            observed_angle = dihedral_angles[i]
            mean_angle = entry[f"chi{i + 1}Val"]
            sigma = entry[f"chi{i + 1}Sig"]
            delta_angle = (observed_angle - mean_angle + 180) % 360 - 180
            log_prob_chi = -0.5 * (delta_angle / sigma) ** 2 - np.log(
                np.sqrt(2 * np.pi) * sigma
            )
            log_prob += log_prob_chi
        total_log_prob.append(log_prob)

    total_log_prob = np.array(total_log_prob)
    log_prob = logsumexp(total_log_prob)
    probability = np.exp(log_prob)

    if return_best_rotamer:
        best_rotamer_idx = np.argmax(total_log_prob)
        best_rotamer = rotamer_df.iloc[best_rotamer_idx]
        return probability, best_rotamer
    if return_score:
        score = float(
            np.log10(np.array(probability))
            / ROTAMER_PROB_SCORE_NORMALIZATIONS[res_name]
        )
        return probability, score
    return probability


def dihedral_calculation(p: np.ndarray) -> float:
    """
    Calculate the dihedral angle from four points.

    Parameters:
        p: Array of shape (4, 3) containing the coordinates of four points.

    Returns:
        Dihedral angle in degrees.
    """
    p = np.asarray(p, dtype=float)

    b0 = -1.0 * (p[1] - p[0])
    b1 = p[2] - p[1]
    b2 = p[3] - p[2]

    b1 /= np.linalg.norm(b1)

    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.degrees(np.arctan2(y, x))


def calculate_phi_psi(prev_residue_xyz, curr_residue_xyz, next_residue_xyz):
    """
    Calculate the phi and psi angles for a residue given neighboring residue coordinates.

    Parameters:
        prev_residue_xyz: Coordinates of the previous residue atoms.
        curr_residue_xyz: Coordinates of the current residue atoms.
        next_residue_xyz: Coordinates of the next residue atoms.

    Returns:
        Tuple of (phi, psi) angles in degrees.
    """
    C_prev = prev_residue_xyz[2]
    N = curr_residue_xyz[0]
    CA = curr_residue_xyz[1]
    C = curr_residue_xyz[2]
    N_next = next_residue_xyz[0]

    phi_coords = np.array([C_prev, N, CA, C], dtype=float)
    psi_coords = np.array([N, CA, C, N_next], dtype=float)

    phi = dihedral_calculation(phi_coords)
    psi = dihedral_calculation(psi_coords)

    return phi, psi
