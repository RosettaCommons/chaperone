"""Utility functions for RMSD metrics using numpy."""

import copy
import logging
import numpy as np
from atomworks.constants import STANDARD_AA, STANDARD_DNA, STANDARD_RNA
from atomworks.ml.transforms.atomize import AtomizeByCCDName
from atomworks.ml.transforms.symmetry import FindAutomorphismsWithNetworkX

logger = logging.getLogger(__name__)


def weighted_rigid_align(
    X_L,  # [B, L, 3]
    X_gt_L,  # [B, L, 3]
    X_exists_L=None,  # [L]
    w_L=None,  # [B, L]
    return_alignment_function=False,  # Func ..., N, 3 -> ..., N, 3 for alignment
):
    """
    Weighted rigid body alignment of X_gt_L onto X_L with weights w_L.
    Allows for "moving target" ground truth that is se3 invariant.
    Following algorithm 28 in AF3 paper.

    Args:
        X_L: [B, L, 3]   # static
        X_gt_L: [B, L, 3]  # mobile
        X_exists_L: [L]  # bool mask for both src and tgt
        w_L: [B, L]     # per-atom weights
        return_alignment_function: bool

    Returns:
      X_align_L: [B, L, 3] or alignment function
    """
    if X_exists_L is None:
        X_exists_L = np.ones((X_L.shape[-2]), dtype=bool)
    if w_L is None:
        w_L = np.ones_like(X_L[..., 0])

    assert X_L.shape == X_gt_L.shape, f"{X_L.shape} != {X_gt_L.shape}"
    assert X_L.shape[:-1] == w_L.shape, f"{X_L.shape[:-1]} != {w_L.shape}"

    X_resolved = X_L[:, X_exists_L]
    X_gt_resolved = X_gt_L[:, X_exists_L]
    w_resolved = w_L[:, X_exists_L]

    # Compute weighted centroids
    u_X = np.sum(X_resolved * w_resolved[..., None], axis=-2) / np.sum(
        w_resolved, axis=-1, keepdims=True
    )
    u_X_gt = np.sum(X_gt_resolved * w_resolved[..., None], axis=-2) / np.sum(
        w_resolved, axis=-1, keepdims=True
    )

    # Center coordinates
    X_resolved = X_resolved - u_X[:, None, :]
    X_gt_resolved = X_gt_resolved - u_X_gt[:, None, :]

    # Computation of the covariance matrix
    C = np.einsum("bji,bjk->bik", w_resolved[..., None] * X_gt_resolved, X_resolved)

    # SVD decomposition
    U, S, Vt = np.linalg.svd(C)

    # Compute rotation matrix
    R = U @ Vt
    B, _, _ = X_L.shape
    F = np.tile(np.eye(3, 3)[None], (B, 1, 1))

    F[..., -1, -1] = np.sign(np.linalg.det(R))
    R = U @ F @ Vt

    if return_alignment_function:
        # Return a function that applies the alignment
        def align_func(X):
            return (X - u_X_gt.squeeze()) @ R.squeeze() + u_X.squeeze()

        return align_func
    else:
        X_gt_L = X_gt_L - u_X_gt[:, None, :]
        X_align_L = X_gt_L @ R + u_X[:, None, :]
        return X_align_L


def get_rmsd(xyz1, xyz2, eps=1e-6):
    """
    Calculate RMSD between two coordinate arrays.

    Args:
        xyz1: [..., L, 3] coordinates
        xyz2: [..., L, 3] coordinates
        eps: small value to avoid division by zero

    Returns:
        RMSD value
    """
    L = xyz1.shape[-2]
    rmsd = np.sqrt(np.sum((xyz2 - xyz1) * (xyz2 - xyz1), axis=(-1, -2)) / (L + eps))
    return rmsd


def greedy_resolve_symmetries(xyz1, xyz2, a1):
    """
    Resolve symmetries between two coordinate sets using greedy automorphism matching.

    Args:
        xyz1: [L, 3] coordinates to swap
        xyz2: [L, 3] reference coordinates
        a1: list of automorphism arrays

    Returns:
        Symmetry-resolved xyz1 coordinates
    """
    xyz1 = np.asarray(xyz1, dtype=np.float32)
    xyz2 = np.asarray(xyz2, dtype=np.float32)

    for a_i in a1:
        if a_i.shape[0] == 1:
            continue
        a_i = np.asarray(a_i)
        xyz1 = _get_best(x_to_swap=xyz1, x_ref=xyz2, a_i=a_i)

    return xyz1


def _get_best(x_to_swap, x_ref, a_i):
    """
    Swaps coordinates in native to match prediction according to the allowed automorphism in a_i.

    Args:
        x_to_swap: coordinates to swap (L, 3)
        x_ref: reference coordinates (L, 3)
        a_i: shape (n_automorphisms, n_coords) indices in L

    Returns:
        Best-matched coordinates

    Adapted from src.modelhub.loss.af3_losses.py
    """
    assert x_to_swap.shape == x_ref.shape, f"{x_to_swap.shape} != {x_ref.shape}"

    # Create a mask for a_i
    mask = np.zeros(x_ref.shape[-2], dtype=bool)
    mask[a_i[0]] = True

    x_swapped = x_to_swap.copy()
    bestj = 0
    bestrmsd = None

    for j in range(a_i.shape[0]):
        x_swapped[a_i[0]] = x_to_swap[a_i[j]]

        # Compute distance for swapped coordinates
        d_swap = np.linalg.norm(
            x_swapped[..., mask, :] - x_ref[..., mask, :], axis=-1
        ).mean()

        if j == 0:
            # Initialize values
            bestj = 0
            bestrmsd = d_swap
        elif bestrmsd > d_swap:
            # Update bestj and bestrmsd if the new swap is better
            bestj = j
            bestrmsd = d_swap

    # Swap using best automorphism
    x_to_swap[a_i[0]] = x_to_swap[a_i[bestj]]

    return x_to_swap


def resolve_subset_symmetries(
    xyz_src_subset, xyz_des_subset, mask, src_atoms, automorphisms
):
    """
    Resolve symmetries for a subset of atoms.

    Args:
        xyz_src_subset: Source coordinates subset (numpy array)
        xyz_des_subset: Design coordinates subset (numpy array)
        mask: Boolean mask indicating which atoms are in subset
        src_atoms: Full source atom array
        automorphisms: Full automorphism list

    Returns:
        Symmetry-resolved source coordinates (numpy array)
    """
    try:
        # Map global atom indices to subset indices
        keep_idx = np.where(mask)[0]
        orig2new = np.full(src_atoms.coord.shape[0], -1, dtype=int)
        orig2new[keep_idx] = np.arange(len(keep_idx))

        # Subset automorphisms to only those within the selection
        automorphisms_subset = [
            orig2new[x] for x in automorphisms if np.all(orig2new[x] >= 0)
        ]

        if automorphisms_subset:
            xyz_src_subset = greedy_resolve_symmetries(
                xyz_src_subset, xyz_des_subset, automorphisms_subset
            )
    except Exception as e:
        logger.debug(f"Failed to resolve subset symmetries: {e}")

    return xyz_src_subset


def find_automorphisms(atom_array):
    """
    Find automorphisms (symmetries) in atom array.

    Args:
        atom_array: Biotite AtomArray structure

    Returns:
        List of automorphism arrays or None
    """
    try:
        temp_array = copy.deepcopy(atom_array)
        atomizer = AtomizeByCCDName(
            atomize_by_default=True,
            res_names_to_ignore=STANDARD_AA + STANDARD_RNA + STANDARD_DNA,
            move_atomized_part_to_end=False,
            validate_atomize=False,
        )
        data = atomizer({"atom_array": temp_array})

        # Find automorphisms
        automorph_finder = FindAutomorphismsWithNetworkX()
        result = automorph_finder(data)

        return result.get("automorphisms")

    except Exception as e:
        logger.warning(
            f"Failed to find automorphisms, skipping symmetry resolution: {e}"
        )
        return None
