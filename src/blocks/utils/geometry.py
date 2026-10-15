import itertools
import logging
import warnings

import biotite.structure as struc
import numpy as np
from biotite.structure import index_distance
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Centralized atom / ligand selection helpers (WS7-5)
#
# These consolidate duplicated inline masks and cdist calls that were scattered
# across the metric blocks (rasa, pocket_coulomb, secondary_structure, hbond,
# contact_ms, foldseek, ...). Each helper preserves the EXACT semantics of the
# inline code it replaces — in particular the FULL-ligand policy for
# contact_ms / hbond / aromatic / rasa is retained (no rigid-core restriction).
# ---------------------------------------------------------------------------

BACKBONE_ATOM_NAMES = ("N", "CA", "C", "O")


def get_ca_mask(atom_array: struc.AtomArray) -> np.ndarray:
    """Boolean mask of CA atoms."""
    return atom_array.atom_name == "CA"


def get_cb_mask(atom_array: struc.AtomArray) -> np.ndarray:
    """Boolean mask of CB atoms."""
    return atom_array.atom_name == "CB"


def get_backbone_mask(atom_array: struc.AtomArray) -> np.ndarray:
    """Boolean mask of protein backbone atoms (N, CA, C, O)."""
    return np.isin(atom_array.atom_name, list(BACKBONE_ATOM_NAMES))


def get_heavy_mask(atom_array: struc.AtomArray) -> np.ndarray:
    """Boolean mask of non-hydrogen (heavy) atoms."""
    return atom_array.element != "H"


def get_ligand_mask(atom_array: struc.AtomArray) -> np.ndarray | None:
    """Boolean mask of ligand atoms from explicit metadata or PDB HETATM.

    The mask spans the FULL ligand (no rigid-core restriction) to preserve the
    full-ligand scoring policy used by contact_ms / hbond / aromatic / rasa.
    PDB files do not carry the mmCIF entity metadata used to derive
    ``is_ligand``; their ``hetero`` annotation is the authoritative fallback.
    """
    if "is_ligand" in atom_array.get_annotation_categories():
        return np.asarray(atom_array.is_ligand, dtype=bool)
    if "hetero" in atom_array.get_annotation_categories():
        return np.asarray(atom_array.hetero, dtype=bool)
    return None


def get_chain_annotation(atom_array: struc.AtomArray) -> np.ndarray:
    """Return the chain annotation array, preferring ``chain_iid`` over
    ``chain_id`` (matches foldseek / hbonds chain-identifier preference)."""
    if "chain_iid" in atom_array.get_annotation_categories():
        return atom_array.chain_iid
    return atom_array.chain_id


def match_atoms_by_identifier(
    src_atoms: struc.AtomArray,
    des_atoms: struc.AtomArray,
    src_mask: np.ndarray,
    des_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Create paired masks for atoms present in BOTH structures.

    When structures have different atoms (e.g., PTMs replacing residues or
    MPNN changing sequences), finds atoms with matching (chain_id, res_id,
    atom_name) in both and returns masks for only those common atoms.

    Falls back to (res_name, atom_name) matching when no exact identifier
    matches — this handles AF3-style residue renumbering where chain/res_id
    change but the atoms themselves are unchanged. This fallback semantics is
    load-bearing (see rmsd.py) and is preserved exactly.

    Returns:
        Tuple of (paired_src_mask, paired_des_mask) with matching atoms,
        ordered to maintain the source structure ordering.
    """
    src_indices = np.where(src_mask)[0]
    des_indices = np.where(des_mask)[0]

    src_ids = {
        (src_atoms.chain_id[i], src_atoms.res_id[i], src_atoms.atom_name[i]): i
        for i in src_indices
    }
    des_ids = {
        (des_atoms.chain_id[i], des_atoms.res_id[i], des_atoms.atom_name[i]): i
        for i in des_indices
    }

    common_ids = set(src_ids.keys()) & set(des_ids.keys())
    if len(common_ids) == 0:
        # Fallback: match by (res_name, atom_name) only — handles AF3-style
        # residue renumbering where chain/res_id change but atoms are the same.
        msg = (
            "No common atoms with exact (chain_id, res_id, atom_name) match. "
            "Falling back to (res_name, atom_name) matching — this typically means "
            "the prediction tool renumbered residues (e.g. AF3 ligand renumbering)."
        )
        logger.warning(msg)
        warnings.warn(msg, stacklevel=2)
        src_ids = {
            (src_atoms.res_name[i], src_atoms.atom_name[i]): i for i in src_indices
        }
        des_ids = {
            (des_atoms.res_name[i], des_atoms.atom_name[i]): i for i in des_indices
        }
        common_ids = set(src_ids.keys()) & set(des_ids.keys())
        if len(common_ids) == 0:
            raise ValueError("No common atoms found between source and design")

    # Maintain source structure ordering
    common_ids_sorted = sorted(common_ids, key=lambda x: src_ids[x])

    paired_src_mask = np.zeros(len(src_atoms), dtype=bool)
    paired_des_mask = np.zeros(len(des_atoms), dtype=bool)

    for atom_id in common_ids_sorted:
        paired_src_mask[src_ids[atom_id]] = True
        paired_des_mask[des_ids[atom_id]] = True

    return paired_src_mask, paired_des_mask


def get_n_closest_residues_to_substructure(
    atom_array: struc.AtomArray, mask_a: np.ndarray, mask_b: np.ndarray, n: int
) -> np.ndarray:
    """
    Returns a mask corresponding to the N residues in mask_a closest to the substructure given by mask_b,
    using the minimum CA-CA distance to determine proximity.

    Parameters:
    -----------
    atom_array : AtomArray
        The atom array containing the protein structure
    mask_a : np.ndarray
        Boolean mask for chain A (shape: (n_atoms,))
    mask_b : np.ndarray
        Boolean mask for chain B (shape: (n_atoms,))
    n : int
        Number of closest residues to return

    Returns:
    --------
    np.ndarray
        Boolean mask for the N residues closest to chain B (shape: (n_atoms,))
    """
    # Get CA atoms for chain A and chain B
    ca_mask_a = mask_a & (atom_array.atom_name == "CA")
    ca_mask_b = mask_b & (atom_array.atom_name == "CA")

    if not np.any(ca_mask_a):
        raise ValueError("No CA atoms found in chain A")
    if not np.any(ca_mask_b):
        raise ValueError("No CA atoms found in chain B")

    # Get CA coordinates
    ca_coords_a = atom_array.coord[ca_mask_a]  # Shape: (n_ca_a, 3)
    ca_coords_b = atom_array.coord[ca_mask_b]  # Shape: (n_ca_b, 3)

    # Calculate pairwise distances between all CA atoms in chain A and chain B
    distances = cdist(ca_coords_a, ca_coords_b)  # Shape: (n_ca_a, n_ca_b)

    # For each residue in chain A, find the minimum distance to any CA in chain B
    min_distances = np.min(distances, axis=1)  # Shape: (n_ca_a,)

    # Get the indices of the N closest residues
    closest_indices = np.argsort(min_distances)[:n]

    # Get the original atom indices for the CA atoms of the closest residues
    ca_indices_a = np.where(ca_mask_a)[0]
    closest_ca_indices = ca_indices_a[closest_indices]

    # Create a mask for all atoms belonging to the closest residues
    result_mask = np.zeros(len(atom_array), dtype=bool)

    if "chain_iid" in atom_array.get_annotation_categories():
        has_chain_iid = True
    else:
        logger.warning(
            "No chain_iid found in atom_array. Can lead to unexpected behavior if multiple symmetry transformations are present."
        )
        has_chain_iid = False

    # For each closest CA atom, find all atoms from the same residue
    for ca_idx in closest_ca_indices:
        # Get the residue identifier for this CA atom
        res_id = atom_array.res_id[ca_idx]
        if has_chain_iid:
            chain_identifier = atom_array.chain_iid[ca_idx]
            residue_mask = (atom_array.res_id == res_id) & (
                atom_array.chain_iid == chain_identifier
            )
        else:
            chain_identifier = atom_array.chain_id[ca_idx]
            residue_mask = (atom_array.res_id == res_id) & (
                atom_array.chain_id == chain_identifier
            )

        # Add to result mask
        result_mask |= residue_mask

    return result_mask


def target_aligned_nearest_neighbors_rmsd(struct1, struct2, n_neighbors):
    """
    Align structures on the target (motif region), then compute the RMSD over the n_neighbors closest binder residues
    to the target. Distance is determined by CA distance, and RMSD is computed over backbone N, CA, Catoms.

    Args:
        struct1: atomarray
        struct2: atomarray
        n_neighbors (int): number of residues on the binder for which to compute the RMSD

    Returns
    -------
    float
        RMSD over n_neighbors closest binder residues
    """
    # Get motif atoms
    motif_atoms_1 = struct1[struct1.is_motif_atom]
    motif_atoms_2 = struct2[struct2.is_motif_atom]

    fitted, sup = struc.superimpose(motif_atoms_1, motif_atoms_2)
    struct2_aligned = sup.apply(struct2)

    binder_interface_mask_1 = get_n_closest_residues_to_substructure(
        struct1,
        mask_a=~struct1.is_motif_atom,
        mask_b=struct1.is_motif_atom,
        n=n_neighbors,
    )
    binder_interface_mask_2 = get_n_closest_residues_to_substructure(
        struct2_aligned,
        mask_a=~struct2_aligned.is_motif_atom,
        mask_b=struct2_aligned.is_motif_atom,
        n=n_neighbors,
    )

    binder_interface_1 = struct1[binder_interface_mask_1]
    binder_interface_2 = struct2_aligned[binder_interface_mask_2]

    n_ca_c_1 = binder_interface_1[
        np.isin(binder_interface_1.atom_name, ["CA", "N", "C"])
    ]
    n_ca_c_2 = binder_interface_2[
        np.isin(binder_interface_2.atom_name, ["CA", "N", "C"])
    ]

    if len(n_ca_c_1) != len(n_ca_c_2):
        raise ValueError("Number of atoms must be equal")

    rmsd = float(struc.rmsd(n_ca_c_1, n_ca_c_2))
    return rmsd


def get_dna_aligned_ca_rmsd(struct1, struct2):
    """
    Align structures by DNA phosphates, calculate protein RMSD with and without terminal loops.

    Args:
        struct1: atomarray
        struct2: atomarray

    Returns
    -------
    tuple(float, float)
        RMSD with terminal loops trimmed, and RMSD without trimming
    """
    p_mask1 = (struct1.atom_name == "P") & (struct1.element == "P")
    p_mask2 = (struct2.atom_name == "P") & (struct2.element == "P")

    p_atoms1 = struct1[p_mask1]
    p_atoms2 = struct2[p_mask2]

    if len(p_atoms1) != len(p_atoms2):
        p_atoms1, p_atoms2 = match_phosphate_atoms(p_atoms1, p_atoms2)

    fitted, sup = struc.superimpose(p_atoms1, p_atoms2)
    struct2_aligned = sup.apply(struct2)

    # Get protein-only structures
    protein1 = get_protein_atoms(struct1)
    protein2 = get_protein_atoms(struct2_aligned)

    # Get all CA atoms for untrimmed RMSD
    ca_mask1 = protein1.atom_name == "CA"
    ca_mask2 = protein2.atom_name == "CA"

    ca_all1 = protein1[ca_mask1]
    ca_all2 = protein2[ca_mask2]

    if len(ca_all1) != len(ca_all2):
        raise ValueError("Number of CA atoms must be equal")

    # Calculate untrimmed RMSD
    untrimmed_rmsd = struc.rmsd(ca_all1, ca_all2)

    # Get trimmed CA atoms and calculate trimmed RMSD
    ca_struct1, ca_struct2 = get_trimmed_ca_atoms(protein1, protein2)
    trimmed_rmsd = struc.rmsd(ca_struct1, ca_struct2)

    return trimmed_rmsd, untrimmed_rmsd


def find_terminus_trim(sse, from_start=True):
    """
    Find how many residues to trim from one terminus.

    Parameters
    ----------
    sse : ndarray
        Array of secondary structure annotations
    from_start : bool
        If True, check from N-terminus, if False, check from C-terminus

    Returns
    -------
    int
        Number of residues to trim from this terminus
    """
    if from_start:
        for i, ss in enumerate(sse):
            if ss in ["a", "b"]:
                return i
        return len(sse)  # All loops
    else:
        for i in range(len(sse) - 1, -1, -1):
            if sse[i] in ["a", "b"]:
                return len(sse) - (i + 1)
        return len(sse)  # All loops


def find_structured_termini(sse1, sse2):
    """
    Find indices where terminal loops end/begin, using the more conservative trim
    between the two structures.

    Parameters
    ----------
    sse1, sse2 : ndarray
        Arrays of secondary structure annotations for both structures

    Returns
    -------
    tuple(int, int, int, int)
        Start and end indices for both structures
    """
    # Find how much each structure would trim
    n_trim1 = find_terminus_trim(sse1, from_start=True)
    n_trim2 = find_terminus_trim(sse2, from_start=True)
    c_trim1 = find_terminus_trim(sse1, from_start=False)
    c_trim2 = find_terminus_trim(sse2, from_start=False)

    # Use the smaller trim for each terminus
    n_trim = min(n_trim1, n_trim2)
    c_trim = min(c_trim1, c_trim2)

    # Calculate end indices
    end1 = len(sse1) - c_trim
    end2 = len(sse2) - c_trim

    return n_trim, end1, n_trim, end2


def get_trimmed_ca_atoms(protein_struct1, protein_struct2):
    """
    Get CA atoms excluding terminal loops, using consistent trimming.
    """
    # Get SSE annotations
    sse1 = struc.annotate_sse(protein_struct1)
    sse2 = struc.annotate_sse(protein_struct2)

    # Get unique residue IDs and their corresponding SSE annotations
    res_ids1 = np.unique(protein_struct1.res_id)
    res_ids2 = np.unique(protein_struct2.res_id)

    # Find trim points
    start1, end1, start2, end2 = find_structured_termini(sse1, sse2)

    # Get the residue IDs we want to keep
    keep_res1 = res_ids1[start1:end1]
    keep_res2 = res_ids2[start2:end2]

    # Create masks for the atoms we want to keep
    ca_mask1 = protein_struct1.atom_name == "CA"
    ca_mask2 = protein_struct2.atom_name == "CA"

    res_mask1 = np.isin(protein_struct1.res_id, keep_res1)
    res_mask2 = np.isin(protein_struct2.res_id, keep_res2)

    final1 = protein_struct1[ca_mask1 & res_mask1]
    final2 = protein_struct2[ca_mask2 & res_mask2]

    return final1, final2


def get_protein_atoms(structure):
    """
    Extract protein atoms from structure.

    Parameters
    ----------
    structure : AtomArray
        Input structure

    Returns
    -------
    AtomArray
        Protein-only atoms
    """
    nucleic_acids = {"DA", "DT", "DG", "DC", "A", "U", "G", "C"}
    protein_mask = ~np.isin(structure.res_name, nucleic_acids)
    return structure[protein_mask]


def match_phosphate_atoms(src_phos_atoms, des_phos_atoms):
    """
    Identify which phosphate atoms to keep for alignment when the two structures
    have different numbers of phosphates.
    (occurs when aligning af3 prediction against design models made with crystal structure dna)

    Args:
        src_phos_atoms: AtomArray containing phosphate atoms from the source structure
        des_phos_atoms: AtomArray containing phosphate atoms from the destination structure

    Returns:
        (phos_src, phos_des): Tuple of AtomArrays containing the matched phosphate atoms
    """
    # Extract residue indices and names
    src_res_indices = src_phos_atoms.res_id
    src_res_names = src_phos_atoms.res_name
    des_res_indices = des_phos_atoms.res_id
    des_res_names = des_phos_atoms.res_name

    # Count unique residues in each structure to handle potential duplicates
    src_unique_res = list(zip(src_res_indices, src_res_names))
    des_unique_res = list(zip(des_res_indices, des_res_names))

    # Get the sequence strings for comparison
    src_seq = "".join([res[1] for res in src_unique_res])
    des_seq = "".join([res[1] for res in des_unique_res])

    # Determine which sequence is longer
    if len(src_unique_res) == len(des_unique_res):
        # Same length, no need for alignment
        return src_phos_atoms, des_phos_atoms

    if len(src_unique_res) > len(des_unique_res):
        longer_atoms = src_phos_atoms
        shorter_atoms = des_phos_atoms
        longer_res = src_unique_res
        shorter_res = des_unique_res
        shorter_seq = des_seq
    else:
        longer_atoms = des_phos_atoms
        shorter_atoms = src_phos_atoms
        longer_res = des_unique_res
        shorter_res = src_unique_res
        shorter_seq = src_seq

    # Calculate how many residues to remove
    diff_length = len(longer_res) - len(shorter_res)

    # Try all possible combinations of residues to remove (up to 4)
    best_match = None
    best_score = -1

    for indices_to_remove in itertools.combinations(
        range(len(longer_res)), diff_length
    ):
        # Create a candidate sequence by removing specific residues
        candidate_res = [
            longer_res[i] for i in range(len(longer_res)) if i not in indices_to_remove
        ]
        candidate_seq = "".join([res[1] for res in candidate_res])

        # Count matching residues
        match_score = sum(a == b for a, b in zip(candidate_seq, shorter_seq))

        # If we found a perfect match, stop searching
        if match_score == len(shorter_seq):
            best_match = indices_to_remove
            break

        # Otherwise, keep track of the best partial match
        if match_score > best_score:
            best_score = match_score
            best_match = indices_to_remove

    # Create masks for the longer sequence to keep only matching residues
    keep_mask = [True] * len(longer_atoms)

    # Map from unique residues back to atoms
    res_to_indices = {}
    for i, (res_id, res_name) in enumerate(
        zip(longer_atoms.res_id, longer_atoms.res_name)
    ):
        key = (res_id, res_name)
        if key not in res_to_indices:
            res_to_indices[key] = []
        res_to_indices[key].append(i)

    # Mark atoms to remove
    for idx in best_match:
        res_key = longer_res[idx]
        for atom_idx in res_to_indices.get(res_key, []):
            keep_mask[atom_idx] = False

    # Apply the mask to get the matched longer sequence
    matched_longer = longer_atoms[keep_mask]

    # Return the results in the original order
    if len(src_unique_res) > len(des_unique_res):
        return matched_longer, shorter_atoms
    else:
        return shorter_atoms, matched_longer


def _find_nearest_neighbors(
    atom_array: struc.AtomArray, ligand_indices: list, n_angstroms: int
) -> np.ndarray:
    """
    Find the nearest neighbors of an atom in an atom array within n angstroms based on
    cb distance
    Args:
        atom_array: AtomArray
        ligand_indices: list of indices of the ligand atoms
        n_angstroms: int
    Returns:
        Indices into `atom_array` (the array as passed in) of the CB atoms
        whose residues sit within the cutoff of any ligand atom.
    """
    non_h_mask = atom_array.element != "H"
    filtered = atom_array[non_h_mask]
    # Remap ligand indices from original array to filtered (H-removed) array
    original_to_filtered = np.cumsum(non_h_mask) - 1
    ligand_indices = [original_to_filtered[i] for i in ligand_indices if non_h_mask[i]]
    # Indices are computed against the filtered array but callers index the
    # array they passed in, so keep the way back. Without it, any structure
    # carrying hydrogens silently reports the wrong residues rather than
    # failing.
    filtered_to_original = np.where(non_h_mask)[0]
    # get all the Cb atoms since it determines if the sidechains are facing the ligand
    cb_indices = np.where(filtered.atom_name == "CB")[0]
    # A structure with no CB atoms at all — backbone-only or all-glycine — is a
    # legitimate zero-contact result. Left to itself the empty comprehension
    # below builds a (0,) array where index_distance demands (N, 2), turning
    # that into a ValueError mid-campaign.
    if len(cb_indices) == 0 or len(ligand_indices) == 0:
        return []
    closest_residues_indices = set()
    for ligand_i in ligand_indices:
        combo = np.column_stack(
            (np.full(len(cb_indices), ligand_i), cb_indices)
        )
        # get the distance between the combo
        dists = index_distance(filtered.coord, combo)
        for i, dist in enumerate(dists):
            if dist <= n_angstroms:
                closest_residues_indices.add(int(filtered_to_original[cb_indices[i]]))
    return sorted(closest_residues_indices)
