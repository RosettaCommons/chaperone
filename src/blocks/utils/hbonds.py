"""
Hydrogen bond calculation utilities using HBPLUS.

This module provides functionality to calculate hydrogen bonds in protein structures
using the HBPLUS program.
"""

import os
import string
import subprocess
import tempfile

import numpy as np
from biotite.structure import AtomArray, filter_amino_acids
from biotite.structure.io.pdb import PDBFile


def _uniquify_ligand_atom_names(atom_array: AtomArray, aa_mask: np.ndarray) -> None:
    """Rename duplicate atom names within non-protein residues in-place.

    For each (chain_id, res_id) group that is NOT an amino acid, if any atom
    names are duplicated, all atoms in that residue are reassigned using
    element-based sequential numbering (e.g. N1, N2, C1, C2, ...).

    PDB atom names are max 4 characters.  The scheme ``<element><counter>``
    always fits because element symbols are 1-2 chars and the counter will
    never exceed 99 for realistic ligands.
    """
    from collections import Counter

    chain_ids = atom_array.chain_id
    res_ids = atom_array.res_id
    atom_names = atom_array.atom_name
    elements = atom_array.element

    # Build set of (chain, res_id) pairs that are non-protein
    non_aa_indices = np.where(~aa_mask)[0]
    if len(non_aa_indices) == 0:
        return

    # Group non-protein atom indices by (chain_id, res_id)
    residue_groups: dict[tuple[str, int], list[int]] = {}
    for idx in non_aa_indices:
        key = (chain_ids[idx], int(res_ids[idx]))
        residue_groups.setdefault(key, []).append(idx)

    for _key, indices in residue_groups.items():
        names_in_res = [atom_names[i] for i in indices]
        counts = Counter(names_in_res)
        if all(v == 1 for v in counts.values()):
            continue  # already unique — skip

        # Reassign: element-based sequential numbering
        elem_counter: dict[str, int] = {}
        for i in indices:
            elem = elements[i]
            elem_counter[elem] = elem_counter.get(elem, 0) + 1
            new_name = f"{elem}{elem_counter[elem]}"
            # PDB atom name field is 4 chars; truncate if needed (unlikely)
            atom_names[i] = new_name[:4]

    atom_array.atom_name = atom_names


def save_atomarray_to_pdb(
    atom_array: AtomArray, output_path: str
) -> tuple[AtomArray, np.ndarray, dict[str, str]]:
    """
    Save an AtomArray to a PDB file, handling chain ID conversions.

    Args:
        atom_array: AtomArray to save.
        output_path: Path to write the PDB file.

    Returns:
        Tuple of (modified atom_array, nan_mask, inverted_chain_map).
    """

    def _handle_nan_coords(atom_array: AtomArray, noise_level: float = 1e-3):
        """Replace NaN coordinates with small random noise."""
        coords = atom_array.coord.copy()
        nan_mask = np.isnan(coords)
        coords[nan_mask] = np.random.uniform(
            -noise_level, noise_level, size=nan_mask.sum()
        )
        atom_array.coord = coords
        return atom_array, nan_mask

    atom_array, nan_mask = _handle_nan_coords(atom_array)
    if "chain_iid" not in atom_array.get_annotation_categories():
        atom_array.set_annotation("chain_iid", atom_array.chain_id)
    chain_iids = np.unique(atom_array.chain_iid)
    if len(chain_iids) > 52:
        raise ValueError("Too many chain_iids, cannot convert to PDB")

    all_possible_chainIDS = string.ascii_letters
    chain_map = {}
    for item in chain_iids:
        if len(item) == 1:
            chain_map[item] = item
            all_possible_chainIDS = all_possible_chainIDS.replace(item, "")
    for item in chain_iids:
        if len(item) > 1:
            chain_map[item] = all_possible_chainIDS[0]
            all_possible_chainIDS = all_possible_chainIDS.replace(chain_map[item], "")

    new_chain_ids = [chain_map[i] for i in atom_array.chain_iid]
    inverted_chain_map = {v: k for k, v in chain_map.items()}
    atom_array.chain_id = new_chain_ids
    atom_array.b_factor = np.zeros(len(atom_array))

    # Uniquify duplicate atom names in non-protein residues (e.g. ligands from
    # RF3 that output repeated names like N3, N3, N3). Both the atom_array and
    # the PDB must agree so the HBPLUS-output lookup at line ~164 works.
    aa_mask = filter_amino_acids(atom_array)
    _uniquify_ligand_atom_names(atom_array, aa_mask)

    # PDB format limits residue names to 3 characters.  Prediction oracles
    # (e.g. Boltz2) may output longer names like "LIG1".  Truncate them so
    # biotite's PDB writer doesn't reject the structure.
    long_mask = np.array([len(n) > 3 for n in atom_array.res_name])
    if long_mask.any():
        res_names = atom_array.res_name.copy()
        res_names[long_mask] = [n[:3] for n in res_names[long_mask]]
        atom_array.res_name = res_names

    pdb = PDBFile()
    pdb.set_structure(atom_array)
    pdb.write(output_path)

    return atom_array, nan_mask, inverted_chain_map


def calculate_hbonds(
    atom_array: AtomArray,
    cutoff_HA_dist: float = 3.0,
    cutoff_DA_distance: float = 3.5,
    atom_groups: dict[str, list[str]] | None = None,
    hbplus_path: str | None = None,
) -> tuple[AtomArray, list[dict], dict[str, dict] | None]:
    """
    Calculate hydrogen bonds using HBPLUS and annotate the AtomArray.

    This function identifies hydrogen bonds between motif and non-motif atoms,
    annotates the AtomArray with donor/acceptor information, and returns
    detailed bond information.

    Args:
        atom_array: Input structure with `is_motif_atom` annotation.
        cutoff_HA_dist: Maximum hydrogen-acceptor distance in Å (default: 3.0).
        cutoff_DA_distance: Maximum donor-acceptor distance in Å (default: 3.5).
        atom_groups: Optional mapping of group name to list of atom names.
            When provided, returns per-group h-bond counts as the third element.
        hbplus_path: Resolved HBPLUS executable. When omitted, resolve it from
            Pipelines settings for backwards compatibility.

    Returns:
        Tuple of:
            - atom_array: Annotated with active_donor and active_acceptor fields
            - motif_hbonds: List of dicts containing H-bond details (between motif and non-motif)
            - group_counts: Dict of {group_name: int} (h-bond count) when atom_groups is provided,
              None otherwise.
    """
    from blocks.utils.geometry import get_ligand_mask

    atom_array.set_annotation("is_motif_atom", get_ligand_mask(atom_array))

    from chaperone import get_setting_field

    if hbplus_path is None:
        hbplus_path = get_setting_field("HBPLUS_COPY", check_bin=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_path = os.path.join(tmpdir, "structure.pdb")
        hb2_path = os.path.join(tmpdir, "structure.hb2")
        atom_array, nan_mask, chain_map = save_atomarray_to_pdb(atom_array, pdb_path)

        subprocess.call(
            [
                hbplus_path,
                "-h",
                str(cutoff_HA_dist),
                "-d",
                str(cutoff_DA_distance),
                pdb_path,
                pdb_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=tmpdir,
        )

        HB = open(hb2_path, "r").readlines()
        hbonds = []
        for i in range(8, len(HB)):
            d_chain = HB[i][0]
            d_resi = str(int(HB[i][1:5].strip()))
            d_resn = HB[i][6:9].strip()
            d_ins = HB[i][5].replace("-", " ")
            d_atom = HB[i][9:13].strip()
            a_chain = HB[i][14]
            a_resi = str(int(HB[i][15:19].strip()))
            a_ins = HB[i][19].replace("-", " ")
            a_resn = HB[i][20:23].strip()
            a_atom = HB[i][23:27].strip()
            dist = float(HB[i][27:32].strip())

            items = {
                "d_chain": chain_map[d_chain],
                "d_resi": d_resi,
                "d_resn": d_resn,
                "d_ins": d_ins,
                "d_atom": d_atom,
                "a_chain": chain_map[a_chain],
                "a_resi": a_resi,
                "a_resn": a_resn,
                "a_ins": a_ins,
                "a_atom": a_atom,
                "dist": dist,
            }
            hbonds.append(items)

    # Prepare output arrays
    n = len(atom_array)

    donor_mask = np.zeros(n, dtype=bool)
    acceptor_mask = np.zeros(n, dtype=bool)
    motif_hbonds = []

    for item in hbonds:
        # Locate atoms
        d_mask = (
            (atom_array.chain_iid == item["d_chain"])
            & (atom_array.res_id == float(item["d_resi"]))
            & (atom_array.atom_name == item["d_atom"])
        )
        a_mask = (
            (atom_array.chain_iid == item["a_chain"])
            & (atom_array.res_id == float(item["a_resi"]))
            & (atom_array.atom_name == item["a_atom"])
        )
        if d_mask.sum() != 1 or a_mask.sum() != 1:
            raise ValueError(f"Cannot uniquely identify atoms in H-bond line: {item}")

        d_idx = np.where(d_mask)[0][0]
        a_idx = np.where(a_mask)[0][0]

        d_motif = atom_array.is_motif_atom[d_idx]
        a_motif = atom_array.is_motif_atom[a_idx]
        if d_motif == a_motif:
            continue

        motif_hbonds.append(item)
        donor_mask[d_idx] = True
        acceptor_mask[a_idx] = True

    # Annotate
    atom_array.set_annotation("active_donor", donor_mask.astype(int))
    atom_array.set_annotation("active_acceptor", acceptor_mask.astype(int))

    # Per-group h-bond counts
    group_counts = None
    if atom_groups:
        group_counts = {gname: 0 for gname in atom_groups}
        for hb in motif_hbonds:
            # Determine which atom is on the motif (ligand) side
            d_idx = np.where(
                (atom_array.chain_iid == hb["d_chain"])
                & (atom_array.res_id == float(hb["d_resi"]))
                & (atom_array.atom_name == hb["d_atom"])
            )[0][0]
            a_idx = np.where(
                (atom_array.chain_iid == hb["a_chain"])
                & (atom_array.res_id == float(hb["a_resi"]))
                & (atom_array.atom_name == hb["a_atom"])
            )[0][0]
            motif_atom_name = (
                atom_array.atom_name[d_idx]
                if atom_array.is_motif_atom[d_idx]
                else atom_array.atom_name[a_idx]
            )
            for gname, gnames in atom_groups.items():
                if motif_atom_name in gnames:
                    group_counts[gname] += 1

    return atom_array, motif_hbonds, group_counts
