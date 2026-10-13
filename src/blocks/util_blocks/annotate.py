import os

from chaperone import BlockFunction, PipelineDataHandler
import numpy as np
from pydantic import Field
from typing import List, Optional
from dataclasses import dataclass, field
from scipy.spatial import cKDTree

all_annotations = ["interface", "hbond", "aromatics_near_ligand", "ligand"]


@dataclass
class InterfaceQuery:
    chain: Optional[str] = None
    residue: Optional[str] = None


@dataclass
class InterfaceConfig:
    min_contact_radius: float = 2.0
    contact_radius: float = 3.5
    query: InterfaceQuery = field(default_factory=InterfaceQuery)


class AnnotateBlock(BlockFunction):
    """Annotate molecular structures with interface, hbond, and ligand info."""

    _name = "annotate"
    _display_name = "Annotate"
    _description = "Annotate molecular structures"
    _inline = True
    _chunked = False

    annotations: List[str] = Field(default=all_annotations)
    overwrite: bool = Field(
        True, description="Whether to overwrite existing annotations"
    )
    interface_config: InterfaceConfig = Field(
        default_factory=InterfaceConfig,
        description="Configuration for interface residue annotation",
        json_schema_extra={"cli_type": None},
    )

    def __call__(self, handler, sandbox_dir=None, **kwargs):
        rows_out = []
        for loaded in handler.get_data(which="last", load=True):
            atom_array = loaded.atom_array
            design_meta = loaded.metadata or {}

            if isinstance(design_meta, dict):
                design_meta = {k: v for k, v in design_meta.items() if v is not None}
            else:
                design_meta = {}

            updates = {}

            if "ligand" in self.annotations and (
                self.overwrite or design_meta.get("ligand") is None
            ):
                updates["ligand"] = find_ligands_in_array(atom_array)
            if "hbond" in self.annotations and (
                self.overwrite or design_meta.get("hbond") is None
            ):
                updates["hbond"] = find_hbonds_in_pdb(atom_array)
            if "aromatics_near_ligand" in self.annotations and (
                self.overwrite or design_meta.get("aromatics_near_ligand") is None
            ):
                updates["aromatics_near_ligand"] = find_aromatic_residues_near_ligand(
                    atom_array
                )
            if "interface" in self.annotations and (
                self.overwrite or design_meta.get("interface") is None
            ):
                updates["interface"] = _get_closest_residues_by_sidechain_dist(
                    atom_array, conf=self.interface_config
                )

            row_out = loaded.to_row_with_updates(metadata=updates)
            rows_out.append(row_out)

        return PipelineDataHandler(rows_out)


def find_ligands_in_array(atom_array, as_resname=True):
    ligand_residues = set()
    if as_resname:
        for index in np.where(atom_array.is_ligand)[0]:
            ligand_residues.add(atom_array.res_name[index])
    else:
        chain_id = atom_array.chain_id[index].capitalize()
        resi = atom_array.res_id[index]
        ligand_residues.add(f"{chain_id}{resi}")
    return list(ligand_residues)


def find_hbonds_in_pdb(atom_array):
    # Directly compute the hbonds given atom array

    fix_hbonds_residues = set()
    ligand_names = atom_array[atom_array.is_ligand].res_name
    from chaperone import get_setting_field

    if len(ligand_names) == 0 or not os.path.exists(get_setting_field("HBPLUS_COPY")):
        return list(fix_hbonds_residues)
    ligand_name = ligand_names[0]

    # Calculate hbonds
    from blocks.utils.hbonds import calculate_hbonds

    atom_array, hbonds, _ = calculate_hbonds(atom_array)

    for hbond in hbonds:
        donor_chain = hbond["d_chain"].split("_")[0]
        donor_resi = hbond["d_resi"]
        donor_name = hbond["d_resn"]
        acceptor_chain = hbond["a_chain"].split("_")[0]
        acceptor_resi = hbond["a_resi"]
        acceptor_name = hbond["a_resn"]
        if donor_name == ligand_name:
            fix_hbonds_residues.add(f"{acceptor_chain}{acceptor_resi}")
        elif acceptor_name == ligand_name:
            fix_hbonds_residues.add(f"{donor_chain}{donor_resi}")

    return list(fix_hbonds_residues)


def find_aromatic_residues_near_ligand(
    atom_array, distance_threshold=8.0, ligand_atom_names=None
):
    """Return unique aromatic protein residues near selected ligand atoms."""
    from blocks.constants import AROMATIC_RESIDUES
    from blocks.utils.geometry import _find_nearest_neighbors

    ligand_mask = np.asarray(atom_array.is_ligand, dtype=bool)
    if ligand_atom_names is not None:
        ligand_mask &= np.isin(atom_array.atom_name, list(ligand_atom_names))
    ligand_indices = np.where(ligand_mask)[0]
    if len(ligand_indices) == 0:
        return []
    closest_residue_indices = _find_nearest_neighbors(
        atom_array, ligand_indices, distance_threshold
    )
    aromatic_neighbor_ids = set()
    for atom_idx in closest_residue_indices:
        res_name = atom_array.res_name[atom_idx]
        if res_name in AROMATIC_RESIDUES:
            chain_id = atom_array.chain_id[atom_idx].capitalize()
            res_id = atom_array.res_id[atom_idx]
            aromatic_neighbor_ids.add(f"{chain_id}{res_id}")
    # Return list to match schema
    return sorted(list(aromatic_neighbor_ids))


##########################################################################
# Helper functions for annotation
###########################################################################


def _get_closest_residues_by_sidechain_dist(
    atom_array, conf: InterfaceConfig
) -> list[str]:
    """Find protein residues near ligands/nucleic acids based on sidechain distances.

    Args:
        atom_array: Biotite AtomArray structure
        conf: Configuration with query, min_contact_radius, and contact_radius attributes

    Returns:
        List of interface residues (e.g., ["A10", "A15", "B3"])
    """

    def query_shell(tree, query_coords, min_r, max_r):
        """Query atoms within an annular shell (min_r <= distance <= max_r)."""
        neighbors = tree.query_ball_point(query_coords, r=max_r)
        filtered_inds = []
        for qi, inds in zip(query_coords, neighbors):
            if len(inds) == 0:
                continue
            dists = np.linalg.norm(tree.data[inds] - qi, axis=1)
            keep = dists >= min_r
            if np.any(keep):
                filtered_inds.append(np.array(inds)[keep])
        return filtered_inds

    # Ensure type annotations exist
    atom_count = getattr(atom_array, "array_length", lambda: len(atom_array))()

    # Filter valid coordinates
    coords = atom_array.coord
    valid_atoms = ~np.isnan(coords).any(axis=1)
    coords_valid = coords[valid_atoms]
    if coords_valid.size == 0:
        return []

    # Extract type annotations
    is_dna_atom = atom_array.get_annotation("is_dna")[valid_atoms]
    is_rna_atom = atom_array.get_annotation("is_rna")[valid_atoms]
    is_ligand_atom = atom_array.get_annotation("is_ligand")[valid_atoms]

    # Build user-specified query mask
    user_query = np.zeros(atom_count, dtype=bool)
    if conf.query.chain:
        for cur_chain_query in conf.query.chain.split(","):
            user_query[atom_array.chain_id == cur_chain_query] = True
    if conf.query.residue:
        for cur_residue_query in conf.query.residue.split(","):
            res_id = (
                int(cur_residue_query)
                if cur_residue_query.isdigit()
                else cur_residue_query
            )
            user_query[atom_array.res_id == res_id] = True

    # Identify sidechain atoms (excluding backbone)
    backbone_names = {"N", "CA", "C", "O"}
    is_sidechain_atom = np.array(
        [name.strip() not in backbone_names for name in atom_array.atom_name]
    )[valid_atoms]

    # Combine query masks
    query_mask = is_ligand_atom | is_dna_atom | is_rna_atom
    query_mask_full = np.zeros(atom_count, dtype=bool)
    query_mask_full[valid_atoms] = query_mask
    query_mask_full |= user_query
    query_mask = query_mask_full[valid_atoms]

    if not np.any(query_mask):
        return []

    # Build spatial index and query shell
    tree = cKDTree(coords_valid)
    inds_list = query_shell(
        tree, coords_valid[query_mask], conf.min_contact_radius, conf.contact_radius
    )

    # Filter for sidechain contacts only
    first_shell_atom_idx_valid = []
    for inds in inds_list:
        sidechain_inds = inds[is_sidechain_atom[inds]]
        if len(sidechain_inds) > 0:
            first_shell_atom_idx_valid.append(sidechain_inds)

    if not first_shell_atom_idx_valid:
        return []

    first_shell_atom_idx_valid = np.unique(np.concatenate(first_shell_atom_idx_valid))

    # Map back to full atom array and exclude query atoms
    first_shell_protein_mask_valid = np.zeros(coords_valid.shape[0], dtype=bool)
    first_shell_protein_mask_valid[first_shell_atom_idx_valid] = True
    first_shell_protein_mask = np.zeros(atom_count, dtype=bool)
    first_shell_protein_mask[valid_atoms] = first_shell_protein_mask_valid
    first_shell_protein_mask &= ~query_mask_full

    # Extract unique residues
    first_shell_residues_atom_array = atom_array[first_shell_protein_mask]
    residues = {
        f"{str(first_shell_residues_atom_array.chain_id[i]).upper()}{first_shell_residues_atom_array.res_id[i]}"
        for i in range(len(first_shell_residues_atom_array))
    }
    return sorted(list(residues))
