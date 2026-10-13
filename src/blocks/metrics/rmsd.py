import logging
from typing import Dict, List, Optional

import numpy as np
from biotite.structure import AtomArray
from pydantic import Field

from blocks.metrics.utils import (
    find_automorphisms,
    get_rmsd,
    greedy_resolve_symmetries,
    resolve_subset_symmetries,
    weighted_rigid_align,
)
from blocks.utils.geometry import match_atoms_by_identifier
from pipelines import InlineMetricBlock, JobArgs
from pipelines.data.components import set_component_remap
from pipelines.data.schema import SELECTIONS

logger = logging.getLogger(__name__)


###########################################################################
# Main Block
###########################################################################


def _design_order_like_source(src_atoms: AtomArray, des_atoms: AtomArray) -> np.ndarray:
    """Row order that lists the design's atoms in the source's order.

    A boolean mask selects atoms in a structure's own row order, so atoms paired by
    name (``select_mask_by_name_pairing``) are only compared correctly when both
    structures list them in the same order. Atoms are matched on
    ``(chain_id, res_id, atom_name)``, then, for those left over, on
    ``(res_name, atom_name)``, the same identifiers ``match_atoms_by_identifier``
    uses. Unmatched design atoms keep their relative order after the matched ones.
    The result is the identity when the structures already agree.
    """
    exact: dict[tuple, int] = {}
    for i in range(len(src_atoms)):
        exact.setdefault(
            (src_atoms.chain_id[i], src_atoms.res_id[i], src_atoms.atom_name[i]), i
        )
    position = np.full(len(des_atoms), len(src_atoms) + len(des_atoms), dtype=np.int64)
    taken: set[int] = set()
    leftover: list[int] = []
    for j in range(len(des_atoms)):
        key = (des_atoms.chain_id[j], des_atoms.res_id[j], des_atoms.atom_name[j])
        i = exact.get(key)
        if i is not None and i not in taken:
            position[j] = i
            taken.add(i)
        else:
            leftover.append(j)
    loose: dict[tuple, list[int]] = {}
    for i in range(len(src_atoms)):
        if i not in taken:
            loose.setdefault(
                (src_atoms.res_name[i], src_atoms.atom_name[i]), []
            ).append(i)
    for j in leftover:
        candidates = loose.get((des_atoms.res_name[j], des_atoms.atom_name[j]))
        if candidates:
            position[j] = candidates.pop(0)
    return np.lexsort((np.arange(len(des_atoms)), position))


def _get_paired_masks(
    src_atoms: AtomArray,
    des_atoms: AtomArray,
    src_mask: np.ndarray,
    des_mask: np.ndarray,
) -> tuple:
    """
    Create paired masks for atoms present in BOTH structures.

    Thin wrapper around :func:`blocks.utils.geometry.match_atoms_by_identifier`.
    The (chain_id, res_id, atom_name) → (res_name, atom_name) fallback semantics
    are preserved exactly there (load-bearing for AF3 ligand renumbering).

    Args:
        src_atoms: Source atom array
        des_atoms: Design atom array
        src_mask: Boolean mask for source
        des_mask: Boolean mask for design

    Returns:
        Tuple of (paired_src_mask, paired_des_mask) with matching atoms
    """
    return match_atoms_by_identifier(src_atoms, des_atoms, src_mask, des_mask)


class RMSDBlock(InlineMetricBlock):
    """Compute alignment and RMSD metrics between two designs in the lineage.

    Uses SELECTIONS enum for both alignment sources and RMSD targets,
    with metric keys formatted as "{alignment}_align.{target}_rmsd"
    """

    _name = "rmsd"
    _display_name = "RMSD"
    _description = "Compute alignment RMSD metrics between two designs"
    _creates_new_step = False
    _resources = JobArgs(partition="cpu")
    which: List[int] = Field(
        default=[0, -1],
        description="Indices into each row's structures list selecting the two "
        "structures to compare. Default [0, -1] compares first vs last. "
        "Use [-2, -1] to compare penultimate (e.g. MPNN input) vs last (e.g. AF3 output).",
    )
    alignment_methods: List[SELECTIONS] = Field(
        default=[SELECTIONS.BKBN, SELECTIONS.MOTIF_BKBN],
        description="List of selections to use for computing alignment transformations",
    )
    rmsd_targets: List[SELECTIONS] = Field(
        default=[SELECTIONS.BKBN, SELECTIONS.MOTIF_ALLATOM, SELECTIONS.LIGANDS],
        description="List of selections to compute RMSD on after alignment",
    )
    resolve_symmetries: bool = Field(
        default=True, description="Whether to resolve automorphisms for all-atom RMSD"
    )
    select_mask_by_name_pairing: List[SELECTIONS] = Field(
        default=[SELECTIONS.LIGANDS],
        description="List of RMSD targets where atom masks are paired by finding "
        "common atoms (via chain_id, res_id, atom_name) instead of requiring "
        "identical mask sizes. Useful when structures differ (e.g. PTMs, ligands).",
    )
    component_remap: dict[str, str] | None = Field(
        default=None,
        description="Remap component names for ligand lookup (e.g., {'LIG_B': 'B;N'}). "
        "Needed when upstream tools rename ligands and RMSD runs on SLURM workers.",
    )
    atom_groups: Optional[dict[str, list[str]]] = Field(
        default=None,
        description="Mapping of group name to list of ligand atom names. "
        "When provided, per-group ligand RMSD values are included in the output.",
    )

    def get_data_kwargs(self):
        """Load the first+last (or ``self.which``) structures of each row with
        the alignment/target atom selections needed for masking."""
        return {
            "which": self.which,
            "load": True,
            "select": list(set(self.alignment_methods) | set(self.rmsd_targets)),
        }

    def setup(self, handler, sandbox_dir, chunk_id):
        if self.component_remap:
            set_component_remap(self.component_remap)
        return None

    def compute_metrics(self, loaded, *, context=None, index=0):
        first_loaded, last_loaded = loaded
        try:
            metrics = self._compute_alignments(
                src_atoms=first_loaded.atom_array,
                src_masks=first_loaded.masks,
                des_atoms=last_loaded.atom_array,
                des_masks=last_loaded.masks,
                atom_groups=self.atom_groups,
            )
            logger.info(
                "Computed RMSDs for {} to {}: {}".format(
                    first_loaded.path, last_loaded.path, metrics
                )
            )
        except Exception as e:
            logger.error(
                "Failed to compute alignment for {} to {}: {}".format(
                    first_loaded.path, last_loaded.path, e
                ),
                exc_info=True,
            )
            raise e

        return metrics

    def _compute_alignments(
        self,
        src_atoms: AtomArray,
        src_masks: Dict[SELECTIONS, np.ndarray],
        des_atoms: AtomArray,
        des_masks: Dict[SELECTIONS, np.ndarray],
        atom_groups: Optional[dict[str, list[str]]] = None,
    ):
        """Compute configured RMSDs after each configured alignment."""
        all_metrics = {}

        if self.select_mask_by_name_pairing:
            order = _design_order_like_source(src_atoms, des_atoms)
            if not np.array_equal(order, np.arange(len(des_atoms))):
                des_atoms = des_atoms[order]
                des_masks = {sele: mask[order] for sele, mask in des_masks.items()}

        xyz_src = src_atoms.coord.reshape(-1, 3)
        xyz_des = des_atoms.coord.reshape(-1, 3)
        automorphisms = None
        if self.resolve_symmetries:
            automorphisms = find_automorphisms(src_atoms)

        # Double for loop over alignment first, then different targets second.
        for align_method in self.alignment_methods:
            # Skip alignment methods that don't have masks (e.g., missing motif)
            if align_method not in src_masks or align_method not in des_masks:
                logger.warning(
                    f"Skipping alignment method {align_method}: missing mask"
                )
                continue

            try:
                mask_src = src_masks[align_method]
                mask_des = des_masks[align_method]

                # Handle mismatched atom counts via name-pairing if requested
                if mask_src.sum() != mask_des.sum():
                    if align_method in self.select_mask_by_name_pairing:
                        mask_src, mask_des = _get_paired_masks(
                            src_atoms, des_atoms, mask_src, mask_des
                        )
                        logger.info(
                            f"Alignment {align_method}: paired {mask_src.sum()} common atoms "
                            f"(source had {src_masks[align_method].sum()}, "
                            f"design had {des_masks[align_method].sum()})"
                        )
                    else:
                        raise ValueError(
                            f"Atom count mismatch for {align_method}: "
                            f"source has {mask_src.sum()} atoms, design has {mask_des.sum()} atoms. "
                            f"Add {align_method} to select_mask_by_name_pairing to handle this, "
                            f"or check that the first/last pipeline inputs are correct."
                        )

                align_func = weighted_rigid_align(
                    xyz_src[mask_src][None],
                    xyz_des[mask_des][None],
                    return_alignment_function=True,
                )
                xyz_des_aligned = align_func(xyz_des)
            except Exception as e:
                logger.error(f"Failed to compute alignment {str(align_method)}: {e}")
                raise e

            for rmsd_target in self.rmsd_targets:
                metric_name = f"{align_method}_align.{rmsd_target}_rmsd"

                # Skip RMSD targets that don't have masks (e.g., missing ligand data)
                if rmsd_target not in src_masks or rmsd_target not in des_masks:
                    logger.warning(f"Skipping RMSD target {rmsd_target}: missing mask")
                    continue

                try:
                    rmsd_mask_src = src_masks[rmsd_target]
                    rmsd_mask_des = des_masks[rmsd_target]

                    # Fuse masks for unsafe targets by finding common atoms
                    if rmsd_target in self.select_mask_by_name_pairing:
                        rmsd_mask_src, rmsd_mask_des = _get_paired_masks(
                            src_atoms, des_atoms, rmsd_mask_src, rmsd_mask_des
                        )
                        logger.debug(
                            f"Using {rmsd_mask_src.sum()} paired atoms for unsafe target {rmsd_target}"
                        )

                    xyz_src_subset = xyz_src[rmsd_mask_src]
                    xyz_des_subset = xyz_des_aligned[rmsd_mask_des]

                    xyz_src_subset = _apply_symmetry_resolution(
                        xyz_src_subset,
                        xyz_des_subset,
                        rmsd_mask_src,
                        src_atoms,
                        automorphisms,
                    )

                    rmsd_value = get_rmsd(xyz_src_subset, xyz_des_subset)
                    all_metrics[metric_name] = float(rmsd_value)

                    if rmsd_target == SELECTIONS.LIGANDS and atom_groups:
                        for group_name, group_atom_names in atom_groups.items():
                            group_set = set(group_atom_names)
                            group_src = rmsd_mask_src & np.isin(
                                src_atoms.atom_name, list(group_set)
                            )
                            group_des = rmsd_mask_des & np.isin(
                                des_atoms.atom_name, list(group_set)
                            )
                            n_src = group_src.sum()
                            n_des = group_des.sum()
                            if n_src == 0 or n_des == 0:
                                logger.warning(
                                    f"Skipping atom group '{group_name}': "
                                    f"0 matching atoms (src={n_src}, des={n_des})"
                                )
                                continue
                            if n_src != n_des:
                                logger.warning(
                                    f"Skipping atom group '{group_name}': "
                                    f"atom count mismatch (src={n_src}, des={n_des})"
                                )
                                continue

                            grp_src_xyz = _apply_symmetry_resolution(
                                xyz_src[group_src],
                                xyz_des_aligned[group_des],
                                group_src,
                                src_atoms,
                                automorphisms,
                            )
                            grp_rmsd = get_rmsd(grp_src_xyz, xyz_des_aligned[group_des])
                            all_metrics.setdefault(group_name, {})[metric_name] = float(
                                grp_rmsd
                            )

                    chain_ids_in_mask = np.unique(src_atoms.chain_id[rmsd_mask_src])
                    if rmsd_target == SELECTIONS.LIGANDS and len(chain_ids_in_mask) > 1:
                        assert (
                            rmsd_mask_src.sum() > 0
                        ), "No ligand atoms found in source for RMSD calculation"
                        for chain_id in chain_ids_in_mask:
                            chain_mask_src = rmsd_mask_src & (
                                src_atoms.chain_id == chain_id
                            )
                            chain_mask_des = rmsd_mask_des & (
                                des_atoms.chain_id == chain_id
                            )

                            xyz_src_subset = xyz_src[chain_mask_src]
                            xyz_des_subset = xyz_des_aligned[chain_mask_des]

                            if xyz_src_subset.shape != xyz_des_subset.shape:
                                src_ids = set(
                                    zip(
                                        src_atoms.chain_id[chain_mask_src],
                                        src_atoms.res_id[chain_mask_src].astype(str),
                                        src_atoms.atom_name[chain_mask_src],
                                    )
                                )
                                des_ids = set(
                                    zip(
                                        des_atoms.chain_id[chain_mask_des],
                                        des_atoms.res_id[chain_mask_des].astype(str),
                                        des_atoms.atom_name[chain_mask_des],
                                    )
                                )
                                only_src = sorted(src_ids - des_ids)
                                only_des = sorted(des_ids - src_ids)
                                logger.error(
                                    f"Shape mismatch for chain {chain_id} "
                                    f"({rmsd_target}): src={xyz_src_subset.shape} "
                                    f"des={xyz_des_subset.shape}. "
                                    f"Only in src: {only_src[:20]}{'...' if len(only_src) > 20 else ''}. "
                                    f"Only in des: {only_des[:20]}{'...' if len(only_des) > 20 else ''}"
                                )

                            xyz_src_subset_sym_resolved = _apply_symmetry_resolution(
                                xyz_src_subset,
                                xyz_des_subset,
                                chain_mask_src,
                                src_atoms,
                                automorphisms,
                            )

                            assert (
                                xyz_src_subset_sym_resolved.shape
                                == xyz_src_subset.shape
                            ), (
                                f"Shape mismatch after symmetry resolution for chain {chain_id} "
                                f"({rmsd_target}): src={xyz_src_subset_sym_resolved.shape} "
                                f"des={xyz_des_subset.shape}"
                            )

                            rmsd_value = get_rmsd(
                                xyz_src_subset_sym_resolved, xyz_des_subset
                            )
                            metric_name = (
                                f"{align_method}_align.{rmsd_target}_{chain_id}_rmsd"
                            )
                            all_metrics[metric_name] = float(rmsd_value)

                except Exception as e:
                    logger.error(f"Error computing {metric_name}: {e}")
                    raise e

        return all_metrics


def _apply_symmetry_resolution(
    xyz_src_subset, xyz_des_subset, mask, src_atoms, automorphisms
):
    """Apply symmetry resolution if enabled."""
    if automorphisms is None:
        return xyz_src_subset

    # Full automorphisms for all-atom
    if mask.sum() == len(src_atoms):
        return greedy_resolve_symmetries(xyz_src_subset, xyz_des_subset, automorphisms)
    elif mask.sum() < len(src_atoms):
        return resolve_subset_symmetries(
            xyz_src_subset, xyz_des_subset, mask, src_atoms, automorphisms
        )

    return xyz_src_subset
