"""Backbone clash metrics block.

Counts non-bonded backbone heavy-atom pairs that sit closer than a cutoff. This is a
*model-level* structure-quality metric: it is computed on a generated backbone before
any sequence design or structure-prediction filtering, so it isolates the generator's
own geometry from downstream pipeline enrichment.
"""

import logging

import biotite.structure as struc
import numpy as np
from pydantic import Field
from scipy.spatial import cKDTree

from chaperone import InlineMetricBlock, JobArgs

logger = logging.getLogger(__name__)

BACKBONE_ATOMS = ("N", "CA", "C", "O")

# Neighbours examined per atom when finding the closest non-bonded contact. Each
# backbone atom has at most a handful of bonded/same-residue neighbours, so the
# nearest 16 comfortably contain the closest non-bonded one.
_MIN_DIST_NEIGHBOURS = 16


class ClashMetricsBlock(InlineMetricBlock):
    """Count non-bonded backbone heavy-atom clashes in a structure.

    A clash is a pair of backbone heavy atoms (N/CA/C/O) closer than ``cutoff``
    angstroms that is not bonded — i.e. not in the same residue and not in
    consecutive residues of the same chain.
    """

    _name = "clash"
    _display_name = "Clash Metrics"
    _description = "Count non-bonded backbone heavy-atom clashes"
    _creates_new_step = False
    _resources = JobArgs(partition="cpu", rows_per_job=500)

    cutoff: float = Field(
        default=1.5,
        description="Distance (A) below which a non-bonded backbone atom pair counts "
        "as a clash.",
    )
    backbone_only: bool = Field(
        default=True,
        description="Restrict to backbone heavy atoms (N/CA/C/O). Set False to use all "
        "heavy atoms — note this is not comparable across generators that emit "
        "different sidechain completeness (e.g. poly-alanine backbones).",
    )
    sequence_separation: int = Field(
        default=1,
        description="Residues this far apart or closer in the same chain are treated as "
        "bonded and excluded. 1 excludes intra-residue and adjacent-residue pairs.",
    )
    exclude_hetero: bool = Field(
        default=True,
        description="Drop hetero (ligand/cofactor) atoms before scoring. Required for "
        "correctness whenever a ligand is a canonical amino acid — e.g. ornithine (ORN) "
        "passes filter_amino_acids, contributes N/CA/C/O to the backbone set, and sits "
        "in the active site on its own chain, so every protein-ligand contact would be "
        "miscounted as a backbone clash.",
    )

    def get_data_kwargs(self):
        return {"which": "last", "load": True}

    def compute_metrics(self, loaded, *, context=None, index=0):
        try:
            atom_array = loaded.atom_array
            keep = struc.filter_amino_acids(atom_array)
            if self.exclude_hetero and hasattr(atom_array, "hetero"):
                keep = keep & ~np.asarray(atom_array.hetero, dtype=bool)
            protein = atom_array[keep]
            if self.backbone_only:
                protein = protein[np.isin(protein.atom_name, BACKBONE_ATOMS)]
            # Drop hydrogens: "heavy atom" is part of the metric definition.
            protein = protein[protein.element != "H"]

            n_atoms = len(protein)
            n_residues = struc.get_residue_count(protein) if n_atoms else 0
            if n_atoms < 2:
                logger.warning(
                    "%s: structure has %d backbone heavy atoms — no clash metric",
                    self.name,
                    n_atoms,
                )
                return {
                    "n_bb_clashes": None,
                    "bb_clash_rate": None,
                    "min_bb_dist": None,
                    "n_residues": n_residues,
                }

            coords = protein.coord.astype(np.float64)
            tree = cKDTree(coords)

            # Residue ordinal per atom, so "same residue" is a single integer compare.
            _, res_ordinal = np.unique(
                np.stack([protein.chain_id, protein.res_id.astype(str)], axis=1),
                axis=0,
                return_inverse=True,
            )

            pairs = np.array(sorted(tree.query_pairs(r=self.cutoff)), dtype=int)
            if pairs.size:
                keep = self._non_bonded_mask(protein, res_ordinal, pairs[:, 0], pairs[:, 1])
                n_clashes = int(keep.sum())
            else:
                n_clashes = 0

            return {
                "n_bb_clashes": n_clashes,
                "bb_clash_rate": float(n_clashes / n_residues) if n_residues else None,
                "min_bb_dist": self._min_non_bonded_distance(protein, res_ordinal, tree, coords),
                "n_residues": int(n_residues),
            }
        except Exception as exc:  # noqa: BLE001 — one bad row must not kill the chunk
            logger.warning("%s: clash metric failed on index %d: %s", self.name, index, exc)
            return {
                "n_bb_clashes": None,
                "bb_clash_rate": None,
                "min_bb_dist": None,
                "n_residues": None,
            }

    def _non_bonded_mask(self, protein, res_ordinal, i, j) -> np.ndarray:
        """True where the atom pair (i, j) is NOT bonded/same-residue."""
        same_residue = res_ordinal[i] == res_ordinal[j]
        same_chain = protein.chain_id[i] == protein.chain_id[j]
        adjacent = same_chain & (
            np.abs(protein.res_id[i].astype(int) - protein.res_id[j].astype(int))
            <= self.sequence_separation
        )
        return ~(same_residue | adjacent)

    def _min_non_bonded_distance(self, protein, res_ordinal, tree, coords) -> float | None:
        """Closest non-bonded backbone contact, regardless of the clash cutoff."""
        k = min(_MIN_DIST_NEIGHBOURS, len(coords))
        distances, neighbours = tree.query(coords, k=k)
        # Column 0 is the atom itself; every remaining column is a candidate.
        source = np.repeat(np.arange(len(coords)), k - 1)
        target = neighbours[:, 1:].ravel()
        dist = distances[:, 1:].ravel()
        keep = self._non_bonded_mask(protein, res_ordinal, source, target)
        return float(dist[keep].min()) if keep.any() else None
