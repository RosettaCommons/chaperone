"""Rotamer metrics block for scoring sidechain rotamer quality."""

import json
import logging
from typing import Literal

import biotite.structure as struc
import numpy as np
from pydantic import Field

from pipelines import InlineMetricBlock, JobArgs
from pipelines.data.schema import SELECTIONS

logger = logging.getLogger(__name__)


class RotamerMetricsBlock(InlineMetricBlock):
    """Compute rotamer quality metrics for protein structures.

    Scores sidechain rotamer conformations using either the peppr library
    (Top8000-based, default) or the Dunbrack backbone-dependent rotamer library.
    Optionally restricts scoring to motif residues only.
    """

    _name = "rotamer_metrics"
    _display_name = "Rotamer Metrics"
    _description = "Compute rotamer quality metrics for protein sidechains"
    _creates_new_step = False
    _resources = JobArgs(partition="cpu")

    backend: Literal["dunbrack", "peppr"] = Field(
        default="dunbrack",
        description="Scoring backend: 'peppr' (Top8000 library via peppr) or "
        "'dunbrack' (backbone-dependent Dunbrack library).",
    )
    motif_only: bool = Field(
        default=True,
        description="Only score residues annotated as motif (is_motif_atom). "
        "If the structure has no motif annotation, all residues are scored.",
    )
    include_rama: bool = Field(
        default=True,
        description="Also compute Ramachandran scores (peppr backend only).",
    )
    detailed_output: bool = Field(
        default=False,
        description="Include per-residue-type breakdown of scores.",
    )
    per_residue_output: bool = Field(
        default=True,
        description="Include per-original-residue scores using diffused_index_map "
        "metadata to map back to original residue identifiers "
        "(e.g. mean_rotamer_pct.A1).",
    )

    def get_data_kwargs(self):
        return {
            "which": "last",
            "load": True,
            "select": [SELECTIONS.MOTIF_ALLATOM],
        }

    def compute_metrics(self, loaded, *, context=None, index=0):
        try:
            atom_array = loaded.atom_array
            motif_mask = loaded.masks.get(SELECTIONS.MOTIF_ALLATOM, None)

            # Build set of motif residue identifiers (chain_id + res_id) for
            # filtering scores *after* computing on the full structure.
            # We do NOT subset the atom array because chi/phi/psi angles
            # require neighboring backbone atoms.
            motif_res_ids = None
            if self.motif_only:
                assert motif_mask is not None, (
                    "motif_only=True but no motif mask found in data"
                )
                motif_res_ids = _get_residue_ids_from_mask(atom_array, motif_mask)

            # Keep the full protein backbone context required for torsion angles.
            protein_mask = struc.filter_amino_acids(atom_array)
            atom_array = atom_array[protein_mask]

            reverse_index_map = None
            if self.per_residue_output:
                reverse_index_map = _build_reverse_index_map(loaded.metadata)

            if len(atom_array) == 0:
                logger.warning("No protein atoms found for %s, skipping", loaded.path)
                metrics = {}
            elif self.backend == "peppr":
                metrics = self._score_peppr(
                    atom_array, motif_res_ids, reverse_index_map
                )
            else:
                metrics = self._score_dunbrack(
                    atom_array, motif_res_ids, reverse_index_map
                )

            logger.info(
                "Computed rotamer metrics for %s: %s",
                loaded.path,
                {
                    k: v
                    for k, v in metrics.items()
                    if not k.startswith("mean_rotamer_prob.")
                    and not k.startswith("mean_rotamer_pct.")
                },
            )
        except Exception as e:
            logger.error(
                "Failed to compute rotamer metrics for %s: %s",
                loaded.path,
                e,
                exc_info=True,
            )
            raise

        return metrics

    def _score_peppr(self, atom_array, motif_res_ids, reverse_index_map) -> dict:
        """Score rotamers using peppr (Top8000 library)."""
        from peppr.rotamer import (
            ConformerClass,
            RamaScore,
            RotamerScore,
        )

        metrics = {}

        rota = RotamerScore.from_atoms(atom_array)
        # ALA, GLY, and other UNKNOWN residues have no chi angles to score.
        all_scores = [
            s for s in rota.rotamer_scores if s.classification != ConformerClass.UNKNOWN
        ]

        if motif_res_ids is not None:
            scores = [s for s in all_scores if (s.chain_id, s.res_id) in motif_res_ids]
        else:
            scores = all_scores

        if scores:
            pcts = [s.pct for s in scores]
            classifications = [s.classification for s in scores]
            n_total = len(scores)

            metrics["mean_rotamer_pct"] = float(np.mean(pcts))
            metrics["frac_rotamer_outliers"] = (
                sum(1 for c in classifications if c == ConformerClass.OUTLIER) / n_total
            )
            metrics["frac_rotamer_favored"] = (
                sum(1 for c in classifications if c == ConformerClass.FAVORED) / n_total
            )
            metrics["n_rotamer_residues"] = n_total

            if self.detailed_output:
                by_resname = {}
                for s in scores:
                    by_resname.setdefault(s.res_name, []).append(s.pct)
                for res_name, pct_list in by_resname.items():
                    metrics[f"mean_rotamer_pct.{res_name}"] = float(np.mean(pct_list))

            if self.per_residue_output and reverse_index_map is not None:
                for s in scores:
                    diffused_key = f"{s.chain_id}{s.res_id}"
                    original_key = reverse_index_map.get(diffused_key)
                    if original_key:
                        metrics[f"mean_rotamer_pct.{original_key}"] = float(s.pct)
                    else:
                        print(
                            f"Warning: no mapping for diffused residue {diffused_key}, skipping per-residue output for this residue"
                        )

        if self.include_rama:
            rama = RamaScore.from_atoms(atom_array)
            all_rama_scores = rama.rama_scores

            if motif_res_ids is not None and all_rama_scores:
                rama_scores = [
                    s
                    for s in all_rama_scores
                    if (s.chain_id, s.res_id) in motif_res_ids
                ]
            else:
                rama_scores = all_rama_scores

            if rama_scores:
                rama_classifications = [s.classification for s in rama_scores]
                n_rama = len(rama_scores)

                metrics["frac_rama_outliers"] = (
                    sum(1 for c in rama_classifications if c == ConformerClass.OUTLIER)
                    / n_rama
                )
                metrics["frac_rama_favored"] = (
                    sum(1 for c in rama_classifications if c == ConformerClass.FAVORED)
                    / n_rama
                )
                metrics["n_rama_residues"] = n_rama

        return metrics

    def _score_dunbrack(self, atom_array, motif_res_ids, reverse_index_map) -> dict:
        """Score rotamers using the Dunbrack backbone-dependent library."""
        from blocks.metrics.utils.dihedrals import (
            RESIDUES_WITHOUT_ROTAMERS,
            calculate_phi_psi,
            get_rotamer_probability,
        )

        metrics = {}
        all_scores = []
        all_probs = []
        prob_by_res_name = {}

        starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
        residues = [
            atom_array[start:end] for start, end in zip(starts[:-1], starts[1:])
        ]

        for index, token in enumerate(residues):
            res_name = token.res_name[0]

            if (
                res_name in RESIDUES_WITHOUT_ROTAMERS
                or index == 0
                or index == len(residues) - 1
            ):
                continue

            if motif_res_ids is not None:
                chain_id = token.chain_id[0]
                res_id = token.res_id[0]
                if (chain_id, res_id) not in motif_res_ids:
                    continue

            phi, psi = calculate_phi_psi(
                residues[index - 1].coord,
                token.coord,
                residues[index + 1].coord,
            )
            out = get_rotamer_probability(
                token.coord, token.atom_name, res_name, phi, psi, return_score=True
            )
            if out is None:
                continue

            prob, score = out
            all_scores.append(score)
            all_probs.append(prob)
            prob_by_res_name.setdefault(res_name, []).append(prob)

            if self.per_residue_output:
                chain_id = token.chain_id[0]
                res_id = token.res_id[0]
                diffused_key = f"{chain_id}{res_id}"
                if reverse_index_map is not None:
                    original_key = reverse_index_map.get(diffused_key)
                    if original_key:
                        metrics[f"rotamer_score.{original_key}"] = float(score)
                    else:
                        logger.warning(
                            "No mapping for diffused residue %s, skipping "
                            "per-residue output",
                            diffused_key,
                        )
                else:
                    metrics[f"rotamer_score.{diffused_key}"] = float(score)

        if all_scores:
            metrics["mean_rotamer_score"] = float(np.mean(all_scores))
            metrics["n_rotamer_residues"] = len(all_scores)

            if self.detailed_output:
                for res_name, probs in prob_by_res_name.items():
                    metrics[f"mean_rotamer_prob.{res_name}"] = float(np.mean(probs))

        return metrics


def _get_residue_ids_from_mask(atom_array, mask):
    """Get set of (chain_id, res_id) tuples for residues covered by mask."""
    masked = atom_array[mask]
    if len(masked) == 0:
        return set()
    return set(zip(masked.chain_id, masked.res_id))


def _build_reverse_index_map(metadata):
    """Build reverse map from diffused residue ID to original residue ID.

    diffused_index_map: {"A1": "A90", "A2": "A91", ...}
    Returns: {"A90": "A1", "A91": "A2", ...}
    """
    raw = metadata.get("diffused_index_map")
    if not raw:
        return None
    if isinstance(raw, str):
        raw = json.loads(raw)
    return {v: k for k, v in raw.items()}
