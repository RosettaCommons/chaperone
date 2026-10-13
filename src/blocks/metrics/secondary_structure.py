"""Secondary-structure annotation block.

Reports SS composition (global + pocket-around-ligand) using biotite's P-SEA-based
``annotate_sse``, plus two disorder views: loop fraction (geometric) and low-pLDDT
fraction (when the input has b_factor populated, as AF3/Boltz/RF3 CIFs do).
"""

import logging
from typing import List, Optional

import biotite.structure as struc
import numpy as np
from pydantic import Field
from scipy.spatial.distance import cdist

from pipelines import InlineMetricBlock

logger = logging.getLogger(__name__)


_SSE_MAP = {"a": "H", "b": "E", "c": "L"}


class SecondaryStructureBlock(InlineMetricBlock):
    """Per-fold SS composition + loop/pLDDT disorder, globally and around the ligand."""

    _name = "secondary_structure"
    _display_name = "Secondary Structure"
    _description = "Per-fold SS composition + loop/pLDDT disorder, globally and pocket-around-ligand"
    _creates_new_step = False

    pocket_cutoff: float = Field(
        default=8.0,
        description="Å cutoff for a residue to be considered in the pocket "
        "(any non-H protein atom within this distance of any ligand atom).",
    )
    plddt_threshold: float = Field(
        default=70.0,
        description="Per-residue mean b_factor below this counts as low-confidence "
        "(pLDDT semantics; b_factor in AF3/Boltz/RF3 CIFs carries pLDDT 0-100).",
    )
    plddt_min_signal: float = Field(
        default=50.0,
        description="Auto-scaling rules for b_factor → pLDDT (0-100):\n"
        "  - max(bf) ≤ 1.5         → assume 0-1 scale (RF3); multiply by 100.\n"
        "  - max(bf) ≥ plddt_min_signal → assume 0-100 scale (AF3/Boltz); use as-is.\n"
        "  - otherwise (crystal B-factor noise, typical max 5-40) → pLDDT metrics null.\n"
        "Default 50 admits AF3/Boltz (typical max >60) and rejects crystal B-factor.",
    )
    verbose: bool = Field(
        default=False,
        description="When True, also emit `ss_string`, segment topology, and a "
        "`residue_records` list with per-residue {residue_id, resname, ss, plddt, in_pocket}.",
    )
    chains: Optional[List[str]] = Field(
        default=None,
        description="Restrict SS computation to these chain IDs (e.g. ['B']). "
        "When None, all chains are included.",
    )

    def compute_metrics(self, loaded, *, context=None, index=0):
        try:
            atom_array = loaded.atom_array
            if self.chains is not None:
                atom_array = atom_array[np.isin(atom_array.chain_id, self.chains)]
            metrics = _compute_ss_metrics(
                atom_array,
                pocket_cutoff=self.pocket_cutoff,
                plddt_threshold=self.plddt_threshold,
                plddt_min_signal=self.plddt_min_signal,
                verbose=self.verbose,
            )
            logger.info(
                "secondary_structure for %s: n_residues=%d helix=%.2f loop=%.2f pocket_n=%s",
                loaded.path,
                metrics.get("n_residues", 0),
                metrics.get("fraction_helix") or 0.0,
                metrics.get("fraction_loop") or 0.0,
                metrics.get("pocket_n_residues"),
            )
        except Exception as e:
            logger.error(
                "secondary_structure failed for %s: %s",
                loaded.path,
                e,
                exc_info=True,
            )
            raise

        return metrics


def _compute_ss_metrics(
    atom_array,
    *,
    pocket_cutoff: float,
    plddt_threshold: float,
    plddt_min_signal: float,
    verbose: bool,
) -> dict:
    """Compute SS composition + disorder metrics for one structure."""
    aa_mask = struc.filter_amino_acids(atom_array)
    protein = atom_array[aa_mask]

    if len(protein) == 0:
        return _empty_metrics(verbose=verbose)

    # Per-residue SS labels (biotite alphabet: 'a'/'b'/'c'/'')
    sse = struc.annotate_sse(protein)
    ss_labels = np.array([_SSE_MAP.get(c, "-") for c in sse])

    # Per-residue identity and b_factor
    res_starts = struc.get_residue_starts(protein)
    chain_ids = protein.chain_id[res_starts]
    res_ids = protein.res_id[res_starts]
    res_names = protein.res_name[res_starts]
    res_ends = np.append(res_starts[1:], len(protein))
    # Crystal/design CIFs are allowed to omit the optional b_factor
    # annotation entirely.  Biotite raises AttributeError for an absent
    # annotation category, so check through getattr before applying the
    # no-pLDDT fallback.
    b_factor_annotation = getattr(protein, "b_factor", None)
    b_factor = (
        b_factor_annotation
        if b_factor_annotation is not None
        else np.zeros(len(protein))
    )
    mean_bf = np.array(
        [float(np.mean(b_factor[s:e])) for s, e in zip(res_starts, res_ends)]
    )

    n_residues = len(res_starts)
    # SSE may return fewer labels than residues if some have no CA — align defensively
    if len(ss_labels) != n_residues:
        # Pad / trim to match
        if len(ss_labels) < n_residues:
            ss_labels = np.concatenate(
                [ss_labels, np.array(["-"] * (n_residues - len(ss_labels)))]
            )
        else:
            ss_labels = ss_labels[:n_residues]

    residue_ids = np.array([f"{c}{r}" for c, r in zip(chain_ids, res_ids)])

    # Pocket mask via cdist on non-H atoms
    pocket_res_mask = _pocket_residue_mask(
        atom_array, aa_mask, chain_ids, res_ids, pocket_cutoff
    )

    # pLDDT signal detection + auto-scaling to 0-100
    bf_max_raw = float(np.nanmax(b_factor)) if len(b_factor) else 0.0
    if 0 < bf_max_raw <= 1.5:
        mean_bf = mean_bf * 100.0  # RF3 normalized pLDDT
        has_plddt = True
    elif bf_max_raw >= plddt_min_signal:
        has_plddt = True  # AF3/Boltz native 0-100
    else:
        has_plddt = False  # crystal B-factor noise or zero

    metrics: dict = {
        "n_residues": int(n_residues),
        **_composition_block(ss_labels, prefix=""),
    }

    if pocket_res_mask is None:
        metrics["pocket_n_residues"] = None
        metrics["pocket_fraction_helix"] = None
        metrics["pocket_fraction_sheet"] = None
        metrics["pocket_fraction_loop"] = None
        metrics["pocket_residue_ids"] = None
    else:
        pocket_ss = ss_labels[pocket_res_mask]
        metrics["pocket_n_residues"] = int(pocket_ss.shape[0])
        metrics.update(_composition_block(pocket_ss, prefix="pocket_"))
        metrics["pocket_residue_ids"] = residue_ids[pocket_res_mask].tolist()

    # pLDDT-based disorder
    if has_plddt:
        low_mask = mean_bf < plddt_threshold
        metrics["mean_plddt"] = float(np.mean(mean_bf))
        metrics["n_low_plddt"] = int(low_mask.sum())
        metrics["fraction_low_plddt"] = float(low_mask.mean())
        if pocket_res_mask is None:
            metrics["pocket_mean_plddt"] = None
            metrics["pocket_n_low_plddt"] = None
            metrics["pocket_fraction_low_plddt"] = None
        else:
            pocket_bf = mean_bf[pocket_res_mask]
            metrics["pocket_mean_plddt"] = (
                float(np.mean(pocket_bf)) if pocket_bf.size else None
            )
            pocket_low = pocket_bf < plddt_threshold
            metrics["pocket_n_low_plddt"] = int(pocket_low.sum())
            metrics["pocket_fraction_low_plddt"] = (
                float(pocket_low.mean()) if pocket_low.size else None
            )
        if verbose:
            metrics["low_plddt_residues"] = residue_ids[low_mask].tolist()
            metrics["pocket_low_plddt_residues"] = (
                residue_ids[pocket_res_mask][pocket_low].tolist()
                if pocket_res_mask is not None
                else None
            )
    else:
        for k in (
            "mean_plddt",
            "n_low_plddt",
            "fraction_low_plddt",
            "pocket_mean_plddt",
            "pocket_n_low_plddt",
            "pocket_fraction_low_plddt",
        ):
            metrics[k] = None
        if verbose:
            metrics["low_plddt_residues"] = None
            metrics["pocket_low_plddt_residues"] = None

    if verbose:
        metrics["ss_string"] = "".join(ss_labels.tolist())
        metrics["segments"] = _count_segments(ss_labels)
        metrics["residue_records"] = [
            {
                "residue_id": rid,
                "resname": rn,
                "ss": ss,
                "plddt": float(bf) if has_plddt else None,
                "in_pocket": bool(pocket_res_mask[i])
                if pocket_res_mask is not None
                else False,
            }
            for i, (rid, rn, ss, bf) in enumerate(
                zip(residue_ids, res_names, ss_labels, mean_bf)
            )
        ]

    return metrics


def _composition_block(ss_labels: np.ndarray, *, prefix: str) -> dict:
    n = int(ss_labels.shape[0])
    if n == 0:
        return {
            f"{prefix}fraction_helix": None,
            f"{prefix}fraction_sheet": None,
            f"{prefix}fraction_loop": None,
            f"{prefix}n_helix": 0,
            f"{prefix}n_sheet": 0,
            f"{prefix}n_loop": 0,
        }
    n_h = int((ss_labels == "H").sum())
    n_e = int((ss_labels == "E").sum())
    n_l = int((ss_labels == "L").sum())
    return {
        f"{prefix}n_helix": n_h,
        f"{prefix}n_sheet": n_e,
        f"{prefix}n_loop": n_l,
        f"{prefix}fraction_helix": n_h / n,
        f"{prefix}fraction_sheet": n_e / n,
        f"{prefix}fraction_loop": n_l / n,
    }


def _count_segments(ss_labels: np.ndarray) -> str:
    """Return a string like '3H,2E,4L' counting contiguous segments per SS type."""
    if len(ss_labels) == 0:
        return ""
    counts = {"H": 0, "E": 0, "L": 0}
    prev = None
    for s in ss_labels:
        if s in counts and s != prev:
            counts[s] += 1
        prev = s
    return ",".join(f"{v}{k}" for k, v in counts.items() if v > 0)


def _pocket_residue_mask(
    atom_array,
    aa_mask: np.ndarray,
    chain_ids: np.ndarray,
    res_ids: np.ndarray,
    cutoff: float,
):
    """Return a boolean mask over residues (aligned with chain_ids/res_ids) marking
    residues with any non-H protein atom within ``cutoff`` Å of any ligand atom.
    Returns None if the structure has no ligand atoms.
    """
    from blocks.utils.geometry import get_heavy_mask, get_ligand_mask

    is_lig = get_ligand_mask(atom_array)
    if is_lig is None:
        return None
    if not is_lig.any():
        return None

    # Restrict to non-H protein atoms vs ligand heavy atoms
    non_h = get_heavy_mask(atom_array)
    prot_atom_mask = aa_mask & non_h & (~is_lig)
    lig_atom_mask = is_lig & non_h

    if not prot_atom_mask.any() or not lig_atom_mask.any():
        return np.zeros(len(chain_ids), dtype=bool)

    prot_coords = atom_array.coord[prot_atom_mask]
    lig_coords = atom_array.coord[lig_atom_mask]

    # Per-protein-atom min distance to any ligand atom
    dmat = cdist(prot_coords, lig_coords)
    atom_min = dmat.min(axis=1)
    within = atom_min <= cutoff

    # Map atom hits back to residues
    hit_chain = atom_array.chain_id[prot_atom_mask][within]
    hit_resid = atom_array.res_id[prot_atom_mask][within]
    hit_set = set(zip(hit_chain.tolist(), hit_resid.tolist()))
    return np.array(
        [(c, int(r)) in hit_set for c, r in zip(chain_ids, res_ids)], dtype=bool
    )


def _empty_metrics(*, verbose: bool) -> dict:
    out = {
        "n_residues": 0,
        "n_helix": 0,
        "n_sheet": 0,
        "n_loop": 0,
        "fraction_helix": None,
        "fraction_sheet": None,
        "fraction_loop": None,
        "pocket_n_residues": None,
        "pocket_fraction_helix": None,
        "pocket_fraction_sheet": None,
        "pocket_fraction_loop": None,
        "pocket_residue_ids": None,
        "mean_plddt": None,
        "n_low_plddt": None,
        "fraction_low_plddt": None,
        "pocket_mean_plddt": None,
        "pocket_n_low_plddt": None,
        "pocket_fraction_low_plddt": None,
    }
    if verbose:
        out.update(
            {
                "ss_string": "",
                "segments": "",
                "residue_records": [],
                "low_plddt_residues": None,
                "pocket_low_plddt_residues": None,
            }
        )
    return out
