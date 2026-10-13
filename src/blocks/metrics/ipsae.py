"""ipSAE — interface predicted Score from Aligned Errors.

Per Park & Cong (https://github.com/flyark42/ipsae): a per-interface confidence
score derived from a structure prediction's full per-token PAE matrix. Higher is
better (closer to 1 = high-confidence interface).

Reads the full PAE matrix from a sibling ``*_confidences.json`` next to the
prediction's ``*_model.cif`` (the format AlphaFold3 emits; RF3 uses the same
layout). For each binder<->target chain pair, applies the pTM-style scaling
``s = 1 / (1 + (PAE / d0)^2)`` with ``d0 = max(1, 1.24*(L-15)^(1/3) - 1.8)``,
masks pairs above ``pae_cutoff``, and reports the max-row mean — taken
direction-symmetrically as ``max(a→b, b→a)``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import ClassVar, List, Optional

import numpy as np
from pydantic import Field

from chaperone import InlineMetricBlock, JobArgs

logger = logging.getLogger(__name__)


def _ptm_d0(length: int) -> float:
    if length < 27:
        return 1.0
    return max(1.0, 1.24 * (length - 15) ** (1.0 / 3.0) - 1.8)


def _ipsae_directional(
    pae: np.ndarray,
    src_idx: np.ndarray,
    tgt_idx: np.ndarray,
    pae_cutoff: float,
) -> float:
    """Compute ipSAE for the direction src -> tgt. Returns 0.0 if no valid pairs."""
    if len(src_idx) == 0 or len(tgt_idx) == 0:
        return 0.0
    sub = pae[np.ix_(src_idx, tgt_idx)]
    valid = sub < pae_cutoff
    n_valid_tgt = int(valid.any(axis=0).sum())
    if n_valid_tgt < 2:
        return 0.0
    d0 = _ptm_d0(n_valid_tgt)
    s = 1.0 / (1.0 + (sub / d0) ** 2)
    s_masked = s * valid
    s_per_src = s_masked.sum(axis=1) / n_valid_tgt
    return float(s_per_src.max())


def compute_ipsae(
    pae: np.ndarray,
    token_chain_ids: List[str],
    binder_chain: str,
    target_chains: Optional[List[str]],
    pae_cutoff: float,
) -> dict:
    """Compute ipSAE for binder<->target chain pairs from a full PAE matrix.

    Returns a dict with:
        - ipsae: min over per-pair ipSAEs (worst interface; the value to maximize)
        - ipsae_max: max over per-pair ipSAEs (best interface)
        - ipsae_per_chain: {"<binder>_<target>": float}
        - min_pae_interface: min PAE value across all binder<->target submatrix entries
    """
    n = len(token_chain_ids)
    if pae.shape != (n, n):
        raise ValueError(
            f"PAE shape {pae.shape} does not match token_chain_ids length {n}"
        )

    chain_to_idx: dict[str, list[int]] = {}
    for i, c in enumerate(token_chain_ids):
        chain_to_idx.setdefault(str(c), []).append(i)
    chain_to_idx_np = {c: np.asarray(ix) for c, ix in chain_to_idx.items()}

    if binder_chain not in chain_to_idx_np:
        raise ValueError(
            f"binder_chain={binder_chain!r} not found among chains {sorted(chain_to_idx_np)}"
        )
    if target_chains is None:
        target_chains = [c for c in chain_to_idx_np if c != binder_chain]
    target_chains = [c for c in target_chains if c in chain_to_idx_np]
    if not target_chains:
        return {
            "ipsae": None,
            "ipsae_max": None,
            "ipsae_per_chain": {},
            "min_pae_interface": None,
        }

    binder_idx = chain_to_idx_np[binder_chain]
    per_chain: dict[str, float] = {}
    min_pae = float("inf")
    for tc in target_chains:
        tc_idx = chain_to_idx_np[tc]
        a_to_b = _ipsae_directional(pae, binder_idx, tc_idx, pae_cutoff)
        b_to_a = _ipsae_directional(pae, tc_idx, binder_idx, pae_cutoff)
        per_chain[f"{binder_chain}_{tc}"] = max(a_to_b, b_to_a)
        sub = pae[np.ix_(binder_idx, tc_idx)]
        sub_rev = pae[np.ix_(tc_idx, binder_idx)]
        m = float(min(sub.min(initial=np.inf), sub_rev.min(initial=np.inf)))
        if m < min_pae:
            min_pae = m

    return {
        "ipsae": float(min(per_chain.values())),
        "ipsae_max": float(max(per_chain.values())),
        "ipsae_per_chain": per_chain,
        "min_pae_interface": float(min_pae) if min_pae != float("inf") else None,
    }


def _resolve_confidences_path(structure_path: str) -> Optional[Path]:
    """Map a prediction's structure file to its sibling confidences JSON.

    Handles AF3 (``*_model.cif`` -> ``*_confidences.json``) and RF3 (same convention).
    """
    p = Path(structure_path)
    name = p.name
    candidates = []
    for suffix in ("_model.cif", "_model.pdb"):
        if name.endswith(suffix):
            candidates.append(p.with_name(name[: -len(suffix)] + "_confidences.json"))
    candidates.append(p.with_suffix("").with_suffix(".confidences.json"))
    candidates.append(p.parent / "_confidences.json")
    for c in candidates:
        if c.exists():
            return c
    return None


class IpSAEBlock(InlineMetricBlock):
    """Compute ipSAE between a binder chain and one or more target chains.

    Reads the full per-token PAE matrix from the AF3/RF3 ``*_confidences.json``
    sibling of the last structure in the lineage. Score follows Park & Cong's
    canonical formulation; reports the worst per-pair score (the value to
    maximize) plus per-pair detail and the minimum interface PAE.
    """

    _name = "ipsae"
    _display_name = "ipSAE"
    _description = "Interface predicted Score from Aligned Errors"
    _creates_new_step = False
    _resources: ClassVar[JobArgs] = JobArgs(partition="cpu")

    binder_chain: str = Field(
        default="A",
        description="Chain ID of the designed binder. Default 'A' matches RFD3 output.",
    )
    target_chains: Optional[List[str]] = Field(
        default=None,
        description=(
            "Chain IDs of the target. If None, all non-binder chains in the "
            "prediction are used."
        ),
    )
    pae_cutoff: float = Field(
        default=10.0,
        description="PAE threshold for inclusion in the score (Å). Default 10.0 per Park & Cong.",
    )

    def get_data_kwargs(self):
        return {"which": "last", "load": False}

    def compute_metrics(self, loaded, *, context=None, index=0):
        return self._score_one(loaded.path)

    def _score_one(self, structure_path: str) -> dict:
        conf_path = _resolve_confidences_path(structure_path)
        if conf_path is None:
            logger.warning(
                "ipsae: no _confidences.json sibling found for %s — emitting None",
                structure_path,
            )
            return {
                "ipsae": None,
                "ipsae_max": None,
                "ipsae_per_chain": {},
                "min_pae_interface": None,
            }
        try:
            with open(conf_path) as f:
                data = json.load(f)
        except Exception as e:
            logger.error("ipsae: failed to read %s: %s", conf_path, e)
            return {
                "ipsae": None,
                "ipsae_max": None,
                "ipsae_per_chain": {},
                "min_pae_interface": None,
            }

        pae_raw = data.get("pae")
        token_chain_ids = data.get("token_chain_ids")
        if pae_raw is None or token_chain_ids is None:
            logger.warning(
                "ipsae: %s missing 'pae' or 'token_chain_ids'; emitting None",
                conf_path,
            )
            return {
                "ipsae": None,
                "ipsae_max": None,
                "ipsae_per_chain": {},
                "min_pae_interface": None,
            }

        pae = np.asarray(pae_raw, dtype=float)
        try:
            return compute_ipsae(
                pae=pae,
                token_chain_ids=list(token_chain_ids),
                binder_chain=self.binder_chain,
                target_chains=self.target_chains,
                pae_cutoff=self.pae_cutoff,
            )
        except Exception as e:
            logger.error("ipsae: scoring failed for %s: %s", structure_path, e)
            return {
                "ipsae": None,
                "ipsae_max": None,
                "ipsae_per_chain": {},
                "min_pae_interface": None,
            }
