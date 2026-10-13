"""Motif-resolved oracle confidence: is the motif *rigidly placed* by the scaffold.

Whole-structure ``plddt``/``ptm`` cannot answer the question a motif-scaffolding
campaign asks. A design can fold confidently and still hold the catalytic site
in the wrong place, and a design can carry one floppy scaffold arm that drags
the global mean down while the site itself is pinned. Both are visible only
once confidence is split into motif and scaffold.

Two sources, both written by AF3-style oracles beside the model:

* **pLDDT** comes off the predicted structure's B-factor column, so it needs no
  side-car file and survives being moved.
* **PAE** comes from ``<model>_confidences.json`` (``pae``, ``token_chain_ids``,
  ``token_res_ids``). RNA and protein are one token per residue, so tokens map
  to residues directly; ligand and ion tokens are dropped by chain before any
  indexing. When the file is absent the pLDDT metrics are still reported and
  ``status`` says why the PAE block is missing.

``pae_motif_scaffold_mean`` is the headline number - it is the "is the motif
rigidly positioned by its scaffold" signal. ``pae_motif_internal_max`` catches
the other failure: a motif that is confidently placed but internally distorted.

Motif residues are addressed by their *reference* ids and resolved through
RFD3's ``diffused_index_map``, the same convention as ``reference_motif_rmsd``,
so one motif spec drives every block in the stack.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from pydantic import Field

from pipelines import InlineMetricBlock, JobArgs

logger = logging.getLogger(__name__)

LIGAND_PREFIX = "L:"


def _index_map(loaded) -> dict:
    raw = loaded.metadata.get("diffused_index_map")
    if not raw:
        return {}
    return json.loads(raw) if isinstance(raw, str) else raw


def _split_resid(resid: str):
    """'A27' -> ('A', 27). Chain ids may be multi-character."""
    i = len(resid)
    while i > 0 and resid[i - 1].isdigit():
        i -= 1
    return resid[:i], int(resid[i:])


def _residue_plddt(atom_array, chain: str) -> Dict[int, float]:
    """Mean B-factor (= pLDDT) per residue of one chain, in file order."""
    mask = atom_array.chain_id == chain
    res_ids = atom_array.res_id[mask]
    b_factors = atom_array.b_factor[mask]
    return {
        int(res): float(b_factors[res_ids == res].mean())
        for res in dict.fromkeys(res_ids.tolist())
    }


def confidences_path(model_path: str) -> Optional[Path]:
    """``..._model.cif`` -> ``..._confidences.json``, if the oracle wrote one."""
    for suffix in ("_model.cif", "_model.mmcif", ".cif"):
        if model_path.endswith(suffix):
            candidate = Path(model_path[: -len(suffix)] + "_confidences.json")
            if candidate.is_file():
                return candidate
    return None


class MotifConfidenceBlock(InlineMetricBlock):
    """Motif vs scaffold pLDDT and PAE of an oracle refold."""

    _name = "motif_confidence"
    _display_name = "Motif confidence"
    _description = "Motif/scaffold pLDDT and PAE split, via diffused_index_map"
    _creates_new_step = False
    _resources = JobArgs(partition="cpu", rows_per_job=500)

    which: List[int] = Field(
        default=[0, -1],
        description="Two indices into each row's structures list: the structure "
        "carrying diffused_index_map, and the prediction to score.",
    )
    residues: Optional[List[str]] = Field(
        default=None,
        description="Reference residue ids to treat as the motif, e.g. "
        "['A10', 'A11']. Defaults to every polymer entry in diffused_index_map.",
    )
    chain: Optional[str] = Field(
        default=None,
        description="Designed chain to score. Defaults to the chain the mapped "
        "motif residues land in.",
    )

    def get_data_kwargs(self) -> dict:
        return {"which": self.which, "load": True}

    def compute_metrics(self, loaded, *, context=None, index: int = 0):
        source, prediction = loaded
        metrics = {
            "status": "ok",
            "plddt_motif": None,
            "plddt_scaffold": None,
            "plddt_motif_min": None,
            "pae_motif_internal_max": None,
            "pae_motif_scaffold_mean": None,
            "pae_motif_scaffold_p90": None,
            "n_motif_residues": 0,
        }

        index_map = _index_map(source)
        if not index_map:
            metrics["status"] = "na"
            metrics["reason"] = "no diffused_index_map"
            return metrics

        wanted = set(self.residues) if self.residues else None
        design_ids = [
            str(value)
            for key, value in index_map.items()
            if not key.startswith(LIGAND_PREFIX) and (wanted is None or key in wanted)
        ]
        motif = {}
        for design_id in design_ids:
            chain, res_id = _split_resid(design_id)
            motif.setdefault(chain, set()).add(res_id)
        if not motif:
            metrics["status"] = "na"
            metrics["reason"] = "index map has no polymer entries"
            return metrics

        chain = self.chain or max(motif, key=lambda c: len(motif[c]))
        motif_residues = motif.get(chain, set())
        metrics["n_motif_residues"] = len(motif_residues)

        plddt = _residue_plddt(prediction.atom_array, chain)
        motif_plddt = [v for res, v in plddt.items() if res in motif_residues]
        scaffold_plddt = [v for res, v in plddt.items() if res not in motif_residues]
        if motif_plddt:
            metrics["plddt_motif"] = float(np.mean(motif_plddt))
            metrics["plddt_motif_min"] = float(np.min(motif_plddt))
        if scaffold_plddt:
            metrics["plddt_scaffold"] = float(np.mean(scaffold_plddt))

        path = confidences_path(str(prediction.path))
        if path is None:
            metrics["status"] = "partial"
            metrics["reason"] = "no _confidences.json beside the prediction"
            return metrics

        data = json.loads(path.read_text())
        pae = np.asarray(data["pae"], dtype=float)
        token_chains = data["token_chain_ids"]
        token_residues = data["token_res_ids"]
        motif_tokens = [
            i
            for i, (c, r) in enumerate(zip(token_chains, token_residues))
            if c == chain and int(r) in motif_residues
        ]
        scaffold_tokens = [
            i
            for i, (c, r) in enumerate(zip(token_chains, token_residues))
            if c == chain and int(r) not in motif_residues
        ]
        if not motif_tokens:
            metrics["status"] = "partial"
            metrics["reason"] = f"no tokens for chain {chain} motif residues"
            return metrics

        internal = pae[np.ix_(motif_tokens, motif_tokens)]
        metrics["pae_motif_internal_max"] = float(internal.max())
        if scaffold_tokens:
            cross = np.concatenate(
                [
                    pae[np.ix_(motif_tokens, scaffold_tokens)].ravel(),
                    pae[np.ix_(scaffold_tokens, motif_tokens)].ravel(),
                ]
            )
            metrics["pae_motif_scaffold_mean"] = float(cross.mean())
            metrics["pae_motif_scaffold_p90"] = float(np.percentile(cross, 90))
        return metrics
