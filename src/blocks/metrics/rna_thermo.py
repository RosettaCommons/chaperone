"""ViennaRNA thermodynamic scoring of a designed RNA against a required pair set.

Answers a question no structure predictor can: does the designed *sequence*
thermodynamically want to form the motif's base pairs? The quantity is the
Boltzmann weight of the constrained ensemble,

    p_motif = exp(-(G_constrained - G_unconstrained) / RT)

which is exact and assumes nothing about the rest of the fold - the scaffold is
novel, so its target secondary structure is unknown by construction.

Pairs are addressed by reference residue id and resolved through RFD3's
``diffused_index_map``, the same convention as ``reference_motif_rmsd``, so the
same target spec drives both. Raw 0-based index pairs are accepted too.

ViennaRNA's model cannot represent pseudoknots. Declare those pairs in
``pseudoknot_pairs``; they are excluded from the constraint and reported, and
the block returns ``status="partial"`` rather than silently scoring a subset.
Vienna also has no Mg2+ term even though many structured RNAs are metal-dependent,
so this metric ranks designs *within* a target and says nothing absolute.
"""

import json
import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from pydantic import Field

from pipelines import InlineMetricBlock, JobArgs

logger = logging.getLogger(__name__)

# kcal/mol at 37 C, the temperature ViennaRNA's default parameter set is fit at.
RT_37C = 0.0019872 * 310.15

_RNA_RESIDUES = {"A": "A", "C": "C", "G": "G", "U": "U", "T": "U"}


def _index_map(loaded) -> dict:
    raw = loaded.metadata.get("diffused_index_map")
    if not raw:
        return {}
    return json.loads(raw) if isinstance(raw, str) else raw


def _split_resid(resid: str) -> Tuple[str, int]:
    """'A27' -> ('A', 27). Chain ids may be multi-character."""
    i = len(resid)
    while i > 0 and resid[i - 1].isdigit():
        i -= 1
    return resid[:i], int(resid[i:])


def rna_sequence_and_index(atom_array, chain: Optional[str] = None):
    """Extract an RNA sequence plus a (chain, res_id) -> 0-based index map.

    Residues are taken in file order, which for a designed chain is 5'->3'.
    Non-RNA components (ions, ligands, protein) are skipped rather than mapped
    to N, so the returned indices always address a valid Vienna sequence.
    """
    seen, seq, index = set(), [], {}
    for chain_id, res_id, res_name in zip(
        atom_array.chain_id, atom_array.res_id, atom_array.res_name
    ):
        key = (str(chain_id), int(res_id))
        if key in seen:
            continue
        letter = _RNA_RESIDUES.get(str(res_name).strip().upper())
        if letter is None:
            continue
        if chain is not None and str(chain_id) != chain:
            continue
        seen.add(key)
        index[key] = len(seq)
        seq.append(letter)
    return "".join(seq), index


# ViennaRNA's model needs a minimum hairpin loop of 3, so residues closer than
# this cannot pair. hc_add_bp SILENTLY DROPS such a constraint, and the
# unconstrained and constrained partition functions then come out identical --
# which reads as p_motif = 1.0, the strongest possible score, for a pair the
# model never even considered. Verified on ViennaRNA 2.7.2: forcing (3, 4) in
# GGGGAAAACCCC returns p = 1.0 with dG = 0 exactly.
MIN_PAIR_SEPARATION = 4

CANONICAL_PAIRS = {
    ("A", "U"),
    ("U", "A"),
    ("G", "C"),
    ("C", "G"),
    ("G", "U"),
    ("U", "G"),
}


def motif_boltzmann_probability(seq: str, required_pairs: List[Tuple[int, int]]):
    """Boltzmann probability of the ensemble constrained to contain all pairs.

    ``required_pairs`` are 0-based (i, j) in design coordinates. Returns
    ``(p, g_unconstrained, g_constrained, dropped)`` where ``dropped`` lists
    pairs ViennaRNA could not apply. ``p`` is capped at 1.0 because a
    constrained ensemble can only ever be a subset of the unconstrained one and
    tiny numerical excesses above 1 are noise, not signal.
    """
    import RNA

    fc0 = RNA.fold_compound(seq)
    _, g0 = fc0.pf()

    dropped, applied = [], []
    for i, j in required_pairs:
        if abs(i - j) < MIN_PAIR_SEPARATION:
            dropped.append((i, j, "separation"))
            continue
        if (seq[i], seq[j]) not in CANONICAL_PAIRS:
            # Vienna scores only canonical pairs. Forcing a noncanonical one
            # does not model it, it just makes the ensemble near-impossible.
            dropped.append((i, j, "noncanonical"))
            continue
        applied.append((i, j))

    if not applied:
        return None, float(g0), None, dropped

    fc1 = RNA.fold_compound(seq)
    for i, j in applied:
        fc1.hc_add_bp(
            i + 1,
            j + 1,
            RNA.CONSTRAINT_CONTEXT_ALL_LOOPS | RNA.CONSTRAINT_CONTEXT_ENFORCE,
        )
    _, g1 = fc1.pf()

    return min(1.0, math.exp(-(g1 - g0) / RT_37C)), float(g0), float(g1), dropped


def _pair_probabilities(seq: str):
    """Upper-triangular base-pair probability matrix, 0-based, from one pf run."""
    import RNA

    fc = RNA.fold_compound(seq)
    fc.pf()
    raw = fc.bpp()  # 1-based, (n+1) x (n+1)
    n = len(seq)
    bpp = np.zeros((n, n))
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            p = raw[i][j]
            if p:
                bpp[i - 1, j - 1] = p
    return bpp


class RNAThermoBlock(InlineMetricBlock):
    """ViennaRNA Boltzmann scoring of a required base-pair set."""

    _name = "rna_thermo"
    _display_name = "RNA thermodynamics"
    _description = (
        "ViennaRNA Boltzmann probability, pair probabilities and ensemble defect "
        "for a required motif pair set"
    )
    _resources = JobArgs(partition="cpu", mem_gb=4, cpus_per_task=1)

    required_pairs: List[List[str]] = Field(
        default_factory=list,
        description=(
            "Base pairs the motif requires, as reference residue ids resolved "
            'through diffused_index_map, e.g. [["A10", "A24"], ["B4", "B5"]]. '
            "Integer strings are treated as 0-based design indices instead."
        ),
    )
    pseudoknot_pairs: List[List[str]] = Field(
        default_factory=list,
        description=(
            "Pairs that form pseudoknots. Excluded from the constraint because "
            "ViennaRNA's model cannot represent them; reported and flagged."
        ),
    )
    chain: Optional[str] = Field(
        None, description="Restrict to one chain id. None uses every RNA residue."
    )

    def _resolve(self, pairs, index_map, res_index) -> Tuple[list, list]:
        """Map declared pairs to 0-based design indices, keeping the unresolved."""
        resolved, missing = [], []
        for pair in pairs:
            if len(pair) != 2:
                raise ValueError(f"pair must have exactly two members, got {pair!r}")
            idx = []
            for ref_id in pair:
                ref_id = str(ref_id)
                if ref_id.lstrip("-").isdigit():
                    idx.append(int(ref_id))
                    continue
                design_id = str(index_map.get(ref_id, ref_id))
                chain, num = _split_resid(design_id)
                pos = res_index.get((chain, num))
                if pos is None:
                    idx = None
                    break
                idx.append(pos)
            if idx is None:
                missing.append(pair)
            else:
                resolved.append((min(idx), max(idx)))
        return resolved, missing

    def compute_metrics(self, loaded, *, context=None, index: int = 0):
        seq, res_index = rna_sequence_and_index(loaded.atom_array, self.chain)
        if not seq:
            logger.warning("No RNA residues found; skipping row %d", index)
            return {"status": "na", "reason": "no RNA residues"}

        index_map = _index_map(loaded)
        pairs, missing = self._resolve(self.required_pairs, index_map, res_index)
        pk_pairs, _ = self._resolve(self.pseudoknot_pairs, index_map, res_index)

        metrics: Dict[str, object] = {
            "sequence_length": len(seq),
            "n_pairs_scored": len(pairs),
            "n_pairs_unresolved": len(missing),
            "pk_pairs_excluded": len(self.pseudoknot_pairs),
        }

        if any(i == j for i, j in pairs):
            raise ValueError("a base pair cannot pair a residue with itself")
        if not pairs:
            metrics["status"] = "na"
            metrics["reason"] = "no required pairs resolved"
            return metrics

        p_motif, g0, g1, dropped = motif_boltzmann_probability(seq, pairs)
        metrics["n_pairs_dropped"] = len(dropped)
        metrics["dropped_reasons"] = ",".join(sorted({d[2] for d in dropped}))
        if p_motif is None:
            metrics["status"] = "na"
            metrics["reason"] = (
                "every required pair is outside ViennaRNA's model "
                f"({metrics['dropped_reasons']}); no thermodynamic score is "
                "defined for it"
            )
            return metrics
        metrics.update(
            {
                "p_motif": p_motif,
                "g_unconstrained": g0,
                "g_constrained": g1,
                "dg_constraint": g1 - g0,
            }
        )

        bpp = _pair_probabilities(seq)
        pair_probs = [float(bpp[i, j]) for i, j in pairs]
        # The mean hides the usual failure mode, which is one weak pair.
        metrics["bpp_min_motif"] = min(pair_probs)
        metrics["bpp_mean_motif"] = float(np.mean(pair_probs))

        # Expected number of motif nucleotides not in their required partner.
        motif_positions = sorted({i for pair in pairs for i in pair})
        paired_prob = bpp + bpp.T
        correct = 0.0
        for i, j in pairs:
            correct += 2 * float(bpp[i, j])
        metrics["ensemble_defect_motif"] = len(motif_positions) - correct

        # Shannon entropy of each motif position's pairing distribution; low
        # means the position knows what it wants to pair with.
        entropies = []
        for pos in motif_positions:
            probs = list(paired_prob[pos][paired_prob[pos] > 0])
            unpaired = max(0.0, 1.0 - sum(probs))
            if unpaired > 0:
                probs.append(unpaired)
            entropies.append(-sum(p * math.log(p) for p in probs if p > 0))
        metrics["positional_entropy_motif"] = float(np.mean(entropies))

        import RNA

        mfe_struct, mfe = RNA.fold_compound(seq).mfe()
        metrics["mfe"] = float(mfe)
        metrics["mfe_structure"] = mfe_struct

        if dropped:
            metrics["status"] = "partial"
            metrics["reason"] = (
                f"{len(dropped)} pair(s) outside ViennaRNA's model "
                f"({metrics['dropped_reasons']})"
            )
        elif self.pseudoknot_pairs:
            metrics["status"] = "partial"
            metrics["reason"] = "pseudoknot pairs excluded from the Vienna model"
        elif missing:
            metrics["status"] = "partial"
            metrics["reason"] = f"{len(missing)} pair(s) unresolved in this design"
        else:
            metrics["status"] = "ok"
        return metrics
