"""Secondary-structure F1 / MCC of a design's motif, against reference pairs.

Implements T3 of the RNA motif-scaffolding benchmark. The 2D prediction comes
from RNet2 (RibonanzaNet2-SS), run on **sequence alone** - deriving 2D from the
design tool's own output would make the metric self-referential.

This block does not run RNet. It ingests the output CSV of an external RNet2
runner (a ``predicted_dbn`` dot-bracket column plus a key column, ``join_on``,
default ``design_path``) and computes the motif-restricted comparison, because
a full-sequence F1 has no reference when the scaffold is novel by construction.
Scoring is confined to the motif index space ``M`` (``n = |M|``):

    TP = |P & R|,  FP = |P - R|,  FN = |R - P|
    TN = n(n-1)/2 - TP - FP - FN

**The negative space is every unordered motif-internal pair**, stated here
because MCC is meaningless without declaring it and papers differ. A motif with
no predicted internal pairs yields ``mcc = 0.0`` with ``degenerate=true``,
never NaN.

The reference set must be **canonical pairs only** (Watson-Crick + G.U wobble).
Scoring noncanonical pairs punishes every predictor equally and tells you
nothing. Pass ``pseudoknot_pairs`` separately if the decoder in use cannot emit
them; they are excluded from ``R`` and reported.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from pydantic import Field

from blocks.metrics.rna_thermo import _split_resid, rna_sequence_and_index
from chaperone import InlineMetricBlock, JobArgs

logger = logging.getLogger(__name__)

# Extended dot-bracket layers the RNet runner emits.
_BRACKETS = [("(", ")"), ("[", "]"), ("{", "}"), ("<", ">")]


def parse_dbn(dbn: str) -> Set[Tuple[int, int]]:
    """Dot-bracket -> set of 0-based (i, j) pairs, i < j, all layers."""
    pairs, stacks = set(), {}
    for idx, ch in enumerate(dbn):
        for opener, closer in _BRACKETS:
            if ch == opener:
                stacks.setdefault(opener, []).append(idx)
                break
            if ch == closer:
                stack = stacks.get(opener)
                if stack:
                    pairs.add((stack.pop(), idx))
                break
        else:
            # ASCII letter layers: uppercase opens, lowercase closes, or the
            # reverse. Either polarity is accepted as long as it is consistent
            # within one structure, so treat the first-seen case as the opener.
            if ch.isalpha():
                key = ch.upper()
                stack = stacks.setdefault(key, [])
                if stack and stacks.get(f"_open_{key}") == ch.swapcase():
                    pairs.add((stack.pop(), idx))
                else:
                    stacks.setdefault(f"_open_{key}", ch)
                    if stacks[f"_open_{key}"] == ch:
                        stack.append(idx)
                    elif stack:
                        pairs.add((stack.pop(), idx))
    return {(min(i, j), max(i, j)) for i, j in pairs}


def f1_mcc(tp: int, fp: int, fn: int, tn: int) -> Tuple[float, Optional[float]]:
    """F1 and MCC. MCC is None when its denominator vanishes."""
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    denom_sq = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom_sq <= 0:
        return f1, None
    return f1, (tp * tn - fp * fn) / (denom_sq**0.5)


class RNASecondaryStructureF1Block(InlineMetricBlock):
    """Motif-restricted secondary-structure F1 / MCC against reference pairs."""

    _name = "rna_ss_f1_mcc"
    _display_name = "RNA SS F1/MCC"
    _description = (
        "Motif-restricted secondary-structure F1 and MCC of an RNet2 prediction "
        "against the reference's canonical pairs"
    )
    _resources = JobArgs(partition="cpu", mem_gb=4, cpus_per_task=1)

    predictions_csv: str = Field(
        ...,
        description=(
            "RNet2 runner output CSV. Must carry 'predicted_dbn' and the key "
            "column named by 'join_on'."
        ),
    )
    join_on: str = Field(
        "design_path",
        description=(
            "Column shared with the handler used to match a prediction to its "
            "design. Row position is NOT used - the runner preserves order, but "
            "a Chaperone chunk does not necessarily present rows in that order."
        ),
    )
    reference_pairs: List[List[str]] = Field(
        default_factory=list,
        description=(
            "Canonical (WC + G.U) reference pairs as reference residue ids, "
            'e.g. [["A5", "A10"], ["A12", "A25"]]. Resolved through '
            "diffused_index_map. Canonical only - see the module docstring."
        ),
    )
    pseudoknot_pairs: List[List[str]] = Field(
        default_factory=list,
        description="Reference pairs excluded from R because they are pseudoknotted.",
    )
    compare_to_vienna: bool = Field(
        True,
        description=(
            "Also report whole-sequence F1 between RNet and ViennaRNA's MFE on "
            "the same design. Cross-method 2D disagreement is a cheap red flag."
        ),
    )

    def setup(self, handler, sandbox_dir, chunk_id):
        path = Path(self.predictions_csv)
        with path.open() as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            raise ValueError(f"No rows in {path}")
        for required in ("predicted_dbn", self.join_on):
            if required not in rows[0]:
                raise ValueError(
                    f"{path} has no '{required}' column; found {list(rows[0])}"
                )
        by_key = {r[self.join_on]: r for r in rows}
        logger.info("Loaded %d RNet predictions from %s", len(by_key), path)
        return by_key

    def _motif_indices(self, loaded, res_index) -> Dict[str, int]:
        """Reference residue id -> 0-based sequence index in this design."""
        raw = loaded.metadata.get("diffused_index_map")
        index_map = (json.loads(raw) if isinstance(raw, str) else raw) if raw else {}
        out = {}
        for ref_id, design_id in index_map.items():
            chain, num = _split_resid(str(design_id))
            pos = res_index.get((chain, num))
            if pos is not None:
                out[str(ref_id)] = pos
        return out

    def compute_metrics(self, loaded, *, context=None, index: int = 0):
        key = str(loaded.path)
        record = context.get(key)
        metrics: Dict[str, object] = {}
        if record is None:
            return {"status": "na", "reason": f"no prediction for {key}"}
        if record.get("status") not in (None, "", "ok"):
            return {
                "status": "na",
                "reason": f"runner status={record.get('status')}: {record.get('error')}",
            }

        dbn = record.get("predicted_dbn") or ""
        if not dbn:
            return {"status": "na", "reason": "empty predicted_dbn"}

        seq, res_index = rna_sequence_and_index(loaded.atom_array)
        ref_to_design = self._motif_indices(loaded, res_index)
        motif = set(ref_to_design.values())
        n = len(motif)
        metrics["n_motif_residues"] = n
        metrics["negative_space"] = n * (n - 1) // 2

        reference = set()
        unresolved = 0
        for a, b in self.reference_pairs:
            i, j = ref_to_design.get(str(a)), ref_to_design.get(str(b))
            if i is None or j is None:
                unresolved += 1
                continue
            reference.add((min(i, j), max(i, j)))
        metrics["n_reference_pairs"] = len(reference)
        metrics["n_reference_pairs_unresolved"] = unresolved
        metrics["pk_pairs_excluded"] = len(self.pseudoknot_pairs)

        predicted = {p for p in parse_dbn(dbn) if p[0] in motif and p[1] in motif}
        metrics["n_predicted_motif_pairs"] = len(predicted)

        tp = len(predicted & reference)
        fp = len(predicted - reference)
        fn = len(reference - predicted)
        tn = metrics["negative_space"] - tp - fp - fn
        f1, mcc = f1_mcc(tp, fp, fn, tn)
        metrics.update({"tp": tp, "fp": fp, "fn": fn, "tn": tn, "f1": f1})
        # Guard rather than emit NaN: a motif with no predicted internal pairs
        # collapses the MCC denominator, which is a real outcome, not an error.
        metrics["mcc"] = 0.0 if mcc is None else mcc
        metrics["degenerate"] = mcc is None
        metrics["motif_pair_recall"] = tp / len(reference) if reference else None

        if self.compare_to_vienna and seq:
            import RNA

            vienna_dbn, _ = RNA.fold_compound(seq).mfe()
            v_pairs, r_pairs = parse_dbn(vienna_dbn), parse_dbn(dbn)
            v_tp = len(v_pairs & r_pairs)
            agree, _ = f1_mcc(v_tp, len(r_pairs - v_pairs), len(v_pairs - r_pairs), 0)
            metrics["rnet_vienna_agreement"] = agree

        metrics["status"] = "ok" if reference else "degenerate_reference"
        if not reference:
            metrics["reason"] = (
                "no canonical reference pairs internal to the motif; F1/MCC "
                "cannot discriminate designs for this target"
            )
        return metrics
