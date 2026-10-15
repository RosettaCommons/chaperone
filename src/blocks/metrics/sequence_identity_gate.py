"""Hard novelty gate: global sequence identity against a set of natural references.

Determines whether a benchmark measures design or recall. A design that copies
a natural RNA can score well on structural metrics because the predictor may
have memorized it too.

Two details decide whether this gate works at all:

* **The motif is excluded from the identity calculation.** Motif identity is
  100% by construction, so scoring the full design makes every submission look
  similarly identical to the native and the gate becomes noise. Identity is
  computed over the scaffold region only - every residue not in the motif map.
* **Global (Needleman-Wunsch) alignment, gaps in the denominator.** Local
  identity lets a design share a long stretch with a natural homolog and still
  report a low number because the alignment was short.

``threshold`` is the gate value and is meant to be set per campaign; the 0.40
default is the benchmark guide's. ``passed=False`` is intended to zero a
design's score outright rather than contribute a penalty term - a soft penalty
gets traded against structural gains and the gate stops working.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import Field

from chaperone import InlineMetricBlock, JobArgs

logger = logging.getLogger(__name__)

_RNA_RESIDUES = {"A": "A", "C": "C", "G": "G", "U": "U", "T": "U"}


def read_fasta(path) -> List[Tuple[str, str]]:
    """Minimal FASTA reader: returns [(header, sequence), ...]."""
    entries, header, chunks = [], None, []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                entries.append((header, "".join(chunks)))
            header, chunks = line[1:].strip(), []
        else:
            chunks.append(line.upper().replace("T", "U"))
    if header is not None:
        entries.append((header, "".join(chunks)))
    return entries


def nw_identity(a: str, b: str) -> float:
    """Global-alignment identity: matches / alignment length, gaps included.

    Uses biotite's optimal aligner with a simple match/mismatch matrix, which is
    the standard nucleotide scoring and keeps this dependency-free relative to
    the rest of the repo.
    """
    from biotite.sequence import NucleotideSequence
    from biotite.sequence.align import SubstitutionMatrix, align_optimal

    if not a or not b:
        return 0.0
    # biotite's NucleotideSequence alphabet is DNA, so U is not a symbol. The
    # U -> T relabel is applied to BOTH sides and is a bijection, so it cannot
    # change which positions match.
    seq_a = NucleotideSequence(a.replace("U", "T"))
    seq_b = NucleotideSequence(b.replace("U", "T"))
    matrix = SubstitutionMatrix.std_nucleotide_matrix()
    alignment = align_optimal(
        seq_a, seq_b, matrix, gap_penalty=(-10, -1), local=False, max_number=1
    )[0]
    symbols = alignment.get_gapped_sequences()
    top, bottom = symbols[0], symbols[1]
    matches = sum(1 for x, y in zip(top, bottom) if x == y and x != "-")
    return matches / len(top) if top else 0.0


class SequenceIdentityGateBlock(InlineMetricBlock):
    """Hard novelty gate on scaffold-region global sequence identity."""

    _name = "sequence_identity_gate"
    _display_name = "Sequence identity gate"
    _description = (
        "Global NW identity of the scaffold region against natural references; "
        "hard novelty gate"
    )
    _resources = JobArgs(partition="cpu", mem_gb=4, cpus_per_task=1)

    references: str = Field(
        ...,
        description=(
            "FASTA of natural sequences to compare against - the native "
            "construct plus Rfam/RNAcentral homologs."
        ),
    )
    threshold: float = Field(
        0.40,
        ge=0.0,
        le=1.0,
        description=(
            "Maximum scaffold identity a design may have and still pass. Set "
            "per campaign; 0.40 is the RNA motif-scaffolding benchmark value."
        ),
    )
    exclude_motif: bool = Field(
        True,
        description=(
            "Score the scaffold region only. Leave True - motif identity is "
            "100% by construction and including it makes the gate uninformative."
        ),
    )
    chain: Optional[str] = Field(
        None, description="Restrict to one chain id. None uses every RNA residue."
    )
    compute_intra_identity: bool = Field(
        False,
        description=(
            "Also report the maximum identity to any OTHER design in this "
            "chunk, which measures whether the designer explored or emitted "
            "near-copies. O(n^2) alignments, so off by default."
        ),
    )

    def setup(self, handler, sandbox_dir, chunk_id):
        refs = read_fasta(self.references)
        if not refs:
            raise ValueError(f"No sequences read from {self.references}")
        logger.info("Novelty gate: %d reference sequences", len(refs))

        scaffolds = None
        if self.compute_intra_identity:
            scaffolds = [
                self._scaffold(item)[0]
                for item in handler.get_data(**self.get_data_kwargs())
            ]
        return {"refs": refs, "scaffolds": scaffolds}

    def _scaffold(self, loaded) -> Tuple[str, int]:
        """Scaffold sequence (motif residues removed) and the motif residue count.

        Motif residues are read from the row's ``motif`` metadata, falling back
        to the values of ``diffused_index_map``. Non-contiguous scaffold
        segments are concatenated in 5'->3' order - one documented policy, held
        fixed, as the benchmark guide requires.
        """
        motif_ids = loaded.metadata.get("motif")
        if not motif_ids:
            raw = loaded.metadata.get("diffused_index_map")
            if raw:
                index_map = json.loads(raw) if isinstance(raw, str) else raw
                motif_ids = [str(v) for v in index_map.values()]
        motif = {str(m) for m in (motif_ids or [])}

        seen, seq, n_motif = set(), [], 0
        atom_array = loaded.atom_array
        for chain_id, res_id, res_name in zip(
            atom_array.chain_id, atom_array.res_id, atom_array.res_name
        ):
            key = (str(chain_id), int(res_id))
            if key in seen:
                continue
            letter = _RNA_RESIDUES.get(str(res_name).strip().upper())
            if letter is None:
                continue
            if self.chain is not None and str(chain_id) != self.chain:
                continue
            seen.add(key)
            if self.exclude_motif and f"{chain_id}{res_id}" in motif:
                n_motif += 1
                continue
            seq.append(letter)
        return "".join(seq), n_motif

    def compute_metrics(self, loaded, *, context=None, index: int = 0):
        scaffold, n_motif = self._scaffold(loaded)
        metrics: Dict[str, object] = {
            "scaffold_length": len(scaffold),
            "n_motif_residues_excluded": n_motif,
            "threshold": self.threshold,
            "n_references": len(context["refs"]),
        }
        if not scaffold:
            metrics.update(
                {"status": "na", "reason": "empty scaffold region", "passed": False}
            )
            return metrics

        best_id, best_name = 0.0, None
        for header, ref_seq in context["refs"]:
            identity = nw_identity(scaffold, ref_seq)
            if identity > best_id:
                best_id, best_name = identity, header

        metrics.update(
            {
                "max_identity": best_id,
                "nearest_natural_id": best_name,
                "passed": bool(best_id < self.threshold),
                "status": "ok",
            }
        )

        scaffolds = context.get("scaffolds")
        if scaffolds is not None:
            others = [s for i, s in enumerate(scaffolds) if i != index and s]
            metrics["n_intra_submission_comparisons"] = len(others)
            # A singleton has no possible duplicate. Treat its maximum peer
            # identity as the vacuous, neutral 0.0 rather than a missing or
            # non-finite value so downstream reports remain serializable.
            metrics["max_intra_submission_identity"] = max(
                (nw_identity(scaffold, other) for other in others), default=0.0
            )
        return metrics
