"""TM-score of a structure against an external reference fold.

DO NOT USE THIS FOR RNA OR DNA. The protein d0 is miscalibrated for nucleic
acids - it inflates the apparent similarity of unrelated RNAs and compresses
the useful range - so use ``tm_score_rna`` (Gong, Zhang & Zhang 2019), which
shells out to US-align with the nucleic-acid d0. This block remains correct for
protein.

RMSD is the wrong global number for motif scaffolding: designs vary in length
and a single flexible arm blows it up. TM-score normalized by the *reference*
length answers "how much of the native fold did this reproduce", and cannot be
gamed by making the design longer.

Both normalizations are reported. A large ``tm_ref`` / ``tm_design`` split means
the design reproduced the fold but carries a big unstructured appendage.

Non-polymer components are stripped from both sides before alignment - ions and
ligands left in place produce garbage alignments. Nucleic acids are represented
by C3' and protein by CA, matching US-align's convention for each molecule type.

This complements rather than replaces ``reference_motif_rmsd``: TM-score is the
global fold number, motif RMSD is the sharper local one, and they disagree in
informative ways.
"""

import logging
from typing import Dict, Optional

import numpy as np
from pydantic import Field

from chaperone import InlineMetricBlock, JobArgs
from chaperone.io import parse_to_atom_array

logger = logging.getLogger(__name__)

NUCLEIC_ANCHOR = "C3'"
PROTEIN_ANCHOR = "CA"

_NUCLEIC = {"A", "C", "G", "U", "T", "DA", "DC", "DG", "DT"}
_THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def polymer_anchors(atom_array, chain: Optional[str] = None):
    """One anchor atom per polymer residue, in file order.

    Returns ``(coords, sequence)``. Ions, ligands and waters have no anchor atom
    and so drop out without needing an explicit HETATM filter.
    """
    coords, seq, seen = [], [], set()
    for i in range(len(atom_array)):
        res_name = str(atom_array.res_name[i]).strip().upper()
        atom_name = str(atom_array.atom_name[i]).strip()
        chain_id = str(atom_array.chain_id[i])
        if chain is not None and chain_id != chain:
            continue
        if res_name in _NUCLEIC:
            if atom_name != NUCLEIC_ANCHOR:
                continue
            letter = res_name[-1]
        elif res_name in _THREE_TO_ONE:
            if atom_name != PROTEIN_ANCHOR:
                continue
            letter = _THREE_TO_ONE[res_name]
        else:
            continue
        key = (chain_id, int(atom_array.res_id[i]))
        if key in seen:
            continue
        seen.add(key)
        coords.append(atom_array.coord[i])
        seq.append(letter)
    return np.asarray(coords, dtype=np.float64), "".join(seq)


class TMScoreBlock(InlineMetricBlock):
    """TM-score against an external reference structure."""

    _name = "tm_score"
    _display_name = "TM-score"
    _description = "TM-score of each structure against an external reference fold"
    _resources = JobArgs(partition="cpu", mem_gb=4, cpus_per_task=1)

    reference: str = Field(
        ..., description="Path to the reference structure (PDB/CIF)."
    )
    reference_chain: Optional[str] = Field(
        None, description="Restrict the reference to one chain id."
    )
    chain: Optional[str] = Field(
        None, description="Restrict the target to one chain id."
    )

    def setup(self, handler, sandbox_dir, chunk_id):
        ref_coords, ref_seq = polymer_anchors(
            parse_to_atom_array(self.reference), self.reference_chain
        )
        if len(ref_coords) < 3:
            raise ValueError(
                f"Reference {self.reference} yielded {len(ref_coords)} anchor "
                "atoms; need at least 3. Check reference_chain."
            )
        logger.info("TM-score reference: %d residues", len(ref_coords))
        return {"coords": ref_coords, "seq": ref_seq}

    def compute_metrics(self, loaded, *, context=None, index: int = 0):
        from tmtools import tm_align

        coords, seq = polymer_anchors(loaded.atom_array, self.chain)
        metrics: Dict[str, object] = {
            "n_residues_design": len(coords),
            "n_residues_reference": len(context["coords"]),
        }
        if len(coords) < 3:
            metrics.update({"status": "na", "reason": "fewer than 3 anchor atoms"})
            return metrics

        # chain1 = design, chain2 = reference, so tm_norm_chain2 is normalized
        # by the reference length. That is the primary number.
        result = tm_align(coords, context["coords"], seq, context["seq"])
        metrics.update(
            {
                "tm_ref": float(result.tm_norm_chain2),
                "tm_design": float(result.tm_norm_chain1),
                "rmsd_aligned": float(result.rmsd),
                "status": "ok",
            }
        )
        return metrics
