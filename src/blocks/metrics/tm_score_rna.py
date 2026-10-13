"""TM-score_RNA (Gong, Zhang & Zhang 2019, Bioinformatics 35:4459-4461) via US-align.

The protein TM-score is miscalibrated for nucleic acids: RNA backbones are more
extended, so the protein ``d0`` is too small and unrelated RNAs score near zero
while related ones score far below the protein cutoffs. The RNA form re-fits d0
on C3' atoms,

    TM = (1/L) * sum_i 1 / (1 + (d_i / d0)^2)
    d0 = 0.6 * sqrt(L - 0.5) - 2.5

with **L the length of the reference structure**, not the model and not the
alignment. Because the score is asymmetric in L, ``tm_ref`` (reference-normalized)
is the primary number and ``tm_design`` is reported beside it - a large gap means
the design reproduced the fold but carries unstructured extra length.

Calibration from the paper, which is NOT the protein calibration:

    ~0.21   random-pair baseline
     0.45   same-Rfam-family decision boundary

Do not carry the protein 0.5 threshold over.

Two hard constraints, both enforced here:

1. **d0 collapses at short length.** It reaches zero near L = 18 and goes
   negative below, so TM-score_RNA is undefined for short subsets - never apply
   it to a motif-only selection, where RMSD is the correct metric. US-align
   clamps d0 to a step table below L = 30 (``TMalign.h``, ``parameter_set4final``);
   rather than depend on that regime we refuse a reference shorter than
   ``min_reference_length``.
2. **Non-polymer components corrupt the alignment.** Ions, ligands and waters are
   stripped from both structures before the call, and ``-ter 0`` makes a
   multi-chain reference align as one unit. Set ``expected_reference_length`` to
   have the parsed L checked against the nucleotide count you expect, because a
   silently mis-parsed reference changes d0 and therefore every score.

The alignment search itself is not reimplemented - US-align is the reference
implementation and is called as a subprocess. What lives here is the input
preparation, the output parser and the fixed-correspondence path (``-TMscore 1``,
which scores a given residue-index correspondence instead of searching for one).
"""

import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

from pydantic import Field

from pipelines import InlineMetricBlock, JobArgs
from pipelines.io import parse_to_atom_array

logger = logging.getLogger(__name__)

# d0 leaves the continuous regime here; below this US-align uses a step table.
MIN_REFERENCE_LENGTH = 30

# Paper calibration for TM-score_RNA. Not the protein numbers.
RNA_RANDOM_BASELINE = 0.21
RNA_FAMILY_THRESHOLD = 0.45

NUCLEIC_ANCHOR = "C3'"
PROTEIN_ANCHOR = "CA"

_WATERS = {"HOH", "WAT", "DOD"}

_PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def rna_d0(length: int) -> float:
    """d0 as US-align computes it for nucleic acids, step table included.

    Ported from ``parameter_set4final`` in TMalign.h rather than using the bare
    formula, which is negative below L = 18. Used only to cross-check the value
    US-align reports; the score itself always comes from US-align.
    """
    if length <= 11:
        return 0.3
    if length <= 15:
        return 0.4
    if length <= 19:
        return 0.5
    if length <= 23:
        return 0.6
    if length < 30:
        return 0.7
    return 0.6 * math.sqrt(length - 0.5) - 2.5


# ── US-align invocation and parsing ──────────────────────────────────────

_RE_VERSION = re.compile(r"US-align \(Version ([^)\s]+)")
_RE_LEN = re.compile(r"Length of Structure_(\d+):\s*(\d+)")
_RE_ALIGNED = re.compile(
    r"Aligned length=\s*(\d+),\s*RMSD=\s*([\d.-]+),\s*"
    r"Seq_ID=n_identical/n_aligned=\s*([\d.-]+)"
)
_RE_TM = re.compile(
    r"TM-score=\s*([\d.]+)\s*\(normalized by length of Structure_(\d+):\s*"
    r"L=(\d+),\s*d0=([\d.-]+)\)"
)


def resolve_usalign(binary: Optional[str] = None) -> str:
    """Absolute path to the US-align executable, or a clear error.

    US-align is a single C++ translation unit, not a package, so it is neither
    a dependency the venv installs nor something a fresh clone has. Resolution
    order: the explicit argument, the ``USALIGN_BIN`` setting, then ``sys.prefix
    /bin`` (where ``src/blocks/scripts/install_usalign.sh`` puts it), then PATH.
    """
    from pipelines import get_setting_field

    candidates = [binary, get_setting_field("USALIGN_BIN", "USalign")]
    for candidate in candidates:
        if not candidate:
            continue
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return os.path.abspath(candidate)
        in_venv = os.path.join(sys.prefix, "bin", os.path.basename(candidate))
        if os.path.isfile(in_venv) and os.access(in_venv, os.X_OK):
            return in_venv
        found = shutil.which(candidate)
        if found is not None:
            return found
    raise FileNotFoundError(
        f"US-align binary {binary or 'USalign'!r} not found. Build it with "
        "src/blocks/scripts/install_usalign.sh, or set the USALIGN_BIN setting to an "
        "existing build."
    )


def parse_usalign(stdout: str) -> Dict[str, object]:
    """Parse one US-align pairwise report.

    Structure_1 is the model (superimposed onto Structure_2), Structure_2 is the
    reference, so ``tm_ref`` is the Structure_2-normalized value.
    """
    lengths = {int(n): int(v) for n, v in _RE_LEN.findall(stdout)}
    tms = {
        int(which): (float(tm), int(length), float(d0))
        for tm, which, length, d0 in _RE_TM.findall(stdout)
    }
    if 1 not in tms or 2 not in tms:
        raise RuntimeError(f"US-align produced no TM-score:\n{stdout}")

    aligned = _RE_ALIGNED.search(stdout)
    version = _RE_VERSION.search(stdout)
    tm_design, l_design_norm, d0_design = tms[1]
    tm_ref, l_ref_norm, d0_ref = tms[2]

    return {
        "tm_ref": tm_ref,
        "tm_design": tm_design,
        "d0_used": d0_ref,
        "d0_design": d0_design,
        "L_ref": lengths.get(2, l_ref_norm),
        "L_design": lengths.get(1, l_design_norm),
        "L_align": int(aligned.group(1)) if aligned else None,
        "rmsd_aligned": float(aligned.group(2)) if aligned else None,
        "seq_id_aligned": float(aligned.group(3)) if aligned else None,
        "usalign_version": version.group(1) if version else None,
    }


def run_usalign(
    model_path: str,
    reference_path: str,
    *,
    binary: Optional[str] = None,
    mol: str = "RNA",
    fixed_correspondence: bool = False,
    model_chain: Optional[str] = None,
    reference_chain: Optional[str] = None,
    extra_args: Tuple[str, ...] = (),
) -> Dict[str, object]:
    """Align ``model_path`` onto ``reference_path`` and parse the report.

    ``-ter 0`` is always passed so a multi-chain reference is treated as one
    unit; without it US-align stops at the first chain terminator and silently
    scores a fragment.
    """
    cmd: List[str] = [resolve_usalign(binary), model_path, reference_path]
    cmd += ["-mol", mol, "-ter", "0"]
    if fixed_correspondence:
        cmd += ["-TMscore", "1"]
    if model_chain:
        cmd += ["-chain1", model_chain]
    if reference_chain:
        cmd += ["-chain2", reference_chain]
    cmd += list(extra_args)

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"US-align failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"{proc.stdout}\n{proc.stderr}"
        )
    parsed = parse_usalign(proc.stdout)
    parsed["usalign_command"] = " ".join(cmd)
    return parsed


# ── structure preparation ────────────────────────────────────────────────


def polymer_only(atom_array, chain: Optional[str] = None, mol: str = "RNA"):
    """Drop everything without a polymer anchor atom.

    Selecting on the presence of C3' (RNA/DNA) or CA (protein) rather than on a
    residue-name whitelist keeps modified nucleotides - 2'-O-methylcytidine and
    friends are real reference residues and dropping them would change L, and so
    d0 - while ions, ligands and waters have no anchor and fall out on their own.
    """
    anchor = NUCLEIC_ANCHOR if mol == "RNA" else PROTEIN_ANCHOR
    keep_residues = set()
    for i in range(len(atom_array)):
        if str(atom_array.res_name[i]).strip().upper() in _WATERS:
            continue
        if str(atom_array.atom_name[i]).strip() != anchor:
            continue
        keep_residues.add((str(atom_array.chain_id[i]), int(atom_array.res_id[i])))

    mask = []
    for i in range(len(atom_array)):
        chain_id = str(atom_array.chain_id[i])
        keep = (chain_id, int(atom_array.res_id[i])) in keep_residues
        if keep and chain is not None and chain_id != chain:
            keep = False
        mask.append(keep)
    return atom_array[mask], len(
        {r for r in keep_residues if chain is None or r[0] == chain}
    )


def write_clean_pdb(source, dest: str, chain: Optional[str] = None, mol: str = "RNA"):
    """Write a polymer-only PDB for US-align; returns the residue count written."""
    from biotite.structure.io.pdb import PDBFile

    atom_array = source if hasattr(source, "res_name") else parse_to_atom_array(source)
    cleaned, n_residues = polymer_only(atom_array, chain=chain, mol=mol)
    if n_residues == 0:
        raise ValueError(f"no {mol} residues left after stripping non-polymer atoms")
    # PDB allows one character of chain id; mmCIF-derived ids like 'bb' would
    # abort the write. -ter 0 makes chain identity irrelevant to the score, so
    # remapping is safe and keeps multi-chain references usable.
    cleaned = cleaned.copy()
    for i, original in enumerate(sorted(set(str(c) for c in cleaned.chain_id))):
        cleaned.chain_id[cleaned.chain_id == original] = _PDB_CHAIN_IDS[
            i % len(_PDB_CHAIN_IDS)
        ]
    pdb = PDBFile()
    pdb.set_structure(cleaned)
    pdb.write(dest)
    return n_residues


class TMScoreRNABlock(InlineMetricBlock):
    """TM-score_RNA against a reference fold, computed by US-align."""

    _name = "tm_score_rna"
    _display_name = "TM-score RNA"
    _description = (
        "TM-score_RNA (Gong 2019) against a reference fold via US-align, "
        "normalized by the reference length"
    )
    _resources = JobArgs(partition="cpu", mem_gb=4, cpus_per_task=1)

    reference: str = Field(
        ..., description="Path to the reference structure (the native fold)."
    )
    reference_chain: Optional[str] = Field(
        None,
        description=(
            "Restrict the reference to one chain. None keeps every chain and "
            "aligns them as one unit (-ter 0)."
        ),
    )
    chain: Optional[str] = Field(
        None, description="Restrict the design to one chain. None uses all."
    )
    expected_reference_length: Optional[int] = Field(
        None,
        description=(
            "Nucleotide count the reference should parse to. Checked at setup - "
            "a mis-parsed reference silently changes d0 and every score with it."
        ),
    )
    fixed_correspondence: bool = Field(
        False,
        description=(
            "Score the residue-index correspondence as given (-TMscore 1) "
            "instead of searching for an alignment. Only meaningful when design "
            "and reference share a numbering."
        ),
    )
    mol: str = Field("RNA", description="US-align molecule type: RNA or prot.")
    usalign_binary: Optional[str] = Field(
        None,
        description="US-align executable name or path. Defaults to the "
        "USALIGN_BIN setting, the venv's bin, then PATH.",
    )
    min_reference_length: int = Field(
        MIN_REFERENCE_LENGTH,
        description=(
            "Refuse a reference shorter than this. d0 goes to zero near L=18, so "
            "TM-score_RNA is undefined on short selections - use RMSD there."
        ),
    )

    def setup(self, handler, sandbox_dir, chunk_id):
        binary = resolve_usalign(self.usalign_binary)
        ref_pdb = os.path.join(str(sandbox_dir), "tm_reference.pdb")
        n_ref = write_clean_pdb(
            self.reference, ref_pdb, chain=self.reference_chain, mol=self.mol
        )

        if n_ref < self.min_reference_length:
            raise ValueError(
                f"reference {self.reference} has {n_ref} residues, below the "
                f"{self.min_reference_length}-residue floor. TM-score_RNA's d0 "
                "is not defined at that length - score this selection with RMSD "
                "instead (see reference_motif_rmsd)."
            )
        if (
            self.expected_reference_length is not None
            and n_ref != self.expected_reference_length
        ):
            raise ValueError(
                f"reference parsed to {n_ref} residues but "
                f"expected_reference_length={self.expected_reference_length}. "
                "Check the chain selection and whether modified nucleotides "
                "survived stripping; d0 depends on this number."
            )
        logger.info(
            "TM-score_RNA reference: %d residues, d0=%.3f", n_ref, rna_d0(n_ref)
        )
        return {"reference_pdb": ref_pdb, "n_ref": n_ref, "binary": binary}

    def compute_metrics(self, loaded, *, context=None, index: int = 0):
        metrics: Dict[str, object] = {
            "reference_length_parsed": context["n_ref"],
            "family_threshold": RNA_FAMILY_THRESHOLD,
            "random_baseline": RNA_RANDOM_BASELINE,
        }
        with tempfile.TemporaryDirectory() as tmp:
            model_pdb = os.path.join(tmp, "model.pdb")
            try:
                n_model = write_clean_pdb(
                    loaded.atom_array, model_pdb, chain=self.chain, mol=self.mol
                )
            except ValueError as exc:
                metrics.update({"status": "na", "reason": str(exc)})
                return metrics

            result = run_usalign(
                model_pdb,
                context["reference_pdb"],
                binary=context["binary"],
                mol=self.mol,
                fixed_correspondence=self.fixed_correspondence,
            )

        metrics.update(result)
        metrics["L_design_parsed"] = n_model
        if result["L_align"] and result["L_ref"]:
            metrics["coverage"] = result["L_align"] / result["L_ref"]
        metrics["above_family_threshold"] = bool(
            result["tm_ref"] >= RNA_FAMILY_THRESHOLD
        )
        # A wide split means the fold is reproduced but padded with loose length.
        metrics["tm_norm_gap"] = result["tm_ref"] - result["tm_design"]
        metrics["status"] = "ok"
        return metrics
