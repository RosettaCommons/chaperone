"""TM-score_RNA via US-align.

The anchor is the published worked example: 4kzz chain j vs 5mrc chain bb, two
tRNAs in Rfam RF00005. Gong, Zhang & Zhang (2019) report TM-score_RNA ~= 0.520
normalized by 5mrc chain bb; US-align reproduces 0.513 here.
"""

import shutil
from pathlib import Path

import pytest

from blocks.metrics.tm_score_rna import (
    MIN_REFERENCE_LENGTH,
    RNA_FAMILY_THRESHOLD,
    parse_usalign,
    rna_d0,
    run_usalign,
    write_clean_pdb,
)

DATA = Path(__file__).parent.parent / "test_data" / "rna_tm"
TRNA_A = str(DATA / "4kzz_j.pdb")  # 75 nt
TRNA_B = str(DATA / "5mrc_bb.pdb")  # 76 nt

needs_usalign = pytest.mark.skipif(
    shutil.which("USalign") is None, reason="US-align binary not on PATH"
)


# ── d0, the whole reason this is a separate metric from protein TM ──────


def test_d0_matches_the_published_formula():
    assert rna_d0(76) == pytest.approx(0.6 * (75.5**0.5) - 2.5, abs=1e-9)
    assert rna_d0(76) == pytest.approx(2.7134, abs=1e-3)


def test_d0_uses_usalign_step_table_below_thirty():
    """d0 crosses zero at L ~= 17.9 and is negative below, so the table is not
    cosmetic - it is what keeps short references from producing nonsense."""
    assert 0.6 * (17.5**0.5) - 2.5 == pytest.approx(0.010, abs=0.001)  # L=18
    assert 0.6 * (16.5**0.5) - 2.5 < 0  # L=17
    assert rna_d0(11) == 0.3
    assert rna_d0(19) == 0.5
    assert rna_d0(29) == 0.7
    assert rna_d0(30) == pytest.approx(0.6 * (29.5**0.5) - 2.5)


def test_protein_threshold_is_not_reused():
    assert RNA_FAMILY_THRESHOLD == 0.45


# ── parser ──────────────────────────────────────────────────────────────

_REPORT = """
 * US-align (Version 20260819)                                      *
Name of Structure_1: a.pdb
Name of Structure_2: b.pdb
Length of Structure_1: 75 residues
Length of Structure_2: 76 residues

Aligned length= 71, RMSD=   3.25, Seq_ID=n_identical/n_aligned= 0.507
TM-score= 0.51562 (normalized by length of Structure_1: L=75, d0=2.68)
TM-score= 0.51314 (normalized by length of Structure_2: L=76, d0=2.71)
"""


def test_parser_normalizes_by_structure_two():
    """Structure_2 is the reference, so tm_ref must come from that line - the
    two values are close here but swapping them is a silent correctness bug."""
    r = parse_usalign(_REPORT)
    assert r["tm_ref"] == pytest.approx(0.51314)
    assert r["tm_design"] == pytest.approx(0.51562)
    assert r["L_ref"] == 76 and r["L_design"] == 75
    assert r["d0_used"] == pytest.approx(2.71)
    assert r["L_align"] == 71
    assert r["rmsd_aligned"] == pytest.approx(3.25)
    assert r["usalign_version"] == "20260819"


def test_parser_rejects_output_with_no_score():
    with pytest.raises(RuntimeError, match="no TM-score"):
        parse_usalign("Warning! Cannot parse file")


# ── the published worked example ────────────────────────────────────────


@needs_usalign
def test_reproduces_the_paper_worked_example():
    r = run_usalign(TRNA_A, TRNA_B)
    assert r["L_ref"] == 76 and r["L_design"] == 75
    assert r["d0_used"] == pytest.approx(rna_d0(76), abs=0.01)
    assert r["tm_ref"] == pytest.approx(0.520, abs=0.02)
    assert r["tm_ref"] >= RNA_FAMILY_THRESHOLD  # same Rfam family


@needs_usalign
def test_identical_structures_score_one():
    r = run_usalign(TRNA_B, TRNA_B)
    assert r["tm_ref"] == pytest.approx(1.0, abs=1e-4)
    assert r["rmsd_aligned"] == pytest.approx(0.0, abs=1e-3)


@needs_usalign
def test_score_is_asymmetric_in_reference_length():
    """L is the reference's, so swapping the arguments changes d0 and the score."""
    forward = run_usalign(TRNA_A, TRNA_B)
    reverse = run_usalign(TRNA_B, TRNA_A)
    assert forward["L_ref"] != reverse["L_ref"]
    assert forward["tm_ref"] == pytest.approx(reverse["tm_design"], abs=1e-4)


@needs_usalign
def test_fixed_correspondence_differs_from_the_alignment_search():
    """-TMscore 1 scores the given numbering rather than searching, so on two
    structures with unrelated numbering it must not beat the search."""
    searched = run_usalign(TRNA_A, TRNA_B)["tm_ref"]
    fixed = run_usalign(TRNA_A, TRNA_B, fixed_correspondence=True)["tm_ref"]
    assert fixed <= searched + 1e-6


# ── input preparation ───────────────────────────────────────────────────


def test_stripping_keeps_nucleotides_and_drops_ions_and_water(tmp_path):
    import numpy as np
    from biotite.structure.io.pdb import PDBFile

    from chaperone.io import parse_to_atom_array

    array = parse_to_atom_array(TRNA_B)
    n_clean = write_clean_pdb(array, str(tmp_path / "clean.pdb"))
    assert n_clean == 76

    contaminated = array + array[:1].copy()
    contaminated.res_name[-1] = "MN"
    contaminated.atom_name[-1] = "MN"
    contaminated.res_id[-1] = 999
    assert write_clean_pdb(contaminated, str(tmp_path / "with_ion.pdb")) == 76
    assert np.all(
        PDBFile.read(str(tmp_path / "with_ion.pdb")).get_structure().res_name[0] != "MN"
    )


def test_multi_character_chain_ids_are_remapped(tmp_path):
    """mmCIF chains like 'bb' cannot be written to PDB; -ter 0 makes the rename
    free, and without it a multi-chain reference aborts the whole run."""
    from chaperone.io import parse_to_atom_array

    array = parse_to_atom_array(TRNA_B)
    array.chain_id[:] = "bb"
    assert write_clean_pdb(array, str(tmp_path / "renamed.pdb")) == 76


def test_empty_selection_is_an_error(tmp_path):
    from chaperone.io import parse_to_atom_array

    array = parse_to_atom_array(TRNA_B)
    with pytest.raises(ValueError, match="no RNA residues"):
        write_clean_pdb(array, str(tmp_path / "none.pdb"), chain="Z")


# ── the short-reference refusal ─────────────────────────────────────────


@needs_usalign
def test_short_reference_is_refused_rather_than_scored(tmp_path):
    """d0 collapses below L~18, so a motif-sized reference must raise instead of
    silently returning a number from US-align's clamped regime."""
    from blocks.metrics.tm_score_rna import TMScoreRNABlock
    from chaperone.io import parse_to_atom_array

    array = parse_to_atom_array(TRNA_B)
    short = array[array.res_id <= 10]
    short_pdb = str(tmp_path / "short.pdb")
    write_clean_pdb(short, short_pdb)

    block = TMScoreRNABlock(reference=short_pdb)
    with pytest.raises(ValueError, match="below the"):
        block.setup(None, tmp_path, 0)
    assert MIN_REFERENCE_LENGTH == 30


@needs_usalign
def test_expected_reference_length_mismatch_is_refused(tmp_path):
    from blocks.metrics.tm_score_rna import TMScoreRNABlock

    block = TMScoreRNABlock(reference=TRNA_B, expected_reference_length=63)
    with pytest.raises(ValueError, match="expected_reference_length"):
        block.setup(None, tmp_path, 0)

    TMScoreRNABlock(reference=TRNA_B, expected_reference_length=76).setup(
        None, tmp_path, 0
    )
