"""Focused tests for residue renumbering in validation lDDT alignment."""

import numpy as np
from biotite.structure import AtomArray

from blocks.networks.prediction.validation.lddt import _align_atoms, _res_id_remap


def _array(entries):
    """entries: list of (chain, res_id, res_name, atom_name)."""
    atoms = AtomArray(len(entries))
    atoms.chain_id = np.array([e[0] for e in entries])
    atoms.res_id = np.array([e[1] for e in entries])
    atoms.res_name = np.array([e[2] for e in entries])
    atoms.atom_name = np.array([e[3] for e in entries])
    atoms.coord = np.arange(len(entries) * 3, dtype=float).reshape(-1, 3)
    return atoms


# Enough protein atoms that a one-atom ligand shortfall stays above the 95%
# aggregate gate -- which is the whole point: the aggregate passes, and only the
# non-protein coverage reveals the loss.
PROTEIN = [("A", i, "GLY", "CA") for i in range(1, 61)]


def test_renumbered_ligand_is_recovered():
    # AF3 writes the ligand the reference numbers 201 as residue 1.
    reference = _array(PROTEIN + [("G", 201, "PLC", "C1"), ("G", 201, "PLC", "C2")])
    prediction = _array(PROTEIN + [("G", 1, "PLC", "C1"), ("G", 1, "PLC", "C2")])

    remap, unmapped = _res_id_remap(reference, prediction)
    assert remap == {("G", 1): 201}
    assert unmapped == []

    ref, pred, report = _align_atoms(reference, prediction)
    assert len(ref) == len(pred) == len(PROTEIN) + 2
    assert report["atom_coverage"] == 1.0
    assert report["non_protein_atom_coverage"] == 1.0
    assert report["n_renumbered_residues"] == 1


def test_mismatched_composition_is_not_guessed():
    # Different ligand identity: pairing by position would be a guess, so the
    # chain is left alone and named rather than silently mis-paired.
    reference = _array(PROTEIN + [("G", 201, "PLC", "C1")])
    prediction = _array(PROTEIN + [("G", 1, "LUT", "C1")])

    remap, unmapped = _res_id_remap(reference, prediction)
    assert remap == {}
    assert unmapped == ["G"]


def test_partial_ligand_is_reported_not_hidden():
    # The prediction models one of two ligand atoms: legitimate, but the
    # shortfall must show up in non-protein coverage.
    reference = _array(PROTEIN + [("G", 201, "PLC", "C1"), ("G", 201, "PLC", "C2")])
    prediction = _array(PROTEIN + [("G", 1, "PLC", "C1")])

    _, _, report = _align_atoms(reference, prediction)
    # The aggregate gate is satisfied and would have hidden this entirely.
    assert report["atom_coverage"] > 0.95
    assert report["non_protein_atom_coverage"] == 0.5


def test_identical_numbering_needs_no_remap():
    reference = _array(PROTEIN + [("G", 201, "PLC", "C1")])
    remap, unmapped = _res_id_remap(reference, reference)
    assert remap == {} and unmapped == []
    _, _, report = _align_atoms(reference, reference)
    assert report["atom_coverage"] == 1.0
    assert report["n_renumbered_residues"] == 0
