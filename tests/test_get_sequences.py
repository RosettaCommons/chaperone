"""Protein sequences read off a structure."""

from biotite.structure import Atom, array

from chaperone.io import get_sequences


def _atom(chain, res_id, res_name, atom_name, element):
    return Atom(
        [0.0, 0.0, float(res_id)],
        chain_id=chain,
        res_id=res_id,
        res_name=res_name,
        atom_name=atom_name,
        element=element,
    )


def _residue(chain, res_id, res_name, names=("N", "CA", "C", "O")):
    return [_atom(chain, res_id, res_name, name, name[0]) for name in names]


def test_ions_and_ligands_are_not_protein_residues():
    atoms = array(
        _residue("A", 1, "GLY")
        + _residue("A", 2, "ALA")
        + [_atom("A", 3, "CA", "CA", "CA")]
        + [_atom("B", 1, "CA", "CA", "CA")]
        + [_atom("C", 1, "NO3", "N", "N"), _atom("C", 1, "NO3", "O1", "O")]
        + _residue("A", 4, "LIG", names=("CA", "CB", "OG"))
        + _residue("D", 1, "LIG", names=("CA", "CB", "OG"))
    )
    assert get_sequences(atoms) == {"A": "GA", "B": None, "C": None, "D": None}


def test_partial_and_reduced_models_keep_their_sequence():
    atoms = array(
        _residue("A", 1, "GLY", names=("CA",))
        + _residue("A", 2, "ALA", names=("CA",))
        + _residue("A", 3, "UNK", names=("CA",))
        + _residue("B", 1, "GLY", names=("N",))
        + _residue("B", 2, "ALA", names=("N",))
        + _residue("C", 1, "GLY")
        + _residue("C", 2, "ALA", names=("N",))
        + _residue("C", 3, "MSE")
        + _residue("C", 4, "UNK")
    )
    assert get_sequences(atoms) == {"A": "GAX", "B": "GA", "C": "GAXX"}


def test_residue_listing_its_ca_first_does_not_take_over_the_chain():
    atoms = array(
        _residue("A", 1, "GLY") + _residue("A", 2, "LYS", names=("CA", "N", "C", "O"))
    )
    assert get_sequences(atoms) == {"A": "GK"}


def test_nucleic_chain_keeps_its_sequence():
    atoms = array(
        _residue("D", 1, "DA", names=("P", "OP1", "C3'"))
        + _residue("D", 2, "DT", names=("P", "OP1", "C3'"))
    )
    assert get_sequences(atoms) == {"D": "AT"}
