"""Regression coverage for shared PDB serialization."""

import numpy as np
import pytest
from biotite.structure import Atom, BadStructureError, array
from biotite.structure.io.pdb import PDBFile

from chaperone.io import write_pdb


@pytest.fixture
def atoms():
    return array(
        [
            Atom(
                [1.25, 2.5, 3.75],
                chain_id="A",
                res_id=1,
                res_name="GLY",
                atom_name="N",
                element="N",
            ),
            Atom(
                [2.5, 3.75, 5.0],
                chain_id="A",
                res_id=1,
                res_name="GLY",
                atom_name="CA",
                element="C",
            ),
        ]
    )


@pytest.mark.parametrize("optional_fields", [False, True])
def test_write_pdb_defaults_and_source_preservation(atoms, tmp_path, optional_fields):
    if optional_fields:
        atoms.set_annotation("b_factor", np.array([np.nan, 12.5]))
        atoms.set_annotation("occupancy", np.array([0.5, np.nan]))
    before = atoms.copy()
    path = tmp_path / "structure.pdb"

    write_pdb(atoms, path)

    written = PDBFile.read(path).get_structure(
        model=1, extra_fields=["b_factor", "occupancy"]
    )
    np.testing.assert_allclose(written.coord, before.coord, atol=1e-3)
    np.testing.assert_allclose(
        written.b_factor, [0.0, 12.5] if optional_fields else 0.0
    )
    np.testing.assert_allclose(
        written.occupancy, [0.5, 1.0] if optional_fields else 1.0
    )
    np.testing.assert_array_equal(atoms.coord, before.coord)
    assert atoms.get_annotation_categories() == before.get_annotation_categories()
    for annotation in before.get_annotation_categories():
        np.testing.assert_array_equal(
            atoms.get_annotation(annotation), before.get_annotation(annotation)
        )
    for annotation in ("chain_id", "res_id", "res_name", "atom_name", "element"):
        np.testing.assert_array_equal(
            written.get_annotation(annotation), before.get_annotation(annotation)
        )


@pytest.mark.parametrize("coordinate", [np.nan, np.inf, -np.inf])
def test_write_pdb_rejects_invalid_coordinates(atoms, tmp_path, coordinate):
    atoms.coord[0, 0] = coordinate
    path = tmp_path / "invalid.pdb"

    with pytest.raises(BadStructureError, match="[Cc]oordinates"):
        write_pdb(atoms, path)

    assert not path.exists()
