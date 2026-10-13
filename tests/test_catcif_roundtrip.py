"""Round-trip structures through a real catcif archive (no catcif_tools fake).

Guards the atomworks >= 3 requirement that CIF writers be given an AtomArray
with a bond list: bond-less sources (PDB files, hand-built arrays) used to
make ``store()`` silently drop every structure.
"""

import numpy as np
import pytest
from biotite.structure import AtomArray, BondList
from biotite.structure.io.pdb import PDBFile

pytest.importorskip("catcif_tools")

from pipelines import PipelineDataHandler  # noqa: E402
from pipelines.data import make_row  # noqa: E402
from pipelines.data.handler import _sequence_to_cif  # noqa: E402
from pipelines.io import ensure_bonds, parse_to_atom_array  # noqa: E402


def _peptide(chain_id: str = "A", offset: float = 0.0) -> AtomArray:
    names = ["N", "CA", "C", "O"]
    residues = ["ALA", "GLY", "SER"]
    n = len(names) * len(residues)
    array = AtomArray(n)
    array.chain_id = np.full(n, chain_id)
    array.res_id = np.repeat(np.arange(1, len(residues) + 1), len(names))
    array.res_name = np.repeat(residues, len(names))
    array.atom_name = np.tile(names, len(residues))
    array.element = np.tile(["N", "C", "C", "O"], len(residues))
    array.hetero = np.zeros(n, dtype=bool)
    array.set_annotation("b_factor", np.zeros(n))
    array.set_annotation("occupancy", np.ones(n))
    array.coord = np.array(
        [[1.5 * i + offset, 0.3 * (i % 4), 0.1 * (i % 3)] for i in range(n)]
    )
    return array


def _assert_same(expected, actual, *, bonds: bool):
    assert len(expected) == len(actual)
    np.testing.assert_allclose(expected.coord, actual.coord, atol=1e-3, equal_nan=True)
    for field in ("chain_id", "res_id", "res_name", "atom_name"):
        assert (getattr(expected, field) == getattr(actual, field)).all(), field
    if bonds:
        assert set(map(tuple, expected.bonds.as_array()[:, :2].tolist())) == set(
            map(tuple, actual.bonds.as_array()[:, :2].tolist())
        )


def _store_and_tag(tmp_path, path):
    handler = PipelineDataHandler([make_row("design", path, design_id="d")])
    handler.store("design", tmp_path / "packed")
    stored = next(handler.get_data(load=False, which="last"))
    assert stored.storage_format == "catcif"
    assert ".catcif:" in stored.path
    return stored.path


def test_ensure_bonds_adds_empty_list_without_mutating_input():
    array = _peptide()
    assert array.bonds is None

    fixed = ensure_bonds(array)

    assert array.bonds is None
    assert fixed.bonds is not None and fixed.bonds.as_array().shape[0] == 0
    assert ensure_bonds(fixed) is fixed


def test_pdb_source_is_stored_in_catcif(tmp_path):
    """The bond-less case: biotite's PDB reader returns ``bonds is None``."""
    pdb = tmp_path / "design.pdb"
    file = PDBFile()
    file.set_structure(_peptide())
    file.write(pdb)
    source = parse_to_atom_array(str(pdb))
    assert source.bonds is None

    tag = _store_and_tag(tmp_path, pdb)

    restored = parse_to_atom_array(tag)
    _assert_same(source, restored, bonds=False)
    assert restored.bonds is not None


def test_cif_source_round_trips_including_bonds(tmp_path):
    from atomworks.io.utils.io_utils import to_cif_file

    array = _peptide()
    array.bonds = BondList(array.array_length(), np.array([[0, 1, 1], [1, 2, 1]]))
    cif = tmp_path / "design.cif"
    to_cif_file(array, str(cif))

    tag = _store_and_tag(tmp_path, cif)

    _assert_same(parse_to_atom_array(str(cif)), parse_to_atom_array(tag), bonds=True)
    # Same through the atomworks parse() path rather than the biotite fast path.
    _assert_same(
        parse_to_atom_array(str(cif), hydrogen_policy="keep"),
        parse_to_atom_array(tag, hydrogen_policy="keep"),
        bonds=True,
    )


def test_sequence_only_row_is_stored_in_catcif(tmp_path):
    cif = tmp_path / "rna.cif"
    _sequence_to_cif("GGACUUCG", cif, molecule_type="rna")

    tag = _store_and_tag(tmp_path, cif)

    restored = parse_to_atom_array(tag)
    _assert_same(parse_to_atom_array(str(cif)), restored, bonds=True)
    assert set(restored.res_name.tolist()) <= {"A", "C", "G", "U"}
