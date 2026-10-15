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

from chaperone import PipelineDataHandler  # noqa: E402
from chaperone.data import make_row  # noqa: E402
from chaperone.data.handler import _sequence_to_cif  # noqa: E402
from chaperone.io import ensure_bonds, parse_to_atom_array  # noqa: E402


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


@pytest.mark.parametrize("compress", [False, True])
def test_staging_distinguishes_same_entry_in_different_archives(tmp_path, compress):
    import catcif_tools
    from pathlib import Path
    from atomworks.io.utils.io_utils import to_cif_file
    from blocks.networks.foundry.mpnn import FoundryMPNNBlock

    block = FoundryMPNNBlock(output_mode="structures")
    staged = []
    for index in range(2):
        source = tmp_path / f"source{index}.cif"
        to_cif_file(ensure_bonds(_peptide(offset=index * 100)), str(source))
        archive = tmp_path / f"archive{index}.catcif"
        catcif_tools.append_to_catcif_file(
            str(archive), source.read_text(), "same", compress=compress,
        )
        tag = f"{archive}:same"
        destination = block.stage_file(tag, tmp_path / "sandbox")
        assert destination.is_relative_to(tmp_path / "sandbox")
        assert destination.read_text() == catcif_tools.get_structure(tag)
        _assert_same(_peptide(offset=index * 100), parse_to_atom_array(destination), bonds=False)
        assert block.stage_file(Path(tag), tmp_path / "sandbox") == destination
        staged.append(destination)
    # Tools also key jobs by basename, so separate parent directories aren't enough.
    assert staged[0].name != staged[1].name


def test_archive_unique_ids_do_not_collide_with_existing_suffixes(tmp_path):
    from atomworks.io.utils.io_utils import to_cif_file

    rows = []
    for index, name in enumerate(("same", "same_1", "same")):
        source = tmp_path / str(index) / f"{name}.cif"
        source.parent.mkdir()
        to_cif_file(ensure_bonds(_peptide(offset=index * 100)), str(source))
        rows.append(make_row("design", source, design_id=f"design{index}"))
    handler = PipelineDataHandler(rows)
    handler.store("design", tmp_path / "packed.catcif", cleanup=True)
    loaded = list(handler.get_data(which="last", load=True))
    assert len({item.path for item in loaded}) == 3
    for index, item in enumerate(loaded):
        _assert_same(_peptide(offset=index * 100), item.atom_array, bonds=False)


@pytest.mark.parametrize("fail", [False, True])
def test_catcif_parser_removes_temporary_file_even_on_failure(tmp_path, monkeypatch, fail):
    import chaperone.io as io
    from pathlib import Path
    from atomworks.io.utils.io_utils import to_cif_file

    source = tmp_path / "source.cif"
    to_cif_file(ensure_bonds(_peptide()), str(source))
    tag = _store_and_tag(tmp_path, source)
    parser = io._parse_structure
    seen = []

    def capture(path, options):
        seen.append(Path(path))
        assert seen[-1].is_file()
        if fail:
            raise ValueError("deliberate parser failure")
        return parser(path, options)

    monkeypatch.setattr(io, "_parse_structure", capture)
    if fail:
        with pytest.raises(ValueError, match="deliberate parser failure"):
            io.parse_to_atom_array(tag)
    else:
        _assert_same(_peptide(), io.parse_to_atom_array(tag), bonds=False)
    assert len(seen) == 1
    assert not seen[0].exists()


def test_foundry_mpnn_archived_inputs_keep_distinct_names_and_parent_lineage(tmp_path):
    import catcif_tools
    from atomworks.io.utils.io_utils import to_cif_file
    from blocks.networks.foundry.mpnn import FoundryMPNNBlock
    from blocks.networks.mpnn_io import load_manifest

    rows = []
    for index in range(2):
        source = tmp_path / f"source{index}.cif"
        to_cif_file(ensure_bonds(_peptide(offset=index * 100)), str(source))
        archive = tmp_path / f"source{index}.catcif"
        catcif_tools.append_to_catcif_file(str(archive), source.read_text(), "same")
        rows.append(make_row("input", f"{archive}:same", design_id=f"parent{index}", metadata={"motif": ["A1"]}))
    sandbox = tmp_path / "mpnn"
    sandbox.mkdir()
    block = FoundryMPNNBlock()
    config = block._build_config(PipelineDataHandler(rows), sandbox)
    assert len({item["name"] for item in config["inputs"]}) == 2
    for index, item in enumerate(config["inputs"]):
        assert ".catcif:" not in item["structure_path"]
        assert item["fixed_residues"] == ["A1"]
        _assert_same(_peptide(offset=index * 100), parse_to_atom_array(item["structure_path"]), bonds=False)
    assert {entry["parent_design_id"] for entry in load_manifest(sandbox).values()} == {"parent0", "parent1"}


def test_rfoptimization_reads_archived_complex_and_preserves_parent(tmp_path):
    import json
    from atomworks.io.utils.io_utils import to_cif_file
    from blocks.networks.rfoptimization.block import RFOptimizationBlock

    atoms = _peptide() + _peptide(chain_id="B", offset=20)
    source = tmp_path / "complex.cif"
    to_cif_file(ensure_bonds(atoms), str(source))
    handler = PipelineDataHandler([make_row("input", source, design_id="parent")])
    handler.store("input", tmp_path / "complex.catcif", cleanup=True)
    assert not source.exists()
    sandbox = tmp_path / "rfo"
    block = RFOptimizationBlock(backprop_fraction=0, boltz_cache=str(tmp_path / "cache"))
    block.create_command(handler, sandbox, 0)
    _assert_same(atoms, parse_to_atom_array(sandbox / "design_0/start.pdb"), bonds=False)
    item = json.loads((sandbox / "rfo_inputs.json").read_text())[0]
    assert item["parent_design_id"] == "parent"
    assert ".catcif:" in item["path_handle"]


@pytest.mark.parametrize("nest_by_step", [False, True])
def test_materialize_separate_archives_with_same_entry(tmp_path, nest_by_step):
    from atomworks.io.utils.io_utils import to_cif_file

    rows = []
    for index in range(2):
        source = tmp_path / str(index) / "same.cif"
        source.parent.mkdir()
        to_cif_file(ensure_bonds(_peptide(offset=index * 100)), str(source))
        handler = PipelineDataHandler([make_row(
            "input", source, design_id=f"design{index}", parent_design_id="ancestor",
            metadata={"motif": ["A1"]}, metrics={"input": {"score": index}},
        )])
        handler.store("input", tmp_path / f"archive{index}.catcif", cleanup=True)
        rows.extend(handler._df.iter_rows(named=True))
    restored = PipelineDataHandler(rows).materialize(tmp_path / "materialized", nest_by_step=nest_by_step)
    items = list(restored.get_data(which="last", load=True))
    assert len({item.path for item in items}) == 2
    for index, item in enumerate(items):
        assert item.design_id == f"design{index}"
        assert item.parent_design_id == "ancestor"
        assert item.metadata["motif"] == ["A1"]
        assert item.metrics["input"]["score"] == index
        _assert_same(_peptide(offset=index * 100), item.atom_array, bonds=False)
