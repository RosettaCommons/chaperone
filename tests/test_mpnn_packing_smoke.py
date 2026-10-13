"""Validate the manual packing smoke's input and output assertions without a GPU."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from atomworks.io.utils.io_utils import to_cif_file
from biotite.structure import Atom, array
import pytest

from chaperone import PipelineDataHandler
from chaperone.data import make_row
from chaperone.io import ensure_bonds


def load_helper():
    path = Path(__file__).with_name("mpnn_packing_smoke.py")
    spec = importlib.util.spec_from_file_location("mpnn_packing_smoke", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preparation_and_output_checks_cover_fixed_residues_chains_and_counts(tmp_path):
    helper = load_helper()
    atoms = []
    for residue in range(1, 4):
        for name in ("N", "CA", "C", "O", "CB"):
            atoms.append(Atom([len(atoms), 0, 0], chain_id="A", res_id=residue,
                              res_name="ALA", atom_name=name, element=name[0], hetero=False))
    path = tmp_path / "source.cif"
    to_cif_file(ensure_bonds(array(atoms)), str(path))
    source = tmp_path / "source.parquet"
    PipelineDataHandler([make_row("input", str(path), design_id=f"parent_{i}") for i in range(2)]).write_parquet(source)
    prepared, expected = helper.prepare_inputs(source, tmp_path / "prepared")
    assert expected == {f"packing_input_{i}": (3, {1: "ALA", 2: "ALA"}) for i in range(2)}
    outputs = []
    for design in prepared.get_data(load=True):
        assert set(design.atom_array.chain_id) == {"X1"}
        assert design.metadata["motif"] == ["X11", "X12"]
        for _ in range(4):
            outputs.append(SimpleNamespace(parent_design_id=design.design_id, atom_array=design.atom_array.copy()))

    class Handler:
        def __len__(self):
            return len(outputs)

        def get_data(self, **kwargs):
            return iter(outputs)

    helper.validate_outputs(Handler(), expected)
    outputs[0].atom_array.res_name[0] = "GLY"
    with pytest.raises(AssertionError):
        helper.validate_outputs(Handler(), expected)
    outputs[0].atom_array.res_name[0] = "ALA"
    outputs[0].atom_array.chain_id[:] = "A"
    with pytest.raises(AssertionError):
        helper.validate_outputs(Handler(), expected)
    outputs[0].atom_array.chain_id[:] = "X1"
    outputs.pop()
    with pytest.raises(AssertionError, match="2 inputs"):
        helper.validate_outputs(Handler(), expected)
