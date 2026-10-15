"""Structure parsing passes atomworks an explicit ParseConfig, never bare kwargs."""

import warnings

import numpy as np
import pytest
from atomworks.io.config import ParseConfig
from biotite.structure import AtomArray
from biotite.structure.io.pdbx import CIFFile, set_structure

from chaperone.io import parse_to_atom_array

DEPRECATIONS = ("Passing processing kwargs", "load_any")


@pytest.fixture
def cif_path(tmp_path):
    atoms = AtomArray(3)
    atoms.coord = np.arange(9, dtype=float).reshape(3, 3)
    atoms.chain_id[:] = "A"
    atoms.res_id[:] = 1
    atoms.res_name[:] = "GLY"
    atoms.atom_name[:] = ["N", "CA", "C"]
    atoms.element[:] = ["N", "C", "C"]
    path = tmp_path / "gly.cif"
    cif = CIFFile()
    set_structure(cif, atoms)
    cif.write(path)
    return path


def _no_atomworks_deprecations():
    for message in DEPRECATIONS:
        warnings.filterwarnings("error", message=message, category=DeprecationWarning)


def test_atomworks_path_receives_a_parse_config(cif_path, monkeypatch):
    import atomworks.io

    real_parse = atomworks.io.parse
    seen = {}

    def spy(path, **kwargs):
        seen.update(kwargs)
        return real_parse(path, **kwargs)

    monkeypatch.setattr("atomworks.io.parse", spy)
    with warnings.catch_warnings():
        _no_atomworks_deprecations()
        # hydrogen_policy is not handled by the fast biotite path
        atoms = parse_to_atom_array(cif_path, hydrogen_policy="keep")

    assert list(seen) == ["config"]
    assert isinstance(seen["config"], ParseConfig)
    assert seen["config"].hydrogen_policy == "keep"
    assert len(atoms) == 3


def test_unknown_parse_option_fails_loudly(cif_path):
    with pytest.raises(TypeError, match="not_a_parse_option"):
        parse_to_atom_array(cif_path, not_a_parse_option=True)


def test_fallback_is_a_raw_parse_not_load_any(cif_path, monkeypatch):
    import atomworks.io

    real_parse = atomworks.io.parse
    calls = []

    def fail_first(path, **kwargs):
        calls.append(kwargs["config"])
        if len(calls) == 1:
            raise ValueError("simulated atomworks failure")
        return real_parse(path, **kwargs)

    monkeypatch.setattr("atomworks.io.parse", fail_first)
    with warnings.catch_warnings():
        _no_atomworks_deprecations()
        atoms = parse_to_atom_array(cif_path, hydrogen_policy="keep")

    assert len(calls) == 2
    assert calls[1].remove_waters is False
    assert calls[1].add_missing_atoms is False
    assert len(atoms) == 3
