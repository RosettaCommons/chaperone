"""Exercise PLIP/OpenBabel native analysis with a cortisol complex."""

from importlib.util import find_spec
from pathlib import Path

import pytest

from blocks.metrics.plip import _run_plip_isolated
from chaperone.io import parse_to_atom_array, write_pdb


@pytest.mark.skipif(
    find_spec("plip") is None or find_spec("openbabel") is None,
    reason="requires chaperone-pipelines[plip]",
)
def test_installed_native_plip_analyzes_public_ligand(tmp_path, monkeypatch):
    fixture = (
        Path(__file__).resolve().parents[1]
        / "examples/benchmarks/multi-mpnn/files/cortisol_complex.cif"
    )
    pdb_path = tmp_path / "cortisol_complex.pdb"
    write_pdb(parse_to_atom_array(fixture), pdb_path)
    pdb = pdb_path.read_text()
    monkeypatch.chdir(tmp_path)
    result = _run_plip_isolated(
        pdb,
        ligand_ids=None,
        per_residue=True,
        include_details=True,
        verbose=True,
    )
    assert result["ligand_count"] == 1
    assert result["selected_site_ids"] == ["HCY:B:1"]
    # Exact counts can change with upstream interaction/protonation rules.
    # A positive result exercises native analysis and metric serialization,
    # beyond merely importing the Python extension.
    assert result["metrics"]["num_binding_sites"] == 1
    assert result["metrics"]["total_interactions"] > 0
