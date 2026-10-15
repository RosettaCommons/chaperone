"""Exercise PLIP/OpenBabel native analysis with the public biotin fixture."""

from pathlib import Path

from blocks.metrics.plip import _run_plip_isolated


def test_installed_native_plip_analyzes_public_ligand(tmp_path, monkeypatch):
    fixture = (
        Path(__file__).resolve().parents[1]
        / "examples/biotin-binder/files/biotin_all.pdb"
    )
    pdb = fixture.read_text()
    monkeypatch.chdir(tmp_path)
    result = _run_plip_isolated(
        pdb,
        ligand_ids=None,
        per_residue=True,
        include_details=True,
        verbose=True,
    )
    assert result["ligand_count"] == 1
    assert result["selected_site_ids"] == ["B;N:B:1"]
    # Exact counts can change with upstream interaction/protonation rules.
    # A positive result exercises native analysis and metric serialization,
    # beyond merely importing the Python extension.
    assert result["metrics"]["num_binding_sites"] == 1
    assert result["metrics"]["total_interactions"] > 0
