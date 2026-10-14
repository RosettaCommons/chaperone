"""Portable CCD detection for ligand-containing predictor inputs."""

from blocks.utils.oracles import is_ccd_name


def test_bundled_ccd_without_site_mirror(monkeypatch):
    monkeypatch.setattr("chaperone.get_setting_field", lambda name, default=None: None)
    assert is_ccd_name("MN")
    assert is_ccd_name("ALA")
    assert not is_ccd_name("NOT_A_COMPONENT")
    assert not is_ccd_name("")
    assert not is_ccd_name(None)


def test_mirror_extensions_and_bundled_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "chaperone.get_setting_field", lambda name, default=None: str(tmp_path)
    )
    entry = tmp_path / "N" / "NEWCCD" / "NEWCCD.cif"
    entry.parent.mkdir(parents=True)
    entry.write_text("data_NEWCCD\n")
    assert is_ccd_name("NEWCCD")
    assert is_ccd_name("MN")
    assert not is_ccd_name("UNKNOWN_CCD")
