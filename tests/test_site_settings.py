"""Explicit shared site files configure clusters without institutional defaults."""

import os

import chaperone.settings as module
from chaperone.settings import ChaperoneSettings, _find_env_files


def test_explicit_site_file_loads_and_environment_map_merges(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    source = root / "src/chaperone"
    source.mkdir(parents=True)
    (root / ".env").write_text("CHAPERONE_COMPUTE_BACKEND=auto\n")
    (root / "cluster.env").write_text(
        "CHAPERONE_DEFAULT_PARTITION=cpu\nCHAPERONE_GPU_PARTITION=gpu\n"
        'CHAPERONE_GRES_MAP=\'{"small":"gpu:small:1","large":"gpu:large:1"}\'\n'
    )
    monkeypatch.setattr(module, "__file__", str(source / "settings.py"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("CHAPERONE_SITE", raising=False)
    monkeypatch.setenv("CHAPERONE_ENV_FILE", str(root / "cluster.env"))
    monkeypatch.setenv("CHAPERONE_GRES_MAP", '{"small":"gpu:t4:1"}')
    settings = ChaperoneSettings(_env_file=_find_env_files())
    assert settings.CHAPERONE_DEFAULT_PARTITION == "cpu"
    assert settings.CHAPERONE_GPU_PARTITION == "gpu"
    assert settings.CHAPERONE_GRES_MAP == {"small": "gpu:t4:1", "large": "gpu:large:1"}


def test_no_site_file_is_portable(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "__file__", str(tmp_path / "package/settings.py"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for name in os.environ:
        if name.startswith("CHAPERONE_"):
            monkeypatch.delenv(name, raising=False)
    settings = ChaperoneSettings(_env_file=_find_env_files())
    assert settings.CHAPERONE_DEFAULT_PARTITION is None
    assert settings.CHAPERONE_GPU_PARTITION is None
    assert settings.CHAPERONE_GRES_MAP == {}
    assert settings.CHAPERONE_ROOT == tmp_path
    assert settings.ESM_WEIGHTS_DIR.startswith(str(tmp_path))


def test_legacy_site_files_are_not_implicitly_loaded(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "__file__", str(tmp_path / "installed/settings.py"))
    (tmp_path / ".env.ipd").write_text("CHAPERONE_GPU_PARTITION=private\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("CHAPERONE_SITE", raising=False)
    monkeypatch.delenv("CHAPERONE_ENV_FILE", raising=False)
    assert str(tmp_path / ".env.ipd") not in _find_env_files()


def test_removed_site_preset_has_actionable_migration(monkeypatch):
    import pytest

    monkeypatch.setenv("CHAPERONE_SITE", "digs")
    with pytest.raises(ValueError, match="CHAPERONE_ENV_FILE"):
        _find_env_files()


def test_installed_package_loads_working_directory_local_overrides(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "__file__", str(tmp_path / "installed/chaperone/settings.py"))
    work = tmp_path / "work"
    work.mkdir()
    (work / ".env.local").write_text("CHAPERONE_GPU_PARTITION=accelerated\n")
    monkeypatch.chdir(work)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("CHAPERONE_SITE", raising=False)
    monkeypatch.delenv("CHAPERONE_ENV_FILE", raising=False)
    settings = ChaperoneSettings(_env_file=_find_env_files())
    assert settings.CHAPERONE_GPU_PARTITION == "accelerated"


def test_local_and_process_overrides_take_precedence(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "__file__", str(tmp_path / "installed/settings.py"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("CHAPERONE_SITE", raising=False)
    shared = tmp_path / "shared.env"
    shared.write_text("CHAPERONE_GPU_PARTITION=shared\nCHAPERONE_DEFAULT_PARTITION=cpu\n")
    (tmp_path / ".env.local").write_text("CHAPERONE_GPU_PARTITION=local\n")
    monkeypatch.setenv("CHAPERONE_ENV_FILE", str(shared))
    settings = ChaperoneSettings(_env_file=_find_env_files())
    assert settings.CHAPERONE_GPU_PARTITION == "local"
    monkeypatch.setenv("CHAPERONE_GPU_PARTITION", "process")
    settings = ChaperoneSettings(_env_file=_find_env_files())
    assert settings.CHAPERONE_GPU_PARTITION == "process"
    assert settings.CHAPERONE_DEFAULT_PARTITION == "cpu"
