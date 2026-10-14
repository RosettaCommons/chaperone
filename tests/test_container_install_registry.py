"""Explicit recipes remain installable irrespective of the active runtime."""

from types import SimpleNamespace

from absl import app
import pytest

import blocks
import blocks.networks.mpnn as mpnn
from chaperone_cli import install


def test_mpnn_is_installable(monkeypatch, tmp_path):
    assert blocks.get_container("ligandmpnn") is mpnn.LIGANDMPNN_CONTAINER
    monkeypatch.setattr(install, "FLAGS", SimpleNamespace(dry_run=False, force=False))
    monkeypatch.setattr(install, "get_writable_container_dir", lambda: tmp_path)
    built = []
    monkeypatch.setattr(install, "build_container", lambda *args, **kwargs: built.append((args, kwargs)))
    install.install(["ligandmpnn"])
    assert len(built) == 1
    assert built[0][0] == (mpnn.LIGANDMPNN_CONTAINER, "ligandmpnn", tmp_path)
    assert blocks.get_all_containers()["ligandmpnn"] is mpnn.LIGANDMPNN_CONTAINER


def test_unknown_name_fails_before_building_or_creating_output(monkeypatch):
    monkeypatch.setattr(blocks, "get_container", lambda name: object() if name == "valid" else None)
    monkeypatch.setattr(install, "get_all_containers", lambda: {"valid": object()})

    def forbidden(*args, **kwargs):
        pytest.fail("An invalid install request began building or creating output")

    monkeypatch.setattr(install, "get_writable_container_dir", forbidden)
    monkeypatch.setattr(install, "build_container", forbidden)
    with pytest.raises(app.UsageError, match="Unknown container: misspelled"):
        install.install(["valid", "misspelled"])
