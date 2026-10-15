"""Directory creation tolerates the stale-attribute FileExistsError seen on NFS."""

from pathlib import Path

import pytest

from chaperone.block_config.manifest import Manifest
from chaperone.common import mkdir_p


def _stale_mkdir(self, mode=0o777, parents=False, exist_ok=False):
    # What a stale NFS attribute cache looks like: the directory exists but the
    # post-EEXIST is_dir() recheck says otherwise.
    raise FileExistsError(str(self))


def test_creates_nested_directories_and_is_idempotent(tmp_path):
    target = tmp_path / "a" / "b" / "c"

    assert mkdir_p(target) == target
    assert mkdir_p(target) == target
    assert target.is_dir()


def test_swallows_stale_file_exists_error(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "mkdir", _stale_mkdir)

    assert mkdir_p(tmp_path / "shared") == tmp_path / "shared"


def test_other_errors_still_propagate(tmp_path, monkeypatch):
    def denied(self, mode=0o777, parents=False, exist_ok=False):
        raise PermissionError(str(self))

    monkeypatch.setattr(Path, "mkdir", denied)

    with pytest.raises(PermissionError):
        mkdir_p(tmp_path / "shared")


def test_standalone_manifest_writer_survives_a_stale_mkdir(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    shared.mkdir()
    monkeypatch.setattr(Path, "mkdir", _stale_mkdir)

    path = Manifest(name="step", command=["echo", "hi"], cwd=str(tmp_path)).write(
        shared
    )

    assert path.is_file()
