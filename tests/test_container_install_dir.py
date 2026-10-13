"""Concurrent container installs must keep the selected output directory."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Lock

import pytest

from chaperone_cli import install


def test_concurrent_installers_probe_independently(monkeypatch, tmp_path):
    cache = tmp_path / "shared-containers"
    cache.mkdir()
    unrelated = cache / ".write_test"
    unrelated.write_text("preserve existing files")
    monkeypatch.setattr("chaperone.containers.get_container_dirs", lambda: [cache])
    count = 8
    rendezvous = Barrier(count)
    lock = Lock()
    probes = []
    real_temporary_file = install.tempfile.NamedTemporaryFile

    def overlapping_probe(*args, **kwargs):
        probe = real_temporary_file(*args, **kwargs)
        with lock:
            probes.append(Path(probe.name))
        try:
            # All probes coexist before any installer can remove its own.
            rendezvous.wait(timeout=10)
        except BaseException:
            probe.close()
            raise
        return probe

    monkeypatch.setattr(install.tempfile, "NamedTemporaryFile", overlapping_probe)
    with ThreadPoolExecutor(max_workers=count) as pool:
        selected = list(pool.map(lambda _: install.get_writable_container_dir(), range(count)))
    assert selected == [cache] * count
    assert len(set(probes)) == count
    assert all(path.parent == cache and not path.exists() for path in probes)
    assert unrelated.read_text() == "preserve existing files"
    assert list(cache.iterdir()) == [unrelated]


@pytest.mark.parametrize("has_writable_candidate", [True, False])
def test_unwritable_directory_falls_back(monkeypatch, tmp_path, has_writable_candidate):
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    writable = tmp_path / "writable"
    writable.mkdir()
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)
    candidates = [blocked, writable] if has_writable_candidate else [blocked]
    monkeypatch.setattr("chaperone.containers.get_container_dirs", lambda: candidates)
    real_temporary_file = install.tempfile.NamedTemporaryFile

    def deny_blocked(*args, **kwargs):
        if Path(kwargs["dir"]) == blocked:
            raise PermissionError("read-only container directory")
        return real_temporary_file(*args, **kwargs)

    monkeypatch.setattr(install.tempfile, "NamedTemporaryFile", deny_blocked)
    expected = writable if has_writable_candidate else home / ".chaperone" / "containers"
    assert install.get_writable_container_dir() == expected
    assert expected.is_dir()
    assert list(blocked.iterdir()) == []
