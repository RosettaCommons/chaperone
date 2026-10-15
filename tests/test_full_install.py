"""Core preset and concurrency behavior without downloading scientific images."""

from threading import Barrier, Lock
from types import SimpleNamespace

import pytest
import blocks
from chaperone_cli import install


def test_core_preset_is_available_and_omits_optional_models():
    assert all(
        blocks.get_container(name) is not None for name in install.CORE_CONTAINERS
    )
    assert {"foundry", "ligandmpnn", "alphafold3", "esmfold2"} <= set(
        install.CORE_CONTAINERS
    )
    assert not {"e1", "esm2", "simplefold"} & set(install.CORE_CONTAINERS)


def test_parallel_builds_are_bounded_deduplicated_and_failures_reported(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(install, "get_writable_container_dir", lambda: tmp_path)
    monkeypatch.setattr(
        blocks,
        "get_container",
        lambda name: SimpleNamespace(name="a" if name == "alias" else name),
    )
    barrier = Barrier(2)
    lock = Lock()
    active = 0
    maximum = 0
    completed = []

    def build(spec, name, *args, **kwargs):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        barrier.wait(timeout=5)
        with lock:
            active -= 1
            completed.append(name)
        if name == "a":
            raise RuntimeError("intentional failure")

    monkeypatch.setattr(install, "build_container", build)
    with pytest.raises(RuntimeError, match="a: intentional failure"):
        install.install(["a", "alias", "b", "c", "d"], jobs=2)
    assert maximum == 2
    assert sorted(completed) == ["a", "b", "c", "d"]


def test_full_preset_can_be_extended_without_duplicate_builds(monkeypatch, tmp_path):
    monkeypatch.setattr(install, "get_writable_container_dir", lambda: tmp_path)
    built = []
    monkeypatch.setattr(
        install, "build_container", lambda spec, name, *a, **kw: built.append(name)
    )
    install.install(["foundry", "tmol"], full=True)
    assert sorted(built) == sorted([*install.CORE_CONTAINERS, "tmol"])


def test_cli_runs_warmup_only_after_successful_builds(monkeypatch, tmp_path):
    from chaperone_cli import warmup

    values = dict(
        full=True,
        jobs=2,
        force=False,
        dry_run=True,
        warmup_rfd3=True,
        warmup_cache_dir=str(tmp_path / "cache"),
        warmup_run_dir=str(tmp_path / "run"),
        warmup_local=False,
    )
    monkeypatch.setattr(install, "FLAGS", SimpleNamespace(**values))
    calls = []
    monkeypatch.setattr(
        install, "install", lambda *args, **kwargs: calls.append("build")
    )
    monkeypatch.setattr(warmup, "main", lambda args: calls.append(args))
    install.main(["chaperone"])
    assert calls[0] == "build"
    assert calls[1][-1] == "--dry-run"

    def fail(*args, **kwargs):
        raise RuntimeError("build failed")

    calls.clear()
    monkeypatch.setattr(install, "install", fail)
    with pytest.raises(RuntimeError, match="build failed"):
        install.main(["chaperone"])
    assert calls == []


def test_existing_image_in_secondary_search_directory_is_reused(monkeypatch, tmp_path):
    from chaperone import get_settings
    from chaperone.containers import Image
    import chaperone.containers as containers

    writable = tmp_path / "new"
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "ready.sif").write_bytes(b"existing image")
    monkeypatch.setattr(get_settings(), "CHAPERONE_CONTAINER_BACKEND", "apptainer")
    monkeypatch.setattr(containers, "get_container_dirs", lambda: [writable, shared])
    built = []
    monkeypatch.setattr(install, "_build_apptainer", lambda *a: built.append(a))
    image = Image("ready").base_image("ubuntu:24.04")
    install.build_container(image, "ready", writable)
    assert built == []
    assert not writable.exists()
    install.build_container(image, "ready", writable, force=True)
    assert len(built) == 1
