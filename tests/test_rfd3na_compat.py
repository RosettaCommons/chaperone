"""Patch application safeguards; the image build tests the real Foundry parser."""

from hashlib import sha256

import pytest

from blocks.networks.foundry import _rfd3na_compat as compat


@pytest.fixture
def source(tmp_path, monkeypatch):
    patches = []
    for name, old, new in (
        ("input.py", "old input", "new input"),
        ("tokens.py", "old tokens", "new tokens"),
    ):
        original = f"header\n{old}\nfooter\n"
        (tmp_path / name).write_text(original)
        patches.append((name, sha256(original.encode()).hexdigest(), old, new))
    monkeypatch.setattr(compat, "_PATCHES", tuple(patches))
    return tmp_path


def test_repair_is_idempotent(source):
    compat.patch(source)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    assert b"new input" in before["input.py"]
    compat.patch(source)
    assert before == {p.name: p.read_bytes() for p in source.iterdir()}


def test_changed_upstream_refuses_entire_patch(source):
    (source / "tokens.py").write_text("unexpected upstream source")
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    with pytest.raises(RuntimeError, match="Unexpected pinned Foundry source"):
        compat.patch(source)
    assert before == {p.name: p.read_bytes() for p in source.iterdir()}


def test_modified_repair_is_not_silently_overwritten(source):
    compat.patch(source)
    with (source / "input.py").open("a") as file:
        file.write("unexpected edit")
    with pytest.raises(RuntimeError, match="Unexpected pinned Foundry source"):
        compat.patch(source)


def test_both_images_apply_and_check_repair_with_valid_shell():
    import subprocess

    from blocks.networks.foundry.container import (
        _FOUNDRY_CUDA_CONTAINER,
        _FOUNDRY_PUBLIC_CONTAINER,
        _RFD3NA_COMPAT_COMMAND,
    )

    for image in (_FOUNDRY_CUDA_CONTAINER, _FOUNDRY_PUBLIC_CONTAINER):
        commands = [command for group in image._commands for command in group]
        assert _RFD3NA_COMPAT_COMMAND in commands
    subprocess.run(["sh", "-n"], input=_RFD3NA_COMPAT_COMMAND, text=True, check=True)
