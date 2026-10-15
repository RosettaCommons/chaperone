"""Guard the pinned RF3 cache override; image builds test the actual pipeline."""

from hashlib import sha256
import subprocess

import pytest

from blocks.networks.foundry import _rf3_portability as portability


@pytest.fixture
def source(tmp_path, monkeypatch):
    path = tmp_path / "rf3.py"
    content = "header\n" + portability._OLD + "footer\n"
    path.write_text(content)
    monkeypatch.setattr(portability, "_EXPECTED_SHA256", sha256(content.encode()).hexdigest())
    return path


def test_cache_override_is_idempotent(source):
    portability.patch(source)
    content = source.read_bytes()
    assert b'"residue_cache_dir": None' in content
    portability.patch(source)
    assert source.read_bytes() == content


@pytest.mark.parametrize("after_patch", [False, True])
def test_changed_upstream_or_repair_is_rejected_without_writing(source, after_patch):
    if after_patch:
        portability.patch(source)
    source.write_text(source.read_text() + "unreviewed upstream change\n")
    content = source.read_bytes()
    with pytest.raises(RuntimeError, match="Unexpected pinned Foundry RF3 source"):
        portability.patch(source)
    assert source.read_bytes() == content


def test_both_images_apply_and_check_real_cache_override():
    from blocks.networks.foundry.container import (
        _FOUNDRY_CUDA_CONTAINER,
        _FOUNDRY_PUBLIC_CONTAINER,
        _RF3_PORTABILITY_COMMAND,
    )

    for image in (_FOUNDRY_CUDA_CONTAINER, _FOUNDRY_PUBLIC_CONTAINER):
        commands = [command for group in image._commands for command in group]
        assert _RF3_PORTABILITY_COMMAND in commands
    assert "--check" in _RF3_PORTABILITY_COMMAND
    subprocess.run(["sh", "-n"], input=_RF3_PORTABILITY_COMMAND, text=True, check=True)
