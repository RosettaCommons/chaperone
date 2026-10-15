"""Foundry 1.0 uses native CLIs with source-checked AtomWorks 3 backports."""

import subprocess
from pathlib import Path

import pytest
from blocks.networks.foundry.container import (
    _FOUNDRY_CUDA_CONTAINER,
    _FOUNDRY_PUBLIC_CONTAINER,
    _FOUNDRY_INFERENCE_CHECK,
    _RFD3_COMPAT_COMMAND,
    _RFD3NA_COMPAT_COMMAND,
    _AW3_COMPAT_COMMAND,
)
from blocks.networks.prediction.rosettafold3 import RosettaFold3Block


def test_both_images_check_native_inference_and_formal_charge_serialization():
    for image in (_FOUNDRY_CUDA_CONTAINER, _FOUNDRY_PUBLIC_CONTAINER):
        commands = [cmd for group in image._commands for cmd in group]
        assert _FOUNDRY_INFERENCE_CHECK in commands
        assert _RFD3_COMPAT_COMMAND in commands
        assert "/rfd3/utils/inference.py" in _RFD3_COMPAT_COMMAND
        assert "/rfd3na/utils/inference.py" in _RFD3_COMPAT_COMMAND
        assert _RFD3NA_COMPAT_COMMAND in commands
        assert _AW3_COMPAT_COMMAND in commands
        assert image._labels["RFD3ChargeFix"] == "integer-formal-charge-v1"
        assert not any(
            ("source.write_text" in cmd or "sha256(original" in cmd)
            for cmd in commands
            if cmd not in {_RFD3_COMPAT_COMMAND, _RFD3NA_COMPAT_COMMAND}
        )
    for command in (_FOUNDRY_INFERENCE_CHECK, _RFD3_COMPAT_COMMAND):
        subprocess.run(["sh", "-n"], input=command, text=True, check=True)


@pytest.mark.parametrize(
    "trajectories,msa",
    [
        (False, None),
        (True, None),
        (False, ["/configured/msa"]),
        (True, ["/configured/msa"]),
    ],
)
def test_native_rf3_cli_with_trajectories_and_msa(trajectories, msa):
    block = RosettaFold3Block(dump_trajectories=trajectories, msa_dirs=msa)
    command = block._build_base_cmd("/input.cif", "/output")
    assert "rf3" in command and "fold" in command
    assert not any(
        "compat" in word or "CHAPERONE_LOCAL_MSA_DIRS" in word for word in command
    )
    if msa is not None:
        assert "LOCAL_MSA_DIRS=" + ",".join(msa) in command
        assert "PROTEIN_MSA_DIRS=" + ",".join(msa) in command


def test_repair_entrypoints_are_removed():
    source = Path(__file__).resolve().parents[1] / "src/blocks"
    for path in [
        "networks/foundry/_rf3_portability.py",
        "scripts/rf3_trajectory_compat.py",
    ]:
        assert not (source / path).exists()


def test_all_foundry_images_pin_the_same_1_0_revision():
    from blocks.networks.foundry.container import _FOUNDRY_REVISION
    from blocks.networks.foundry.rfoptimization.container import (
        RFOPTIMIZATION_CONTAINER,
    )

    for image in (
        _FOUNDRY_CUDA_CONTAINER,
        _FOUNDRY_PUBLIC_CONTAINER,
        RFOPTIMIZATION_CONTAINER,
    ):
        recipe = image.to_dockerfile()
        assert f"checkout --detach {_FOUNDRY_REVISION}" in recipe
        assert "refs/pull/404/head" in recipe
        commands = "\n".join(cmd for group in image._commands for cmd in group)
        assert 'version("rc-foundry") == "1.0.0"' in commands
        assert "atomworks3.patch --check" in recipe


def test_aw3_backport_checks_all_sources_before_writing(tmp_path, monkeypatch):
    from hashlib import sha256
    import pytest
    from blocks.networks.foundry import _aw3_compat

    source = tmp_path / "first.py"
    source.write_text("unchanged")
    (tmp_path / "second.py").write_text("modified upstream")
    monkeypatch.setattr(
        _aw3_compat,
        "_SOURCE_HASHES",
        {
            "first.py": sha256(source.read_bytes()).hexdigest(),
            "second.py": sha256(b"reviewed upstream").hexdigest(),
        },
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Patch application started before all sources were checked")

    monkeypatch.setattr(_aw3_compat.subprocess, "run", forbidden)
    with pytest.raises(RuntimeError, match="second.py"):
        _aw3_compat.patch(tmp_path, tmp_path / "unused.patch")
    assert source.read_text() == "unchanged"
