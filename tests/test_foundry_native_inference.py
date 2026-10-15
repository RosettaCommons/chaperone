"""Foundry fixes live upstream; Chaperone uses its unmodified native CLI."""

import subprocess
from pathlib import Path

import pytest
from blocks.networks.foundry.container import (
    _FOUNDRY_CUDA_CONTAINER,
    _FOUNDRY_PUBLIC_CONTAINER,
    _FOUNDRY_INFERENCE_CHECK,
)
from blocks.networks.prediction.rosettafold3 import RosettaFold3Block


def test_both_images_check_native_inference_without_source_rewriting():
    for image in (_FOUNDRY_CUDA_CONTAINER, _FOUNDRY_PUBLIC_CONTAINER):
        commands = [cmd for group in image._commands for cmd in group]
        assert _FOUNDRY_INFERENCE_CHECK in commands
        assert not any(
            "source.write_text" in cmd or "sha256(original" in cmd for cmd in commands
        )
    subprocess.run(["sh", "-n"], input=_FOUNDRY_INFERENCE_CHECK, text=True, check=True)


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
        "networks/foundry/_rfd3na_compat.py",
        "scripts/rf3_trajectory_compat.py",
    ]:
        assert not (source / path).exists()
