"""Regression tests for isolated, bounded Proteina-Complexa generation."""

import json
from pathlib import Path

import pytest
import yaml

from blocks import Blocks
from blocks.networks.design.proteina_complexa.proteina_complexa import (
    ProteinaComplexaBlock,
)
from blocks.networks.design.proteina_complexa.container import (
    COMPLEXA_CONTAINER,
    COMPLEXA_REVISION,
)


@pytest.fixture
def block(tmp_path):
    target = tmp_path / "target with space.pdb"
    target.write_text(_pdb([("A", 37), ("A", 39)]))
    weights = tmp_path / "weights"
    weights.mkdir()
    for name in ("complexa.ckpt", "complexa_ae.ckpt"):
        (weights / name).write_text("test checkpoint placeholder")
    return ProteinaComplexaBlock(
        target_pdb=str(target),
        target_selection="A1-115",
        hotspots=["A37", "A39"],
        checkpoint_dir=str(weights),
    )


def test_registry():
    assert Blocks["proteina_complexa"] is ProteinaComplexaBlock


def test_chunk_configuration_isolated_and_seeded(block, tmp_path):
    original = Path(block.target_pdb).read_bytes()
    for chunk in (0, 7):
        sandbox = tmp_path / f"job {chunk}"
        command = block.create_command(block.init_handler(), sandbox, chunk)
        cfg = yaml.safe_load((sandbox / "job.yaml").read_text())
        assert cfg["seed"] == chunk
        assert cfg["hydra"]["job"]["chdir"] is True
        assert cfg["hydra"]["run"]["dir"] == str(sandbox)
        assert cfg["gen_njobs"] == 1
        generation = cfg["generation"]
        assert generation["search"]["algorithm"] == "single-pass"
        assert generation["reward_model"] is None
        assert generation["args"]["nsteps"] == 400
        assert generation["dataloader"]["dataset"]["nres"]["nsamples"] == 1
        assert generation["target_dict_cfg"]["target"]["target_path"] == str(
            sandbox / "target.pdb"
        )
        assert (sandbox / "target.pdb").read_bytes() == original
        assert command[-2:] == [f"--config-path={sandbox}", "--config-name=job"]
    assert Path(block.target_pdb).read_bytes() == original


def test_missing_weights_fail_before_submission(block, tmp_path):
    Path(block.checkpoint_dir, "complexa_ae.ckpt").unlink()
    with pytest.raises(ValueError, match="Missing Proteina-Complexa checkpoint"):
        ProteinaComplexaBlock(**block.model_dump())


@pytest.mark.parametrize("length", [(0, 10), (100, 20)])
def test_invalid_length_rejected(block, length):
    with pytest.raises(ValueError, match="binder_length"):
        ProteinaComplexaBlock(**{**block.model_dump(), "binder_length": length})


def test_compile_does_not_ingest_staged_target_or_partial_output(block, tmp_path):
    sandbox = tmp_path / "job"
    block.create_command(block.init_handler(), sandbox, 0)
    with pytest.raises(RuntimeError, match="produced 0 structures"):
        block.compile(sandbox)


def test_compile_exact_sample_count_and_provenance(block, tmp_path):
    sandbox = tmp_path / "job"
    block.create_command(block.init_handler(), sandbox, 3)
    out = sandbox / "inference" / "job_target" / "job_0_n_195_id_0"
    out.mkdir(parents=True)
    (out / "job_0_n_195_id_0.pdb").write_text(
        _pdb([("A", 1), ("A", 2)] + [("B", i) for i in range(1, 81)])
    )
    handler = block.compile(sandbox)
    assert len(handler) == 1
    meta = json.loads((sandbox / "input_metadata.json").read_text())
    assert meta["seed"] == 3
    assert meta["upstream_revision"] == COMPLEXA_REVISION
    (out / "unexpected.pdb").write_text("END\n")
    with pytest.raises(RuntimeError, match="produced 2 structures"):
        block.compile(sandbox)


def test_recipe_has_no_site_dependencies():
    recipe = COMPLEXA_CONTAINER.to_singularity()
    assert COMPLEXA_REVISION in recipe
    assert "atomworks[ml,openbabel]==2.2.1" in recipe
    for site_path in ("/net/", "/mnt/home", "/projects", "foundry-dev"):
        assert site_path not in recipe


def _pdb(residues):
    lines = []
    for serial, (chain, residue) in enumerate(residues, 1):
        lines.append(
            f"ATOM  {serial:5d}  CA  ALA {chain}{residue:4d}    {serial * 3.8:8.3f}{0:8.3f}{0:8.3f}{1:6.2f}{0:6.2f}           C  "
        )
    return "\n".join(lines) + "\nEND\n"


@pytest.mark.parametrize("hotspots", [["A38"], ["B37"]])
def test_invalid_hotspots_fail_before_gpu(block, tmp_path, hotspots):
    block.hotspots = hotspots
    with pytest.raises(ValueError, match="Hotspots outside"):
        block.create_command(block.init_handler(), tmp_path / "job", 0)


def test_compile_rejects_binder_only(block, tmp_path):
    sandbox = tmp_path / "job"
    block.create_command(block.init_handler(), sandbox, 0)
    out = sandbox / "inference" / "job_target" / "job_0_n_80_id_0"
    out.mkdir(parents=True)
    (out / "sample.pdb").write_text(_pdb([("B", i) for i in range(1, 81)]))
    with pytest.raises(RuntimeError, match="Missing target or binder"):
        block.compile(sandbox)


def test_compile_rejects_target_geometry_change(block, tmp_path):
    sandbox = tmp_path / "job"
    block.create_command(block.init_handler(), sandbox, 0)
    out = sandbox / "inference" / "job_target" / "job_0_n_82_id_0"
    out.mkdir(parents=True)
    pdb = _pdb([("A", 1), ("A", 2)] + [("B", i) for i in range(1, 81)])
    pdb = pdb.replace("   7.600", "  12.600")
    (out / "sample.pdb").write_text(pdb)
    with pytest.raises(RuntimeError, match="Target coordinates changed"):
        block.compile(sandbox)


def test_invalid_hotspot_rejected_at_construction(block):
    with pytest.raises(ValueError, match="Hotspots outside"):
        ProteinaComplexaBlock(**{**block.model_dump(), "hotspots": ["Z999"]})


def test_external_checkpoint_dir_explicitly_mounted(block, tmp_path):
    command = block.create_command(block.init_handler(), tmp_path / "job", 0)
    assert f"{block.checkpoint_dir}:{block.checkpoint_dir}" in command


def test_grouped_batches_rejected_without_silent_design_loss(block, tmp_path):
    block.n_batches = 2
    with pytest.raises(ValueError, match="one batch per chunk"):
        block.create_command(block.init_handler(), tmp_path / "grouped", 0)


def test_chunk_outputs_have_distinct_ids_and_basenames(block, tmp_path):
    identifiers, paths = [], []
    for chunk in (0, 1):
        sandbox = tmp_path / f"chunk_{chunk}"
        block.create_command(block.init_handler(), sandbox, chunk)
        output = sandbox / "inference" / "job_target" / "job_0_n_82_id_0"
        output.mkdir(parents=True)
        native = output / "job_0_n_82_id_0.pdb"
        native.write_text(_pdb([("A", 1), ("A", 2)] + [("B", i) for i in range(1, 81)]))
        handler = block.compile(sandbox)
        row = handler.unpack(to_pandas=False).row(0, named=True)
        identifiers.append(row["design_id"])
        result = next(handler.get_data())
        paths.append(Path(result.path))
        assert Path(result.path).read_bytes() == native.read_bytes()
        assert native.is_file()  # retain upstream CSV/output references
    assert len(set(identifiers)) == 2
    assert len({path.name for path in paths}) == 2
    assert identifiers == [path.stem for path in paths]
