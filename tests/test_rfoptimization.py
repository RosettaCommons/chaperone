"""RFO's boundaries: input identity, completed cycles, and score provenance."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from biotite.structure import Atom, array

from blocks.networks.rfoptimization.block import RFOptimizationBlock
from blocks.networks.rfoptimization.container import RFOPTIMIZATION_CONTAINER


@pytest.fixture
def handler():
    atoms = array(
        [
            Atom(
                [float(i), 0, 0],
                chain_id=chain,
                res_id=1,
                res_name="GLY",
                atom_name=name,
                element=name[0],
            )
            for chain in ("A", "B")
            for i, name in enumerate(("N", "CA", "C", "O"))
        ]
    )
    item = SimpleNamespace(
        atom_array=atoms,
        metadata={},
        path="/source/original.cif",
        design_id="parent-id",
        basename="original",
    )
    return SimpleNamespace(get_data=lambda **kwargs: iter([item]), item=item)


def build(block, handler, tmp_path):
    block.create_command(handler, tmp_path, 2)
    return yaml.safe_load((tmp_path / "design_0/config.yaml").read_text())


def test_input_staging_and_seed_and_identity(handler, tmp_path):
    block = RFOptimizationBlock(
        backprop_fraction=0, boltz_cache=str(tmp_path / "cache")
    )
    before = handler.item.atom_array.coord.copy()
    config = build(block, handler, tmp_path)
    assert config["seed"] == 42 + 2 * 1000003
    assert config["assets"]["mpnn_model_type"] == "protein_mpnn"
    assert Path(config["input_structure"]).is_file()
    np.testing.assert_array_equal(before, handler.item.atom_array.coord)
    entry = json.loads((tmp_path / "rfo_inputs.json").read_text())[0]
    assert entry["path_handle"] == "/source/original.cif"
    assert entry["parent_design_id"] == "parent-id"


def test_requires_explicit_designed_chain(handler, tmp_path):
    handler.item.atom_array.chain_id[:] = "B"
    with pytest.raises(ValueError, match="designs chain A"):
        build(RFOptimizationBlock(backprop_fraction=0), handler, tmp_path)


def test_does_not_silently_drop_motif_fixing(handler, tmp_path):
    handler.item.metadata["motif"] = ["A1"]
    with pytest.raises(ValueError, match="motif masks"):
        build(RFOptimizationBlock(backprop_fraction=0), handler, tmp_path)


def test_missing_checkpoint_fails_before_gpu(handler, tmp_path):
    with pytest.raises(ValueError, match="checkpoint does not exist"):
        build(
            RFOptimizationBlock(rf3_checkpoint=str(tmp_path / "absent")),
            handler,
            tmp_path,
        )


def test_compile_final_structure_without_pre_redesign_scores(handler, tmp_path):
    block = RFOptimizationBlock(
        backprop_fraction=0, total_cycles=1, boltz_cache=str(tmp_path / "cache")
    )
    build(block, handler, tmp_path)
    output = tmp_path / "design_0/output"
    (output / "inputs").mkdir(parents=True)
    (output / "inputs/start_1.pdb").write_text(
        (tmp_path / "design_0/start.pdb").read_text()
    )
    (output / "start_records.json").write_text(
        json.dumps([{"cycle": 0, "iptm": 0.9, "mpnn_sequence": "G"}])
    )
    result = block.compile(tmp_path)
    row = next(result.get_data(which="last"))
    assert Path(row.path).parent.name == "outputs"
    assert Path(row.path).name.startswith("rfo_")
    assert row.metrics[block.name]["completed_cycles"] == 1
    assert "iptm" not in row.metrics[block.name]
    assert row.parent_design_id == "parent-id"
    assert row.design_id.startswith("parent-id_rfoptimization_")


@pytest.mark.parametrize("records,final", [([], True), ([{"cycle": 0}], False)])
def test_partial_runs_fail_compile(handler, tmp_path, records, final):
    block = RFOptimizationBlock(
        backprop_fraction=0, total_cycles=1, boltz_cache=str(tmp_path / "cache")
    )
    build(block, handler, tmp_path)
    output = tmp_path / "design_0/output"
    (output / "inputs").mkdir(parents=True)
    (output / "start_records.json").write_text(json.dumps(records))
    if final:
        (output / "inputs/start_1.pdb").write_text("ATOM\n")
    with pytest.raises(RuntimeError, match="did not complete"):
        block.compile(tmp_path)


def test_image_is_portable_and_dependency_isolated():
    recipe = RFOPTIMIZATION_CONTAINER.to_singularity()
    assert "atomworks[ml]==2.2.1" in recipe
    assert "/opt/boltz-env" in recipe and "/opt/mpnn-env" in recipe
    assert "/net/" not in recipe and "/mnt/home" not in recipe
    assert "sha256sum -c" in recipe
    assert RFOPTIMIZATION_CONTAINER._command_config.get("binds", []) == []
    assert (
        RFOPTIMIZATION_CONTAINER._command_config["env"]["DISABLE_CUEQUIVARIANCE"] == "1"
    )


def test_failed_record_is_not_a_completed_run(handler, tmp_path):
    block = RFOptimizationBlock(
        backprop_fraction=0, total_cycles=1, boltz_cache=str(tmp_path / "cache")
    )
    build(block, handler, tmp_path)
    output = tmp_path / "design_0/output"
    (output / "inputs").mkdir(parents=True)
    (output / "inputs/start_1.pdb").write_text(
        (tmp_path / "design_0/start.pdb").read_text()
    )
    (output / "start_records.json").write_text(
        json.dumps([{"cycle": 0, "RF3_Status": "Failed"}])
    )
    with pytest.raises(RuntimeError, match="failed cycles"):
        block.compile(tmp_path)


def test_multiple_same_basename_inputs_are_isolated(handler, tmp_path):
    handler.get_data = lambda **kwargs: iter([handler.item, handler.item])
    block = RFOptimizationBlock(
        backprop_fraction=0, boltz_cache=str(tmp_path / "cache")
    )
    build(block, handler, tmp_path)
    paths = json.loads((tmp_path / "rfo_configs.json").read_text())
    configs = [yaml.safe_load(Path(path).read_text()) for path in paths]
    assert len({c["input_structure"] for c in configs}) == 2
    assert len({c["out_dir"] for c in configs}) == 2
    assert len({c["seed"] for c in configs}) == 2
    manifest = json.loads((tmp_path / "rfo_inputs.json").read_text())
    assert len({entry["design_id"] for entry in manifest}) == 2


def test_designed_chain_is_first_for_rf3_without_input_mutation(handler, tmp_path):
    # Upstream RF3 optimizes the first chain whereas MPNN names chain A.
    original = handler.item.atom_array
    handler.item.atom_array = original[np.r_[4:8, 0:4]]
    before = handler.item.atom_array.chain_id.copy()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_text("dummy")
    config = build(
        RFOptimizationBlock(
            rf3_checkpoint=str(checkpoint), boltz_cache=str(tmp_path / "cache")
        ),
        handler,
        tmp_path,
    )
    from biotite.structure.io.pdb import PDBFile

    staged = PDBFile.read(config["input_structure"]).get_structure(model=1)
    assert list(staged.chain_id) == ["A"] * 4 + ["B"] * 4
    np.testing.assert_array_equal(handler.item.atom_array.chain_id, before)


def test_auxiliary_inputs_copied_into_sandbox(handler, tmp_path):
    source = tmp_path / "outside"
    source.mkdir()
    msa = source / "target.a3m"
    msa.write_text(">target\nG\n")
    sandbox = tmp_path / "run"
    config = build(
        RFOptimizationBlock(
            backprop_fraction=0,
            msa_paths={"B": str(msa)},
            boltz_cache=str(tmp_path / "cache"),
        ),
        handler,
        sandbox,
    )
    staged = Path(config["msa_paths"]["B"])
    assert staged.is_relative_to(sandbox)
    assert staged.read_text() == msa.read_text()
    assert staged != msa


def test_compile_rejects_lost_partner(handler, tmp_path):
    from chaperone.io import write_pdb

    block = RFOptimizationBlock(
        backprop_fraction=0, total_cycles=1, boltz_cache=str(tmp_path / "cache")
    )
    build(block, handler, tmp_path)
    output = tmp_path / "design_0/output"
    (output / "inputs").mkdir(parents=True)
    atoms = handler.item.atom_array
    write_pdb(atoms[atoms.chain_id == "A"], output / "inputs/start_1.pdb")
    (output / "start_records.json").write_text(json.dumps([{"cycle": 0}]))
    with pytest.raises(RuntimeError, match="interaction partner"):
        block.compile(tmp_path)


def test_published_basenames_unique_across_inputs_and_chunks(handler, tmp_path):
    handler.get_data = lambda **kwargs: iter([handler.item, handler.item])
    block = RFOptimizationBlock(
        backprop_fraction=0, total_cycles=1, boltz_cache=str(tmp_path / "cache")
    )
    paths = []
    for chunk in (0, 1):
        sandbox = tmp_path / f"chunk_{chunk}"
        block.create_command(handler, sandbox, chunk)
        for index in (0, 1):
            job = sandbox / f"design_{index}"
            output = job / "output"
            (output / "inputs").mkdir(parents=True)
            (output / "inputs/start_1.pdb").write_text((job / "start.pdb").read_text())
            (output / "start_records.json").write_text(json.dumps([{"cycle": 0}]))
        paths.extend(d.path for d in block.compile(sandbox).get_data(which="last"))
    assert len({Path(p).name for p in paths}) == 4


def test_cyclic_repair_guards_source_and_disables_only_geometry_guess(
    tmp_path, monkeypatch
):
    import hashlib
    from blocks.networks.rfoptimization import _compat

    path = tmp_path / "pipeline.py"
    source = 'transform_overrides = {\n            "run_confidence_head": True,\n}\n'
    path.write_text(source)
    with pytest.raises(RuntimeError, match="Unexpected upstream"):
        _compat.repair(path)
    assert path.read_text() == source
    monkeypatch.setattr(
        _compat, "EXPECTED_SHA256", hashlib.sha256(source.encode()).hexdigest()
    )
    _compat.repair(path)
    namespace = {}
    exec(path.read_text(), namespace)
    assert namespace["transform_overrides"] == {
        "run_confidence_head": True,
        "add_cyclic_bonds": False,
        "residue_cache_dir": None,
    }
