"""BindCraft2 input, scheduler, and accepted-refold boundary regressions."""

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from biotite.structure import Atom, array
from biotite.structure.io.pdbx import CIFFile, set_structure

from blocks import Blocks
from blocks.networks.design.bindcraft2.block import BindCraft2Block
from blocks.networks.design.bindcraft2.container import BINDCRAFT2_CONTAINER
from chaperone import PipelineDataHandler
from chaperone.data import make_row


def protein(chains=("C",), count=2):
    return array(
        [
            Atom(
                [float(i), float(r), 0],
                chain_id=chain,
                res_id=r,
                res_name="GLY",
                atom_name=name,
                element=name[0],
            )
            for chain in chains
            for r in range(1, count + 1)
            for i, name in enumerate(("N", "CA", "C", "O"))
        ]
    )


@pytest.fixture
def handler():
    item = SimpleNamespace(
        atom_array=protein(),
        metadata={},
        path="/source/target.cif",
        design_id="parent-id",
        basename="target",
    )
    return SimpleNamespace(get_data=lambda **kwargs: iter([item]), item=item)


@pytest.fixture
def block(tmp_path):
    return BindCraft2Block(
        chain="C", hotspots="1-2", lengths=(2, 2), weights_dir=str(tmp_path / "weights")
    )


def prepare(block, handler, tmp_path, chunk=0):
    command = block.create_command(handler, tmp_path, chunk)
    return command, json.loads((tmp_path / "job.json").read_text())


def ranked(tmp_path, records=None, atoms=None):
    folder = tmp_path / "designs/3_Ranked"
    folder.mkdir(parents=True, exist_ok=True)
    records = (
        records
        if records is not None
        else [{"design": "good_seq1", "Binder_Sequence": "GG", "i_pTM": ".83"}]
    )
    with (folder / "!_Ranked.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["design", "Binder_Sequence", "i_pTM"]
        )
        writer.writeheader()
        writer.writerows(records)
    cif = CIFFile()
    set_structure(cif, protein(("A", "B")) if atoms is None else atoms)
    cif.write(folder / "good_seq1.cif")
    return folder


def test_registry_and_portable_image():
    assert Blocks["bindcraft2"] is BindCraft2Block
    recipe = BINDCRAFT2_CONTAINER.to_singularity()
    assert "cuda12" in recipe and "jax[cuda12]==0.11.0" in recipe
    assert "/net/" not in recipe and "/mnt/home" not in recipe
    assert BINDCRAFT2_CONTAINER._command_config["env"]["JAX_PLATFORMS"] == "cuda"


def test_stages_target_and_owns_single_gpu_budget(block, handler, tmp_path):
    before = handler.item.atom_array.copy()
    command, config = prepare(block, handler, tmp_path, 7)
    assert config["campaign_seed"] == 49 and config["max_trajectories"] == 10
    assert config["design_workers"] == 1 and not config["auto_multi_gpu"]
    assert not config["trajectory_only"] and not config["resume"]
    assert "filters" not in config
    assert Path(config["targets"][0]["target_path"]).is_file()
    assert command[-1] == str(tmp_path / "job.json")
    assert "JAX_COMPILATION_CACHE_DIR" in " ".join(command)
    np.testing.assert_array_equal(before.coord, handler.item.atom_array.coord)


@pytest.mark.parametrize(
    "overrides",
    [
        {"trajectory_only": True},
        {"targets": []},
        {"design_workers": 2},
        {"project_folder": "elsewhere"},
        {"parameter_sweep": {}},
        {"binder_scaffold": "not-staged.cif"},
    ],
)
def test_overrides_cannot_bypass_contract(overrides):
    with pytest.raises(ValueError, match="Unsupported"):
        BindCraft2Block(chain="A", overrides=overrides)


@pytest.mark.parametrize(
    "change,match",
    [
        ("missing", "absent"),
        ("insertion", "insertion"),
        ("ligand", "standard protein"),
        ("nonfinite", "non-finite"),
        ("backbone", "backbone"),
        ("motif", "motif"),
    ],
)
def test_invalid_inputs_fail_before_execution(block, handler, tmp_path, change, match):
    atoms = handler.item.atom_array
    if change == "missing":
        atoms.chain_id[:] = "D"
    if change == "insertion":
        atoms.ins_code[:] = "A"
    if change == "ligand":
        atoms.res_name[:] = "LIG"
    if change == "nonfinite":
        atoms.coord[0, 0] = np.nan
    if change == "backbone":
        atoms.atom_name[0] = "CB"
    if change == "motif":
        handler.item.metadata["motif"] = ["C1"]
    with pytest.raises(ValueError, match=match):
        prepare(block, handler, tmp_path)


@pytest.mark.parametrize("hotspots", ["3", "2-1", "A1", "1,", "0-2"])
def test_hotspots_checked_against_selected_residues(block, handler, tmp_path, hotspots):
    block.hotspots = hotspots
    with pytest.raises(ValueError, match="[Hh]otspot"):
        prepare(block, handler, tmp_path)


def test_incomplete_weights_fail(block, handler, tmp_path):
    block.af2_params_dir = str(tmp_path / "missing")
    with pytest.raises(ValueError, match="checkpoints"):
        prepare(block, handler, tmp_path)


def test_one_input_per_chunk(block, handler, tmp_path):
    handler.get_data = lambda **kwargs: iter([handler.item, handler.item])
    with pytest.raises(ValueError, match="one target"):
        prepare(block, handler, tmp_path)


def test_only_ranked_complexes_with_lineage_and_unique_names(block, handler, tmp_path):
    ids = []
    for chunk in (0, 1):
        root = tmp_path / str(chunk)
        prepare(block, handler, root, chunk)
        folder = ranked(root)
        (folder / "good_seq1_monomer.cif").write_text("do not collect")
        (folder / "rejected.cif").write_text("do not collect")
        result = block.compile(root)
        parent = PipelineDataHandler(
            [
                make_row(
                    name="input",
                    path=handler.item.path,
                    design_id="parent-id",
                    metadata={"ligand": ["C999"], "interface": ["C1"]},
                )
            ]
        )
        result = PipelineDataHandler.merge(parent, result)
        assert len(result) == 1
        row = next(result.get_data(which="last"))
        assert row.parent_design_id == "parent-id"
        assert row.metrics["bindcraft2"]["i_pTM"] == 0.83
        provenance = row.metrics["bindcraft2"]["provenance"]
        assert provenance["binder_chain"] == "B"
        assert provenance["target_chains"] == ["A"]
        assert row.metadata["motif"] == ["A1", "A2"]
        assert row.metadata["ligand"] == [] and row.metadata["interface"] == []
        assert Path(provenance["settings_path"]).is_file()
        assert Path(row.path).name == row.design_id + ".cif"
        ids.append(row.design_id)
    assert ids[0] != ids[1]


@pytest.mark.parametrize(
    "problem",
    [
        "no_table",
        "empty",
        "missing_cif",
        "wrong_sequence",
        "wrong_target",
        "nan",
        "lost_chain",
        "traversal",
        "duplicate",
    ],
)
def test_partial_or_corrupt_results_cannot_pass(block, handler, tmp_path, problem):
    prepare(block, handler, tmp_path)
    records = [{"design": "good_seq1", "Binder_Sequence": "GG", "i_pTM": ".8"}]
    atoms = protein(("A", "B"))
    if problem == "empty":
        records = []
    if problem == "wrong_sequence":
        records[0]["Binder_Sequence"] = "AA"
    if problem == "wrong_target":
        atoms.res_name[atoms.chain_id == "A"] = "ALA"
    if problem == "nan":
        atoms.coord[0, 0] = np.nan
    if problem == "lost_chain":
        atoms = atoms[atoms.chain_id == "A"]
    if problem == "traversal":
        records[0]["design"] = "../good_seq1"
    if problem == "duplicate":
        records *= 2
    folder = ranked(tmp_path, records, atoms)
    if problem == "no_table":
        (folder / "!_Ranked.csv").unlink()
    if problem == "missing_cif":
        (folder / "good_seq1.cif").unlink()
    with pytest.raises(RuntimeError):
        block.compile(tmp_path)
