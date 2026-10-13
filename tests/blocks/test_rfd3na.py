"""Tests for the RFdiffusion3NA generation block."""

import json

import pytest

from blocks import Blocks
from blocks.networks.foundry.container import FOUNDRY_CONTAINER
from blocks.networks.foundry.rfd3na import RFD3NABlock


def test_registry_discovers_rfd3na_generation_block():
    assert Blocks["rfd3na"] is RFD3NABlock
    assert RFD3NABlock._get_image() is FOUNDRY_CONTAINER
    assert RFD3NABlock._resources.partition == "gpu"
    assert RFD3NABlock._resources.gpu_class == "small"
    assert RFD3NABlock._resources.gres is None

    definition = FOUNDRY_CONTAINER.to_singularity()
    assert "pip install -e '.[all]'" in definition
    assert "rfd3na --help" in definition


def test_rfd3na_builds_safe_default_command_and_input(tmp_path):
    block = RFD3NABlock(
        inputs={
            "mixed": {
                "contig": "10-10R,/0,12-12D,/0,40-50",
                "length": "62-72",
            }
        },
        n_batches=2,
        batch_size=3,
        seed_offset=100,
    )

    command = block.create_command(block.init_handler(), tmp_path, chunk_id=4)

    expected = {
        "rfd3na",
        "design",
        "diffusion_batch_size=3",
        "n_batches=2",
        "seed=104",
        "global_prefix=input_chunk4_",
        "inference_sampler.num_timesteps=200",
        "inference_sampler.use_classifier_free_guidance=False",
        "inference_sampler.cfg_scale=1.5",
        "inference_sampler.step_scale=1.5",
        "read_sequence_from_sequence_head=False",
        "prevalidate_inputs=True",
        "low_memory_mode=False",
        "dump_trajectories=False",
        "skip_existing=True",
        f"out_dir={tmp_path}",
        "ckpt_path=/projects/ml/pipelines/weights/foundry/checkpoints/rfd3na_1190.ckpt",
    }
    assert expected.issubset(command)

    input_path = tmp_path / "inputs" / "input_chunk4.json"
    assert json.loads(input_path.read_text()) == {
        "mixed": {
            "contig": "10-10R,/0,12-12D,/0,40-50",
            "length": "62-72",
        }
    }


def test_rfd3na_rejects_unsafe_design_names(tmp_path):
    block = RFD3NABlock(inputs={"bad.name": {"contig": "20-20R"}})

    with pytest.raises(ValueError, match="Invalid spec"):
        block.create_command(block.init_handler(), tmp_path, chunk_id=0)


def test_rfd3na_rejects_cloud_checkpoint_substitution(tmp_path):
    block = RFD3NABlock(inputs={"rna": {"contig": "20-20R"}}, cloud=True)

    with pytest.raises(NotImplementedError, match="no cloud checkpoint"):
        block.create_command(block.init_handler(), tmp_path, chunk_id=0)


def test_rfd3na_compiles_cif_and_metadata_but_not_trajectories(tmp_path):
    structure = tmp_path / "chunk0_mixed_0_model_0.cif.gz"
    structure.write_bytes(b"")
    (tmp_path / "chunk0_mixed_0_model_0.json").write_text(
        json.dumps(
            {
                "specification": {
                    "input": None,
                    "contig": "20-20R,/0,40-40",
                    "ligand": None,
                    "extra": {"example": "mixed"},
                },
                "diffused_index_map": {"A1": "A1", "B1": "B1"},
                "metrics": {"na_chainbreaks": 0, "radius_of_gyration": 12.5},
            }
        )
    )
    (tmp_path / "chunk0_mixed_0_denoised_model_0.cif.gz").write_bytes(b"")

    unpacked = (
        RFD3NABlock(inputs={"mixed": {"contig": "20-20R"}}).compile(tmp_path).unpack()
    )

    assert len(unpacked) == 1
    assert unpacked["rfd3na.path"].item() == str(structure)
    assert unpacked["rfd3na.metadata.task"].item() == "mixed"
    assert unpacked["rfd3na.na_chainbreaks"].item() == 0
    assert unpacked["rfd3na.radius_of_gyration"].item() == 12.5
