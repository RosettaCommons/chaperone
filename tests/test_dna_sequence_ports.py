"""Regression checks for public DNA and sequence-design recipe configuration."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_example(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "examples" / name / "pipeline.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dna_uses_public_checkpoint_by_default(tmp_path):
    module = load_example("dna-binder")
    config = module.build_rfd3_config(
        module.DNA_SEQUENCE, module.FILES_DIR / "dna_target.pdb", tmp_path
    )
    first, second = module.rounds(config)
    assert first[0].ckpt_path == module.Blocks["rfd3"]().ckpt_path
    assert first[2].model_type == second[0].model_type == "ligand_mpnn"
    assert first[3].diffusion_batch_size == second[1].diffusion_batch_size == 1


def test_examples_request_exactly_one_packed_structure_per_sequence(tmp_path):
    (tmp_path / "input.json").write_text('{"smoke": {"length": 32}}')
    dna = load_example("dna-binder")
    first, second = dna.rounds(tmp_path / "input.json")
    peptide = load_example("peptide-binder")
    pfirst, psecond = peptide.rounds(tmp_path / "input.json")
    multi = load_example("benchmarks/multi-mpnn")
    for block in [first[2], second[0], pfirst[1], psecond[0], *multi.sequence_blocks()]:
        assert block.output_mode == "structures"
        assert block.args.number_of_packs_per_design == 1


def test_whole_duplex_hotspots_use_finite_com_origin(tmp_path):
    import json

    import numpy as np

    from chaperone.io import parse_to_atom_array

    example = load_example("dna-binder")
    target = example.FILES_DIR / "dna_target.pdb"
    spec = json.loads(
        example.build_rfd3_config(example.DNA_SEQUENCE, target, tmp_path).read_text()
    )["dna_binder"]
    assert spec["select_hotspots"] == "A1-12,B1-12"
    assert spec["infer_ori_strategy"] == "com"
    coordinates = parse_to_atom_array(target).coord
    assert np.isfinite(coordinates.mean(axis=0)).all()
    assert spec["select_hbond_donor"] and spec["select_hbond_acceptor"]
