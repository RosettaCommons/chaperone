"""Consolidated functional tests for network blocks."""

from __future__ import annotations

import json
from types import SimpleNamespace
import numpy as np
import pytest
from biotite.structure import AtomArray
from blocks.networks.prediction.alphafold3 import (
    AF3_RUNTIME_SCRIPT_DIR,
    SCRIPT_DIR,
    AlphaFold3Block,
)
from blocks.networks.prediction.af3_with_target_template import (
    AF3WithTargetTemplate,
    _to_cif_string,
)
from pathlib import Path
from blocks import Blocks, get_container
from blocks.networks.design.bindcraft import BindCraftBlock
from pipelines import PipelineDataHandler
from pipelines.data import make_row
from blocks.networks.design.boltzgen import BoltzGenBlock
from blocks.networks.mpnn import ClassicMPNNBlock, MODELHUB_CONTAINER
from blocks.networks.hallucination.protein_hunter import (
    PROTEIN_HUNTER_CONTAINER,
    ProteinHunterBlock,
)
from blocks.networks.prediction.rosettafold3 import (
    _RF3_INPUT_MANIFEST,
    _rf3_input_records,
    RF3DevBlock,
    RosettaFold3Block,
    compile_rf3,
)
from blocks.networks.rfdiffusion2 import _RFD2_SIF_CONTAINER


# test_af3_seeds.py


class _Handler:
    def get_data(self, **kwargs):
        yield SimpleNamespace(basename="design", atom_array=object(), metadata={})


def test_af3_runner_uses_explicit_container_bind(monkeypatch):
    monkeypatch.setenv("PIPELINES_CONTAINER_BACKEND", "apptainer")

    command = AlphaFold3Block().get_command_template()

    bind = f"{SCRIPT_DIR}:{AF3_RUNTIME_SCRIPT_DIR}"
    assert bind in command
    assert str(AF3_RUNTIME_SCRIPT_DIR / "run_alphafold_AF.py") in command
    assert str(SCRIPT_DIR / "run_alphafold_AF.py") not in command


def test_af3_model_seeds_are_written_to_generated_json(tmp_path, monkeypatch):
    for block_cls, module in [
        (AlphaFold3Block, "alphafold3"),
        (AF3WithTargetTemplate, "af3_with_target_template"),
    ]:
        monkeypatch.setattr(
            f"blocks.networks.prediction.{module}.create_af3_json_from_design",
            lambda **kwargs: {"name": "design", "modelSeeds": [1], "sequences": []},
        )
        kwargs = {"target_chain": "A"} if block_cls is AF3WithTargetTemplate else {}
        block = block_cls(model_seeds=[101, 202, 303], input_mode="json", **kwargs)
        if isinstance(block, AF3WithTargetTemplate):
            monkeypatch.setattr(block, "_inject_target_template", lambda *args: None)

        block._prepare_af3_inputs(_Handler(), tmp_path)
        assert json.loads((tmp_path / "design.json").read_text())["modelSeeds"] == [
            101,
            202,
            303,
        ]


def test_target_template_cif_supplies_required_numeric_annotations(monkeypatch):
    array = AtomArray(1)
    array.coord = np.zeros((1, 3), dtype=float)
    captured = {}

    def fake_to_cif_string(template, *, id):
        captured["template"] = template
        return f"data_{id}\n"

    monkeypatch.setattr("atomworks.io.utils.io_utils.to_cif_string", fake_to_cif_string)

    _to_cif_string(array, "template")

    template = captured["template"]
    assert template.occupancy.tolist() == [1.0]
    assert template.b_factor.tolist() == [0.0]


def test_af3_model_seeds_reject_invalid_values():
    for seeds in ([], [-1], [7, 7]):
        with pytest.raises(ValueError, match="model_seeds"):
            AlphaFold3Block(model_seeds=seeds)


# test_bindcraft.py


def _target_handler(path: Path) -> PipelineDataHandler:
    path.write_text(
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N\n"
        "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 20.00           C\n"
        "ATOM      3  C   ALA A   1       2.000   1.420   0.000  1.00 20.00           C\n"
        "ATOM      4  O   ALA A   1       1.400   2.450   0.000  1.00 20.00           O\n"
        "END\n"
    )
    return PipelineDataHandler.from_paths([str(path)], name="target")


def test_bindcraft_is_discoverable_and_buildable() -> None:
    assert Blocks["bindcraft"] is BindCraftBlock
    assert get_container("bindcraft").name == "bindcraft"


def test_create_command_writes_pipeline_native_configs(tmp_path: Path) -> None:
    target_path = tmp_path / "target.pdb"
    handler = _target_handler(target_path)
    sandbox = tmp_path / "chunk_0"
    block = BindCraftBlock(
        chains="A",
        target_hotspot_residues="A1",
        lengths=(55, 75),
        number_of_final_designs=2,
        max_trajectories=7,
        advanced_overrides={"soft_iterations": 1},
    )

    command = block.create_command(handler, sandbox, chunk_id=0)

    assert "${CONTAINER:bindcraft}" in command
    options = "settings filters advanced filter-overrides advanced-overrides config-dir"
    assert all(f"--{option}" in command for option in options.split())

    settings = json.loads((sandbox / "configs" / "target.json").read_text())
    assert settings["chains"] == "A"
    assert settings["target_hotspot_residues"] == "A1"
    assert settings["lengths"] == [55, 75]
    assert settings["number_of_final_designs"] == 2
    assert settings["starting_pdb"].endswith("/inputs/target.pdb")
    assert settings["design_path"].endswith("/chunk_0/designs")
    assert json.loads(
        (sandbox / "configs" / "advanced_overrides.json").read_text()
    ) == {"max_trajectories": 7, "soft_iterations": 1}


def _write_fake_outputs(sandbox: Path, *, ranked: bool) -> Path:
    design_dir = sandbox / "designs"
    accepted = design_dir / "Accepted"
    output_dir = accepted / "Ranked" if ranked else accepted
    output_dir.mkdir(parents=True)
    prefix = "1_" if ranked else ""
    name = f"{prefix}binder_chunk0_l65_s123_mpnn1_model2.pdb"
    pdb = output_dir / name
    pdb.write_text("MODEL\nEND\n")
    (design_dir / "final_design_stats.csv").write_text(
        "Rank,Design,Average_i_pTM,Sequence\n1,binder_chunk0_l65_s123_mpnn1,0.81,ACDE\n"
    )
    (sandbox / "bindcraft_manifest.json").write_text(
        json.dumps(
            {
                "parent_design_id": "target-1",
                "source_path": "/inputs/target.pdb",
                "chains": "A",
                "target_hotspot_residues": "A10-12",
            }
        )
    )
    return pdb


def test_compile_emits_accepted_designs_with_metrics_and_lineage(
    tmp_path: Path,
) -> None:
    for ranked in (True, False):
        sandbox = tmp_path / str(ranked)
        expected = _write_fake_outputs(sandbox, ranked=ranked)
        loaded = next(
            BindCraftBlock(chains="A").compile(sandbox).get_data(which="last")
        )

        assert loaded.path == str(expected)
        assert loaded.parent_design_id == "target-1"
        assert loaded.design_id.startswith("target-1__bindcraft_")
        metrics = loaded.metrics["bindcraft"]
        assert metrics["Average_i_pTM"] == 0.81
        assert metrics["accepted"] is True
        assert metrics["target_chains"] == "A"


def test_compile_prefers_ranked_outputs(tmp_path: Path) -> None:
    ranked = _write_fake_outputs(tmp_path, ranked=True)
    accepted = tmp_path / "designs" / "Accepted" / "unranked_model1.pdb"
    accepted.write_text("MODEL\nEND\n")

    handler = BindCraftBlock(chains="A").compile(tmp_path)

    assert len(handler) == 1
    assert next(handler.get_data(which="last")).path == str(ranked)


def test_compile_rejects_empty_run(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="produced no accepted PDB"):
        BindCraftBlock(chains="A").compile(tmp_path)


# test_boltzgen.py


def test_boltzgen_uses_stable_container_tmpdir(tmp_path: Path) -> None:
    config = tmp_path / "design.yaml"
    config.write_text("entities: []\n")

    block = BoltzGenBlock(config=str(config), num_designs=1, budget=1)

    assert "--env" in block.command_template
    assert "TMPDIR=/tmp" in block.command_template


# test_mpnn_container.py


def test_existing_modelhub_image_uses_exec_not_runscript():
    if not MODELHUB_CONTAINER.is_existing:
        return

    command = ClassicMPNNBlock(num_seq_per_pdb=1).get_command_template()

    assert command[:2] == ["apptainer", "exec"]
    assert command[-3:-1] == ["python", "-u"]
    assert command[-1].endswith("/fused_mpnn/v1.1.1/run.py")


def test_explicit_fixed_residues_are_merged_into_multi_input_constraints(
    tmp_path,
):
    structure = tmp_path / "input.pdb"
    structure.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "END\n"
    )
    handler = PipelineDataHandler.from_paths([str(structure)], name="input")
    block = ClassicMPNNBlock(
        num_seq_per_pdb=1,
        keep_motif_sequences_fixed=False,
        args={"fixed_residues": "A1 A7"},
    )

    command = block.create_command(handler, tmp_path / "chunk_0", chunk_id=0)

    constraints = command[command.index("--fixed_residues_multi") + 1]
    payload = json.loads(
        (tmp_path / "chunk_0" / "inputs" / "pdb_position_fixed_chunk0.json").read_text()
    )
    assert constraints.endswith("pdb_position_fixed_chunk0.json")
    assert next(iter(payload.values())) == ["A1", "A7"]


# test_protein_hunter.py


def test_container_recipe_pins_a_consistent_cuda_runtime():
    definition = PROTEIN_HUNTER_CONTAINER.to_singularity()

    expected = (
        "a3e048e96c93953f022d15a4d35cd744b7e39e41 torch==2.11.0 "
        "https://download.pytorch.org/whl/cu128 cuda-toolkit-12-8 pip check"
    ).split() + ["CUDA 12.8"]
    assert all(value in definition for value in expected)


def _fake_cycles(sandbox: Path) -> None:
    run = sandbox / "0_protein_hunter_design" / "run_0"
    run.mkdir(parents=True)
    for cycle in range(3):
        (run / f"design_predicted_cycle_{cycle}.pdb").write_text("MODEL\nEND\n")


def test_compile_can_emit_each_protein_hunter_cycle(tmp_path: Path):
    _fake_cycles(tmp_path)
    (tmp_path / "summary_all_runs.csv").write_text(
        "run_id,cycle_0_iptm,cycle_1_iptm,cycle_2_iptm,best_seq\n0,0.5,0.7,0.9,AAAA\n"
    )

    handler = ProteinHunterBlock(ligand_ccd="ADN", emit_all_cycles=True).compile(
        tmp_path
    )

    assert len(handler) == 3
    unpacked = handler.unpack()
    assert "protein_hunter_design.path" in unpacked.columns
    assert not any("/cycle_" in column for column in unpacked.columns)
    assert unpacked["protein_hunter_design.iptm"].tolist() == [0.5, 0.7, 0.9]
    cycles = unpacked["protein_hunter_design.metadata.protein_hunter_cycle"]
    assert cycles.tolist() == [0, 1, 2]


def test_compile_keeps_trajectory_lineage_by_default(tmp_path: Path):
    _fake_cycles(tmp_path)

    handler = ProteinHunterBlock(ligand_ccd="ADN").compile(tmp_path)

    assert len(handler) == 1


def test_compile_accepts_cycle_zero_only_diagnostic(tmp_path: Path):
    run = tmp_path / "0_protein_hunter_design" / "run_0"
    run.mkdir(parents=True)
    (run / "design_predicted_cycle_0.pdb").write_text("MODEL\nEND\n")
    (tmp_path / "summary_all_runs.csv").write_text(
        "run_id,cycle_0_iptm,cycle_0_plddt\n0,0.42,0.56\n"
    )

    handler = ProteinHunterBlock(num_cycles=0).compile(tmp_path)

    assert len(handler) == 1
    unpacked = handler.unpack()
    assert unpacked["protein_hunter_design.cycle_0_iptm"].item() == 0.42
    assert "protein_hunter_design/cycle_0.path" in unpacked.columns


# test_rf3_seeds.py


def test_rf3_single_seed_uses_supported_seed_setting():
    for block_cls, seed_arg in (
        (RosettaFold3Block, "seed=20"),
        (RF3DevBlock, "+seed=20"),
    ):
        command = block_cls(seed_offset=17, model_seeds=None)._append_common_cli_args(
            ["rf3", "fold"], chunk_id=3
        )
        assert seed_arg in command
        assert not any("model_seeds" in arg for arg in command)


def test_rf3_model_seeds_expand_to_independent_invocations():
    base = ["rf3", "fold", "inputs=in.json", "out_dir=outputs"]
    for block_cls, seed_prefix in ((RosettaFold3Block, ""), (RF3DevBlock, "+")):
        command = block_cls(model_seeds=[101, 202])._expand_model_seed_commands(
            base, "outputs"
        )
        assert command[:2] == ["bash", "-lc"]
        assert all(f"out_dir=outputs/seed_{seed}" in command[2] for seed in (101, 202))
        assert all(f"{seed_prefix}seed={seed}" in command[2] for seed in (101, 202))
        assert " && " in command[2]


def test_rf3_all_samples_excludes_top_model_alias(tmp_path):
    design = tmp_path / "outputs" / "seed_101" / "design"
    (design / "seed-101_sample-0").mkdir(parents=True)
    (design / "seed-101_sample-1").mkdir()
    (design / "design_model.cif").write_text("alias")
    (design / "seed-101_sample-0" / "design_seed-101_sample-0_model.cif").write_text(
        "sample 0"
    )
    (design / "seed-101_sample-1" / "design_seed-101_sample-1_model.cif").write_text(
        "sample 1"
    )

    result = compile_rf3(tmp_path, read_most_confident_model=False)

    assert len(result) == 2
    assert all("_sample-" in row.path for row in result.get_data())
    _check_rf3_most_confident_models_support_multi_seed_layout(tmp_path / "multi")


def _check_rf3_most_confident_models_support_multi_seed_layout(tmp_path):
    for seed in (101, 202):
        design = tmp_path / "outputs" / f"seed_{seed}" / "design"
        design.mkdir(parents=True)
        (design / "design_model.cif").write_text(f"seed {seed}")
        sample = design / f"seed-{seed}_sample-0"
        sample.mkdir()
        (sample / f"design_seed-{seed}_sample-0_model.cif").write_text("sample")

    result = compile_rf3(tmp_path, read_most_confident_model=True)

    assert len(result) == 2
    assert {row.path for row in result.get_data()} == {
        str(tmp_path / "outputs" / "seed_101" / "design" / "design_model.cif"),
        str(tmp_path / "outputs" / "seed_202" / "design" / "design_model.cif"),
    }


def test_rf3_staged_names_disambiguate_duplicate_basenames_and_ids():
    items = [
        SimpleNamespace(
            basename="design",
            design_id="duplicate",
            path=f"/branch-{branch}/design.cif",
            sequence={"A": sequence},
        )
        for branch, sequence in ((1, "AAAA"), (2, "CCCC"))
    ]

    names = [name for _, name in _rf3_input_records(items)]

    assert len(set(names)) == 2
    assert all(name.startswith("design__rf3_") for name in names)


def test_rf3_manifest_backlinks_duplicate_ids_by_exact_source_path(tmp_path):
    source_paths = ["/branch-1/design.cif", "/branch-2/design.cif"]
    base = PipelineDataHandler(
        [
            make_row(name="input", path=path, design_id="duplicate")
            for path in source_paths
        ]
    )
    items = [
        SimpleNamespace(
            basename="design",
            design_id="duplicate",
            path=path,
            sequence={"A": sequence},
        )
        for path, sequence in zip(source_paths, ("AAAA", "CCCC"))
    ]
    records = _rf3_input_records(items)
    manifest = {
        name: {"path_handle": item.path, "parent_design_id": item.design_id}
        for item, name in records
    }
    (tmp_path / _RF3_INPUT_MANIFEST).write_text(json.dumps(manifest))
    for _, name in records:
        output = tmp_path / "outputs" / name
        output.mkdir(parents=True)
        (output / f"{name}_model.cif").write_text(name)

    predicted = compile_rf3(tmp_path)
    merged = PipelineDataHandler.merge(base, predicted).unpack(all=True)

    assert len(merged) == 2
    assert set(merged["input.path"]) == set(source_paths)
    for source, (_, staged_name) in zip(source_paths, records):
        row = merged.loc[merged["input.path"] == source].iloc[0]
        assert staged_name in row["rf3.path"]


# test_rfdiffusion2_container.py


def test_rfdiffusion2_rebinds_external_code_and_weights():
    command = _RFD2_SIF_CONTAINER.build_command("python", "--version")

    assert (
        "/net/tukwila/rohith/RFdiffusion2:/net/tukwila/rohith/RFdiffusion2" in command
    )
    assert "/net/databases/lab/aa-diffusion:/net/databases/lab/aa-diffusion" in command
