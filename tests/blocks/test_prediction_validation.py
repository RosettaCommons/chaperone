"""Focused tests for validation reference preparation and RF3 lDDT scoring."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from biotite.structure import AtomArray

from blocks.networks.prediction.validation.lddt import rf3_allatom_lddt
from blocks.networks.prediction.validation.manifest import (
    ManifestValStructuresBlock,
    VALIDATION_BUCKETS,
    VALIDATION_COLUMNS,
)
from blocks.networks.prediction.validation.score import (
    ScoreValStructuresBlock,
    ValidationMetric,
)
from chaperone import InlineMetricBlock
from chaperone.io import (
    clean_atom_array,
    disambiguate_assembly_chain_ids,
    parse_pdb_assembly,
    pdb_mirror_path,
)


def _validation_frame(**overrides) -> pd.DataFrame:
    row = {
        "pdb_id": "9etw",
        "assembly_id": "1",
        "interfaces_to_score": "[('A_1', 'B_1', 'protein-protein')]",
        "pn_units_to_score": "[('A_1', 'protein'), ('B_1', 'protein')]",
        "example_id": "9etw_1",
        "bucket": "protein-protein",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _two_chain_atom_array() -> AtomArray:
    atoms = AtomArray(6)
    atoms.coord = np.array(
        [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [3.0, 4.0, 0.0],
            [6.0, 4.0, 0.0],
        ]
    )
    atoms.chain_id = np.array(["A", "A", "A", "B", "B", "B"])
    atoms.res_id = np.array([1, 2, 3, 1, 2, 3])
    atoms.res_name = np.array(["ALA"] * 6)
    atoms.atom_name = np.array(["CA"] * 6)
    atoms.element = np.array(["C"] * 6)
    atoms.set_annotation("chain_iid", np.array(["A_1"] * 3 + ["B_1"] * 3))
    atoms.set_annotation("token_id", np.arange(6))
    atoms.set_annotation("occupancy", np.ones(6))
    return atoms


def test_prepare_bucket_defaults_to_all_and_can_select_protein(tmp_path):
    parquet = tmp_path / "validation.parquet"
    frame = pd.concat(
        [
            _validation_frame(
                pdb_id="8urn",
                interfaces_to_score="[]",
                pn_units_to_score="[('A_1', 'protein')]",
                bucket="protein",
            ),
            _validation_frame(pdb_id="9etw", bucket="protein-protein"),
        ],
        ignore_index=True,
    )
    frame.to_parquet(parquet)

    all_rows = ManifestValStructuresBlock(
        parquet_path=str(parquet), pdb_mirror=str(tmp_path)
    ).init_handler()
    protein_rows = ManifestValStructuresBlock(
        parquet_path=str(parquet), pdb_mirror=str(tmp_path), bucket="protein"
    ).init_handler()

    assert len(all_rows) == 2
    assert len(protein_rows) == 1
    assert protein_rows.unpack().iloc[0]["validation.pdb_id"] == "8urn"
    assert protein_rows.unpack().iloc[0]["validation.bucket"] == "protein"


def test_prepare_rejects_unknown_bucket(tmp_path):
    parquet = tmp_path / "validation.parquet"
    _validation_frame().to_parquet(parquet)
    block = ManifestValStructuresBlock(
        parquet_path=str(parquet), pdb_mirror=str(tmp_path), bucket="not_a_bucket"
    )

    with pytest.raises(ValueError, match="unknown validation bucket"):
        block.init_handler()


def test_prepare_checks_source_when_loading(tmp_path):
    block = ManifestValStructuresBlock(
        parquet_path=str(tmp_path / "missing.parquet"), pdb_mirror=str(tmp_path)
    )

    with pytest.raises(FileNotFoundError, match="missing.parquet"):
        block.init_handler()


def test_prepare_rejects_missing_parquet_column(tmp_path):
    parquet = tmp_path / "validation.parquet"
    _validation_frame().drop(columns="interfaces_to_score").to_parquet(parquet)
    block = ManifestValStructuresBlock(
        parquet_path=str(parquet), pdb_mirror=str(tmp_path)
    )

    with pytest.raises(ValueError, match="interfaces_to_score"):
        block.init_handler()


def test_prepare_carries_assembly_and_scoring_fields(tmp_path):
    parquet = tmp_path / "validation.parquet"
    _validation_frame().to_parquet(parquet)
    block = ManifestValStructuresBlock(
        parquet_path=str(parquet), pdb_mirror=str(tmp_path)
    )

    row = block.init_handler().unpack().iloc[0]

    assert row["validation.pdb_id"] == "9etw"
    assert row["validation.assembly_id"] == "1"
    assert row["validation.interfaces_to_score"].startswith("[('A_1', 'B_1'")
    assert row["validation.pn_units_to_score"].startswith("[('A_1', 'protein')")


def test_pdb_mirror_path_is_canonical_and_has_no_download_fallback(tmp_path):
    mirrored = tmp_path / "et" / "9etw.cif.gz"
    mirrored.parent.mkdir()
    mirrored.touch()

    assert pdb_mirror_path("9ETW", tmp_path) == mirrored
    with pytest.raises(FileNotFoundError, match="8abc"):
        pdb_mirror_path("8abc", tmp_path)


def test_parse_pdb_assembly_uses_rcsb_settings(monkeypatch, tmp_path):
    mirrored = tmp_path / "et" / "9etw.cif.gz"
    mirrored.parent.mkdir()
    mirrored.touch()
    expected = _two_chain_atom_array()
    called = {}

    def fake_parse(path, *, config):
        called.update(path=path, config=config)
        return {"assemblies": {"1": [expected]}}

    monkeypatch.setattr("atomworks.io.parse", fake_parse)

    assert (
        parse_pdb_assembly(
            "9etw",
            "1",
            tmp_path,
            clean_atom_array=False,
            disambiguate_assembly_chain_ids=False,
        )
        is expected
    )
    config = called["config"]
    assert called["path"] == mirrored
    assert config.build_assembly == ("1",)
    assert config.convert_mse_to_met is True
    assert config.remove_waters is False
    assert config.hydrogen_policy == "keep"


def test_clean_inputs_removes_solvents_and_hydrogens():
    atoms = AtomArray(4)
    atoms.coord = np.zeros((4, 3))
    atoms.chain_id = np.array(["A"] * 4)
    atoms.res_id = np.arange(1, 5)
    atoms.res_name = np.array(["ALA", "ALA", "HOH", "GOL"])
    atoms.atom_name = np.array(["CA", "H", "O", "C1"])
    atoms.element = np.array(["C", "H", "O", "C"])

    cleaned = clean_atom_array(atoms)

    assert cleaned.atom_name.tolist() == ["CA"]


def test_assembly_chain_iids_disambiguate_repeated_author_chain_ids():
    atoms = _two_chain_atom_array()
    atoms.chain_id[:] = "A"
    atoms.chain_iid[:3] = "A_1"
    atoms.chain_iid[3:] = "A_2"

    disambiguated = disambiguate_assembly_chain_ids(atoms)

    assert disambiguated.chain_id.tolist() == ["A"] * 3 + ["B"] * 3
    assert atoms.chain_id.tolist() == ["A"] * 6


def test_rf3_allatom_lddt_scores_global_units_and_interfaces():
    reference = _two_chain_atom_array()
    prediction = reference.copy()

    result = rf3_allatom_lddt(
        reference,
        prediction,
        interfaces_to_score="[('A_1', 'B_1', 'protein-protein')]",
        pn_units_to_score="[('A_1', 'protein'), ('B_1', 'protein')]",
    )

    assert result["rf3_all_atom_lddt"]["best_of_1_lddt"] == pytest.approx(1.0)
    assert [entry["pn_units"] for entry in result["rf3_by_type_lddt"]] == [
        ["A_1", "B_1"],
        ["A_1"],
        ["B_1"],
    ]
    assert all(
        entry["best_of_1_lddt"] == pytest.approx(1.0)
        for entry in result["rf3_by_type_lddt"]
    )


def test_rf3_lddt_uses_chain_id_for_disambiguated_assembly_units():
    reference = _two_chain_atom_array()
    reference.chain_id[:3] = "A_1"
    reference.chain_id[3:] = "A_2"
    reference.chain_iid[:3] = "A_1_1"
    reference.chain_iid[3:] = "A_2_1"

    result = rf3_allatom_lddt(
        reference,
        reference.copy(),
        interfaces_to_score="[('A_1', 'A_2', 'protein-protein')]",
        pn_units_to_score="[('A_2', 'protein')]",
    )

    assert [entry["pn_units"] for entry in result["rf3_by_type_lddt"]] == [
        ["A_1", "A_2"],
        ["A_2"],
    ]
    assert all(
        entry["best_of_1_lddt"] == pytest.approx(1.0)
        for entry in result["rf3_by_type_lddt"]
    )


def test_score_val_structures_is_in_place_and_uses_carried_schema():
    reference = _two_chain_atom_array()
    validation = _validation_frame().iloc[0].to_dict()
    loaded = (
        SimpleNamespace(atom_array=reference, metrics={"validation": validation}),
        SimpleNamespace(
            atom_array=reference.copy(),
            metrics={"validation": validation},
            metadata={},
        ),
    )
    block = ScoreValStructuresBlock()

    result = block.compute_metrics(loaded)

    assert isinstance(block, InlineMetricBlock)
    assert block._creates_new_step is False
    assert result["rf3_all_atom_lddt"]["best_of_1_lddt"] == pytest.approx(1.0)


def test_validation_blocks_expose_the_renamed_registry_surface(tmp_path):
    manifest = ManifestValStructuresBlock(
        parquet_path=str(tmp_path / "validation.parquet"),
        pdb_mirror=str(tmp_path),
    )
    score = ScoreValStructuresBlock()

    assert manifest._name == "manifest_val_structures"
    assert score._name == "score_val_structures"
    assert score.metrics == [ValidationMetric.RF3_ALLATOM_LDDT]
    assert score.remap_outputs_to_input is False


def test_rf3_file_mode_stages_emitted_cif_without_rewriting(tmp_path):
    from blocks.networks.prediction.rosettafold3 import RosettaFold3Block
    from chaperone import PipelineDataHandler

    emitted_cif = tmp_path / "emitted.cif"
    emitted_cif.write_text("data_emitted\n")
    handler = PipelineDataHandler.from_paths(
        [str(emitted_cif)], name="validation_target"
    )
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    block = RosettaFold3Block(command_template=["rf3", "fold"], ckpt_path=None)

    command = block.create_command(handler, sandbox, chunk_id=0)

    (staged,) = (sandbox / "inputs").iterdir()
    assert staged.is_symlink()
    assert staged.resolve() == emitted_cif.resolve()
    assert f"inputs={sandbox / 'inputs'}" in command


def test_score_val_structures_loops_over_atom_array_scorers(monkeypatch):
    reference = _two_chain_atom_array()
    prediction = reference.copy()
    validation = _validation_frame().iloc[0].to_dict()
    loaded = (
        SimpleNamespace(atom_array=reference, metrics={"validation": validation}),
        SimpleNamespace(
            atom_array=prediction,
            metrics={"validation": validation},
            metadata={},
        ),
    )
    calls = []

    def fake_rf3(reference_atoms, prediction_atoms, **kwargs):
        calls.append(("rf3_allatom_lddt", reference_atoms, prediction_atoms, kwargs))
        return {"rf3": 1.0}

    def fake_peppr(reference_atoms, prediction_atoms, **kwargs):
        calls.append(("peppr", reference_atoms, prediction_atoms, kwargs))
        return {"peppr": 2.0}

    monkeypatch.setattr(
        "blocks.networks.prediction.validation.score.rf3_allatom_lddt", fake_rf3
    )
    monkeypatch.setattr(
        "blocks.networks.prediction.validation.score.peppr_metrics", fake_peppr
    )

    result = ScoreValStructuresBlock(
        metrics=["rf3_allatom_lddt", "peppr"]
    ).compute_metrics(loaded)

    assert result == {"rf3": 1.0, "peppr": 2.0}
    assert [call[0] for call in calls] == ["rf3_allatom_lddt", "peppr"]
    assert all(call[1] is reference and call[2] is prediction for call in calls)
    assert calls[0][3] == {
        "interfaces_to_score": validation["interfaces_to_score"],
        "pn_units_to_score": validation["pn_units_to_score"],
        "pn_unit_id_map": None,
    }

    assert calls[1][3] == {
        "min_sequence_identity": 0.0,
        "allow_unmatched_entities": True,
    }


def test_score_val_structures_optionally_remaps_output_to_input():
    assembly = _two_chain_atom_array()
    prediction = assembly.copy()
    prediction.chain_id = np.asarray(["X"] * 3 + ["Y"] * 3)
    prediction.chain_iid = np.asarray(["X_1"] * 3 + ["Y_1"] * 3)
    assembly.res_id += 9
    validation = {
        "interfaces_to_score": "[]",
        "pn_units_to_score": "[('B_1', 'protein')]",
    }
    loaded = (
        SimpleNamespace(atom_array=assembly, metrics={"validation": validation}),
        SimpleNamespace(
            atom_array=prediction,
            metrics={"validation": validation},
            metadata={},
        ),
    )

    result = ScoreValStructuresBlock(remap_outputs_to_input=True).compute_metrics(
        loaded
    )

    assert result["atom_coverage"] == pytest.approx(1.0)
    assert result["rf3_all_atom_lddt"]["best_of_1_lddt"] == pytest.approx(1.0)
    assert result["rf3_by_type_lddt"][0]["pn_units"] == ["B_1"]


def test_score_val_structures_requires_scoring_metadata():
    reference = _two_chain_atom_array()
    validation = _validation_frame().iloc[0].to_dict()
    validation.pop("interfaces_to_score")
    loaded = (
        SimpleNamespace(atom_array=reference, metrics={"validation": validation}),
        SimpleNamespace(
            atom_array=reference.copy(),
            metrics={"validation": validation},
            metadata={},
        ),
    )

    with pytest.raises(ValueError, match="interfaces_to_score"):
        ScoreValStructuresBlock().compute_metrics(loaded)


COHORT_CSV = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "benchmarks"
    / "prediction"
    / "validation"
    / "entry_level_val_df.csv"
)


def test_cohort_csv_has_the_expected_buckets():
    cohort = pd.read_csv(COHORT_CSV)

    assert set(VALIDATION_COLUMNS) <= set(cohort.columns)
    assert set(cohort["bucket"]) == set(VALIDATION_BUCKETS)
    assert cohort["bucket"].value_counts().to_dict() == {
        "protein": 50,
        "protein-protein": 325,
        "protein-nucleic": 164,
        "protein-ligand": 636,
        "misc": 553,
    }


def test_manifest_reads_the_cohort_csv_by_bucket(tmp_path):
    handler = ManifestValStructuresBlock(
        parquet_path=str(COHORT_CSV), pdb_mirror=str(tmp_path), bucket="protein"
    ).init_handler()

    assert len(handler.unpack()) == 50
