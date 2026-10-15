"""Consolidated functional tests for sequence workflows."""

from __future__ import annotations

import json
import polars as pl
import pytest
from blocks.networks.foundry.mpnn import FoundryMPNNBlock
from blocks.metrics.mpnn_score import MpnnScoreBlock
from blocks.networks.mpnn import ClassicMPNNBlock
from blocks.networks.mpnn_io import write_manifest
from blocks.networks.prediction.alphafold3 import AlphaFold3Block
from blocks.networks.prediction.helpers import (
    create_af3_json_from_design,
    create_boltz_yaml_from_design,
    create_chai_fasta_from_design,
    create_colabfold_fasta_from_design,
    create_esmfold2_spec_from_design,
    create_openfold3_json_from_design,
    create_protenix_json_from_design,
    create_rf3_json_from_design,
)
from chaperone.data import (
    PipelineDataHandler,
    SequenceOnlyInputError,
    make_row,
    make_sequence_row,
)
from chaperone.data.schema import Column, get_sequences_from_row
from types import SimpleNamespace
import pandas as pd
from blocks.util_blocks import select_diverse as select_diverse_module
from blocks.util_blocks.package_submission import PackageSubmissionBlock
from blocks.util_blocks.rna_designs import write_rna_design_cif
from blocks.util_blocks.select_diverse import SelectDiverseBlock
from chaperone.io import parse_to_atom_array


# test_sequence_native_mpnn.py


def test_prediction_length_sort_preserves_rewound_context_and_residue_maps():
    rows = [
        make_row(
            name="mpnn",
            path=f"/{design}.cif",
            design_id=design,
            sequences={"A": "ACDE"},
            metadata={"diffused_index_map": json.dumps({"A86": f"A{position}"})},
            metrics={
                "rfd3": {"num_residues": length},
                "af3_apo": {"path": f"/{design}_apo.cif"},
            },
        )
        for design, length, position in [
            ("long", 200, 140),
            ("short", 100, 62),
            ("tied", 100, 80),
        ]
    ]
    handler = PipelineDataHandler(rows)
    handler._producing_steps = ["af3_apo_filter"]
    sorted_handler = AlphaFold3Block().pre_chunk_sort(handler)
    assert sorted_handler._df.to_dicts() == [
        handler._df.to_dicts()[i] for i in [1, 2, 0]
    ]
    assert sorted_handler._producing_steps == handler._producing_steps
    inputs = list(sorted_handler.get_prediction_inputs(load_context=False))
    assert [item.path for item in inputs] == ["/short.cif", "/tied.cif", "/long.cif"]
    assert [
        json.loads(item.metadata["diffused_index_map"])["A86"] for item in inputs
    ] == ["A62", "A80", "A140"]


def test_sequence_table_is_native_and_structure_consumers_fail_clearly(tmp_path):
    handler = PipelineDataHandler.from_sequences(
        pl.DataFrame(
            {
                "design_id": ["design_1"],
                "parent_design_id": ["backbone_1"],
                "sequence_A": ["ACDE"],
                "sequence_B": ["FGHI"],
            }
        )
    )

    row = handler._df.to_dicts()[0]
    assert row[str(Column.STRUCTURES)] == []
    assert get_sequences_from_row(row) == {"A": "ACDE", "B": "FGHI"}
    prediction_input = next(handler.get_prediction_inputs(load_context=False))
    assert prediction_input.design_id == "design_1"
    assert prediction_input.sequences == {"A": "ACDE", "B": "FGHI"}

    with pytest.raises(SequenceOnlyInputError, match="sequence-only"):
        list(handler.get_data())

    csv_path = tmp_path / "sequences.csv"
    handler.write_sequences_csv(csv_path)
    reloaded = PipelineDataHandler.from_file(csv_path, allow_sequences=True)
    assert get_sequences_from_row(reloaded._df.to_dicts()[0]) == {
        "A": "ACDE",
        "B": "FGHI",
    }


def test_classic_mpnn_compiles_generated_fasta_without_structures(tmp_path):
    write_manifest(
        tmp_path,
        {
            "backbone": {
                "input_name": "backbone",
                "parent_design_id": "backbone",
                "chain_ids": ["A", "B"],
                "chain_lengths": [4, 3],
            }
        },
    )
    seq_dir = tmp_path / "seqs"
    seq_dir.mkdir()
    (seq_dir / "backbone.fa").write_text(
        ">backbone, score=0.0\nAAAA/CCC\n"
        ">backbone, id=0, overall_confidence=0.81, ligand_confidence=0.72, seq_rec=0.5\n"
        "ACDE/FGH\n"
    )

    handler = ClassicMPNNBlock().compile(tmp_path)
    row = handler._df.to_dicts()[0]
    assert row[str(Column.STRUCTURES)] == []
    assert row[str(Column.PARENT_DESIGN_ID)] == "backbone"
    assert get_sequences_from_row(row) == {"A": "ACDE", "B": "FGH"}
    assert json.loads(row[str(Column.METRICS)])["mpnn"]["overall_confidence"] == 0.81

    scored = MpnnScoreBlock()(handler, sandbox_dir=tmp_path)
    scored_metrics = json.loads(scored._df.to_dicts()[0][str(Column.METRICS)])
    assert scored_metrics["mpnn_score"]["overall_confidence"] == 0.81


def test_sequence_lineage_backlinks_by_design_id_and_survives_prediction(tmp_path):
    context = tmp_path / "backbone.cif"
    context.write_text("data_backbone\n")
    base = PipelineDataHandler.from_paths([context], name="backbone")
    sequence = PipelineDataHandler(
        [
            make_sequence_row(
                design_id="backbone__mpnn_0",
                parent_design_id="backbone",
                sequences={"A": "ACDE"},
                metrics={"mpnn": {"confidence": 0.8}},
            )
        ]
    )

    designed = PipelineDataHandler.merge(base, sequence)
    designed_row = designed._df.to_dicts()[0]
    assert len(designed_row[str(Column.STRUCTURES)]) == 1
    assert designed_row[str(Column.DESIGN_ID)] == "backbone__mpnn_0"
    assert next(designed.get_prediction_inputs(load_context=False)).sequences == {
        "A": "ACDE"
    }

    predicted = PipelineDataHandler(
        [
            make_row(
                name="af3",
                path=tmp_path / "backbone__mpnn_0_model.cif",
            )
        ]
    )
    folded = PipelineDataHandler.merge(designed, predicted)
    folded_row = folded._df.to_dicts()[0]
    assert folded_row[str(Column.ARTIFACT_KIND)] == "structure"
    assert get_sequences_from_row(folded_row) == {"A": "ACDE"}


def test_foundry_mpnn_compiles_concatenated_fasta_by_parent_chain_lengths(tmp_path):
    write_manifest(
        tmp_path,
        {
            "backbone": {
                "input_name": "backbone",
                "parent_design_id": "parent-id",
                "chain_ids": ["A", "C"],
                "chain_lengths": [4, 3],
            }
        },
    )
    (tmp_path / "backbone.fa").write_text(
        ">backbone_b0_d0, confidence=0.91, sequence_recovery=0.4\nACDEFGH\n"
    )

    handler = FoundryMPNNBlock().compile(tmp_path)
    row = handler._df.to_dicts()[0]
    assert row[str(Column.DESIGN_ID)] == "backbone_b0_d0"
    assert row[str(Column.PARENT_DESIGN_ID)] == "parent-id"
    assert get_sequences_from_row(row) == {"A": "ACDE", "C": "FGH"}


def test_prediction_builders_accept_sequence_only_inputs():
    sequences = {"A": "ACDE", "B": "FGH"}

    af3 = create_af3_json_from_design(None, "d", protein_sequences=sequences)
    assert [entry["protein"]["sequence"] for entry in af3["sequences"]] == [
        "ACDE",
        "FGH",
    ]
    boltz = create_boltz_yaml_from_design(None, protein_sequences=sequences)
    assert len(boltz["sequences"]) == 2
    assert ">protein|name=d_chain_A" in create_chai_fasta_from_design(
        None, "d", protein_sequences=sequences
    )
    assert (
        create_esmfold2_spec_from_design(None, "d", protein_sequences=sequences)[
            "chains"
        ][1]["sequence"]
        == "FGH"
    )
    assert create_colabfold_fasta_from_design(
        None, "d", protein_sequences=sequences
    ).endswith("ACDE\n")
    assert (
        len(
            create_protenix_json_from_design(None, "d", protein_sequences=sequences)[0][
                "sequences"
            ]
        )
        == 2
    )
    assert (
        len(
            create_openfold3_json_from_design(None, "d", protein_sequences=sequences)[
                "d"
            ]["chains"]
        )
        == 2
    )
    assert create_rf3_json_from_design(None, "d", protein_sequences=sequences)[
        "components"
    ] == [
        {"seq": "ACDE", "chain_id": "A"},
        {"seq": "FGH", "chain_id": "B"},
    ]


def test_nucleic_context_sequence_override_does_not_duplicate_chain(tmp_path):
    """An RNA context plus its extracted sequence is still one RNA entity.

    Prediction handlers pass extracted sequences through the historically named
    ``protein_sequences`` argument. A nucleic chain already present in the
    context must therefore count as seen, or builders append a second protein
    with the same chain id and AF3 rejects the input.
    """
    path = write_rna_design_cif("GAGGCC", tmp_path / "rna.cif")
    atoms = parse_to_atom_array(path)

    af3 = create_af3_json_from_design(
        atoms, "rna-design", protein_sequences={"A": "GAGGCC"}
    )
    assert len(af3["sequences"]) == 1
    assert af3["sequences"][0]["rna"]["id"] == "A"
    assert af3["sequences"][0]["rna"]["sequence"] == "GAGGCC"

    boltz = create_boltz_yaml_from_design(atoms, protein_sequences={"A": "GAGGCC"})
    assert boltz["sequences"] == [{"rna": {"id": "A", "sequence": "GAGGCC"}}]


def test_rna_design_cif_is_byte_stable_for_oracle_cache(tmp_path):
    path = tmp_path / "rna.cif"
    write_rna_design_cif("GAGGCC", path, ligands=["MN"])
    first = path.read_bytes()

    write_rna_design_cif("GAGGCC", path, ligands=["MN"])

    assert path.read_bytes() == first
    text = first.decode()
    assert "_entry.date     1970-01-01" in text
    assert "_entry.time     00:00:00" in text


def test_af3_block_reads_sequence_handler_without_structure(tmp_path):
    handler = PipelineDataHandler.from_sequences(
        pl.DataFrame({"design_id": ["sequence-design"], "sequence_A": ["ACDE"]})
    )
    AlphaFold3Block()._prepare_af3_inputs(handler, tmp_path)

    payload = json.loads((tmp_path / "sequence-design.json").read_text())
    assert payload["name"] == "sequence-design"
    assert payload["sequences"][0]["protein"]["sequence"] == "ACDE"


# test_sequence_util_blocks.py


def _no_conflicts(sequences):
    size = len(sequences)
    return [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]


def test_select_diverse_accepts_native_chain_qualified_sequences(monkeypatch):
    monkeypatch.setattr(select_diverse_module, "_identity_matrix", _no_conflicts)
    handler = PipelineDataHandler(
        [
            make_sequence_row(design_id="one", sequences={"A": "AAAA"}),
            make_sequence_row(design_id="two", sequences={"A": "CCCC"}),
            make_sequence_row(design_id="three", sequences={"A": "DDDD"}),
        ]
    )

    selected = SelectDiverseBlock(target_count=2, require_target=True)(handler)

    assert selected.unpack(all=True)["sequence_A"].tolist() == ["AAAA", "CCCC"]


def test_select_diverse_derives_varying_chain_from_structures(tmp_path, monkeypatch):
    monkeypatch.setattr(select_diverse_module, "_identity_matrix", _no_conflicts)
    handler = PipelineDataHandler(
        [
            make_row(name="fold", path=tmp_path / f"{name}.cif", design_id=name)
            for name in ("one", "two", "three")
        ]
    )
    sequence_maps = [
        {"A": "MMMM", "B": "AAAA"},
        {"A": "MMMM", "B": "CCCC"},
        {"A": "MMMM", "B": "DDDD"},
    ]
    monkeypatch.setattr(
        handler,
        "get_prediction_inputs",
        lambda load_context=True: (
            SimpleNamespace(sequences=sequences) for sequences in sequence_maps
        ),
    )

    selected = SelectDiverseBlock(target_count=2, require_target=True)(handler)
    unpacked = selected.unpack(all=True)

    assert unpacked["sequence_A"].tolist() == ["MMMM", "MMMM"]
    assert unpacked["sequence_B"].tolist() == ["AAAA", "CCCC"]


def _mixed_screening_handler() -> PipelineDataHandler:
    return PipelineDataHandler(
        [
            make_row(
                name="rf3",
                path="screened.cif",
                design_id="screened",
                sequences={"A": "ACDEFGHIKLMNPQRSTVWY"},
            ),
            make_row(
                name="mpnn",
                path="unscreened.cif",
                design_id="unscreened",
                sequences={"A": "YWVTSRQPNMLKIHGFEDCA"},
            ),
        ]
    )


def test_select_diverse_excludes_unscreened_topups(monkeypatch):
    monkeypatch.setattr(select_diverse_module, "_identity_matrix", _no_conflicts)
    selected = SelectDiverseBlock(
        max_identity=1.0,
        target_count=1,
        require_target=True,
        required_structure_steps=["rf3"],
    )(_mixed_screening_handler())

    frame = selected.unpack(all=True)
    assert frame["design_id"].tolist() == ["screened"]


def test_select_diverse_fails_when_screened_pool_cannot_meet_target(monkeypatch):
    monkeypatch.setattr(select_diverse_module, "_identity_matrix", _no_conflicts)
    block = SelectDiverseBlock(
        max_identity=1.0,
        target_count=2,
        require_target=True,
        required_structure_steps=["rf3"],
    )

    with pytest.raises(ValueError, match="found 1 mutually dissimilar designs"):
        block(_mixed_screening_handler())


def test_package_submission_resolves_default_sequence_alias(tmp_path):
    handler = PipelineDataHandler(
        [
            make_sequence_row(design_id="one", sequences={"A": "AAAA"}),
            make_sequence_row(design_id="two", sequences={"A": "CCCC"}),
        ]
    )
    output = tmp_path / "submission"

    PackageSubmissionBlock(output_dir=str(output), expected_count=2)(handler)

    written = pd.read_csv(output / "designs.csv")
    assert written.to_dict("records") == [
        {"sequence": "AAAA"},
        {"sequence": "CCCC"},
    ]
