import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
from biotite.structure import AtomArray

from scoring import (
    FixedRegionSpec,
    ScoreReport,
    SequenceSubmissionError,
    SequenceSubmissionSpec,
    VariableRegion,
    attach_measurements_by_design_id,
    ca_radius_of_gyration,
    isolated_scoring_root,
    load_sequence_submission,
    merge_submission_columns,
    run_sequence_scorer,
    sequence_submission_to_handler,
    write_score_outputs,
)
from pipelines.data.handler import PipelineDataHandler
from pipelines.data.schema import make_row


def _write(path: Path, sequences: list[str], column: str = "sequence") -> Path:
    pl.DataFrame(
        {
            "design_id": [f"d{i}" for i in range(len(sequences))],
            column: sequences,
        }
    ).write_csv(path)
    return path


def test_single_chain_uses_sequence_by_default_and_accepts_legacy_alias(
    tmp_path: Path,
) -> None:
    spec = SequenceSubmissionSpec(required_n=2, chains=("A",))
    sequences = ["AAAAAAAAAA", "CCCCCCCCCC"]

    canonical = load_sequence_submission(
        _write(tmp_path / "canonical.csv", sequences), spec
    )
    assert canonical.columns == ["design_id", "sequence_A"]
    assert canonical["sequence_A"].to_list() == sequences

    legacy = load_sequence_submission(
        _write(tmp_path / "legacy.csv", sequences, column="sequence_A"), spec
    )
    assert legacy.to_dicts() == canonical.to_dicts()

    both = pl.DataFrame(
        {
            "design_id": ["d0", "d1"],
            "sequence": sequences,
            "sequence_A": sequences,
        }
    )
    path = tmp_path / "both.csv"
    both.write_csv(path)
    with pytest.raises(SequenceSubmissionError, match="supply either 'sequence'"):
        load_sequence_submission(path, spec)


def test_pairwise_identity_allows_boundary_and_rejects_close_siblings(
    tmp_path: Path,
) -> None:
    spec = SequenceSubmissionSpec(required_n=2, chains=("A",))
    submission = _write(tmp_path / "boundary.csv", ["AAAAAAAAAA", "AAAAAAACCC"])
    assert len(load_sequence_submission(submission, spec)) == 2

    submission = _write(tmp_path / "too_similar.csv", ["AAAAAAAAAA", "AAAAAAAACC"])
    with pytest.raises(SequenceSubmissionError, match="pairwise global identity"):
        load_sequence_submission(submission, spec)


def test_fixed_framework_similarity_uses_only_designed_regions(tmp_path: Path) -> None:
    fixed = FixedRegionSpec(
        reference={"H": "MMMMMAAAAQQQQQ"},
        regions=(VariableRegion("CDR-H", "H", 6, 9),),
    )
    spec = SequenceSubmissionSpec(
        required_n=2,
        chains=("H",),
        fixed_regions=fixed,
    )

    # The full sequences are >70% identical because their fixed framework is
    # identical, but their designed regions share no identity and must pass.
    submission = _write(
        tmp_path / "diverse_cdrs.csv",
        ["MMMMMCCCCQQQQQ", "MMMMMDDDDQQQQQ"],
        column="sequence_H",
    )
    assert len(load_sequence_submission(submission, spec)) == 2

    submission = _write(
        tmp_path / "similar_cdrs.csv",
        ["MMMMMCCCCQQQQQ", "MMMMMCCCDQQQQQ"],
        column="sequence_H",
    )
    with pytest.raises(
        SequenceSubmissionError, match="concatenated designed-region sequence"
    ):
        load_sequence_submission(submission, spec)


def _ribozyme_spec() -> SequenceSubmissionSpec:
    return SequenceSubmissionSpec(
        required_n=1,
        chains=("A",),
        molecule_type="rna",
        sequence_length_range=(50, 90),
        allow_motif_indices=False,
        motif_reference_ids=("A20", "A21", "A22", "A36", "B6", "B7"),
        motif_residue_types={
            "A20": "G",
            "A21": "A",
            "A22": "G",
            "A36": "G",
            "B6": "C",
            "B7": "C",
        },
        motif_contiguity_groups=(("A20", "A21", "A22"), ("B6", "B7")),
        motif_map_format="zero_based_indices",
    )


def _write_ribozyme(path: Path, sequence: str) -> Path:
    motif_map = '{"A20":0,"A21":1,"A22":2,"A36":10,"B6":20,"B7":21}'
    pl.DataFrame(
        {"design_id": ["rz0"], "sequence": [sequence], "motif_map": [motif_map]}
    ).write_csv(path)
    return path


def test_rna_motif_map_rejects_a_mutated_motif_base(tmp_path: Path) -> None:
    # Positions 0-2 are GAG, position 10 is G, and positions 20-21 are CC.
    valid = "GAG" + "A" * 7 + "G" + "A" * 9 + "CC" + "A" * 28
    mutated = "C" + valid[1:]
    submission = _write_ribozyme(tmp_path / "mutated.csv", mutated)

    with pytest.raises(SequenceSubmissionError, match="expected G"):
        load_sequence_submission(submission, _ribozyme_spec())


def test_valid_rna_submission_materializes_as_an_rna_chain(tmp_path: Path) -> None:
    from atomworks.enums import ChainType
    from pipelines.data.schema import Column
    from pipelines.io import parse_to_atom_array

    sequence = "GAG" + "A" * 7 + "G" + "A" * 9 + "CC" + "A" * 28
    submission = load_sequence_submission(
        _write_ribozyme(tmp_path / "valid.csv", sequence), _ribozyme_spec()
    )
    handler = sequence_submission_to_handler(
        submission,
        tmp_path / "run",
        molecule_type="rna",
        ligands=("MN",),
    )

    assert len(handler) == 1
    structures_row = handler.to_pandas().iloc[0][str(Column.STRUCTURES)]
    metadata = structures_row[0]["metadata"]
    diffused_index_map = metadata["diffused_index_map"]
    if isinstance(diffused_index_map, str):
        diffused_index_map = json.loads(diffused_index_map)
    assert {
        reference: diffused_index_map[reference]
        for reference in _ribozyme_spec().motif_reference_ids
    } == {
        "A20": "A1",
        "A21": "A2",
        "A22": "A3",
        "A36": "A11",
        "B6": "A21",
        "B7": "A22",
    }
    structures = list((tmp_path / "run" / "submission_inputs").glob("*.cif"))
    assert len(structures) == 1
    atoms = parse_to_atom_array(str(structures[0]))
    rna_chain = atoms[atoms.chain_id == "A"]
    assert len(rna_chain) == len(sequence)
    assert set(rna_chain.chain_type) == {ChainType.RNA}


def test_ca_radius_of_gyration_uses_requested_chain_ca_atoms(monkeypatch) -> None:
    atoms = AtomArray(5)
    atoms.chain_id = np.array(["A", "A", "A", "B", "A"])
    atoms.atom_name = np.array(["CA", "CA", "CB", "CA", "N"])
    atoms.coord = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [200.0, 0.0, 0.0],
            [300.0, 0.0, 0.0],
        ]
    )
    monkeypatch.setattr("pipelines.io.parse_to_atom_array", lambda _: atoms)

    assert ca_radius_of_gyration("prediction.cif", chain_id="A") == pytest.approx(1.0)


def test_ca_radius_of_gyration_requires_two_ca_atoms(monkeypatch) -> None:
    atoms = AtomArray(1)
    atoms.chain_id = np.array(["A"])
    atoms.atom_name = np.array(["CA"])
    atoms.coord = np.array([[0.0, 0.0, 0.0]])
    monkeypatch.setattr("pipelines.io.parse_to_atom_array", lambda _: atoms)

    with pytest.raises(ValueError, match="fewer than two C-alpha atoms"):
        ca_radius_of_gyration("prediction.cif", chain_id="A")


def test_attach_measurements_bridges_logical_and_prediction_design_ids() -> None:
    handler = PipelineDataHandler(
        [
            make_row(
                "oracle",
                "/tmp/physical-d1-model.cif",
                design_id="physical-d1-model",
                parent_design_id="physical-d1",
                metrics={"design_id": "submitted-d1"},
            ),
            make_row(
                "oracle",
                "/tmp/physical-d0-model.cif",
                design_id="physical-d0-model",
                parent_design_id="physical-d0",
                metrics={"design_id": "submitted-d0"},
            ),
        ]
    )

    measured = attach_measurements_by_design_id(
        handler,
        {
            "submitted-d0": {"bespoke": {"distance": 1.25}},
            "submitted-d1": {"bespoke": {"distance": 2.5}},
        },
    ).unpack(all=True)

    by_id = measured.set_index("design_id")
    assert by_id.loc["submitted-d0", "bespoke.distance"] == pytest.approx(1.25)
    assert by_id.loc["submitted-d1", "bespoke.distance"] == pytest.approx(2.5)


def test_attach_measurements_rejects_incomplete_logical_key_set() -> None:
    handler = PipelineDataHandler(
        [
            make_row(
                "oracle",
                "/tmp/physical-d0-model.cif",
                design_id="physical-d0-model",
                metrics={"design_id": "submitted-d0"},
            )
        ]
    )

    with pytest.raises(KeyError, match=r"missing=\['submitted-d0'\]"):
        attach_measurements_by_design_id(handler, {})


def test_diagnostic_columns_are_optional_and_write_nonnumeric_vectors(
    tmp_path: Path,
) -> None:
    import inspect
    import pandas as pd

    parameter = inspect.signature(run_sequence_scorer).parameters["diagnostic_columns"]
    assert parameter.default == ()

    detail = '[{"DSSR":"cW-W","LW":"cWW","partner_id":"A38"}]'
    frame = pd.DataFrame(
        [
            {
                "design_id": "rz0",
                "metric": 1.0,
                "t6.pair_classifications_B6": detail,
                "pass": True,
                "soft_pass": True,
            }
        ]
    )
    report = ScoreReport(
        eval="diagnostic-test",
        n_reported=1,
        required_n=1,
        contract_ok=True,
        validation_oracle="none",
    )
    out = tmp_path / "score.json"
    per_design = tmp_path / "score.csv"

    write_score_outputs(
        report,
        frame,
        out=out,
        per_design_out=per_design,
        metric_columns=["metric", "t6.pair_classifications_B6"],
    )

    written = pl.read_csv(per_design)
    assert written["t6.pair_classifications_B6"].to_list() == [detail]


def test_scored_frame_must_contain_every_submitted_design() -> None:
    submission = pl.DataFrame({"design_id": ["d0", "d1", "d2"]})
    partial = pd.DataFrame({"design_id": ["d0", "d2"]})

    with pytest.raises(RuntimeError, match=r"missing=\['d1'\].*n_frame=2"):
        merge_submission_columns(partial, submission, ())


def test_managed_scorer_uses_controller_assigned_rundir(
    tmp_path: Path, monkeypatch
) -> None:
    assigned = tmp_path / "trial" / "scoring" / "score_run"
    monkeypatch.setenv("RUNDIR", str(assigned))

    assert isolated_scoring_root(tmp_path / "source-local-cache") == assigned.resolve()


def test_scorer_default_root_applies_without_controller_rundir(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("RUNDIR", raising=False)
    default = tmp_path / "source-local-cache"

    assert isolated_scoring_root(default) == default.resolve()
