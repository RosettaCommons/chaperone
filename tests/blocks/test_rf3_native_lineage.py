"""Prediction siblings may share coordinates but have distinct designed sequences."""

import json
import pytest
from types import SimpleNamespace

from blocks.networks.prediction.rosettafold3 import (
    _RF3_INPUT_MANIFEST,
    _rf3_input_records,
    compile_rf3,
)
from chaperone import PipelineDataHandler
from chaperone.data import make_sequence_row


def test_rf3_native_siblings_backlink_by_design_id_not_shared_backbone(tmp_path):
    base = PipelineDataHandler(
        [
            make_sequence_row(
                design_id=f"native/{index}",
                parent_design_id="backbone",
                sequences={"A": sequence},
                context_path="/source/backbone.cif",
            )
            for index, sequence in enumerate(("AAA", "WWW"))
        ]
    )
    records = _rf3_input_records(
        [
            SimpleNamespace(
                basename=f"native/{index}",
                design_id=f"native/{index}",
                path="/source/backbone.cif",
                sequences={"A": sequence},
            )
            for index, sequence in enumerate(("AAA", "WWW"))
        ]
    )
    (tmp_path / _RF3_INPUT_MANIFEST).write_text(
        json.dumps(
            {
                name: {"path_handle": item.path, "parent_design_id": item.design_id}
                for item, name in records
            }
        )
    )
    for _, name in records:
        output = tmp_path / "outputs" / name
        output.mkdir(parents=True)
        (output / f"{name}_model.cif").write_text("data_model\n")
    merged = PipelineDataHandler.merge(base, compile_rf3(tmp_path))
    rows = merged._df.to_dicts()
    assert {row["parent_design_id"] for row in rows} == {"native/0", "native/1"}
    for item, name in records:
        row = next(row for row in rows if row["parent_design_id"] == item.design_id)
        assert name in row["structures"][-1]["path"]
        assert json.loads(row["sequences"]) == item.sequences


def test_rf3_partial_success_does_not_cache_missing_input(tmp_path):
    import pytest

    manifest = {name: {"parent_design_id": name} for name in ("done", "missing")}
    (tmp_path / _RF3_INPUT_MANIFEST).write_text(json.dumps(manifest))
    output = tmp_path / "outputs/done"
    output.mkdir(parents=True)
    (output / "done_model.cif").write_text("data_model\n")
    with pytest.raises(RuntimeError, match="required inputs: missing"):
        compile_rf3(tmp_path)


def test_rf3_each_requested_seed_requires_all_inputs(tmp_path):
    import pytest

    (tmp_path / _RF3_INPUT_MANIFEST).write_text(
        json.dumps(
            {
                "done": {"parent_design_id": "done"},
            }
        )
    )
    output = tmp_path / "outputs/seed_101/done"
    output.mkdir(parents=True)
    (output / "done_model.cif").write_text("data_model\n")
    with pytest.raises(RuntimeError, match=r"required inputs: done \(seed 202\)"):
        compile_rf3(tmp_path, expected_seeds=[101, 202])


def test_rf3_revision_invalidates_pre_fix_cache():
    import hashlib
    from blocks.networks.prediction.rosettafold3 import RosettaFold3Block
    from chaperone import ExecutionEngine
    from chaperone_cli.registry import _extract_field_type

    block = RosettaFold3Block()
    config = block.model_dump(mode="json")
    assert config.pop("output_lineage_version") == 2
    assert (
        _extract_field_type(type(block).model_fields["output_lineage_version"]) is None
    )
    old_hash = hashlib.sha256(
        json.dumps(
            [
                {
                    "class": f"{type(block).__module__}.{type(block).__qualname__}",
                    "config": config,
                }
            ],
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()
    assert old_hash != ExecutionEngine._cache_fingerprint([block])


def _write_early_stop(root, name, contents="early_stopped,mean_plddt\nTrue,0.31\n"):
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}_ranking_scores.csv").write_text(contents)


def test_rf3_accepts_explicit_early_stop_and_keeps_completed_lineage(tmp_path):
    manifest = {name: {"parent_design_id": name} for name in ("done", "stopped")}
    (tmp_path / _RF3_INPUT_MANIFEST).write_text(json.dumps(manifest))
    output = tmp_path / "outputs/done"
    output.mkdir(parents=True)
    (output / "done_model.cif").write_text("data_model\n")
    _write_early_stop(tmp_path / "outputs", "stopped")
    result = compile_rf3(tmp_path)
    assert len(result) == 1
    assert result._df["parent_design_id"].to_list() == ["done"]


def test_rf3_all_early_stopped_returns_empty_child_without_parent_passthrough(tmp_path):
    (tmp_path / _RF3_INPUT_MANIFEST).write_text(json.dumps({"stopped": {"parent_design_id": "stopped"}}))
    _write_early_stop(tmp_path / "outputs", "stopped")
    result = compile_rf3(tmp_path)
    assert len(result) == 0
    base = PipelineDataHandler([make_sequence_row(design_id="stopped", sequences={"A": "AAA"})])
    assert len(PipelineDataHandler.merge(base, result)) == 0


def test_rf3_early_stop_marker_is_scoped_to_requested_seed(tmp_path):
    import pytest

    (tmp_path / _RF3_INPUT_MANIFEST).write_text(json.dumps({"stopped": {"parent_design_id": "stopped"}}))
    _write_early_stop(tmp_path / "outputs/seed_101", "stopped")
    with pytest.raises(RuntimeError, match=r"required inputs: stopped \(seed 202\)"):
        compile_rf3(tmp_path, expected_seeds=[101, 202])
    _write_early_stop(tmp_path / "outputs/seed_202", "stopped")
    assert len(compile_rf3(tmp_path, expected_seeds=[101, 202])) == 0


def test_rf3_disabled_early_stopping_does_not_accept_marker(tmp_path):
    import pytest

    (tmp_path / _RF3_INPUT_MANIFEST).write_text(json.dumps({"stopped": {"parent_design_id": "stopped"}}))
    _write_early_stop(tmp_path / "outputs", "stopped")
    with pytest.raises(RuntimeError, match="required inputs: stopped"):
        compile_rf3(tmp_path, early_stopping_plddt_threshold=0)


@pytest.mark.parametrize("contents", [
    "early_stopped,mean_plddt\nFalse,0.31\n",
    "early_stopped,mean_plddt\nTrue,nan\n",
    "early_stopped,mean_plddt\nTrue,inf\n",
    "early_stopped,mean_plddt\nTrue,0.7\n",
    "early_stopped,mean_plddt\nTrue,-0.2\n",
    "early_stopped,mean_plddt\nTrue,\n",
    "early_stopped,mean_plddt\nTrue,0.31\nTrue,0.2\n",
    "early_stopped\nTrue\n",
    "",
])
def test_rf3_malformed_or_inconsistent_stop_does_not_hide_missing_output(tmp_path, contents):
    (tmp_path / _RF3_INPUT_MANIFEST).write_text(json.dumps({"stopped": {"parent_design_id": "stopped"}}))
    _write_early_stop(tmp_path / "outputs", "stopped", contents)
    with pytest.raises(RuntimeError, match="required inputs: stopped"):
        compile_rf3(tmp_path)


def test_rf3_preserves_default_early_stop_and_passes_threshold_to_compiler(monkeypatch, tmp_path):
    from blocks.networks.prediction import rosettafold3

    received = {}
    monkeypatch.setattr(rosettafold3, "compile_rf3", lambda *args, **kwargs: received.update(kwargs))
    block = rosettafold3.RosettaFold3Block()
    assert block.early_stopping_plddt_threshold == 0.5
    block.compile(tmp_path)
    assert received["early_stopping_plddt_threshold"] == 0.5
