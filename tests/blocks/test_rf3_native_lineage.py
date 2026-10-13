"""Prediction siblings may share coordinates but have distinct designed sequences."""

import json
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
