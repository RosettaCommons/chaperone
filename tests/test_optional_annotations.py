"""Optional annotations survive table operations and old packed files."""

import json

import numpy as np
import polars as pl
import pytest
from biotite.structure import AtomArray
from pydantic import ValidationError

from blocks.util_blocks.annotate import AnnotateBlock
from chaperone import PipelineDataHandler
from chaperone.data.schema import get_mask_for_metadata, make_row


def test_annotations_are_optional_and_late_rows_are_not_truncated(tmp_path):
    rows = [make_row("input", f"/tmp/design{i}.cif") for i in range(105)]
    assert "aromatics_near_ligand" not in rows[0]["structures"][0]["metadata"]
    rows[-1]["structures"][0]["metadata"]["aromatics_near_ligand"] = ["A10"]
    handler = PipelineDataHandler(rows)
    path = tmp_path / "annotations.parquet"
    handler.to_parquet(path)
    loaded = PipelineDataHandler(path)
    assert loaded._df.to_dicts()[-1]["structures"][0]["metadata"][
        "aromatics_near_ligand"
    ] == ["A10"]


def test_old_annotation_file_concat_pack_and_merge(tmp_path):
    annotated = make_row(
        "input",
        "/tmp/old.cif",
        metadata={
            "aromatics_near_ligand": ["A10"],
            "hbond": None,
            "interface": ["A11"],
            "custom_selection": ["A12"],
        },
    )
    old_path = tmp_path / "old.parquet"
    pl.DataFrame([annotated]).write_parquet(old_path)
    old = PipelineDataHandler(old_path)
    plain = PipelineDataHandler([make_row("input", "/tmp/new.cif")])
    combined = old + plain
    restored = PipelineDataHandler.pack(combined.unpack(all=True, to_pandas=False))
    first = restored._df.to_dicts()[0]
    assert first["structures"][0]["metadata"]["custom_selection"] == ["A12"]
    assert json.loads(first["metrics"]) == {}

    child = PipelineDataHandler(
        [make_row("child", "/tmp/old_child.cif", path_handle="/tmp/old.cif")]
    )
    merged = PipelineDataHandler.merge(old, child)
    meta = next(merged.get_data()).metadata
    assert meta["aromatics_near_ligand"] == ["A10"]
    assert meta["custom_selection"] == ["A12"]
    atoms = AtomArray(2)
    atoms.chain_id = np.array(["A", "A"])
    atoms.res_id = np.array([10, 11])
    assert get_mask_for_metadata("aromatics_near_ligand", atoms, meta).tolist() == [
        True,
        False,
    ]


def test_annotate_computes_only_requested_fields(monkeypatch):
    import blocks.util_blocks.annotate as module

    handler = PipelineDataHandler([make_row("input", "/tmp/input.cif")])
    loaded = next(handler.get_data())
    loaded.atom_array = AtomArray(0)
    monkeypatch.setattr(handler, "get_data", lambda **kwargs: iter([loaded]))
    monkeypatch.setattr(
        module, "find_aromatic_residues_near_ligand", lambda atoms: ["A10"]
    )

    def unwanted(*args, **kwargs):
        raise AssertionError("Unrequested annotation was computed")

    monkeypatch.setattr(module, "find_hbonds_in_pdb", unwanted)
    monkeypatch.setattr(module, "find_ligands_in_array", unwanted)
    monkeypatch.setattr(module, "_get_closest_residues_by_sidechain_dist", unwanted)
    result = AnnotateBlock(annotations=["aromatics_near_ligand"])(handler)
    assert next(result.get_data()).metadata["aromatics_near_ligand"] == ["A10"]
    with pytest.raises(ValidationError):
        AnnotateBlock()
    with pytest.raises(ValidationError):
        AnnotateBlock(annotations=["typo"])


@pytest.mark.parametrize("how", ["inner", "left"])
def test_large_merge_does_not_discard_updated_annotations(how):
    # Exercise the native-merge threshold with the same schema but new values.
    parent = PipelineDataHandler(
        [
            make_row("input", f"/tmp/design{i}.cif", metadata={"interface": ["A1"]})
            for i in range(105)
        ]
    )
    child = PipelineDataHandler(
        [
            make_row("input", f"/tmp/design{i}.cif", metadata={"interface": ["A2"]})
            for i in range(105)
        ]
    )
    result = PipelineDataHandler.merge(parent, child, how=how)
    assert len(result) == 105
    assert all(
        row["structures"][0]["metadata"]["interface"] == ["A2"]
        for row in result._df.to_dicts()
    )


def test_large_metric_merge_preserves_design_identity():
    rows = [
        make_row(
            "input",
            f"/tmp/design{i}.cif",
            design_id=f"stable{i}",
            parent_design_id=f"ancestor{i}",
        )
        for i in range(105)
    ]
    parent = PipelineDataHandler(rows)
    child = PipelineDataHandler(
        [{**row, "metrics": '{"score": {"x": 1}}'} for row in rows]
    )
    result = PipelineDataHandler.merge(parent, child)
    assert result._df["design_id"].to_list() == parent._df["design_id"].to_list()
    assert (
        result._df["parent_design_id"].to_list()
        == parent._df["parent_design_id"].to_list()
    )
    assert json.loads(result._df["metrics"][0]) == {"score": {"x": 1}}
