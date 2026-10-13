"""Consolidated functional tests for data model."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import get_args
from pipelines import BlockFunction, PipelineDataHandler
import numpy as np
from biotite.structure import AtomArray
from blocks.utils.geometry import get_ligand_mask
from pipelines.data import make_row
import os
import subprocess
from pathlib import Path
import json
import cloudpickle
import polars as pl
from pipelines.execution_tracking import ExecutionSummary
from pipelines import (
    CompiledBlockFunction,
    InlineMetricBlock,
)
from pipelines.block_config import runner
from pipelines.data import make_sequence_row
from pipelines.data.handler import _csv_safe_value
from pipelines.data.schema import Column, get_sequences_from_row
from pipelines.manifest import PipelineManifest


# test_catcif_storage.py


def _install_fake_catcif(monkeypatch):
    entries: dict[str, str] = {}

    def append_to_catcif_file(path, content, entry_id, **_kwargs):
        entries[f"{path}:{entry_id}"] = content
        open(path, "a").close()

    monkeypatch.setitem(
        sys.modules,
        "catcif_tools",
        SimpleNamespace(
            append_to_catcif_file=append_to_catcif_file,
            get_structure=entries.__getitem__,
        ),
    )


def test_store_is_catcif_only(tmp_path, monkeypatch):
    _install_fake_catcif(monkeypatch)
    source = tmp_path / "design.cif"
    source.write_text("data_design\n#\n")
    handler = PipelineDataHandler([make_row("design", source)])

    handler.store("design", tmp_path / "packed")

    archived = next(handler.get_data(load=False, which="last"))
    assert archived.path.endswith("packed.catcif:design")
    assert archived.storage_format == "catcif"
    assert set(get_args(BlockFunction.model_fields["store"].annotation)) == {
        "none",
        "catcif",
    }

    materialized = handler.materialize(tmp_path / "materialized")
    extracted = next(materialized.get_data(load=False, which="last"))
    assert extracted.path.endswith("materialized/design/design.cif")
    assert open(extracted.path).read() == "data_design\n#\n"


# test_geometry.py


def test_get_ligand_mask_falls_back_to_pdb_hetero_annotation():
    array = AtomArray(3)
    array.hetero = np.array([False, False, True])

    mask = get_ligand_mask(array)

    assert mask.tolist() == [False, False, True]
    _check_get_ligand_mask_prefers_explicit_annotation()


def _check_get_ligand_mask_prefers_explicit_annotation():
    array = AtomArray(3)
    array.hetero = np.array([False, False, True])
    array.set_annotation("is_ligand", np.array([False, True, False]))

    mask = get_ligand_mask(array)

    assert mask.tolist() == [False, True, False]


# test_handler_pack.py


def test_pack_accepts_default_pandas_unpack():
    handler = PipelineDataHandler(
        [make_row(name="input", path="/tmp/input.pdb", design_id="design-1")]
    )
    frame = handler.unpack()
    frame["acceptance.rank_score"] = [12.5]

    restored = PipelineDataHandler.pack(frame).unpack()

    assert restored["design_id"].tolist() == ["design-1"]
    assert restored["acceptance.rank_score"].tolist() == [12.5]
    assert restored["input.path"].tolist() == ["/tmp/input.pdb"]


# test_lean_imports.py

REPO = Path(__file__).resolve().parents[1]


def _run_isolated(source: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src")
    env["PYDANTIC_DISABLE_PLUGINS"] = "1"
    return subprocess.run(
        [sys.executable, "-c", source],
        cwd=REPO,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )


def test_selected_cli_blocks_do_not_scan_the_registry():
    result = _run_isolated(
        """
from blocks import Blocks
from blocks.metrics.clash import ClashMetricsBlock
from pipelines import init_cli
assert not Blocks._scanned
init_cli(['test-program'], blocks=[ClashMetricsBlock])
assert not Blocks._scanned
"""
    )
    assert result.returncode == 0, result.stderr


# test_manifest.py


def _add_step(manifest: PipelineManifest, name: str, parents: list[str]) -> None:
    manifest.add_step(
        block_name=name.split("/")[-1],
        name=name,
        parquet=manifest.rundir / name / "output.parquet",
        pipeline_input_names=parents,
    )


def test_pipeline_json_is_the_only_manifest(tmp_path):
    (tmp_path / "state.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "block_name": "stale",
                        "name": "stale",
                        "pipeline_input_names": [],
                        "parquet": "stale/output.parquet",
                    }
                ]
            }
        )
    )

    manifest = PipelineManifest.load(tmp_path)
    assert manifest.steps == []

    _add_step(manifest, "design", [])
    manifest.save()

    assert manifest.path == tmp_path / "pipeline.json"
    assert not (tmp_path / "pipeline.json.tmp").exists()
    payload = json.loads(manifest.path.read_text())
    assert set(payload) == {"steps"}
    assert "rows_in" not in payload["steps"][0]


def test_manifest_preserves_lineage_and_execution_totals(tmp_path):
    manifest = PipelineManifest(tmp_path)
    _add_step(manifest, "design", [])
    _add_step(manifest, "sequence", ["design"])
    _add_step(manifest, "predict", ["sequence"])
    manifest.set_step_execution_summary(
        "predict",
        ExecutionSummary(
            records=2,
            successful_records=2,
            wall_seconds=90,
            gpu_seconds=180,
        ),
    )
    manifest.save()

    loaded = PipelineManifest.load(tmp_path)
    assert loaded.get_step("sequence").pipeline_input_names == ["design"]
    assert loaded.get_step("predict").pipeline_input_names == ["sequence"]
    assert loaded.execution_totals() == ExecutionSummary(
        records=2,
        successful_records=2,
        wall_seconds=90,
        gpu_seconds=180,
    )


def test_reregister_moves_step_without_clobbering_lineage(tmp_path):
    manifest = PipelineManifest(tmp_path)
    _add_step(manifest, "design", [])
    _add_step(manifest, "sequence", ["design"])
    _add_step(manifest, "analysis", ["sequence"])

    _add_step(manifest, "sequence", [])

    assert manifest.step_names() == ["design", "analysis", "sequence"]
    assert manifest.get_step("sequence").pipeline_input_names == ["design"]


class _RegressionSourceBlock(CompiledBlockFunction):
    _name = "regression_source"
    _requires_input = False

    def get_command_template(self):
        return []

    def create_command(self, handler, sandbox_dir, chunk_id):
        return []

    def compile(self, sandbox_dir):
        return PipelineDataHandler(
            [
                make_row(
                    name=self.name,
                    path=Path(sandbox_dir) / "generated.cif",
                    design_id="generated",
                )
            ]
        )


class _RegressionNoLoadMetric(InlineMetricBlock):
    _name = "regression_metric"

    def get_data_kwargs(self):
        return {"which": "last", "load": False}

    def compute_metrics(self, loaded, *, context=None, index=0):
        return {"row": loaded.design_id}


def _scheduling_handler(tmp_path):
    return PipelineDataHandler(
        [
            make_row(
                name="scheduler",
                path=tmp_path / "synthetic-input.cif",
                metrics={"synthetic": True},
                design_id="synthetic",
            )
        ]
    )


def test_source_compiled_call_returns_output_without_scheduling_rows(tmp_path):
    result = _RegressionSourceBlock()(
        _scheduling_handler(tmp_path), tmp_path, chunk_id=0, recompile=True
    )

    rows = result._df.to_dicts()
    assert len(rows) == 1
    assert rows[0][str(Column.DESIGN_ID)] == "generated"
    assert json.loads(rows[0][str(Column.METRICS)]) == {}


def test_source_runner_finisher_returns_output_without_scheduling_rows(
    tmp_path, monkeypatch
):
    block_path = tmp_path / "source.pkl"
    with block_path.open("wb") as handle:
        cloudpickle.dump(_RegressionSourceBlock(), handle)

    input_path = tmp_path / "input.parquet"
    _scheduling_handler(tmp_path).write_parquet(input_path)
    output_path = tmp_path / "output.parquet"
    state = {
        "sandbox_dir": str(tmp_path),
        "chunk_id": 0,
        "output_path": str(output_path),
        "block_path": str(block_path),
        "input_handler_path": str(input_path),
        "result_pickle": str(tmp_path / "result.pkl"),
        "terminal_path": str(tmp_path / "terminal.json"),
        "block_name": "regression_source",
    }
    monkeypatch.setattr(runner, "_write_result_pickle", lambda *args: None)
    monkeypatch.setattr(runner, "_terminal", lambda *args, **kwargs: None)

    assert runner._phase_finish(state) == 0
    rows = PipelineDataHandler(output_path)._df.to_dicts()
    assert len(rows) == 1
    assert rows[0][str(Column.DESIGN_ID)] == "generated"
    assert json.loads(rows[0][str(Column.METRICS)]) == {}


def test_inline_metric_overlays_exact_current_design_before_older_parent(tmp_path):
    sequences = PipelineDataHandler(
        [
            make_sequence_row(
                design_id="backbone",
                sequences={"A": "AAAA"},
                context_path=tmp_path / "backbone.cif",
            ),
            make_sequence_row(
                design_id="variant",
                parent_design_id="backbone",
                sequences={"A": "CCCC"},
                context_path=tmp_path / "variant.cif",
            ),
        ]
    )
    prediction_output = PipelineDataHandler(
        [
            make_row(name="prediction", path=tmp_path / "backbone.cif"),
            make_row(name="prediction", path=tmp_path / "variant.cif"),
        ]
    )
    predicted = PipelineDataHandler.merge(sequences, prediction_output)

    updated = _RegressionNoLoadMetric()(predicted, tmp_path)
    rows = {row[str(Column.DESIGN_ID)]: row for row in updated._df.to_dicts()}

    assert set(rows) == {"backbone", "variant"}
    assert get_sequences_from_row(rows["backbone"]) == {"A": "AAAA"}
    assert get_sequences_from_row(rows["variant"]) == {"A": "CCCC"}
    assert json.loads(rows["variant"][str(Column.METRICS)])["regression_metric"] == {
        "row": "variant"
    }


def test_sequence_csv_json_encodes_nested_rfd3_metrics(tmp_path):
    handler = PipelineDataHandler(
        [
            make_sequence_row(
                design_id="rfd3-design",
                sequences={"A": "ACDE"},
                metrics={
                    "rfd3": {
                        "diffused_com": [1.0, 2.0, 3.0],
                        "fixed_com": {"xyz": [4.0, 5.0, 6.0]},
                    }
                },
            )
        ]
    )
    csv_path = tmp_path / "sequences.csv"

    handler.write_sequences_csv(csv_path)
    row = pl.read_csv(csv_path).to_dicts()[0]

    assert json.loads(row["rfd3.diffused_com"]) == [1.0, 2.0, 3.0]
    assert json.loads(row["rfd3.fixed_com"]) == {"xyz": [4.0, 5.0, 6.0]}
    assert _csv_safe_value((1, {"two": 2})) == '[1, {"two": 2}]'
