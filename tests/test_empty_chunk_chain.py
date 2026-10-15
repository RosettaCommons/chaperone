"""Filtering out a chunk must not launch models or revive cached designs."""

import json
from types import SimpleNamespace

import pytest

from chaperone import PipelineDataHandler
from chaperone.data import make_row
from chaperone.engine.chunking import execute_chunk_job


@pytest.mark.parametrize("force_exec", [False, True])
def test_empty_filter_skips_models_and_overwrites_stale_child_cache(
    tmp_path, force_exec
):
    calls = []

    class Block:
        _requires_input = True

        def __init__(self, name):
            self.name = name

        def __call__(self, handler):
            calls.append(self.name)
            assert self.name == "filter", "Empty cohorts must not launch a model"
            return handler[:0]

    original = PipelineDataHandler([make_row("input", "/design.cif", design_id="one")])
    steps = []
    for name in ["filter", "mpnn", "af3"]:
        block = Block(name)
        step = SimpleNamespace(
            chunk_dir=tmp_path / name / "chunk_0",
            block_path=None,
            get_block=lambda block=block: block,
        )
        if name != "filter":
            step.chunk_dir.mkdir(parents=True)
            original.to_parquet(step.chunk_dir / "output.parquet")
        steps.append(step)
    context = SimpleNamespace(
        steps=steps, chunk_id=0, chunk_slice=(0, 1), get_handler=lambda: original
    )
    result = execute_chunk_job(context, force_exec=force_exec)
    assert calls == ["filter"]
    assert len(PipelineDataHandler(result)) == 0
    assert all(
        len(PipelineDataHandler(step.chunk_dir / "output.parquet")) == 0
        for step in steps
    )
    report = json.loads((steps[-1].chunk_dir / "terminal.json").read_text())
    assert report["status"] == "success"
    assert report["output_count"] == 0
