"""Changed chains cannot replay intermediate results from earlier settings."""

from dataclasses import asdict
from pathlib import Path

import pytest

from blocks.util_blocks.testing import CopyBlock
from chaperone import ExecutionEngine, PipelineDataHandler


def engine_at(root):
    return ExecutionEngine(
        rundir=root,
        run_args_default=dict(mode="local", max_workers=1, use_local_gpus=False),
        job_args_default=dict(cpus_per_task=1, partition="cpu", gres=""),
    )


@pytest.mark.parametrize(
    "partial,skip_cache,force_exec",
    [
        (False, False, False),
        (True, False, False),
        (True, True, False),
        (True, False, True),
    ],
)
def test_changed_chain_reruns_real_intermediate_and_identical_replay_is_cached(
    tmp_path, partial, skip_cache, force_exec
):
    source = (
        Path(__file__).resolve().parents[1]
        / "examples/benchmarks/ppi/files/4zxb_cropped.pdb"
    )
    original = PipelineDataHandler.from_paths([str(source)], name="input")
    engine = engine_at(tmp_path / "run")
    upstream = engine.run(CopyBlock(name="upstream"), data=original)
    upstream_output = engine.rundir / "upstream/output.parquet"
    upstream_stat = upstream_output.stat().st_mtime_ns
    upstream_structure = Path(next(upstream.get_data()).path)
    unrelated = engine.rundir / "unrelated/keep.txt"
    unrelated.parent.mkdir()
    unrelated.write_text("not part of the chain")
    chain = [CopyBlock(name="intermediate", n_multiply=1), CopyBlock(name="tail")]
    assert len(engine.run(chain, data=upstream)) == 1
    first_chunk = engine.rundir / "intermediate/chunk_0/output.parquet"
    assert first_chunk.is_file()
    if partial:
        # Simulate an interrupted chain with only its first step completed.
        (engine.rundir / "tail/output.parquet").unlink()
        (engine.rundir / "tail/chunk_0/output.parquet").unlink()
    stale_input = first_chunk.parent / "inputs/obsolete.json"
    stale_input.parent.mkdir(exist_ok=True)
    stale_input.write_text("old staged input")
    chain = [CopyBlock(name="intermediate", n_multiply=2), CopyBlock(name="tail")]
    resumed = engine_at(engine.rundir)
    output = resumed.run(
        chain,
        data=upstream,
        run_args=dict(skip_cache=skip_cache, force_exec=force_exec),
    )
    assert len(output) == 2
    assert not stale_input.exists()
    assert len(PipelineDataHandler(first_chunk)) == 2
    assert upstream_output.stat().st_mtime_ns == upstream_stat
    assert upstream_structure.is_file()
    assert unrelated.read_text() == "not part of the chain"
    tail_output = engine.rundir / "tail/output.parquet"
    before = (first_chunk.stat().st_mtime_ns, tail_output.stat().st_mtime_ns)
    totals = asdict(resumed.manifest.execution_totals())
    replay = engine_at(engine.rundir)
    assert len(replay.run(chain, data=upstream)) == 2
    assert before == (first_chunk.stat().st_mtime_ns, tail_output.stat().st_mtime_ns)
    assert asdict(replay.manifest.execution_totals()) == totals


def test_invalidation_preserves_current_inputs_and_does_not_partially_delete(tmp_path):
    engine = engine_at(tmp_path / "run")
    first = engine.rundir / "intermediate"
    tail = engine.rundir / "tail"
    first.mkdir(parents=True)
    tail.mkdir()
    sentinel = first / "keep.txt"
    sentinel.write_text("keep")
    current_input = tail / "input.cif"
    current_input.write_text("input")
    handler = PipelineDataHandler.from_paths([str(current_input)], name="input")
    with pytest.raises(ValueError, match="containing input artifact"):
        engine._invalidate_chain_cache([first, tail], handler)
    assert sentinel.is_file() and current_input.is_file()


@pytest.mark.parametrize("name", ["..", ".", "../outside"])
def test_invalidation_rejects_paths_outside_step_directories(tmp_path, name):
    engine = engine_at(tmp_path / "run")
    with pytest.raises(ValueError, match="must be inside"):
        engine._invalidate_chain_cache([engine.rundir / name], None)


def test_invalidation_preserves_unrelated_nested_step(tmp_path):
    engine = engine_at(tmp_path / "run")
    nested = engine.rundir / "intermediate/unrelated/output.parquet"
    nested.parent.mkdir(parents=True)
    nested.write_text("keep")
    engine.manifest.add_step(
        block_name="copy",
        name="intermediate/unrelated",
        parquet=nested,
        pipeline_input_names=[],
    )
    with pytest.raises(ValueError, match="another step"):
        engine._invalidate_chain_cache([engine.rundir / "intermediate"], None)
    assert nested.is_file()
