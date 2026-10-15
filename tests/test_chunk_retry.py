"""Retry behaviour of ``execute_chunks`` when a chunk failure names no node.

``extract_failed_nodes`` only yields a node when sacct blames one. A transient
fault spread evenly across the allocation yields nothing, and the loop used to
treat "no new nodes to exclude" as "nothing left to try" and stop before
re-executing anything -- so those chunks were never retried even once, despite
``max_retries=2``, and their rows were silently dropped from the step.
"""

from types import SimpleNamespace

import polars as pl
import pytest

from chaperone.data import PipelineDataHandler, make_row
from chaperone.engine import chunking


def _run_args(max_retries=2):
    return SimpleNamespace(
        max_retries=max_retries,
        is_local=True,
        mode="local",
        recompile=False,
        force_exec=False,
        skip_chunk_cache=False,
        job_args=SimpleNamespace(
            rows_per_job=1,
            num_chunks=None,
            excluded_nodes_set=set(),
            job_name="job",
        ),
        for_attempt=lambda **kwargs: SimpleNamespace(is_local=True),
    )


@pytest.fixture
def dispatch_attempts(tmp_path, monkeypatch):
    """Record each dispatch, returning the per-attempt outcome scripted by a test."""
    monkeypatch.setattr(chunking, "extract_failed_nodes", lambda name: set())
    monkeypatch.setattr(
        chunking.ChunkContext, "cache_blocks", staticmethod(lambda *a, **k: None)
    )
    monkeypatch.setattr(
        "chaperone.engine.resolve._create_runner", lambda *a, **k: SimpleNamespace()
    )
    attempts = []

    def install(per_attempt_failures):
        def fake_dispatch(job_fn, contexts, runner, run_args, label, **kwargs):
            n_failed = per_attempt_failures[len(attempts)]
            attempts.append(n_failed)
            return [
                None if i < n_failed else ctx.steps[-1].chunk_dir / "output.parquet"
                for i, ctx in enumerate(contexts)
            ]

        monkeypatch.setattr(chunking, "_dispatch_jobs", fake_dispatch)
        return attempts

    return install


def _execute(tmp_path, max_retries=2):
    handler = PipelineDataHandler(
        [make_row(name="step", path=f"/runs/step/d{i}.pdb") for i in range(3)]
    )
    blocks = [SimpleNamespace(name="step")]
    return chunking.execute_chunks(
        blocks, handler, tmp_path, _run_args(max_retries=max_retries)
    )


def test_failures_with_no_node_to_exclude_are_still_retried(
    tmp_path, dispatch_attempts
):
    """The regression: 2 chunks fail, sacct blames nobody, a plain retry recovers them."""
    attempts = dispatch_attempts([2, 0])

    outputs = _execute(tmp_path)

    assert attempts == [2, 0], "failed chunks were never re-executed"
    assert all(p is not None for p in outputs)


def test_stops_once_a_retry_recovers_nothing(tmp_path, dispatch_attempts):
    """Two identical attempts recovering nothing: stop rather than burn the last retry."""
    attempts = dispatch_attempts([2, 2, 2])

    with pytest.raises(RuntimeError, match=r"2 chunk\(s\) still failing"):
        _execute(tmp_path)

    assert attempts == [2, 2], "retried again after a retry that recovered nothing"


def test_partial_recovery_keeps_retrying(tmp_path, dispatch_attempts):
    """Progress on the same node set earns the remaining attempts."""
    attempts = dispatch_attempts([3, 2, 0])

    _execute(tmp_path)

    assert attempts == [3, 2, 0]


def test_exhausted_retry_budget_fails_explicitly(tmp_path, dispatch_attempts):
    attempts = dispatch_attempts([2])
    with pytest.raises(RuntimeError, match=r"2 chunk\(s\) still failing"):
        _execute(tmp_path, max_retries=0)
    assert attempts == [2]


@pytest.mark.parametrize("flag", ["force_exec", "skip_chunk_cache", "recompile"])
def test_forced_slurm_execution_never_recovers_an_old_output(
    tmp_path, monkeypatch, dispatch_attempts, flag
):
    dispatch_attempts([2])
    args = _run_args(max_retries=0)
    args.is_local = False
    args.mode = "slurm"
    setattr(args, flag, True)

    def unexpected_recovery(*args, **kwargs):
        raise AssertionError("must not recover stale output from a forced run")

    monkeypatch.setattr(chunking, "recover_written_outputs", unexpected_recovery)
    with pytest.raises(RuntimeError, match="still failing"):
        chunking.execute_chunks(
            [SimpleNamespace(name="step")],
            PipelineDataHandler([make_row(name="step", path="/runs/step/d.pdb")]),
            tmp_path,
            args,
        )


def test_last_slurm_attempt_can_recover_a_completed_chunk(tmp_path, monkeypatch):
    from chaperone.engine.resolve import RuntimeArgs
    from chaperone.orchestration import JobArgs

    monkeypatch.setattr(chunking.ChunkContext, "cache_blocks", lambda *a, **k: None)
    runner = SimpleNamespace(fail_on_error=True)
    monkeypatch.setattr("chaperone.engine.resolve._create_runner", lambda *a: runner)

    def dispatch(job_fn, contexts, runner, run_args, label, **kwargs):
        assert run_args.allow_errors
        assert not runner.fail_on_error
        for ctx in contexts:
            ctx.steps[-1].chunk_dir.mkdir(parents=True)
            pl.DataFrame({"id": ["design"]}).write_parquet(
                ctx.steps[-1].chunk_dir / "output.parquet"
            )
        return [None] * len(contexts)

    monkeypatch.setattr(chunking, "_dispatch_jobs", dispatch)
    outputs = chunking.execute_chunks(
        [SimpleNamespace(name="step")],
        PipelineDataHandler([make_row(name="step", path="/runs/step/d.pdb")]),
        tmp_path,
        RuntimeArgs(
            mode="slurm",
            max_retries=0,
            use_local_gpus=False,
            job_args=JobArgs(rows_per_job=1),
        ),
    )
    assert len(outputs) == 1
    assert outputs[0].is_file()
