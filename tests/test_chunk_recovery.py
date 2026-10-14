"""Recovery of chunks whose future failed but whose parquet is on disk.

The race this guards: submitit raises UncompletedJobError when the login node is
still serving a directory listing older than the compute node's result pickle.
That is indistinguishable from a job that genuinely wrote nothing, so without
this the driver discards completed work and raises "N chunk(s) still failing".
"""

import json
from types import SimpleNamespace

import polars as pl
import pytest

from chaperone.engine.chunking import recover_written_outputs


def _ctx(tmp_path, chunk_id, *, write=False):
    chunk_dir = tmp_path / f"step/chunk_{chunk_id}"
    chunk_dir.mkdir(parents=True)
    if write:
        pl.DataFrame({"id": ["design"]}).write_parquet(chunk_dir / "output.parquet")
    return SimpleNamespace(
        chunk_id=chunk_id, steps=[SimpleNamespace(chunk_dir=chunk_dir)]
    )


def test_recovers_a_chunk_whose_parquet_exists(tmp_path):
    contexts = [_ctx(tmp_path, 0, write=True)]
    outputs = [None]
    assert recover_written_outputs(contexts, outputs, wait_s=0) == 1
    assert outputs[0] == contexts[0].steps[-1].chunk_dir / "output.parquet"


def test_leaves_genuinely_failed_chunks_as_none(tmp_path):
    contexts = [_ctx(tmp_path, 0, write=False)]
    outputs = [None]
    assert recover_written_outputs(contexts, outputs, wait_s=0) == 0
    assert outputs == [None]


def test_does_not_touch_chunks_that_already_succeeded(tmp_path):
    contexts = [_ctx(tmp_path, 0, write=True)]
    sentinel = "already-there"
    outputs = [sentinel]
    assert recover_written_outputs(contexts, outputs, wait_s=0) == 0
    assert outputs == [sentinel]


def test_mixed_batch_recovers_only_the_written_ones(tmp_path):
    contexts = [
        _ctx(tmp_path, 0, write=True),
        _ctx(tmp_path, 1, write=False),
        _ctx(tmp_path, 2, write=True),
    ]
    outputs = [None, None, None]
    assert recover_written_outputs(contexts, outputs, wait_s=0) == 2
    assert outputs[0] is not None and outputs[2] is not None
    assert outputs[1] is None


def test_waits_for_a_parquet_that_appears_late(tmp_path):
    """The whole point of the bound: a parquet that becomes visible one poll
    later must still be recovered, not discarded."""
    contexts = [_ctx(tmp_path, 0, write=False)]
    outputs = [None]
    parquet = contexts[0].steps[-1].chunk_dir / "output.parquet"

    calls = []

    def fake_sleep(_seconds):
        calls.append(_seconds)
        pl.DataFrame({"id": ["design"]}).write_parquet(parquet)

    assert recover_written_outputs(contexts, outputs, wait_s=10, sleep=fake_sleep) == 1
    assert outputs[0] == parquet
    assert len(calls) == 1


def test_gives_up_at_the_deadline(tmp_path):
    """A genuinely failed chunk must not stall the driver indefinitely."""
    contexts = [_ctx(tmp_path, 0, write=False)]
    outputs = [None]
    calls = []
    assert (
        recover_written_outputs(
            contexts, outputs, wait_s=0, sleep=lambda s: calls.append(s)
        )
        == 0
    )
    assert calls == []  # deadline already passed, so no sleeping at all


def test_no_pending_chunks_does_no_filesystem_work(tmp_path):
    """A healthy run must not pay for this path."""
    contexts = [_ctx(tmp_path, 0, write=True)]
    outputs = ["ok"]

    def explode(_):
        raise AssertionError("should not sleep when nothing is pending")

    assert recover_written_outputs(contexts, outputs, wait_s=30, sleep=explode) == 0


def test_terminal_failure_is_not_recovered_from_a_leftover_parquet(tmp_path):
    contexts = [_ctx(tmp_path, 0, write=True)]
    contexts[0].steps[-1].chunk_dir.joinpath("terminal.json").write_text(
        '{"status":"error"}'
    )
    outputs = [None]
    assert recover_written_outputs(contexts, outputs, wait_s=0) == 0
    assert outputs == [None]


@pytest.mark.parametrize("payload", [b"", b"PAR1", b"not parquet"])
def test_corrupt_output_is_not_recovered(tmp_path, payload):
    contexts = [_ctx(tmp_path, 0)]
    (contexts[0].steps[-1].chunk_dir / "output.parquet").write_bytes(payload)
    outputs = [None]
    assert recover_written_outputs(contexts, outputs, wait_s=0) == 0
    assert outputs == [None]


@pytest.mark.parametrize(
    "report",
    [
        None,
        [],
        {"status": "running"},
        {"status": "success", "output_sha256": "stale"},
        {"status": "success", "output_count": 2},
    ],
)
def test_inconsistent_terminal_report_is_not_recovered(tmp_path, report):
    contexts = [_ctx(tmp_path, 0, write=True)]
    (contexts[0].steps[-1].chunk_dir / "terminal.json").write_text(json.dumps(report))
    outputs = [None]
    assert recover_written_outputs(contexts, outputs, wait_s=0) == 0
    assert outputs == [None]


def test_recovers_verified_empty_output(tmp_path):
    from chaperone.engine.chunking import _sha256

    contexts = [_ctx(tmp_path, 0)]
    parquet = contexts[0].steps[-1].chunk_dir / "output.parquet"
    pl.DataFrame(schema={"id": pl.String}).write_parquet(parquet)
    parquet.with_name("terminal.json").write_text(
        json.dumps(
            {"status": "success", "output_count": 0, "output_sha256": _sha256(parquet)}
        )
    )
    outputs = [None]
    assert recover_written_outputs(contexts, outputs, wait_s=0) == 1


def test_retries_incomplete_output_until_readable(tmp_path):
    contexts = [_ctx(tmp_path, 0)]
    parquet = contexts[0].steps[-1].chunk_dir / "output.parquet"
    parquet.write_bytes(b"PAR1")
    outputs = [None]

    def finish_write(_):
        pl.DataFrame({"id": ["design"]}).write_parquet(parquet)

    assert (
        recover_written_outputs(contexts, outputs, wait_s=10, sleep=finish_write) == 1
    )


def test_invalid_json_terminal_report_is_not_recovered(tmp_path):
    contexts = [_ctx(tmp_path, 0, write=True)]
    (contexts[0].steps[-1].chunk_dir / "terminal.json").write_text("{")
    outputs = [None]
    assert recover_written_outputs(contexts, outputs, wait_s=0) == 0
    assert outputs == [None]
