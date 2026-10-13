"""Consolidated functional tests for runtime."""

from __future__ import annotations

from types import SimpleNamespace
from chaperone import ExecutionEngine, InlineMetricBlock, JobArgs, PipelineDataHandler
import importlib
import sys
from chaperone.engine.chunking import ChainStep, ChunkContext
from chaperone.data.schema import Column
from chaperone.data import make_row
from chaperone.engine.chunking import (
    _reset_chunk_sandbox_for_execution,
    execute_chunk_job,
)
from pathlib import Path
import polars as pl
from chaperone.engine import chunking
import chaperone
from chaperone import containers
import pytest
from chaperone.execution_tracking import load_execution_records, track_execution
import json
import numpy as np
from biotite.structure import AtomArray
from blocks.util_blocks.annotate import find_aromatic_residues_near_ligand
from chaperone.orchestration.slurm_executors import _sanitize_env
from chaperone.engine.resolve import RuntimeArgs, _resolve_job_args
from chaperone.orchestration import job_args as job_args_module
from chaperone.orchestration.slurm_futures import SlurmFuture


# test_cache_fingerprint.py


class _ChunkedMetric(InlineMetricBlock):
    _name = "chunked_cache_regression"

    def compute_metrics(self, loaded, *, context=None, index=0):
        raise AssertionError("fingerprint tests must not execute the block")


def _handler(size: int) -> PipelineDataHandler:
    return PipelineDataHandler(
        [
            {
                "design_id": f"design-{index}",
                "structures": [],
                "metrics": "{}",
            }
            for index in range(size)
        ]
    )


def _fingerprint(
    *, rows: int = 48, rows_per_job: int = 8, exclude: str | None = None
) -> str:
    run_args = _run_args(rows_per_job=rows_per_job, exclude=exclude)
    return ExecutionEngine._cache_fingerprint(
        [_ChunkedMetric()], handler_in=_handler(rows), run_args=run_args
    )


def _run_args(*, rows_per_job: int, exclude: str | None = None):
    return SimpleNamespace(
        job_args=JobArgs(rows_per_job=rows_per_job, exclude=exclude),
        recompile=False,
        skip_cache=False,
        force_exec=False,
    )


def test_chunk_cache_fingerprint_changes_with_rows_per_job():
    assert _fingerprint(rows_per_job=32) != _fingerprint(rows_per_job=8)


def test_chunk_cache_fingerprint_changes_with_input_row_count():
    assert _fingerprint(rows=48) != _fingerprint(rows=56)


def test_cache_fingerprint_changes_with_same_size_replacement_rows():
    first = _handler(48)
    second = PipelineDataHandler(
        [
            {"design_id": f"replacement-{index}", "structures": [], "metrics": "{}"}
            for index in range(48)
        ]
    )
    run_args = _run_args(rows_per_job=8)

    assert ExecutionEngine._cache_fingerprint(
        [_ChunkedMetric()], handler_in=first, run_args=run_args
    ) != ExecutionEngine._cache_fingerprint(
        [_ChunkedMetric()], handler_in=second, run_args=run_args
    )


def test_cache_fingerprint_changes_when_artifact_bytes_change(tmp_path):
    structure = tmp_path / "same-name.cif"
    structure.write_text("first structure")
    handler = PipelineDataHandler(
        [make_row(name="input", path=structure, design_id="design")]
    )
    run_args = _run_args(rows_per_job=8)
    before = ExecutionEngine._cache_fingerprint(
        [_ChunkedMetric()], handler_in=handler, run_args=run_args
    )

    structure.write_text("replacement structure with different bytes")
    after = ExecutionEngine._cache_fingerprint(
        [_ChunkedMetric()], handler_in=handler, run_args=run_args
    )

    assert before != after


def test_chunk_cache_fingerprint_ignores_node_exclusions():
    assert _fingerprint(exclude=None) == _fingerprint(exclude="g1113")


def test_changed_chunk_layout_invalidates_partial_chunk_cache(tmp_path):
    engine = ExecutionEngine(rundir=tmp_path)
    block = _ChunkedMetric()
    handler = _handler(48)
    output = tmp_path / block.name / "output.parquet"
    chunk_output = output.parent / "chunk_1" / "output.parquet"
    chunk_output.parent.mkdir(parents=True)
    _handler(16).write_parquet(chunk_output)

    engine._write_cache_fingerprint(
        output,
        engine._cache_fingerprint(
            [block], handler_in=handler, run_args=_run_args(rows_per_job=32)
        ),
    )
    new_fingerprint = engine._cache_fingerprint(
        [block], handler_in=handler, run_args=_run_args(rows_per_job=8)
    )

    cached = engine._check_cache(
        [block],
        output,
        handler,
        _run_args(rows_per_job=8),
        new_fingerprint,
    )

    assert cached is None
    assert not output.parent.exists()


# test_chunk_block_serialization.py


def test_campaign_local_block_is_cached_by_value(tmp_path):
    """A worker can unpickle a custom block without its source module."""
    module_dir = tmp_path / "campaign"
    module_dir.mkdir()
    module_path = module_dir / "score.py"
    module_path.write_text(
        "class CampaignMetric:\n"
        "    name = 'campaign_metric'\n"
        "    def compute(self):\n"
        "        return 17\n"
    )

    sys.path.insert(0, str(module_dir))
    try:
        score = importlib.import_module("score")
        block = score.CampaignMetric()
        context = ChunkContext(
            input_path=tmp_path / "input.parquet",
            chunk_slice=(0, 1),
            chunk_id=0,
            steps=[ChainStep(chunk_dir=tmp_path / "run" / "chunk_0")],
        )
        ChunkContext.cache_blocks([block], [context], tmp_path / "run")
    finally:
        sys.path.remove(str(module_dir))
        sys.modules.pop("score", None)

    module_path.unlink()
    restored = context.get_block()

    assert restored.compute() == 17
    assert type(restored).__module__ == "score"


# test_chunk_retry_isolation.py


class _AssertingBlock:
    name = "rf3"

    def __call__(self, handler, sandbox_dir, **_kwargs):
        sandbox_dir = sandbox_dir.resolve()
        assert not (sandbox_dir / "outputs" / "old_model.cif").exists()
        (sandbox_dir / "outputs").mkdir()
        (sandbox_dir / "outputs" / "new_model.cif").write_text("new")
        return handler


def test_retry_reset_removes_stale_inputs_outputs_and_terminal_state(tmp_path):
    chunk_dir = tmp_path / "rf3" / "chunk_0"
    stale_output = chunk_dir / "outputs" / "old_design" / "old_model.cif"
    stale_input = chunk_dir / "inputs" / "old_chunk.json"
    stale_output.parent.mkdir(parents=True)
    stale_input.parent.mkdir(parents=True)
    stale_output.write_text("old model")
    stale_input.write_text("old input")
    (chunk_dir / "terminal.json").write_text('{"status": "error"}')

    _reset_chunk_sandbox_for_execution(chunk_dir)

    assert chunk_dir.is_dir()
    assert list(chunk_dir.iterdir()) == []


def test_execute_chunk_job_resets_failed_sandbox_before_running(tmp_path):
    input_path = tmp_path / ".pipeline" / "input.rf3.parquet"
    input_path.parent.mkdir()
    PipelineDataHandler(
        [{str(Column.STRUCTURES): [], str(Column.METRICS): "{}"}]
    ).write_parquet(input_path)

    chunk_dir = tmp_path / "rf3" / "chunk_0"
    stale = chunk_dir / "outputs" / "old_model.cif"
    stale.parent.mkdir(parents=True)
    stale.write_text("old")
    context = ChunkContext(
        input_path=input_path,
        chunk_slice=(0, 1),
        chunk_id=0,
        steps=[ChainStep(chunk_dir=chunk_dir, _cached_block=_AssertingBlock())],
    )

    result = execute_chunk_job(context)

    assert result == chunk_dir / "output.parquet"
    assert result.is_file()
    assert not stale.exists()
    assert (chunk_dir / "outputs" / "new_model.cif").is_file()


# test_chunking_nfs_retry.py


def test_parquet_visibility_retry_survives_more_than_eight_negative_lookups(
    monkeypatch, tmp_path: Path
):
    expected = pl.DataFrame({"value": [1]})
    calls = 0

    def delayed_read(_path):
        nonlocal calls
        calls += 1
        if calls <= 8:
            raise FileNotFoundError
        return expected

    monkeypatch.setattr(chunking.pl, "read_parquet", delayed_read)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    result = chunking._read_parquet_with_nfs_retry(tmp_path / "output.parquet", delay=0)

    assert calls == 9
    assert result.equals(expected)


def test_parquet_visibility_retry_bypasses_stale_polars_path_cache(
    monkeypatch, tmp_path: Path
):
    path = tmp_path / "output.parquet"
    expected = pl.DataFrame({"value": [1]})
    expected.write_parquet(path)
    real_read_parquet = pl.read_parquet
    calls = []

    def path_is_stale(source):
        calls.append(source)
        if isinstance(source, Path):
            raise FileNotFoundError
        return real_read_parquet(source)

    monkeypatch.setattr(chunking.pl, "read_parquet", path_is_stale)

    result = chunking._read_parquet_with_nfs_retry(path, delay=0)

    assert len(calls) == 2
    assert calls[0] == path
    assert hasattr(calls[1], "read")
    assert result.equals(expected)


# test_container_namespace_fallback.py


def _settings():
    return SimpleNamespace(CHAPERONE_CONTAINER_BACKEND="apptainer")


def test_apptainer_runs_directly_when_private_mount_namespace_is_unavailable(
    monkeypatch,
):
    monkeypatch.setattr(chaperone, "get_settings", _settings)
    monkeypatch.setattr(containers, "_supports_private_mount_namespace", lambda: False)
    original_is_dir = Path.is_dir
    monkeypatch.setattr(Path, "is_dir", lambda path: True if str(path) == "/projects" else original_is_dir(path))
    image = containers.Image.from_existing("test", "/tmp/test.sif")

    command = image.build_command("true", exec_mode="exec")

    assert command[:2] == ["apptainer", "exec"]
    assert "unshare" not in command
    assert "/projects:/projects" in command
    assert "/projects:/mnt/projects" in command


def test_apptainer_keeps_private_mount_wrapper_when_supported(monkeypatch):
    monkeypatch.setattr(chaperone, "get_settings", _settings)
    monkeypatch.setattr(containers, "_supports_private_mount_namespace", lambda: True)
    image = containers.Image.from_existing("test", "/tmp/test.sif")

    command = image.build_command("true", exec_mode="exec")

    assert command[:2] == ["unshare", "--mount"]
    assert command[command.index("apptainer") : command.index("apptainer") + 2] == [
        "apptainer",
        "exec",
    ]


# test_execution_tracking.py


def test_execution_tracking_writes_one_terminal_record(tmp_path):
    step_dir = tmp_path / "fold"
    sandbox = step_dir / "chunk_0"

    with track_execution(
        step_dir=step_dir,
        sandbox_dir=sandbox,
        block_name="fold",
        source="test",
        chunk_id=0,
    ):
        pass

    records = load_execution_records(step_dir)
    assert len(records) == 1
    assert records[0].status == "succeeded"
    assert records[0].wall_seconds >= 0


def test_execution_tracking_keeps_failure_accounting(tmp_path):
    step_dir = tmp_path / "fold"
    sandbox = step_dir / "chunk_0"

    with pytest.raises(RuntimeError, match="bad fold"):
        with track_execution(
            step_dir=step_dir,
            sandbox_dir=sandbox,
            block_name="fold",
            source="test",
            chunk_id=0,
        ):
            raise RuntimeError("bad fold")

    records = load_execution_records(step_dir)
    assert len(records) == 1
    assert records[0].status == "failed"
    assert records[0].error == "RuntimeError: bad fold"


# test_harness_hardening.py


def test_aromatic_neighbors_can_be_restricted_to_ligand_core():
    atoms = AtomArray(4)
    atoms.coord = np.array(
        [[0.0, 0.0, 0.0], [20.0, 0.0, 0.0], [20.5, 0.0, 0.0], [0.5, 0.0, 0.0]]
    )
    atoms.atom_name = np.array(["CORE", "TAIL", "CB", "CB"])
    atoms.res_name = np.array(["BTN", "BTN", "PHE", "TRP"])
    atoms.chain_id = np.array(["B", "B", "A", "A"])
    atoms.res_id = np.array([1, 1, 10, 11])
    atoms.set_annotation("is_ligand", np.array([True, True, False, False]))

    assert find_aromatic_residues_near_ligand(
        atoms, distance_threshold=6.0, ligand_atom_names=["CORE"]
    ) == ["A11"]


def test_mount_control_environment_is_never_forwarded():
    clean = _sanitize_env(
        {
            "KEEP": "yes",
            "APPTAINER_BIND": "/controller:/escape",
            "SINGULARITY_BINDPATH": "/controller:/escape",
        }
    )
    assert clean == {"KEEP": "yes"}


class _FailingBlock:
    name = "fail"

    def __call__(self, handler, **kwargs):
        raise RuntimeError("primary worker failure")


def test_worker_failure_is_nonzero_and_has_terminal_report(tmp_path: Path):
    input_path = tmp_path / "input.parquet"
    PipelineDataHandler([{"structures": [], "metrics": "{}"}]).write_parquet(input_path)
    chunk_dir = tmp_path / "fail" / "chunk_0"
    context = ChunkContext(
        input_path=input_path,
        chunk_slice=(0, 1),
        chunk_id=0,
        steps=[ChainStep(chunk_dir=chunk_dir, _cached_block=_FailingBlock())],
    )

    with pytest.raises(RuntimeError, match="primary worker failure"):
        execute_chunk_job(context, allow_errors=True)

    terminal = json.loads((chunk_dir / "terminal.json").read_text())
    assert terminal["status"] == "error"
    assert terminal["exit_code"] == 1
    assert terminal["error"]["type"] == "RuntimeError"
    assert terminal["error"]["message"] == "primary worker failure"
    assert terminal["input_count"] == 1
    assert terminal["input_row_ids"] == [0]


# test_job_args.py


class _Executor:
    def __init__(self):
        self.parameters = None

    def update_parameters(self, **parameters):
        self.parameters = parameters


class _ResourceFreeBlock:
    _chunked = True
    _rows_per_job = 7

    def __init__(self, name):
        self.name = name

    def get_resources(self):
        return None


@pytest.fixture
def executor(monkeypatch):
    executor = _Executor()
    monkeypatch.setattr(
        job_args_module, "_create_slurm_executor", lambda **_kwargs: executor
    )
    return executor


def test_submitit_parameters_are_forwarded(executor):
    cases = [
        ({"constraint": "Ampere"}, {}, "slurm_constraint", "Ampere"),
        (
            {"rows_per_job": 32, "slurm_array_parallelism": 5},
            {},
            "slurm_array_parallelism",
            5,
        ),
        (
            {"slurm_array_parallelism": 5},
            {"slurm_array_parallelism": 2},
            "slurm_array_parallelism",
            2,
        ),
    ]
    for config, overrides, parameter, expected in cases:
        assert JobArgs(**config).create_executor(**overrides) is executor
        assert executor.parameters[parameter] == expected


def test_array_parallelism_falls_back_to_pipeline_setting(monkeypatch):
    monkeypatch.setattr(
        chaperone,
        "get_setting_field",
        lambda name, default: (
            5 if name == "CHAPERONE_SLURM_ARRAY_PARALLELISM" else default
        ),
    )

    assert JobArgs().effective_slurm_array_parallelism() == 5


def test_array_parallelism_must_be_positive():
    with pytest.raises(ValueError, match="must be >= 1"):
        JobArgs(slurm_array_parallelism=0)


def test_reusing_job_args_does_not_leak_generated_job_name_between_blocks():
    shared = JobArgs(partition="cpu", mem_gb=8)
    run_args = RuntimeArgs(use_local_gpus=False)

    first = _resolve_job_args(shared, _ResourceFreeBlock("first"), run_args)
    second = _resolve_job_args(shared, _ResourceFreeBlock("second"), run_args)

    assert first.job_name == "first"
    assert second.job_name == "second"
    assert first.rows_per_job == second.rows_per_job == 7
    assert shared.job_name is None
    assert shared.rows_per_job is None


# test_slurm_future_nfs.py


def test_result_ready_refreshes_parent_directory_before_exists(tmp_path, monkeypatch):
    result = tmp_path / "job_result.pkl"
    result.write_bytes(b"done")
    job = SimpleNamespace(paths=SimpleNamespace(result_pickle=result))
    future = SlurmFuture(job, poll_interval=0, max_poll_time=1)
    calls = []

    def record_listdir(path):
        calls.append(path)
        return [result.name]

    monkeypatch.setattr(
        "chaperone.orchestration.slurm_futures.os.listdir", record_listdir
    )

    assert future._result_ready() is True
    assert calls == [tmp_path]
