"""A local pipeline inside a submitit allocation must retain its driver."""

from chaperone import (
    CompiledBlockFunction,
    ExecutionEngine,
    JobArgs,
    PipelineDataHandler,
)
from chaperone.data import make_row
from chaperone.engine.chunking import _should_use_exec_finisher, execute_chunks
from chaperone.engine.resolve import RuntimeArgs


class ReturningBlock(CompiledBlockFunction):
    _name = "returning"
    _chunked = True

    def create_command(self, handler, sandbox_dir, chunk_id):
        raise AssertionError("no external command is needed")

    def compile(self, sandbox_dir):
        raise AssertionError("no external command is needed")

    def __call__(self, handler, **kwargs):
        return handler

    def stage_then_exec_finisher(self, **kwargs):
        raise AssertionError("a local driver must not be replaced")


def test_local_driver_inherits_submitit_environment_and_finishes(monkeypatch, tmp_path):
    monkeypatch.setenv("SUBMITIT_FOLDER", str(tmp_path / "outer-job"))
    monkeypatch.setenv("SUBMITIT_EXECUTOR", "slurm")
    block = ReturningBlock(command_template=["unused"])
    h = PipelineDataHandler([make_row("input", "/input.cif", design_id="one")])
    outputs = execute_chunks(
        [block],
        h,
        tmp_path / "pipeline",
        RuntimeArgs(
            mode="local",
            use_local_gpus=False,
            max_retries=0,
            job_args=JobArgs(rows_per_job=1),
        ),
    )
    assert len(outputs) == 1
    assert len(PipelineDataHandler(outputs[0])) == 1


def test_environment_alone_does_not_authorize_exec(monkeypatch):
    monkeypatch.setenv("SUBMITIT_FOLDER", "/outer-job")
    monkeypatch.setenv("SUBMITIT_EXECUTOR", "slurm")
    kw = {
        "block": ReturningBlock(command_template=["unused"]),
        "is_last": True,
        "handler": object(),
        "recompile": False,
    }
    assert not _should_use_exec_finisher(**kw)
    assert _should_use_exec_finisher(**kw, dedicated_submitit_worker=True)
    assert not _should_use_exec_finisher(
        **{**kw, "is_last": False}, dedicated_submitit_worker=True
    )
    assert not _should_use_exec_finisher(
        **{**kw, "recompile": True}, dedicated_submitit_worker=True
    )
    monkeypatch.delenv("SUBMITIT_EXECUTOR")
    assert not _should_use_exec_finisher(**kw, dedicated_submitit_worker=True)


def test_cycling_retains_container_context_and_runs_all_rounds(monkeypatch, tmp_path):
    monkeypatch.setenv("SUBMITIT_FOLDER", str(tmp_path / "outer-job"))
    monkeypatch.setenv("SUBMITIT_EXECUTOR", "slurm")
    engine = ExecutionEngine(
        tmp_path / "pipeline",
        run_args_default={"mode": "local", "use_local_gpus": False, "max_retries": 0},
    )
    h = PipelineDataHandler([make_row("input", "/input.cif", design_id="one")])
    for i, child in enumerate(engine.cycle("refine", 2)):
        assert child._container_bind_settings == engine._container_bind_settings
        block = ReturningBlock(command_template=["unused"])
        h = child.run(block, data=h)
        assert len(h) == 1
        assert (engine.rundir / f"refine/cycle_{i}/returning/output.parquet").is_file()
