"""Real command failures must stop both compiled execution paths before compile."""

import json
from pathlib import Path
import pickle
import subprocess
import sys

import pytest

from chaperone import CompiledBlockFunction, PipelineDataHandler
from chaperone.block_config.manifest import Manifest
from chaperone.block_config import runner
from chaperone.execution_tracking import load_execution_records


class _CommandBlock(CompiledBlockFunction):
    _name = "command_failure_regression"
    _requires_input = False
    exit_code: int = 0

    def get_command_template(self):
        return []

    def create_command(self, handler, sandbox_dir, chunk_id):
        script = Path(sandbox_dir) / "command.py"
        script.write_text(
            "from pathlib import Path\n"
            f"exit_code = {self.exit_code}\n"
            "if exit_code == 0:\n"
            "    Path('sequences.csv').write_text('design_id,sequence_A\\nfresh,ACDE\\n')\n"
            "print('command ran', flush=True)\n"
            "raise SystemExit(exit_code)\n"
        )
        return [sys.executable, str(script)]

    def compile(self, sandbox_dir):
        sandbox = Path(sandbox_dir)
        (sandbox / "compile.called").touch()
        return PipelineDataHandler.from_sequences(sandbox / "sequences.csv")


def stale_sandbox(tmp_path):
    sandbox = tmp_path / "step" / "chunk_0"
    sandbox.mkdir(parents=True)
    (sandbox / "sequences.csv").write_text("design_id,sequence_A\nstale,GGGG\n")
    return sandbox


def test_failed_real_command_does_not_compile_stale_outputs(tmp_path):
    sandbox = stale_sandbox(tmp_path)
    with pytest.raises(subprocess.CalledProcessError) as caught:
        _CommandBlock(exit_code=17)(
            PipelineDataHandler([{"structures": [], "metrics": "{}"}]),
            sandbox,
            chunk_id=0,
        )
    assert caught.value.returncode == 17
    assert "command ran" in caught.value.output
    assert not (sandbox / "compile.called").exists()
    assert "stale,GGGG" in (sandbox / "sequences.csv").read_text()
    records = load_execution_records(sandbox.parent)
    assert len(records) == 1 and records[0].status == "failed"
    assert "CalledProcessError" in records[0].error


def test_successful_real_command_compiles_new_outputs(tmp_path):
    sandbox = stale_sandbox(tmp_path)
    result = _CommandBlock()(
        PipelineDataHandler([{"structures": [], "metrics": "{}"}]), sandbox, chunk_id=0
    )
    assert (sandbox / "compile.called").exists()
    assert len(result) == 1
    design = next(result.get_prediction_inputs(load_context=False))
    assert design.design_id == "fresh" and design.sequences == {"A": "ACDE"}
    records = load_execution_records(sandbox.parent)
    assert len(records) == 1 and records[0].status == "succeeded"


def test_finisher_wait_rejects_failed_command_before_compile(monkeypatch, tmp_path):
    sandbox = stale_sandbox(tmp_path)
    block = _CommandBlock(exit_code=19)
    block.create_manifest(
        PipelineDataHandler([{"structures": [], "metrics": "{}"}]), sandbox, chunk_id=0
    )
    state = {
        "sandbox_dir": str(sandbox),
        "step_dir": str(sandbox.parent),
        "chunk_id": 0,
        "block_name": block.name,
        "result_pickle": str(tmp_path / "result.pkl"),
        "terminal_path": str(sandbox / "terminal.json"),
        "output_path": str(sandbox / "output.parquet"),
        "block_path": str(tmp_path / "unused-block.pkl"),
    }

    def must_not_compile(_state):
        raise AssertionError("Failed command reached the compile phase")

    monkeypatch.setattr(runner, "_phase_finish", must_not_compile)
    assert runner._phase_wait(state) == 1
    terminal = json.loads(Path(state["terminal_path"]).read_text())
    assert terminal["status"] == "error"
    assert terminal["error"]["type"] == "ManifestExecutionError"
    assert "CalledProcessError" in terminal["error"]["message"]
    with Path(state["result_pickle"]).open("rb") as handle:
        status, traceback = pickle.load(handle)
    assert status == "error" and "CalledProcessError" in traceback
    records = load_execution_records(sandbox.parent)
    assert len(records) == 1 and records[0].status == "failed"
    assert "stale,GGGG" in (sandbox / "sequences.csv").read_text()


def test_manifest_exec_still_returns_inspectable_nonzero_result(tmp_path):
    Manifest(
        name="inspection",
        command=[sys.executable, "-c", "raise SystemExit(23)"],
        cwd=str(tmp_path),
    ).write(tmp_path)
    result = Manifest.exec(tmp_path)
    assert result.returncode == 23
