from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts.trial_audit import (
    parse_sacct_allocations,
    reconcile_compute,
    write_trial_audit,
)


def _completed(stdout: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")


def _registry(trial: Path, *job_ids: str) -> Path:
    path = trial / ".runtime" / "slurm-broker" / "submissions.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        "".join(
            json.dumps({"schema_version": 1, "job_id": job_id}) + "\n"
            for job_id in job_ids
        )
    )
    return path


def test_sacct_counts_array_tasks_failures_and_cancellations_without_parents():
    output = "\n".join(
        (
            "100|COMPLETED|100|cpu=2,gres/gpu=2|2|",
            "100_0|COMPLETED|10|cpu=1,gres/gpu:a4000=1,gres/gpu=1|1|",
            "100_0.batch|COMPLETED|10|cpu=1,gres/gpu=1|1|",
            "100_1|FAILED|20|cpu=1,gres/gpu:a4000=1,gres/gpu=1|1|",
            "200|CANCELLED|30|cpu=2,gres/gpu:a4000=2,gres/gpu=2|2|",
        )
    )

    jobs, warnings = parse_sacct_allocations(output, ["100", "200"])

    assert not warnings
    assert [row["job_id"] for row in jobs["100"]["rows"]] == ["100_0", "100_1"]
    assert jobs["100"]["gpu_seconds"] == 30
    assert jobs["200"]["gpu_seconds"] == 60
    assert jobs["200"]["rows"][0]["state"] == "CANCELLED"


def test_execution_records_reconcile_a_job_missing_from_sacct(tmp_path: Path):
    trial = tmp_path / "trial"
    _registry(trial, "300")
    execution = trial / "workspace" / "run" / "step" / ".execution"
    execution.mkdir(parents=True)
    (execution / "attempt.json").write_text(
        json.dumps(
            {
                "backend": "slurm",
                "job_id": "300",
                "status": "failed",
                "gpu_seconds": 45,
            }
        )
    )

    compute, warnings = reconcile_compute(trial, runner=lambda *_a, **_k: _completed())

    assert compute["accounting_complete"] is True
    assert compute["gpu_hours"] == 45 / 3600
    assert compute["jobs"]["300"]["source"] == "pipelines_execution"
    assert any("execution fallback" in warning for warning in warnings)


def test_uncovered_registered_job_marks_accounting_incomplete(tmp_path: Path):
    trial = tmp_path / "trial"
    _registry(trial, "400")

    compute, warnings = reconcile_compute(trial, runner=lambda *_a, **_k: _completed())

    assert compute["accounting_complete"] is False
    assert compute["uncovered_job_ids"] == ["400"]
    assert any("accounting is incomplete" in warning for warning in warnings)


def test_audit_indexes_transcript_and_accepts_designs_only(tmp_path: Path):
    trial = tmp_path / "trial"
    submission = trial / "artifacts" / "workspace" / "submission"
    submission.mkdir(parents=True)
    designs = submission / "designs.csv"
    designs.write_text("sequence,design_id\nAAAA,d0\nCCCC,d1\n")
    transcript = trial / "agent" / "codex.txt"
    transcript.parent.mkdir()
    transcript.write_text(
        json.dumps(
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 100,
                    "cached_input_tokens": 80,
                    "output_tokens": 20,
                    "reasoning_output_tokens": 5,
                },
            }
        )
        + "\n"
    )
    _registry(trial)
    (trial / "result.json").write_text(
        json.dumps(
            {
                "agent_execution": {
                    "started_at": "2026-08-18T00:00:00Z",
                    "finished_at": "2026-08-18T00:02:00Z",
                }
            }
        )
    )

    audit_path, run_meta_path, audit = write_trial_audit(
        trial,
        designs,
        "ppi_pdl1",
        2,
        runner=lambda *_a, **_k: _completed(),
    )

    assert audit["contract_ok"] is True
    assert audit["audit_ok"] is True
    assert audit["submission"]["row_count"] == 2
    assert audit["transcript"]["path"] == str(transcript.absolute())
    assert audit["transcript"]["final_usage"]["output_tokens"] == 20
    assert json.loads(audit_path.read_text()) == audit
    run_meta = json.loads(run_meta_path.read_text())
    assert run_meta["source"] == "harbor-controller"
    assert run_meta["wallclock_min"] == 2


def test_missing_audit_data_does_not_change_submission_contract(tmp_path: Path):
    trial = tmp_path / "trial"
    designs = trial / "workspace" / "submission" / "designs.csv"
    designs.parent.mkdir(parents=True)
    designs.write_text("sequence\nAAAA\n")

    _audit_path, _run_meta_path, audit = write_trial_audit(
        trial,
        designs,
        "ppi_pdl1",
        1,
        runner=lambda *_a, **_k: _completed(),
    )

    assert audit["contract_ok"] is True
    assert audit["audit_ok"] is False
    assert audit["warnings"]


def test_legacy_auxiliary_files_do_not_affect_contract(tmp_path: Path):
    trial = tmp_path / "trial"
    submission = trial / "workspace" / "submission"
    submission.mkdir(parents=True)
    designs = submission / "designs.csv"
    designs.write_text("sequence\nAAAA\n")
    (submission / "terminal_report.json").write_text("not json")
    (submission / "run_meta.json").write_text('{"gpu_hours": 999999}')
    (submission / "lineage.jsonl").write_text("not json\n")

    _audit_path, run_meta_path, audit = write_trial_audit(
        trial,
        designs,
        "ppi_pdl1",
        1,
        runner=lambda *_a, **_k: _completed(),
    )

    assert audit["contract_ok"] is True
    assert json.loads(run_meta_path.read_text())["gpu_hours"] == 0
