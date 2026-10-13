"""Consolidated functional tests for monitoring."""

from __future__ import annotations

import json
from pathlib import Path
import polars as pl
import pytest
from pipelines_cli.campaign_inspect import InspectError, inspect_campaign
import os
import subprocess
import sys
from unittest.mock import patch
from pipelines_cli import job_wait


# test_campaign_inspect.py


def _write_campaign(root: Path) -> Path:
    campaign = root / "campaigns" / "example"
    rundir = campaign / "run"
    generate = rundir / "generate"
    screen = rundir / "screen"
    generate.mkdir(parents=True)
    screen.mkdir()
    pl.DataFrame({"design_id": ["d1", "d2", "d3"]}).write_parquet(
        generate / "output.parquet"
    )
    pl.DataFrame({"design_id": ["d1", "d2"]}).write_parquet(screen / "output.parquet")
    (rundir / "pipeline.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "block_name": "source",
                        "name": "generate",
                        "pipeline_input_names": [],
                        "parquet": "generate/output.parquet",
                    },
                    {
                        "block_name": "filter",
                        "name": "screen",
                        "pipeline_input_names": ["generate"],
                        "parquet": "screen/output.parquet",
                    },
                ]
            }
        )
    )

    execution = screen / ".execution"
    execution.mkdir()
    attempt = {
        "block_name": "screen",
        "status": "failed",
        "ended_at": "2026-08-18T12:00:00+00:00",
        "wall_seconds": 7200,
        "gpu_seconds": 3600,
        "attempt": 2,
        "chunk_id": 7,
        "job_id": "123",
        "error": "OutOfMemoryError: allocation failed",
    }
    (execution / "attempt.json").write_text(json.dumps(attempt))
    logs = rundir / "logs"
    logs.mkdir()
    (logs / "screen.err").write_text(
        "ordinary progress\nTraceback: CUDA out of memory\n"
    )

    submission = campaign / "submission"
    submission.mkdir()
    (submission / "designs.csv").write_text("design_id,sequence\nd1,AAAA\nd2,CCCC\n")
    (campaign / "score.json").write_text(
        json.dumps(
            {
                "eval": "example",
                "contract_ok": True,
                "n_reported": 2,
                "required_n": 2,
                "n_pass": 1,
                "pass_rate": 0.5,
                "per_criterion": {"fold": 2, "binding": 1},
                "metric_summaries": {"large": {str(i): i for i in range(100)}},
            }
        )
    )
    (campaign / "score_per_design.csv").write_text(
        "design_id,sequence,loss,pass_fold,pass\n"
        "d1,AAAA,2.0,true,true\n"
        "d2,CCCC,1.0,true,true\n"
        "d3,DDDD,0.5,false,false\n"
    )
    return campaign


@pytest.fixture
def campaign(tmp_path: Path) -> Path:
    return _write_campaign(tmp_path)


def test_summary_aggregates_existing_artifacts_without_large_metrics(campaign: Path):
    result = inspect_campaign(campaign=campaign, view="summary")

    assert result["pipelines"]["runs"] == 1
    assert result["pipelines"]["completed_steps"] == 2
    assert result["pipelines"]["execution"] == {
        "attempts": 1,
        "statuses": {"failed": 1},
        "wall_hours": 2.0,
        "gpu_hours": 1.0,
    }
    assert result["submissions"]["sample"][0]["designs"] == 2
    score = result["scores"]["sample"][0]
    assert score["n_pass"] == 1
    assert "metric_summaries" not in score


def test_failures_returns_bounded_records_and_relevant_log_lines(campaign: Path):
    result = inspect_campaign(campaign=campaign, view="failures", max_items=1)

    assert result["failed_attempts"] == 1
    assert result["failures"][0]["chunk"] == 7
    assert "OutOfMemoryError" in result["failures"][0]["error"]
    assert len(result["log_evidence"]) == 1
    assert "out of memory" in result["log_evidence"][0]["excerpt"]
    assert "ordinary progress" not in result["log_evidence"][0]["excerpt"]


def test_funnel_reads_only_row_counts_and_honors_output_limit(campaign: Path):
    result = inspect_campaign(campaign=campaign, view="funnel", max_items=1)

    assert result["total_steps"] == 2
    assert result["steps"] == [
        {"rundir": "run", "step": "generate", "parents": [], "rows": 3}
    ]
    assert result["truncated"] is True


def test_candidates_returns_only_passing_ids_and_requested_public_metric(
    campaign: Path,
):
    result = inspect_campaign(
        campaign=campaign,
        view="candidates",
        max_items=1,
        rank_by="loss",
        direction="min",
    )

    assert result["total"] == 3
    assert result["eligible"] == 2
    assert result["criterion_pass_counts"] == {"pass": 2, "pass_fold": 2}
    assert result["candidates"] == [{"design_id": "d2", "pass": True, "loss": 1.0}]
    assert "sequence" not in json.dumps(result)


def test_candidates_rejects_an_unknown_rank_column(campaign: Path):
    with pytest.raises(InspectError, match="rank_by column"):
        inspect_campaign(campaign=campaign, view="candidates", rank_by="secret")


def test_candidates_trim_artifact_fields_and_keep_response_bounded(campaign: Path):
    (campaign / "score_per_design.csv").write_text(
        "design_id,pass\n" + "d" * 10_000 + ",true\n"
    )

    result = inspect_campaign(campaign=campaign, view="candidates", max_items=20)

    assert len(result["candidates"][0]["design_id"]) == 200
    assert len(json.dumps(result).encode()) < 32 * 1024


# test_init_cli_help.py

REPO = Path(__file__).resolve().parents[1]


def test_init_cli_supports_conventional_help_flags():
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO / "src")
    for help_flag in ("-h", "--help", "--helpshort", "--helpfull"):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from pipelines import init_cli; "
                    f"init_cli(['campaign.py', {help_flag!r}], blocks=[])"
                ),
            ],
            cwd=REPO,
            env=environment,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert "--config" in result.stdout
        assert "UnrecognizedFlagError" not in result.stderr


# test_job_wait.py


def test_workspace_paths_accept_harbor_alias_and_real_path(tmp_path: Path):
    host_path = tmp_path / "results" / "done"
    with patch.dict(os.environ, {"HARBOR_HOST_WORKSPACE": str(tmp_path)}):
        assert job_wait._workspace_path("/workspace/results/done", None) == host_path
        assert job_wait._workspace_path(str(host_path), None) == host_path


def test_workspace_paths_accept_equivalent_shared_filesystem_spellings():
    cases = (
        ("/net/scratch", "/mnt/net/scratch"),
        ("/mnt/net/scratch", "/net/scratch"),
        ("/projects", "/mnt/projects"),
        ("/mnt/projects", "/projects"),
    )
    for workspace_root, requested_root in cases:
        suffix = "ml/user" if workspace_root.endswith("projects") else "user"
        workspace = Path(workspace_root) / suffix / "trial/workspace"
        requested = f"{requested_root}/{suffix}/trial/workspace/results/done"
        expected = workspace / "results" / "done"
        assert (
            job_wait._translate_workspace_path(Path(requested), workspace) == expected
        )
        assert job_wait._workspace_path(requested, workspace) == expected.resolve(
            strict=False
        )


def test_shared_filesystem_aliases_do_not_weaken_workspace_confinement():
    workspace = Path("/net/scratch/user/trial/workspace")
    for path in (
        "/mnt/net/scratch/user/trial/workspace-other/done",
        "/mnt/net/scratch/user/trial/workspace/../escaped",
        "/mnt/projects/ml/user/trial/workspace/results/done",
    ):
        with pytest.raises(job_wait.WaitError, match="outside the workspace"):
            job_wait._workspace_path(path, workspace)


def test_workspace_paths_reject_escape(tmp_path: Path):
    with patch.dict(os.environ, {"HARBOR_HOST_WORKSPACE": str(tmp_path)}):
        for path in ("../escaped", "/etc/passwd", "/workspace/../etc/passwd"):
            with pytest.raises(job_wait.WaitError, match="outside the workspace"):
                job_wait._workspace_path(path, None)


def test_wait_reports_accounted_slurm_failure():
    with (
        patch.object(job_wait, "active_slurm_state", return_value=None),
        patch.object(job_wait, "accounted_slurm_state", return_value="OUT_OF_MEMORY"),
    ):
        event = job_wait.wait_for_completion(job_ids=["123"], poll_seconds=0.01)

    assert event["status"] == "failed"
    assert event["slurm_state"] == "OUT_OF_MEMORY"


def test_launch_and_wait_uses_argv_without_shell_and_confines_paths(tmp_path: Path):
    marker = tmp_path / "done"
    injected = tmp_path / "injected"
    driver = tmp_path / "driver"
    driver.write_text("#!/bin/sh\ntouch done\n")
    driver.chmod(0o700)
    with patch.dict(os.environ, {"HARBOR_HOST_WORKSPACE": str(tmp_path)}):
        event = job_wait.launch_and_wait(
            argv=[str(driver), f"; touch {injected}"],
            cwd="/workspace",
            completion_markers=[str(marker)],
            poll_seconds=0.01,
            timeout_seconds=5,
        )

    assert event["status"] == "completed"
    assert marker.is_file()
    assert not injected.exists()


def test_python_launch_rejects_code_modules_and_outside_scripts(tmp_path: Path):
    python = str(sys.executable)
    outside = tmp_path.parent / "outside.py"
    outside.write_text("print('outside')\n")
    for argv in (
        [python, "-c", "print('unsafe')"],
        [python, "-m", "module"],
        [python, str(outside)],
        [python, "/workspace/missing.py"],
        [python, "-I", "/workspace/missing.py"],
    ):
        with pytest.raises(job_wait.WaitError):
            job_wait._validated_argv(argv, tmp_path, tmp_path)


def test_mcp_exposes_wait_launch_and_inspect_from_pipelines(tmp_path: Path):
    marker = tmp_path / "done"
    marker.touch()
    campaign = tmp_path / "campaigns" / "example"
    campaign.mkdir(parents=True)
    (campaign / "submission").mkdir()
    (campaign / "submission" / "designs.csv").write_text(
        "design_id,sequence\nd1,AAAA\n"
    )

    def request(id, method, **params):
        return {"jsonrpc": "2.0", "id": id, "method": method, "params": params}

    requests = (
        request(1, "initialize"),
        request(2, "tools/list"),
        request(
            3, "tools/call", name="wait", arguments={"completion_markers": ["done"]}
        ),
        request(
            4,
            "tools/call",
            name="inspect",
            arguments={"campaign": "campaigns/example", "view": "summary"},
        ),
    )
    environment = os.environ.copy()
    environment["HARBOR_HOST_WORKSPACE"] = str(tmp_path)
    source = str(Path(__file__).resolve().parents[1] / "src")
    environment["PYTHONPATH"] = source + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pipelines_cli.main",
            "monitor-mcp",
            "--server-name",
            "campaign-monitor",
        ],
        input="\n".join(map(json.dumps, requests)) + "\n",
        text=True,
        capture_output=True,
        timeout=10,
        check=True,
        env=environment,
    )
    responses = [json.loads(line) for line in completed.stdout.splitlines()]

    assert responses[0]["result"]["serverInfo"] == {
        "name": "campaign-monitor",
        "version": "2.1",
    }
    assert [tool["name"] for tool in responses[1]["result"]["tools"]] == [
        "wait",
        "launch_and_wait",
        "inspect",
    ]
    event = json.loads(responses[2]["result"]["content"][0]["text"])
    assert event["status"] == "completed"
    inspection = json.loads(responses[3]["result"]["content"][0]["text"])
    assert inspection["view"] == "summary"
    assert inspection["submissions"]["sample"] == [
        {"path": "submission/designs.csv", "designs": 1}
    ]
