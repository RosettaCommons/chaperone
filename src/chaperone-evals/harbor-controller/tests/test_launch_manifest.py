"""The launch manifest records provenance without exposing held-out locations."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

import launch_manifest
from launch_manifest import (
    asset_record,
    build_manifest,
    mark_submitted,
    redacted_argv,
    write_manifest,
)
from task_paths import PRIVATE_TASKS_ENV

SECRET_ROOT = "/very/secret/private/evals_task"


@pytest.fixture
def launch(tmp_path, monkeypatch):
    public = tmp_path / "public" / "demo"
    (public / "inputs").mkdir(parents=True)
    (public / "inputs" / "target.pdb").write_text("ATOM\n")
    private = tmp_path / "private"
    (private / "demo" / "score").mkdir(parents=True)
    (private / "demo" / "score" / "score.py").write_text("print('scorer')\n")
    monkeypatch.setenv(PRIVATE_TASKS_ENV, str(private))
    wrapper = tmp_path / "wrapper"
    wrapper.mkdir()
    (wrapper / "prompt.md").write_text("Submit 10 rows\n")
    (wrapper / "instruction.md").write_text("Follow the prompt\n")
    agent, worker = tmp_path / "agent.sif", tmp_path / "worker.sif"
    agent.write_bytes(b"agent image")
    worker.write_bytes(b"worker image")
    args = Namespace(
        arm="harness",
        no_score=False,
        image=agent,
        worker_image=worker,
        workspace_seed=tmp_path / "seeds" / "abc123",
        private_tasks_path=Path(SECRET_ROOT),
        required_n=10,
        model="m",
    )
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    return args, public, wrapper, jobs


def test_manifest_records_the_launch(launch):
    args, task, wrapper, jobs = launch

    manifest = build_manifest(
        args, task, wrapper, "run-1", jobs, ["demo", "--arm", "harness"]
    )

    assert manifest["status"] == "prepared"
    assert manifest["bundle_sha256"] == "abc123"
    assert manifest["task"]["name"] == "demo"
    assert "target.pdb" in manifest["task"]["public_inputs"]
    assert manifest["task"]["rendered_prompt_sha256"]
    assert (
        manifest["images"]["agent"]["sha256"] != manifest["images"]["worker"]["sha256"]
    )
    assert manifest["scorer_sha256"]
    assert manifest["controller"]["python"]


def test_manifest_never_contains_the_private_location(launch):
    args, task, wrapper, jobs = launch

    manifest = build_manifest(
        args,
        task,
        wrapper,
        "run-1",
        jobs,
        ["demo", "--private-tasks-path", SECRET_ROOT],
    )
    text = json.dumps(manifest)

    assert SECRET_ROOT not in text
    assert str(Path(launch[0].image).parent / "private") not in text
    assert "score.py" not in text  # only the scorer's hash is kept
    assert manifest["args"]["private_tasks_path"] == "<redacted>"


def test_redacted_argv_handles_both_option_spellings():
    assert redacted_argv(["a", "--private-tasks-path", "/x", "b"]) == [
        "a",
        "--private-tasks-path",
        "<redacted>",
        "b",
    ]
    assert redacted_argv(["--private-tasks-path=/x"]) == [
        "--private-tasks-path=<redacted>"
    ]


def test_scorer_is_not_hashed_when_scoring_is_off(launch):
    args, task, wrapper, jobs = launch
    args.no_score = True

    assert (
        build_manifest(args, task, wrapper, "run-1", jobs, [])["scorer_sha256"] is None
    )


def test_baseline_arm_has_no_bundle(launch):
    args, task, wrapper, jobs = launch
    args.arm = "baseline"

    assert (
        build_manifest(args, task, wrapper, "run-1", jobs, [])["bundle_sha256"] is None
    )


def test_image_is_hashed_once_while_unchanged(tmp_path, monkeypatch):
    image = tmp_path / "big.sif"
    image.write_bytes(b"x" * 1000)
    cache = tmp_path / "cache.json"
    calls = []
    real = launch_manifest._sha256
    monkeypatch.setattr(
        launch_manifest, "_sha256", lambda p: calls.append(p) or real(p)
    )

    first = asset_record(image, cache)
    second = asset_record(image, cache)

    assert len(calls) == 1
    assert first["sha256"] == second["sha256"]
    assert second["digest_cached"] is True


def test_changed_image_is_rehashed(tmp_path):
    image = tmp_path / "big.sif"
    cache = tmp_path / "cache.json"
    image.write_bytes(b"one")
    first = asset_record(image, cache)
    image.write_bytes(b"two!")

    assert asset_record(image, cache)["sha256"] != first["sha256"]


def test_skipping_digests_records_identity_only(tmp_path):
    image = tmp_path / "big.sif"
    image.write_bytes(b"abc")

    record = asset_record(image, tmp_path / "cache.json", digest=False)

    assert record["sha256"] is None
    assert record["size"] == 3
    assert not (tmp_path / "cache.json").exists()


def test_missing_image_is_reported_not_fatal(tmp_path):
    assert asset_record(tmp_path / "gone.sif", tmp_path / "c.json") == {
        "path": str(tmp_path / "gone.sif"),
        "present": False,
    }


def test_write_then_mark_submitted(launch):
    args, task, wrapper, jobs = launch
    manifest = build_manifest(args, task, wrapper, "run-1", jobs, [])

    path = write_manifest(manifest, jobs)
    mark_submitted(path, "4242", jobs / "results" / "run-1-4242")

    saved = json.loads(path.read_text())
    assert path.parent == jobs / "manifests"
    assert saved["status"] == "submitted"
    assert saved["slurm_job_id"] == "4242"
    assert not list(path.parent.glob("*.tmp"))
