"""Portable and digs policies retain the same enforcement and accounting paths."""

import argparse
import json

import pytest
from site_config import get_site, configure_site, policy_instructions
from environments.apptainer import ApptainerEnvironment, _runtime_bind_pairs
from scripts.slurm_gpu_guard import (
    GuardError,
    prepare_generic_submission,
    validate_request,
)


def configure(monkeypatch, tmp_path, **overrides):
    monkeypatch.setenv("CHAPERONE_EVAL_SITE", "portable")
    config = tmp_path / "site.json"
    config.write_text(json.dumps(overrides))
    monkeypatch.setenv("CHAPERONE_EVAL_SITE_CONFIG", str(config))
    return get_site()


def test_portable_defaults_have_no_institutional_assets(monkeypatch):
    monkeypatch.setenv("CHAPERONE_EVAL_SITE", "portable")
    site = get_site()
    assert site["agent_image"] == site["worker_image"] == site["baseline_image"] == ""
    assert not any(
        p.startswith(("/net/", "/projects/"))
        for pair in _runtime_bind_pairs()
        for p in pair
    )
    assert site["gpu_type"] == site["gpu_constraint"] == ""
    assert json.loads(site["environment"]["CHAPERONE_GRES_MAP"])["small"] == "gpu:1"
    assert "gpu:N" in policy_instructions("harness")
    assert "A4000" not in policy_instructions("baseline")


def test_digs_preserves_images_and_gpu_accounting():
    site = get_site()
    assert site["worker_python"] == "/opt/pipelines/.venv/bin/python"
    assert site["gpu_type"] == "small"
    assert site["gpu_constraint"] == "A4000"
    assert site["max_total_gpus"] == 5
    assert site["max_requested_gpus_per_submission"] == 30
    assert "A4000" in policy_instructions("harness")
    assert (
        validate_request([], ["#SBATCH --gres=gpu:small:1"], 5, site["gpu_type"]) == 1
    )
    with pytest.raises(GuardError):
        validate_request([], ["#SBATCH --gres=gpu:1"], 5, site["gpu_type"])


@pytest.mark.parametrize(
    "gpu_type,partition,constraint", [("", "", ""), ("a100", "accelerated", "ampere")]
)
def test_external_generic_submission_policy(
    monkeypatch, tmp_path, gpu_type, partition, constraint
):
    site = configure(
        monkeypatch,
        tmp_path,
        gpu_type=gpu_type,
        gpu_partition=partition,
        cpu_partition="batch" if partition else "",
        gpu_constraint=constraint,
        account="lab",
        qos="normal",
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    script = workspace / "job.sh"
    gres = "gpu:" + (gpu_type + ":" if gpu_type else "") + "1"
    script.write_text(
        "#!/bin/bash\n#SBAT" + "CH --job-name=test\n#SBATCH --output=logs/%j.out\n"
        "#SBATCH --error=logs/%j.err\n#SBATCH --mem=1G\n#SBATCH --cpus-per-task=1\n"
        "#SBATCH --time=10\n"
        + (f"#SBATCH --partition={partition}\n" if partition else "")
        + f"#SBATCH --gres={gres}\n#SBATCH --array=0-2%2\npython run.py\n"
    )
    kwargs = {
        key: site[key]
        for key in (
            "cpu_partition",
            "gpu_partition",
            "gpu_constraint",
            "account",
            "qos",
        )
    }
    argv, count, payload, directives = prepare_generic_submission(
        [str(script)],
        workspace,
        workspace,
        workspace,
        tmp_path / "logs",
        allowed_gpu_type=gpu_type,
        **kwargs,
    )
    assert count == 1
    assert f"--gres={gres}" in argv
    assert "--account=lab" in argv and "--qos=normal" in argv
    assert (f"--constraint={constraint}" in argv) == bool(constraint)
    assert (f"--partition={partition}" in argv) == bool(partition)
    with pytest.raises(GuardError, match="trial"):
        prepare_generic_submission(
            [str(script)],
            workspace,
            workspace,
            workspace,
            tmp_path / "logs",
            max_total_gpus=1,
            allowed_gpu_type=gpu_type,
            **kwargs,
        )
    with pytest.raises(GuardError, match="across all array tasks"):
        prepare_generic_submission(
            [str(script)],
            workspace,
            workspace,
            workspace,
            tmp_path / "logs",
            max_requested_gpus_per_submission=2,
            allowed_gpu_type=gpu_type,
            **kwargs,
        )


def test_wrapper_uses_external_python_assets_and_settings(monkeypatch, tmp_path):
    asset = tmp_path / "weights"
    asset.mkdir()
    configure(
        monkeypatch,
        tmp_path,
        worker_python="/opt/custom/bin/python",
        runtime_binds=[str(asset)],
        environment={"HBPLUS_COPY": str(asset / "hbplus")},
    )
    workspace = tmp_path / "workspace"
    (workspace / "chaperone/src").mkdir(parents=True)
    (tmp_path / "image.sif").touch()
    wrapper = tmp_path / "worker"
    ApptainerEnvironment._prepare_slurm_python(
        wrapper, workspace, tmp_path / "image.sif"
    )
    text = wrapper.read_text()
    assert "exec /opt/custom/bin/python" in text
    assert str(asset) + ":" + str(asset) + ":ro" in text
    assert "export HBPLUS_COPY=" in text
    assert "/projects/ml/" not in text and "/opt/pipelines/" not in text


def test_supplied_env_file_is_allowlisted(monkeypatch, tmp_path):
    env = tmp_path / ".env.ipd"
    env.write_text("HBPLUS_COPY=/custom/hbplus\nOPENAI_API_KEY=secret\n")
    site = configure(monkeypatch, tmp_path, env_file=str(env))
    assert site["environment"]["HBPLUS_COPY"] == "/custom/hbplus"
    assert "OPENAI_API_KEY" not in site["environment"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"environment": {"OPENAI_API_KEY": "secret"}},
        {"runtime_binds": ["relative"]},
        {"max_total_gpus": 0},
        {"gpu_type": "a100;evil"},
        {"unknown": 1},
    ],
)
def test_bad_site_config_fails_before_submission(monkeypatch, tmp_path, overrides):
    with pytest.raises(ValueError):
        configure(monkeypatch, tmp_path, **overrides)


def test_cli_propagates_images_and_profile(monkeypatch, tmp_path):
    monkeypatch.setenv("CHAPERONE_EVAL_SITE", "portable")
    args = argparse.Namespace(
        site="portable",
        site_config=None,
        image=tmp_path / "agent.sif",
        worker_image=tmp_path / "worker.sif",
        arm="harness",
        no_score=False,
        cpu_partition=None,
    )
    configure_site(args)
    assert get_site()["agent_image"] == str(args.image)
    assert get_site()["worker_image"] == str(args.worker_image)
    assert args.cpu_partition == ""


def test_no_score_still_rejects_private_mount(monkeypatch, tmp_path):
    import run_eval
    import argparse

    private = tmp_path / "private"
    private.mkdir()
    monkeypatch.setenv("CHAPERONE_EVALS_PRIVATE", str(private))
    configure(monkeypatch, tmp_path, runtime_binds=[str(tmp_path)])
    with pytest.raises(SystemExit, match="mounted into the agent"):
        run_eval._preflight(argparse.Namespace(no_score=True), tmp_path / "task")
