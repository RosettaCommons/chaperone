"""Portable and digs policies retain the same enforcement and accounting paths."""

import argparse
import json
from types import SimpleNamespace

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
        environment={
            "HBPLUS_COPY": str(asset / "hbplus"),
            "CHAPERONE_MPNN_CONTAINER": "legacy",
        },
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
    assert "export CHAPERONE_MPNN_CONTAINER=legacy" in text
    assert "/projects/ml/" not in text and "/opt/pipelines/" not in text


@pytest.mark.parametrize("profile", ["portable", "digs"])
def test_trial_uses_same_writable_af3_cache_in_agent_and_worker(monkeypatch, tmp_path, profile):
    """A site cache under read-only assets must not shadow the writable trial."""
    image = tmp_path / "agent.sif"
    image.touch()
    assets = tmp_path / "readonly-assets"
    assets.mkdir()
    shared_cache = assets / "af3-cache"
    shared_cache.mkdir()
    configure(monkeypatch, tmp_path, worker_image=str(image),
              runtime_binds=[str(assets)],
              environment={"AF3_JAX_CACHE_DIR": str(shared_cache)})
    monkeypatch.setenv("CHAPERONE_EVAL_SITE", profile)
    captured = {}

    def initialize_base(self, *args, **kwargs):
        captured.update(kwargs)

    def prepare_workspace(workspace, seed, inputs):
        (workspace / "chaperone" / "src").mkdir(parents=True)
        # The worker must override even a checkout default after sourcing it.
        (workspace / "chaperone" / ".env").write_text(f"AF3_JAX_CACHE_DIR={shared_cache}\n")

    monkeypatch.setattr("environments.apptainer.SingularityEnvironment.__init__", initialize_base)
    monkeypatch.setattr(ApptainerEnvironment, "_prepare_workspace", staticmethod(prepare_workspace))
    monkeypatch.setattr(ApptainerEnvironment, "_find_codex_vendor", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(ApptainerEnvironment, "_find_resolver_config", staticmethod(lambda: tmp_path / "resolv.conf"))
    monkeypatch.setattr("environments.apptainer.SlurmBrokerProcess",
                        lambda **kwargs: SimpleNamespace(socket_dir=tmp_path / "broker"))
    trial = tmp_path / "trial"
    ApptainerEnvironment(
        environment_dir=tmp_path,
        environment_name="nested-smoke",
        session_id="nested-smoke",
        trial_paths=SimpleNamespace(trial_dir=trial),
        task_env_config=SimpleNamespace(docker_image=str(image)),
        workspace_seed=tmp_path / "seed", task_inputs=tmp_path / "inputs",
        persistent_env={"AF3_JAX_CACHE_DIR": "/untrusted/override"},
    )
    workspace = (trial / "workspace").resolve()
    cache = workspace / ".cache" / "af3_jax"
    assert cache.is_dir()
    assert captured["persistent_env"]["AF3_JAX_CACHE_DIR"] == str(cache)
    script = (trial / ".runtime" / "slurm-python").read_text()
    assert f"export AF3_JAX_CACHE_DIR={cache}" in script
    assert script.index("export AF3_JAX_CACHE_DIR=") > script.index("source ")
    assert f"export AF3_JAX_CACHE_DIR={shared_cache}" not in script
    assert any(mount["source"] == str(assets) and mount["read_only"]
               for mount in captured["mounts"])


def test_supplied_env_file_is_allowlisted(monkeypatch, tmp_path):
    env = tmp_path / ".env.ipd"
    env.write_text("HBPLUS_COPY=/custom/hbplus\nOPENAI_API_KEY=secret\n")
    site = configure(monkeypatch, tmp_path, env_file=str(env))
    assert site["environment"]["HBPLUS_COPY"] == "/custom/hbplus"
    assert "OPENAI_API_KEY" not in site["environment"]


@pytest.mark.parametrize("selector", ["auto", "public", "legacy"])
@pytest.mark.parametrize("source", ["json", "env_file"])
def test_mpnn_selector_reaches_scientific_environment(
    monkeypatch, tmp_path, selector, source
):
    from site_config import scoring_environment

    environment = {"CHAPERONE_MPNN_CONTAINER": selector}
    if source == "env_file":
        env = tmp_path / ".env.ipd"
        env.write_text(f"CHAPERONE_MPNN_CONTAINER={selector}\n")
        site = configure(monkeypatch, tmp_path, env_file=str(env))
    else:
        site = configure(monkeypatch, tmp_path, environment=environment)
    assert site["environment"]["CHAPERONE_MPNN_CONTAINER"] == selector
    assert scoring_environment()["CHAPERONE_MPNN_CONTAINER"] == selector


def test_mpnn_json_selector_overrides_supplied_env_file(monkeypatch, tmp_path):
    env = tmp_path / ".env.ipd"
    env.write_text("CHAPERONE_MPNN_CONTAINER=legacy\n")
    site = configure(
        monkeypatch, tmp_path, env_file=str(env),
        environment={"CHAPERONE_MPNN_CONTAINER": "public"},
    )
    assert site["environment"]["CHAPERONE_MPNN_CONTAINER"] == "public"


@pytest.mark.parametrize("source", ["json", "env_file"])
def test_invalid_mpnn_selector_fails_before_submission(monkeypatch, tmp_path, source):
    overrides = {"environment": {"CHAPERONE_MPNN_CONTAINER": "other"}}
    if source == "env_file":
        env = tmp_path / ".env.ipd"
        env.write_text("CHAPERONE_MPNN_CONTAINER=other\n")
        overrides = {"env_file": str(env)}
    with pytest.raises(ValueError, match="CHAPERONE_MPNN_CONTAINER"):
        configure(monkeypatch, tmp_path, **overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"environment": {"OPENAI_API_KEY": "secret"}},
        {"environment": {"CHAPERONE_CONTAINER_WORKSPACE": "/untrusted"}},
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
