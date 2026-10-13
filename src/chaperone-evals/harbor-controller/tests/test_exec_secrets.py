"""Container environment variables must not appear on the apptainer command line."""

from __future__ import annotations

import asyncio
import json
import stat
from pathlib import Path

import pytest

from environments.apptainer import ApptainerEnvironment

FAKE_APPTAINER = """#!/usr/bin/env python3
import json, os, sys
record = {
    "argv": sys.argv[1:],
    "env": {k: v for k, v in os.environ.items() if k.startswith("APPTAINERENV_")},
}
with open(os.environ["FAKE_APPTAINER_LOG"], "w") as handle:
    json.dump(record, handle)
print("done")
"""

SECRET = "sk-test-0123456789-secret"


@pytest.fixture
def environment(tmp_path, monkeypatch):
    fake = tmp_path / "bin" / "apptainer"
    fake.parent.mkdir()
    fake.write_text(FAKE_APPTAINER)
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{fake.parent}:/usr/bin:/bin")
    monkeypatch.setenv("FAKE_APPTAINER_LOG", str(tmp_path / "log.json"))
    instance = object.__new__(ApptainerEnvironment)
    instance.environment_dir = tmp_path
    instance._sif_path = tmp_path / "image.sif"
    instance._staging_dir = tmp_path / "staging"
    instance._singularity_no_mount = None
    instance._gpu_enabled = False
    instance._workdir = "/workspace"
    instance._merge_env = lambda env: dict(env or {})
    instance._bind_args = lambda: []
    instance._output_callback = lambda: None
    return instance, tmp_path / "log.json"


def run(instance, env):
    return asyncio.run(instance.exec("echo hi", env=env))


def test_secret_is_not_in_argv_but_reaches_the_container(environment):
    instance, log = environment

    result = run(instance, {"OPENAI_API_KEY": SECRET, "PLAIN": "value"})

    record = json.loads(log.read_text())
    assert result.return_code == 0
    assert SECRET not in " ".join(record["argv"])
    assert "--env" not in record["argv"]
    assert record["env"]["APPTAINERENV_OPENAI_API_KEY"] == SECRET
    assert record["env"]["APPTAINERENV_PLAIN"] == "value"


def test_home_is_still_not_forwarded(environment):
    instance, log = environment

    run(instance, {"HOME": "/elsewhere", "KEEP": "1"})

    record = json.loads(log.read_text())
    assert "APPTAINERENV_HOME" not in record["env"]
    assert record["env"]["APPTAINERENV_KEEP"] == "1"


def test_command_and_flags_are_unchanged(environment):
    instance, log = environment

    run(instance, {"A": "1"})

    argv = json.loads(log.read_text())["argv"]
    assert argv[0] == "exec"
    for flag in ("--cleanenv", "--containall", "--pid", "--no-eval"):
        assert flag in argv
    assert argv[argv.index("--pwd") + 1] == "/workspace"
    assert argv[-3:] == ["bash", "-lc", "echo hi"]


def test_an_explicit_value_beats_a_host_apptainerenv_of_the_same_name(
    environment, monkeypatch
):
    instance, log = environment
    monkeypatch.setenv("APPTAINERENV_TOKEN", "from-host-shell")

    run(instance, {"TOKEN": "from-trial"})

    assert json.loads(log.read_text())["env"]["APPTAINERENV_TOKEN"] == "from-trial"


def test_staging_dir_is_recreated(environment):
    instance, _ = environment
    run(instance, {})
    assert Path(instance._staging_dir).is_dir()
