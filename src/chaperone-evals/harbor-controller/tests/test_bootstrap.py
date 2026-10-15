"""The public controller defaults to its containing checkout and supports an override."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/bootstrap.sh"


@pytest.mark.parametrize("external", [False, True])
def test_bootstrap_separates_environments_and_installs_local_source(tmp_path, external):
    containing = tmp_path / "public chaperone"
    project = containing / "src/chaperone-evals"
    controller = project / "harbor-controller"
    script = controller / "scripts/bootstrap.sh"
    script.parent.mkdir(parents=True)
    shutil.copy2(SCRIPT, script)
    public = tmp_path / "other chaperone" if external else containing
    (public / "src/chaperone").mkdir(parents=True)
    (public / "pyproject.toml").write_text("[project]\nname='chaperone-pipelines'\n")
    for checkout in (project, public):
        python = checkout / ".venv/bin/python"
        python.parent.mkdir(parents=True)
        python.write_text("#!/bin/sh\nexit 0\n")
        python.chmod(0o755)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "uv-args"
    uv = bindir / "uv"
    uv.write_text('#!/bin/sh\nprintf "%s\\n" "$@" >> "$UV_TEST_LOG"\n')
    uv.chmod(0o755)
    env = {**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}", "UV_TEST_LOG": str(log)}
    env.pop("CHAPERONE_REPO", None)
    if external:
        env["CHAPERONE_REPO"] = str(public)
    subprocess.run(["bash", str(script)], env=env, check=True, capture_output=True)
    args = log.read_text().splitlines()
    assert str(project / ".venv/bin/python") in args
    assert str(public / ".venv/bin/python") in args
    assert str(public) in args
    assert str(project) + "[test]" in args
    assert "--override" not in args
    assert (controller / ".venv").resolve() == project / ".venv"


def test_bootstrap_rejects_non_chaperone_checkout(tmp_path):
    env = {**os.environ, "CHAPERONE_REPO": str(tmp_path)}
    result = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert "Chaperone checkout not found" in result.stderr
