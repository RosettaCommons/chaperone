"""The broker owns authentication; status queries cannot escape a trial."""

import json
import socket
import threading

import pytest

from environments.apptainer import ApptainerEnvironment
from scheduler_isolation import validate_runtime_binds
from scripts.slurm_gpu_guard import _BrokerServer, _send_message, _receive_message
from site_config import get_site


@pytest.mark.parametrize(
    "source",
    [
        "/run/munge",
        "/run",
        "/etc",
        "/",
        "/usr",
        "/etc/slurm",
        "/usr/bin/sacct",
        "/var/run/munge/munge.socket.2",
    ],
)
def test_credentials_and_enclosing_mounts_are_rejected(source):
    with pytest.raises(ValueError, match="forbidden"):
        validate_runtime_binds([source])
    with pytest.raises(ValueError, match="forbidden"):
        validate_runtime_binds([["/safe/assets", source]])


def test_credential_alias_is_rejected(tmp_path):
    alias = tmp_path / "assets"
    alias.symlink_to("/run/munge")
    with pytest.raises(ValueError, match="forbidden"):
        validate_runtime_binds([str(alias)])


@pytest.mark.parametrize("site", ["portable", "digs"])
def test_site_presets_and_worker_have_no_scheduler_mounts(tmp_path, monkeypatch, site):
    monkeypatch.setenv("CHAPERONE_EVAL_SITE", site)
    validate_runtime_binds(get_site()["runtime_binds"])
    workspace = tmp_path / "workspace"
    (workspace / "chaperone/src").mkdir(parents=True)
    image = tmp_path / "worker.sif"
    image.touch()
    wrapper = tmp_path / ".runtime/slurm-python"
    wrapper.parent.mkdir()
    ApptainerEnvironment._prepare_slurm_python(wrapper, workspace, image)
    script = wrapper.read_text()
    for forbidden in (
        "/run/munge",
        "/etc/slurm",
        "/usr/bin/srun",
        "/usr/bin/scontrol",
        "SLURM_JWT",
        "compgen -A variable SLURM_",
    ):
        assert forbidden not in script
    assert "--no-mount hostfs" in script
    assert "--no-mount cwd" in script
    assert "SLURM_JOB_ID" in script


@pytest.fixture
def broker(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    executable = tmp_path / "fake-slurm"
    executable.write_text(
        "#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n"
    )
    executable.chmod(0o700)
    sock = tmp_path / "broker.sock"
    server = _BrokerServer(
        str(sock),
        *([str(executable)] * 5),
        workspace,
        executable,
        1,
        "a100",
        real_sacct=str(executable),
    )
    server._job_ids.add("101")
    server._job_names.add("trial-job")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def request(argv):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect(str(sock))
            _send_message(
                client,
                {
                    "command": "sacct",
                    "argv": argv,
                    "cwd": str(workspace),
                    "stdin": "",
                    "env": {},
                },
            )
            return _receive_message(client)

    yield request
    server.shutdown()
    thread.join()
    server.server_close()


@pytest.mark.parametrize(
    "args",
    [
        ["-o", "JobID,State,NodeList", "--parsable2", "-j", "101"],
        ["-j", "101_2", "--format=JobID,State", "--parsable2", "--noheader"],
        [
            "--name=trial-job",
            "--state=FAILED,NODE_FAIL,TIMEOUT,OUT_OF_MEMORY",
            "--starttime=now-2hours",
            "--format=NodeList%50",
            "--parsable2",
            "--noheader",
        ],
        [],
    ],
)
def test_status_queries_are_scoped_over_real_broker_socket(broker, args):
    response = broker(args)
    assert response["returncode"] == 0, response
    argv = json.loads(response["stdout"])
    assert argv[0] in ("--jobs=101", "--jobs=101_2")


@pytest.mark.parametrize(
    "args",
    [
        ["-j", "102"],
        ["-j", "101,102"],
        ["--name=another-trial"],
        ["--allusers"],
        ["--clusters=other"],
        ["--format=SubmitLine"],
        ["-j"],
        ["--jobs=101", "--jobs=102"],
        ["--jobs=101;id"],
    ],
)
def test_status_query_escape_is_rejected(broker, args):
    response = broker(args)
    assert response["returncode"] == 2
    assert response["stdout"] == ""
