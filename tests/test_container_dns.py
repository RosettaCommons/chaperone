"""Contained tools need a resolver, without widening evaluation asset mounts."""

from types import SimpleNamespace

import pytest

import chaperone
from chaperone import containers


@pytest.fixture
def settings(monkeypatch):
    settings = SimpleNamespace(CHAPERONE_CONTAINER_BACKEND="apptainer")
    monkeypatch.setattr(chaperone, "get_settings", lambda: settings)
    monkeypatch.setattr(containers, "_supports_private_mount_namespace", lambda: False)
    return settings


@pytest.mark.parametrize("private_namespace", [False, True])
def test_apptainer_binds_existing_upstream_resolver_read_only(monkeypatch, tmp_path, settings, private_namespace):
    upstream = tmp_path / "upstream-resolv.conf"
    upstream.write_text("nameserver 192.0.2.1\n")
    stub = tmp_path / "stub-resolv.conf"
    stub.write_text("nameserver 127.0.0.53\n")
    monkeypatch.setattr(containers, "_RESOLVER_PATHS", (upstream, stub))
    monkeypatch.setattr(containers, "_supports_private_mount_namespace", lambda: private_namespace)
    command = containers.Image("probe").build_command("python", "-c", "pass")
    assert f"{upstream}:/etc/resolv.conf:ro" in command
    assert f"{stub}:/etc/resolv.conf:ro" not in command
    assert command[command.index("--no-mount") + 1] == "home,tmp,bind-paths"


def test_resolver_falls_back_to_regular_symlink_target(monkeypatch, tmp_path, settings):
    missing = tmp_path / "missing"
    empty = tmp_path / "empty"
    empty.touch()
    directory = tmp_path / "directory"
    directory.mkdir()
    actual = tmp_path / "actual.conf"
    actual.write_text("nameserver 192.0.2.2\n")
    alias = tmp_path / "resolv.conf"
    alias.symlink_to(actual)
    monkeypatch.setattr(containers, "_RESOLVER_PATHS", (missing, empty, directory, alias))
    command = containers.Image("probe").build_command("true")
    assert f"{actual}:/etc/resolv.conf:ro" in command


def test_missing_resolver_adds_no_invalid_bind(monkeypatch, tmp_path, settings):
    monkeypatch.setattr(containers, "_RESOLVER_PATHS", (tmp_path / "missing",))
    command = containers.Image("probe").build_command("true")
    assert not any("/etc/resolv.conf" in arg for arg in command)


@pytest.mark.parametrize("destination", ["/etc/resolv.conf", "/etc/resolv.conf:ro", "/etc:ro"])
def test_explicit_resolver_override_is_preserved(monkeypatch, tmp_path, settings, destination):
    automatic = tmp_path / "automatic.conf"
    automatic.write_text("nameserver 192.0.2.1\n")
    monkeypatch.setattr(containers, "_RESOLVER_PATHS", (automatic,))
    explicit = str(tmp_path / "assigned.conf")
    command = containers.Image("probe").build_command("true", binds=[(explicit, destination)])
    assert f"{explicit}:{destination}" in command
    assert f"{automatic}:/etc/resolv.conf:ro" not in command


def test_docker_keeps_its_own_dns(monkeypatch, settings):
    settings.CHAPERONE_CONTAINER_BACKEND = "docker"

    def forbidden(*args):
        pytest.fail("Docker must not probe or bind the host resolver")

    monkeypatch.setattr(containers, "_runtime_resolver_bind", forbidden)
    command = containers.Image("probe").build_command("true")
    assert command[:2] == ["docker", "run"]
    assert not any("/etc/resolv.conf" in arg for arg in command)


def test_harbor_dns_does_not_enable_inferred_source_mounts(monkeypatch, tmp_path, settings):
    workspace = tmp_path / "assigned-workspace"
    workspace.mkdir()
    resolver = tmp_path / "assigned-resolv.conf"
    resolver.write_text("nameserver 192.0.2.1\n")
    outside = tmp_path / "private-source"
    outside.mkdir()
    script = outside / "wrapper.py"
    script.write_text("pass\n")
    settings.CHAPERONE_CONTAINER_BIND_WORKSPACE = True
    monkeypatch.setenv("HARBOR_HOST_WORKSPACE", str(workspace))
    monkeypatch.setattr(containers, "_RESOLVER_PATHS", (resolver,))
    command = containers.Image("probe").build_command("python", str(script))
    assert f"{workspace}:{workspace}" in command
    assert f"{resolver}:/etc/resolv.conf:ro" in command
    assert not any(arg.startswith(f"{outside}:") for arg in command)
