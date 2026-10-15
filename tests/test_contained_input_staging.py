"""External inputs remain readable without mounting an entire host filesystem."""
from types import SimpleNamespace

import pytest

import chaperone
from chaperone.block_config.compiled import CompiledBlockFunction, ScriptSpec
from chaperone.containers import Image


class StagingProbe(CompiledBlockFunction):
    _name = "staging_probe"
    _script_spec = ScriptSpec(command=["true"], image=Image("staging_probe"))

    def create_command(self, handler, sandbox_dir, chunk_id):
        return list(self.command_template)

    def compile(self, sandbox_dir):
        return None


@pytest.fixture
def staged_context(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    run = tmp_path / "runs"
    run.mkdir()
    settings = SimpleNamespace(
        CHAPERONE_CONTAINER_BACKEND="apptainer",
        CHAPERONE_CONTAINER_BIND_WORKSPACE=True,
        CHAPERONE_CONTAINER_WORKSPACE=workspace,
        CHAPERONE_CONTAINER_BIND_RUNDIR=True,
        RUNDIR=run,
    )
    monkeypatch.setattr(chaperone, "get_settings", lambda: settings)
    monkeypatch.delenv("HARBOR_HOST_WORKSPACE", raising=False)
    block = StagingProbe()
    sandbox = run / "chunk_0"
    sandbox.mkdir()
    return block, workspace, sandbox


def test_external_inputs_with_equal_names_are_copied_without_collisions(staged_context, tmp_path):
    block, workspace, sandbox = staged_context
    staged = []
    for name in ("first", "second"):
        source = tmp_path / name / "input.cif"
        source.parent.mkdir()
        source.write_text(name)
        target = block.stage_file(source, sandbox)
        assert target.is_relative_to(sandbox)
        assert target.name == source.name
        assert target.read_text() == name
        assert not target.is_symlink()
        staged.append(target)
    assert len(set(staged)) == 2


def test_visible_inputs_reuse_existing_mounts(staged_context):
    block, workspace, sandbox = staged_context
    for source in (workspace / "input.cif", sandbox.parent / "previous.cif"):
        source.write_text("structure")
        assert block.stage_file(source, sandbox) == source
        destination = sandbox / source.name
        block.stage_or_copy(source, destination)
        assert destination.is_symlink()
        assert destination.read_text() == "structure"


def test_workspace_symlink_to_external_input_is_materialized(staged_context, tmp_path):
    block, workspace, sandbox = staged_context
    external = tmp_path / "outside.cif"
    external.write_text("external structure")
    alias = workspace / "alias.cif"
    alias.symlink_to(external)
    target = block.stage_file(alias, sandbox)
    assert target.is_relative_to(sandbox)
    assert not target.is_symlink()
    destination = sandbox / "prepared.cif"
    destination.symlink_to(external)
    block.stage_or_copy(alias, destination)
    assert not destination.is_symlink()
    external.unlink()
    assert target.read_text() == destination.read_text() == "external structure"


def test_serialized_worker_uses_submission_mount_context(staged_context, tmp_path):
    block, workspace, sandbox = staged_context
    worker = tmp_path / "different-worker-directory"
    worker.mkdir()
    source = workspace / "input.cif"
    source.write_text("structure")
    block._container_bind_settings = {
        "CHAPERONE_CONTAINER_BIND_WORKSPACE": True,
        "CHAPERONE_CONTAINER_WORKSPACE": worker,
        "CHAPERONE_CONTAINER_BIND_RUNDIR": True,
        "RUNDIR": sandbox.parent,
    }
    target = block.stage_file(source, sandbox)
    assert target.is_relative_to(sandbox)
    assert not target.is_symlink()


def test_cloud_still_copies_visible_host_inputs(staged_context):
    block, workspace, sandbox = staged_context
    block.cloud = True
    source = workspace / "input.cif"
    source.write_text("structure")
    target = block.stage_file(source, sandbox)
    assert target.is_relative_to(sandbox)
    destination = sandbox / "cloud-input.cif"
    block.stage_or_copy(source, destination)
    assert not destination.is_symlink()
    assert destination.read_text() == "structure"


def test_active_runtime_context_is_preserved_without_block_snapshot(staged_context, tmp_path):
    from chaperone.containers import container_runtime_bind_settings
    block, workspace, sandbox = staged_context
    source = tmp_path / "assigned-input.cif"
    source.write_text("structure")
    # The enclosing worker explicitly mounts its assigned input directory.
    context = {
        "CHAPERONE_CONTAINER_BIND_WORKSPACE": True,
        "CHAPERONE_CONTAINER_WORKSPACE": tmp_path,
        "CHAPERONE_CONTAINER_BIND_RUNDIR": True,
        "RUNDIR": sandbox.parent,
    }
    with container_runtime_bind_settings(context):
        assert block.stage_file(source, sandbox) == source
