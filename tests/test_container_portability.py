"""Release checks that do not require Docker, Apptainer, model weights or a GPU."""
from pathlib import Path
import shlex
import subprocess

import pytest

from chaperone import get_settings
from chaperone.containers import Image
from chaperone_cli.install import _build_apptainer, _build_docker


def test_docker_label_preserves_access_instructions_as_one_value():
    message = 'Requires authorized access; model "legacy" is private'
    image = Image("label-test").base_image("ubuntu:24.04").label(BuildAccess=message)
    line = next(line for line in image.to_dockerfile().splitlines() if line.startswith("LABEL "))
    assert shlex.split(line) == ["LABEL", f"BuildAccess={message}"]
    assert f"BuildAccess {message}" in image.to_singularity()
    literal = Image("literal-test").base_image("ubuntu:24.04").label(Example="$HOME")
    assert 'LABEL Example="\\$HOME"' in literal.to_dockerfile()


def test_dry_run_does_not_clone_or_invoke_runtime(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("A dry run invoked a build/runtime/source download")
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(Image, "prepare_git_clones", forbidden)
    image = Image("preview").base_image("ubuntu:24.04").git_clone(
        "https://github.com/example/private", "/opt/private"
    )
    for build in (_build_docker, _build_apptainer):
        build(image, image.name, tmp_path, dry_run=True)
    assert not list(tmp_path.iterdir())


def test_docker_stages_only_recipe_sources_inside_context(monkeypatch, tmp_path):
    runner = tmp_path / "runner.py"
    runner.write_text("print('hello')")
    secret = tmp_path / "private.env"
    secret.write_text("not a build input")
    cloned = tmp_path / "source-repo"
    cloned.mkdir()
    (cloned / "run.py").write_text("pass")
    image = Image("portable").base_image("ubuntu:24.04").copy(str(runner), "/opt/run.py")
    image = image.copy(str(cloned), "/opt/source")
    captured = []
    def run(command, **kwargs):
        assert command[:2] == ["docker", "build"]
        context = Path(command[-1])
        captured.append(context)
        definition = (context / "Dockerfile").read_text()
        assert str(tmp_path) not in definition
        assert "COPY sources/0/runner.py /opt/run.py" in definition
        assert (context / "sources/0/runner.py").read_text() == runner.read_text()
        assert (context / "sources/1/source-repo/run.py").is_file()
        assert not list(context.rglob("private.env"))
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(subprocess, "run", run)
    _build_docker(image, image.name, tmp_path / "built", force=True)
    assert len(captured) == 1
    assert not captured[0].exists()
    assert image._copies[0][0] == str(runner)  # immutable original recipe


def test_optional_site_mounts_do_not_break_external_linux(monkeypatch, tmp_path):
    monkeypatch.setattr(get_settings(), "CHAPERONE_CONTAINER_BACKEND", "apptainer")
    monkeypatch.setattr("chaperone.common.is_macos", lambda: False)
    missing = str(tmp_path / "missing-institutional-directory")
    explicit = str(tmp_path / "required-input")
    image = Image("portable").base_image("ubuntu:24.04").template_command(
        binds=[(missing, missing)]
    )
    command = image.build_command("true", binds=[(explicit, explicit)])
    assert f"{missing}:{missing}" not in command
    assert f"{explicit}:{explicit}" in command
    required_image = Image.from_existing("legacy", "/tmp/legacy.sif").template_command(
        binds=[(missing, missing)]
    )
    assert f"{missing}:{missing}" in required_image.build_command("true")


def test_foundry_checkpoint_resolution_and_mounts_follow_user_config(monkeypatch, tmp_path):
    from blocks.networks.foundry.container import default_checkpoint, FOUNDRY_CONTAINER
    from blocks.networks.foundry.rfdiffusion3 import RFD3Block
    from blocks.networks.foundry.rfd3na import RFD3NABlock
    from blocks.networks.prediction.rosettafold3 import RosettaFold3Block
    monkeypatch.setattr(get_settings(), "CHAPERONE_CONTAINER_BACKEND", "apptainer")
    monkeypatch.setattr(get_settings(), "FOUNDRY_CHECKPOINT_DIRS", str(tmp_path))
    assert default_checkpoint("rfd3", "rfd3_latest.ckpt") == "rfd3"
    for model, filename, block in (
        ("rfd3", "rfd3_latest.ckpt", RFD3Block),
        ("rf3", "rf3_foundry_01_24_latest_remapped.ckpt", RosettaFold3Block),
        ("rfd3na", "rfd3na_1190.ckpt", RFD3NABlock),
    ):
        assert block().ckpt_path == model
        (tmp_path / filename).touch()
        assert Path(block().ckpt_path) == (tmp_path / filename).resolve()
    # Settings may be supplied after the module is imported.
    command = FOUNDRY_CONTAINER.build_command("true")
    assert f"{tmp_path.resolve()}:{tmp_path.resolve()}" in command


def test_all_reinstallable_recipes_render_and_have_local_copy_sources():
    from blocks import get_all_containers
    from blocks.networks.foundry.container import _FOUNDRY_CUDA_CONTAINER
    images = get_all_containers()
    images["foundry"] = _FOUNDRY_CUDA_CONTAINER
    assert len(images) >= 20
    for name, image in images.items():
        if image.is_existing:
            continue
        assert image.to_dockerfile(), name
        assert image.to_singularity(), name
        for source, _ in image._copies:
            assert Path(source).is_file() or Path(source).is_dir(), (name, source)
            assert not Path(source).is_symlink(), (name, source)
        for url, _, _ in image._git_clones:
            assert not url.startswith("git@"), (name, url)
        commands = "\n".join(command for group in image._commands for command in group)
        checked = subprocess.run(["bash", "-n"], input=commands, text=True, capture_output=True)
        assert checked.returncode == 0, (name, checked.stderr)
        if "gpg --dearmor" in commands:
            assert "gnupg" in image._apt_packages, name
            if image._base_image == "ubuntu:22.04":
                assert "/ubuntu2204/" in commands and "/ubuntu2404/" not in commands, name


def test_ligand_ddg_reports_missing_dalphaball(monkeypatch):
    from blocks.metrics.ligand_ddg import LigandDDGBlock
    monkeypatch.setattr("blocks.metrics.ligand_ddg.shutil.which", lambda _: None)
    block = LigandDDGBlock(rosetta_bin="rosetta_scripts", dalphaball_bin=None)
    with pytest.raises(FileNotFoundError, match="DALPHABALL_BIN"):
        block._dalphaball_command()


def test_ligand_ddg_dalphaball_override_and_custom_protocol(tmp_path):
    from blocks.metrics.ligand_ddg import LigandDDGBlock
    binary = tmp_path / "DAlphaBall with spaces"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    block = LigandDDGBlock(rosetta_bin="rosetta_scripts", dalphaball_bin=str(binary))
    assert block._dalphaball_command() == ["-holes:dalphaball", str(binary)]
    protocol = tmp_path / "simple.xml"
    protocol.write_text("<ROSETTASCRIPTS><FILTERS/></ROSETTASCRIPTS>")
    block = LigandDDGBlock(rosetta_bin="rosetta_scripts", protocol=str(protocol))
    assert block._dalphaball_command() == []


def test_private_source_failure_is_actionable(monkeypatch, tmp_path):
    image = (
        Image("private-test")
        .base_image("ubuntu:24.04")
        .git_clone("https://github.com/example/private", "/opt/private")
        .label(BuildAccess="Requires authorized access to private example/private")
    )
    def unavailable(command, **kwargs):
        raise subprocess.CalledProcessError(128, command)
    monkeypatch.setattr(subprocess, "run", unavailable)
    with pytest.raises(RuntimeError, match="Requires authorized access to private"):
        image.prepare_git_clones(tmp_path)


def test_esmfold2_recipe_uses_released_public_model_and_legacy_images_still_work(monkeypatch):
    import importlib.util
    import sys
    from types import ModuleType, SimpleNamespace
    from blocks.networks.prediction.esmfold2 import ESMFOLD2_CONTAINER, ESMFOLD2_RUNNER_PATH
    definition = ESMFOLD2_CONTAINER.to_dockerfile()
    assert "pip install esm==3.4.1.post1" in definition
    assert "Biohub/transformers.git" not in definition
    assert "torch==2.11.0" in definition
    spec = importlib.util.spec_from_file_location("esmfold2_runner", ESMFOLD2_RUNNER_PATH)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    public = object()
    assert runner._model_class(SimpleNamespace(EsmFold2Model=public)) is public
    legacy_module = ModuleType("transformers.models.esmfold2.modeling_esmfold2")
    legacy = object()
    legacy_module.ESMFold2Model = legacy
    monkeypatch.setitem(sys.modules, legacy_module.__name__, legacy_module)
    assert runner._model_class(SimpleNamespace()) is legacy


def test_ligand_ddg_respects_dalphaball_in_custom_flags(tmp_path):
    import shlex
    from blocks.metrics.ligand_ddg import LigandDDGBlock
    binary = tmp_path / "DAlphaBall custom"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    flags = tmp_path / "flags"
    flags.write_text("-holes:dalphaball " + shlex.quote(str(binary)))
    block = LigandDDGBlock(rosetta_bin="rosetta_scripts", dalphaball_bin=None, flags=str(flags))
    assert block._dalphaball_command() == ["-holes:dalphaball", str(binary)]


def test_fresh_gpu_recipes_have_consistent_blackwell_toolchain():
    from blocks.networks.foundry.container import _FOUNDRY_CUDA_CONTAINER
    from blocks.networks.prediction.simplefold import SIMPLEFOLD_CONTAINER
    foundry = _FOUNDRY_CUDA_CONTAINER.to_dockerfile()
    simplefold = SIMPLEFOLD_CONTAINER.to_dockerfile()
    assert "torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128" in foundry
    assert "12.0+PTX" in foundry
    assert "import fused_layer_norm_cuda" in foundry
    assert "FROM ubuntu:24.04" in simplefold
    assert "python3.12 -m venv" in simplefold
    assert "torch==2.7.1 torchvision==0.22.1" in simplefold
    for definition in (foundry, simplefold):
        assert "cuda-toolkit-12-8" in definition
        assert "/whl/cu128" in definition
        assert "pip check" in definition


def test_default_engine_mounts_resolve_late_and_keep_aliases(monkeypatch, tmp_path):
    from chaperone import ExecutionEngine, PipelineDataHandler, get_settings
    from chaperone.block_config.compiled import CompiledBlockFunction, ScriptSpec

    class MountBlock(CompiledBlockFunction):
        _name = "mount_probe"
        _script_spec = ScriptSpec(command=["true"], image=Image("mount_probe").base_image("ubuntu:24.04"))
        def create_command(self, handler, sandbox_dir, chunk_id):
            return list(self.command_template)
        def compile(self, sandbox_dir):
            return PipelineDataHandler([])

    monkeypatch.delenv("HARBOR_HOST_WORKSPACE", raising=False)
    settings = get_settings()
    monkeypatch.setattr(settings, "RUNDIR", None)
    monkeypatch.setattr(settings, "CHAPERONE_CONTAINER_BACKEND", "apptainer")
    monkeypatch.setattr(settings, "CHAPERONE_CONTAINER_BIND_RUNDIR", True)
    monkeypatch.setattr(settings, "CHAPERONE_CONTAINER_BIND_WORKSPACE", True)
    real = tmp_path / "actual_workspace"
    real.mkdir()
    alias = tmp_path / "alias_workspace"
    alias.symlink_to(real, target_is_directory=True)
    monkeypatch.setattr(settings, "CHAPERONE_CONTAINER_WORKSPACE", alias)
    block = MountBlock()  # Must work before a run directory exists.
    run = alias / "run"
    ExecutionEngine(rundir=run)
    monkeypatch.setattr(block, "_resolve_command", lambda command: command)
    manifest = block.create_manifest(None, run / "chunk_0", 0, write=False)
    command = manifest.command
    for path in (alias, real, run, run.resolve()):
        assert f"{path}:{path}" in command
    assert "/net:/net" not in command
    assert "/mnt/net:/mnt/net" not in command

    custom = MountBlock(command_template=["my-custom-runner"])
    monkeypatch.setattr(custom, "_resolve_command", lambda command: command)
    assert custom.create_manifest(None, run / "custom", 0, write=False).command == ["my-custom-runner"]


def test_eval_workspace_wins_and_mount_optouts_are_preserved(monkeypatch, tmp_path):
    from chaperone.containers import _runtime_workspace_bind, _runtime_script_binds
    from types import SimpleNamespace
    workspace = tmp_path / "eval-workspace"
    workspace.mkdir()
    outside = tmp_path / "private-repo"
    outside.mkdir()
    script = outside / "private.py"
    script.touch()
    settings = SimpleNamespace(CHAPERONE_CONTAINER_BIND_WORKSPACE=True, CHAPERONE_CONTAINER_WORKSPACE=outside)
    monkeypatch.setenv("HARBOR_HOST_WORKSPACE", str(workspace))
    assert _runtime_workspace_bind(settings) == (str(workspace), str(workspace))
    assert _runtime_script_binds(settings, (str(script),)) == []
    settings.CHAPERONE_CONTAINER_BIND_WORKSPACE = False
    assert _runtime_workspace_bind(settings) is None
    assert _runtime_script_binds(settings, (str(script),)) == []


def test_runtime_mount_rejects_symlink_to_broad_directory(tmp_path):
    from chaperone.containers import _runtime_rundir_bind
    from types import SimpleNamespace
    alias = tmp_path / "looks-specific"
    alias.symlink_to("/", target_is_directory=True)
    with pytest.raises(ValueError, match="Refusing broad"):
        _runtime_rundir_bind(SimpleNamespace(CHAPERONE_CONTAINER_BIND_RUNDIR=True, RUNDIR=alias))


def test_engine_captures_container_paths_before_worker_serialization(monkeypatch, tmp_path):
    import cloudpickle
    import chaperone.engine.base as engine_module
    from chaperone import ExecutionEngine, PipelineDataHandler
    from chaperone.block_config.compiled import CompiledBlockFunction, ScriptSpec
    from chaperone.containers import _runtime_workspace_bind

    class WorkerMountBlock(CompiledBlockFunction):
        _name = "worker_mount_probe"
        _script_spec = ScriptSpec(command=["true"], image=Image("worker_mount_probe"))
        def get_command_template(self):
            from chaperone import get_settings
            return self._script_spec.image.build_command("true", env={"MODEL_CACHE": get_settings().ESM_WEIGHTS_DIR})
        def create_command(self, handler, sandbox_dir, chunk_id):
            # Also cover blocks that construct commands dynamically at staging.
            return self.command_template + self._script_spec.to_template()
        def _resolve_command(self, command):
            return command
        def compile(self, sandbox_dir):
            return PipelineDataHandler([])

    settings = get_settings()
    monkeypatch.delenv("HARBOR_HOST_WORKSPACE", raising=False)
    monkeypatch.setattr(settings, "CHAPERONE_CONTAINER_BACKEND", "apptainer")
    monkeypatch.setattr(settings, "CHAPERONE_CONTAINER_BIND_RUNDIR", True)
    monkeypatch.setattr(settings, "CHAPERONE_CONTAINER_BIND_WORKSPACE", True)
    monkeypatch.setattr(settings, "CHAPERONE_CONTAINER_WORKSPACE", None)
    source = tmp_path / "source"
    worker_cwd = tmp_path / "worker"
    source.mkdir()
    worker_cwd.mkdir()
    monkeypatch.chdir(source)
    monkeypatch.setattr(settings, "ESM_WEIGHTS_DIR", str(tmp_path / "submission-cache"))
    engine = ExecutionEngine(rundir=tmp_path / "run")
    block = WorkerMountBlock(store="none")
    monkeypatch.chdir(worker_cwd)

    def worker(blocks, handler, rundir, run_args):
        restored = cloudpickle.loads(cloudpickle.dumps(blocks[0]))
        # Fresh worker process has no in-memory engine settings.
        monkeypatch.setattr(settings, "RUNDIR", None)
        monkeypatch.setattr(settings, "ESM_WEIGHTS_DIR", str(tmp_path / "worker-cache"))
        command = restored.create_manifest(None, rundir / "chunk_0", 0, write=False).command
        for path in (source, rundir):
            assert command.count(f"{path}:{path}") == 2
        assert f"{worker_cwd}:{worker_cwd}" not in command
        assert f"MODEL_CACHE={tmp_path / 'submission-cache'}" in command
        assert f"MODEL_CACHE={tmp_path / 'worker-cache'}" not in command
        # Context does not leak to another block/thread after command staging.
        assert _runtime_workspace_bind(settings) == (str(worker_cwd), str(worker_cwd))
        return []

    monkeypatch.setattr(engine_module, "execute_chunks", worker)
    monkeypatch.setattr(engine_module, "concat_outputs", lambda outputs: None)
    engine._run([block], None, None)


def test_serialized_eval_bind_context_never_infers_private_script_mounts(monkeypatch, tmp_path):
    from chaperone.containers import (
        capture_container_runtime_bind_settings,
        container_runtime_bind_settings,
        _runtime_script_binds,
        _runtime_workspace_bind,
    )
    settings = get_settings()
    monkeypatch.setattr(settings, "CHAPERONE_CONTAINER_BIND_WORKSPACE", True)
    workspace = tmp_path / "eval"
    workspace.mkdir()
    private = tmp_path / "private"
    private.mkdir()
    script = private / "run.py"
    script.touch()
    monkeypatch.setenv("HARBOR_HOST_WORKSPACE", str(workspace))
    captured = capture_container_runtime_bind_settings(workspace / "run")
    monkeypatch.delenv("HARBOR_HOST_WORKSPACE")
    monkeypatch.chdir(private)
    with container_runtime_bind_settings(captured):
        assert _runtime_workspace_bind(settings) == (str(workspace), str(workspace))
        assert _runtime_script_binds(settings, (str(script),)) == []
        # A worker's explicit trusted workspace still takes precedence.
        replacement = tmp_path / "replacement_eval"
        monkeypatch.setenv("HARBOR_HOST_WORKSPACE", str(replacement))
        assert _runtime_workspace_bind(settings) == (str(replacement), str(replacement))


def test_esmfold2_runtime_caches_avoid_home_symlinks(monkeypatch, tmp_path):
    import importlib.util
    from blocks.networks.prediction.esmfold2 import ESMFOLD2_RUNNER_PATH
    spec = importlib.util.spec_from_file_location("esmfold2_cache_runner", ESMFOLD2_RUNNER_PATH)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    variables = ("XDG_CACHE_HOME", "CUEQ_TRITON_CACHE_DIR", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR")
    for variable in (*variables, "ESMCFOLD_CCD_PATH"):
        monkeypatch.delenv(variable, raising=False)
    home = tmp_path / "home"
    home.mkdir()
    (home / ".cache").symlink_to(tmp_path / "unmounted")
    monkeypatch.setenv("HOME", str(home))
    snapshot = tmp_path / "model"
    snapshot.mkdir()
    (snapshot / "ccd.pkl").touch()
    output = tmp_path / "run" / "chunk_0" / "outputs"
    runner._configure_runtime_caches(output, str(snapshot))
    import os
    for variable in variables:
        cache = Path(os.environ[variable])
        assert cache.is_dir()
        assert cache.is_relative_to(output.parent)
    assert os.environ["ESMCFOLD_CCD_PATH"] == str(snapshot / "ccd.pkl")
    explicit = tmp_path / "shared-kernel-cache"
    monkeypatch.setenv("CUEQ_TRITON_CACHE_DIR", str(explicit))
    runner._configure_runtime_caches(output, str(snapshot))
    assert Path(os.environ["CUEQ_TRITON_CACHE_DIR"]) == explicit
    assert explicit.is_dir()


def test_esmfold2_runner_supplies_staged_msa(monkeypatch, tmp_path):
    import importlib.util
    import sys
    from types import ModuleType, SimpleNamespace
    from blocks.networks.prediction.esmfold2 import ESMFOLD2_RUNNER_PATH
    spec = importlib.util.spec_from_file_location("esmfold2_msa_runner", ESMFOLD2_RUNNER_PATH)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    staged = tmp_path / "input.a3m"
    staged.write_text(">query\nACDE\n>hit\nACdDE\n")
    msa = object()
    calls = []
    def from_a3m(path, **kwargs):
        calls.append((path, kwargs))
        assert path.read_text() == staged.read_text()
        return msa
    module = ModuleType("esm.utils.msa")
    module.MSA = SimpleNamespace(from_a3m=from_a3m)
    monkeypatch.setitem(sys.modules, "esm.utils.msa", module)
    spi = runner._build_input({"chains": [
        {"id": "A", "sequence": "ACDE", "msa_path": str(staged)},
        {"id": "B", "sequence": "ACDE"},
    ]}, (SimpleNamespace, SimpleNamespace, SimpleNamespace))
    assert spi.sequences[0].msa is msa
    assert spi.sequences[1].msa is None
    assert calls == [(staged, {"remove_insertions": True})]


def test_af3_build_uses_upstream_lock_and_checks_runner_imports():
    from blocks.networks.prediction.alphafold3 import ALPHAFOLD3_CONTAINER
    definition = ALPHAFOLD3_CONTAINER.to_dockerfile()
    assert "git checkout --detach 71e671112ab5f287fdabd3ce86b5d05805dd0d81" in definition
    assert "uv==0.9.24" in definition
    assert "uv sync --frozen --no-dev --no-editable" in definition
    assert "uv pip check --python /opt/af3-env/bin/python" in definition
    assert "jax[cuda12]>=0.5.0" not in definition
    assert "JAX_PLATFORMS=cpu" in definition
    assert "runpy.run_path('/opt/chaperone-build-smoke/run_alphafold_AF.py'" in definition
    assert any(dst == "/opt/chaperone-build-smoke/run_alphafold_AF.py" for _, dst in ALPHAFOLD3_CONTAINER._copies)


def test_script_bind_does_not_shadow_writable_workspace(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from chaperone.containers import _runtime_script_binds
    monkeypatch.delenv("HARBOR_HOST_WORKSPACE", raising=False)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    script = workspace / "runner.py"
    script.touch()
    settings = SimpleNamespace(CHAPERONE_CONTAINER_BIND_WORKSPACE=True, CHAPERONE_CONTAINER_WORKSPACE=workspace)
    assert _runtime_script_binds(settings, (str(script),)) == []
    outside = tmp_path / "installed-wrappers"
    outside.mkdir()
    wrapper = outside / "runner.py"
    wrapper.touch()
    assert _runtime_script_binds(settings, (str(wrapper),)) == [(str(outside), str(outside) + ":ro")]


def test_wrapper_alias_survives_resolved_cwd_and_serialized_worker(monkeypatch, tmp_path):
    import cloudpickle
    from chaperone import ExecutionEngine, PipelineDataHandler
    from chaperone.block_config.compiled import CompiledBlockFunction, ScriptSpec
    from chaperone.containers import _runtime_script_binds
    real = tmp_path / "mnt-net" / "workspace"
    scripts = real / "src" / "blocks" / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "runner.py").write_text("pass\n")
    alias = tmp_path / "net"
    alias.symlink_to(real.parent, target_is_directory=True)
    wrapper = alias / "workspace" / "src" / "blocks" / "scripts" / "runner.py"
    class AliasWrapperBlock(CompiledBlockFunction):
        _name = "alias_wrapper_probe"
        _script_spec = ScriptSpec(command=["python", str(wrapper)], image=Image("alias_wrapper_probe"))
        def create_command(self, handler, sandbox_dir, chunk_id):
            return list(self.command_template)
        def compile(self, sandbox_dir):
            return PipelineDataHandler([])
    settings = get_settings()
    monkeypatch.delenv("HARBOR_HOST_WORKSPACE", raising=False)
    monkeypatch.setattr(settings, "CHAPERONE_CONTAINER_BACKEND", "apptainer")
    monkeypatch.setattr(settings, "CHAPERONE_CONTAINER_BIND_WORKSPACE", True)
    monkeypatch.setattr(settings, "CHAPERONE_CONTAINER_WORKSPACE", None)
    monkeypatch.setattr(settings, "RUNDIR", None)
    monkeypatch.chdir(real)
    engine = ExecutionEngine(rundir=real / "run")
    block = AliasWrapperBlock(store="none")
    engine._configure_container_binds([block])
    restored = cloudpickle.loads(cloudpickle.dumps(block))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(restored, "_resolve_command", lambda command: command)
    command = restored.create_manifest(None, real / "run/chunk_0", 0, write=False).command
    assert str(wrapper) in command
    assert f"{real}:{real}" in command
    assert f"{wrapper.parent}:{wrapper.parent}" in command
    assert f"{scripts}:{scripts}:ro" not in command
    assert f"{wrapper.parent}:{wrapper.parent}:ro" not in command
    monkeypatch.setenv("HARBOR_HOST_WORKSPACE", str(real))
    assert _runtime_script_binds(settings, (str(wrapper),)) == []


def test_engine_resolves_container_template_before_cache_fingerprint(monkeypatch, tmp_path):
    from chaperone import ExecutionEngine, PipelineDataHandler
    from chaperone.block_config.compiled import CompiledBlockFunction, ScriptSpec

    class CachedMountBlock(CompiledBlockFunction):
        _name = "cached_mount_probe"
        _script_spec = ScriptSpec(command=["true"], image=Image("cached_mount_probe"))
        def create_command(self, handler, sandbox_dir, chunk_id):
            return list(self.command_template)
        def _resolve_command(self, command):
            return command
        def compile(self, sandbox_dir):
            return PipelineDataHandler([])

    settings = get_settings()
    monkeypatch.delenv("HARBOR_HOST_WORKSPACE", raising=False)
    monkeypatch.setattr(settings, "RUNDIR", None)
    monkeypatch.setattr(settings, "CHAPERONE_CONTAINER_BIND_RUNDIR", True)
    monkeypatch.setattr(settings, "CHAPERONE_CONTAINER_WORKSPACE", tmp_path)
    block = CachedMountBlock(store="none")
    engine = ExecutionEngine(rundir=tmp_path / "run")
    data = PipelineDataHandler([{"design_id": "probe", "structures": [], "metrics": "{}"}])
    prepared = engine._prepare(block, run_args={"mode": "local"}, data=data)
    assert str(engine.rundir) in " ".join(prepared.blocks[0].command_template)
    block.create_manifest(data, engine.rundir / "chunk_0", 0, write=False)
    repeated = engine._prepare(block, run_args={"mode": "local"}, data=data)
    assert prepared.cache_fingerprint == repeated.cache_fingerprint


def test_disco_blackwell_recipe_resolves_torch_from_cuda128():
    import tomllib
    from blocks.networks.design.disco.disco import DISCO_CONTAINER, _DISCO_UV_CONFIG
    # Preserve upstream workspace/source tables while adding precise Torch index.
    source = '[tool.uv.sources]\nopenfold = { workspace = true }\n[tool.uv.workspace]\nmembers = ["packages/openfold"]\n'
    config = tomllib.loads(source + _DISCO_UV_CONFIG)
    assert config["tool"]["uv"]["sources"]["torch"]["index"] == "pytorch-cu128"
    assert "torch==2.11.0" in config["tool"]["uv"]["constraint-dependencies"]
    assert "transformers==4.57.6" in config["tool"]["uv"]["constraint-dependencies"]
    definition = DISCO_CONTAINER.to_dockerfile()
    assert "cu124" not in definition
    assert "398098c57e3f1dd6dd6205c059f7792dcc9b9147" in definition
    assert "uv pip check --python /opt/disco/.venv/bin/python" in definition
    assert "import runner.inference" in definition
    assert "ESM rotary buffer layout is incompatible" in definition
    # The inline Python TOML edit must remain a valid shell command in both
    # Docker RUN and Apptainer %post (without executing builds in this test).
    for commands in DISCO_CONTAINER._commands:
        for command in commands:
            subprocess.run(["/bin/sh", "-n"], input=command, text=True, check=True)


@pytest.mark.parametrize("layer_local", [True, False])
def test_disco_esm_checkpoint_gate_rejects_model_level_rotary_layout(monkeypatch, layer_local):
    import sys
    from types import SimpleNamespace
    from blocks.networks.design.disco.disco import _DISCO_ESM_COMPATIBILITY_CHECK
    class ProbeModel:
        def __init__(self, config):
            self.config = config
        def state_dict(self):
            if layer_local:
                return {f"esm.encoder.layer.{i}.attention.self.rotary_embeddings.inv_freq": None for i in range(self.config.num_hidden_layers)}
            return {"esm.rotary_embeddings.inv_freq": None}
        def load_state_dict(self, values, *, strict):
            assert strict is True
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(EsmConfig=lambda **kwargs: SimpleNamespace(**kwargs), EsmForMaskedLM=ProbeModel))
    if layer_local:
        exec(_DISCO_ESM_COMPATIBILITY_CHECK, {})
    else:
        with pytest.raises(RuntimeError, match="incompatible with the DISCO checkpoint"):
            exec(_DISCO_ESM_COMPATIBILITY_CHECK, {})


def test_foundry_build_pins_compatible_atomworks_and_imports_real_engines():
    from blocks.networks.foundry.container import _FOUNDRY_CUDA_CONTAINER, _FOUNDRY_PUBLIC_CONTAINER
    for image in (_FOUNDRY_CUDA_CONTAINER, _FOUNDRY_PUBLIC_CONTAINER):
        definition = image.to_dockerfile()
        assert "git -C /opt/foundry checkout --detach c97ad246877e2c7cb3b50e60aab7d76d02e8d393" in definition
        assert "atomworks[ml]==2.2.1" in definition
        assert "biotite==1.4.0" in definition
        for module in ("rfd3.engine", "rfd3na.engine", "rf3.inference_engines.rf3", "mpnn.inference_engines.mpnn"):
            assert f"import {module}" in definition
        assert "pip check" in definition


@pytest.mark.parametrize("name", ["boltz2", "chai1"])
def test_prediction_cache_outside_workspace_is_bound_and_created_lazily(monkeypatch, tmp_path, name):
    from blocks import Blocks
    from types import SimpleNamespace
    cache = tmp_path / "independent-model-cache" / name
    settings = get_settings()
    monkeypatch.setattr(settings, "CHAPERONE_CONTAINER_BACKEND", "apptainer")
    block = Blocks[name](cache_dir=str(cache))
    assert not cache.exists(), "Constructing a block must not create model caches"
    assert f"{cache}:{cache}" in block.command_template
    assert f"{'BOLTZ_CACHE_DIR' if name == 'boltz2' else 'CHAI_DOWNLOADS_DIR'}={cache}" in block.command_template
    handler = SimpleNamespace(get_prediction_inputs=lambda **kwargs: iter(()))
    block.create_command(handler, tmp_path / "sandbox" / name, 0)
    assert cache.is_dir()


def test_chai_unconfigured_cache_has_a_real_default(monkeypatch):
    from blocks.networks.prediction.chai1 import Chai1Block
    monkeypatch.setattr(get_settings(), "CHAI_DOWNLOADS_DIR", None)
    block = Chai1Block()
    assert block.cache_dir == str((Path.home() / ".chai").resolve())
    assert "CHAI_DOWNLOADS_DIR=None" not in block.command_template


def test_boltz_and_chai_recipes_match_working_blackwell_versions():
    from blocks.networks.prediction.boltz2 import BOLTZ2_CONTAINER
    from blocks.networks.prediction.chai1 import CHAI1_CONTAINER
    for image, torch_version, model in (
        (BOLTZ2_CONTAINER, "2.11.0", "boltz[cuda]==2.2.1"),
        (CHAI1_CONTAINER, "2.7.1", "chai_lab==0.6.1"),
    ):
        definition = image.to_dockerfile()
        assert f"torch=={torch_version} --index-url https://download.pytorch.org/whl/cu128" in definition
        assert model in definition
        assert "numpy==1.26.4" in definition
        if image is BOLTZ2_CONTAINER:
            assert "pip check" in definition
        else:
            assert "only the documented Torch 2.7.1/cu128 override is allowed" in definition
        assert "torch.version.cuda" in definition
        assert "cu126" not in definition


@pytest.mark.parametrize("extra_conflict", [False, True])
def test_chai_dependency_gate_allows_only_documented_torch_override(monkeypatch, extra_conflict):
    from types import SimpleNamespace
    from blocks.networks.prediction.chai1 import _CHAI_DEPENDENCY_CHECK
    output = "chai-lab 0.6.1 has requirement torch<2.7,>=2.3.1, but you have torch 2.7.1+cu128.\n"
    if extra_conflict:
        output += "other-package 1.0 has requirement numpy<1.0, but you have numpy 1.26.4.\n"
    def fake_check(command, **kwargs):
        assert command[-3:] == ["-m", "pip", "check"]
        return SimpleNamespace(returncode=1, stdout=output, stderr="")
    monkeypatch.setattr(subprocess, "run", fake_check)
    if extra_conflict:
        with pytest.raises(RuntimeError, match="Unexpected Chai dependency conflict"):
            exec(_CHAI_DEPENDENCY_CHECK, {})
    else:
        exec(_CHAI_DEPENDENCY_CHECK, {})


def test_chai_build_commands_keep_python_scripts_on_one_docker_line():
    from blocks.networks.prediction.chai1 import CHAI1_CONTAINER
    for commands in CHAI1_CONTAINER._commands:
        for command in commands:
            assert "\n" not in command
            subprocess.run(["/bin/sh", "-n"], input=command, text=True, check=True)


@pytest.mark.parametrize("name", ["boltz2", "chai1", "e1", "protenix"])
def test_new_cache_mount_survives_engine_prepare_and_worker_manifest(monkeypatch, tmp_path, name):
    import cloudpickle
    from blocks import Blocks
    from chaperone import ExecutionEngine, PipelineDataHandler

    cache = tmp_path / "external-model-cache" / name
    settings = get_settings()
    monkeypatch.setattr(settings, "CHAPERONE_CONTAINER_BACKEND", "apptainer")
    monkeypatch.setattr(settings, "CHAPERONE_CONTAINER_WORKSPACE", tmp_path / "workspace")
    monkeypatch.setattr(settings, "RUNDIR", None)
    monkeypatch.delenv("HARBOR_HOST_WORKSPACE", raising=False)
    source = Path(__file__).parents[1] / "examples/benchmarks/ppi/files/4zxb_cropped.pdb"
    if name == "e1":
        from biotite.structure.io.pdb import PDBFile
        from chaperone.io import parse_to_atom_array
        atoms = parse_to_atom_array(str(source))
        atoms.chain_id[:] = "A"
        source = tmp_path / "reference-a.pdb"
        pdb = PDBFile()
        pdb.set_structure(atoms)
        pdb.write(str(source))
    handler = PipelineDataHandler.from_paths([str(source)], name="reference")
    block = Blocks[name](cache_dir=str(cache), store="none")
    engine = ExecutionEngine(rundir=tmp_path / "run")
    prepared = engine._prepare(block, run_args={"mode": "local"}, data=handler)
    assert not cache.exists()
    worker = cloudpickle.loads(cloudpickle.dumps(prepared.blocks[0]))
    monkeypatch.setattr(worker, "_resolve_command", lambda command: command)
    manifest = worker.create_manifest(handler, engine.rundir / "chunk_0", 0, write=False)
    assert cache.is_dir()
    assert f"{cache}:{cache}" in manifest.command


def test_e1_recipe_pins_working_source_and_dependencies():
    from blocks.networks.plm.e1 import E1_CONTAINER
    definition = E1_CONTAINER.to_dockerfile()
    assert "219c7c7a3a803b8552ae6a44ce6dd24f133fefbd" in definition
    assert "torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128" in definition
    assert "transformers==4.56.2" in definition
    assert "flash-attn==2.8.3" in definition
    assert "MAX_JOBS=4" in definition
    assert "from E1.predictor import E1Predictor" in definition
    assert "pip check" in definition
    for commands in E1_CONTAINER._commands:
        for command in commands:
            subprocess.run(["/bin/sh", "-n"], input=command, text=True, check=True)


@pytest.mark.parametrize("name", ["af3", "esmfold2", "simplefold", "disco"])
def test_external_model_caches_survive_worker_serialization(monkeypatch, tmp_path, name):
    import cloudpickle
    from blocks import Blocks
    from chaperone import ExecutionEngine, PipelineDataHandler

    settings = get_settings()
    monkeypatch.setattr(settings, "CHAPERONE_CONTAINER_BACKEND", "apptainer")
    monkeypatch.setattr(settings, "CHAPERONE_CONTAINER_WORKSPACE", tmp_path / "workspace")
    monkeypatch.delenv("HARBOR_HOST_WORKSPACE", raising=False)
    weights = tmp_path / "outside-workspace" / "model-weights"
    cache = tmp_path / "outside-workspace" / "writable-cache"
    weights.mkdir(parents=True)
    source = Path(__file__).parents[1] / "examples/benchmarks/ppi/files/4zxb_cropped.pdb"
    handler = PipelineDataHandler.from_paths([str(source)], name="reference")
    kwargs = {"store": "none"}
    if name == "af3":
        monkeypatch.setattr(settings, "ALPHAFOLD3_WEIGHTS_DIR", str(weights))
        monkeypatch.setattr(settings, "AF3_JAX_CACHE_DIR", str(cache))
    elif name == "esmfold2":
        monkeypatch.setattr(settings, "ESM_WEIGHTS_DIR", str(weights))
        kwargs["model"] = str(weights)
    elif name == "simplefold":
        kwargs["weights_dir"] = str(cache)
    else:
        config = tmp_path / "input.json"
        config.write_text("[]")
        checkpoint = weights / "disco.pt"
        checkpoint.touch()
        kwargs.update(inputs=str(config), cache_dir=str(cache), ckpt_path=str(checkpoint))
    block = Blocks[name](**kwargs)
    engine = ExecutionEngine(rundir=tmp_path / "run")
    prepared = engine._prepare(block, run_args={"mode": "local"}, data=handler)
    assert not cache.exists()
    worker = cloudpickle.loads(cloudpickle.dumps(prepared.blocks[0]))
    for setting in ("ALPHAFOLD3_WEIGHTS_DIR", "AF3_JAX_CACHE_DIR", "ESM_WEIGHTS_DIR"):
        monkeypatch.setattr(settings, setting, str(tmp_path / "wrong-worker-default"))
    monkeypatch.setattr(worker, "_resolve_command", lambda command: command)
    command = worker.create_manifest(handler, engine.rundir / "chunk_0", 0, write=False).command
    if name == "af3":
        assert f"{weights}:{weights}:ro" in command
        assert f"--model_dir={weights}" in command
        assert f"--jax_compilation_cache_dir={cache}" in command
    elif name == "esmfold2":
        assert f"{weights}:{weights}" in command
        assert f"HF_HUB_CACHE={weights}" in command
    elif name == "disco":
        assert f"{checkpoint}:{checkpoint}" in command
    if name != "esmfold2":
        assert f"{cache}:{cache}" in command
        assert cache.is_dir()


def test_protenix_recipe_keeps_declared_dependency_stack_and_checked_arch_patch(tmp_path):
    import ast
    import sys
    from blocks.networks.prediction.protenix import PROTENIX_CONTAINER
    definition = PROTENIX_CONTAINER.to_dockerfile()
    assert "torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1" in definition
    assert "--index-url https://download.pytorch.org/whl/cu128" in definition
    assert "pip install protenix==2.0.0 && pip check" in definition
    assert "import runner.batch_inference" in definition
    assert "force-reinstall" not in definition
    commands = [command for group in PROTENIX_CONTAINER._commands for command in group]
    for command in commands:
        assert "\n" not in command
        subprocess.run(["/bin/sh", "-n"], input=command, text=True, check=True)
    patch = next(command for command in commands if "architecture patch no longer matches" in command)
    code = shlex.split(patch)[-1]
    target = tmp_path / "torch_ext_compile.py"
    target.write_text('_wanted = [("100", "100"),]\n')
    code = code.replace(
        "/opt/protenix-env/lib/python3.12/site-packages/protenix/model/layer_norm/torch_ext_compile.py",
        str(target),
    )
    subprocess.run([sys.executable, "-c", code], check=True)
    architectures = ast.literal_eval(ast.parse(target.read_text()).body[0].value)
    assert architectures == [("100", "100"), ("120", "120")]
    # A source with no matching anchor must fail instead of silently omitting sm_120.
    target.write_text('_wanted = []\n')
    failed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert failed.returncode != 0 and "patch no longer matches" in failed.stderr
