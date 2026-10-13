"""Release checks that do not require Docker, Apptainer, model weights or a GPU."""
from pathlib import Path
import subprocess

import pytest

from chaperone import get_settings
from chaperone.containers import Image
from chaperone_cli.install import _build_apptainer, _build_docker


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


def test_classic_mpnn_advertises_private_dependency():
    from blocks.networks.mpnn import MODELHUB_PORTABLE_CONTAINER
    assert "private" in MODELHUB_PORTABLE_CONTAINER._labels["BuildAccess"]


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


def test_portable_variants_can_be_installed_even_when_site_selects_legacy(monkeypatch):
    import blocks.networks.mpnn as mpnn
    from blocks import get_container, get_all_containers
    monkeypatch.setattr(mpnn, "MODELHUB_CONTAINER", mpnn._MODELHUB_SIF_CONTAINER)
    assert get_container("mpnn") is mpnn._MODELHUB_SIF_CONTAINER
    assert get_container("modelhub_portable") is mpnn.MODELHUB_PORTABLE_CONTAINER
    assert get_all_containers()["modelhub_portable"] is mpnn.MODELHUB_PORTABLE_CONTAINER


def test_private_source_failure_is_actionable(monkeypatch, tmp_path):
    from blocks.networks.mpnn import MODELHUB_PORTABLE_CONTAINER
    def unavailable(command, **kwargs):
        raise subprocess.CalledProcessError(128, command)
    monkeypatch.setattr(subprocess, "run", unavailable)
    with pytest.raises(RuntimeError, match="Requires authorized access to private"):
        MODELHUB_PORTABLE_CONTAINER.prepare_git_clones(tmp_path)


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
