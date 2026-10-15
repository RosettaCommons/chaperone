"""CPU portability and GPU affinity at command generation in the worker."""
import pytest

from chaperone import get_settings
from chaperone.containers import Image
from chaperone.block_config.compiled import CompiledBlockFunction, ScriptSpec
from chaperone.orchestration.job_args import JobArgs


@pytest.fixture
def image(monkeypatch):
    monkeypatch.setattr(get_settings(), "CHAPERONE_CONTAINER_BACKEND", "docker")
    monkeypatch.setattr("chaperone.common.is_macos", lambda: False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    return Image("gpu_test").base_image("ubuntu:24.04")


def test_cpu_container_does_not_require_nvidia(image, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    command = image.build_command("true")
    assert "--gpus" not in command
    assert "NVIDIA_VISIBLE_DEVICES=void" in command
    assert "CUDA_VISIBLE_DEVICES=" in command


@pytest.mark.parametrize("mask", ["", "-1", "none", "void"])
def test_disabled_allocation_never_expands_to_all(image, monkeypatch, mask):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    assert "--gpus" not in image.build_command("true", gpu=True)


@pytest.mark.parametrize("mask,inner", [("3", "0"), ("2,5", "0,1"), ("GPU-abc", "0")])
def test_docker_restricts_and_renumbers_devices(image, monkeypatch, mask, inner):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    command = image.build_command("true", gpu=True)
    assert command[command.index("--gpus") + 1] == f'"device={mask}"'
    assert f"CUDA_VISIBLE_DEVICES={inner}" in command


def test_apptainer_preserves_mask_despite_cleanenv(image, monkeypatch):
    monkeypatch.setattr(get_settings(), "CHAPERONE_CONTAINER_BACKEND", "apptainer")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,5")
    command = image.build_command("true", gpu=True)
    assert "--nv" in command
    assert "CUDA_VISIBLE_DEVICES=2,5" in command
    command = image.build_command("true", gpu=False)
    assert "--nv" not in command
    assert "CUDA_VISIBLE_DEVICES=" in command


def test_generated_templates_use_worker_affinity_and_keep_user_templates(image, monkeypatch, tmp_path):
    class GPUBlock(CompiledBlockFunction):
        _name = "gpu_test"
        _resources = JobArgs(gpu_class="small")
        _script_spec = ScriptSpec(command=["true"], image=image)

        def create_command(self, handler, sandbox_dir, chunk_id):
            return self.command_template

        def compile(self, sandbox_dir):
            raise NotImplementedError

    block = GPUBlock()
    block._set_container_bind_settings({})
    monkeypatch.setattr(block, "_resolve_command", lambda command: command)
    for mask in ("3", "5"):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
        command = block.create_manifest(None, tmp_path, 0, write=False).command
        assert command[command.index("--gpus") + 1] == f'"device={mask}"'
    block.command_template = ["custom-launcher"]
    assert block.create_manifest(None, tmp_path, 0, write=False).command == ["custom-launcher"]


def test_explicit_job_gpu_request_overrides_cpu_block_default(image, monkeypatch, tmp_path):
    class CPUDefault(CompiledBlockFunction):
        _name = "cpu_default"
        _resources = JobArgs(partition="cpu")
        _script_spec = ScriptSpec(command=["true"], image=image)

        def create_command(self, handler, sandbox_dir, chunk_id):
            return self.command_template

        def compile(self, sandbox_dir):
            raise NotImplementedError

    block = CPUDefault()
    block._set_container_bind_settings({})
    block._container_job_args = JobArgs(gres="gpu:1")
    monkeypatch.setattr(block, "_resolve_command", lambda command: command)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    command = block.create_manifest(None, tmp_path, 0, write=False).command
    assert command[command.index("--gpus") + 1] == '"device=4"'


def test_mpnn_explicit_cpu_mode_overrides_gpu_job(image):
    from blocks.networks.mpnn import ClassicMPNNBlock

    block = ClassicMPNNBlock(device="cpu")
    block._container_job_args = JobArgs(gres="gpu:1")
    assert not block._wants_container_gpu()
