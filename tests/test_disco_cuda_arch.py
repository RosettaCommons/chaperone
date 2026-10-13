"""DISCO must not overwrite DeepSpeed's architecture list with commas."""

from types import SimpleNamespace

import pytest

from blocks.networks.design.disco.disco import (
    DISCO_CONTAINER,
    _DISCO_CUDA_ARCH_ASSIGNMENT,
    _DISCO_CUDA_ARCH_PATCH,
    _DISCO_DEEPSPEED_ARCH_CHECK,
)


@pytest.mark.parametrize("capability", [(8, 6), (12, 0)])
@pytest.mark.parametrize("override", [None, "", "8.0;8.9+PTX", "8.0,8.9"])
def test_architecture_assignment_detects_gpu_and_normalizes_explicit_lists(capability, override):
    environ = {} if override is None else {"TORCH_CUDA_ARCH_LIST": override}
    scope = {
        "os": SimpleNamespace(environ=environ),
        "torch": SimpleNamespace(cuda=SimpleNamespace(get_device_capability=lambda device: capability)),
        "self": SimpleNamespace(device="cuda:0"),
    }
    exec(_DISCO_CUDA_ARCH_ASSIGNMENT, scope)
    expected = (override or ".".join(map(str, capability))).replace(",", ";")
    assert environ["TORCH_CUDA_ARCH_LIST"] == expected
    assert "," not in expected


def test_recipe_patches_exact_upstream_assignment_and_checks_real_deepspeed_parser(tmp_path):
    runner = tmp_path / "inference.py"
    runner.write_text('def init_env(self):\n    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0,8.9"\n')
    patch = _DISCO_CUDA_ARCH_PATCH.replace("/opt/disco/runner/inference.py", str(runner))
    exec(patch, {})
    assert _DISCO_CUDA_ARCH_ASSIGNMENT in runner.read_text()
    compile(runner.read_text(), str(runner), "exec")
    with pytest.raises(RuntimeError, match="assignment changed"):
        exec(patch, {})
    recipe = DISCO_CONTAINER.to_singularity()
    assert "DISCO CUDA architecture assignment changed" in recipe
    assert "CUDAOpBuilder" in recipe and "builder.compute_capability_args()" in recipe
    compile(_DISCO_DEEPSPEED_ARCH_CHECK, "deepspeed_architecture_build_gate", "exec")
