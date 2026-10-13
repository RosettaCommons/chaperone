"""Prepare the real nested GPU probe without executing an image or a job."""

from pathlib import Path
import runpy
import subprocess


def test_prepare_only_stages_sanitized_source_and_real_worker(monkeypatch, tmp_path):
    helper = runpy.run_path(str(Path(__file__).with_name("nested_gpu_smoke.py")), run_name="nested_gpu_helper")
    image = tmp_path / "agent.sif"
    image.touch()  # Placeholder images are never executed in preparation mode.
    containers = tmp_path / "containers"
    containers.mkdir()
    (containers / "alphafold3.sif").touch()
    weights = tmp_path / "weights"
    weights.mkdir()
    root = tmp_path / "probe"
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)

    def no_processes(*args, **kwargs):
        raise AssertionError("Preparation must not start a container or submit work")

    monkeypatch.setattr(subprocess, "run", no_processes)
    helper["main"]([
        "--prepare-only", "--root", str(root), "--agent-image", str(image),
        "--container-dir", str(containers), "--af3-weights", str(weights),
    ])
    workspace = root / "workspace"
    assert (workspace / "chaperone/src/blocks/scripts/run_alphafold_AF.py").is_file()
    assert not (workspace / "chaperone/src/chaperone-evals").exists()
    assert (workspace / "inputs/reference.pdb").is_file()
    cache = workspace / ".cache/af3_jax"
    assert cache.is_dir()
    wrapper = (root / ".runtime/slurm-python").read_text()
    assert "exec /opt/chaperone/.venv/bin/python" in wrapper
    assert f"{containers}:{containers}:ro" in wrapper
    assert f"{weights}:{weights}:ro" in wrapper
    assert f"export AF3_JAX_CACHE_DIR={cache}" in wrapper
    assert str(workspace / "chaperone/src") in wrapper
    driver = (workspace / "nested_af3.py").read_text()
    compile(driver, "nested_af3.py", "exec")
    assert '"mode": "local"' in driver
    assert 'os.environ["HARBOR_HOST_WORKSPACE"]' in driver
