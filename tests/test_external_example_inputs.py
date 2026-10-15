"""Withdrawn fixtures must not cause an example to submit work without inputs."""

from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("example", [
    "enzyme-protease", "enzyme-glycosidase", "nanoparticle-design",
    "benchmarks/multi-mpnn",
])
def test_external_inputs_required_before_engine_start(example, tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "examples" / example / "pipeline.py"),
         "--test", f"--rundir={tmp_path / 'run'}"],
        cwd=tmp_path, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode != 0
    assert "input_dir" in result.stdout + result.stderr
    assert not (tmp_path / "run").exists()


def test_params_use_explicit_user_directory(tmp_path):
    from blocks.utils.params import list_available_ligands, resolve_param_file

    assert list_available_ligands(str(tmp_path)) == []
    with pytest.raises(KeyError, match="CHAPERONE_LIGAND_PARAMS_DIR"):
        resolve_param_file("TEST", str(tmp_path))
    supplied = tmp_path / "supplied"
    supplied.mkdir()
    params = supplied / "TEST.params"
    params.write_text("NAME TEST\n")
    assert list_available_ligands(str(supplied)) == ["TEST"]
    assert resolve_param_file("TEST", str(supplied)) == str(params)
