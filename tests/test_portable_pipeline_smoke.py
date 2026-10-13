"""Opt-in GPU smoke must resolve valid resources without submitting jobs."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("mapping", [False, True])
def test_public_packing_smoke_respects_site_and_explicit_mode(tmp_path, local, mapping):
    env = {
        key: os.environ[key] for key in ("PATH", "HOME", "TMPDIR") if key in os.environ
    }
    env.update(
        CHAPERONE_SITE="portable",
        CHAPERONE_MPNN_CONTAINER="public",
        PYTHONPATH=str(ROOT / "src"),
    )
    if mapping:
        env.update(
            CHAPERONE_GPU_PARTITION="hpc-mid",
            CHAPERONE_GRES_MAP='{"large":"gpu:h200:1"}',
        )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tests/public_mpnn_packing_smoke.py"),
            "--inputs=unused.parquet",
            "--check-config",
            *(["--local"] if local else []),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    plan = json.loads(result.stdout.strip().splitlines()[-1])
    assert plan["expected_outputs"] == 8
    assert plan["runtime"]["mode"] == ("local" if local else "slurm")
    assert not plan["runtime"]["use_local_gpus"]
    resources = plan["resources"]
    assert resources["gres"] == ("gpu:h200:1" if mapping else "gpu:1")
    assert resources["partition"] == ("hpc-mid" if mapping else None)
    assert resources["constraint"] is None
    assert resources["num_chunks"] == resources["slurm_array_parallelism"] == 1
    assert (
        resources["mem_gb"],
        resources["cpus_per_task"],
        resources["time_limit"],
    ) == (16, 2, "00:15:00")


def test_cpu_smoke_rejects_unknown_backend_before_work(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tests/cpu_functional_smoke.py"),
            "--mode=cluster",
            "--rundir",
            str(tmp_path / "run"),
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode != 0
    assert "invalid choice" in result.stderr
    assert not (tmp_path / "run").exists()
