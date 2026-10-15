"""Resolve the opt-in core GPU smoke plan without running models or Slurm."""

import json
import os
from pathlib import Path
from importlib.util import find_spec
import subprocess
import sys
import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def resolve(tmp_path, *arguments, success=True):
    env = {key: os.environ[key] for key in ("PATH", "HOME", "TMPDIR") if key in os.environ}
    env.update(PYTHONPATH=str(Path(find_spec("chaperone").origin).parent.parent), MPLCONFIGDIR=str(tmp_path / ".mpl"),
               CHAPERONE_GPU_PARTITION="portable-gpu",
               CHAPERONE_GRES_MAP='{"large": "gpu:b4000:1"}')
    proc = subprocess.run([sys.executable, str(ROOT / "tests/core_gpu_batch_smoke.py"),
                           "--test", "--check-config", *arguments], cwd=tmp_path,
                          env=env, text=True, capture_output=True, timeout=90)
    if not success:
        assert proc.returncode != 0, proc.stdout + proc.stderr
        assert "AssertionError" in proc.stderr or "ValueError" in proc.stderr, proc.stderr
        return
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("local", [False, True])
def test_core_smoke_keeps_bounded_cohort_and_normal_schedules(tmp_path, local):
    plan = resolve(tmp_path, *(["--local"] if local else []))
    assert (plan["backbones"], plan["sequences"]) == (2, 4)
    blocks = plan["blocks"]
    assert (blocks["rfd3"]["n_batches"], blocks["rfd3"]["batch_size"]) == (2, 1)
    assert blocks["rfd3"]["inputs"] == {"smoke40": {"length": "40"}}
    assert blocks["rfd3"]["inference_sampler"]["num_timesteps"] == 200
    assert not blocks["rfd3"]["convert_config_to_test_config"]
    assert blocks["mpnn"]["num_seq_per_pdb"] == 2
    assert blocks["mpnn"]["output_mode"] == "sequences"
    assert blocks["rf3"]["n_recycles"] == 10
    assert blocks["rf3"]["num_diffusion_steps"] == 50
    assert blocks["rf3"]["early_stopping_plddt_threshold"] == 0
    assert blocks["af3"]["n_recycles"] is None
    assert blocks["esmfold2"]["num_diffusion_steps"] == 200
    assert blocks["esmfold2"]["num_loops"] == 3
    for name in ("rf3", "af3", "esmfold2"):
        assert blocks[name]["diffusion_batch_size"] == 1
    assert plan["runtime"]["mode"] == ("local" if local else "slurm")
    assert plan["runtime"]["max_workers"] == (1 if local else 2)
    assert plan["runtime"]["max_retries"] == 0
    assert plan["resources"]["num_chunks"] == (1 if local else 2)
    assert plan["resources"]["slurm_array_parallelism"] == 2
    assert plan["resources"]["gres"] == "gpu:b4000:1"
    assert plan["resources"]["mem_gb"] == 48


def test_core_smoke_allows_generation_only_for_incremental_resume(tmp_path):
    plan = resolve(tmp_path, "--predictors=none")
    assert set(plan["blocks"]) == {"rfd3", "mpnn"}


def test_core_smoke_explicit_small_gpu_request_keeps_slurm_caps(tmp_path):
    plan = resolve(tmp_path, "--gres=gpu:small:1", "--nouse_local_gpus")
    assert plan["resources"]["gres"] == "gpu:small:1"
    assert plan["resources"]["num_chunks"] == 2
    assert plan["resources"]["slurm_array_parallelism"] == 2
    assert plan["runtime"]["max_workers"] == 2
    assert not plan["runtime"]["use_local_gpus"]


@pytest.mark.parametrize("argument", [
    "--rfd3.n_batches=3", "--mpnn.num_seq_per_pdb=3",
    "--esmfold2.num_loops=1", "--rf3.num_diffusion_steps=2",
    "--force_exec", "--predictors=rf3,rf3",
])
def test_core_smoke_rejects_expansion_shortened_science_or_cache_bypass(tmp_path, argument):
    resolve(tmp_path, argument, success=False)


@pytest.mark.parametrize("child_ids", [False, True])
@pytest.mark.parametrize("corruption", [None, "coordinates", "sequence", "parent", "ancestor", "confidence"])
def test_prediction_integrity_checks_reject_mislinked_or_bad_outputs(tmp_path, corruption, child_ids):
    spec = importlib.util.spec_from_file_location("core_gpu_batch_smoke", ROOT / "tests/core_gpu_batch_smoke.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cohort = {f"design{i}": ("A" * 40, f"parent{i // 2}") for i in range(4)}
    parent_paths = {f"parent{i}": str(tmp_path / f"parent{i}.cif") for i in range(2)}
    inputs, outputs, ancestors = [], [], []
    for design_id, (sequence, parent) in cohort.items():
        inputs.append(SimpleNamespace(design_id=f"{design_id}_model" if child_ids else design_id,
                                      parent_design_id=design_id if child_ids else parent, sequences={"A": sequence}))
        atoms = SimpleNamespace(coord=np.zeros((40, 3)), chain_id=["A"] * 40, res_id=list(range(40)))
        # _validate_atoms only needs this tiny structural protocol.
        class Atoms(SimpleNamespace):
            def __len__(self):
                return len(self.coord)
        outputs.append(SimpleNamespace(name="rf3", atom_array=Atoms(**vars(atoms)), sequence={"A": sequence},
                                       metrics={"rf3": {"plddt": 0.5, "ptm": 0.4}}))
        ancestors.append(SimpleNamespace(path=parent_paths[parent]))
    if corruption == "coordinates":
        outputs[0].atom_array.coord[0, 0] = np.nan
    elif corruption == "sequence":
        outputs[0].sequence = {"A": "G" * 40}
    elif corruption == "parent":
        inputs[0].parent_design_id = "parent1"
    elif corruption == "ancestor":
        ancestors[0].path = parent_paths["parent1"]
    elif corruption == "confidence":
        outputs[0].metrics["rf3"]["ptm"] = None

    class Handler:
        def __len__(self):
            return 4

        def get_prediction_inputs(self, **kwargs):
            return iter(inputs)

        def get_data(self, *, which, **kwargs):
            return iter(outputs if which == "last" else ancestors)

    if corruption is None:
        module.validate_prediction(Handler(), "rf3", cohort, parent_paths)
    else:
        with pytest.raises(AssertionError):
            module.validate_prediction(Handler(), "rf3", cohort, parent_paths)
