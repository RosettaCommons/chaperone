"""CLI smoke runs must stay usable without checkout-only test fixtures."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_slurm_batch_smoke_resolves_two_bounded_gpu_chunks(tmp_path):
    completed = run_python('''
import os, runpy, sys
os.environ["CHAPERONE_GPU_PARTITION"] = "portable-gpu"
os.environ["CHAPERONE_GRES_MAP"] = '{"small": "gpu:b4000:1"}'
os.environ["SMOKE_GPU_CONSTRAINT"] = "Blackwell"
script = sys.argv[1]
sys.argv = [script, "--test", "--check-config"]
runpy.run_path(script, run_name="__main__")
''', tmp_path, ROOT / "tests/slurm_batch_smoke.py")
    plan = json.loads(completed.stdout.strip().splitlines()[-1])
    assert (plan["n_batches"], plan["batch_size"], plan["sequence_count"]) == (2, 1, 1)
    assert plan["runtime"]["mode"] == "slurm"
    assert plan["runtime"]["use_local_gpus"] is False
    assert plan["runtime"]["max_retries"] == 0
    resources = plan["resources"]
    assert resources["partition"] == "portable-gpu"
    assert resources["gres"] == "gpu:b4000:1"
    assert resources["constraint"] == "Blackwell"
    assert (resources["num_chunks"], resources["slurm_array_parallelism"]) == (2, 2)
    assert (resources["mem_gb"], resources["cpus_per_task"], resources["time_limit"]) == (24, 2, "00:15:00")


@pytest.mark.parametrize("invalid_flag", ["--rfd3.n_batches=3", "--local"])
def test_slurm_batch_smoke_rejects_changed_scope(tmp_path, invalid_flag):
    run_python('''
import runpy, sys
script, argument = sys.argv[1:]
sys.argv = [script, "--test", "--check-config", argument]
try:
    runpy.run_path(script, run_name="__main__")
except AssertionError:
    pass
else:
    raise AssertionError("Unsafe scope override was accepted")
''', tmp_path, ROOT / "tests/slurm_batch_smoke.py", invalid_flag)


def run_python(source, cwd, *arguments):
    env = {key: os.environ[key] for key in ("PATH", "HOME", "TMPDIR") if key in os.environ}
    env.update({"PYTHONPATH": str(ROOT / "src"), "MPLCONFIGDIR": str(cwd / ".mpl")})
    completed = subprocess.run(
        [sys.executable, "-c", source, *map(str, arguments)],
        cwd=cwd, env=env, text=True, capture_output=True, timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed


def test_packaged_smoke_mode_and_override_precedence(tmp_path):
    config = tmp_path / "custom.yaml"
    config.write_text("rfd3:\n  batch_size: 2\n  n_batches: 2\n")
    run_python('''
import sys
from blocks import Blocks
from chaperone import init_cli, get_runtime_overrides
from chaperone.engine.resolve import _resolve_block
from chaperone.smoke_defaults import smoke_defaults
init_cli(["test", "--test", "--config=" + sys.argv[1], "--rfd3.n_batches=3"])
rfd = _resolve_block(Blocks["rfd3"](inputs={"one": {"length": "80"}}, n_batches=99, batch_size=99), {}, None)[0]
mpnn = _resolve_block(Blocks["foundry_mpnn"](num_seq_per_pdb=99), {}, None)[0]
assert (rfd.n_batches, rfd.batch_size, mpnn.num_seq_per_pdb) == (3, 2, 1)
assert get_runtime_overrides()["auto_store_format"] == "none"
assert get_runtime_overrides()["auto_store_cleanup"] is False
assert get_runtime_overrides()["cleanup_logs"] is False
assert smoke_defaults()["rfd3"]["n_batches"] == 1
''', tmp_path, config)


def test_test_mode_without_config_and_production_defaults(tmp_path):
    source = '''
import sys
from blocks import Blocks
from chaperone import init_cli
from chaperone.engine.resolve import _resolve_block
init_cli(["test", *sys.argv[1:]])
block = _resolve_block(Blocks["foundry_mpnn"](num_seq_per_pdb=7), {}, None)[0]
assert block.num_seq_per_pdb == (1 if "--test" in sys.argv else 7)
'''
    run_python(source, tmp_path, "--test")
    run_python(source, tmp_path)


def test_smoke_parameters_are_registered_fields():
    from blocks import Blocks
    from chaperone.smoke_defaults import smoke_defaults

    for name, fields in smoke_defaults().items():
        assert set(fields) <= set(Blocks[name].model_fields), name


EXAMPLES = sorted(
    path.relative_to(ROOT / "examples").as_posix()
    for path in (ROOT / "examples").rglob("pipeline*.py")
    if not any(part.startswith("rundir") or part in {"runs", "__pycache__"}
               for part in path.relative_to(ROOT / "examples").parts)
)


@pytest.mark.parametrize("example", EXAMPLES)
@pytest.mark.parametrize("test_mode", [True, False], ids=["smoke", "production"])
def test_example_smoke_configuration_and_exports(tmp_path, example, test_mode):
    # Exercise every entrypoint in both modes: test-only subsets can hide missing
    # production inputs. Resolve actual blocks and stage inputs, without GPU models.
    # A private copy also catches paths accidentally anchored outside --rundir.
    source = ROOT / "examples" / example
    copied = tmp_path / "example"
    shutil.copytree(source.parent, copied, ignore=shutil.ignore_patterns("rundir*", "runs", "__pycache__"))
    input_files = {
        path.relative_to(copied): path.read_bytes()
        for path in (copied / "files").rglob("*") if path.is_file()
    }
    result = tmp_path / "resolved.json"
    run_python('''
import json, runpy, sys
from pathlib import Path
import chaperone
from chaperone.data import PipelineDataHandler, make_row
from chaperone.engine.resolve import _resolve_block
seen, exports = [], []
class Handler:
    names = ["example"]
    def __add__(self, other): return self
    def rewind_to(self, step): return self
    def __len__(self): return 1
    def unpack(self, **kwargs):
        return PipelineDataHandler([make_row(
            name="af3", path="/test/design.cif", design_id="one",
            metadata={"motif": ["A1"]},
            metrics={"af3_ipsae": {"ipsae": 0.5}, "af3": {"iptm": 0.7, "chain_pair_pae_min": [[1.0, 2.0], [2.0, 1.0]]}},
        )]).unpack(**kwargs)
    def store(self, which, path, *, cleanup=False): exports.append(str(path))
    def materialize(self, path, **kwargs): exports.append(str(path))
def run(self, blocks, data=None, **kwargs):
    self.rundir.mkdir(parents=True, exist_ok=True)
    for raw in blocks if isinstance(blocks, list) else [blocks]:
        block = _resolve_block(raw, {}, None)[0]
        record = {"type": block._name, "name": block.name, "config": block.model_dump()}
        staging = self.rundir / "staging" / str(len(seen))
        if hasattr(block, "_load_inputs_to_dict"):
            specs = block._load_inputs_to_dict()
            record["input_keys"] = list(specs)
            # Run the actual staging code, including JSON-relative structure paths.
            # Loading only the JSON does not prove its input files are present.
            block.create_command(block.init_handler(), staging, 0)
        elif block._name in {"disco", "rfd1"}:
            if block._name == "disco":
                block.cache_dir = str(staging / "cache")
            block.create_command(block.init_handler(), staging, 0)
            if block._name == "disco":
                jobs = json.loads((staging / "input_config.json").read_text())
                for job in jobs:
                    for entry in job["sequences"]:
                        ligand = entry.get("ligand", {}).get("ligand", "")
                        if ligand.startswith("FILE_"):
                            assert Path(ligand[5:]).is_file(), ligand
        for field in ("reference", "input_pdb", "motif_pdb", "clash_ref_pdb", "json_template"):
            path = getattr(block, field, None)
            if path:
                assert Path(path).is_file(), (block.name, field, path)
        seen.append(record)
    return Handler()
chaperone.ExecutionEngine.run = run
script, output, mode = sys.argv[1:]
sys.argv = [script, *(["--test"] if mode == "smoke" else []), "--rundir=" + str(Path(output).parent / "custom-run")]
runpy.run_path(script, run_name="__main__")
Path(output).write_text(json.dumps({"blocks": seen, "exports": exports}, default=str))
''', tmp_path, copied / source.name, result, "smoke" if test_mode else "production")
    assert {
        path.relative_to(copied): path.read_bytes()
        for path in (copied / "files").rglob("*") if path.is_file()
    } == input_files, "Pipeline modified its bundled inputs instead of --rundir"
    report = json.loads(result.read_text())
    blocks = report["blocks"]
    assert blocks, "Example did not construct any pipeline blocks"
    for block in blocks:
        if not test_mode:
            continue
        values = block["config"]
        if block["type"] in {"rfd3", "rfd3na"}:
            assert values["batch_size"] == values["n_batches"] == 1
            assert len(block["input_keys"]) == (2 if "enzyme-" in example else 1)
        if block["type"] in {"mpnn", "foundry_mpnn"}:
            assert values["num_seq_per_pdb"] == 1
            if block["type"] == "mpnn" and values["output_mode"] == "structures":
                assert values["args"]["number_of_packs_per_design"] == 1
        if block["type"] in {"af3", "af3_with_target_template", "rf3"}:
            assert values["diffusion_batch_size"] == 1
            assert len(values["model_seeds"]) == 1
        if block["type"] == "rfd1":
            assert values["num_designs"] == 1
        if block["type"] == "rpxdock":
            assert values["nout_top"] == 1
        if block["type"] == "disco":
            assert values["num_seeds"] == values["n_batches"] == 1
    if "ame" in example:
        assert sum(block["type"] == "rfd3" for block in blocks) == (1 if "full" in example else 2)
    if "fgfr2" in example:
        import polars as pl
        scores = pl.read_csv(tmp_path / "custom-run/scores.csv")
        assert scores.height == 1
        assert "af3.iptm" in scores.columns
        assert "af3.chain_pair_pae_min" not in scores.columns
        assert "af3.metadata.motif" not in scores.columns
    if "catcif" in example:
        assert report["exports"] == [str(tmp_path / "custom-run" / x) for x in ("packaged", "materialized")]
