"""AF3 must predict sequence artifacts and honor seeds in native mmCIF mode."""

import ast
import hashlib
import json
from pathlib import Path
import pathlib

import numpy as np
import pytest
from biotite.structure.io.pdb import PDBFile

from blocks.networks.prediction.alphafold3 import AlphaFold3Block, SCRIPT_DIR
from chaperone import ExecutionEngine, PipelineDataHandler
from chaperone.data import make_sequence_row
from chaperone.io import parse_to_atom_array
from chaperone.engine.resolve import RuntimeArgs


@pytest.fixture
def backbone(tmp_path):
    source = (
        Path(__file__).resolve().parents[2]
        / "examples/benchmarks/ppi/files/4zxb_cropped.pdb"
    )
    atoms = parse_to_atom_array(str(source))
    atoms = atoms[atoms.chain_id == atoms.chain_id[0]]
    residues = list(dict.fromkeys(atoms.res_id))[:3]
    atoms = atoms[np.isin(atoms.res_id, residues)]
    atoms.chain_id[:] = "A"
    target = tmp_path / "backbone.pdb"
    pdb = PDBFile()
    pdb.set_structure(atoms)
    pdb.write(target)
    return PipelineDataHandler.from_paths([str(target)], name="backbone")


@pytest.mark.parametrize("artifact_kind", ["sequence", "structure"])
def test_default_af3_uses_each_mpnn_sequence_not_its_backbone_context(
    tmp_path, backbone, artifact_kind
):
    sequences = PipelineDataHandler(
        [
            make_sequence_row(
                design_id=f"backbone__mpnn_{index}",
                parent_design_id="backbone",
                sequences={"A": sequence},
            )
            | {"artifact_kind": artifact_kind}
            for index, sequence in enumerate(["WWW", "FFF"])
        ]
    )
    designed = PipelineDataHandler.merge(backbone, sequences)
    assert designed.has_sequence_data
    assert designed.has_sequence_artifacts == (artifact_kind == "sequence")
    assert all(
        item.atom_array is not None
        for item in designed.get_prediction_inputs(load_context=True)
    )
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    manifest = AlphaFold3Block(model_seeds=[101, 202])._prepare_af3_inputs(
        designed, inputs
    )
    assert manifest is None, (
        "Sequence artifacts require the sequence-authoritative JSON path"
    )
    files = sorted(inputs.glob("*.json"))
    assert len(files) == 2
    payloads = [json.loads(path.read_text()) for path in files]
    assert [payload["sequences"][0]["protein"]["sequence"] for payload in payloads] == [
        "WWW",
        "FFF",
    ]
    assert all(payload["modelSeeds"] == [101, 202] for payload in payloads)
    assert len({payload["name"] for payload in payloads}) == 2


def test_default_mmcif_manifest_carries_all_model_seeds(tmp_path, backbone):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    manifest = AlphaFold3Block(model_seeds=[101, 202])._prepare_af3_inputs(
        backbone, inputs
    )
    payloads = json.loads(manifest.read_text())
    assert len(payloads) == 1 and "cif" in payloads[0]
    assert payloads[0]["seeds"] == [101, 202]
    assert payloads[0]["seed"] == 101


def test_input_preparation_revision_invalidates_old_outputs_and_staged_inputs(
    tmp_path, backbone
):
    block = AlphaFold3Block()
    assert block.model_dump(mode="json")["input_preparation_version"] == 2
    from chaperone_cli.registry import _extract_field_type

    assert (
        _extract_field_type(type(block).model_fields["input_preparation_version"])
        is None
    )
    config = block.model_dump(mode="json")
    config.pop("input_preparation_version")
    old_payload = [
        {
            "class": f"{block.__class__.__module__}.{block.__class__.__qualname__}",
            "config": config,
        },
        {"engine_input_content": backbone.content_fingerprint()},
    ]
    old_hash = hashlib.sha256(
        json.dumps(
            old_payload, sort_keys=True, separators=(",", ":"), default=str
        ).encode()
    ).hexdigest()
    engine = ExecutionEngine(rundir=tmp_path / "run")
    output = engine.rundir / "af3/output.parquet"
    output.parent.mkdir(parents=True)
    backbone.write_parquet(output)
    stale_input = output.parent / "chunk_0/inputs/old.json"
    stale_input.parent.mkdir(parents=True)
    stale_input.write_text('{"modelSeeds": [1], "wrong_old_sequence": true}')
    engine._write_cache_fingerprint(output, old_hash)
    runtime = RuntimeArgs(mode="local")
    current_hash = engine._cache_fingerprint([block], backbone, runtime)
    assert current_hash != old_hash
    assert engine._check_cache([block], output, backbone, runtime, current_hash) is None
    assert not output.exists() and not stale_input.exists()


@pytest.mark.parametrize(
    "seeds,expected",
    [({"seed": 17}, (17,)), ({"seed": 17, "seeds": [101, 202]}, (101, 202))],
)
def test_actual_manifest_loader_forwards_seeds_without_importing_gpu_stack(
    tmp_path, seeds, expected
):
    # Execute the runner's actual stdlib-only adapter; the heavyweight AF3
    # builder is captured at its call boundary rather than requiring JAX/GPU.
    source = SCRIPT_DIR / "run_alphafold_AF.py"
    tree = ast.parse(source.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "load_fold_inputs_from_manifest"
    )
    calls = []
    namespace = {
        "json": json,
        "pathlib": pathlib,
        "_resolve_path_or_string": lambda value: value,
        "build_fold_input_from_cif": lambda **kwargs: calls.append(kwargs) or kwargs,
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"name": "test", "cif": "data_test\n", **seeds}]))
    results = namespace["load_fold_inputs_from_manifest"](manifest)
    assert len(results) == len(calls) == 1
    assert calls[0]["rng_seeds"] == expected


@pytest.mark.parametrize(
    "completed,skip_existing,should_run",
    [(False, True, True), (True, True, False), (True, False, True)],
)
def test_actual_runner_retries_partial_output_directory(
    tmp_path, completed, skip_existing, should_run
):
    import datetime
    import os
    from types import SimpleNamespace

    tree = ast.parse((SCRIPT_DIR / "run_alphafold_AF.py").read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_check_job_already_completed", "process_fold_input"}
        and node.body
        and not isinstance(node.body[-1], ast.Expr)
    ]
    # The runner also contains overload declarations. Select its implemented
    # function by the presence of a real return statement in the outer body.
    functions = [
        node
        for node in functions
        if node.name == "_check_job_already_completed"
        or any(isinstance(child, ast.Return) for child in node.body)
    ]
    namespace = {"os": os, "datetime": datetime}
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *functions,
        ],
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(module), "run_alphafold_AF.py", "exec"),
        namespace,
    )
    output = tmp_path / "partial"
    output.mkdir()
    (output / "partial_data.json").write_text("{}")
    if completed:
        (output / "ranking_scores.csv").write_text(
            "seed,sample,ranking_score\n1,0,0.8\n"
        )
        (output / "partial_model.cif").write_text("data_test\n")
    calls = []
    namespace.update(
        write_fold_input_json=lambda fold_input, output_dir: calls.append(output_dir),
        predict_structure=lambda **kwargs: ["predicted"],
        write_outputs=lambda **kwargs: None,
    )
    fold_input = SimpleNamespace(
        name="partial", chains=["A"], rng_seeds=[1], sanitised_name=lambda: "partial"
    )
    result = namespace["process_fold_input"](
        fold_input=fold_input,
        data_pipeline_config=None,
        model_runner=SimpleNamespace(model_params={}),
        output_dir=output,
        skip_existing=skip_existing,
    )
    assert bool(calls) is should_run
    assert result == (["predicted"] if should_run else [])


def test_actual_runner_propagates_required_input_inference_failure(tmp_path):
    # The same function is executed by main without an exception handler.
    import os
    from types import SimpleNamespace

    tree = ast.parse((SCRIPT_DIR / "run_alphafold_AF.py").read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_check_job_already_completed", "process_fold_input"}
        and any(isinstance(child, ast.Return) for child in node.body)
    ]
    namespace = {"os": os, "write_fold_input_json": lambda *args: None}

    def fail(**kwargs):
        raise RuntimeError("required input inference failed")

    namespace["predict_structure"] = fail
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *functions,
        ],
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(module), "run_alphafold_AF.py", "exec"),
        namespace,
    )
    with pytest.raises(RuntimeError, match="required input inference failed"):
        namespace["process_fold_input"](
            fold_input=SimpleNamespace(
                name="input",
                chains=["A"],
                rng_seeds=[1],
                sanitised_name=lambda: "input",
            ),
            data_pipeline_config=None,
            model_runner=SimpleNamespace(model_params={}),
            output_dir=tmp_path / "outputs",
        )
