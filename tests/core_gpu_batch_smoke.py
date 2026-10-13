#!/usr/bin/env python
"""Bounded multi-input check of the core design and prediction stack.

    CHAPERONE_MPNN_CONTAINER=portable python tests/core_gpu_batch_smoke.py \
        --test --local --rundir=/absolute/new/run

Two independent 40-residue RFD3 batches produce two backbones; classic MPNN
produces two sequences each. RF3, AF3 and ESMFold2 independently predict that
same four-sequence cohort with one sample and normal sampling schedules.
This checks execution and data integrity, not scientific accuracy or production
throughput. RF3 early stopping is disabled here to require a completed prediction
for every input; its normal recycle and diffusion schedules are unchanged. This
execution diagnostic does not promote designs based on confidence.
Select CHAPERONE_MPNN_CONTAINER=public for the public LigandMPNN image or
portable for the authorized fused-MPNN image; select it explicitly for a build check.

Use --predictors=rf3,af3 (or none) to select stages. Reuse the same run directory
to add predictors later: generation and sequences use the engine's normal cache.
Without --local, engine-managed Slurm chunks request one large-class GPU,
48 GB RAM, four CPUs and 30 minutes, with at most two simultaneous tasks.
Local mode uses one worker and a single multi-input chunk per stage, exercising
multiple inputs in one model process. SMOKE_GPU_CONSTRAINT optionally restricts
Slurm nodes; SMOKE_ESMFOLD2_SNAPSHOT selects an already downloaded snapshot.
--check-config resolves and validates the plan without executing any model.
Every successful invocation reopens the engine and checks a complete disk-cache
replay without new execution records or changed output-table timestamps.
"""

from collections import Counter
from dataclasses import asdict
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from blocks import Blocks
from chaperone import ExecutionEngine, PipelineDataHandler, get_runtime_overrides, init_cli
from chaperone.engine.resolve import _resolve_block, _resolve_job_args, _resolve_run_args
from chaperone.execution_tracking import load_execution_records


PREDICTORS = ("rf3", "af3", "esmfold2")
INPUTS = {"smoke40": {"length": "40"}}


def resolve_plan(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--predictors", default=",".join(PREDICTORS))
    parser.add_argument("--check-config", action="store_true")
    options, remaining = parser.parse_known_args(argv[1:])
    selected = [] if options.predictors == "none" else options.predictors.split(",")
    if len(selected) != len(set(selected)) or not set(selected) <= set(PREDICTORS):
        raise ValueError("--predictors must be a unique subset of rf3,af3,esmfold2 or none")
    classes = {name: Blocks[name] for name in ("rfd3", "mpnn", *selected)}
    init_cli([
        argv[0], "--test", "--rfd3.n_batches=2", "--rfd3.batch_size=1",
        "--mpnn.num_seq_per_pdb=2", *remaining,
    ], blocks=list(classes.values()))
    generation = _resolve_block([
        classes["rfd3"](inputs=INPUTS, n_batches=2, batch_size=1, seed_offset=1741),
        classes["mpnn"](model_type="protein_mpnn", num_seq_per_pdb=2),
    ], {}, None)
    predictors = []
    for name in selected:
        kwargs = {"diffusion_batch_size": 1}
        if name == "rf3":
            kwargs["early_stopping_plddt_threshold"] = 0
        if name == "esmfold2":
            kwargs["use_local_msas"] = False
            if snapshot := os.environ.get("SMOKE_ESMFOLD2_SNAPSHOT"):
                kwargs["model"] = snapshot
        predictors.extend(_resolve_block(classes[name](**kwargs), {}, None))
    rfd3, mpnn = generation
    assert rfd3.inputs == INPUTS and (rfd3.n_batches, rfd3.batch_size) == (2, 1)
    assert mpnn.num_seq_per_pdb == 2 and mpnn.output_mode == "sequences"
    assert mpnn.model_type == "protein_mpnn"
    defaults = classes["rfd3"](inputs=INPUTS)
    assert rfd3.inference_sampler == defaults.inference_sampler
    assert not rfd3.convert_config_to_test_config
    assert not rfd3.json_keys_subset
    for block in predictors:
        default = classes[block._name]()
        assert block.diffusion_batch_size == 1 and block.read_most_confident_model
        for field in ("num_diffusion_steps", "n_recycles", "num_loops"):
            if hasattr(block, field):
                assert getattr(block, field) == getattr(default, field), f"Keep normal {block.name}.{field}"
        if block._name == "rf3":
            assert block.early_stopping_plddt_threshold == 0
        if hasattr(block, "model_seeds"):
            assert len(block.model_seeds or []) == 1
        assert not block.compile_as_metric
    runtime = _resolve_run_args(dict(mode="slurm", use_local_gpus=False, max_retries=0), block_name="rfd3")
    runtime.max_workers = 1 if runtime.mode == "local" else 2
    assert runtime.mode == "local" or not runtime.use_local_gpus, "Use --local explicitly for local execution"
    assert not any((runtime.recompile, runtime.force_exec, runtime.skip_cache, runtime.skip_chunk_cache))
    jobs = dict(mem_gb=48, cpus_per_task=4, time_limit="00:30:00", gpu_class="large",
                num_chunks=1 if runtime.mode == "local" else 2, slurm_array_parallelism=2)
    if constraint := os.environ.get("SMOKE_GPU_CONSTRAINT"):
        jobs["constraint"] = constraint
    resources = _resolve_job_args(jobs, rfd3, runtime)
    # Bound even explicit CLI resource overrides: this is a smoke check.
    assert resources.mem_gb <= 48 and resources.cpus_per_task <= 4
    assert resources.num_chunks == jobs["num_chunks"] and resources.slurm_array_parallelism <= 2
    assert resources.time_limit == "00:30:00"
    return generation, predictors, runtime, resources, options.check_config


def _validate_atoms(atoms):
    assert atoms is not None and len(atoms) > 0 and np.isfinite(atoms.coord).all()
    assert float(np.ptp(atoms.coord, axis=0).max()) < 1000
    assert len(set(zip(atoms.chain_id, atoms.res_id))) == 40


def validate_cohort(backbones, sequences):
    assert len(backbones) == 2 and len(sequences) == 4
    parent_paths = {}
    for design in backbones.get_data(load=True):
        _validate_atoms(design.atom_array)
        parent_paths[design.design_id] = str(Path(design.path).resolve())
    assert len(parent_paths) == 2
    cohort = {}
    parents = Counter()
    for design in sequences.get_prediction_inputs(load_context=True):
        assert len(design.sequences) == 1
        sequence = next(iter(design.sequences.values()))
        assert len(sequence) == 40 and set(sequence) <= set("ACDEFGHIKLMNPQRSTVWY")
        assert design.parent_design_id in parent_paths
        assert str(Path(design.path).resolve()) == parent_paths[design.parent_design_id]
        _validate_atoms(design.atom_array)
        assert design.design_id not in cohort
        cohort[design.design_id] = (sequence, design.parent_design_id)
        parents[design.parent_design_id] += 1
    assert set(parents) == set(parent_paths) and set(parents.values()) == {2}
    return cohort, parent_paths


def validate_prediction(handler, name, cohort, parent_paths):
    assert len(handler) == 4
    seen = set()
    inputs = list(handler.get_prediction_inputs(load_context=False))
    outputs = list(handler.get_data(which="last", load=True))
    ancestors = list(handler.get_data(which="first", load=False))
    for design, output, ancestor in zip(inputs, outputs, ancestors):
        # A predictor may advance the same logical design (ESMFold2), or
        # create a named child whose parent is the sequence design (AF3/RF3).
        sequence_id = design.design_id if design.design_id in cohort else design.parent_design_id
        assert sequence_id in cohort and sequence_id not in seen
        expected_sequence, parent = cohort[sequence_id]
        if design.design_id == sequence_id:
            assert design.parent_design_id == parent
        else:
            assert design.parent_design_id == sequence_id
        assert str(Path(ancestor.path).resolve()) == parent_paths[parent]
        assert next(iter(design.sequences.values())) == expected_sequence
        assert output.name == name
        _validate_atoms(output.atom_array)
        assert len(output.sequence) == 1 and next(iter(output.sequence.values())) == expected_sequence
        for metric in ("plddt", "ptm"):
            value = output.metrics[name].get(metric)
            assert value is not None and np.isfinite(float(value)), f"Missing {name}.{metric}"
        seen.add(sequence_id)
    assert seen == set(cohort)


def run_plan(engine, generation, predictors, runtime, resources):
    sequences = engine.run(generation, run_args=runtime, job_args=resources)
    backbones = PipelineDataHandler(engine.rundir / "rfd3/output.parquet")
    cohort, parent_paths = validate_cohort(backbones, sequences)
    results = {"rfd3": backbones, "mpnn": sequences}
    for block in predictors:
        result = engine.run(block, data=sequences, run_args=runtime, job_args=resources)
        validate_prediction(result, block.name, cohort, parent_paths)
        results[block.name] = result
    return results


def main(argv=None):
    generation, predictors, runtime, resources, config_only = resolve_plan(list(sys.argv if argv is None else argv))
    if config_only:
        print(json.dumps({"backbones": 2, "sequences": 4, "blocks": {
            block.name: block.model_dump() for block in [*generation, *predictors]},
            "runtime": asdict(runtime), "resources": asdict(resources)}, default=str, sort_keys=True))
        return
    assert os.environ.get("CHAPERONE_MPNN_CONTAINER") in {"public", "portable"}, "Select the freshly built public or portable MPNN image explicitly"
    rundir = get_runtime_overrides().get("rundir")
    if not rundir:
        raise ValueError("Supply --rundir=/absolute/path/to/core-batch-smoke")
    engine = ExecutionEngine(rundir=Path(rundir).expanduser().resolve())
    results = run_plan(engine, generation, predictors, runtime, resources)
    records = {name: load_execution_records(engine.rundir / name) for name in results}
    for name, attempts in records.items():
        assert len(attempts) == resources.num_chunks, (name, attempts)
        for record in attempts:
            assert record.status == "succeeded" and record.gpu_seconds > 0
            assert record.allocated_gpus == 1
            expected_backend = "slurm" if runtime.mode == "slurm" or os.environ.get("SLURM_JOB_ID") else "gpu_local"
            assert record.backend == expected_backend
            if runtime.mode == "slurm":
                assert record.job_id and record.job_id != os.environ.get("SLURM_JOB_ID")
    parquets = [engine.rundir / name / "output.parquet" for name in results]
    mtimes = [path.stat().st_mtime_ns for path in parquets]
    totals = asdict(engine.manifest.execution_totals())
    resumed = ExecutionEngine(rundir=engine.rundir)
    replay = run_plan(resumed, generation, predictors, runtime, resources)
    assert [path.stat().st_mtime_ns for path in parquets] == mtimes
    assert asdict(resumed.manifest.execution_totals()) == totals
    for name, handler in results.items():
        assert load_execution_records(engine.rundir / name) == records[name]
        assert handler.unpack(all=True, to_pandas=False).equals(replay[name].unpack(all=True, to_pandas=False))
    print(f"CORE GPU BATCH PASS: 2 backbones, 4 sequences, predictors {[b.name for b in predictors]}; "
          f"{totals['gpu_seconds']:.1f} GPU seconds; disk replay added no work", flush=True)


if __name__ == "__main__":
    main()
