"""Two bounded Slurm array chunks, each chaining a 40mer RFD3 design and MPNN.

    python tests/slurm_batch_smoke.py --test --rundir=/new/empty/slurm-smoke

Requires configured Foundry image/checkpoints and the site's Slurm GPU mapping.
SMOKE_GPU_CONSTRAINT optionally supplies a node constraint. Each chunk requests
24 GB RAM, 2 CPUs, one small-class GPU and 15 minutes; concurrency is capped at 2.
The driver can run inside an interactive allocation: it deliberately submits
engine-managed inner jobs instead of consuming the driver's GPU. Do not monitor
those inner jobs separately. No scientific design-quality claim is made.

--check-config prints the resolved bounded configuration without submitting.
The script supplies --test, --rfd3.n_batches=2, --rfd3.batch_size=1 and
--foundry_mpnn.num_seq_per_pdb=1 as CLI defaults; this explicitly overrides the
usual --test single-batch default. Changed sampling limits are rejected.
"""

from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

import numpy as np

from blocks import Blocks
from chaperone import ExecutionEngine, get_runtime_overrides, init_cli
from chaperone.engine.resolve import (
    _resolve_block,
    _resolve_job_args,
    _resolve_run_args,
)
from chaperone.execution_tracking import load_execution_records


def resolve_plan(argv):
    rfd3, mpnn = Blocks.getall("rfd3", "foundry_mpnn")
    init_cli(
        [
            argv[0],
            "--test",
            "--rfd3.n_batches=2",
            "--rfd3.batch_size=1",
            "--foundry_mpnn.num_seq_per_pdb=1",
            *argv[1:],
        ],
        blocks=[rfd3, mpnn],
    )
    blocks = _resolve_block(
        [
            rfd3(
                inputs={"smoke40": {"length": "40"}},
                n_batches=2,
                batch_size=1,
                seed_offset=1729,
            ),
            mpnn(model_type="protein_mpnn", num_seq_per_pdb=1, seed=1730),
        ],
        {},
        None,
    )
    assert (blocks[0].n_batches, blocks[0].batch_size, blocks[1].num_seq_per_pdb) == (
        2,
        1,
        1,
    ), "Smoke requires exactly two single-design chunks"
    assert blocks[0].inputs == {"smoke40": {"length": "40"}}
    assert blocks[1].output_mode == "sequences"
    runtime = _resolve_run_args(
        dict(mode="slurm", use_local_gpus=False, max_retries=0), block_name="rfd3"
    )
    jobs = dict(
        mem_gb=24,
        cpus_per_task=2,
        time_limit="00:15:00",
        rows_per_job=1,
        num_chunks=2,
        slurm_array_parallelism=2,
    )
    if constraint := os.environ.get("SMOKE_GPU_CONSTRAINT"):
        jobs["constraint"] = constraint
    resources = _resolve_job_args(jobs, blocks[0], runtime)
    assert runtime.mode == "slurm" and not runtime.use_local_gpus, (
        "Do not pass --local or enable local GPU routing"
    )
    assert not any(
        (
            runtime.recompile,
            runtime.force_exec,
            runtime.skip_cache,
            runtime.skip_chunk_cache,
        )
    ), "Cache overrides would invalidate the resume test"
    assert resources.wants_gpu, (
        "Configure a Slurm GPU partition and small-class GPU mapping"
    )
    return blocks, runtime, resources


def validate_outputs(handler, rundir):
    from chaperone import PipelineDataHandler

    assert len(handler) == 2
    backbones = PipelineDataHandler(rundir / "rfd3/output.parquet")
    assert len(backbones) == 2
    parent_ids = set()
    for design in backbones.get_data(load=True):
        assert (
            design.atom_array is not None and np.isfinite(design.atom_array.coord).all()
        )
        assert sum(map(len, design.sequence.values())) == 40
        assert len(set(zip(design.atom_array.chain_id, design.atom_array.res_id))) == 40
        parent_ids.add(design.design_id)
    assert len(parent_ids) == 2
    observed_parents = set()
    for design in handler.get_prediction_inputs(load_context=True):
        assert sum(map(len, design.sequences.values())) == 40
        assert all(
            set(seq) <= set("ACDEFGHIKLMNPQRSTVWY") for seq in design.sequences.values()
        )
        assert design.context is not None and np.isfinite(design.atom_array.coord).all()
        assert design.parent_design_id in parent_ids
        observed_parents.add(design.parent_design_id)
    assert observed_parents == parent_ids


def main(argv=None):
    argv = list(sys.argv if argv is None else argv)
    config_only = "--check-config" in argv
    argv = [arg for arg in argv if arg != "--check-config"]
    blocks, runtime, resources = resolve_plan(argv)
    if config_only:
        print(
            json.dumps(
                {
                    "n_batches": blocks[0].n_batches,
                    "batch_size": blocks[0].batch_size,
                    "sequence_count": blocks[1].num_seq_per_pdb,
                    "runtime": asdict(runtime),
                    "resources": asdict(resources),
                },
                sort_keys=True,
            )
        )
        return
    rundir = get_runtime_overrides().get("rundir")
    if not rundir:
        raise ValueError("Supply --rundir=/new/empty/slurm-smoke")
    root = Path(rundir).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Choose a new, empty output directory: {root}")
    engine = ExecutionEngine(rundir=root)
    result = engine.run(blocks, run_args=runtime, job_args=resources)
    validate_outputs(result, engine.rundir)
    records = {
        block.name: load_execution_records(engine.rundir / block.name)
        for block in blocks
    }
    job_ids = set()
    for step_records in records.values():
        assert len(step_records) == 2
        assert {record.chunk_id for record in step_records} == {0, 1}
        for record in step_records:
            assert record.status == "succeeded" and record.backend == "slurm"
            assert record.job_id and record.job_id != os.environ.get("SLURM_JOB_ID")
            assert record.gpu_seconds > 0 and record.allocated_gpus == 1
            job_ids.add(record.job_id)
    assert len(job_ids) == 2, f"Expected two chained Slurm array tasks: {job_ids}"
    before = asdict(engine.manifest.execution_totals())
    parquets = [engine.rundir / block.name / "output.parquet" for block in blocks]
    mtimes = [path.stat().st_mtime_ns for path in parquets]
    resumed = ExecutionEngine(rundir=engine.rundir)
    cached = resumed.run(blocks, run_args=runtime, job_args=resources)
    validate_outputs(cached, resumed.rundir)
    assert cached.unpack(all=True, to_pandas=False).equals(
        result.unpack(all=True, to_pandas=False)
    )
    assert [path.stat().st_mtime_ns for path in parquets] == mtimes
    assert asdict(resumed.manifest.execution_totals()) == before
    for name, original in records.items():
        assert load_execution_records(resumed.rundir / name) == original
    print(
        f"SLURM BATCH SMOKE PASS: two 40mers, two sequences, jobs {sorted(job_ids)}, {before['gpu_seconds']:.1f} GPU seconds; disk resume submitted no additional work",
        flush=True,
    )


if __name__ == "__main__":
    main()
