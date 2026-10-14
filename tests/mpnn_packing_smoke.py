#!/usr/bin/env python
"""Run LigandMPNN packing on two small existing backbones.

    python tests/mpnn_packing_smoke.py \
        --inputs=/path/to/core-run/rfd3/output.parquet --test --local \
        --rundir=/path/to/new/mpnn-packing-run

Each input must have one protein chain of at most 48 residues. The helper
renames it X1 and fixes its first two residues, requests two sequences and two
packs per sequence (eight outputs total), and verifies fixed identities,
restored chain IDs, finite coordinates, side chains, lineage and cache replay.
Normal model sampling defaults are retained. This is an execution check,
not an accuracy benchmark.

Omit --local to submit one engine-managed Slurm GPU job using the site's GPU
mapping (or gpu:1 on the default partition). Requests are bounded to 16 GB RAM,
two CPUs, one chunk and 15 minutes. --check-config resolves without submission.
"""

from collections import Counter
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
from atomworks.io.utils.io_utils import to_cif_file
from biotite.structure import get_residue_starts

from blocks.networks.mpnn import ClassicMPNNBlock, LIGANDMPNN_CONTAINER
from chaperone import (
    ExecutionEngine,
    PipelineDataHandler,
    get_runtime_overrides,
    init_cli,
)
from chaperone.data import make_row
from chaperone.engine.resolve import (
    _resolve_block,
    _resolve_job_args,
    _resolve_run_args,
)
from chaperone.execution_tracking import load_execution_records
from chaperone.io import ensure_bonds
from chaperone.orchestration.job_args import site_gres, site_partition


def resolve_plan(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--check-config", action="store_true")
    options, remaining = parser.parse_known_args(argv[1:])
    init_cli(
        [argv[0], "--test", "--mpnn.num_seq_per_pdb=2", *remaining],
        blocks=[ClassicMPNNBlock],
    )
    (block,) = _resolve_block(
        ClassicMPNNBlock(
            model_type="ligand_mpnn",
            output_mode="structures",
            num_seq_per_pdb=2,
            args={"number_of_packs_per_design": 2},
            store="none",
        ),
        {},
        None,
    )
    assert block._get_image() is LIGANDMPNN_CONTAINER
    assert block.output_mode == "structures" and block.num_seq_per_pdb == 2
    assert (
        block.args.number_of_packs_per_design == 2 and block.args.pack_side_chains == 1
    )
    assert block.keep_motif_sequences_fixed
    default = ClassicMPNNBlock()
    for field in (
        "temperature",
        "sc_num_denoising_steps",
        "sc_num_samples",
        "pack_with_ligand_context",
        "repack_everything",
    ):
        assert getattr(block.args, field) == getattr(default.args, field)
    runtime = _resolve_run_args(
        dict(mode="slurm", max_retries=0, use_local_gpus=False), block_name="mpnn"
    )
    assert (
        runtime.mode == "local" or not runtime.use_local_gpus
    ), "Use --local explicitly inside a GPU allocation"
    runtime.max_workers = 1
    assert not any(
        (
            runtime.recompile,
            runtime.force_exec,
            runtime.skip_cache,
            runtime.skip_chunk_cache,
        )
    )
    resources = _resolve_job_args(
        # MPNN defaults to CPU, so explicitly opt this GPU smoke in.
        # A shared gpu_class override alone intentionally leaves CPU blocks alone.
        dict(
            partition=site_partition(gpu=True),
            gres=site_gres("large"),
            mem_gb=16,
            cpus_per_task=2,
            num_chunks=1,
            slurm_array_parallelism=1,
            time_limit="00:15:00",
        ),
        block,
        runtime,
    )
    assert resources.mem_gb <= 16 and resources.cpus_per_task <= 2
    assert resources.num_chunks == resources.slurm_array_parallelism == 1
    assert resources.time_limit == "00:15:00"
    return options, block, runtime, resources


def prepare_inputs(source, directory):
    designs = list(PipelineDataHandler(source).get_data(which="last", load=True))
    assert len(designs) == 2, "Supply exactly two small backbone rows"
    directory.mkdir(parents=True, exist_ok=True)
    rows, expected = [], {}
    for index, design in enumerate(designs):
        atoms = design.atom_array.copy()
        assert len(set(atoms.chain_id)) == 1 and not atoms.hetero.any()
        starts = get_residue_starts(atoms)
        assert 3 <= len(starts) <= 48
        assert np.isfinite(atoms.coord).all()
        atoms.chain_id[:] = "X1"
        parent = f"packing_input_{index}"
        fixed = {int(atoms.res_id[i]): str(atoms.res_name[i]) for i in starts[:2]}
        path = directory / f"{parent}.cif"
        to_cif_file(ensure_bonds(atoms), str(path))
        rows.append(
            make_row(
                "input",
                str(path),
                design_id=parent,
                metadata={"motif": [f"X1{residue}" for residue in fixed]},
            )
        )
        expected[parent] = (len(starts), fixed)
    return PipelineDataHandler(rows), expected


def validate_outputs(handler, expected):
    assert len(handler) == 8, "Expected 2 inputs × 2 sequences × 2 packs"
    counts = Counter()
    for design in handler.get_data(which="last", load=True):
        parent = design.parent_design_id
        assert parent in expected, (parent, list(expected))
        residues, fixed = expected[parent]
        atoms = design.atom_array
        assert set(atoms.chain_id) == {"X1"}
        assert (
            np.isfinite(atoms.coord).all()
            and float(np.ptp(atoms.coord, axis=0).max()) < 1000
        )
        assert len(get_residue_starts(atoms)) == residues
        assert np.isin(
            atoms.atom_name, ["CB", "CG", "CG1", "OG", "OG1", "SG"]
        ).any(), "No packed side-chain atoms"
        for residue, name in fixed.items():
            assert set(atoms.res_name[atoms.res_id == residue]) == {name}
        counts[parent] += 1
    assert counts == Counter({parent: 4 for parent in expected})


def main(argv=None):
    options, block, runtime, jobs = resolve_plan(
        list(sys.argv if argv is None else argv)
    )
    if options.check_config:
        print(
            json.dumps(
                {
                    "block": block.model_dump(),
                    "runtime": asdict(runtime),
                    "resources": asdict(jobs),
                    "expected_outputs": 8,
                },
                default=str,
            )
        )
        return
    rundir = get_runtime_overrides().get("rundir")
    if not rundir:
        raise ValueError("Supply a new --rundir for the packing check")
    engine = ExecutionEngine(rundir=Path(rundir).expanduser().resolve())
    inputs, expected = prepare_inputs(options.inputs, engine.rundir / "smoke_inputs")
    output = engine.run(block, data=inputs, run_args=runtime, job_args=jobs)
    validate_outputs(output, expected)
    records = load_execution_records(engine.rundir / block.name)
    assert (
        len(records) == 1
        and records[0].status == "succeeded"
        and records[0].gpu_seconds > 0
    )
    assert records[0].allocated_gpus == 1
    if runtime.mode == "slurm":
        assert records[0].backend == "slurm" and records[0].job_id
    parquet = engine.rundir / block.name / "output.parquet"
    mtime = parquet.stat().st_mtime_ns
    replay = ExecutionEngine(rundir=engine.rundir).run(
        block, data=inputs, run_args=runtime, job_args=jobs
    )
    validate_outputs(replay, expected)
    assert parquet.stat().st_mtime_ns == mtime
    assert load_execution_records(engine.rundir / block.name) == records
    print(
        f"MPNN PACKING PASS: 8 outputs, fixed residues and X1 chains preserved; {records[0].gpu_seconds:.1f} GPU seconds; cache replay added no work"
    )


if __name__ == "__main__":
    main()
