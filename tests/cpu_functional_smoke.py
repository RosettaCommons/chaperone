"""Exercise installed Chaperone on real CPU inputs, without mocks or model containers.

    python tests/cpu_functional_smoke.py --repo . --rundir /new/empty/smoke-dir

Add --mode=slurm to validate a shared installation on compute nodes. Resource
requests use the configured CPU partition, or the cluster default when unset.

Uses 24 residues from a bundled structure, two copies and three short sequences.
Preserves ordinary engine manifests, execution records, structures and tables.
This is an integration check, not a scientific performance benchmark.
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import polars as pl
from biotite.structure.io.pdb import PDBFile

from blocks import Blocks
from chaperone import ExecutionEngine, PipelineDataHandler, get_settings
from chaperone.data.schema import SELECTIONS
from chaperone.execution_tracking import load_execution_records
from chaperone.io import parse_to_atom_array


def assert_same_atoms(expected, actual):
    assert len(actual) == len(expected)
    np.testing.assert_allclose(actual.coord, expected.coord, atol=1e-3)
    for field in ("chain_id", "res_id", "res_name", "atom_name"):
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--rundir", required=True, type=Path)
    parser.add_argument("--mode", choices=("local", "slurm"), default="local")
    args = parser.parse_args()
    root = args.rundir.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    source = args.repo.resolve() / "examples/benchmarks/ppi/files/4zxb_cropped.pdb"
    reference = parse_to_atom_array(str(source))
    chain = reference.chain_id[0]
    reference = reference[reference.chain_id == chain]
    residue_ids = list(dict.fromkeys(reference.res_id))[:24]
    assert len(residue_ids) == 24
    reference = reference[np.isin(reference.res_id, residue_ids)]
    input_path = root / "input_first24.pdb"
    pdb = PDBFile()
    pdb.set_structure(reference)
    pdb.write(input_path)
    inputs = PipelineDataHandler.from_paths([str(input_path)], name="reference")

    # Explicit mode prevents accidental submission; Slurm uses site CPU defaults.
    get_settings().CHAPERONE_COMPUTE_BACKEND = args.mode
    runtime = dict(mode=args.mode, max_workers=1, max_retries=0, use_local_gpus=False)
    resources = dict(cpus_per_task=1, mem_gb=4, time_limit="00:05:00")
    engine = ExecutionEngine(
        rundir=root / "pipeline",
        run_args_default=runtime,
        job_args_default=resources,
    )
    copied = engine.run(Blocks["copy"](n_multiply=2), data=inputs)
    assert len(copied) == 2
    for row in copied.get_data(which="last", load=True):
        assert_same_atoms(reference, row.atom_array)
    output_parquet = engine.rundir / "copy/output.parquet"
    original_stat = output_parquet.stat()
    summary = asdict(engine.manifest.execution_totals())
    assert summary["successful_records"] > 0
    assert summary["failed_records"] == summary["gpu_seconds"] == 0
    records = load_execution_records(engine.rundir / "copy")
    assert records and all(record.backend == args.mode for record in records)
    if args.mode == "slurm":
        assert all(record.job_id and record.allocated_gpus == 0 for record in records)

    # Resume from disk in another engine; a real cache hit preserves artifacts
    # and execution accounting rather than rerunning a fast copy unnoticed.
    resumed = ExecutionEngine(
        rundir=engine.rundir,
        run_args_default=runtime,
        job_args_default=resources,
    )
    assert len(resumed.manifest) == len(engine.manifest)
    cached = resumed.run(Blocks["copy"](n_multiply=2), data=inputs)
    assert len(cached) == 2
    assert output_parquet.stat().st_mtime_ns == original_stat.st_mtime_ns
    assert asdict(resumed.manifest.execution_totals()) == summary
    assert cached.unpack(all=True, to_pandas=False).equals(
        copied.unpack(all=True, to_pandas=False)
    )
    print(
        f"PASS real {args.mode} execution, disk resume and unchanged cache accounting",
        flush=True,
    )

    scored = resumed.run(
        Blocks["rmsd"](
            alignment_methods=[SELECTIONS.BKBN],
            rmsd_targets=[SELECTIONS.BKBN],
            resolve_symmetries=False,
        ),
        data=cached,
    )
    metric = "rmsd.bkbn_align.bkbn_rmsd"
    values = scored.unpack(to_pandas=False)[metric].to_numpy()
    assert len(values) == 2 and np.isfinite(values).all()
    np.testing.assert_allclose(values, 0.0, atol=1e-4)
    passed = resumed.run(
        Blocks["filter"](name="passing_gate", filters={metric: {"max": 1e-4}}),
        data=scored,
    )
    assert len(passed) == 2
    failed = resumed.run(
        Blocks["filter"](name="failing_gate", filters={metric: {"min": 1.0}}),
        data=scored,
    )
    assert len(failed) == 0
    missing = resumed.run(
        Blocks["filter"](name="missing_gate", filters={"missing.value": {"max": 1.0}}),
        data=scored,
    )
    assert len(missing) == 0
    selected = resumed.run(
        Blocks["select_diverse"](
            name="structure_selection",
            max_identity=0.99,
            target_count=1,
            require_target=True,
            required_structure_steps=["copy"],
            rank_by=metric,
        ),
        data=scored,
    )
    assert len(selected) == 1
    print(
        "PASS lineage RMSD, strict passing/failing/missing gates and structural selection",
        flush=True,
    )

    selected.store("copy", root / "archived_copy.catcif")
    tagged = next(selected.get_data(which="last", load=False))
    assert tagged.storage_format == "catcif" and ".catcif:" in tagged.path
    assert_same_atoms(reference, parse_to_atom_array(tagged.path))
    selected.write_parquet(root / "archived.parquet")
    reloaded = PipelineDataHandler(root / "archived.parquet")
    assert_same_atoms(
        reference, next(reloaded.get_data(which="last", load=True)).atom_array
    )
    materialized = reloaded.materialize(root / "materialized")
    real_path = next(materialized.get_data(which="last", load=False)).path
    assert ".catcif:" not in real_path and Path(real_path).is_file()
    assert_same_atoms(reference, parse_to_atom_array(real_path))
    # Exercise the actual external command's archive-input staging too.
    staged = resumed.run(Blocks["copy"](name="archive_copy"), data=reloaded)
    assert len(staged) == 1
    assert_same_atoms(
        reference, next(staged.get_data(which="last", load=True)).atom_array
    )
    print(
        "PASS CatCIF archive/table/materialization roundtrip and command staging",
        flush=True,
    )

    sequence = next(iter(next(inputs.get_data(load=True)).sequence.values()))
    assert len(sequence) == 24
    alternative = "A" * 24 if sequence != "A" * 24 else "G" * 24
    sequences = PipelineDataHandler.from_sequences(
        pl.DataFrame(
            {
                "design_id": ["original", "duplicate", "alternative"],
                "sequence_A": [sequence, sequence, alternative],
            }
        )
    )
    sequences.write_sequences_csv(root / "sequences.csv")
    restored = PipelineDataHandler.from_sequences(root / "sequences.csv")
    expected = [
        item.sequences for item in sequences.get_prediction_inputs(load_context=False)
    ]
    observed = [
        item.sequences for item in restored.get_prediction_inputs(load_context=False)
    ]
    assert observed == expected
    diverse = resumed.run(
        Blocks["select_diverse"](
            name="sequence_selection",
            max_identity=0.99,
            target_count=2,
            require_target=True,
        ),
        data=restored,
    )
    assert len(diverse) == 2
    picked = [
        item.sequences["A"]
        for item in diverse.get_prediction_inputs(load_context=False)
    ]
    assert set(picked) == {sequence, alternative}
    assert diverse.has_sequence_artifacts
    assert all(len(diverse[index][0]) == 0 for index in range(len(diverse)))
    diverse.write_sequences_csv(root / "selected_sequences.csv")
    print(
        "PASS native sequence CSV roundtrip and deterministic duplicate removal",
        flush=True,
    )
    assert resumed.manifest.execution_totals().failed_records == 0
    assert resumed.manifest.execution_totals().gpu_seconds == 0
    print(f"CPU FUNCTIONAL SMOKE PASS: artifacts in {root}", flush=True)


if __name__ == "__main__":
    main()
