"""GPU warmup of the pinned public RFD3 compiler through the normal engine."""

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--rundir", type=Path, required=True)
    parser.add_argument(
        "--inputs", type=Path, help="Representative public RFD3 input JSON"
    )
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--local", action="store_true", help="Use an already allocated local GPU"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.length < 1 or args.batch_size < 1:
        parser.error("length and batch-size must be positive")
    if args.inputs and not args.inputs.is_file():
        parser.error("inputs must be an existing RFD3 JSON file")
    cache = args.cache_dir.expanduser().resolve()
    rundir = args.rundir.expanduser().resolve()
    if args.dry_run:
        print(
            f"RFD3 compiled GPU warmup: cache={cache}; output={rundir}; "
            f"batch_size={args.batch_size}; inputs={args.inputs or f'length {args.length}'}"
        )
        return 0

    from chaperone import ExecutionEngine, JobArgs, get_settings, init_cli
    from chaperone.engine.resolve import RuntimeArgs
    from blocks.networks.foundry.rfdiffusion3 import RFD3Block

    init_cli(argv=["chaperone-warmup-rfd3"])
    cache.mkdir(parents=True, exist_ok=True)
    get_settings().RFD3_COMPILE_CACHE_DIR = str(cache)
    block = RFD3Block(
        inputs=str(args.inputs.resolve())
        if args.inputs
        else {"warmup": {"length": str(args.length)}},
        compile_model=True,
        batch_size=args.batch_size,
        n_batches=2,
        seed_offset=0,
        # Exercise repeated compiled forwards; these outputs are not quality samples.
        inference_sampler={"num_timesteps": 3},
    )
    engine = ExecutionEngine(rundir=str(rundir))
    result = engine.run(
        block,
        run_args=RuntimeArgs(
            mode="local" if args.local else "slurm",
            force_exec=True,
            max_retries=0,
            job_args=JobArgs(cpus_per_task=4, mem_gb=32, time_limit="00:45:00"),
        ),
    )
    if len(result.unpack()) == 0:
        raise RuntimeError("RFD3 warmup produced no structures")
    print(
        f"Warmup completed. Reuse RFD3_COMPILE_CACHE_DIR={cache} and compile_model=True."
    )
    print(
        "Other GPU types, software versions, batch sizes or shapes can compile again."
    )
    return 0
