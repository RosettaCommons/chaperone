#!/usr/bin/env -S bash -c '"${CHAPERONE_REPO:-.}/.venv/bin/python" "$0" "$@"'
"""Compare three public LigandMPNN model families on identical input structures."""

from pathlib import Path

from blocks import Blocks
from blocks.util_blocks.filter import MetricFilter
from chaperone import ExecutionEngine, PipelineDataHandler, init_cli
from chaperone.data.schema import SELECTIONS

FILES = Path(__file__).resolve().parent / "files"
MODELS = ("protein_mpnn", "ligand_mpnn", "soluble_mpnn")


def sequence_blocks():
    return [
        Blocks["mpnn"](
            name=f"mpnn_{model}",
            model_type=model,
            output_mode="sequences",
            num_seq_per_pdb=4,
        )
        for model in MODELS
    ]


def refolding_blocks():
    return [
        Blocks["af3"](diffusion_batch_size=1),
        Blocks["rmsd"](
            alignment_methods=[SELECTIONS.BKBN],
            rmsd_targets=[SELECTIONS.BKBN, SELECTIONS.CA],
        ),
        Blocks["filter"](
            name="af3_filter",
            filters={
                "af3.ranking_score": MetricFilter(min=0.5),
                "rmsd.bkbn_align.bkbn_rmsd": MetricFilter(max=3.0),
            },
        ),
    ]


def main():
    from absl import flags

    flags.DEFINE_bool(
        "refold", False, "Run AF3 and RMSD; requires separately obtained AF3 weights."
    )
    init_cli()
    engine = ExecutionEngine(rundir=Path(__file__).resolve().parent / "rundir")
    inputs = PipelineDataHandler.from_dir(FILES, name="input")
    outputs = [engine.run(block, data=inputs) for block in sequence_blocks()]
    result = outputs[0]
    for output in outputs[1:]:
        result = result + output
    # This checks preservation of structure context and native sequence rows.
    repacked = PipelineDataHandler(result.unpack())
    if len(repacked) != len(result):
        raise RuntimeError("Unpack/repack lost rows")
    if flags.FLAGS.refold:
        result = engine.run(refolding_blocks(), data=result)
    result.unpack().to_csv(engine.rundir / "results.csv", index=False)
    print(engine)


if __name__ == "__main__":
    main()
