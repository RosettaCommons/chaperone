#!/usr/bin/env -S bash -c '"${CHAPERONE_REPO:-.}/.venv/bin/python" "$0" "$@"'
"""Compare three public LigandMPNN model families on identical input structures."""

from pathlib import Path

from blocks import Blocks
from blocks.util_blocks.filter import MetricFilter
from chaperone import ExecutionEngine, PipelineDataHandler, init_cli
from chaperone.data.schema import SELECTIONS

FILES = Path(__file__).resolve().parent / "files"
MODELS = ("protein_mpnn", "ligand_mpnn", "soluble_mpnn")
# The fixture labels its biotin pentylamide BTN; CCD BTN is unmodified biotin.
LIGAND_SMILES = "N1C(=O)N[C@@H]2[C@@H](SC[C@H]12)CCCCC(=O)NCCCCC"


def sequence_blocks():
    return [
        Blocks["mpnn"](
            name=f"mpnn_{model}",
            model_type=model,
            output_mode="structures",
            args={"number_of_packs_per_design": 1, "file_ending": f"_{model}"},
            num_seq_per_pdb=4,
        )
        for model in MODELS
    ]


def refolding_blocks(model_name: str = ""):
    suffix = f"_{model_name}" if model_name else ""
    af3_name, rmsd_name = f"af3{suffix}", f"rmsd{suffix}"
    return [
        Blocks["af3"](
            name=af3_name,
            diffusion_batch_size=1,
            input_mode="json",
            specified_smiles={"BTN": LIGAND_SMILES},
        ),
        Blocks["rmsd"](
            name=rmsd_name,
            alignment_methods=[SELECTIONS.BKBN],
            rmsd_targets=[SELECTIONS.BKBN, SELECTIONS.CA],
        ),
        Blocks["filter"](
            name=f"af3_filter{suffix}",
            filters={
                f"{af3_name}.ranking_score": MetricFilter(min=0.5),
                f"{rmsd_name}.bkbn_align.bkbn_rmsd": MetricFilter(max=3.0),
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
    outputs = []
    for block in sequence_blocks():
        output = engine.run(block, data=inputs)
        if flags.FLAGS.refold:
            # Packed filenames may match across model families. Keep AF3 input
            # directories separate so one branch cannot overwrite another.
            output = engine.run(refolding_blocks(block.model_type), data=output)
        outputs.append(output)
    result = outputs[0]
    for output in outputs[1:]:
        result = result + output
    # This checks preservation of structure context and packed structure rows.
    repacked = PipelineDataHandler(result.unpack())
    if repacked.unpack()["design_id"].nunique() != len(repacked):
        raise RuntimeError("Model branches produced duplicate design IDs")
    if len(repacked) != len(result):
        raise RuntimeError("Unpack/repack lost rows")
    result.unpack().to_csv(engine.rundir / "results.csv", index=False)
    print(engine)


if __name__ == "__main__":
    main()
