#!/usr/bin/env -S bash -c '"${CHAPERONE_REPO:-.}/.venv/bin/python" "$0" "$@"'
"""Redesign one antibody CDR loop with regular RFD3 and fixed Fv context."""

from pathlib import Path

from blocks import Blocks
from chaperone import ExecutionEngine, init_cli
from chaperone.data.schema import SELECTIONS

SCRIPT_DIR = Path(__file__).resolve().parent
FILES_DIR = SCRIPT_DIR / "files"


def pipeline():
    return [
        Blocks["rfd3"](
            inputs=str(FILES_DIR / "antibody.json"), n_batches=1, batch_size=1
        ),
        Blocks["mpnn"](
            model_type="protein_mpnn",
            output_mode="structures",
            num_seq_per_pdb=1,
            keep_motif_sequences_fixed=True,
            args={"chains_to_design": "A", "number_of_packs_per_design": 1},
        ),
        Blocks["af3"](name="af3_complex", diffusion_batch_size=1),
        Blocks["rmsd"](
            alignment_methods=[SELECTIONS.BKBN],
            rmsd_targets=[SELECTIONS.BKBN],
            select_mask_by_name_pairing=[SELECTIONS.BKBN],
        ),
    ]


def main():
    init_cli()
    engine = ExecutionEngine(rundir=SCRIPT_DIR / "rundir")
    data = engine.run(pipeline())
    data.unpack().to_csv(engine.rundir / "results.csv", index=False)
    print(engine)


if __name__ == "__main__":
    main()
