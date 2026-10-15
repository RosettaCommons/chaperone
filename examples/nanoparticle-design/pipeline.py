#!/usr/bin/env -S bash -c '"${CHAPERONE_REPO:-.}/.venv/bin/python" "$0" "$@"'
"""C3 scaffold generation, I3 docking, tied sequence design, and trimer refolding."""

from pathlib import Path

from blocks import Blocks
from blocks.util_blocks.filter import MetricFilter
from chaperone import ExecutionEngine, init_cli
from chaperone.data.schema import SELECTIONS

def pipeline_blocks(files: Path):
    return [
        Blocks["rfd1"](
            input_pdb=str(files / "dn5_stump.pdb"),
            contigs="A1-7/150/0 B1-7/150/0 C1-7/150/0",
            symmetry="c3",
            num_designs=1,
        ),
        Blocks["nanoparticle_prepare"](
            motif_pdb=str(files / "dn5_stump.pdb"),
            clash_ref_pdb=str(files / "dn5b_stalk.pdb"),
            max_clashes=8,
        ),
        # RFD1 emits glycine placeholders; score their backbone geometry too.
        Blocks["rpxdock"](nout_top=1, ignored_aas=""),
        Blocks["filter"](
            name="rpxdock_filter", filters={"rpxdock.rpx_score": MetricFilter(min=70.0)}
        ),
        Blocks["mpnn"](
            keep_motif_sequences_fixed=True,
            num_seq_per_pdb=2,
            model_type="protein_mpnn",
            output_mode="structures",
            args={"homo_oligomer": 1, "number_of_packs_per_design": 1},
        ),
        Blocks["af3"](diffusion_batch_size=1),
        Blocks["rmsd"](
            alignment_methods=[SELECTIONS.BKBN],
            rmsd_targets=[SELECTIONS.BKBN],
            which=[2, -1],
        ),
        Blocks["filter"](
            name="af3_filter",
            filters={
                "rmsd.bkbn_align.bkbn_rmsd": MetricFilter(max=2.0),
                "af3.plddt": MetricFilter(min=90.0),
            },
        ),
    ]


def main():
    from absl import flags

    flags.DEFINE_string("input_dir", None, "Directory with authorized dn5_stump.pdb and dn5b_stalk.pdb inputs.")
    flags.mark_flag_as_required("input_dir")
    init_cli()
    files = Path(flags.FLAGS.input_dir).expanduser().resolve()
    for filename in ("dn5_stump.pdb", "dn5b_stalk.pdb"):
        if not (files / filename).is_file():
            raise FileNotFoundError(f"Required user-supplied input is missing: {files / filename}")
    engine = ExecutionEngine(rundir=Path(__file__).resolve().parent / "rundir")
    result = engine.run(pipeline_blocks(files))
    result.unpack().to_csv(engine.rundir / "results.csv", index=False)
    print(engine)


if __name__ == "__main__":
    main()
