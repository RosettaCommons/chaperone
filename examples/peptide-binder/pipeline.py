#!/usr/bin/env -S bash -c '"${CHAPERONE_REPO:-.}/.venv/bin/python" "$0" "$@"'
"""Two-round binder design with fixed peptide sequence and flexible coordinates."""

import json
from pathlib import Path

from blocks import Blocks
from blocks.util_blocks.filter import MetricFilter
from chaperone import ExecutionEngine, init_cli
from chaperone.data.schema import SELECTIONS

SEQUENCE = "TVGSDGPPSFRSLSG"


def prepare_inputs(directory: Path) -> Path:
    from atomworks.enums import ChainType
    from atomworks.io.tools.inference import sequence_to_annotated_atom_array
    from atomworks.io.utils.io_utils import to_cif_file

    directory.mkdir(parents=True, exist_ok=True)
    atoms = sequence_to_annotated_atom_array(
        SEQUENCE, "B", chain_type=ChainType.POLYPEPTIDE_L, is_polymer=True
    )
    atoms = atoms[atoms.element != "H"]
    atoms.set_annotation("chain_iid", atoms.chain_id)
    target = directory / "peptide_target.cif"
    to_cif_file(atoms, target)
    config = directory / "rfd3_config.json"
    config.write_text(
        json.dumps(
            {
                "peptide_binder": {
                    "input": str(target.resolve()),
                    "contig": f"60-140,/0,B1-{len(SEQUENCE)}",
                    "length": f"{60 + len(SEQUENCE)}-{140 + len(SEQUENCE)}",
                    "redesign_motif_sidechains": False,
                    "select_fixed_atoms": False,
                }
            },
            indent=2,
        )
    )
    return config


def rounds(config: Path):
    RFD3, MPNN, AF3, RMSD, Filter = Blocks.getall(
        "rfd3", "mpnn", "af3", "rmsd", "filter"
    )
    first = [
        RFD3(
            inputs=str(config),
            n_batches=1,
            batch_size=4,
            jitter_ori_sigma=1.0,
            seed_offset=42,
        ),
        MPNN(
            num_seq_per_pdb=2, model_type="ligand_mpnn", keep_motif_sequences_fixed=True
        ),
        AF3(diffusion_batch_size=1),
        RMSD(
            name="af3_rmsd",
            alignment_methods=[SELECTIONS.BKBN],
            rmsd_targets=[SELECTIONS.BKBN, SELECTIONS.MOTIF_ALLATOM],
        ),
        Filter(
            name="round_1_filter",
            filters={
                "af3.iptm": MetricFilter(min=0.7),
                "af3.min_pae_interaction": MetricFilter(max=2.0),
                "af3_rmsd.bkbn_align.bkbn_rmsd": MetricFilter(max=3.0),
            },
        ),
    ]
    second = [
        MPNN(
            name="mpnn_2",
            num_seq_per_pdb=2,
            model_type="ligand_mpnn",
            keep_motif_sequences_fixed=True,
        ),
        AF3(name="af3_2", diffusion_batch_size=1),
        RMSD(
            name="af3_2_rmsd",
            alignment_methods=[SELECTIONS.BKBN],
            rmsd_targets=[SELECTIONS.BKBN, SELECTIONS.MOTIF_ALLATOM],
        ),
        Filter(
            name="round_2_filter",
            filters={
                "af3_2.iptm": MetricFilter(min=0.85),
                "af3_2.min_pae_interaction": MetricFilter(max=1.1),
                "af3_2_rmsd.bkbn_align.bkbn_rmsd": MetricFilter(max=2.0),
            },
        ),
    ]
    return first, second


def main():
    init_cli()
    engine = ExecutionEngine(rundir=Path(__file__).resolve().parent / "rundir")
    config = prepare_inputs(engine.rundir / "inputs")
    first, second = rounds(config)
    result = engine.run(first)
    if len(result):
        result = engine.run(second, data=result)
    result.unpack().to_csv(engine.rundir / "results.csv", index=False)
    print(engine)


if __name__ == "__main__":
    main()
