#!/usr/bin/env -S bash -c '"${CHAPERONE_REPO:-.}/.venv/bin/python" "$0" "$@"'
"""Zinc metalloprotease design pipeline.

De novo design of a Zn-dependent protease targeting fibrin Aa peptide.
RFD3 generates backbones with HEXXH + guide-strand + Tyr + Glu catalytic
motifs around a Zn coordination site, with a substrate peptide (chain B).

Pipeline::

  rfd3 -> filter -> mpnn -> [rf3/mpnn cycling] -> af3_monomer (apo) -> rmsd -> filter
                                                   -> af3_complex (holo) -> rmsd -> filter

Adapted from Ashida et al. protease design campaign.
"""

from pathlib import Path

from blocks import Blocks
from blocks.util_blocks.filter import MetricFilter
from chaperone import ExecutionEngine, init_cli
from chaperone.data.schema import SELECTIONS


SCRIPT_DIR = Path(__file__).resolve().parent


def main():
    from absl import flags

    flags.DEFINE_string("input_dir", None, "Directory of authorized workflow inputs; see the example README.")
    flags.mark_flag_as_required("input_dir")
    init_cli()
    files_dir = Path(flags.FLAGS.input_dir).expanduser().resolve()
    for filename in ['protease.json', 'GPRVVERHQS78.pdb']:
        if not (files_dir / filename).is_file():
            raise FileNotFoundError(f"Required user-supplied input is missing: {files_dir / filename}")
    RFD3, MPNN, RF3, AF3, RMSD, Filter = Blocks.getall(
        "rfd3", "mpnn", "rf3", "af3", "rmsd", "filter"
    )

    # -- Block configuration -----------------------------------------------------

    # Phase 1: backbone generation -> filtering -> sequence design
    design_blocks = [
        RFD3(
            inputs=str(files_dir / "protease.json"),
            n_batches=1,
            batch_size=2,
            inference_sampler={"s_jitter_origin": 1},
        ),
        Filter(
            name="rfd3_filter",
            filters={
                "rfd3.n_chainbreaks": MetricFilter(max=0),
                "rfd3.n_clashing.ligand_clashes": MetricFilter(max=0),
                "rfd3.n_clashing.interresidue_clashes_w_backbone": MetricFilter(max=0),
                "rfd3.n_clashing.interresidue_clashes_w_sidechain": MetricFilter(max=0),
                "rfd3.non_loop_fraction": MetricFilter(min=0.2),
                "rfd3.n_clashing.ligand_min_distance": MetricFilter(min=2.0),
            },
        ),
        MPNN(
            model_type="ligand_mpnn",
            output_mode="structures",
            keep_motif_sequences_fixed=True,
            num_seq_per_pdb=2,
            args={
                "temperature": 0.3,
                "number_of_packs_per_design": 1,
                "omit_AA": "CH",
                "chains_to_design": "A",
            },
        ),
    ]

    # Phase 2: RF3/MPNN cycling for backbone refinement
    NUM_CYCLES = 1  # number of RF3/MPNN cycles; increase for production
    cycle_blocks = []
    for cycle in range(NUM_CYCLES):
        if cycle > 0:
            cycle_blocks.append(
                MPNN(
                    name=f"cycles/mpnn_{cycle}",
                    model_type="ligand_mpnn",
                    output_mode="structures",
                    keep_motif_sequences_fixed=True,
                    num_seq_per_pdb=2,
                    args={
                        "temperature": 0.3,
                        "number_of_packs_per_design": 1,
                        "omit_AA": "CH",
                        "chains_to_design": "A",
                    },
                )
            )
        cycle_blocks.append(
            RF3(
                name=f"cycles/rf3_{cycle}",
                diffusion_batch_size=1,
                template_selection=["B"],
                num_diffusion_steps=50,
                n_recycles=5,
                ground_truth_conformer_selection=["B"],
            )
        )

    # Branch 1: Monomer folding (enzyme chain A only, no Zn or peptide)
    monomer_blocks = [
        Blocks["select_chains"](name="enzyme_only", chains=["A"]),
        AF3(
            name="af3_monomer",
            diffusion_batch_size=1,
            remove_ligands=True,
        ),
        RMSD(
            name="af3_monomer_rmsd",
            which=[-2, -1],
            alignment_methods=[SELECTIONS.BKBN],
            rmsd_targets=[SELECTIONS.BKBN, SELECTIONS.CA, SELECTIONS.MOTIF_ALLATOM],
        ),
        Filter(
            name="monomer_filter",
            filters={
                "af3_monomer.plddt": MetricFilter(min=80.0),
                "af3_monomer_rmsd.bkbn_align.ca_rmsd": MetricFilter(max=2.0),
            },
        ),
    ]

    # Branch 2: Complex folding (enzyme + peptide + Zn/water representation)
    complex_blocks = [
        AF3(
            name="af3_complex",
            diffusion_batch_size=1,
            input_mode="json",
            specified_smiles={"ZnO": "[Zn+2].O"},
        ),
        RMSD(
            name="af3_complex_rmsd",
            alignment_methods=[SELECTIONS.BKBN],
            rmsd_targets=[SELECTIONS.BKBN, SELECTIONS.CA, SELECTIONS.MOTIF_ALLATOM],
        ),
        Filter(
            name="complex_filter",
            filters={
                "af3_complex.plddt": MetricFilter(min=80.0),
                "af3_complex_rmsd.bkbn_align.bkbn_rmsd": MetricFilter(max=2.5),
            },
        ),
    ]

    # -- Execute ------------------------------------------------------------------

    engine = ExecutionEngine(rundir=SCRIPT_DIR / "rundir", reinit=True)

    # Design: rfd3 -> filter -> mpnn
    handler = engine.run(design_blocks)

    # Cycling: rf3/mpnn refinement
    for block in cycle_blocks:
        if not len(handler):
            break
        handler = engine.run(block, data=handler)

    # Branch 1: Monomer folding (fast apo screen)
    handler = engine.run(monomer_blocks, data=handler)

    # Branch 2: Complex folding (rewind to last rf3, stash monomer metrics)
    if not len(handler):
        handler.unpack().to_csv(engine.rundir / "results.csv", index=False)
        return
    handler = handler.rewind_to(f"cycles/rf3_{NUM_CYCLES - 1}")
    handler = engine.run(complex_blocks, data=handler)

    handler.unpack().to_csv(engine.rundir / "results.csv", index=False)
    print(engine)


if __name__ == "__main__":
    main()
