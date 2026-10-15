#!/usr/bin/env -S bash -c '"${CHAPERONE_REPO:-.}/.venv/bin/python" "$0" "$@"'
"""Glycosidase enzyme design pipeline.

De novo enzyme design with active site constraints using a theozyme definition.
Three AF3 evaluation branches from a shared packed MPNN step:
  1. Apo folding     — fast backbone quality check (no ligand)
  2. PTM folding     — substrate covalently attached at catalytic GLU A86
  3. Ligand folding  — substrate as a separate ligand chain

Pipeline: rfd3 -> filter -> mpnn -> {apo, ptm, ligand} AF3 branches
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
    for filename in ['glycosidase.json', 'theozyme.pdb', 'af3_template_ptm.json', 'af3_template_ligand.json']:
        if not (files_dir / filename).is_file():
            raise FileNotFoundError(f"Required user-supplied input is missing: {files_dir / filename}")
    RFD3, MPNN, AF3, RMSD, Filter = Blocks.getall(
        "rfd3", "mpnn", "af3", "rmsd", "filter"
    )

    # ── Block configuration ─────────────────────────────────────────────────

    # Shared: backbone generation -> filtering -> sequence design
    design_blocks = [
        RFD3(
            inputs=str(files_dir / "glycosidase.json"),
            n_batches=1,
            batch_size=2,
        ),
        Filter(
            name="rfd3_filter",
            filters={
                "rfd3.n_chainbreaks": MetricFilter(max=0),
                "rfd3.n_clashing.ligand_clashes": MetricFilter(max=0),
                "rfd3.non_loop_fraction": MetricFilter(min=0.2),
                "rfd3.n_clashing.ligand_min_distance": MetricFilter(min=2.0),
                "rfd3.insertion_rmsd": MetricFilter(max=0.5),
                "rfd3.n_conjoined_residues": MetricFilter(max=0),
            },
        ),
        MPNN(
            name="mpnn",
            model_type="ligand_mpnn",
            output_mode="structures",
            keep_motif_sequences_fixed=True,
            num_seq_per_pdb=2,
            args={"number_of_packs_per_design": 1},
        ),
    ]

    # Branch 1: Apo folding (no ligand, fast backbone screen)
    apo_blocks = [
        AF3(
            name="af3_apo",
            diffusion_batch_size=1,
            remove_ligands=True,
        ),
        RMSD(
            name="af3_apo_rmsd",
            alignment_methods=[SELECTIONS.BKBN],
            rmsd_targets=[SELECTIONS.BKBN, SELECTIONS.MOTIF_ALLATOM],
        ),
        Filter(
            name="af3_apo_filter",
            filters={
                "af3_apo_rmsd.bkbn_align.bkbn_rmsd": MetricFilter(max=2.0),
                "af3_apo.plddt": MetricFilter(min=80.0),
            },
        ),
    ]

    # Branch 2: PTM folding (glycosyl-enzyme intermediate at catalytic GLU A86)
    ptm_blocks = [
        AF3(
            name="af3_ptm",
            diffusion_batch_size=1,
            input_mode="json",
            template_json=str(files_dir / "af3_template_ptm.json"),
            add_ptm_at_mapped_residues={"L:1": "A86"},
        ),
        RMSD(
            name="af3_ptm_rmsd",
            alignment_methods=[SELECTIONS.BKBN],
            rmsd_targets=[SELECTIONS.BKBN],
            # The custom PTM is HETATM; align common unmodified backbone atoms.
            select_mask_by_name_pairing=[SELECTIONS.BKBN],
        ),
        Blocks["reference_motif_rmsd"](
            name="af3_ptm_catalytic",
            reference=str(files_dir / "theozyme.pdb"),
            residues=["A77", "A86", "A88", "A122", "A136", "A177"],
        ),
        Filter(
            name="af3_ptm_filter",
            filters={
                "af3_ptm_rmsd.bkbn_align.bkbn_rmsd": MetricFilter(max=2.0),
                "af3_ptm_catalytic.motif_rmsd": MetricFilter(max=2.0),
                "af3_ptm_catalytic.n_residues": MetricFilter(min=6, max=6),
                "af3_ptm_catalytic.n_atoms": MetricFilter(min=38, max=38),
            },
        ),
    ]

    # Branch 3: Ligand folding (substrate as separate chain B)
    ligand_blocks = [
        AF3(
            name="af3_ligand",
            diffusion_batch_size=1,
            input_mode="json",
            template_json=str(files_dir / "af3_template_ligand.json"),
        ),
        RMSD(
            name="af3_ligand_rmsd",
            alignment_methods=[SELECTIONS.BKBN],
            rmsd_targets=[
                SELECTIONS.BKBN,
                SELECTIONS.MOTIF_ALLATOM,
                SELECTIONS.LIGANDS,
            ],
        ),
        Filter(
            name="af3_ligand_filter",
            filters={
                "af3_ligand_rmsd.bkbn_align.bkbn_rmsd": MetricFilter(max=2.0),
                "af3_ligand.ligand_plddt": MetricFilter(min=80.0),
            },
        ),
    ]

    # ── Execute ──────────────────────────────────────────────────────────────

    engine = ExecutionEngine(rundir=SCRIPT_DIR / "rundir", reinit=True)

    # Design: rfd3 -> filter -> mpnn
    handler = engine.run(design_blocks)

    # Branch 1: Apo folding (fast backbone screen)
    handler = engine.run(apo_blocks, data=handler)

    # Branch 2: PTM folding (restore packed MPNN structure, retain apo metrics)
    if not len(handler):
        handler.unpack().to_csv(engine.rundir / "results.csv", index=False)
        return
    handler = handler.rewind_to("mpnn")
    handler = engine.run(ptm_blocks, data=handler)

    # Branch 3: Ligand folding (restore packed MPNN structure, retain PTM metrics)
    if not len(handler):
        handler.unpack().to_csv(engine.rundir / "results.csv", index=False)
        return
    handler = handler.rewind_to("mpnn")
    handler = engine.run(ligand_blocks, data=handler)

    handler.unpack().to_csv(engine.rundir / "results.csv", index=False)
    print(engine)


if __name__ == "__main__":
    main()
