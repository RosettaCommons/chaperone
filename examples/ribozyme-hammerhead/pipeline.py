#!/usr/bin/env -S bash -c '"${CHAPERONE_REPO:-.}/.venv/bin/python" "$0" "$@"'
"""Hammerhead ribozyme rescaffolding with RFdiffusion3NA.

Hands RFD3NA only the six catalytic nucleotides of the full-length Schistosoma
hammerhead (PDB 2OEU) plus its active-site Mn2+, unindexed, and asks it to build
a new fixed-length RNA around them. Every metric is scored against the canonical
crystal structure or against absolute catalytic geometry - never against the
diffusion output.

Pipeline: rfd3na -> filter -> af3 -> reference_motif_rmsd -> motif_geometry -> filter
"""

from pathlib import Path

from blocks import Blocks
from blocks.networks.foundry.rfd3na_input import build_motif_spec, write_rfd3na_json
from blocks.util_blocks.filter import MetricFilter
from chaperone import ExecutionEngine, init_cli


SCRIPT_DIR = Path(__file__).resolve().parent
FILES_DIR = SCRIPT_DIR / "files"
MOTIF = FILES_DIR / "2oeu_motif.pdb"

RFD3NA, AF3, RefMotifRMSD, MotifGeometry, Filter = Blocks.getall(
    "rfd3na", "af3", "reference_motif_rmsd", "motif_geometry", "filter"
)

# ── Motif configuration ─────────────────────────────────────────────────

# 2OEU numbering -> hammerhead numbering: A20=G8, A21=A9, A22=G10.1, A36=G12,
# B6=C17 (nucleophile), B7=C1.1 (leaving group). A20-22 are tied because they
# are a contiguous trinucleotide in the native; the other two groups are free to
# land anywhere. All six come from two native strands but are rescaffolded into
# a single chain, i.e. a cis-acting ribozyme.
#
# Commas keep the unindexed groups in one designed molecule. A "/0" would
# request separate molecules and change the chain connectivity problem.
UNINDEX = ["A20-22", "A36", "B6-7"]
LIGAND = "MN"
LENGTH = "60-60"

# Catalytic geometry, addressed by 2OEU residue id and resolved through RFD3NA's
# diffused_index_map. Native values in the comments.
CATALYTIC = {
    "general_base_dist": ["A36.N1", "B6.O2'"],  # G12 N1 -- C17 2'-OH   3.26 A
    "general_acid_dist": ["A20.O2'", "B7.O5'"],  # G8 2'-OH -- O5' LG   3.51 A
    "inline_attack_dist": ["B6.O2'", "B7.P"],  # nucleophile -- P       3.30 A
    "inline_attack_angle": ["B6.O2'", "B7.P", "B7.O5'"],  # in-line     152.0 deg
    "metal_a9_dist": ["L:MN.MN", "A21.OP2"],  # A9 pro-R phosphate      2.01 A
    "metal_g10_dist": ["L:MN.MN", "A22.N7"],  # G10.1 N7                2.08 A
}

# ── Execute ──────────────────────────────────────────────────────────────


def main():
    init_cli()
    engine = ExecutionEngine(rundir=SCRIPT_DIR / "rundir")

    spec_json = write_rfd3na_json(
        {
            "hammerhead": build_motif_spec(
                MOTIF, UNINDEX, nucleic="RNA", length=LENGTH, ligand=LIGAND
            )
        },
        engine.rundir / "rfd3na_input.json",
    )

    blocks = [
        RFD3NA(inputs=spec_json, n_batches=1, batch_size=2),
        # Reject poorly inserted motifs and explicit clashes before refolding.
        # These metrics do not establish RNA phosphodiester connectivity.
        Filter(
            name="motif_filter",
            filters={
                "rfd3na.insertion_rmsd": MetricFilter(max=1.0),
                "rfd3na.n_conjoined_residues": MetricFilter(max=0),
                "rfd3na.n_clashing.interresidue_clashes_w_backbone": MetricFilter(
                    max=0
                ),
                "rfd3na.n_clashing.interresidue_clashes_w_sidechain": MetricFilter(
                    max=0
                ),
            },
        ),
        # MN is a CCD code, so AF3 carries the ion through without a SMILES override.
        AF3(diffusion_batch_size=1),
        # Score the refold against the crystal motif. Mn placement is reported
        # separately and excluded from the RNA motif RMSD.
        RefMotifRMSD(reference=str(MOTIF), ligands=[]),
        MotifGeometry(measurements=CATALYTIC),
        # Metal distances are diagnostic outputs, not acceptance gates.
        Filter(
            name="refold_filter",
            filters={
                "reference_motif_rmsd.motif_rmsd": MetricFilter(max=2.0),
                "motif_geometry.inline_attack_dist": MetricFilter(max=4.0),
                "motif_geometry.inline_attack_angle": MetricFilter(min=140.0),
            },
        ),
    ]

    handler = engine.run(blocks)
    handler.unpack().to_csv(engine.rundir / "results.csv", index=False)
    print(engine)


if __name__ == "__main__":
    main()
