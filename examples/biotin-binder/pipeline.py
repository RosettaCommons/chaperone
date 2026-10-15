#!/usr/bin/env -S bash -c '"${CHAPERONE_REPO:-.}/.venv/bin/python" "$0" "$@"'
"""Biotin binder — single-pass AF3 pipeline (linker-based BTN, residue B;N).

One linear pass, no cycling:
  rfd3 -> contact_ms -> rasa -> rfd3_filter
    -> mpnn -> ligand_ddg_soft -> ddg_filter           (sequence design + triage)
      -> af3_final -> remap -> {hbond, rasa, contact_ms, rmsd} -> af3_final_filter
        -> af3_apo -> apo_holo_rmsd -> apo_filter       (single-chain stability)

AF3 is the single oracle: it designs the holo validation and the apo stability
check. Tune everything in the CONFIG section below, then run top-to-bottom.

Run:
    ./pipeline.py --test                 # 1 design end-to-end (SLURM)
    ./pipeline.py --test --use_local_gpus # 1 design, local GPU, smoke test
    ./pipeline.py                        # production defaults below
    ./pipeline.py --rfd3.n_batches=50 --partition=gpu-bf
"""

from __future__ import annotations

from pathlib import Path

from blocks import Blocks
from blocks.util_blocks.filter import MetricFilter
from chaperone import ExecutionEngine, init_cli
from chaperone.data.schema import SELECTIONS


# ── CONFIG — edit this section for your ligand, then run top-to-bottom ────────

SCRIPT_DIR = Path(__file__).resolve().parent
FILES_DIR = SCRIPT_DIR / "files"

RUNDIR = SCRIPT_DIR / "biotin"
INPUT_JSON = str(FILES_DIR / "biotin.json")

# Bounded default: two backbones; --test reduces this to one.
N_BATCHES = 1
BATCH_SIZE = 2
JITTER_ORI_SIGMA = 1.0

# Ligand chemistry (linker-based B;N throughout).
LIGAND_NAME = "B;N"
LIGAND_SMILES = "N1C(=O)N[C@@H]2[C@@H](SC[C@H]12)CCCCC(=O)NCCCCC"
LIGAND_CIF = str(FILES_DIR / "biotin_all.pdb")
# Full core (matches files/biotin.json `select_buried`) — must be buried.
BURIED_ATOMS = [
    "N1",
    "N2",
    "S1",
    "O3",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "C7",
    "C8",
    "C9",
    "C10",
]
# Linker atoms that must stay solvent-exposed (OR-gate).
EXPOSED_ATOMS = ["C14", "C15", "C16"]
BINDER_CHAINS = ["A"]
TARGET_CHAINS = ["B"]

# MPNN + DDG triage.
NUM_SEQ_PER_PDB = 2
DDG_TOP_FRACTION = 0.1  # keep top 10% sequences per parent backbone
POLAR_BIAS = (
    "D:1.39,E:1.39,H:1.39,K:1.39,N:1.39,Q:1.39,R:1.39,S:1.39,T:1.39,W:1.39,Y:1.39"
)

PROBE_RADIUS = 6.0  # RASA probe (Å)

# RFD3 filters.
RFD3_MAX_LIGAND_CLASHES = 0
RFD3_MIN_NON_LOOP_FRACTION = 0.2
RFD3_MIN_LIGAND_DISTANCE = 2.0
RFD3_MIN_CONTACT_MS = 110

# AF3 holo filter.
AF3_MIN_PAE = 1.2
AF3_MIN_HBOND = 3
AF3_MIN_CONTACT_MS = 120

# AF3 apo filter (single-chain — no interface metrics; gate on RMSD + pLDDT).
APO_MAX_BKBN_RMSD = 2
APO_MIN_PLDDT = 85

SMILES_KW = {"specified_smiles": {LIGAND_NAME: LIGAND_SMILES}, "input_mode": "json"}

# ── Block factories ──────────────────────────────────────────────────────────

RFD3, MPNN, AF3, RMSD, Filter, ContactMS, RASA, HBond, DDG, RemapLigand = Blocks.getall(
    "rfd3",
    "mpnn",
    "af3",
    "rmsd",
    "filter",
    "contact_ms",
    "rasa",
    "hbond_metrics",
    "ligand_ddg_soft",
    "remap_ligand",
)


def contact_ms_block(name, *, target_atoms=False):
    return ContactMS(
        name=name,
        binder_chains=BINDER_CHAINS,
        target_chains=TARGET_CHAINS,
        target_atom_names=BURIED_ATOMS if target_atoms else None,
    )


def rasa_block(name):
    return RASA(name=name, probe_radius=PROBE_RADIUS, backbone_only=True)


def rmsd_block(name, **kw):
    return RMSD(
        name=name,
        alignment_methods=[SELECTIONS.BKBN, SELECTIONS.CA],
        rmsd_targets=[SELECTIONS.BKBN, SELECTIONS.CA],
        **kw,
    )


def linker_exposed_groups(step: str, min_rasa: float = 0.1) -> dict:
    """Disjunctive (OR) groups: at least one linker atom must stay exposed.

    Core atoms are conditioned for burial in RFD3; contact_ms measures
    aggregate contacts, not burial of every atom. This OR verifies the pocket didn't swallow the solvent-facing linker. Lenient
    enough to survive per-atom RASA noise on a single refold.
    """
    return {
        f"{a}_exposed": {f"{step}.ligand_atom_rasa.{a}": MetricFilter(min=min_rasa)}
        for a in EXPOSED_ATOMS
    }


RFD3_FILTERS = {
    "rfd3.n_clashing.ligand_clashes": MetricFilter(max=RFD3_MAX_LIGAND_CLASHES),
    "rfd3.non_loop_fraction": MetricFilter(min=RFD3_MIN_NON_LOOP_FRACTION),
    "rfd3.n_clashing.ligand_min_distance": MetricFilter(min=RFD3_MIN_LIGAND_DISTANCE),
    "rfd3_contact_ms": MetricFilter(min=RFD3_MIN_CONTACT_MS),
}


# ── Pipeline execution ───────────────────────────────────────────────────────


def main():
    init_cli()
    engine = ExecutionEngine(RUNDIR, reinit=False)

    # ── Stage 1: RFD3 — backbone generation ──────────────────────────────────
    h = engine.run(
        RFD3(
            inputs=INPUT_JSON,
            n_batches=N_BATCHES,
            batch_size=BATCH_SIZE,
            jitter_ori_sigma=JITTER_ORI_SIGMA,
        ),
    )
    n_rfd3 = len(h)
    print(f"RFD3 generated: {n_rfd3} designs")

    # ── Stage 2: RFD3 metrics + filter (bundled, fast CPU) ───────────────────
    h = engine.run(
        [
            contact_ms_block("rfd3_contact_ms", target_atoms=True),
            rasa_block("rfd3_rasa"),
            Filter(
                name="rfd3_filter",
                filters=RFD3_FILTERS,
                filter_groups=linker_exposed_groups("rfd3_rasa"),
            ),
        ],
        data=h,
    )
    print(f"RFD3 filter passed: {len(h)} / {n_rfd3} designs")

    if not len(h):
        h.unpack().to_csv(engine.rundir / "results.csv", index=False)
        return

    # ── Stage 3: MPNN + DDG triage — sequence design (bundled CPU) ───────────
    # MPNN locks the RFD3 motif, biases polar AAs, and fixes hbond/aromatic
    # contacts; ligand_ddg_soft scores binding; ddg_filter keeps the top
    # fraction per parent backbone.
    h = engine.run(
        [
            MPNN(
                num_seq_per_pdb=NUM_SEQ_PER_PDB,
                output_mode="structures",
                keep_motif_sequences_fixed=True,
                hbond_fix=True,
                aromatic_fix=True,
                args={"bias_AA": POLAR_BIAS, "number_of_packs_per_design": 1},
            ),
            DDG(ref_smiles=LIGAND_SMILES, interesting_atoms=BURIED_ATOMS),
            Filter(
                name="ddg_filter",
                filters={
                    "ligand_ddg_soft.ddg_soft": MetricFilter(
                        quantile=DDG_TOP_FRACTION,
                        quantile_direction="lower",
                        parent_based=True,
                    ),
                },
            ),
        ],
        data=h,
    )
    print(f"MPNN + DDG triage: {len(h)} sequences")

    if len(h) == 0:
        print("  0 sequences survived DDG triage — done")
        h.unpack().to_csv(engine.rundir / "results.csv", index=False)
        return

    # ── Stage 4: AF3 final holo — oracle validation ──────────────────────────
    h = engine.run(
        AF3(name="af3_final", diffusion_batch_size=1, **SMILES_KW),
        data=h,
    )
    # Remap predicted ligand atom/residue names back to the reference (B;N)
    # BEFORE the metric blocks read them. As a block it slots into the chain,
    # writes remapped CIFs into its own sandbox (upstream untouched), and
    # splits across chunks like every other block.
    h = engine.run(
        [
            RemapLigand(
                name="remap",
                reference_cif=LIGAND_CIF,
                target_name=LIGAND_NAME,
                buried_atoms=BURIED_ATOMS,
            ),
            HBond(name="af3_final_hbond"),
            rasa_block("af3_final_rasa"),
            contact_ms_block("af3_final_contact_ms", target_atoms=True),
            rmsd_block("af3_final_rmsd"),
            Filter(
                name="af3_final_filter",
                filters={
                    "af3_final.min_pae_interaction": MetricFilter(max=AF3_MIN_PAE),
                    "af3_final_hbond.num_hbonds": MetricFilter(min=AF3_MIN_HBOND),
                    "af3_final_contact_ms": MetricFilter(min=AF3_MIN_CONTACT_MS),
                },
                filter_groups=linker_exposed_groups("af3_final_rasa"),
            ),
        ],
        data=h,
    )
    holo_survivors = len(h)
    print(f"AF3 final holo filter: {holo_survivors} pass")

    if holo_survivors == 0:
        print("  0 designs passed AF3 final filter — skipping apo")
        h.unpack().to_csv(engine.rundir / "results.csv", index=False)
        return

    # ── Stage 5: AF3 apo — ligand-independent stability check ─────────────────
    # Apo is single-chain (ligand removed): interface metrics are undefined, so
    # we only gate on apo-vs-holo backbone RMSD and apo pLDDT.
    h = engine.run(
        AF3(name="af3_apo", remove_ligands=True, diffusion_batch_size=1),
        data=h,
    )
    h = engine.run(
        [
            rmsd_block("apo_holo_rmsd", which=[-2, -1]),
            Filter(
                name="apo_filter",
                filters={
                    "apo_holo_rmsd.bkbn_align.bkbn_rmsd": MetricFilter(
                        max=APO_MAX_BKBN_RMSD
                    ),
                    "af3_apo.plddt": MetricFilter(min=APO_MIN_PLDDT),
                },
            ),
        ],
        data=h,
    )
    print(f"AF3 apo filter: {len(h)} pass")

    print(f"\n  All done! Results in: {engine.rundir}")
    h.unpack().to_csv(engine.rundir / "results.csv", index=False)
    print(engine)


if __name__ == "__main__":
    main()
