#!/usr/bin/env -S bash -c '"${CHAPERONE_REPO:-.}/.venv/bin/python" "$0" "$@"'
"""Held-out scorer for the dna-binder eval — major-groove dsDNA recognition.

Supplies the canonical duplex (`inputs/target.pdb`, chains B and C) as
co-folded context, refolds every submitted binder sequence as chain A with
plain AF3, and gates on fold quality, interface confidence, base-edge readout,
packing, duplex preservation, and sterics.

Plain `af3` is used deliberately: `af3_with_target_template` only injects
templates into `entry["protein"]` and silently skips a DNA/RNA target chain,
so a nucleic-acid target would be folded with no template at all while the
report claimed otherwise. The duplex is instead pinned by an explicit
nucleotide RMSD gate.

Usage:
    ./score.py --designs=/absolute/path/to/submission.csv --nouse_local_gpus
"""

from __future__ import annotations

from pathlib import Path

from absl import flags

from blocks import Blocks
from chaperone import init_cli
from chaperone.data.schema import SELECTIONS
from scoring import (
    SequenceSubmissionSpec,
    define_score_flags,
    load_context_chains,
    run_sequence_scorer,
)

define_score_flags()
FLAGS = flags.FLAGS

init_cli()

EVAL_NAME = "dna-binder"
ORACLE = "af3"

SCRIPT_DIR = Path(__file__).resolve().parent
TARGET_PDB = (SCRIPT_DIR.parent / "inputs" / "target.pdb").resolve()

BINDER_CHAINS = ["A"]
TARGET_CHAINS = ["B", "C"]

# ── _SYNC: scientific thresholds (see score/readme.md) ───────────────────────
PLDDT_MIN = 80.0
INTERFACE_PAE_MAX = 5.0
IPSAE_MIN = 0.30  # lowered from the spec's 0.50; see score/readme.md
MAJOR_HBOND_MIN = 3  # lowered from the spec's 4; see score/readme.md
TOTAL_HBOND_MIN = 8
NT_AA_MIN = 10  # lowered from the spec's 12; see score/readme.md
CONTACT_MS_MIN = 400.0
DNA_RMSD_MAX = 2.0
BB_CLASHES_MAX = 0
IPTM_MIN = 0.60  # soft
IPSAE_PAE_CUTOFF = 10.0
# ── end _SYNC ────────────────────────────────────────────────────────────────

PLDDT_COL = "score_af3.plddt"
IPTM_COL = "score_af3.iptm"
AF3_MIN_PAE_COL = "score_af3.min_pae_interaction"  # reported, never gated
IPSAE_COL = "score_ipsae.ipsae"
INTERFACE_PAE_COL = "score_ipsae.min_pae_interface"
MAJOR_HBOND_COL = "score_dssr.Major_hbond"
TOTAL_HBOND_COL = "score_dssr.Total_hbond"
NT_AA_COL = "score_dssr.nt_aa"
CONTACT_MS_COL = "score_contact_ms"  # contact_ms emits a bare float
DNA_RMSD_COL = (
    f"score_rmsd.{SELECTIONS.NUCLEOTIDE_BKBN.value}_align."
    f"{SELECTIONS.NUCLEOTIDE.value}_rmsd"
)
BB_CLASH_COL = "score_clash.n_bb_clashes"

# One SLURM GPU; `rows_per_job` keeps the fold as a single chunk.
GPU_ARGS = {
    "partition": "gpu",
    "gres": "gpu:small:1",
    "mem_gb": 32,
    "time_limit": "8:00:00",
    "rows_per_job": 4096,
}
CPU_ARGS = {"partition": "local"}


def main() -> None:
    AF3, IpSAE, DSSR, ContactMS, RMSD, Clash = Blocks.getall(
        ORACLE, "ipsae", "dssr_snap_metrics", "contact_ms", "rmsd", "clash"
    )

    run_sequence_scorer(
        eval_name=EVAL_NAME,
        validation_oracle=ORACLE,
        designs=FLAGS.designs,
        out=FLAGS.out,
        per_design_out=FLAGS.per_design_out,
        required_n=FLAGS.required_n,
        run_meta=FLAGS.run_meta,
        run_root=SCRIPT_DIR / "score_runs",
        spec=SequenceSubmissionSpec(
            required_n=FLAGS.required_n,
            chains=tuple(BINDER_CHAINS),
            max_pairwise_identity=0.70,
        ),
        # The dsDNA duplex (chains B, C) is supplied by the scorer, never by the
        # submitter, and is co-folded with every binder sequence.
        context=load_context_chains(TARGET_PDB, TARGET_CHAINS),
        oracle=AF3(name="score_af3", diffusion_batch_size=1),
        oracle_job_args=GPU_ARGS,
        metrics=[
            IpSAE(
                name="score_ipsae",
                binder_chain=BINDER_CHAINS[0],
                target_chains=TARGET_CHAINS,
                pae_cutoff=IPSAE_PAE_CUTOFF,
            ),
            DSSR(name="score_dssr"),
            ContactMS(
                name="score_contact_ms",
                binder_chains=BINDER_CHAINS,
                target_chains=TARGET_CHAINS,
            ),
            RMSD(
                name="score_rmsd",
                alignment_methods=[SELECTIONS.NUCLEOTIDE_BKBN],
                rmsd_targets=[SELECTIONS.NUCLEOTIDE],
            ),
            Clash(name="score_clash"),
        ],
        metric_job_args=CPU_ARGS,
        # Gates are filter-expression strings (the repo's FilterBlock DSL); each
        # threshold lives in the expression that enforces it. `iptm` is soft.
        gates={
            "plddt": f"{PLDDT_COL} >= {PLDDT_MIN}",
            "interface_pae": f"{INTERFACE_PAE_COL} <= {INTERFACE_PAE_MAX}",
            "ipsae": f"{IPSAE_COL} >= {IPSAE_MIN}",
            "major_hbond": f"{MAJOR_HBOND_COL} >= {MAJOR_HBOND_MIN}",
            "total_hbond": f"{TOTAL_HBOND_COL} >= {TOTAL_HBOND_MIN}",
            "nt_aa": f"{NT_AA_COL} >= {NT_AA_MIN}",
            "contact_ms": f"{CONTACT_MS_COL} >= {CONTACT_MS_MIN}",
            "dna_preserved": f"{DNA_RMSD_COL} <= {DNA_RMSD_MAX}",
            "no_bb_clash": f"{BB_CLASH_COL} <= {BB_CLASHES_MAX}",
            "iptm": f"{IPTM_COL} >= {IPTM_MIN}",
        },
        soft=["iptm"],
        # A NaN metric is an upstream bug, not a failing design. In particular
        # the SNAP-derived counts (nt_aa) go NaN when the x3dna-snap binary is
        # missing, which must fail the run loudly rather than fail every design.
        metric_columns=[
            PLDDT_COL,
            IPTM_COL,
            AF3_MIN_PAE_COL,
            IPSAE_COL,
            INTERFACE_PAE_COL,
            MAJOR_HBOND_COL,
            TOTAL_HBOND_COL,
            NT_AA_COL,
            CONTACT_MS_COL,
            DNA_RMSD_COL,
            BB_CLASH_COL,
        ],
        extra={
            "target": TARGET_PDB.name,
            "binder_chains": BINDER_CHAINS,
            "target_chains": TARGET_CHAINS,
        },
    )


if __name__ == "__main__":
    main()
