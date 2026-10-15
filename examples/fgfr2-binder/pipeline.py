#!/usr/bin/env -S bash -c '"${CHAPERONE_REPO:-.}/.venv/bin/python" "$0" "$@"'
"""FGFR2 competitive binder design — variable-length inhibitors of FGF binding to FGFR2 (PDB 1DJS).

Single-round pipeline: RFD3 generates backbones around auto-detected FGFR2 hotspots,
MPNN designs sequences, AF3 (template-aware: receptor pinned) validates, and ipSAE
ranks the binder–FGFR2 interface confidence.

Pipeline: rfd3 -> mpnn -> af3 -> rmsd -> ipsae -> filter

Run prep first to fetch 1DJS and emit `files/1djs_target.pdb` + `files/rfd3_config.json`:
    ./examples/fgfr2-binder/files/prep_target.py --binder-length-min 20 --binder-length-max 50
"""

import json
from pathlib import Path

from blocks import Blocks
from blocks.util_blocks.filter import MetricFilter
from chaperone import ExecutionEngine, init_cli
from chaperone.data.schema import SELECTIONS

init_cli()

SCRIPT_DIR = Path(__file__).resolve().parent
FILES_DIR = SCRIPT_DIR / "files"

# ── Conditioning override ───────────────────────────────────────────────────
# Hotspots = geometric constraint (contact this patch).
# H-bond conditioning = chemical constraint (contacts must be polar with this
# directionality). Best practice is to use BOTH together.
#
# Convention for the H-bond dicts:
#   HBOND_DONOR    = TARGET atoms that DONATE H-bonds (binder will accept here)
#   HBOND_ACCEPTOR = TARGET atoms that ACCEPT H-bonds (binder will donate here)
#
# 5 hotspots on D3 outer face (orthogonal to the FGF-binding region).
# RFD3 contig (set by prep_target.py --receptor-contig) covers B287-296 + B307-328 + B339-350.
HOTSPOTS: list[str] | None = ["B289", "B291", "B311", "B344", "B352"]
# H-bond conditioning disabled this iteration — hotspots only.
HBOND_DONOR: dict[str, str] | None = None
HBOND_ACCEPTOR: dict[str, str] | None = None
# Encourage helical/sheet-rich (non-loopy) backbones.
IS_NON_LOOPY: bool = True


def _resolve_rfd3_config(output_dir: Path) -> str:
    """Return the rfd3_config.json path, with HOTSPOTS / HBOND_* / IS_NON_LOOPY patched in."""
    base_path = FILES_DIR / "rfd3_config.json"
    cfg = json.loads(base_path.read_text())
    (key,) = cfg.keys()
    changed = False
    if HOTSPOTS is not None:
        cfg[key]["select_hotspots"] = ",".join(HOTSPOTS)
        print(
            f"[pipeline] HOTSPOTS override active ({len(HOTSPOTS)} residues): {HOTSPOTS}"
        )
        changed = True
    if HBOND_DONOR is not None:
        cfg[key]["select_hbond_donor"] = HBOND_DONOR
        print(f"[pipeline] HBOND_DONOR override active: {HBOND_DONOR}")
        changed = True
    if HBOND_ACCEPTOR is not None:
        cfg[key]["select_hbond_acceptor"] = HBOND_ACCEPTOR
        print(f"[pipeline] HBOND_ACCEPTOR override active: {HBOND_ACCEPTOR}")
        changed = True
    if IS_NON_LOOPY:
        cfg[key]["is_non_loopy"] = True
        print("[pipeline] is_non_loopy=True (favors helices/sheets)")
        changed = True
    if not changed:
        return str(base_path)
    # The override lives with this run; bundled example assets may be read-only.
    # Preserve the input reference when relocating the JSON away from files/.
    if cfg[key].get("input"):
        cfg[key]["input"] = str((FILES_DIR / cfg[key]["input"]).resolve())
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "rfd3_config_override.json"
    out_path.write_text(json.dumps(cfg, indent=4))
    return str(out_path)


RFD3, MPNN, MpnnScore, AF3, RMSD, IpSAE, ContactMS, Peppr, Filter = Blocks.getall(
    "rfd3",
    "mpnn",
    "mpnn_score",
    "af3_with_target_template",
    "rmsd",
    "ipsae",
    "contact_ms",
    "peppr",
    "filter",
)

# Resources for the bundled chunk (rfd3 → mpnn → af3 → rmsd → ipsae → filter).
# Templated AF3 with variable-length binders wants the headroom of a large GPU
# (~48 GB); GPU blocks get one, CPU blocks (mpnn, metrics) get none. The site maps
# the "large" class to a real --gres through CHAPERONE_GRES_MAP (at IPD, gpu:large:1),
# and to a partition through CHAPERONE_GPU_PARTITION. Use "small" if your queue is
# contended and the model fits.
JOB_ARGS = {"gpu_class": "large", "mem_gb": 16}

# ── Pipeline blocks ─────────────────────────────────────────────────────────────────────
# 4 RFD3 designs × 8 MPNN sequences (hbond_fix + aromatic_fix preserve H-bond
# residue identities across MPNN samples). MpnnScore parses FASTA confidence,
# the parent-based Filter keeps the top-1 sequence per RFD3 backbone, then AF3
# runs only on those 4 selected sequences (one per backbone).
N_MPNN_SEQS = 8

engine = ExecutionEngine(rundir=SCRIPT_DIR / "rundir", reinit=True)

blocks = [
    RFD3(
        inputs=_resolve_rfd3_config(engine.rundir / "inputs"),
        n_batches=4,
        batch_size=1,
        jitter_ori_sigma=1.0,
        seed_offset=42,
    ),
    MPNN(
        num_seq_per_pdb=N_MPNN_SEQS,
        model_type="ligand_mpnn",
        hbond_fix=True,  # preserve identity of residues in H-bonds
        aromatic_fix=True,  # preserve aromatic packers near interface
    ),
    MpnnScore(name="mpnn_score"),
    Filter(
        name="select_top_seq",
        filters={
            "mpnn_score.overall_confidence": MetricFilter(
                quantile=1.0 / N_MPNN_SEQS,
                quantile_direction="higher",
                parent_based=True,  # top-1 PER RFD3 backbone
            ),
        },
    ),
    AF3(
        name="af3",
        target_chain="B",
        # Use the FULL receptor as AF3's chain B + template, even though RFD3
        # only diffused against a sub-contig. The full PDB stays as the prep
        # output; AF3 swaps it in regardless of what RFD3/MPNN passed through.
        full_receptor_pdb=str(FILES_DIR / "1djs_target.pdb"),
    ),
    RMSD(
        name="af3_rmsd",
        alignment_methods=[SELECTIONS.BKBN],
        rmsd_targets=[SELECTIONS.BKBN, SELECTIONS.MOTIF_ALLATOM],
        # AF3 sees the FULL receptor (full_receptor_pdb) but RFD3 only the
        # truncated contig — so the two structures' BKBN atom counts differ.
        # Name-pair to compare only the residues both structures have.
        select_mask_by_name_pairing=[
            SELECTIONS.BKBN,
            SELECTIONS.MOTIF_ALLATOM,
            SELECTIONS.LIGANDS,
        ],
    ),
    IpSAE(name="af3_ipsae", binder_chain="A", target_chains=["B"]),
    ContactMS(name="cms", binder_chains=["A"], target_chains=["B"]),
    Peppr(name="peppr"),
    Filter(
        name="round_1_filter",
        filters={
            "af3.iptm": MetricFilter(min=0.7),
            "af3.min_pae_interaction": MetricFilter(max=2.0),
            "af3_rmsd.bkbn_align.bkbn_rmsd": MetricFilter(max=3.0),
            "af3_ipsae.ipsae": MetricFilter(min=0.4),
        },
    ),
]

# ── Execute ─────────────────────────────────────────────────────────────────────────────
handler = engine.run(blocks, job_args=JOB_ARGS)

df = handler.unpack(all=False, to_pandas=False)
if "af3_ipsae.ipsae" in df.columns:
    df = df.sort("af3_ipsae.ipsae", descending=True)

# Write a metrics-only CSV next to the run log. Drops nested structure columns
# (paths, metadata blobs) and keeps just the scoring columns for easy ranking.
rundir = engine.rundir
metric_prefixes = (
    "af3.",
    "af3_rmsd.",
    "af3_ipsae.",
    "cms",
    "peppr.",
    "rfd3.",
    "mpnn_score.",
)
metric_cols = [
    c
    for c in df.columns
    if (c.startswith(metric_prefixes) or c in {"name", "design_id", "step_idx"})
    and not df.schema[c].is_nested()
]
scores = df.select(metric_cols)
scores_path = rundir / "scores.csv"
scores.write_csv(scores_path)
print(f"\nWrote {scores_path}\n")
print(df)
