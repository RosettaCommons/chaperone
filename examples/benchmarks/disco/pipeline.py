#!/usr/bin/env -S bash -c '"${PIPELINES_REPO:-.}/.venv/bin/python" "$0" "$@"'
"""DISCO ligand-conditioned protein co-design benchmark.

Evaluates DISCO's ability to jointly generate protein sequences and structures
conditioned on a small molecule (heme B). Designs are validated by AF3 refolding
and backbone RMSD, following the co-designability metric from the DISCO paper.

Pipeline: disco -> af3 -> rmsd
"""

from pathlib import Path

from blocks import Blocks
from pipelines import ExecutionEngine, init_cli
from pipelines.data.schema import SELECTIONS

init_cli()

SCRIPT_DIR = Path(__file__).resolve().parent
FILES_DIR = SCRIPT_DIR / "files"

# ── Block configuration ─────────────────────────────────────────────────

DISCO, AF3, RMSD = Blocks.getall("disco", "af3", "rmsd")

blocks = [
    # Generate protein sequences + structures conditioned on heme B
    DISCO(
        inputs=str(FILES_DIR / "heme_b_config.json"),
        num_seeds=10,
        experiment="diverse",
        effort="max",
        n_batches=2,
    ),
    # Refold with AF3 — DISCO labels the ligand as 'l01' in its PDB output,
    # so we map it to heme B's CCD code for AF3 to recognise it.
    AF3(specified_ccd={"l01": "HEM"}),
    RMSD(
        alignment_methods=[SELECTIONS.BKBN],
        rmsd_targets=[SELECTIONS.BKBN, SELECTIONS.CA],
    ),
]

# ── Execute ──────────────────────────────────────────────────────────────

engine = ExecutionEngine(rundir=SCRIPT_DIR / "rundir", reinit=True)
handler = engine.run(blocks)

df = handler.unpack()
print(df)
