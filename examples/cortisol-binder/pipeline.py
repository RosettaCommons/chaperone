#!/usr/bin/env -S bash -c '"${CHAPERONE_REPO:-.}/.venv/bin/python" "$0" "$@"'
"""Cortisol: RFD3 -> cycle(MPNN -> AF3 -> H-bonds -> gate) -> final AF3.

Run from the repository root with --test for one backbone/sequence per round.
Two cycles by default; --cycles sets the bounded number of refinement rounds.
The scientific gates are identical in test and normal runs.
"""

from pathlib import Path

from score import gate_filters, write_scores

from blocks import Blocks
from chaperone import ExecutionEngine, init_cli

SCRIPT_DIR = Path(__file__).resolve().parent
FILES_DIR = SCRIPT_DIR / "files"
LIGAND_NAME = "HCY"
LIGAND_SMILES = (FILES_DIR / "cortisol.smi").read_text().strip()
SMILES_KW = {"specified_smiles": {LIGAND_NAME: LIGAND_SMILES}, "input_mode": "json"}
N_CYCLES = 2

RFD3, MPNN, AF3, RemapLigand, HBond, Filter = Blocks.getall(
    "rfd3", "mpnn", "af3", "remap_ligand", "hbond_metrics", "filter"
)


def sequence_block():
    return MPNN(
        model_type="ligand_mpnn",
        num_seq_per_pdb=1,
        output_mode="structures",
        keep_motif_sequences_fixed=True,
        aromatic_fix=True,
        hbond_fix=True,
        args={"number_of_packs_per_design": 1},
    )


def prediction_blocks(*, final=False):
    name = "af3_final" if final else "af3"
    return [
        AF3(name=name, diffusion_batch_size=1, **SMILES_KW),
        RemapLigand(
            name=f"{name}_remap",
            reference_cif=str(FILES_DIR / "cortisol.cif"),
            target_name=LIGAND_NAME,
        ),
        HBond(name=f"{name}_hbond", verbose=True),
    ]


def run_pipeline(engine, *, cycles=N_CYCLES):
    if cycles < 1:
        raise ValueError("cycles must be at least 1")
    h = engine.run(
        RFD3(
            inputs=str(FILES_DIR / "cortisol.json"),
            n_batches=1,
            batch_size=2,
            jitter_ori_sigma=1.0,
        )
    )
    for i, cycle_engine in enumerate(engine.cycle("refine", cycles)):
        if not len(h):
            break
        # Explicit data preserves the surviving AF3 structure as the next
        # MPNN input. Namespaced keys prevent reuse of an earlier round's scores.
        h = cycle_engine.run([sequence_block(), *prediction_blocks()], data=h)
        af3 = f"refine/cycle_{i}/af3"
        hbonds = f"{af3}_hbond"
        write_scores(h, engine.rundir / "scores" / f"cycle_{i}", af3, hbonds)
        h = cycle_engine.run(
            Filter(name="cycle_filter", filters=gate_filters(af3, hbonds)), data=h
        )
        print(f"Cycle {i + 1}/{cycles}: {len(h)} pass", flush=True)

    # All final metrics come from a fresh final prediction. A failed cycle
    # cannot advance to either the next cycle or this final oracle.
    if len(h):
        h = engine.run(prediction_blocks(final=True), data=h)
    final_scores = write_scores(
        h,
        engine.rundir / "scores" / "final",
        "af3_final",
        "af3_final_hbond",
        final=True,
    )
    if len(h):
        h = engine.run(
            Filter(
                name="af3_final_filter",
                filters=gate_filters("af3_final", "af3_final_hbond", final=True),
            ),
            data=h,
        )
    # unpack() loses columns for an empty handler; retain a readable schema.
    results = h.unpack() if len(h) else final_scores.iloc[:0]
    results.to_csv(engine.rundir / "results.csv", index=False)
    print(engine)
    return h


def main():
    from absl import flags

    flags.DEFINE_integer(
        "cycles", N_CYCLES, "Number of gated MPNN/AF3 rounds", lower_bound=1
    )
    init_cli()
    engine = ExecutionEngine(rundir=SCRIPT_DIR / "rundir")
    for marker in ("driver.done", "driver.failed"):
        (engine.rundir / marker).unlink(missing_ok=True)
    try:
        run_pipeline(engine, cycles=flags.FLAGS.cycles)
    except Exception:
        (engine.rundir / "driver.failed").touch()
        raise
    (engine.rundir / "driver.done").touch()


if __name__ == "__main__":
    main()
