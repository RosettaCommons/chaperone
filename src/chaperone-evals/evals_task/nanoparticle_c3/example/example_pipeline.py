#!/usr/bin/env -S bash -c '"${CHAPERONE_REPO:-.}/.venv/bin/python" "$0" "$@"'
"""Reference pipeline for the nanoparticle-c3 eval: 96 C3 protomer sequences.

    rfd1 (C3 symmetric)  ->  mpnn (tied homo-oligomer)  ->  af3  ->  rank

Route note — this uses **RFdiffusion v1**, not RFD3, because RFD3 has no
symmetry mode: generating a genuinely C3 backbone is the whole point here, and
`rfd1` takes `symmetry="c3"` directly. It deliberately does *not* reuse
`examples/nanoparticle-design`'s `SymmetrizeBlock` / `RPXDockBlock`: those exist
to convert the trimer to poly-alanine and dock it into an I3 icosahedron, which
throws away the very sequence-designable trimer this benchmark scores. RFD1's
symmetric output is already exactly C3 (the `symmetry` block scores a raw rfd1
trimer at ~1e-4 degrees of angle deviation), so the "idealize" half of
`SymmetrizeBlock` has nothing to do either.

The submission is the *protomer* only: MPNN ties every position across the three
chains (see `tied_positions`), so chain A's sequence is the whole design. The
eval contract replicates it back into A/B/C at scoring time.

Ranking is by AF3 assembly quality (ipTM, then pLDDT). The eval contract is a
fixed count (96), not "however many were good", so the table is always filled
to 96 and the printed summary reports the ipTM of the worst row kept — a thin
run is never mistaken for a strong one.

    # single-design smoke test
    ./example_pipeline.py --test --required_n=1 --nouse_local_gpus

    # production
    ./example_pipeline.py --nouse_local_gpus
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
from absl import flags

from blocks import Blocks
from chaperone import ExecutionEngine, init_cli

SCRIPT_DIR = Path(__file__).resolve().parent
INPUTS_DIR = SCRIPT_DIR.parent / "inputs"
CONFIG = json.loads((INPUTS_DIR / "config.json").read_text())
SUBMISSION = SCRIPT_DIR / "submission.csv"

flags.DEFINE_integer(
    "required_n", 96, "Number of protomer sequences to write to submission.csv."
)

init_cli()
FLAGS = flags.FLAGS

POINT_GROUP = CONFIG["point_group"]
N_COPIES = CONFIG["n_copies"]
PROTOMER_LENGTH = CONFIG["protomer_length"]

# RFdiffusion v1 symmetric contigs take the TOTAL assembly length, which must be
# divisible by the symmetry order. Deriving it from the protomer length is the
# only place this factor of N should ever appear.
TOTAL_LENGTH = PROTOMER_LENGTH * N_COPIES
CONTIGS = f"{TOTAL_LENGTH}-{TOTAL_LENGTH}"

# One SLURM GPU for the whole run: rfd1 and af3 both fit in one chunk, so
# rows_per_job is set above the row count rather than left to fan out.
BACKBONE_GPU = {
    "partition": "gpu",
    "gres": "gpu:small:1",
    "mem_gb": 32,
    "time_limit": "8:00:00",
    "rows_per_job": 1000,
}
# A C3 trimer is a 3x-larger fold than a mini-binder, so the oracle gets the
# large-GPU allocation.
ORACLE_GPU = {
    "partition": "gpu",
    "gres": "gpu:large:1",
    "mem_gb": 64,
    "time_limit": "8:00:00",
    "rows_per_job": 1000,
}

RFD1, MPNN, AF3 = Blocks.getall("rfd1", "mpnn", "af3")

rfd1 = RFD1(
    contigs=CONTIGS,
    symmetry=CONFIG["rfd1"]["symmetry"],
    num_designs=CONFIG["rfd1"]["num_designs"],
    diffuser_T=CONFIG["rfd1"]["diffuser_T"],
)
af3 = AF3(diffusion_batch_size=1)


def tied_positions(layout: dict[str, tuple[int, int]]) -> tuple[str, str]:
    """MPNN ``symmetry_residues`` / ``symmetry_weights`` tying chains position-wise.

    ``{"A": (1, 70), "B": (71, 140), ...}`` becomes
    ``"A1,B71,C141|A2,B72,C142|..."`` with equal weights, so MPNN designs one
    sequence shared by all three chains.

    This is written out explicitly rather than using MPNN's ``homo_oligomer=1``
    shortcut, so the tied positions come from the generated backbones' own chain
    and residue numbering. Explicit tying takes the same code path in MPNN's
    feature dict.
    """
    chains = sorted(layout)
    lengths = {chain: layout[chain][1] - layout[chain][0] + 1 for chain in chains}
    if len(set(lengths.values())) != 1:
        raise RuntimeError(f"Protomers differ in length, cannot tie: {lengths}")
    groups = [
        ",".join(f"{chain}{layout[chain][0] + offset}" for chain in chains)
        for offset in range(next(iter(lengths.values())))
    ]
    weights = "|".join(",".join("1.0" for _ in chains) for _ in groups)
    return "|".join(groups), weights


def backbone_layout(handler) -> dict[str, tuple[int, int]]:
    """``{chain: (first_resid, last_resid)}`` of the generated backbones.

    Read from the structures rather than assumed, and required to be identical
    across every backbone: one ``symmetry_residues`` string is applied to the
    whole chunk, so a design whose numbering differs would be tied at the wrong
    positions without ever erroring.
    """
    import numpy as np

    layouts = set()
    for loaded in handler.get_data(which="last", load=True):
        atom_array = loaded.atom_array
        layouts.add(
            tuple(
                (
                    str(chain),
                    int(atom_array.res_id[atom_array.chain_id == chain].min()),
                    int(atom_array.res_id[atom_array.chain_id == chain].max()),
                )
                for chain in sorted(np.unique(atom_array.chain_id).tolist())
            )
        )
    if len(layouts) != 1:
        raise RuntimeError(
            f"Backbones do not share one chain/residue layout: {sorted(layouts)}"
        )
    layout = {chain: (first, last) for chain, first, last in next(iter(layouts))}
    if len(layout) != N_COPIES:
        raise RuntimeError(
            f"{POINT_GROUP} expects {N_COPIES} chains, backbones have {sorted(layout)}"
        )
    return layout


def main() -> None:
    engine = ExecutionEngine(rundir=SCRIPT_DIR / "rundir", reinit=False)
    handler = engine.run(rfd1, job_args=BACKBONE_GPU)

    symmetry_residues, symmetry_weights = tied_positions(backbone_layout(handler))
    handler = engine.run(
        MPNN(
            model_type=CONFIG["mpnn"]["model_type"],
            num_seq_per_pdb=CONFIG["mpnn"]["num_seq_per_pdb"],
            args={
                "symmetry_residues": symmetry_residues,
                "symmetry_weights": symmetry_weights,
                "omit_AA": CONFIG["mpnn"]["omit_AA"],
            },
        ),
        data=handler,
    )
    handler = engine.run(af3, data=handler, job_args=ORACLE_GPU)

    frame = handler.unpack(all=True)
    required_n = FLAGS.required_n

    # Chain A is the whole design: MPNN tied every position across A/B/C, so the
    # three chains of each prediction are identical. That is checked, not
    # assumed — a silently untied run would submit one third of a design.
    sequences = []
    for index, loaded in enumerate(handler.get_data(which="last", load=True)):
        chains = {
            chain: loaded.sequence.get(chain) for chain in ("A", "B", "C")[:N_COPIES]
        }
        if len(set(chains.values())) != 1 or not chains["A"]:
            raise RuntimeError(
                f"Row {index}: the {N_COPIES} chains are not identical, so MPNN did "
                f"not tie them. Chain lengths: "
                f"{ {c: len(s or '') for c, s in chains.items()} }"
            )
        sequences.append(chains["A"])
    if len(sequences) != len(frame):
        raise RuntimeError(
            f"Recovered {len(sequences)} chain-A sequences for {len(frame)} "
            "predictions; refusing to guess which rows they belong to"
        )
    frame = frame.assign(protomer_sequence=sequences)

    rank_by = [("af3.iptm", False), ("af3.plddt", False)]
    columns = [(c, asc) for c, asc in rank_by if c in frame.columns]
    if not columns:
        raise RuntimeError(
            f"AF3 produced none of {[c for c, _ in rank_by]}; cannot rank designs. "
            f"Columns present: {sorted(frame.columns)}"
        )
    ranked = frame.sort_values(
        by=[c for c, _ in columns], ascending=[asc for _, asc in columns]
    )

    records: list[dict] = []
    seen: set[str] = set()
    for _, row in ranked.iterrows():
        if len(records) >= required_n:
            break
        sequence = row["protomer_sequence"]
        if sequence in seen:
            continue
        seen.add(sequence)
        records.append(
            {
                "design_id": f"c3_{len(records):04d}",
                "sequence": sequence,
                "_iptm": float(row["af3.iptm"]),
            }
        )

    if len(records) < required_n:
        raise RuntimeError(
            f"Only {len(records)} unique protomer sequences available from "
            f"{len(frame)} predictions; need {required_n}. Scale the run up with "
            "--rfd1.num_designs / --mpnn.num_seq_per_pdb."
        )

    worst_iptm = min(record.pop("_iptm") for record in records)
    submission = pl.DataFrame(records)
    if submission["sequence"].n_unique() != len(submission):
        raise ValueError("Selected submission contains duplicate sequences")
    submission.write_csv(SUBMISSION)

    lengths = submission["sequence"].str.len_chars()
    print(submission)
    print(
        f"\nWrote {len(submission)} {POINT_GROUP} protomer sequences to {SUBMISSION}\n"
        f"  protomer length {lengths.min()}-{lengths.max()} residues\n"
        f"  worst ipTM kept: {worst_iptm:.3f} "
        f"(of {len(frame)} predicted assemblies)"
    )


if __name__ == "__main__":
    main()
