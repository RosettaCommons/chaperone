#!/usr/bin/env -S bash -c '"${CHAPERONE_REPO:-.}/.venv/bin/python" "$0" "$@"'
"""Reference antibody-CDR submission: MPNN CDR redesign on the D1.3 Fv.

The supplied framework is the point of this benchmark, so the reference
implementation is an *inverse-folding* baseline rather than a de novo
generator: the D1.3 Fv/lysozyme complex is rebuilt from the committed inputs,
every framework and antigen position is fixed, and MPNN redesigns only the 54
AbM CDR positions.

Why not RFdiffusion3? The antibody RFD3 stack in ``examples/antibody-design``
needs an external modelhub checkout, an antibody-finetuned checkpoint and the
abMPNN-PoE container, none of which ship with this repository — the committed
``files/rfd3_config.yaml`` there is an unfilled placeholder. A reference run
must be reproducible from the repo alone, so this pipeline uses only blocks
that are.

Deliberately no oracle-based selection: the submission is the raw MPNN sample,
so the committed ``scored/`` output calibrates the thresholds instead of being
pre-filtered by the very metrics it is meant to calibrate.

    # smoke test (1 design)
    ./example_pipeline.py --required_n=1 --n_samples=2 \
        --rundir=/tmp/antibody_cdr_smoke --nouse_local_gpus

    # production
    ./example_pipeline.py --nouse_local_gpus
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import polars as pl
from absl import flags

from blocks import Blocks
from chaperone import ExecutionEngine, init_cli
from chaperone.data.handler import PipelineDataHandler
from chaperone.data.schema import make_row
from scoring import FixedRegionSpec, VariableRegion
from chaperone.io import parse_to_atom_array

flags.DEFINE_integer(
    "required_n", 96, "Sequences to submit. Override to 1 only for a smoke test."
)
flags.DEFINE_integer(
    "n_samples", 160, "MPNN samples drawn before de-duplicating down to required_n."
)
flags.DEFINE_list(
    "design_cdrs",
    ["CDR-H1", "CDR-H2", "CDR-H3", "CDR-L1", "CDR-L2", "CDR-L3"],
    "Which declared CDR spans this reference redesigns. Defaults to all six, "
    "which is what produced the committed submission.csv. Narrowing it to the "
    "contact-dominant loops (CDR-H3,CDR-H2,CDR-L3) improves the interface but "
    "does not clear the gate - see score/readme.md.",
)
FLAGS = flags.FLAGS

init_cli()

SCRIPT_DIR = Path(__file__).resolve().parent
INPUTS_DIR = SCRIPT_DIR.parent / "inputs"
FRAMEWORK_PDB = (INPUTS_DIR / "framework.pdb").resolve()
TARGET_PDB = (INPUTS_DIR / "target.pdb").resolve()
CONFIG = json.loads((INPUTS_DIR / "config.json").read_text())

ANTIGEN_CHAIN = CONFIG["antigen"]["chain"]
REFERENCE = {
    "H": CONFIG["framework"]["sequence_H"],
    "L": CONFIG["framework"]["sequence_L"],
}
FRAMEWORK = FixedRegionSpec(
    reference=REFERENCE,
    regions=tuple(
        VariableRegion(
            name=name,
            chain=span["chain"],
            start=span["start"],
            end=span["end"],
            min_length=span.get("min_length"),
            max_length=span.get("max_length"),
        )
        for name, span in CONFIG["cdr_spans"].items()
    ),
)

GPU_ARGS = {
    "partition": "gpu",
    "gres": "gpu:small:1",
    "mem_gb": 32,
    "time_limit": "4:00:00",
    "rows_per_job": 1,
}


def build_complex(output_path: Path) -> Path:
    """The Fv/antigen complex MPNN designs on, rebuilt from committed inputs."""
    framework = parse_to_atom_array(str(FRAMEWORK_PDB))
    antigen = parse_to_atom_array(str(TARGET_PDB))
    from atomworks.io.utils.io_utils import to_cif_file

    to_cif_file(framework + antigen, str(output_path))
    return output_path


def fixed_residue_ids() -> list[str]:
    """Every residue MPNN must not touch: all framework, all antigen."""
    chosen = set(FLAGS.design_cdrs)
    unknown = chosen - set(CONFIG["cdr_spans"])
    if unknown:
        raise ValueError(f"--design_cdrs names unknown spans {sorted(unknown)}")
    designable = {
        f"{span['chain']}{index}"
        for name, span in CONFIG["cdr_spans"].items()
        if name in chosen
        for index in range(span["start"], span["end"] + 1)
    }
    fixed = [
        f"{chain}{index}"
        for chain, sequence in REFERENCE.items()
        for index in range(1, len(sequence) + 1)
        if f"{chain}{index}" not in designable
    ]
    fixed += [
        f"{ANTIGEN_CHAIN}{index}"
        for index in range(1, len(CONFIG["antigen"]["sequence"]) + 1)
    ]
    return fixed


engine = ExecutionEngine(rundir=SCRIPT_DIR / "rundir", reinit=True)
engine.rundir.mkdir(parents=True, exist_ok=True)

complex_path = build_complex(engine.rundir / "d1p3_fv_lysozyme.cif")
fixed = fixed_residue_ids()
print(
    f"Redesigning {sorted(FLAGS.design_cdrs)}: {223 + 129 - len(fixed)} positions; "
    f"fixing {len(fixed)}"
)

template = PipelineDataHandler(
    [
        make_row(
            name="template",
            path=complex_path,
            metadata={"motif": fixed},
            metrics={"design_id": "d1p3_template"},
        )
    ]
)

MPNN = Blocks.get("mpnn")
# ``num_seq_per_pdb`` becomes MPNN's ``--batch_size``, and side-chain packing a
# 352-residue complex costs ~0.1 GB of VRAM per sequence in the batch: sampling
# all 160 in one forward pass OOMs a 16 GB card. Draw them in small batches.
batch_size = min(8, FLAGS.n_samples)
n_batches = -(-FLAGS.n_samples // batch_size)
designs = engine.run(
    MPNN(
        name="cdr_mpnn",
        model_type="protein_mpnn",
        num_seq_per_pdb=batch_size,
        keep_motif_sequences_fixed=True,
        args={
            "temperature": 0.3,
            "omit_AA": "XC",
            "seed": 42,
            "number_of_batches": n_batches,
        },
    ),
    data=template,
    job_args=GPU_ARGS,
)

records: list[dict] = []
seen: set[tuple[str, str]] = set()
for index, loaded in enumerate(designs.get_data(which="last", load=True)):
    sequences = loaded.sequence or {}
    pair = (sequences.get("H"), sequences.get("L"))
    if not pair[0] or not pair[1]:
        raise ValueError(f"MPNN design {index} is missing an H or L sequence")
    # Fail loudly rather than silently submitting a broken framework.
    FRAMEWORK.check({"H": pair[0], "L": pair[1]})
    if pair in seen:
        continue
    seen.add(pair)
    records.append(
        {
            "design_id": f"d1p3_cdr_{len(records):04d}",
            "sequence_H": pair[0],
            "sequence_L": pair[1],
        }
    )

if len(records) < FLAGS.required_n:
    raise RuntimeError(
        f"Only {len(records)} unique designs from {FLAGS.n_samples} MPNN samples; "
        f"need {FLAGS.required_n}. Raise --n_samples or the sampling temperature."
    )
submission = pl.DataFrame(records[: FLAGS.required_n])

out_path = SCRIPT_DIR / "submission.csv"
submission.write_csv(out_path)
subprocess.run(["chmod", "-R", "g+w", str(engine.rundir)], check=False)
subprocess.run(["chmod", "g+w", str(out_path)], check=False)
print(engine)
print(submission)
print(f"\nWrote {len(submission)} D1.3-framework CDR designs to {out_path}")
