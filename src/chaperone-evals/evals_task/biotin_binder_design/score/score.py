#!/usr/bin/env python3
"""Held-out scorer for the biotin-binder evaluation.

Given a CSV of exactly 96 sequences, independently co-fold every design with
biotin using AF3, measure the holo interface, and report the pass rate. Agent
self-reported structural metrics are intentionally ignored. This file belongs
to the host-side task definition and is not mounted into the agent container.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from absl import flags

from blocks import Blocks
from blocks.util_blocks.select_diverse import _identity_matrix
from pipelines import ExecutionEngine, init_cli
from pipelines.data.handler import PipelineDataHandler
from pipelines.io import parse_to_atom_array
from pipelines.settings import get_setting_field


flags.DEFINE_string(
    "designs",
    None,
    "Agent submission: CSV with a `sequence` column and optional `design_id`.",
)
flags.DEFINE_string("out", "score.json", "Where to write the score report JSON.")
flags.DEFINE_string(
    "per_design_out",
    None,
    "Per-design metrics table. Defaults to score_per_design.csv beside --out.",
)
flags.DEFINE_integer(
    "required_n",
    None,
    "Submission size the contract enforces. Defaults to REQUIRED_N; override "
    "only to match a reduced test instruction, never to relax a real run.",
)
flags.DEFINE_string(
    "run_meta",
    None,
    "Optional cost JSON. Harbor supplies its controller-owned measurement; "
    "standalone scoring may pass an external record.",
)
flags.DEFINE_string(
    "pipelines_repo",
    os.environ.get("PIPELINES_REPO"),
    "Harness checkout. Passed by the controller; this scorer resolves its own "
    "assets and does not read it.",
)
FLAGS = flags.FLAGS

# The agent-facing count lives in the arm instruction (instruction_*.md), so a
# reduced test instruction can be swapped in without editing the task prompt.
# This is the scorer's own default; --required_n must track whichever
# instruction was served or the contract check will disagree with the prompt.
REQUIRED_N = 96
LIGAND_NAME = "B;N"
LIGAND_SMILES = "N1C(=O)N[C@@H]2[C@@H](SC[C@H]12)CCCCC(=O)NCCCCC"
CORE_ATOMS = [
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
EXPOSED_ATOMS = ["C14", "C15", "C16"]
BINDER_CHAINS = ["A"]
TARGET_CHAINS = ["B"]
PROBE_RADIUS = 6.0

EXPOSED_MIN_RASA = 0.1
AF3_MIN_PAE = 1.5
AF3_MIN_HBOND = 3
AF3_MIN_CONTACT_MS = 140
MIN_AROMATIC_NEIGHBORS = 1
MAX_SEQUENCE_IDENTITY = 0.70

GPU_ARGS = {
    "partition": "gpu",
    # The cluster's canonical GRES for the B4000-class small-GPU nodes. The
    # agent-facing guard performs the same b4000 -> small translation.
    "gres": "gpu:small:1",
    "mem_gb": 32,
    "time_limit": "3:00:00",
    # One GPU is sufficient for the normal 96-design contract. Split only
    # submissions larger than 200 rows; five GPUs is a ceiling, not a target.
    "rows_per_job": 200,
}
CPU_ARGS = {"partition": "local"}
# The submission is sequence-only, so the fold has to supply the ligand itself.
# specified_smiles cannot: it is a lookup keyed on ligand residues already
# present in the input (`if res_name in specified_smiles`), and a bare sequence
# has none — every design folded apo and failed has_ligand, pinning n_pass at 0
# no matter what was submitted. The template declares chain A protein (filled
# per design) plus chain B ligand, so AF3 co-folds biotin every time.
SCRIPT_DIR = Path(__file__).resolve().parent
HOLO_TEMPLATE = SCRIPT_DIR / "biotin_holo_template.json"
LIGAND_REFERENCE = SCRIPT_DIR / "biotin_reference.pdb"


def validate_submission(
    designs_path: str | Path,
    required_n: int,
) -> tuple[pd.DataFrame, float]:
    """Validate count, sequences, and pairwise identity."""
    designs = pd.read_csv(designs_path, dtype=str, keep_default_na=False)
    sequence_columns = {"sequence", "sequence_A"}
    allowed_columns = sequence_columns | {"design_id"}
    present_sequence_columns = sequence_columns & set(designs.columns)
    missing = [] if present_sequence_columns else ["sequence"]
    unexpected = sorted(set(designs.columns) - allowed_columns)
    ambiguous = len(present_sequence_columns) > 1
    if missing or unexpected or ambiguous:
        details = []
        if missing:
            details.append(f"missing required columns {missing}")
        if unexpected:
            details.append(f"unexpected columns {unexpected}")
        if ambiguous:
            details.append("supply either 'sequence' or legacy 'sequence_A', not both")
        raise ValueError(
            "; ".join(details)
            + f"; allowed columns are {sorted(allowed_columns)}"
        )
    if "sequence_A" in designs.columns:
        designs = designs.rename(columns={"sequence_A": "sequence"})
    if len(designs) != required_n:
        raise ValueError(f"expected {required_n} designs, got {len(designs)}")
    sequences = designs["sequence"].tolist()
    invalid = [
        i
        for i, sequence in enumerate(sequences)
        if not re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+", sequence)
    ]
    if invalid:
        raise ValueError(f"invalid or empty protein sequence rows: {invalid[:10]}")
    if len(set(sequences)) != len(sequences):
        raise ValueError("submitted protein sequences must be unique")
    if "design_id" in designs.columns:
        design_ids = designs["design_id"].tolist()
        invalid_ids = [
            i
            for i, design_id in enumerate(design_ids)
            if not design_id or design_id != design_id.strip()
        ]
        if invalid_ids:
            raise ValueError(
                "design_id values must be non-empty with no surrounding "
                f"whitespace; invalid rows: {invalid_ids[:10]}"
            )
        if len(set(design_ids)) != len(design_ids):
            raise ValueError("design_id values must be unique")

    identity = _identity_matrix(sequences)
    max_identity = max(
        (
            identity[i][j]
            for i in range(len(identity))
            for j in range(i + 1, len(identity))
        ),
        default=0.0,
    )
    if max_identity > MAX_SEQUENCE_IDENTITY:
        raise ValueError(
            "pairwise global sequence identity exceeds 0.70: "
            f"observed {max_identity:.6f}"
        )

    return designs, float(max_identity)


def evaluate_physical_gates(
    frame: pd.DataFrame, has_ligand: pd.Series
) -> dict[str, object]:
    """Evaluate exact held-out gate boundaries; missing values always fail."""

    def numeric_column(name: str, missing: float = 0.0) -> pd.Series:
        if name not in frame.columns:
            return pd.Series(
                np.full(len(frame), missing), index=frame.index, dtype=float
            )
        return pd.to_numeric(frame[name], errors="coerce").fillna(missing)

    exposed_columns = [f"holo_rasa.ligand_atom_rasa.{a}" for a in EXPOSED_ATOMS]
    exposed_rasa = pd.DataFrame(
        {column: numeric_column(column) for column in exposed_columns}
    )
    gates = {
        "linker_exposed": (exposed_rasa >= EXPOSED_MIN_RASA).any(axis=1),
        "low_pae": numeric_column("holo.min_pae_interaction", np.inf) <= AF3_MIN_PAE,
        "enough_hbonds": numeric_column("holo_hbond.num_hbonds") >= AF3_MIN_HBOND,
        "enough_contact": numeric_column("holo_contact_ms") >= AF3_MIN_CONTACT_MS,
        "enough_aromatics": numeric_column("holo_aromatic.num_aromatic_neighbors")
        >= MIN_AROMATIC_NEIGHBORS,
    }
    gates["passes"] = has_ligand.copy()
    for name, value in gates.items():
        if name != "passes":
            gates["passes"] &= value
    gates["exposed_rasa"] = exposed_rasa
    return gates


def load_sequences(source: pd.DataFrame) -> PipelineDataHandler:
    """Load submitted sequences into a handler ready for holo co-folding."""
    handler = PipelineDataHandler.from_sequences(source)
    if len(handler) == 0:
        raise ValueError("No sequences loaded from the validated submission.")
    return handler


def _design_ids(frame) -> list:
    """Stable per-row identifier, preferring one the submission supplied."""
    for candidate in ("design_id", "name", "holo.name"):
        if candidate in frame.columns:
            return list(frame[candidate])
    return [f"design_{i:04d}" for i in range(len(frame))]


def structure_has_ligand(path: str) -> bool:
    """Return whether the folded structure contains at least one hetero atom."""
    try:
        atom_array = parse_to_atom_array(path)
    except Exception:
        return False
    return bool(
        getattr(atom_array, "hetero", np.zeros(len(atom_array), dtype=bool)).any()
    )


def main() -> None:
    init_cli()
    if not FLAGS.designs:
        raise SystemExit("--designs is required (a CSV with a `sequence` column)")

    out_path = Path(FLAGS.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rundir = (
        Path(FLAGS.rundir).expanduser().resolve()
        if FLAGS.rundir
        else out_path.parent / "score_run"
    )
    if not LIGAND_REFERENCE.is_file():
        raise SystemExit(f"Ligand reference not found: {LIGAND_REFERENCE}")
    ligand_cif = str(LIGAND_REFERENCE)

    required_n = FLAGS.required_n or REQUIRED_N
    try:
        submission_frame, max_sequence_identity = validate_submission(
            FLAGS.designs, required_n
        )
    except (OSError, ValueError) as error:
        try:
            n_reported = len(pd.read_csv(FLAGS.designs))
        except Exception:
            n_reported = 0
        report = {
            "eval": "biotin-binder",
            "n_reported": int(n_reported),
            "required_n": required_n,
            "contract_ok": False,
            "n_pass": None,
            "pass_rate": None,
            "contract_error": str(error),
        }
        out_path.write_text(json.dumps(report, indent=2) + "\n")
        raise SystemExit(
            f"Contract violation: {error}. Wrote {out_path}; no held-out "
            "compute was run."
        ) from error

    # Construct the engine before loading sequences so settings.RUNDIR is
    # available to PipelineDataHandler.from_file(..., allow_sequences=True).
    engine = ExecutionEngine(rundir=rundir, reinit=False)
    _ = get_setting_field("RUNDIR")

    handler = load_sequences(submission_frame)
    n_reported = len(submission_frame)
    contract_ok = True
    print(f"Loaded and validated {n_reported} sequences from {FLAGS.designs}")

    AF3, ContactMS, RASA, HBond, Aromatic, RemapLigand = Blocks.getall(
        "af3",
        "contact_ms",
        "rasa",
        "hbond_metrics",
        "aromatic_neighbors",
        "remap_ligand",
    )

    holo = engine.run(
        AF3(
            name="holo",
            template_json=str(HOLO_TEMPLATE),
            mean_pae_core_atoms=CORE_ATOMS,
        ),
        data=handler,
        job_args=GPU_ARGS,
    )
    holo = engine.run(
        [
            RemapLigand(
                name="remap",
                reference_cif=ligand_cif,
                target_name=LIGAND_NAME,
                core_atoms=CORE_ATOMS,
            ),
            HBond(name="holo_hbond"),
            RASA(name="holo_rasa", probe_radius=PROBE_RADIUS, backbone_only=True),
            ContactMS(
                name="holo_contact_ms",
                binder_chains=BINDER_CHAINS,
                target_chains=TARGET_CHAINS,
                target_atom_names=CORE_ATOMS,
            ),
            Aromatic(
                name="holo_aromatic",
                distance_threshold=6.0,
                ligand_atom_names=CORE_ATOMS,
            ),
        ],
        data=holo,
        job_args=CPU_ARGS,
    )

    frame = holo.unpack(all=True)
    harness_faults = []
    if len(frame) != n_reported:
        harness_faults.append(
            f"Held-out pipeline returned {len(frame)} rows for {n_reported} "
            "submitted designs. Worker failures or dropped chunks make this "
            "run invalid; missing Parquet is not a scientific gate result."
        )
    if "holo.path" in frame.columns:
        has_ligand = frame["holo.path"].map(structure_has_ligand)
    else:
        has_ligand = pd.Series(False, index=frame.index, dtype=bool)
    gate_values = evaluate_physical_gates(frame, has_ligand)
    linker_exposed = gate_values["linker_exposed"]
    low_pae = gate_values["low_pae"]
    enough_hbonds = gate_values["enough_hbonds"]
    enough_contact = gate_values["enough_contact"]
    enough_aromatics = gate_values["enough_aromatics"]
    exposed_rasa = gate_values["exposed_rasa"]
    passes = gate_values["passes"]
    n_pass = int(passes.sum())
    n_ligandless = int((~has_ligand).sum())
    pass_rate = n_pass / required_n

    # A holo co-fold that produced no ligand anywhere is a harness fault, not a
    # verdict on the submission: every remaining gate reads ligand geometry, so
    # the report collapses to a clean-looking 0/96 that is indistinguishable
    # from a genuinely hard benchmark. This exact failure (an apo fold from a
    # sequence-only submission) went unnoticed until the AF3 inputs were read
    # by hand. Say so loudly and mark the report invalid rather than scoring it.
    if len(frame) and n_ligandless == len(frame):
        harness_faults.append(
            "No folded structure contains the ligand. The holo co-fold did not "
            "supply it, so every ligand-dependent gate fails by construction "
            "and this score says nothing about the submission."
        )
    harness_fault = " ".join(harness_faults) or None
    if harness_fault:
        print(f"ERROR: {harness_fault}")

    per_criterion = {
        "has_ligand": int(has_ligand.sum()),
        "linker_exposed": int(linker_exposed.sum()),
        f"min_pae<={AF3_MIN_PAE:g}": int(low_pae.sum()),
        f"num_hbonds>={AF3_MIN_HBOND}": int(enough_hbonds.sum()),
        f"contact_ms>={AF3_MIN_CONTACT_MS}": int(enough_contact.sum()),
        f"num_aromatics>={MIN_AROMATIC_NEIGHBORS}": int(enough_aromatics.sum()),
        f"pairwise_identity<={MAX_SEQUENCE_IDENTITY}": n_reported,
    }

    status = "INVALID" if harness_fault else "PASS"
    print(
        f"[{status}] {n_pass} / {n_reported} (pass_rate={pass_rate:.3f}, "
        f"{n_ligandless} missing ligand)"
    )
    print("[breakdown] " + "  ".join(f"{k}={v}" for k, v in per_criterion.items()))

    cost = efficiency = None
    if FLAGS.run_meta:
        cost = json.loads(Path(FLAGS.run_meta).read_text())
        gpu_hours = cost.get("gpu_hours") or 0.0
        cpu_hours = cost.get("cpu_hours") or 0.0
        efficiency = {
            "pass_per_gpu_hour": n_pass / gpu_hours if gpu_hours else None,
            "pass_per_cpu_hour": n_pass / cpu_hours if cpu_hours else None,
        }
    else:
        print("WARNING: no --run_meta; pass-rate-per-cost unavailable.")

    # Every gate is a threshold over a continuous quantity, but only the
    # booleans were ever persisted — so a run that scores 0/96 looked identical
    # whether it missed by 0.01 or by an order of magnitude, and thresholds
    # could not be recalibrated without recomputing the whole sweep. Persist the
    # underlying values per design, plus the gate outcomes, so future
    # calibration is a table read rather than a rerun.
    per_design = pd.DataFrame({"design_id": _design_ids(frame)})
    for column in frame.columns:
        if column.startswith(("holo.", "holo_", "remap.")) and column not in (
            "holo.path",
        ):
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.notna().any():
                per_design[column] = values.to_numpy()
    per_design["sequence_length"] = (
        pd.to_numeric(frame.get("sequence_length"), errors="coerce").to_numpy()
        if "sequence_length" in frame.columns
        else np.nan
    )
    # Aggregates the gates actually read, so a calibration sweep needs no
    # knowledge of which per-atom columns make up "the core".
    per_design["linker_max_rasa"] = exposed_rasa.max(axis=1).to_numpy()
    per_design["submission_max_pairwise_identity"] = max_sequence_identity
    for name, series in (
        ("gate_has_ligand", has_ligand),
        ("gate_linker_exposed", linker_exposed),
        ("gate_low_pae", low_pae),
        ("gate_hbonds", enough_hbonds),
        ("gate_contact_ms", enough_contact),
        ("gate_aromatic_neighbors", enough_aromatics),
        ("passes_all", passes),
    ):
        per_design[name] = np.asarray(series).astype(int)

    per_design_path = Path(
        FLAGS.per_design_out or out_path.parent / "score_per_design.csv"
    ).expanduser()
    per_design_path.parent.mkdir(parents=True, exist_ok=True)
    per_design.to_csv(per_design_path, index=False)
    print(
        f"Wrote per-design metrics: {per_design_path} ({per_design.shape[1]} columns)"
    )

    report = {
        "eval": "biotin-binder",
        "n_reported": int(n_reported),
        "required_n": required_n,
        "contract_ok": contract_ok,
        "n_pass": None if harness_fault else n_pass,
        "pass_rate": None if harness_fault else pass_rate,
        "n_ligandless": n_ligandless,
        "n_incomplete": n_ligandless + max(0, n_reported - len(frame)),
        "per_design": str(per_design_path),
        # Non-null means the run is void; do not aggregate it with real scores.
        "harness_fault": harness_fault,
        "per_criterion": per_criterion,
        "cost": cost,
        "efficiency": efficiency,
        "filter_thresholds": {
            "exposed_min_rasa": EXPOSED_MIN_RASA,
            "core_atoms": CORE_ATOMS,
            "exposed_atoms": EXPOSED_ATOMS,
            "min_pae_interaction": AF3_MIN_PAE,
            "num_hbonds": AF3_MIN_HBOND,
            "contact_ms": AF3_MIN_CONTACT_MS,
            "min_aromatic_neighbors": MIN_AROMATIC_NEIGHBORS,
            "aromatic_distance_angstrom": 6.0,
            "max_sequence_identity": MAX_SEQUENCE_IDENTITY,
        },
        "validation_oracle": "af3",
    }
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if harness_fault:
        print(f"\nWrote invalid harness report {out_path}; no score is reportable.")
    else:
        print(
            f"\nWrote {out_path} — n_pass={n_pass} / {n_reported} "
            f"(pass_rate={pass_rate:.3f})"
        )


if __name__ == "__main__":
    main()
