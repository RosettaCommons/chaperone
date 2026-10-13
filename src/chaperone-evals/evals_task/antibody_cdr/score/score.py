#!/usr/bin/env -S bash -c '"${CHAPERONE_REPO:-.}/.venv/bin/python" "$0" "$@"'
"""Held-out scorer for antibody CDR designs on the supplied D1.3 framework.

The framework is *given*, so preserving it is a contract obligation, not a
metric: :class:`~pipelines.evals.FixedRegionSpec` rejects any submission whose
sequence does not decompose as ``framework + CDR + framework + …`` before a
single GPU is touched. CDR-H3 length is part of that same contract, and is
enforced by ``run_sequence_scorer`` via the ``spec`` below.

What is then measured, on an independent AF3 co-fold of H + L with the
scorer-supplied antigen chain T:

  interface confidence  — paratope↔antigen PAE and ipSAE
  fold confidence       — complex pLDDT
  framework in 3D       — backbone RMSD of the fixed positions vs the reference
  packing / sterics     — contact molecular surface and backbone clashes
  developability (soft) — CDR glycosylation sequons, unpaired Cys, net charge,
                          hydrophobic runs in CDR-H3

Usage:
    ./score.py --designs=/absolute/path/to/submission.csv --nouse_local_gpus
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from absl import flags

from blocks import Blocks
from chaperone import init_cli
from scoring import (
    FixedRegionSpec,
    SequenceSubmissionSpec,
    VariableRegion,
    define_score_flags,
    load_context_chains,
    merge_submission_columns,
    metric_series,
    run_sequence_scorer,
)
from chaperone.io import parse_to_atom_array

define_score_flags()
FLAGS = flags.FLAGS

init_cli()

EVAL_NAME = "antibody-cdr"
ORACLE = "af3_with_target_template"

SCRIPT_DIR = Path(__file__).resolve().parent
INPUTS_DIR = SCRIPT_DIR.parent / "inputs"
TARGET_PDB = (INPUTS_DIR / "target.pdb").resolve()
FRAMEWORK_PDB = (INPUTS_DIR / "framework.pdb").resolve()
CONFIG = json.loads((INPUTS_DIR / "config.json").read_text())

HEAVY_CHAIN = "H"
LIGHT_CHAIN = "L"
ANTIGEN_CHAIN = CONFIG["antigen"]["chain"]
ANTIBODY_CHAINS = (HEAVY_CHAIN, LIGHT_CHAIN)

FRAMEWORK = FixedRegionSpec(
    reference={
        HEAVY_CHAIN: CONFIG["framework"]["sequence_H"],
        LIGHT_CHAIN: CONFIG["framework"]["sequence_L"],
    },
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

# ── Thresholds ───────────────────────────────────────────────────────────────
# Deviations from evals/PLAN.md's starting spec are documented in readme.md.

MIN_PAE_INTERFACE_MAX = 8.0  # paratope↔antigen, NOT whole-complex min PAE
# ipSAE masks token pairs at or above its cutoff. At the block default of
# 10 A the *native* D1.3/lysozyme complex scores 0.012 — saturated at zero —
# so no antibody can clear a 0.30 gate. At 15 A the native scores 0.114 while
# every sampled CDR redesign scores exactly 0.000. See readme.md.
IPSAE_PAE_CUTOFF = 15.0
IPSAE_MIN = 0.05
PLDDT_MIN = 80.0
FRAMEWORK_RMSD_MAX = 1.5
CONTACT_MS_MIN = 300.0
N_BB_CLASHES_MAX = 0

NET_CHARGE_RANGE = (-5.0, 5.0)
MAX_UNPAIRED_CYS = 1
MAX_HYDROPHOBIC_RUN_H3 = 4

HYDROPHOBIC = set("ACFILMVWY")
BACKBONE_ATOMS = ("N", "CA", "C", "O")

MIN_PAE_COL = "ipsae.min_pae_interface"
IPSAE_COL = "ipsae.ipsae"
IPSAE_MAX_COL = "ipsae.ipsae_max"
PLDDT_COL = "fold.plddt"
CONTACT_MS_COL = "contact_ms"
CLASH_COL = "clash.n_bb_clashes"
FV_PLDDT_COL = "fv.mean_plddt"
AF3_MIN_PAE_COL = "fold.min_pae_interaction"

GPU_ARGS = {
    "partition": "gpu",
    "gres": "gpu:small:1",
    "mem_gb": 32,
    "time_limit": "8:00:00",
    "rows_per_job": 128,
}
CPU_ARGS = {"partition": "local"}


# ── Framework preservation in 3D ─────────────────────────────────────────────


def _backbone_index(atom_array) -> dict[tuple[str, int, str], int]:
    keep = np.isin(atom_array.atom_name, BACKBONE_ATOMS)
    return {
        (str(atom_array.chain_id[i]), int(atom_array.res_id[i]),
         str(atom_array.atom_name[i])): int(i)
        for i in np.flatnonzero(keep)
    }  # fmt: skip


_REFERENCE = parse_to_atom_array(str(FRAMEWORK_PDB))
_REFERENCE_INDEX = _backbone_index(_REFERENCE)


def framework_backbone_rmsd(structure_path: str, sequences: dict[str, str]) -> float:
    """Backbone RMSD of the fixed positions against the reference Fv.

    A sequence-identical framework that refolds elsewhere is not a preserved
    framework, so the contract check is not sufficient on its own.
    ``FixedRegionSpec.fixed_positions`` supplies the design↔reference residue
    correspondence, which stays correct even when CDR-H3 changed length and
    shifted the design's numbering.
    """
    design = parse_to_atom_array(structure_path)
    design_index = _backbone_index(design)

    design_indices: list[int] = []
    reference_indices: list[int] = []
    for chain in ANTIBODY_CHAINS:
        for design_res, reference_res in FRAMEWORK.fixed_positions(
            chain, sequences[chain]
        ):
            for atom in BACKBONE_ATOMS:
                d = design_index.get((chain, design_res, atom))
                r = _REFERENCE_INDEX.get((chain, reference_res, atom))
                if d is not None and r is not None:
                    design_indices.append(d)
                    reference_indices.append(r)

    if len(design_indices) < 4 * 50:
        raise ValueError(
            f"Only {len(design_indices)} framework backbone atoms could be paired "
            f"for {structure_path}; the prediction is missing chains H/L or their "
            "residue numbering does not match the submitted sequences"
        )

    from biotite.structure import superimpose

    reference_atoms = _REFERENCE[reference_indices]
    fitted, _ = superimpose(reference_atoms, design[design_indices])
    deviation = fitted.coord - reference_atoms.coord
    return float(np.sqrt((deviation**2).sum(axis=1).mean()))


# ── Developability liabilities ───────────────────────────────────────────────


def developability(sequences: dict[str, str]) -> dict[str, float]:
    """Sequence liabilities that sink an otherwise good binder downstream.

    Every check is scoped to what the submitter actually controls: the CDRs.
    Framework sequons, framework cysteines and framework charge are supplied by
    the benchmark and are not the design's fault.
    """
    spans = FRAMEWORK.region_spans(sequences)
    by_name = {region.name: region for region in FRAMEWORK.regions}
    cdr_positions: dict[str, set[int]] = {chain: set() for chain in ANTIBODY_CHAINS}
    for name, (start, end) in spans.items():
        cdr_positions[by_name[name].chain].update(range(start, end + 1))

    # N-X-S/T with X != P, counted when the Asn itself sits in a CDR.
    n_sequons = 0
    for chain, sequence in sequences.items():
        for i in range(len(sequence) - 2):
            if (
                sequence[i] == "N"
                and sequence[i + 1] != "P"
                and sequence[i + 2] in "ST"
                and (i + 1) in cdr_positions[chain]
            ):
                n_sequons += 1

    # Framework cysteines are the canonical, paired intradomain disulfides. A
    # CDR cysteine is unpaired unless a second one in the same chain's CDRs can
    # pair with it, so an odd count leaves exactly one free thiol.
    n_cdr_cys = 0
    n_unpaired_cys = 0
    for chain, sequence in sequences.items():
        count = sum(
            1 for position in cdr_positions[chain] if sequence[position - 1] == "C"
        )
        n_cdr_cys += count
        n_unpaired_cys += count % 2

    whole = sequences[HEAVY_CHAIN] + sequences[LIGHT_CHAIN]
    net_charge = sum(whole.count(a) for a in "KR") - sum(whole.count(a) for a in "DE")

    h3 = FRAMEWORK.region_sequences(sequences)["CDR-H3"]
    longest = current = 0
    for residue in h3:
        current = current + 1 if residue in HYDROPHOBIC else 0
        longest = max(longest, current)

    return {
        "n_cdr_sequons": float(n_sequons),
        "n_cdr_cys": float(n_cdr_cys),
        "n_unpaired_cys": float(n_unpaired_cys),
        "net_charge": float(net_charge),
        "cdr_h3_length": float(len(h3)),
        "cdr_h3_max_hydrophobic_run": float(longest),
    }


def _liability_checks(liabilities: dict[str, float]) -> dict[str, bool]:
    """The four developability gates, evaluated on one design's liabilities.

    Reported individually as well as ANDed, so a single dominant liability is
    visible instead of hidden behind one combined soft criterion.
    """
    net_charge = liabilities["net_charge"]
    return {
        "no_cdr_sequon": liabilities["n_cdr_sequons"] == 0,
        "cys_paired": liabilities["n_unpaired_cys"] <= MAX_UNPAIRED_CYS,
        "net_charge_ok": (NET_CHARGE_RANGE[0] <= net_charge <= NET_CHARGE_RANGE[1]),
        "no_h3_hydrophobic_patch": (
            liabilities["cdr_h3_max_hydrophobic_run"] <= MAX_HYDROPHOBIC_RUN_H3
        ),
    }


# ── Bespoke per-design measurements ──────────────────────────────────────────


def measure(handler, submission):
    """Framework backbone RMSD and developability, per design.

    The AF3 fold cannot produce either: the RMSD needs the scorer's reference
    geometry, and the liabilities are sequence-level. The submitted H/L
    sequences are reached by ``design_id`` (never row position — the scored
    handler comes back in filesystem order). ``framework_rmsd`` and each
    liability count become dotted metrics the gates and ``metric_columns``
    address like any other; ``dev.ok`` is 1 exactly when all four soft
    developability checks hold.
    """
    frame = merge_submission_columns(
        handler.unpack(all=True), submission, ["sequence_H", "sequence_L"]
    )
    values: dict[str, dict] = {}
    for design_id, path, seq_h, seq_l in zip(
        frame["design_id"],
        frame["fold.path"],
        frame["sequence_H"],
        frame["sequence_L"],
    ):
        sequences = {HEAVY_CHAIN: seq_h, LIGHT_CHAIN: seq_l}
        try:
            rmsd = framework_backbone_rmsd(path, sequences)
        except Exception as exc:
            raise RuntimeError(
                f"framework_backbone_rmsd failed for {design_id}: {exc}"
            ) from exc
        liabilities = developability(sequences)
        checks = _liability_checks(liabilities)
        values[design_id] = {
            "framework_rmsd": rmsd,
            "n_cdr_sequons": liabilities["n_cdr_sequons"],
            "n_unpaired_cys": liabilities["n_unpaired_cys"],
            "net_charge": liabilities["net_charge"],
            "cdr_h3_length": liabilities["cdr_h3_length"],
            "cdr_h3_max_hydrophobic_run": liabilities["cdr_h3_max_hydrophobic_run"],
            "dev": {"ok": 1 if all(checks.values()) else 0},
        }
    return values


def per_liability_extra(frame):
    """Per-liability pass counts, alongside the antigen/framework provenance.

    Recomputes the four developability checks over the scored frame so the
    breakdown that explains a soft failure is reported next to the combined
    ``developability`` gate.
    """
    n_sequons = metric_series(frame, "n_cdr_sequons")
    n_unpaired = metric_series(frame, "n_unpaired_cys")
    net_charge = metric_series(frame, "net_charge")
    h3_run = metric_series(frame, "cdr_h3_max_hydrophobic_run")
    per_liability = {
        "no_cdr_sequon": int((n_sequons == 0).sum()),
        "cys_paired": int((n_unpaired <= MAX_UNPAIRED_CYS).sum()),
        "net_charge_ok": int(
            (
                (net_charge >= NET_CHARGE_RANGE[0])
                & (net_charge <= NET_CHARGE_RANGE[1])
            ).sum()
        ),
        "no_h3_hydrophobic_patch": int((h3_run <= MAX_HYDROPHOBIC_RUN_H3).sum()),
    }
    return {
        "antigen": TARGET_PDB.name,
        "framework": FRAMEWORK_PDB.name,
        "cdr_definition": CONFIG["cdr_definition"],
        "per_liability": per_liability,
    }


def main() -> None:
    AF3, IpSAE, ContactMS, Clash, SecondaryStructure = Blocks.getall(
        ORACLE, "ipsae", "contact_ms", "clash", "secondary_structure"
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
            chains=ANTIBODY_CHAINS,
            fixed_regions=FRAMEWORK,
            max_pairwise_identity=0.70,
        ),
        # The antigen is supplied by the scorer and co-folded with every design.
        context=load_context_chains(TARGET_PDB, [ANTIGEN_CHAIN]),
        # The antigen is supplied, so its fold is not what is being tested.
        # Folded single-sequence (AF3 here runs with no data pipeline and empty
        # MSAs) lysozyme comes out at pLDDT ~30 — an unfolded antigen that
        # nothing can dock to, which drives every interface metric to a
        # structural zero. Injecting the supplied coordinates as an AF3 template
        # restores it.
        oracle=AF3(
            name="fold",
            target_chain=ANTIGEN_CHAIN,
            full_receptor_pdb=str(TARGET_PDB),
        ),
        oracle_job_args=GPU_ARGS,
        metrics=[
            # binder_chain is a single chain, so the antigen plays that role and
            # H/L are the "targets": one block then yields both paratope pairs.
            IpSAE(
                name="ipsae",
                binder_chain=ANTIGEN_CHAIN,
                target_chains=list(ANTIBODY_CHAINS),
                pae_cutoff=IPSAE_PAE_CUTOFF,
            ),
            ContactMS(
                name="contact_ms",
                binder_chains=list(ANTIBODY_CHAINS),
                target_chains=[ANTIGEN_CHAIN],
            ),
            Clash(name="clash"),
            SecondaryStructure(name="fv", chains=list(ANTIBODY_CHAINS)),
        ],
        metric_job_args=CPU_ARGS,
        measure=measure,
        # Gates are filter-expression strings (the repo's FilterBlock DSL); each
        # threshold lives in the expression that enforces it. The interface PAE
        # gate reads ipsae's chain-restricted paratope↔antigen PAE, NOT AF3's
        # whole-complex min PAE (chain-agnostic across H/L/T). developability is
        # soft: ``dev.ok`` is 1 only when all four liability checks hold.
        gates={
            "interface_pae": f"{MIN_PAE_COL} <= {MIN_PAE_INTERFACE_MAX}",
            "ipsae": f"{IPSAE_COL} >= {IPSAE_MIN}",
            "plddt": f"{PLDDT_COL} >= {PLDDT_MIN}",
            "framework_preserved": f"framework_rmsd <= {FRAMEWORK_RMSD_MAX}",
            "contact_ms": f"{CONTACT_MS_COL} >= {CONTACT_MS_MIN}",
            "no_clashes": f"{CLASH_COL} <= {N_BB_CLASHES_MAX}",
            "developability": "dev.ok >= 1",
        },
        soft=["developability"],
        metric_columns=[
            MIN_PAE_COL,
            IPSAE_COL,
            IPSAE_MAX_COL,
            PLDDT_COL,
            FV_PLDDT_COL,
            AF3_MIN_PAE_COL,
            "framework_rmsd",
            CONTACT_MS_COL,
            CLASH_COL,
            "n_cdr_sequons",
            "n_unpaired_cys",
            "net_charge",
            "cdr_h3_length",
            "cdr_h3_max_hydrophobic_run",
        ],
        extra=per_liability_extra,
    )


if __name__ == "__main__":
    main()
