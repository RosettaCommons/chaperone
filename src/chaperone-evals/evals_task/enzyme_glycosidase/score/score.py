#!/usr/bin/env -S bash -c '"${CHAPERONE_REPO:-.}/.venv/bin/python" "$0" "$@"'
"""Held-out scorer for de novo glycosidase submissions.

A submission is sequences plus a ``motif_index_map`` naming where each of the
six reference theozyme positions landed. Every sequence is re-folded here with
AF3 together with the ``L:1`` substrate (chain **B**, supplied by the scorer as
a ``userCCD`` ligand), and then measured on:

  motif fidelity     — all-atom RMSD of the six mapped catalytic side chains
                       against ``inputs/theozyme.pdb``
  **dyad geometry**  — the E86 ↔ E177 carboxylate-carbon separation, which is
                       what distinguishes a glycosidase from a protein that
                       merely holds two glutamates near a sugar. The reference
                       theozyme encodes a *retaining* (double-displacement)
                       dyad, so the gate is a window centred on the reference
                       distance, not a simple maximum. See ``score/readme.md``.
  fold confidence    — AF3 pLDDT
  substrate pose     — substrate RMSD after superimposing on the mapped motif
  substrate conf.    — AF3 ligand pLDDT
  pocket burial      — mean relative SASA of the substrate atoms
  packing            — buried molecular surface, chain A vs the substrate
  polar recognition  — PLIP hydrogen bonds (soft)

Self-reported metrics are never trusted; every sequence is re-folded here.

Usage:
    ./score.py --designs=/absolute/path/to/submission.csv --nouse_local_gpus
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from absl import flags

from blocks import Blocks
from chaperone import init_cli
from scoring import (
    SequenceSubmissionSpec,
    define_score_flags,
    load_context_chains,
    merge_submission_columns,
    motif_rmsd,
    run_sequence_scorer,
)
from chaperone.io import parse_to_atom_array

define_score_flags()
FLAGS = flags.FLAGS

init_cli()

EVAL_NAME = "enzyme-glycosidase"
ORACLE = "af3"

SCRIPT_DIR = Path(__file__).resolve().parent
THEOZYME_PDB = (SCRIPT_DIR.parent / "inputs" / "theozyme.pdb").resolve()
# AF3 template carrying the ``L:1`` substrate as a userCCD ligand on chain B.
# The substrate is a custom molecule with no CCD code, so it cannot be rebuilt
# from the input structure alone — the template is part of the protocol.
HOLO_TEMPLATE = SCRIPT_DIR / "glycosidase_holo_template.json"
# Bump when the AF3 input composition changes so a stale fold can never
# masquerade as a result for the same submission.
SCORER_CACHE_VERSION = "holo-ligand-v1"

DESIGN_CHAIN = "A"
SUBSTRATE_CHAIN = "B"

# Reference catalytic positions, in the reference enzyme's own numbering.
MOTIF_REFERENCE_IDS = ("Y77", "E86", "Y88", "R122", "Q136", "E177")
MOTIF_RESIDUE_TYPES = {
    "Y77": "TYR",
    "E86": "GLU",
    "Y88": "TYR",
    "R122": "ARG",
    "Q136": "GLN",
    "E177": "GLU",
}

# The catalytic dyad. E86 is the nucleophile (2.5 Å from the anomeric carbon in
# the reference), E177 the general acid/base (2.8 Å from the glycosidic
# oxygen). The carboxylate carbon is CD for Glu (CG for Asp), verified against
# inputs/theozyme.pdb.
DYAD_REFERENCE_IDS = ("E86", "E177")
CARBOXYLATE_CARBON = {"GLU": "CD", "ASP": "CG"}

# ── Thresholds (held out; see score/readme.md for the rationale) ─────────────
MOTIF_RMSD_MAX = 1.0
# Reference dyad separation measured in inputs/theozyme.pdb: 7.387 Å between
# the E86 and E177 CD atoms (closest carboxylate oxygens 5.87 Å) — a retaining
# geometry. The window is the reference ± 1.5 Å, which still excludes the
# ≈10 Å inverting arrangement.
REFERENCE_DYAD_DISTANCE = 7.387
DYAD_DISTANCE_MIN = 5.9
DYAD_DISTANCE_MAX = 8.9
MIN_PLDDT = 85.0
SUBSTRATE_RMSD_MAX = 2.0
MIN_LIGAND_PLDDT = 70.0
MAX_MEAN_LIGAND_RASA = 0.15
MIN_CONTACT_MS = 200.0
MIN_HBONDS = 3  # soft
# ── end thresholds ──────────────────────────────────────────────────────────

GPU_ARGS = {
    "partition": "gpu",
    "gres": "gpu:small:1",
    "mem_gb": 32,
    "time_limit": "8:00:00",
    "rows_per_job": 10_000,  # one GPU, one chunk
}
CPU_ARGS = {"partition": "local"}


def _residue_atom(atom_array, residue_id: str, atom_name: str) -> np.ndarray:
    """The coordinate of one named atom of one ``A45``-style residue."""
    chain, index = residue_id[0], int(residue_id[1:])
    mask = (
        (atom_array.chain_id == chain)
        & (atom_array.res_id == index)
        & (atom_array.atom_name == atom_name)
    )
    hits = int(mask.sum())
    if hits != 1:
        raise ValueError(
            f"expected exactly one {atom_name} atom in residue {residue_id}, "
            f"found {hits}"
        )
    return atom_array.coord[mask][0]


def dyad_distance(design_structure: str, motif_index_map: dict[str, str]) -> float:
    """Distance between the two catalytic carboxylate carbons of a design.

    This is the benchmark's defining measurement. A retaining glycosidase holds
    its nucleophile and its acid/base close enough that the covalent
    glycosyl-enzyme intermediate can form and then be hydrolysed in place; an
    inverting one holds them ≈10 Å apart to fit a water molecule between the
    acid/base and the anomeric centre. Motif RMSD alone cannot see this,
    because a design can satisfy the per-residue superposition on average while
    the dyad itself opens or collapses.
    """
    atom_array = parse_to_atom_array(design_structure)
    coordinates = []
    for reference in DYAD_REFERENCE_IDS:
        residue_id = motif_index_map[reference]
        residue_type = MOTIF_RESIDUE_TYPES[reference]
        coordinates.append(
            _residue_atom(atom_array, residue_id, CARBOXYLATE_CARBON[residue_type])
        )
    return float(np.linalg.norm(coordinates[0] - coordinates[1]))


def substrate_rmsd(design_structure: str, motif_index_map: dict[str, str]) -> float:
    """Substrate RMSD after superimposing the design's motif on the theozyme.

    The submission carries no coordinates, so there is no design backbone to
    align against — and the theozyme itself is side-chains-only, so it has no
    backbone either. The only frame both structures share is the catalytic
    motif. Superimposing on the mapped motif atoms and then measuring the
    substrate therefore asks exactly the right question: does the substrate sit
    where the catalytic machinery needs it, relative to that machinery?
    """
    from biotite.structure import superimpose

    design = parse_to_atom_array(design_structure)
    reference = parse_to_atom_array(THEOZYME_PDB)

    # Pair motif atoms by name within each mapped residue, exactly as
    # ``pipelines.evals.motif_rmsd`` does, so the two metrics share a frame.
    design_indices: list[int] = []
    reference_indices: list[int] = []
    for position, residue_id in motif_index_map.items():
        chain, index = residue_id[0], int(residue_id[1:])
        reference_residue = int(str(position)[1:])
        design_mask = np.flatnonzero(
            (design.chain_id == chain) & (design.res_id == index)
        )
        reference_mask = np.flatnonzero(reference.res_id == reference_residue)
        if design_mask.size == 0:
            raise ValueError(f"design has no residue {residue_id}")
        if reference_mask.size == 0:
            raise ValueError(f"reference has no residue {position}")
        reference_atoms = {str(reference.atom_name[i]): i for i in reference_mask}
        for atom_index in design_mask:
            atom_name = str(design.atom_name[atom_index])
            if atom_name in reference_atoms:
                design_indices.append(int(atom_index))
                reference_indices.append(int(reference_atoms[atom_name]))
    if len(design_indices) < 3:
        raise ValueError("Fewer than 3 motif atoms could be paired with the reference")

    _, transform = superimpose(reference[reference_indices], design[design_indices])

    reference_substrate = reference[reference.chain_id == SUBSTRATE_CHAIN]
    if len(reference_substrate) == 0:
        raise ValueError(f"reference has no chain {SUBSTRATE_CHAIN}")
    # Take the folded substrate as every non-amino-acid heavy atom, rather than
    # trusting AF3 to keep the chain label, then pair to the reference by atom
    # name (AF3 emits the userCCD atom_ids verbatim).
    from biotite.structure import filter_amino_acids

    design_substrate = design[~filter_amino_acids(design) & (design.element != "H")]
    if len(design_substrate) == 0:
        raise ValueError(f"{design_structure} contains no substrate atoms")

    design_by_name = {str(n): i for i, n in enumerate(design_substrate.atom_name)}
    pairs = [
        (design_by_name[str(name)], i)
        for i, name in enumerate(reference_substrate.atom_name)
        if str(name) in design_by_name
    ]
    if len(pairs) != len(reference_substrate):
        missing = sorted(
            {str(n) for n in reference_substrate.atom_name} - set(design_by_name)
        )
        raise ValueError(
            f"{design_structure}: substrate atoms missing from the fold: {missing}"
        )

    moved = transform.apply(design_substrate.coord[[p[0] for p in pairs]])
    target = reference_substrate.coord[[p[1] for p in pairs]]
    return float(np.sqrt(((moved - target) ** 2).sum(axis=1).mean()))


def measure_structure(handler, submission):
    """Bespoke structure-level metrics the oracle and metric blocks cannot emit.

    Computes, per design:
      ``motif.rmsd``     all-atom RMSD of the mapped catalytic side chains vs
                         the reference geometry
      ``dyad.distance``  the E86 ↔ E177 carboxylate-carbon separation
      ``substrate.rmsd`` substrate RMSD after superimposing on the mapped motif

    Each is returned as a dotted metric so the gates and ``metric_columns``
    address it like any block metric. The submitted map is per design and joined
    on ``design_id`` — never by row position, since the scored handler comes back
    in filesystem order.
    """
    frame = merge_submission_columns(
        handler.unpack(all=True), submission, ["motif_index_map"]
    )
    values = {}
    for design_id, path, motif_map in zip(
        frame["design_id"], frame["holo.path"], frame["motif_index_map"]
    ):
        try:
            values[design_id] = {
                "motif": {"rmsd": motif_rmsd(path, THEOZYME_PDB, motif_map)},
                "dyad": {"distance": dyad_distance(path, motif_map)},
                "substrate": {"rmsd": substrate_rmsd(path, motif_map)},
            }
        except Exception as exc:
            raise RuntimeError(
                f"structural scoring failed for {design_id} ({path}): {exc}"
            ) from exc
    return values


def main() -> None:
    AF3, ContactMS, RASA, PLIP = Blocks.getall("af3", "contact_ms", "rasa", "plip")

    run_sequence_scorer(
        eval_name=EVAL_NAME,
        validation_oracle=ORACLE,
        designs=FLAGS.designs,
        out=FLAGS.out,
        per_design_out=FLAGS.per_design_out,
        required_n=FLAGS.required_n,
        run_meta=FLAGS.run_meta,
        run_root=SCRIPT_DIR / "score_runs" / SCORER_CACHE_VERSION,
        spec=SequenceSubmissionSpec(
            required_n=FLAGS.required_n,
            chains=(DESIGN_CHAIN,),
            motif_reference_ids=MOTIF_REFERENCE_IDS,
            motif_residue_types=MOTIF_RESIDUE_TYPES,
            max_pairwise_identity=0.70,
        ),
        # The substrate is supplied by the scorer, never by the submitter. It is
        # attached to every design's input structure so the input is
        # self-describing; its chemistry (bonds, elements) reaches AF3 through the
        # template's userCCD.
        context=load_context_chains(THEOZYME_PDB, (SUBSTRATE_CHAIN,)),
        oracle=AF3(name="holo", template_json=str(HOLO_TEMPLATE)),
        oracle_job_args=GPU_ARGS,
        metrics=[
            RASA(name="holo_rasa"),
            ContactMS(
                name="holo_contact_ms",
                binder_chains=[DESIGN_CHAIN],
                target_chains=[SUBSTRATE_CHAIN],
            ),
            PLIP(name="holo_plip"),
        ],
        metric_job_args=CPU_ARGS,
        measure=measure_structure,
        # Gates are filter-expression strings (the repo's FilterBlock DSL); each
        # threshold lives in the expression that enforces it. The retaining dyad
        # is a window, not a maximum.
        gates={
            "motif_fidelity": f"motif.rmsd <= {MOTIF_RMSD_MAX}",
            "dyad_geometry": (
                f"dyad.distance >= {DYAD_DISTANCE_MIN} "
                f"and dyad.distance <= {DYAD_DISTANCE_MAX}"
            ),
            "plddt": f"holo.plddt >= {MIN_PLDDT}",
            "substrate_pose": f"substrate.rmsd <= {SUBSTRATE_RMSD_MAX}",
            "substrate_confidence": f"holo.ligand_plddt >= {MIN_LIGAND_PLDDT}",
            "pocket_burial": (
                f"holo_rasa.mean_ligand_atom_rasa <= {MAX_MEAN_LIGAND_RASA}"
            ),
            "packing": f"holo_contact_ms >= {MIN_CONTACT_MS}",
            "hbonds": f"holo_plip.hydrogen_bonds >= {MIN_HBONDS}",
        },
        soft=["hbonds"],
        metric_columns=[
            "motif.rmsd",
            "dyad.distance",
            "substrate.rmsd",
            "holo.plddt",
            "holo.ligand_plddt",
            "holo_rasa.mean_ligand_atom_rasa",
            "holo_contact_ms",
            "holo_plip.hydrogen_bonds",
        ],
        extra={
            "reference": THEOZYME_PDB.name,
            "mechanism": "retaining",
        },
    )


if __name__ == "__main__":
    main()
