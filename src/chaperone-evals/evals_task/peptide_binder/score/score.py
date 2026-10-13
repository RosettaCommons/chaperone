#!/usr/bin/env -S bash -c '"${CHAPERONE_REPO:-.}/.venv/bin/python" "$0" "$@"'
"""Held-out scorer for sequence-only EIN2 site-1 peptide binder submissions.

The target is a 15-residue, glycine/serine/proline-rich peptide — genuinely
disordered on its own. A binder can therefore "succeed" on every interface
metric while having dragged the peptide into a pose that has nothing to do with
the one it was asked to recognize. This scorer exists to make that failure mode
visible:

  * every design is co-folded with the peptide by **plain AF3**, from sequence
    only. The peptide is *not* supplied as a template — see ``readme.md``;
  * the predicted peptide is then compared, backbone-to-backbone, against the
    supplied ``inputs/target.pdb`` conformation — over the **buried groove
    epitope** (``CORE_RESIDUES``), because the exposed flank is not held by any
    binder and its scatter measures the oracle rather than the design. Whole-
    peptide and flank RMSDs are reported per design so nothing is hidden.

Everything else — interface confidence (AF3 + ipSAE), binder fold quality,
buried surface, sterics, and interface energetics — is measured on that same
independent refold. Self-reported metrics are never trusted.

Usage:
    ./score.py --designs=/absolute/path/to/submission.csv --nouse_local_gpus
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from absl import flags

from blocks import Blocks
from chaperone import init_cli
from scoring import (
    SequenceSubmissionSpec,
    define_score_flags,
    load_context_chains,
    run_sequence_scorer,
)
from chaperone.io import parse_to_atom_array

define_score_flags()
FLAGS = flags.FLAGS

init_cli()

EVAL_NAME = "peptide-binder"
# Plain AF3 with the peptide co-folded as sequence context. Deliberately *not*
# af3_with_target_template: templating the peptide would pre-place the very
# conformation the peptide-pose gate is supposed to measure independently.
ORACLE = "af3"

SCRIPT_DIR = Path(__file__).resolve().parent
TARGET_PDB = (SCRIPT_DIR.parent / "inputs" / "target.pdb").resolve()

BINDER_CHAIN = "A"
TARGET_CHAIN = "B"
BACKBONE_ATOMS = ("N", "CA", "C", "O")

# The buried groove epitope of the peptide, defined from the *reference complex*
# (not from any design distribution) by per-residue burial on binder removal and
# minimum heavy-atom distance to the binder. B9-B15 is the contiguous segment
# that the groove actually encloses: it holds every deeply buried residue
# (S9 0.91, F10 0.98, S12 0.98, L13 0.59 buried fraction) and every residue in
# it contacts the binder within 3.4 Å. B1-B8 is a solvent-exposed flank the
# binder barely touches (B1/B2 burial 0.07/0.01, closest approach 4.4/4.8 Å).
# See "Deviations from the plan spec" in readme.md and
# ``inputs/prepare_target.py --report_epitope`` for the derivation.
CORE_RESIDUES = tuple(range(9, 16))

# ── Thresholds (see score/readme.md for the rationale of each) ───────────────
MIN_PAE_INTERACTION_MAX = 1.5
IPSAE_MIN = 0.30
PEPTIDE_CORE_RMSD_MAX = 1.5
MEAN_PLDDT_MIN = 85.0
FRACTION_LOOP_MAX = 0.35
CONTACT_MS_MIN = 300.0
N_BB_CLASHES_MAX = 0
DDG_SOFT_MAX = -15.0
# ── end thresholds ───────────────────────────────────────────────────────────

GPU_ARGS = {
    "partition": "gpu",
    "gres": "gpu:small:1",
    "mem_gb": 32,
    "time_limit": "8:00:00",
    "rows_per_job": 10_000,  # one chunk, one GPU
}
CPU_ARGS = {
    "partition": "cpu",
    "mem_gb": 16,
    "cpus_per_task": 2,
    "time_limit": "8:00:00",
    "rows_per_job": 12,
}


def _peptide_backbone(path: str | Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Chain-B backbone coordinates in a canonical residue/atom order.

    Returns the ``(4N, 3)`` coordinate block, the per-residue three-letter
    names (so a caller can assert the two structures describe the same peptide
    before superimposing them), and the residue ID of every returned atom.
    """
    atoms = parse_to_atom_array(str(path))
    mask = (atoms.chain_id == TARGET_CHAIN) & np.isin(
        atoms.atom_name, list(BACKBONE_ATOMS)
    )
    chain = atoms[mask]
    if len(chain) == 0:
        raise ValueError(f"no chain {TARGET_CHAIN} backbone atoms in {path}")

    res_ids = sorted({int(r) for r in chain.res_id})
    index = {
        (int(r), str(a)): i
        for i, (r, a) in enumerate(zip(chain.res_id, chain.atom_name))
    }
    order: list[int] = []
    for res_id in res_ids:
        for atom_name in BACKBONE_ATOMS:
            key = (res_id, atom_name)
            if key not in index:
                raise ValueError(
                    f"{path}: chain {TARGET_CHAIN} residue {res_id} has no {atom_name}"
                )
            order.append(index[key])

    names = [str(chain.res_name[index[(res_id, "CA")]]).strip() for res_id in res_ids]
    atom_res_ids = np.repeat(np.asarray(res_ids), len(BACKBONE_ATOMS))
    return chain.coord[order], names, atom_res_ids


@lru_cache(maxsize=1)
def _reference_peptide() -> tuple[np.ndarray, tuple[str, ...], np.ndarray, np.ndarray]:
    coord, residues, atom_res_ids = _peptide_backbone(TARGET_PDB)
    core = np.isin(atom_res_ids, list(CORE_RESIDUES))
    if not core.any():
        raise ValueError(
            f"core residues {list(CORE_RESIDUES)} are absent from {TARGET_PDB}"
        )
    return coord, tuple(residues), atom_res_ids, core


def _rmsd_after_fit(
    design: np.ndarray,
    reference: np.ndarray,
    fit_mask: np.ndarray,
    measure_mask: np.ndarray,
) -> float:
    """Superimpose on ``fit_mask`` atoms, then score RMSD over ``measure_mask``."""
    from biotite.structure import AtomArray, superimpose

    mobile, fixed = AtomArray(int(fit_mask.sum())), AtomArray(int(fit_mask.sum()))
    mobile.coord, fixed.coord = design[fit_mask], reference[fit_mask]
    _, transform = superimpose(fixed, mobile)

    full = AtomArray(len(design))
    full.coord = design
    moved = transform.apply(full)
    deviation = moved.coord[measure_mask] - reference[measure_mask]
    return float(np.sqrt((deviation**2).sum(axis=1).mean()))


def _aligned_peptide(structure_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validated design peptide backbone plus the reference and its core mask."""
    reference_coord, reference_residues, _, core = _reference_peptide()
    coord, residues, _ = _peptide_backbone(structure_path)
    if tuple(residues) != reference_residues:
        raise ValueError(
            f"{structure_path}: chain {TARGET_CHAIN} is {residues}, "
            f"expected the supplied target {list(reference_residues)}"
        )
    return coord, reference_coord, core


def peptide_core_rmsd(structure_path: str) -> float:
    """**The gated pose metric.** Backbone RMSD over the buried groove epitope.

    The two chain-B backbones are superimposed on each other directly. There is
    no shared frame to align in: a sequence-only submission carries no reference
    binder, so "align on the binder, then measure the peptide" — the convention
    used inside a design pipeline, where the pre- and post-prediction complexes
    share a binder — has no counterpart here. What is measurable, and what the
    benchmark cares about, is whether the design induces the *supplied bound
    conformation* rather than an arbitrary one.

    Only ``CORE_RESIDUES`` are fitted and scored. The exposed N-terminal flank
    is excluded because the binder does not hold it: including it measures the
    oracle's sampling of a free peptide tail, not the design (see readme).
    """
    coord, reference, core = _aligned_peptide(structure_path)
    return _rmsd_after_fit(coord, reference, core, core)


def peptide_pose_rmsd(structure_path: str) -> float:
    """Diagnostic: whole-peptide backbone RMSD, fitted on the whole peptide."""
    coord, reference, _ = _aligned_peptide(structure_path)
    everything = np.ones(len(reference), dtype=bool)
    return _rmsd_after_fit(coord, reference, everything, everything)


def peptide_flank_rmsd(structure_path: str) -> float:
    """Diagnostic: where the exposed flank lands once the core is superimposed."""
    coord, reference, core = _aligned_peptide(structure_path)
    return _rmsd_after_fit(coord, reference, core, ~core)


def measure_pose(handler, submission):
    """Backbone pose RMSDs of the refolded peptide vs the supplied conformation.

    Returned under the ``pose`` namespace so the gate addresses the buried-groove
    epitope RMSD as ``pose.epitope_rmsd`` like any other metric; the whole-peptide
    and flank RMSDs ride along as diagnostics. Keyed on ``design_id`` and joined
    by identity, never by row position: the scored handler comes back in
    filesystem order, not submission order.
    """
    frame = handler.unpack(all=True)
    values = {}
    for design_id, path in zip(frame["design_id"], frame["score_af3.path"]):
        try:
            values[design_id] = {
                "pose": {
                    "epitope_rmsd": peptide_core_rmsd(path),
                    "whole_rmsd": peptide_pose_rmsd(path),
                    "flank_rmsd": peptide_flank_rmsd(path),
                }
            }
        except Exception as exc:
            raise RuntimeError(
                f"peptide pose RMSD failed for {design_id}: {exc}"
            ) from exc
    return values


def main() -> None:
    AF3, IpSAE, SecondaryStructure, ContactMS, Clash, PPIDDG = Blocks.getall(
        ORACLE, "ipsae", "secondary_structure", "contact_ms", "clash", "ppi_ddg"
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
            chains=(BINDER_CHAIN,),
            max_pairwise_identity=0.70,
        ),
        # The peptide is supplied by the scorer, never by the submitter, and is
        # co-folded with every design as sequence context (no template).
        context=load_context_chains(TARGET_PDB, (TARGET_CHAIN,)),
        oracle=AF3(name="score_af3", diffusion_batch_size=1),
        oracle_job_args=GPU_ARGS,
        metrics=[
            IpSAE(
                name="score_ipsae",
                binder_chain=BINDER_CHAIN,
                target_chains=[TARGET_CHAIN],
            ),
            SecondaryStructure(name="score_ss", chains=[BINDER_CHAIN]),
            ContactMS(
                name="score_contact_ms",
                binder_chains=[BINDER_CHAIN],
                target_chains=[TARGET_CHAIN],
            ),
            Clash(name="score_clash"),
            PPIDDG(
                name="score_ddg",
                binder_chain=BINDER_CHAIN,
                target_chain=TARGET_CHAIN,
                relax_rounds=1,
                replicates=1,
            ),
        ],
        metric_job_args=CPU_ARGS,
        # The buried-groove epitope RMSD is a bespoke geometry computation on the
        # refolded chain-B backbone; the whole-peptide and flank RMSDs ride along
        # as diagnostics under the same ``pose`` namespace.
        measure=measure_pose,
        # Gates are filter-expression strings (the repo's FilterBlock DSL); each
        # threshold lives in the expression that enforces it. AF3's own
        # `min_pae_interaction` is gated (not ipsae's `min_pae_interface`, which
        # is reported for provenance only — see readme.md).
        gates={
            "interface_pae": f"score_af3.min_pae_interaction <= {MIN_PAE_INTERACTION_MAX}",
            "ipsae": f"score_ipsae.ipsae >= {IPSAE_MIN}",
            "peptide_core_pose": f"pose.epitope_rmsd <= {PEPTIDE_CORE_RMSD_MAX}",
            "binder_plddt": f"score_ss.mean_plddt >= {MEAN_PLDDT_MIN}",
            "binder_fraction_loop": f"score_ss.fraction_loop < {FRACTION_LOOP_MAX}",
            "contact_ms": f"score_contact_ms >= {CONTACT_MS_MIN}",
            "no_bb_clashes": f"score_clash.n_bb_clashes == {N_BB_CLASHES_MAX}",
            "ddg": f"score_ddg.ddg <= {DDG_SOFT_MAX}",
        },
        metric_columns=[
            "score_af3.min_pae_interaction",
            "score_ipsae.ipsae",
            # Reported for provenance only. NOT gated: chain-restricted A<->B PAE
            # is a diagnostic; the interface gate uses AF3's min_pae_interaction.
            "score_ipsae.min_pae_interface",
            "pose.epitope_rmsd",
            "pose.whole_rmsd",
            "pose.flank_rmsd",
            "score_ss.mean_plddt",
            "score_ss.fraction_loop",
            "score_contact_ms",
            "score_clash.n_bb_clashes",
            "score_ddg.ddg",
        ],
        extra={
            "target": TARGET_PDB.name,
            "target_chain": TARGET_CHAIN,
            "binder_chain": BINDER_CHAIN,
            "peptide_length": len(_reference_peptide()[1]),
        },
    )


if __name__ == "__main__":
    main()
