"""Small RF3-compatible all-atom and by-type lDDT scorer for validation."""

from __future__ import annotations

import ast
from collections.abc import Sequence

import numpy as np
from biotite.structure import AtomArray, lddt

MIN_ATOM_OVERLAP = 0.95


def _parse_specification(value, *, field: str) -> list[tuple]:
    if value is None:
        raise ValueError(f"{field} is required for validation scoring")
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError) as error:
            raise ValueError(f"invalid {field}: {value!r}") from error
    if not isinstance(value, Sequence):
        raise TypeError(f"{field} must be a list of tuples, got {type(value).__name__}")
    return [tuple(item) for item in value]


def _atom_keys(
    atom_array: AtomArray, remap: dict[tuple[str, int], int] | None = None
) -> list[tuple[str, int, str]]:
    """Per-atom (chain, residue, atom-name) keys, optionally renumbered."""
    remap = remap or {}
    return [
        (str(chain), remap.get((str(chain), int(residue)), int(residue)), str(atom))
        for chain, residue, atom in zip(
            atom_array.chain_id, atom_array.res_id, atom_array.atom_name
        )
    ]


def _residues_by_chain(atom_array: AtomArray) -> dict[str, list[tuple[int, str]]]:
    """Ordered (res_id, res_name) per chain, in file order, deduplicated."""
    chains: dict[str, list[tuple[int, str]]] = {}
    seen: set[tuple[str, int]] = set()
    for chain, residue, name in zip(
        atom_array.chain_id, atom_array.res_id, atom_array.res_name
    ):
        chain, residue, name = str(chain), int(residue), str(name)
        if (chain, residue) in seen:
            continue
        seen.add((chain, residue))
        chains.setdefault(chain, []).append((residue, name))
    return chains


def _res_id_remap(
    reference: AtomArray, prediction: AtomArray
) -> tuple[dict[tuple[str, int], int], list[str]]:
    """Map the prediction's residue numbering onto the reference's, per chain.

    Some predictors renumber non-polymer residues: AlphaFold3 writes a ligand the
    reference numbers 201 as residue 1, keeping the chain and every atom name
    identical. Joining on (chain, res_id, atom_name) then drops every ligand
    atom -- and because ligands are ~2% of a typical assembly, the overlap gate
    still passes and the target is silently scored protein-only. On the
    protein-ligand cohort that happened on 116 of 120 targets.

    A chain is remapped only when its residue-*name* sequence is identical on
    both sides, so residues are paired by position with their identity
    confirmed. A chain whose composition differs is left alone and named in the
    returned list: a wrong pairing is worse than a missing one.
    """
    reference_chains = _residues_by_chain(reference)
    prediction_chains = _residues_by_chain(prediction)

    remap: dict[tuple[str, int], int] = {}
    unmapped: list[str] = []
    for chain, predicted in prediction_chains.items():
        expected = reference_chains.get(chain)
        if expected is None:
            unmapped.append(chain)
            continue
        if [name for _, name in expected] != [name for _, name in predicted]:
            unmapped.append(chain)
            continue
        for (predicted_id, _), (reference_id, _) in zip(predicted, expected):
            if predicted_id != reference_id:
                remap[(chain, predicted_id)] = reference_id
    return remap, unmapped


def _non_protein_mask(atom_array: AtomArray) -> np.ndarray:
    if "is_protein" in atom_array.get_annotation_categories():
        return ~np.asarray(atom_array.is_protein, dtype=bool)
    from biotite.structure import filter_amino_acids

    return ~filter_amino_acids(atom_array)


def _align_atoms(
    reference: AtomArray,
    prediction: AtomArray,
) -> tuple[AtomArray, AtomArray, dict]:
    """Pair atoms by (chain, residue, atom name), renumbering where it is safe.

    Returns the paired arrays and a coverage report. The report is not optional
    detail: this scorer previously exposed only an aggregate gate, and a
    whole-cohort ligand misalignment sat underneath it undetected because
    ligands are a small fraction of atoms. Coverage is therefore reported
    separately for non-protein atoms, where that failure shows up at once.
    """
    remap, unmapped_chains = _res_id_remap(reference, prediction)

    reference_index: dict[tuple[str, int, str], int] = {}
    for index, key in enumerate(_atom_keys(reference)):
        reference_index.setdefault(key, index)
    prediction_index: dict[tuple[str, int, str], int] = {}
    for index, key in enumerate(_atom_keys(prediction, remap)):
        prediction_index.setdefault(key, index)

    common = [key for key in reference_index if key in prediction_index]
    overlap = len(common) / max(len(reference_index), len(prediction_index), 1)
    if not common or overlap < MIN_ATOM_OVERLAP:
        raise ValueError(
            f"RF3 lDDT atom overlap {overlap:.2%} is below the required "
            f"{MIN_ATOM_OVERLAP:.0%} (reference={len(reference_index)}, "
            f"prediction={len(prediction_index)}, common={len(common)}, "
            f"renumbered={len(remap)} residues, unmapped chains={unmapped_chains})"
        )
    reference_indices = np.fromiter(
        (reference_index[key] for key in common), dtype=np.int64
    )
    prediction_indices = np.fromiter(
        (prediction_index[key] for key in common), dtype=np.int64
    )

    non_protein = _non_protein_mask(reference)
    matched = np.zeros(len(reference), dtype=bool)
    matched[reference_indices] = True
    n_non_protein = int(non_protein.sum())
    report = {
        "atom_coverage": round(float(overlap), 6),
        "n_reference_atoms": int(len(reference)),
        "n_matched_atoms": int(len(common)),
        "non_protein_atom_coverage": (
            round(float(matched[non_protein].sum() / n_non_protein), 6)
            if n_non_protein
            else None
        ),
        "n_non_protein_atoms": n_non_protein,
        "n_renumbered_residues": len(remap),
        "unmapped_chains": unmapped_chains,
    }
    return reference[reference_indices], prediction[prediction_indices], report


def _chain_ids(reference: AtomArray, requested: set[str]) -> np.ndarray:
    """Choose the chain annotation that covers the most requested units.

    Requiring an annotation to contain *every* requested unit fails a whole
    target when one of them is absent, which on multi-entity assemblies is the
    common case rather than the exception: ligand, water and alternate-copy PN
    units routinely do not survive assembly construction. Scoring the units that
    are present and reporting the rest is strictly more informative than
    refusing to score the target at all, so the caller resolves what it can and
    records what it could not.
    """
    annotations = reference.get_annotation_categories()
    candidates = []
    if "chain_iid" in annotations:
        candidates.append(np.asarray(reference.chain_iid, dtype=str))
    candidates.append(np.asarray(reference.chain_id, dtype=str))
    if not requested:
        return candidates[0]
    # Ties keep the earlier candidate, so chain_iid wins over chain_id.
    return max(candidates, key=lambda candidate: len(requested & set(candidate)))


def _token_ids(reference: AtomArray) -> np.ndarray:
    if "token_id" in reference.get_annotation_categories():
        return np.asarray(reference.token_id)

    from atomworks.ml.transforms.atom_array import AddGlobalTokenIdAnnotation
    from atomworks.ml.transforms.atomize import AtomizeByCCDName
    from atomworks.ml.transforms.base import Compose

    pipeline = Compose(
        [AtomizeByCCDName(atomize_by_default=True), AddGlobalTokenIdAnnotation()]
    )
    annotated = pipeline({"atom_array": reference.copy()})["atom_array"]
    return np.asarray(annotated.token_id)


def _score(
    reference: AtomArray,
    prediction: AtomArray,
    token_ids: np.ndarray,
    *,
    atom_mask: np.ndarray,
    partner_mask: np.ndarray,
    inclusion_radius: float,
) -> float:
    value = float(
        lddt(
            reference,
            prediction,
            atom_mask=atom_mask,
            partner_mask=partner_mask,
            inclusion_radius=inclusion_radius,
            exclude_same_residue=False,
            filter_function=lambda pairs: token_ids[pairs[:, 0]]
            != token_ids[pairs[:, 1]],
        )
    )
    # RF3's denominator includes an epsilon, so an empty valid-pair set scores
    # 0.0 rather than emitting NaN (e.g. a monatomic ion pn-unit).
    if np.isnan(value):
        return 0.0
    if not np.isfinite(value):
        raise ValueError(f"RF3 lDDT produced a non-finite value: {value}")
    return value


def rf3_allatom_lddt(
    reference: AtomArray,
    prediction: AtomArray,
    *,
    interfaces_to_score,
    pn_units_to_score,
    pn_unit_id_map=None,
) -> dict:
    """Score one prediction with RF3 global and per-unit/interface lDDT masks."""
    interfaces = _parse_specification(interfaces_to_score, field="interfaces_to_score")
    pn_units = _parse_specification(pn_units_to_score, field="pn_units_to_score")
    pn_unit_id_map_raw = dict(pn_unit_id_map or {})
    # A None target means the manifest saw that PN unit vanish while building the
    # assembly. Coercing it with str() turned it into the literal "None", which
    # then failed to resolve and took the whole target down with it.
    pn_unit_id_map = {
        str(source): str(target)
        for source, target in (pn_unit_id_map or {}).items()
        if target is not None
    }
    dropped_by_manifest = {
        str(source)
        for source, target in (pn_unit_id_map_raw or {}).items()
        if target is None
    }

    def serialized_unit(unit) -> str:
        unit = str(unit)
        return pn_unit_id_map.get(unit, unit)

    reference, prediction, alignment = _align_atoms(reference, prediction)
    token_ids = _token_ids(reference)
    resolved = np.isfinite(reference.coord).all(axis=-1)
    if "occupancy" in reference.get_annotation_categories():
        resolved &= np.asarray(reference.occupancy) > 0
    if not resolved.any():
        raise ValueError("reference assembly contains no resolved atoms")

    all_atom_score = _score(
        reference,
        prediction,
        token_ids,
        atom_mask=resolved,
        partner_mask=resolved,
        inclusion_radius=15.0,
    )

    requested_units = {
        serialized_unit(unit) for interface in interfaces for unit in interface[:2]
    } | {serialized_unit(unit) for unit, _ in pn_units}
    chain_annotation = _chain_ids(reference, requested_units)
    present = set(chain_annotation)
    unscored: list[dict] = []

    def missing(unit) -> str | None:
        """Why this unit cannot be scored, or None if it can."""
        serialized = serialized_unit(unit)
        if serialized in present:
            return None
        if str(unit) in dropped_by_manifest:
            return "absent from the built assembly"
        return "not present in the reference chain annotation"

    by_type = []
    for unit_i, unit_j, interface_type in interfaces:
        reasons = {unit: missing(unit) for unit in (unit_i, unit_j)}
        if any(reasons.values()):
            unscored.append(
                {
                    "pn_units": [unit_i, unit_j],
                    "type": interface_type,
                    "reason": "; ".join(
                        f"{unit}: {why}" for unit, why in reasons.items() if why
                    ),
                }
            )
            continue
        value = _score(
            reference,
            prediction,
            token_ids,
            atom_mask=resolved & (chain_annotation == serialized_unit(unit_i)),
            partner_mask=resolved & (chain_annotation == serialized_unit(unit_j)),
            inclusion_radius=30.0,
        )
        by_type.append(
            {
                "pn_units": [unit_i, unit_j],
                "type": interface_type,
                "best_of_1_lddt": value,
            }
        )

    for unit, unit_type in pn_units:
        why = missing(unit)
        if why is not None:
            unscored.append({"pn_units": [unit], "type": unit_type, "reason": why})
            continue
        unit_mask = resolved & (chain_annotation == serialized_unit(unit))
        value = _score(
            reference,
            prediction,
            token_ids,
            atom_mask=unit_mask,
            partner_mask=unit_mask,
            inclusion_radius=15.0,
        )
        by_type.append(
            {
                "pn_units": [unit],
                "type": unit_type,
                "best_of_1_lddt": value,
            }
        )

    return {
        "rf3_all_atom_lddt": {"best_of_1_lddt": all_atom_score},
        "rf3_by_type_lddt": by_type,
        # Never silently short: a caller comparing per-unit coverage across arms
        # needs to see which units were skipped and why, not infer it from a
        # shorter list.
        "rf3_unscored_pn_units": unscored,
        # Surfaced per row so a systematic misalignment cannot hide under an
        # aggregate gate again.
        "rf3_atom_alignment": alignment,
    }
