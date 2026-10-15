"""Optional oracle-output remapping for validation scoring."""

from __future__ import annotations

import numpy as np
from biotite.structure import filter_amino_acids, get_residue_starts

from blocks.networks.prediction.monomer_reference import residue_sequence

from .lddt import MIN_ATOM_OVERLAP


def _protein_atoms(atom_array):
    if "is_protein" in atom_array.get_annotation_categories():
        atoms = atom_array[np.asarray(atom_array.is_protein, dtype=bool)]
    else:
        atoms = atom_array[filter_amino_acids(atom_array)]
    if len(atoms) == 0:
        raise ValueError("remap_outputs_to_input requires protein atoms")
    return atoms


def _split_chains(atom_array) -> list:
    chain_category = (
        "chain_iid"
        if "chain_iid" in atom_array.get_annotation_categories()
        else "chain_id"
    )
    chain_ids = list(
        dict.fromkeys(str(value) for value in atom_array.get_annotation(chain_category))
    )
    return [
        atom_array[
            np.asarray(atom_array.get_annotation(chain_category), dtype=str) == chain_id
        ]
        for chain_id in chain_ids
    ]


def _project_chain(
    reference,
    prediction,
    projected,
    canonical_res_names,
    canonical_hetero,
    reference_offset: int,
) -> None:
    reference_starts = get_residue_starts(reference, add_exclusive_stop=True)
    prediction_starts = get_residue_starts(prediction, add_exclusive_stop=True)
    n_reference = len(reference_starts) - 1
    n_prediction = len(prediction_starts) - 1
    if n_prediction != n_reference:
        raise ValueError(
            "prediction/input residue counts differ for a protein chain: "
            f"prediction={n_prediction}, input={n_reference}"
        )
    if residue_sequence(prediction) != residue_sequence(reference):
        raise ValueError("prediction/input protein-chain sequences differ")

    for residue_index in range(n_reference):
        reference_slice = slice(
            reference_starts[residue_index], reference_starts[residue_index + 1]
        )
        prediction_slice = slice(
            prediction_starts[residue_index], prediction_starts[residue_index + 1]
        )
        projected_slice = slice(
            reference_offset + reference_slice.start,
            reference_offset + reference_slice.stop,
        )
        canonical_res_names[projected_slice] = prediction.res_name[
            prediction_slice.start
        ]
        if canonical_hetero is not None:
            canonical_hetero[projected_slice] = prediction.hetero[
                prediction_slice.start
            ]
        predicted_atoms = {
            str(name): coord
            for name, coord in zip(
                prediction.atom_name[prediction_slice],
                prediction.coord[prediction_slice],
                strict=True,
            )
        }
        for atom_index in range(reference_slice.start, reference_slice.stop):
            coord = predicted_atoms.get(str(reference.atom_name[atom_index]))
            if coord is not None:
                projected[reference_offset + atom_index] = coord


def remap_output_to_input(
    reference,
    prediction,
    validation: dict,
    *,
    min_atom_coverage: float = MIN_ATOM_OVERLAP,
):
    """Map protein chains and residues from oracle order onto the input layout."""
    del validation  # Reserved for future non-positional chain mappings.
    reference = _protein_atoms(reference)
    prediction = _protein_atoms(prediction)
    reference_chains = _split_chains(reference)
    prediction_chains = _split_chains(prediction)
    if len(prediction_chains) != len(reference_chains):
        raise ValueError(
            "prediction/input protein-chain counts differ: "
            f"prediction={len(prediction_chains)}, input={len(reference_chains)}"
        )

    projected = np.full((len(reference), 3), np.nan, dtype=float)
    canonical_res_names = reference.res_name.copy()
    canonical_hetero = (
        reference.hetero.copy()
        if "hetero" in reference.get_annotation_categories()
        and "hetero" in prediction.get_annotation_categories()
        else None
    )
    offset = 0
    for reference_chain, prediction_chain in zip(
        reference_chains, prediction_chains, strict=True
    ):
        _project_chain(
            reference_chain,
            prediction_chain,
            projected,
            canonical_res_names,
            canonical_hetero,
            offset,
        )
        offset += len(reference_chain)

    resolved = np.isfinite(reference.coord).all(axis=-1)
    if "occupancy" in reference.get_annotation_categories():
        resolved &= np.asarray(reference.occupancy) > 0
    matched = resolved & np.isfinite(projected).all(axis=-1)
    coverage = float(matched.sum() / max(resolved.sum(), 1))
    if coverage < min_atom_coverage:
        raise ValueError(
            f"validation atom coverage {coverage:.2%} is below the required "
            f"{min_atom_coverage:.0%}"
        )

    # Score only resolved atoms the oracle can represent. This is important for
    # monomer predictors that canonicalize modified amino acids (for example,
    # phosphothreonine to threonine) and cannot emit modification-group atoms.
    reference = reference[matched]
    reference.res_name = canonical_res_names[matched]
    if canonical_hetero is not None:
        reference.hetero = canonical_hetero[matched]
    remapped = reference.copy()
    remapped.coord = projected[matched]
    return reference, remapped, {"atom_coverage": coverage}
