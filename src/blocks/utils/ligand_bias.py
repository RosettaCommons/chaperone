"""Geometry-derived per-position biases for ligand-aware inverse folding."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from biotite.structure import filter_amino_acids, get_residue_starts


DEFAULT_POCKET_BIAS = {
    "F": 1.0,
    "Y": 0.8,
    "W": 0.8,
    "L": 0.35,
    "I": 0.35,
    "V": 0.35,
    "M": 0.25,
}
DEFAULT_POLAR_BIAS = {
    "S": 1.2,
    "T": 1.2,
    "N": 1.0,
    "Q": 1.0,
    "H": 0.8,
    "D": 0.5,
    "E": 0.5,
    "Y": 0.5,
}
DEFAULT_SUGAR_OXYGENS = frozenset({"O2'", "O3'", "O5'"})
STANDARD_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


def _canonical_atom_name(name: str) -> str:
    return str(name).strip().replace("*", "'").replace("’", "'")


def _proximity_scale(distance: float, radius: float) -> float:
    if radius <= 0:
        raise ValueError("proximity radii must be positive")
    if radius <= 2.0:
        return float(distance <= radius)
    return max(0.0, min(1.0, (radius - distance) / (radius - 2.0)))


def ligand_proximity_bias(
    atom_array,
    *,
    ligand_resnames: Iterable[str],
    pocket_radius: float = 8.0,
    polar_radius: float = 5.0,
    polar_atom_names: Iterable[str] = DEFAULT_SUGAR_OXYGENS,
    pocket_bias: Mapping[str, float] = DEFAULT_POCKET_BIAS,
    polar_bias: Mapping[str, float] = DEFAULT_POLAR_BIAS,
    position_bias: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, dict[str, float]]:
    """Build LigandMPNN logit biases from the current ligand geometry.

    Every protein position contacting the ligand receives a modest
    hydrophobic/aromatic bias. Positions near the named polar ligand atoms
    additionally receive a hydrogen-bond-capable residue bias. Overlapping
    contributions are added, allowing Tyr to serve both roles without forcing
    any identity. Optional position-specific contributions are also additive:
    they can encode a known motif as preferences while leaving MPNN free to
    choose another identity. Fixed-position constraints remain authoritative.
    """
    ligand_names = {str(name) for name in ligand_resnames}
    polar_names = {_canonical_atom_name(name) for name in polar_atom_names}
    heavy = np.asarray(atom_array.element) != "H"
    ligand_mask = np.isin(np.asarray(atom_array.res_name), list(ligand_names)) & heavy
    ligand = atom_array[ligand_mask]
    if len(ligand) == 0:
        raise ValueError(f"No ligand atoms found for residue names {sorted(ligand_names)}")
    polar_mask = np.asarray(
        [_canonical_atom_name(name) in polar_names for name in ligand.atom_name]
    )
    polar = ligand[polar_mask]
    if len(polar) == 0:
        raise ValueError(f"No requested polar ligand atoms found: {sorted(polar_names)}")

    protein = atom_array[filter_amino_acids(atom_array) & heavy]
    if len(protein) == 0:
        raise ValueError("No protein residues found")
    starts = get_residue_starts(protein, add_exclusive_stop=True)
    result: dict[str, dict[str, float]] = {}
    protein_positions: set[str] = set()
    for start, end in zip(starts[:-1], starts[1:]):
        residue = protein[start:end]
        key = f"{residue.chain_id[0]}{int(residue.res_id[0])}"
        protein_positions.add(key)
        ligand_distance = float(
            np.linalg.norm(residue.coord[:, None, :] - ligand.coord[None, :, :], axis=-1).min()
        )
        polar_distance = float(
            np.linalg.norm(residue.coord[:, None, :] - polar.coord[None, :, :], axis=-1).min()
        )
        bias: dict[str, float] = {}
        if ligand_distance <= pocket_radius:
            pocket_scale = _proximity_scale(ligand_distance, pocket_radius)
            bias.update(
                {
                    aa: float(value) * pocket_scale
                    for aa, value in pocket_bias.items()
                    if float(value) * pocket_scale > 0
                }
            )
        if polar_distance <= polar_radius:
            polar_scale = _proximity_scale(polar_distance, polar_radius)
            for aa, value in polar_bias.items():
                contribution = float(value) * polar_scale
                if contribution > 0:
                    bias[aa] = bias.get(aa, 0.0) + contribution
        if bias:
            result[key] = dict(sorted(bias.items()))

    for position, additions in (position_bias or {}).items():
        if position not in protein_positions:
            raise ValueError(f"Position-specific bias refers to missing residue {position!r}")
        merged = result.setdefault(position, {})
        for amino_acid, raw_value in additions.items():
            amino_acid = str(amino_acid).upper()
            value = float(raw_value)
            if amino_acid not in STANDARD_AMINO_ACIDS:
                raise ValueError(f"Invalid amino-acid bias key {amino_acid!r}")
            if not math.isfinite(value):
                raise ValueError(
                    f"Non-finite bias for {position}:{amino_acid}: {raw_value!r}"
                )
            if value:
                merged[amino_acid] = merged.get(amino_acid, 0.0) + value
        result[position] = dict(sorted(merged.items()))
    return dict(sorted(result.items()))


def write_ligand_proximity_bias(
    structure: str | Path,
    output: str | Path,
    **kwargs,
) -> dict[str, dict[str, float]]:
    """Write a geometry-derived LigandMPNN ``bias_AA_per_residue`` JSON."""
    from chaperone.io import parse_to_atom_array

    bias = ligand_proximity_bias(parse_to_atom_array(str(structure)), **kwargs)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bias, indent=2, sort_keys=True) + "\n")
    return bias
