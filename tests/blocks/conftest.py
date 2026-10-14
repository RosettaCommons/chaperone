"""Shared fixtures for the monomer-predictor block tests."""

import numpy as np
import pytest
from biotite.structure import AtomArray


def make_protein(chain_id: str = "A", residue_offset: int = 1) -> AtomArray:
    """Build a three-residue single-chain protein with full occupancy."""
    residues = [
        ("ALA", ["N", "CA", "C", "O", "CB"]),
        ("GLY", ["N", "CA", "C", "O"]),
        ("SER", ["N", "CA", "C", "O", "CB", "OG"]),
    ]
    n_atoms = sum(len(atoms) for _, atoms in residues)
    array = AtomArray(n_atoms)
    array.chain_id = np.full(n_atoms, chain_id)
    array.res_id = np.concatenate(
        [
            np.full(len(atoms), residue_offset + i)
            for i, (_, atoms) in enumerate(residues)
        ]
    )
    array.res_name = np.concatenate(
        [np.full(len(atoms), name) for name, atoms in residues]
    )
    array.atom_name = np.concatenate([np.asarray(atoms) for _, atoms in residues])
    array.element = np.asarray([name[0] for name in array.atom_name])
    array.hetero = np.zeros(n_atoms, dtype=bool)
    array.set_annotation("occupancy", np.ones(n_atoms))
    array.coord = np.asarray(
        [
            (3.1 * residue_index, 0.8 * atom_index, 0.3 * (atom_index % 2))
            for residue_index, (_, atoms) in enumerate(residues)
            for atom_index in range(len(atoms))
        ]
    )
    return array


@pytest.fixture
def protein():
    """Factory fixture returning :func:`make_protein`."""
    return make_protein
