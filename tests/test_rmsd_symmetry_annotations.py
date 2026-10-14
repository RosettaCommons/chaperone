"""Real symmetry matching on minimal arrays under AtomWorks 3."""

import numpy as np
from biotite.structure.info import residue

from blocks.metrics.utils import find_automorphisms, get_rmsd, greedy_resolve_symmetries


def test_minimal_aspartate_swapped_oxygens_resolve_without_mutating_input():
    atoms = residue("ASP")
    atoms = atoms[atoms.element != "H"]
    atoms.chain_id[:] = "A"
    atoms.res_id[:] = 1
    categories = atoms.get_annotation_categories()
    coordinates = atoms.coord.copy()
    permutations = find_automorphisms(atoms)
    assert permutations is not None
    assert atoms.get_annotation_categories() == categories
    np.testing.assert_array_equal(atoms.coord, coordinates)
    swapped = coordinates.copy()
    oxygen_indices = np.flatnonzero(np.isin(atoms.atom_name, ["OD1", "OD2"]))
    swapped[oxygen_indices] = swapped[oxygen_indices[::-1]]
    assert get_rmsd(swapped, coordinates) > 0.5
    resolved = greedy_resolve_symmetries(swapped, coordinates, permutations)
    assert get_rmsd(resolved, coordinates) < 1e-6


def test_existing_nonpolymer_instance_boundaries_are_preserved():
    ligand = residue("ACT")
    ligand = ligand[ligand.element != "H"]
    ligand.chain_id[:] = "L"
    first, second = ligand.copy(), ligand.copy()
    first.set_annotation("pn_unit_iid", np.full(len(first), "L_1"))
    second.set_annotation("pn_unit_iid", np.full(len(second), "L_2"))
    atoms = first + second
    identities = atoms.pn_unit_iid.copy()
    permutations = find_automorphisms(atoms)
    assert permutations is not None and len(permutations) == 2
    assert all((p < len(first)).all() or (p >= len(first)).all() for p in permutations)
    np.testing.assert_array_equal(atoms.pn_unit_iid, identities)
