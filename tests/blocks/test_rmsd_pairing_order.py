"""Name-paired atoms are compared in the same order even when rows are shuffled."""

import numpy as np
import pytest
from biotite.structure import AtomArray

from blocks.metrics.rmsd import RMSDBlock, _design_order_like_source
from chaperone.data.schema import SELECTIONS, get_selection_mask

LIGAND_ATOMS = [
    ("C1", "C", (2.0, 6.0, 0.0)),
    ("C2", "C", (3.4, 6.3, 0.2)),
    ("N1", "N", (4.1, 7.5, 0.1)),
    ("O1", "O", (3.7, 8.6, -0.2)),
    ("C3", "C", (5.5, 7.4, 0.3)),
    ("S1", "S", (6.4, 6.1, 0.9)),
]


def _structure(ligand_order=None, ligand_chain="B", ligand_res=1, extra=False):
    rows = []
    for i in range(6):
        for j, (name, element) in enumerate(
            [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")]
        ):
            rows.append(
                (
                    "A",
                    i + 1,
                    "GLY",
                    name,
                    element,
                    False,
                    (3.8 * i + 1.2 * j, 0.7 * j, 0.3 * i),
                )
            )
    ligand = list(LIGAND_ATOMS)
    if ligand_order is not None:
        ligand = [ligand[k] for k in ligand_order]
    for name, element, xyz in ligand:
        rows.append((ligand_chain, ligand_res, "LIG", name, element, True, xyz))
    if extra:
        rows.append(
            (ligand_chain, ligand_res, "LIG", "EXTRA", "C", True, (9.0, 9.0, 9.0))
        )
    atoms = AtomArray(len(rows))
    for k, (chain, res, res_name, name, element, hetero, xyz) in enumerate(rows):
        atoms.chain_id[k], atoms.res_id[k], atoms.res_name[k] = chain, res, res_name
        atoms.atom_name[k], atoms.element[k], atoms.hetero[k] = name, element, hetero
        atoms.coord[k] = xyz
    return atoms


def _masks(atoms, ligand_chain="B", ligand_res=1):
    metadata = {"ligand": [f"{ligand_chain}{ligand_res}"]}
    return {
        sele: get_selection_mask(sele, atom_array=atoms, metadata=metadata)
        for sele in (SELECTIONS.BKBN, SELECTIONS.LIGANDS)
    }


def _ligand_rmsd(src, des, **kwargs):
    block = RMSDBlock(
        name="r",
        alignment_methods=[SELECTIONS.BKBN],
        rmsd_targets=[SELECTIONS.LIGANDS],
        select_mask_by_name_pairing=[SELECTIONS.LIGANDS],
        resolve_symmetries=False,
    )
    des_kwargs = {
        k: v for k, v in kwargs.items() if k in ("ligand_chain", "ligand_res")
    }
    metrics = block._compute_alignments(
        src, _masks(src), des, _masks(des, **des_kwargs)
    )
    return next(v for k, v in metrics.items() if k.endswith("ligand_rmsd"))


def test_matching_structures_need_no_reordering():
    src = _structure()

    order = _design_order_like_source(src, _structure())

    assert np.array_equal(order, np.arange(len(src)))


def test_shuffled_ligand_rows_are_put_back_in_source_order():
    src, des = _structure(), _structure(ligand_order=[3, 1, 0, 5, 2, 4])

    reordered = des[_design_order_like_source(src, des)]

    assert list(reordered.atom_name) == list(src.atom_name)
    assert np.allclose(reordered.coord, src.coord)


def test_shuffled_ligand_has_zero_rmsd():
    assert _ligand_rmsd(
        _structure(), _structure(ligand_order=[3, 1, 0, 5, 2, 4])
    ) == pytest.approx(0.0, abs=1e-5)


def test_same_order_is_unchanged():
    assert _ligand_rmsd(_structure(), _structure()) == pytest.approx(0.0, abs=1e-5)


def test_renumbered_ligand_falls_back_to_residue_name_and_is_still_ordered():
    des = _structure(ligand_order=[5, 4, 3, 2, 1, 0], ligand_chain="C", ligand_res=7)

    assert _ligand_rmsd(
        _structure(), des, ligand_chain="C", ligand_res=7
    ) == pytest.approx(0.0, abs=1e-5)


def test_unmatched_design_atoms_go_last_in_their_original_order():
    src, des = _structure(), _structure(extra=True)

    order = _design_order_like_source(src, des)

    assert order[-1] == len(des) - 1
    assert list(des[order].atom_name[:-1]) == list(src.atom_name)


def test_genuinely_displaced_ligand_still_reports_its_distance():
    src, des = _structure(), _structure(ligand_order=[3, 1, 0, 5, 2, 4])
    des.coord[des.hetero] += np.array([0.0, 0.0, 2.0])

    assert _ligand_rmsd(src, des) == pytest.approx(2.0, abs=1e-5)
