"""PEPPR must compare the explicitly selected structure states."""

from types import SimpleNamespace

from blocks.metrics.peppr import PepprBlock


def test_peppr_reference_can_be_the_packed_designed_sequence(monkeypatch):
    reference, prediction = object(), object()
    calls = []

    def score(first, last, **kwargs):
        calls.append((first, last, kwargs))
        return {"dockq": 0.7}

    monkeypatch.setattr("blocks.metrics.peppr.peppr_metrics", score)
    block = PepprBlock(which=[-2, -1], metrics=["dockq"], tolerate_exceptions=False)
    assert block.get_data_kwargs() == {"which": [-2, -1], "load": True}
    result = block.compute_metrics(
        (
            SimpleNamespace(atom_array=reference, path="packed.cif"),
            SimpleNamespace(atom_array=prediction, path="af3.cif"),
        )
    )
    assert result == {"dockq": 0.7}
    assert calls[0][:2] == (reference, prediction)
    assert calls[0][2]["tolerate_exceptions"] is False
    assert PepprBlock().get_data_kwargs()["which"] == [0, -1]


def test_peppr_accepts_protein_structures_loaded_without_pdb_bonds():
    import numpy as np
    from biotite.structure import concatenate
    from biotite.structure.info import residue
    from blocks.metrics.peppr import peppr_metrics

    monomer = residue("ALA")
    monomer = monomer[monomer.element != "H"]
    components = []
    for chain, y in [("A", 0), ("B", 5)]:
        for number in range(1, 4):
            part = monomer.copy()
            part.chain_id[:] = chain
            part.res_id[:] = number
            part.coord += np.array([4 * number, y, 0])
            components.append(part)
    atoms = concatenate(components)
    atoms.bonds = None
    result = peppr_metrics(
        atoms, atoms, metrics=["fnat", "irmsd", "dockq"], tolerate_exceptions=False
    )
    assert all(np.isfinite(value) for value in result.values())
    assert result["dockq"] > 0.99
    assert atoms.bonds is None
