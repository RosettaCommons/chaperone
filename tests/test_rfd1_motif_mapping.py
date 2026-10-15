"""RFD1 symmetric PDB numbering differs from its TRB residue labels."""

import json
import pickle

import biotite.structure as struc
import numpy as np
import pytest

from blocks.networks.rfdiffusion1 import _extract_trb_info
from chaperone.io import write_pdb


def fixture(tmp_path, indices):
    atoms = struc.AtomArray(6)
    atoms.chain_id = ["A", "A", "B", "B", "C", "C"]
    atoms.res_id = [1, 2, 1, 2, 1, 2]
    atoms.res_name[:] = "ALA"
    atoms.atom_name[:] = "CA"
    atoms.element[:] = "C"
    atoms.coord = np.arange(18).reshape(6, 3)
    path = tmp_path / "design.pdb"
    write_pdb(atoms, path)
    with path.with_suffix(".trb").open("wb") as out:
        pickle.dump(
            {
                "con_ref_pdb_idx": [("A", 7), ("B", 7), ("C", 7)],
                "con_hal_pdb_idx": [("A", 1), ("B", 3), ("C", 5)],
                "con_hal_idx0": indices,
            },
            out,
        )
    return path


def test_symmetric_motif_mapping_uses_emitted_residue_ids(tmp_path):
    result = _extract_trb_info(str(fixture(tmp_path, [0, 2, 4])))
    assert result["motif"] == ["A1", "B1", "C1"]
    assert json.loads(result["diffused_index_map"]) == {
        "A7": "A1",
        "B7": "B1",
        "C7": "C1",
    }


@pytest.mark.parametrize("indices", [[0, 2, 6], [-1, 2, 4], [0, 2]])
def test_invalid_motif_ordinals_fail(tmp_path, indices):
    with pytest.raises(ValueError, match="motif ordinals"):
        _extract_trb_info(str(fixture(tmp_path, indices)))
