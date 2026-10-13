"""Actual AtomWorks 3 serialization for both AF3 template input paths."""

from io import StringIO
from pathlib import Path

import numpy as np
import pytest
from biotite.structure.io.pdbx import CIFFile, get_structure

from blocks.networks.prediction.af3_with_target_template import _to_cif_string
from blocks.networks.prediction.helpers import create_af3_json_from_design
from chaperone.io import parse_to_atom_array


@pytest.mark.parametrize("pathway", ["target", "residue_selection"])
def test_real_af3_template_serialization_has_coordinates_and_required_categories(pathway):
    source = Path(__file__).resolve().parents[2] / "examples/benchmarks/ppi/files/4zxb_cropped.pdb"
    atoms = parse_to_atom_array(str(source))
    atoms = atoms[atoms.chain_id == atoms.chain_id[0]]
    first_residues = list(dict.fromkeys(atoms.res_id))[:3]
    atoms = atoms[np.isin(atoms.res_id, first_residues)]
    atoms.chain_id[:] = "A"
    for name in ("occupancy", "b_factor"):
        if name in atoms.get_annotation_categories():
            atoms.del_annotation(name)
    assert atoms.bonds is None  # Exercise bond-less PDB input too.
    if pathway == "target":
        text = _to_cif_string(atoms, "receptor")
    else:
        payload = create_af3_json_from_design(
            atoms, "receptor", template_resi="A0,A1,A2",
        )
        template = payload["sequences"][0]["protein"]["templates"][0]
        assert template["queryIndices"] == template["templateIndices"] == [0, 1, 2]
        text = template["mmcif"]
    parsed = CIFFile.read(StringIO(text))
    block = parsed.block
    assert "entity_poly_seq" in block and "entity_poly" in block
    assert len(block["entity_poly_seq"]["num"].as_array()) == 3
    assert "pdbx_audit_revision_history" in block
    assert block["pdbx_audit_revision_history"]["revision_date"].as_array()[-1] == "2023-08-23"
    assert "occupancy" in block["atom_site"] and "B_iso_or_equiv" in block["atom_site"]
    restored = get_structure(parsed, model=1)
    np.testing.assert_allclose(restored.coord, atoms.coord, atol=1e-3)
    np.testing.assert_array_equal(restored.atom_name, atoms.atom_name)
    assert atoms.bonds is None
    assert "occupancy" not in atoms.get_annotation_categories()
