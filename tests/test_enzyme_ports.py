"""Synthetic enzyme inputs exercise AF3 staging without restricted fixtures."""
import json

import pytest

from blocks.networks.prediction.helpers import (
    create_af3_json_from_design,
    fill_af3_json_from_template,
)
from chaperone.io import parse_to_atom_array

@pytest.fixture
def enzyme_inputs(tmp_path):
    import numpy as np
    from biotite.structure import AtomArray
    from chaperone.io import write_pdb

    names = ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"]
    residues = [77, 86, 88, 122, 136, 177]
    atoms = AtomArray(len(names) * len(residues))
    atoms.chain_id[:] = "A"
    atoms.res_name[:] = "GLU"
    atoms.res_id = np.repeat(residues, len(names))
    atoms.atom_name = np.tile(names, len(residues))
    atoms.element = np.tile(["N", "C", "C", "O", "C", "C", "C", "O", "O"], len(residues))
    # Arbitrary test coordinates; this is not a catalytic structure.
    atoms.coord = np.random.default_rng(13).normal(size=(len(atoms), 3)) * 5
    write_pdb(atoms, tmp_path / "theozyme.pdb")
    template = {
        "name": "synthetic", "modelSeeds": [1], "dialect": "alphafold3", "version": 1,
        "sequences": [{"protein": {"id": "A", "sequence": None, "modifications": []}}],
        "userCCD": "data_L:1\n# synthetic payload; input-assembly test only\nOE2 C9 SING\n",
    }
    (tmp_path / "af3_template_ptm.json").write_text(json.dumps(template))
    template["sequences"].append({"ligand": {"id": "B", "ccdCodes": ["L:1"]}})
    (tmp_path / "af3_template_ligand.json").write_text(json.dumps(template))
    peptide = AtomArray(10)
    peptide.chain_id[:] = "B"
    peptide.res_id = np.arange(1, 11)
    peptide.res_name[:] = "GLY"
    peptide.atom_name[:] = "CA"
    peptide.element[:] = "C"
    peptide.coord = np.arange(30).reshape(10, 3)
    ligand = AtomArray(2)
    ligand.chain_id[:] = "C"
    ligand.res_id[:] = 1
    ligand.res_name[:] = "ZnO"
    ligand.atom_name = np.array(["ZN", "O"])
    ligand.element = np.array(["ZN", "O"])
    ligand.hetero[:] = True
    ligand.coord = np.array([[0., 0., 0.], [0., 0., 2.]])
    write_pdb(atoms + peptide + ligand, tmp_path / "complex.pdb")
    return tmp_path


def test_glycosidase_ptm_uses_remapped_catalytic_residue_and_keeps_custom_chemistry(enzyme_inputs):
    files = enzyme_inputs
    template = files / "af3_template_ptm.json"
    original = json.loads(template.read_text())
    atoms = parse_to_atom_array(files / "theozyme.pdb")
    # Generation changes residue numbering: attach at output position 42,
    # not the original theozyme position 86.
    sequence = "G" * 41 + "E" + "G" * 58
    result = fill_af3_json_from_template(
        template, atoms, "generated", json.dumps({"A86": "A42"}),
        add_ptm_at_mapped_residues={"L:1": "A86"},
        protein_sequences={"A": sequence},
    )
    protein = result["sequences"][0]["protein"]
    assert protein["sequence"][protein["modifications"][0]["ptmPosition"] - 1] == "E"
    assert protein["modifications"] == [{"ptmType": "L:1", "ptmPosition": 42}]
    assert result["userCCD"] == original["userCCD"]
    assert "OE2 C9 SING" in result["userCCD"]  # glycosyl–glutamate covalent bond
    assert json.loads(template.read_text()) == original
    with pytest.raises(KeyError):
        fill_af3_json_from_template(
            template, atoms, "missing", {},
            add_ptm_at_mapped_residues={"L:1": "A86"},
        )


def test_glycosidase_ligand_branch_preserves_separate_substrate(enzyme_inputs):
    files = enzyme_inputs
    atoms = parse_to_atom_array(files / "theozyme.pdb")
    result = fill_af3_json_from_template(
        files / "af3_template_ligand.json", atoms, "ligand", {},
        protein_sequences={"A": "GEG"},
    )
    assert result["sequences"][0]["protein"]["sequence"] == "GEG"
    assert result["sequences"][0]["protein"]["modifications"] == []
    assert result["sequences"][1] == {"ligand": {"id": "B", "ccdCodes": ["L:1"]}}


def test_protease_complex_smiles_matches_actual_ligand_and_apo_has_no_peptide(enzyme_inputs):
    files = enzyme_inputs
    atoms = parse_to_atom_array(files / "complex.pdb")
    complex_input = create_af3_json_from_design(
        atoms, "complex", specified_smiles={"ZnO": "[Zn+2].O"},
    )
    proteins = [x["protein"] for x in complex_input["sequences"] if "protein" in x]
    assert {x["id"] for x in proteins} == {"A", "B"}
    assert len(next(x["sequence"] for x in proteins if x["id"] == "B")) == 10
    ligands = [x["ligand"] for x in complex_input["sequences"] if "ligand" in x]
    assert len(ligands) == 1 and ligands[0]["smiles"] == "[Zn+2].O"
    apo = create_af3_json_from_design(atoms[atoms.chain_id == "A"], "apo", remove_ligands=True)
    assert len(apo["sequences"]) == 1
    assert apo["sequences"][0]["protein"]["id"] == "A"


def test_glycosidase_branch_rewind_preserves_packed_design_and_motif_map(enzyme_inputs):
    from chaperone import PipelineDataHandler
    from chaperone.data import make_row

    path = enzyme_inputs / "theozyme.pdb"
    designed = PipelineDataHandler([make_row(
        "mpnn", path, design_id="designed",
        metadata={"diffused_index_map": json.dumps({"A86": "A86"})},
    )])
    predicted = PipelineDataHandler.merge(designed, PipelineDataHandler([make_row(
        "af3_apo", path, design_id="predicted", parent_design_id="designed",
        path_handle=str(path), metrics={"af3_apo": {"plddt": 90.}},
    )]))
    restored = predicted.rewind_to("mpnn")
    assert restored.names == ["mpnn"]
    item = next(restored.get_data(load=True))
    assert item.path == str(path)
    assert json.loads(item.metadata["diffused_index_map"]) == {"A86": "A86"}
    assert item.metrics["af3_apo"]["plddt"] == 90.


def test_ptm_catalytic_metric_counts_modified_glutamate_and_detects_missing_atoms(enzyme_inputs):
    from types import SimpleNamespace
    from blocks.metrics.motif_reference import ReferenceMotifRMSDBlock

    path = enzyme_inputs / "theozyme.pdb"
    reference = parse_to_atom_array(path)
    residues = ["A77", "A86", "A88", "A122", "A136", "A177"]
    modified = reference.copy()
    glu = (modified.chain_id == "A") & (modified.res_id == 86)
    modified.res_name[glu] = "L:1"
    modified.hetero[glu] = True
    source = SimpleNamespace(metadata={"diffused_index_map": dict(zip(residues, residues))}, path=str(path))
    target = SimpleNamespace(atom_array=modified)
    block = ReferenceMotifRMSDBlock(reference=str(path), residues=residues)
    metric = block._compute(reference, source, target, set(residues))
    assert metric["n_residues"] == 6 and metric["n_atoms"] == 54
    assert metric["motif_rmsd"] == pytest.approx(0., abs=1e-5)
    # The public PTM filter requires all synthetic reference atoms; dropping the catalytic oxygen
    # cannot silently improve the RMSD and still pass the coverage gate.
    target.atom_array = modified[~(glu & (modified.atom_name == "OE2"))]
    missing = block._compute(reference, source, target, set(residues))
    assert missing["n_atoms"] == 53
