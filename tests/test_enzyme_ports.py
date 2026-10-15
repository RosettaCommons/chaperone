"""Real bundled chemistry and AF3 staging contracts for the enzyme recipes."""
import json
from pathlib import Path

import pytest

from blocks.networks.prediction.helpers import (
    create_af3_json_from_design,
    fill_af3_json_from_template,
)
from chaperone.io import parse_to_atom_array

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_glycosidase_ptm_uses_remapped_catalytic_residue_and_keeps_custom_chemistry():
    files = EXAMPLES / "enzyme-glycosidase/files"
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


def test_glycosidase_ligand_branch_preserves_separate_substrate():
    files = EXAMPLES / "enzyme-glycosidase/files"
    atoms = parse_to_atom_array(files / "theozyme.pdb")
    result = fill_af3_json_from_template(
        files / "af3_template_ligand.json", atoms, "ligand", {},
        protein_sequences={"A": "GEG"},
    )
    assert result["sequences"][0]["protein"]["sequence"] == "GEG"
    assert result["sequences"][0]["protein"]["modifications"] == []
    assert result["sequences"][1] == {"ligand": {"id": "B", "ccdCodes": ["L:1"]}}


def test_protease_complex_smiles_matches_actual_ligand_and_apo_has_no_peptide():
    files = EXAMPLES / "enzyme-protease/files"
    atoms = parse_to_atom_array(files / "GPRVVERHQS78.pdb")
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


@pytest.mark.parametrize("case", ["enzyme-protease", "enzyme-glycosidase"])
def test_enzyme_specifications_reference_bundled_inputs(case):
    files = EXAMPLES / case / "files"
    specs = json.loads((files / ("protease.json" if "protease" in case else "glycosidase.json")).read_text())
    assert len(specs) == 2
    for spec in specs.values():
        atoms = parse_to_atom_array(files / spec["input"])
        assert len(atoms) > 0
        if "contig" in spec:
            assert "A28-A30" not in spec["contig"]
        else:
            catalytic = atoms[(atoms.chain_id == "A") & (atoms.res_id == 86)]
            assert set(catalytic.res_name) == {"GLU"}


def test_glycosidase_branch_rewind_preserves_packed_design_and_motif_map():
    from chaperone import PipelineDataHandler
    from chaperone.data import make_row

    path = EXAMPLES / "enzyme-glycosidase/files/theozyme.pdb"
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


def test_ptm_catalytic_metric_counts_modified_glutamate_and_detects_missing_atoms():
    from types import SimpleNamespace
    from blocks.metrics.motif_reference import ReferenceMotifRMSDBlock

    path = EXAMPLES / "enzyme-glycosidase/files/theozyme.pdb"
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
    assert metric["n_residues"] == 6 and metric["n_atoms"] == 38
    assert metric["motif_rmsd"] == pytest.approx(0., abs=1e-5)
    # The public PTM filter requires all 38 atoms; dropping the catalytic oxygen
    # cannot silently improve the RMSD and still pass the coverage gate.
    target.atom_array = modified[~(glu & (modified.atom_name == "OE2"))]
    missing = block._compute(reference, source, target, set(residues))
    assert missing["n_atoms"] == 37


def test_ame_phosphotriesterase_origin_is_valid_pdb_and_conditioning_atoms_exist():
    """The virtual origin must not spill its name into PDB element/charge columns."""
    import numpy as np
    from biotite.structure.io.pdb import PDBFile

    files = EXAMPLES / "benchmarks/ame/files"
    spec = json.loads((files / "ame-2.json").read_text())["phosphotriesterase"]
    # AtomWorks also builds bonds using PDB serials. The virtual serial 999
    # must follow the real atoms (1..101), not interrupt them before ligand 63.
    serials = [int(line[6:11]) for line in (files / spec["input"]).read_text().splitlines()
               if line.startswith(("ATOM  ", "HETATM"))]
    assert serials == sorted(set(serials))
    # Foundry requests charges while reading inputs. Without this extra field,
    # the malformed historical ORI marker could escape a lightweight PDB check.
    atoms = PDBFile.read(files / spec["input"]).get_structure(
        model=1, extra_fields=["charge", "occupancy", "b_factor"], include_bonds=True
    )
    assert np.isfinite(atoms.coord).all()
    origin = atoms[atoms.res_name == "ORI"]
    assert len(origin) == 1 and origin.element.tolist() == ["X"]
    np.testing.assert_allclose(origin.coord[0], spec["ori_token"], atol=1e-6)
    ligand = atoms[atoms.res_name == spec["ligand"]]
    assert set(ligand.atom_name[ligand.element == "ZN"]) == {"ZN1", "ZN2"}
    for residue, names in spec["select_fixed_atoms"].items():
        selected = atoms[(atoms.chain_id == residue[0]) & (atoms.res_id == int(residue[1:]))]
        assert set(names.split(",")) <= set(selected.atom_name)
    for field in ("select_buried", "select_partially_buried", "select_exposed"):
        assert set(spec[field][spec["ligand"]].split(",")) <= set(ligand.atom_name)
