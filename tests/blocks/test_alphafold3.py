"""Focused tests for AtomArray-to-AF3 mmCIF preparation."""

import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
from atomworks.enums import ChainType
from biotite import structure as struc
from biotite.structure.io.pdbx import CIFFile

from blocks.networks.prediction.alphafold3 import (
    AlphaFold3Block,
    create_af3_mmcif_payload,
)


def _covalent_design():
    atom_array = struc.AtomArray(5)
    atom_array.chain_id = np.array(["A", "A", "A", "B", "B"])
    atom_array.res_id = np.array([42, 42, 42, 7, 7])
    atom_array.res_name = np.array(["CYS", "CYS", "CYS", "L:1", "L:1"])
    atom_array.hetero = np.array([False, False, False, True, True])
    atom_array.atom_name = np.array(["CA", "CB", "SG", "C1", "O1"])
    atom_array.element = np.array(["C", "C", "S", "C", "O"])
    atom_array.coord = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [2.8, 0.0, 0.0],
            [4.5, 0.0, 0.0],
            [5.7, 0.0, 0.0],
        ]
    )
    atom_array.set_annotation("charge", np.zeros(5, dtype=int))
    atom_array.set_annotation(
        "chain_type",
        np.array(
            [ChainType.POLYPEPTIDE_L] * 3 + [ChainType.NON_POLYMER] * 2,
            dtype=int,
        ),
    )
    atom_array.set_annotation("is_polymer", np.array([True, True, True, False, False]))
    atom_array.set_annotation(
        "chem_comp_type",
        np.array(["L-PEPTIDE LINKING"] * 3 + ["NON-POLYMER"] * 2),
    )
    atom_array.set_annotation(
        "chain_iid", np.array(["A_1", "A_1", "A_1", "B_1", "B_1"])
    )
    atom_array.bonds = struc.BondList(
        5,
        np.array(
            [
                [0, 1, int(struc.BondType.SINGLE)],
                [1, 2, int(struc.BondType.SINGLE)],
                [2, 3, int(struc.BondType.SINGLE)],
                [3, 4, int(struc.BondType.DOUBLE)],
            ]
        ),
    )
    return atom_array


def test_mmcif_payload_preserves_covalent_bond_and_custom_ccd():
    payload = create_af3_mmcif_payload(_covalent_design(), "covalent")

    structure_cif = CIFFile.read(io.StringIO(payload["cif"]))
    structure_block = structure_cif.block
    struct_conn = structure_block["struct_conn"]
    assert struct_conn["ptnr1_label_seq_id"].as_array(int).tolist() == [1]
    assert struct_conn["ptnr2_label_seq_id"].as_array(int).tolist() == [1]

    user_ccd = CIFFile.read(io.StringIO(payload["user_ccd"]))
    ligand_block = user_ccd["L:1"]
    assert ligand_block["chem_comp_atom"]["atom_id"].as_array(str).tolist() == [
        "C1",
        "O1",
    ]
    assert ligand_block["chem_comp_bond"]["value_order"].as_array(str).tolist() == [
        "DOUB"
    ]

    atom_site = structure_block["atom_site"]
    assert "0" not in atom_site["label_entity_id"].as_array(str)
    assert structure_block["entity_poly"]["entity_id"].as_array(str).tolist() == ["1"]
    assert structure_block["entity_poly"]["type"].as_array(str).tolist() == [
        "polypeptide(L)"
    ]
    assert structure_block["entity"]["type"].as_array(str).tolist() == [
        "polymer",
        "non-polymer",
    ]
    assert structure_block["struct_asym"]["id"].as_array(str).tolist() == ["A", "B"]
    assert structure_block["entity_poly_seq"]["mon_id"].as_array(str).tolist() == [
        "CYS"
    ]
    assert structure_block["pdbx_poly_seq_scheme"]["asym_id"].as_array(
        str
    ).tolist() == ["A"]
    assert structure_block["pdbx_nonpoly_scheme"]["asym_id"].as_array(str).tolist() == [
        "B"
    ]


def test_mmcif_payload_can_remove_ligands():
    payload = create_af3_mmcif_payload(_covalent_design(), "apo", remove_ligands=True)

    assert "user_ccd" not in payload
    structure_cif = CIFFile.read(io.StringIO(payload["cif"]))
    assert "struct_conn" not in structure_cif.block


def test_conflicting_custom_components_fail_loudly():
    design = _covalent_design()
    second = design[design.chain_id == "B"].copy()
    second.chain_id[:] = "C"
    second.chain_iid[:] = "C_1"
    second.atom_name[1] = "N1"
    duplicated = struc.concatenate([design, second])

    with pytest.raises(ValueError, match="different atom/bond topologies"):
        create_af3_mmcif_payload(duplicated, "conflicting")


def test_monoatomic_component_relies_on_af3_builtin_ccd():
    design = _covalent_design()
    ligand = design[design.chain_id == "B"][:1]
    monoatomic = struc.concatenate([design[design.chain_id == "A"], ligand])

    payload = create_af3_mmcif_payload(monoatomic, "monoatomic")

    assert "user_ccd" not in payload


def test_builtin_unk_placeholder_does_not_create_user_ccd():
    design = _covalent_design()[_covalent_design().chain_id == "A"]
    design.res_name[:] = "UNK"

    payload = create_af3_mmcif_payload(design, "unknown_residue")

    assert "user_ccd" not in payload


def test_unresolved_residue_keeps_full_sequence_position():
    design = _covalent_design()[_covalent_design().chain_id == "A"]
    missing = design.copy()
    missing.res_id[:] = 1
    missing.res_name[:] = "ALA"
    missing.coord[:] = np.nan
    observed = design.copy()
    observed.res_id[:] = 2
    observed.res_name[:] = "CYS"
    observed.ins_code[:] = "A"
    assembly = struc.concatenate([missing, observed])

    payload = create_af3_mmcif_payload(assembly, "missing_n_terminus")

    structure = CIFFile.read(io.StringIO(payload["cif"])).block
    assert structure["entity_poly_seq"]["mon_id"].as_array(str).tolist() == [
        "ALA",
        "CYS",
    ]
    assert set(structure["atom_site"]["label_seq_id"].as_array(int)) == {2}
    assert set(structure["atom_site"]["auth_seq_id"].as_array(int)) == {2}
    assert structure["pdbx_poly_seq_scheme"]["pdb_ins_code"].as_array(str).tolist() == [
        ".",
        "A",
    ]


def test_repeated_chain_ids_are_disambiguated_by_chain_iid():
    design = _covalent_design()
    copy = design[design.chain_id == "A"].copy()
    copy.chain_iid[:] = "A_2"
    repeated = struc.concatenate([design, copy])

    payload = create_af3_mmcif_payload(repeated, "assembly")

    chain_map = payload["af3_chain_id_to_original_iid"]
    assert set(chain_map.values()) == {"A_1", "A_2", "B_1"}
    assert len(chain_map) == 3


def test_d_polypeptide_is_retained_as_bonded_nonpolymer_for_af3():
    design = _covalent_design()
    d_chain = design[design.chain_id == "A"].copy()
    d_chain.chain_id[:] = "C"
    d_chain.chain_iid[:] = "C_1"
    d_chain.chain_type[:] = ChainType.POLYPEPTIDE_D
    assembly = struc.concatenate([design, d_chain])

    payload = create_af3_mmcif_payload(assembly, "d_peptide")

    structure = CIFFile.read(io.StringIO(payload["cif"])).block
    assert structure["entity"]["type"].as_array(str).tolist() == [
        "polymer",
        "non-polymer",
        "non-polymer",
    ]
    assert structure["entity_poly"]["entity_id"].as_array(str).tolist() == ["1"]
    assert structure["pdbx_nonpoly_scheme"]["asym_id"].as_array(str).tolist() == [
        "B",
        "C",
    ]


def test_af3_mmcif_is_default_and_json_is_explicit_compatibility_mode():
    block = AlphaFold3Block()
    assert block.input_mode == "mmcif"
    assert block._use_mmcif_input()
    assert not AlphaFold3Block(input_mode="json")._use_mmcif_input()

    auto_block = AlphaFold3Block(input_mode="auto", specified_smiles={"LIG": "C"})
    assert not auto_block._use_mmcif_input()

    with pytest.raises(ValueError, match="specified_smiles"):
        AlphaFold3Block(specified_smiles={"LIG": "C"})


def test_af3_block_stages_and_selects_mmcif_manifest(tmp_path):
    loaded = SimpleNamespace(
        atom_array=_covalent_design(), basename="design", path="/input/design.cif"
    )
    handler = SimpleNamespace(get_data=lambda **_kwargs: iter([loaded]))
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()
    output_dir.mkdir()

    block = AlphaFold3Block()
    manifest_path = block._prepare_af3_inputs(handler, input_dir)
    command = block._build_af3_command(
        input_dir, output_dir, input_manifest=manifest_path
    )

    payloads = json.loads(manifest_path.read_text())
    assert [payload["name"] for payload in payloads] == ["design"]
    assert f"--input_manifest={manifest_path}" in command
    assert not any(arg.startswith("--input_dir=") for arg in command)


def test_af3_mmcif_manifest_can_include_msas(tmp_path, monkeypatch):
    loaded = SimpleNamespace(
        atom_array=_covalent_design(), basename="design", path="/input/design.cif"
    )
    handler = SimpleNamespace(get_data=lambda **_kwargs: iter([loaded]))
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    msa = tmp_path / "source.a3m"
    msa.write_text(">query\nC\n>hit\nC\n")

    monkeypatch.setattr(
        "blocks.networks.prediction.alphafold3.resolve_msa_path",
        lambda *_args, **_kwargs: msa,
    )
    block = AlphaFold3Block(use_local_msas=True, msa_dirs=[str(tmp_path)])
    manifest_path = block._prepare_af3_inputs(handler, input_dir)

    payload = json.loads(manifest_path.read_text())[0]
    assert payload["unpaired_msa"] == {"A": ">query\nC\n>hit\nC\n"}
    metadata = json.loads((tmp_path / "af3_manifest" / "design.json").read_text())
    assert metadata["msa_depth"] == 2
    assert metadata["msa_depths"] == {"A": 2}
