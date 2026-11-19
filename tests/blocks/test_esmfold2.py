"""Focused tests for the ESMFold2 prediction block."""

from pathlib import Path
import runpy

import numpy as np
from atomworks.enums import ChainType
from biotite.structure.info import residue

from blocks.networks.prediction.esmfold2 import ESMFold2Block
from blocks.networks.prediction.helpers import create_esmfold2_spec_from_design

RUNNER = Path(__file__).parents[2] / "src/blocks/scripts/run_esmfold2_runner.py"


def test_compile_uses_stable_block_name_for_all_samples(tmp_path):
    outputdir = tmp_path / "outputs"
    outputdir.mkdir()
    first = outputdir / "toy_sample0.cif"
    second = outputdir / "toy_sample1.cif"
    first.write_text("data_first\n")
    second.write_text("data_second\n")
    (outputdir / "summary.csv").write_text(
        "name,length,mean_plddt,ptm,iptm\n"
        "toy_sample0,10,0.7,0.6,0.5\n"
        "toy_sample1,10,0.8,0.7,0.6\n"
    )

    result = ESMFold2Block(command_template=["true"]).compile(tmp_path).unpack()

    assert result["esmfold2.path"].tolist() == [str(first), str(second)]
    assert result["esmfold2.plddt"].tolist() == [0.7, 0.8]


def test_spec_keeps_the_ccd_code_of_ccd_ligands():
    parts = []
    for name, chain, chain_type in [
        ("ALA", "A", ChainType.POLYPEPTIDE_L),
        ("HEM", "B", ChainType.NON_POLYMER),
        ("EOH", "C", ChainType.NON_POLYMER),
    ]:
        part = residue(name)
        part.chain_id[:] = chain
        part.set_annotation("chain_type", np.full(len(part), chain_type))
        parts.append(part)
    parts[2].res_name[:] = "LIG"
    atoms = parts[0] + parts[1] + parts[2]

    spec = create_esmfold2_spec_from_design(
        atoms, "design_0", specified_smiles={"LIG": "CCO"}
    )

    heme, ethanol = spec["ligands"]
    assert heme["id"] == "B" and heme["ccd"] == ["HEM"] and heme["smiles"]
    assert ethanol == {"id": "C", "smiles": "CCO"}


def test_runner_folds_ccd_ligands_from_esmfold2_ccd():
    build_input = runpy.run_path(str(RUNNER))["_build_input"]
    spec = {
        "ligands": [
            {"id": "B", "smiles": "heme", "ccd": ["HEM"]},
            {"id": "C", "smiles": "fallback", "ccd": ["NEW"]},
            {"id": "D", "smiles": "CCO"},
        ]
    }

    built = build_input(spec, (dict, dict, dict), {"HEM"})

    assert built["sequences"] == [
        {"id": "B", "ccd": ["HEM"]},
        {"id": "C", "smiles": "fallback"},
        {"id": "D", "smiles": "CCO"},
    ]


def test_runner_keeps_smiles_when_esm_has_no_ccd():
    build_input = runpy.run_path(str(RUNNER))["_build_input"]
    spec = {"ligands": [{"id": "B", "smiles": "heme", "ccd": ["HEM"]}]}

    built = build_input(spec, (dict, dict, dict))

    assert built["sequences"] == [{"id": "B", "smiles": "heme"}]
