"""Focused tests for the SimpleFold prediction block."""

import json

from blocks import Blocks
from blocks.networks.prediction.simplefold import SimpleFoldBlock
from conftest import make_protein as _protein


def test_simplefold_is_discoverable():
    assert Blocks.get("simplefold") is SimpleFoldBlock


def test_compile_converts_to_cif_without_inline_scoring(tmp_path):
    from atomworks.io.utils.io_utils import to_cif_file
    from chaperone.io import ensure_bonds

    reference_path = tmp_path / "references" / "toy.cif"
    prediction_dir = tmp_path / "outputs" / "predictions_simplefold_100M"
    manifest_dir = tmp_path / "simplefold_manifest"
    reference_path.parent.mkdir()
    prediction_dir.mkdir(parents=True)
    manifest_dir.mkdir()
    to_cif_file(
        ensure_bonds(_protein(chain_id="R", residue_offset=10)),
        reference_path,
    )
    to_cif_file(
        ensure_bonds(_protein()),
        prediction_dir / "toy_sampled_0.cif",
    )
    (manifest_dir / "toy.json").write_text(
        json.dumps(
            {
                "target": "toy",
                "source_path": str(reference_path),
                "reference_path": str(reference_path),
                "sequence": "AGS",
                "pdb_id": "test",
                "assembly_id": "1",
                "pn_unit_iid": "R_1",
                "dropped_nonprotein_atoms": 0,
            }
        )
    )
    block = SimpleFoldBlock(command_template=["true"])

    result = block.compile(tmp_path).unpack()

    assert "simplefold.all_atom_lddt" not in result.columns
    output_path = result.loc[0, "simplefold.path"]
    assert output_path.endswith(".cif")
