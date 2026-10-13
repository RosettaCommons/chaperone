"""generate_params fails clearly when Rosetta's molfile_to_params.py is missing."""

from unittest.mock import patch

import pytest

pytest.importorskip("rdkit")

from atomworks.io.tools.rdkit import atom_array_from_rdkit  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

from blocks.utils.params import generate_params  # noqa: E402


@pytest.fixture
def ethanol():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    array = atom_array_from_rdkit(mol)
    array.res_name[:] = "LIG"
    return array


@pytest.mark.parametrize(
    ("setting", "message"),
    [(None, "is not set"), ("/nonexistent/molfile_to_params.py", "does not exist")],
)
def test_missing_script_names_the_setting_and_the_fix(
    ethanol, tmp_path, setting, message
):
    with patch("chaperone.get_setting_field", return_value=setting):
        with pytest.raises(FileNotFoundError, match=message) as error:
            generate_params(ethanol, tmp_path)

    assert "molfile_to_params.py" in str(error.value)
    assert not any(tmp_path.glob(".*_tmp"))
