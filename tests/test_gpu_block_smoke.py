"""CPU checks for the assertions used by the opt-in real GPU smoke script."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from chaperone import PipelineDataHandler

_spec = importlib.util.spec_from_file_location("gpu_block_smoke", Path(__file__).with_name("gpu_block_smoke.py"))
smoke = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(smoke)


def test_smoke_crops_real_48_residue_chain_and_checks_prediction(tmp_path):
    from biotite.structure import get_residue_count
    from chaperone.io import parse_to_atom_array
    source = smoke.REPO_ROOT / "examples/benchmarks/ppi/files/4zxb_cropped.pdb"
    staged = tmp_path / "input.pdb"
    smoke.stage_reference(source, staged)
    atoms = parse_to_atom_array(str(staged))
    assert get_residue_count(atoms) == 48
    assert set(atoms.chain_id) == {"A"}
    handler = PipelineDataHandler.from_paths([str(staged)], name="boltz2")
    assert smoke.validate_artifacts(handler, "boltz2") == [str(staged)]
    with pytest.raises(AssertionError, match="Last artifact"):
        smoke.validate_artifacts(handler, "chai1")
    with pytest.raises(ValueError, match="between 1 and 48"):
        smoke.stage_reference(source, staged, residues=49)


def test_smoke_checks_embedding_tensors_and_rejects_nonfinite_values(tmp_path):
    from safetensors.numpy import save_file
    path = tmp_path / "embedding.safetensors"
    arrays = {
        "logits": np.zeros((48, 32), dtype=np.float32),
        "token_embeddings": np.zeros((48, 4), dtype=np.float32),
        "mean_token_embeddings": np.zeros(4, dtype=np.float32),
    }
    save_file(arrays, str(path))
    handler = PipelineDataHandler.from_paths([str(path)], name="e1")
    assert smoke.validate_artifacts(handler, "e1") == [str(path)]
    arrays["mean_token_embeddings"][0] = np.nan
    save_file(arrays, str(path))
    with pytest.raises(AssertionError, match="nonfinite"):
        smoke.validate_artifacts(handler, "e1")


def test_smoke_rejects_finite_but_exploded_coordinates(tmp_path):
    from biotite.structure.io.pdb import PDBFile
    from chaperone.io import parse_to_atom_array
    source = smoke.REPO_ROOT / "examples/benchmarks/ppi/files/4zxb_cropped.pdb"
    staged = tmp_path / "exploded.pdb"
    smoke.stage_reference(source, staged)
    atoms = parse_to_atom_array(str(staged))
    atoms.coord[0, 0] += 2000
    pdb = PDBFile()
    pdb.set_structure(atoms)
    pdb.write(str(staged))
    handler = PipelineDataHandler.from_paths([str(staged)], name="boltz2")
    with pytest.raises(AssertionError, match="Coordinate extent"):
        smoke.validate_artifacts(handler, "boltz2")
