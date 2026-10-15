"""The retained rotamer relaxation block is discoverable and preserves lineage."""

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
from biotite.structure import AtomArray

from blocks import Blocks
from blocks.util_blocks import rotamer_relax
from chaperone import PipelineDataHandler
from chaperone.data.schema import make_row


@pytest.mark.parametrize("pose_index", [1, 0])
def test_registered_relaxation_block_executes_or_rejects_missing_motif(
    tmp_path, monkeypatch, pose_index
):
    block = Blocks["rotamer_relax"]()
    assert isinstance(block, rotamer_relax.PyrosettaRotamerRelaxBlock)
    handler = PipelineDataHandler(
        [make_row("input", "/tmp/input.cif", metadata={"motif": ["A10"]})]
    )
    loaded = next(handler.get_data())
    loaded.atom_array = AtomArray(0)
    monkeypatch.setattr(handler, "get_data", lambda **kwargs: iter([loaded]))
    pose = SimpleNamespace(
        pdb_info=lambda: SimpleNamespace(pdb2pose=lambda chain, residue: pose_index),
        dump_cif=lambda filename: Path(filename).write_text("data_relaxed\n#\n"),
    )
    monkeypatch.setitem(
        sys.modules,
        "blocks.utils.pyrosetta",
        SimpleNamespace(
            init_pyrosetta=lambda: None,
            atom_array_to_pose=lambda atoms: pose,
        ),
    )
    minimized = []
    monkeypatch.setattr(
        rotamer_relax,
        "min_sc_fix_tip_fix_bb",
        lambda pose, indices, **kwargs: minimized.extend(indices),
    )
    if pose_index == 0:
        with pytest.raises(ValueError, match="Motif residues.*not found"):
            block(handler, tmp_path / "relaxed")
        assert not minimized
    else:
        result = block(handler, tmp_path / "relaxed")
        output = next(result.get_data())
        assert minimized == [1]
        assert output.name == "rotamer_relax"
        assert Path(output.path).read_text() == "data_relaxed\n#\n"
        assert output.metadata["motif"] == ["A10"]
        assert result._df["structures"].list.len().to_list() == [2]
