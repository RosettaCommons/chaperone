"""Chai output integration: artifact steps stay distinct from design IDs."""

import numpy as np
import pytest

from blocks.networks.prediction.chai1 import Chai1Block


@pytest.mark.parametrize("best_only", [True, False])
def test_compile_keeps_block_name_for_multiple_designs_and_samples(tmp_path, best_only):
    for design in ("design_a", "design_b"):
        folder = tmp_path / "outputs" / design
        folder.mkdir(parents=True)
        for index, score in enumerate((0.2, 0.8)):
            (folder / f"pred.model_idx_{index}.cif").write_text("data_test\n")
            np.savez(folder / f"scores.model_idx_{index}.npz", aggregate_score=[score])
    block = Chai1Block(name="chai_check", command_template=["true"], read_most_confident_model=best_only)
    result = block.compile(tmp_path)
    outputs = list(result.get_data(which="last", load=False))
    assert len(outputs) == (2 if best_only else 4)
    assert {output.name for output in outputs} == {"chai_check"}
    assert len({output.design_id for output in outputs}) == len(outputs)
    frame = result.unpack()
    assert "chai_check.path" in frame.columns
    assert sorted(frame["chai_check.ranking_score"]) == ([0.8, 0.8] if best_only else [0.2, 0.2, 0.8, 0.8])
