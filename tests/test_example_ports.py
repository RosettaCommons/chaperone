"""Public example inputs and native sequence branch contracts."""

import json
from pathlib import Path
import runpy

from chaperone import PipelineDataHandler
from chaperone.data.schema import make_sequence_row
from chaperone.io import parse_to_atom_array

ROOT = Path(__file__).resolve().parents[1]


def test_peptide_is_sequence_conditioned_with_unfixed_coordinates(tmp_path):
    example = runpy.run_path(str(ROOT / "examples/peptide-binder/pipeline.py"))
    config = example["prepare_inputs"](tmp_path)
    spec = json.loads(config.read_text())["peptide_binder"]
    atoms = parse_to_atom_array(spec["input"])
    ca = atoms[atoms.atom_name == "CA"]
    assert ca.chain_id.tolist() == ["B"] * len(example["SEQUENCE"])
    assert ca.res_id.tolist() == list(range(1, len(example["SEQUENCE"]) + 1))
    assert ca.res_name.tolist() == [
        "THR",
        "VAL",
        "GLY",
        "SER",
        "ASP",
        "GLY",
        "PRO",
        "PRO",
        "SER",
        "PHE",
        "ARG",
        "SER",
        "LEU",
        "SER",
        "GLY",
    ]
    assert spec["select_fixed_atoms"] is False
    assert spec["redesign_motif_sidechains"] is False
    assert Path(spec["input"]).parent == tmp_path


def test_three_sequence_branches_keep_context_identity_and_payload_after_repack():
    example = runpy.run_path(str(ROOT / "examples/benchmarks/multi-mpnn/pipeline.py"))
    blocks = example["sequence_blocks"]()
    path = next(example["FILES"].glob("*.cif"))
    outputs = [
        PipelineDataHandler(
            [
                make_sequence_row(
                    design_id=block.name,
                    parent_design_id="shared_backbone",
                    sequences={"A": seq},
                    context_path=path,
                    context_name="input",
                    context_metadata={"motif": ["A1"]},
                    metrics={block.name: {"score": 1.0}},
                )
            ]
        )
        for block, seq in zip(blocks, ["GAS", "GAL", "GAV"])
    ]
    combined = outputs[0] + outputs[1] + outputs[2]
    repacked = PipelineDataHandler(combined.unpack())
    rows = repacked.unpack().to_dict("records")
    assert [row["sequence_A"] for row in rows] == ["GAS", "GAL", "GAV"]
    assert repacked.sequence_dataframe().equals(combined.sequence_dataframe())
    assert [row["parent_design_id"] for row in rows] == ["shared_backbone"] * 3
    assert [row["design_id"] for row in rows] == [block.name for block in blocks]
    assert all(
        item.path == str(path) and item.metadata["motif"] == ["A1"]
        for item in repacked.get_data(which="first")
    )
