"""Public example inputs and packed structure branch contracts."""

import json
import runpy
import shutil
from pathlib import Path

from chaperone import PipelineDataHandler
from chaperone.data.schema import make_row
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


def test_three_packed_branches_keep_lineage_and_distinct_structures_after_repack(
    tmp_path,
):
    example = runpy.run_path(str(ROOT / "examples/benchmarks/multi-mpnn/pipeline.py"))
    blocks = example["sequence_blocks"]()
    path = next(example["FILES"].glob("*.cif"))
    source = PipelineDataHandler(
        [
            make_row(
                name="input",
                path=path,
                design_id="shared_backbone",
                metadata={"motif": ["A1"]},
            )
        ]
    )
    outputs = []
    output_paths = []
    for block in blocks:
        assert block.output_mode == "structures"
        assert block.args.number_of_packs_per_design == 1
        assert block.args.file_ending == f"_{block.model_type}"
        # Stand-ins for independently produced packed files: this tests the
        # example's branch merge/repack contract, not MPNN inference itself.
        packed = tmp_path / block.name / "shared_backbone__0_1.cif"
        packed.parent.mkdir()
        shutil.copyfile(path, packed)
        output_paths.append(str(packed))
        child = PipelineDataHandler(
            [
                make_row(
                    name=block.name,
                    path=packed,
                    path_handle=path,
                    design_id=block.name,
                    parent_design_id="shared_backbone",
                    metadata={"motif": ["A1"]},
                    metrics={block.name: {"score": 1.0}},
                )
            ]
        )
        outputs.append(PipelineDataHandler.merge(source, child))
    combined = outputs[0] + outputs[1] + outputs[2]
    repacked = PipelineDataHandler(combined.unpack())
    rows = repacked.unpack().to_dict("records")
    assert len(rows) == 3
    assert [row["parent_design_id"] for row in rows] == ["shared_backbone"] * 3
    assert [row["design_id"] for row in rows] == [block.name for block in blocks]
    assert all(item.path == str(path) for item in repacked.get_data(which="first"))
    final = list(repacked.get_data(which="last"))
    assert [item.path for item in final] == output_paths
    assert all(item.metadata["motif"] == ["A1"] for item in final)
    for block, item in zip(blocks, final):
        assert item.metrics[block.name]["score"] == 1.0


def test_multi_mpnn_refolding_uses_separate_input_directories_per_model():
    from collections import Counter

    from rdkit import Chem

    from blocks.networks.prediction.helpers import build_remap_reference

    example = runpy.run_path(str(ROOT / "examples/benchmarks/multi-mpnn/pipeline.py"))
    path = next(example["FILES"].glob("*.cif"))
    reference = build_remap_reference(str(path), target_name="HCY")
    assert reference["n_ref_atoms"] == 26
    assert Chem.MolToInchiKey(
        Chem.Mol(reference["ref_mol_bytes"])
    ) == "JYGXADMDTFJGBT-VWUMJDOOSA-N"
    branches = [example["refolding_blocks"](model) for model in example["MODELS"]]
    assert len({branch[0].name for branch in branches}) == 3
    assert len({branch[1].name for branch in branches}) == 3
    for af3, rmsd, gate in branches:
        assert af3.input_mode == "json"
        ligand = Chem.MolFromSmiles(af3.specified_smiles["HCY"])
        atoms = parse_to_atom_array(path)
        fixture_elements = atoms.element[
            (atoms.res_name == "HCY") & (atoms.element != "H")
        ]
        assert Counter(atom.GetSymbol() for atom in ligand.GetAtoms()) == Counter(
            fixture_elements
        )
        assert set(gate.filters) == {
            f"{af3.name}.ranking_score",
            f"{rmsd.name}.bkbn_align.bkbn_rmsd",
        }


def test_fgfr2_hotspot_origin_has_atoms_in_the_retained_target():
    """A cropped target with no retained hotspot cannot define the origin."""
    import re

    files = ROOT / "examples/fgfr2-binder/files"
    for config in ("rfd3_config.json", "rfd3_config_override.json"):
        spec = json.loads((files / config).read_text())["fgfr2_binder"]
        atoms = parse_to_atom_array(files / spec["input"])
        retained = set()
        for component in spec["contig"].split(","):
            match = re.fullmatch(r"([A-Za-z])(\d+)(?:-(\d+))?", component)
            if match:
                chain, first, last = match.groups()
                retained.update(f"{chain}{i}" for i in range(int(first), int(last or first) + 1))
        hotspots = set(spec["select_hotspots"].split(","))
        present = {f"{chain}{residue}" for chain, residue in zip(atoms.chain_id, atoms.res_id)}
        assert hotspots <= present
        assert hotspots & retained, "Hotspot origin requires a hotspot in the retained contig"
