"""Meaningful data and configuration contracts for the remaining public examples."""

import json
from pathlib import Path
import runpy

import numpy as np
import pytest
from biotite.structure import Atom, array

from blocks.util_blocks.select_chains import SelectChainsBlock
from chaperone import PipelineDataHandler
from chaperone.data import make_row
from chaperone.io import write_pdb

ROOT = Path(__file__).resolve().parents[1]


def protein(chains=("H", "L")):
    atoms = array(
        [
            Atom(
                [i * 4.0, j * 15.0, 0.0],
                chain_id=c,
                res_id=i + 1,
                res_name="GLY",
                atom_name="CA",
                element="C",
            )
            for j, c in enumerate(chains)
            for i in range(8)
        ]
    )
    return atoms


def test_chain_selection_really_removes_substrate_and_keeps_lineage(tmp_path):
    atoms = protein(("A", "B"))
    path = tmp_path / "complex.pdb"
    write_pdb(atoms, path)
    h = PipelineDataHandler(
        [
            make_row(
                "mpnn",
                path,
                design_id="parent",
                metadata={
                    "motif": ["A2", "B2"],
                    "diffused_index_map": json.dumps({"A86": "A2", "B7": "B2"}),
                    "motif_atom_contigs": {"A2": ["CA"], "B2": ["CA"]},
                },
            )
        ]
    )
    result = SelectChainsBlock(chains=["A"])(h, tmp_path / "apo")
    loaded = next(result.get_data(load=True))
    assert set(loaded.atom_array.chain_id) == {"A"}
    assert loaded.metadata["motif"] == ["A2"]
    assert json.loads(loaded.metadata["diffused_index_map"]) == {"A86": "A2"}
    assert result.names == ["mpnn", "select_chains"]
    assert result.unpack()["parent_design_id"][0] == "parent"
    with pytest.raises(ValueError, match="missing chains"):
        SelectChainsBlock(chains=["Z"])(h, tmp_path / "absent")


def test_dna_conditioning_matches_real_duplex_and_rejects_sequence_edits_without_new_coordinates(
    tmp_path,
):
    m = runpy.run_path(str(ROOT / "examples/dna-binder/pipeline.py"))
    target = m["FILES_DIR"] / "dna_target.pdb"
    path = m["build_rfd3_config"](m["DNA_SEQUENCE"], target, tmp_path)
    spec = json.loads(path.read_text())["dna_binder"]
    assert spec["input"] == str(target.resolve())
    assert spec["select_hbond_donor"]["A5"] == "N6"
    assert spec["select_hbond_acceptor"]["A2"] == "O6,N7"
    assert spec["length"] == "144-174"
    with pytest.raises(ValueError, match="sequence/numbering"):
        m["build_rfd3_config"]("AAAAAAAAAAAA", target, tmp_path)
    first, second = m["rounds"](path, "/test/dna-specific.ckpt")
    assert first[0].ckpt_path == "/test/dna-specific.ckpt"
    assert len(second) == 4


def test_regular_rfd3_antibody_scaffold_and_loop_contract():
    from chaperone.io import parse_to_atom_array
    from biotite.structure import filter_amino_acids

    m = runpy.run_path(str(ROOT / "examples/antibody-design/pipeline.py"))
    blocks = m["pipeline"]()
    spec = json.loads(Path(blocks[0].inputs).read_text())["cdr_h3"]
    assert spec["contig"] == "A1-98,8,A107-117,/0,B1-113,/0,C1-118"
    atoms = parse_to_atom_array(m["FILES_DIR"] / spec["input"])
    assert filter_amino_acids(atoms).all()
    assert set(atoms.chain_id) == {"A", "B", "C"}
    for chain, length in [("A", 117), ("B", 113), ("C", 118)]:
        ca = atoms[(atoms.chain_id == chain) & (atoms.atom_name == "CA")]
        np.testing.assert_array_equal(ca.res_id, np.arange(1, length + 1))
    mapping = json.loads((m["FILES_DIR"] / "residue_mapping.json").read_text())
    loop = [
        x["example_residue"]
        for x in mapping
        if x["chain"] == "A" and 95 <= x["source_residue"] <= 102
    ]
    assert loop == list(range(99, 107))
    assert blocks[1].output_mode == "structures"
    assert blocks[1].args.number_of_packs_per_design == 1
    assert blocks[1].keep_motif_sequences_fixed
    assert blocks[1].args.chains_to_design == "A"
