"""Verify the bundled RNA motif and its atom-addressed catalytic measurements."""

from pathlib import Path
import runpy

from blocks.networks.foundry.rfd3na_input import build_motif_spec
from chaperone.io import parse_to_atom_array

ROOT = Path(__file__).resolve().parents[1]


def test_hammerhead_motif_is_single_molecule_and_all_metric_atoms_exist():
    example = runpy.run_path(str(ROOT / "examples/ribozyme-hammerhead/pipeline.py"))
    spec = build_motif_spec(
        example["MOTIF"],
        example["UNINDEX"],
        nucleic="RNA",
        length=example["LENGTH"],
        ligand=example["LIGAND"],
    )
    assert spec["contig"] == "60-60R"
    assert spec["unindex"] == "A20-22,A36,B6-7"
    assert spec["ligand"] == "MN"
    atoms = parse_to_atom_array(str(example["MOTIF"]))
    for addresses in example["CATALYTIC"].values():
        for address in addresses:
            residue, atom = address.split(".")
            if residue.startswith("L:"):
                mask = atoms.res_name == residue[2:]
            else:
                mask = (atoms.chain_id == residue[0]) & (
                    atoms.res_id == int(residue[1:])
                )
            assert ((atoms.atom_name == atom) & mask).sum() == 1, address
