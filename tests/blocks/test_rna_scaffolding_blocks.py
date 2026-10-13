"""Tests for the RNA motif-scaffolding blocks that score the design itself.

``rna_connectivity`` and ``motif_confidence`` both answer questions the shipped
metrics get wrong for nucleic acids: whether the backbone is actually bonded,
and whether the *motif* - not the whole structure - is confidently placed. Both
are checked against constructed cases with a known answer.
"""

import json

import numpy as np
import pytest
from biotite.structure import AtomArray

from blocks.metrics.motif_confidence import MotifConfidenceBlock
from blocks.metrics.rna_connectivity import RNAConnectivityBlock
from chaperone import PipelineDataHandler
from chaperone.data import make_row
from chaperone.data.handler import _sequence_to_cif
from chaperone.io import parse_to_atom_array

BOND = 1.6


def _write_rna(path, n_residues: int, break_after: int | None = None, b_factors=None):
    """An n-residue RNA chain, optionally with one junction pulled apart.

    Each residue carries P, C3' and O3'. Residue i's O3' sits one bond length
    before residue i+1's P, so the chain is continuous except at the junction
    after ``break_after`` (0-based), where the next residue is displaced 10 A.
    """
    from atomworks.io.utils.io_utils import to_cif_file
    from chaperone.io import ensure_bonds

    starts = [
        10.0 * i + (10.0 if break_after is not None and i > break_after else 0.0)
        for i in range(n_residues)
    ]
    atoms = AtomArray(3 * n_residues)
    atoms.set_annotation("b_factor", np.zeros(len(atoms)))
    for i in range(n_residues):
        placements = (
            ("P", starts[i], "P"),
            ("C3'", starts[i] + 5.0, "C"),
            ("O3'", starts[i] + 10.0 - BOND, "O"),
        )
        for j, (name, x, element) in enumerate(placements):
            k = 3 * i + j
            atoms.coord[k] = [x, 0.0, 0.0]
            atoms.chain_id[k] = "A"
            atoms.res_id[k] = i + 1
            atoms.res_name[k] = "A"
            atoms.atom_name[k] = name
            atoms.element[k] = element
            if b_factors is not None:
                atoms.b_factor[k] = b_factors[i]

    to_cif_file(ensure_bonds(atoms), str(path))
    return path


def _row(path, metadata=None):
    return PipelineDataHandler(
        [make_row(name="design", path=str(path), metadata=metadata or {})]
    )


def _metrics(handler, block_name: str, row: int = 0) -> dict:
    df = handler.unpack(all=True)
    prefix = f"{block_name}."
    return {c[len(prefix) :]: df[c][row] for c in df.columns if c.startswith(prefix)}


# ── rna_connectivity ────────────────────────────────────────────────────


def test_intact_chain_has_no_breaks(tmp_path):
    handler = _row(_write_rna(tmp_path / "intact.cif", 5))
    metrics = _metrics(RNAConnectivityBlock().execute(handler), "rna_connectivity")
    assert metrics["n_breaks"] == 0
    assert metrics["n_junctions"] == 4
    assert metrics["n_residues"] == 5
    assert metrics["max_gap"] == pytest.approx(BOND, abs=1e-3)


def test_one_pulled_junction_is_one_break(tmp_path):
    """The metric must count the broken bond, not the junction: a designed
    60-mer has 59 junctions and that number says nothing about connectivity."""
    handler = _row(_write_rna(tmp_path / "broken.cif", 5, break_after=2))
    metrics = _metrics(RNAConnectivityBlock().execute(handler), "rna_connectivity")
    assert metrics["n_breaks"] == 1
    assert metrics["break_positions"] == "A3->A4"
    assert metrics["max_gap"] > 10.0
    assert metrics["n_breaks_at_motif"] is None  # no index map on this row


def test_breaks_at_motif_junctions_are_counted_separately(tmp_path):
    """On unindexed scaffolding every break lands at a motif junction, which is
    a different failure from a scaffold the model could not close."""
    handler = _row(
        _write_rna(tmp_path / "broken.cif", 5, break_after=2),
        metadata={"diffused_index_map": json.dumps({"X20": "A4"})},
    )
    metrics = _metrics(RNAConnectivityBlock().execute(handler), "rna_connectivity")
    assert metrics["n_breaks"] == 1
    assert metrics["n_breaks_at_motif"] == 1


# ── motif_confidence ────────────────────────────────────────────────────


def test_motif_and_scaffold_confidence_are_split(tmp_path):
    """A design whose motif is confident and whose scaffold is not must not be
    summarized by one mean - that is exactly the case the block exists for."""
    model = tmp_path / "design_model.cif"
    _write_rna(model, 4, b_factors=[90.0, 90.0, 40.0, 40.0])
    pae = np.full((4, 4), 20.0)
    pae[:2, :2] = 2.0
    (tmp_path / "design_confidences.json").write_text(
        json.dumps(
            {
                "pae": pae.tolist(),
                "token_chain_ids": ["A"] * 4,
                "token_res_ids": [1, 2, 3, 4],
            }
        )
    )

    handler = _row(
        model, metadata={"diffused_index_map": json.dumps({"X1": "A1", "X2": "A2"})}
    )
    block = MotifConfidenceBlock(which=[0, 0])
    metrics = _metrics(block.execute(handler), "motif_confidence")

    assert metrics["status"] == "ok"
    assert metrics["n_motif_residues"] == 2
    assert metrics["plddt_motif"] == pytest.approx(90.0)
    assert metrics["plddt_scaffold"] == pytest.approx(40.0)
    assert metrics["pae_motif_internal_max"] == pytest.approx(2.0)
    assert metrics["pae_motif_scaffold_mean"] == pytest.approx(20.0)


def test_missing_confidences_file_is_partial_not_silent(tmp_path):
    """pLDDT lives in the B-factor column and must still be reported; the
    absent PAE has to be named rather than come back as a bare None."""
    model = tmp_path / "nopae_model.cif"
    _write_rna(model, 4, b_factors=[90.0, 90.0, 40.0, 40.0])
    handler = _row(model, metadata={"diffused_index_map": json.dumps({"X1": "A1"})})
    metrics = _metrics(
        MotifConfidenceBlock(which=[0, 0]).execute(handler), "motif_confidence"
    )
    assert metrics["status"] == "partial"
    assert metrics["plddt_motif"] == pytest.approx(90.0)
    assert metrics["pae_motif_scaffold_mean"] is None


# ── RNA sequences as pipeline inputs ────────────────────────────────────


def test_rna_sequence_cif_is_typed_as_rna(tmp_path):
    """A sequence-only RNA submission has to enter the pipeline as an RNA
    chain: oracles build their entity from the chain type, so a CA-atom
    amino-acid spoof would be folded as a peptide."""
    from atomworks.enums import ChainType

    path = tmp_path / "rna.cif"
    _sequence_to_cif("GGACUUCGGUCC", path, molecule_type="rna")
    atoms = parse_to_atom_array(str(path))

    assert set(atoms.res_name.tolist()) <= {"A", "C", "G", "U"}
    assert set(atoms.atom_name.tolist()) == {"C3'"}
    assert atoms.chain_type[0] == ChainType.RNA


def test_dna_and_protein_sequences_stay_distinguishable(tmp_path):
    from atomworks.enums import ChainType

    dna = tmp_path / "dna.cif"
    _sequence_to_cif("GGACTTCGGTCC", dna, molecule_type="dna")
    assert parse_to_atom_array(str(dna)).chain_type[0] == ChainType.DNA

    protein = tmp_path / "protein.cif"
    _sequence_to_cif("MKTAYIAKQR", protein)
    assert parse_to_atom_array(str(protein)).chain_type[0] == ChainType.POLYPEPTIDE_L
