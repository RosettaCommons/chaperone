"""Tests for the RNA motif-scaffolding benchmark metric blocks.

Each block is checked against a case with a known answer rather than a golden
number: a hairpin whose stem is enforced must return p ~ 1, a structure scored
against itself must return TM = 1, and a sequence aligned to itself must return
identity 1.
"""

import importlib.util
import json
from types import SimpleNamespace

import numpy as np
import pytest

from blocks.metrics import dssr_snap_metrics
from blocks.metrics.dssr_snap_metrics import DSSRMotifBasePairingBlock, run_dssr_pairs
from blocks.metrics.rna_thermo import (
    RNAThermoBlock,
    motif_boltzmann_probability,
    rna_sequence_and_index,
)
from blocks.metrics.sequence_identity_gate import (
    SequenceIdentityGateBlock,
    nw_identity,
    read_fasta,
)
from blocks.metrics.tm_score import TMScoreBlock, polymer_anchors
from chaperone import PipelineDataHandler
from chaperone.data import make_row
from chaperone.io import parse_to_atom_array

RNA_FIXTURE = "tests/test_data/rna_metrics/synthetic_six_nt.pdb"


def test_dssr_uses_relative_output_arguments_in_deep_workdir(tmp_path, monkeypatch):
    """Old DSSR builds segfault when output arguments contain long paths."""
    binary = tmp_path / "fake-x3dna-dssr"
    log = tmp_path / "invocations.jsonl"
    binary.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

with open(os.environ["FAKE_DSSR_LOG"], "a") as handle:
    handle.write(json.dumps(sys.argv[1:]) + "\\n")
if "--cleanup" not in sys.argv:
    output = next(arg.split("=", 1)[1] for arg in sys.argv if arg.startswith("--o="))
    Path(output).write_text(json.dumps({"pairs": []}))
"""
    )
    binary.chmod(0o755)
    monkeypatch.setenv("FAKE_DSSR_LOG", str(log))

    input_path = tmp_path / "input.cif"
    input_path.write_text("data_test\n")
    work_dir = tmp_path / ("deep-" + "x" * 120) / ("nested-" + "y" * 120)

    assert (
        run_dssr_pairs(
            input_path,
            work_dir,
            x3dna_dssr_bin=binary,
            base="row 1",
        )
        == []
    )

    invocations = [json.loads(line) for line in log.read_text().splitlines()]
    assert "--o=row_1-dssr.json" in invocations[0]
    assert "--prefix=row_1" in invocations[0]
    assert invocations[1] == ["--cleanup", "--prefix=row_1"]
    assert all(
        str(work_dir) not in argument for call in invocations for argument in call
    )


def test_dssr_motif_pairing_can_report_row_error_without_aborting(
    tmp_path, monkeypatch
):
    def fail_dssr(*args, **kwargs):
        raise RuntimeError("native tool crashed")

    monkeypatch.setattr(dssr_snap_metrics, "run_dssr_pairs", fail_dssr)
    source = SimpleNamespace(
        path=RNA_FIXTURE,
        metadata={"diffused_index_map": '{"A1": "A1"}'},
    )
    prediction = SimpleNamespace(path=RNA_FIXTURE)
    block = DSSRMotifBasePairingBlock(
        reference=RNA_FIXTURE,
        residues=["A1"],
        continue_on_error=True,
        x3dna_dssr_bin="/unused/in-this-unit-test",
    )

    value = block.compute_metrics(
        (source, prediction),
        context={"work_root": tmp_path, "native_pairs": []},
    )

    assert value == {
        "status": "error",
        "error_type": "RuntimeError",
        "error": "native tool crashed",
    }


def test_dssr_motif_pairing_remains_strict_by_default(tmp_path, monkeypatch):
    def fail_dssr(*args, **kwargs):
        raise RuntimeError("native tool crashed")

    monkeypatch.setattr(dssr_snap_metrics, "run_dssr_pairs", fail_dssr)
    source = SimpleNamespace(
        path=RNA_FIXTURE,
        metadata={"diffused_index_map": '{"A1": "A1"}'},
    )
    prediction = SimpleNamespace(path=RNA_FIXTURE)
    block = DSSRMotifBasePairingBlock(
        reference=RNA_FIXTURE,
        residues=["A1"],
        x3dna_dssr_bin="/unused/in-this-unit-test",
    )

    with pytest.raises(RuntimeError, match="native tool crashed"):
        block.compute_metrics(
            (source, prediction),
            context={"work_root": tmp_path, "native_pairs": []},
        )


# ViennaRNA and tmtools ship in the optional ``rna`` extra, so a default
# install - CI's included - does not have them. Skip rather than fail: the
# tests that need neither still run, exactly as the US-align tests do.
requires_vienna = pytest.mark.skipif(
    importlib.util.find_spec("RNA") is None,
    reason="ViennaRNA not installed (pip install 'chaperone[rna]')",
)
requires_tmtools = pytest.mark.skipif(
    importlib.util.find_spec("tmtools") is None,
    reason="tmtools not installed (pip install 'chaperone[rna]')",
)


def _metrics(handler, block_name: str, row: int = 0) -> dict:
    """Metric dict for one row, from the flattened '<block>.<key>' columns."""
    df = handler.unpack(all=True)
    prefix = f"{block_name}."
    return {c[len(prefix) :]: df[c][row] for c in df.columns if c.startswith(prefix)}


# ── T2: ViennaRNA ──────────────────────────────────────────────────────


@requires_vienna
def test_enforced_stem_is_certain():
    """A hairpin whose own stem is enforced must have p ~ 1: the constraint
    selects the ensemble the sequence already prefers."""
    seq = "GGGGAAAACCCC"
    pairs = [(0, 11), (1, 10), (2, 9), (3, 8)]
    p, g0, g1, dropped = motif_boltzmann_probability(seq, pairs)
    # Not 1.0: even a perfect stem's ensemble contains frayed states.
    assert p > 0.8
    assert dropped == []
    assert g1 >= g0 - 1e-6  # constraining can only raise the ensemble energy


@requires_vienna
def test_impossible_pairing_is_reported_not_scored():
    """Poly-A cannot pair at all, so the pairs are dropped as noncanonical
    rather than assigned a number the model cannot justify."""
    seq = "AAAAAAAAAAAAAAAAAAAA"
    p, _, _, dropped = motif_boltzmann_probability(seq, [(0, 19), (1, 18), (2, 17)])
    assert p is None
    assert {d[2] for d in dropped} == {"noncanonical"}


@requires_vienna
def test_a_real_stem_competing_with_the_fold_is_penalised():
    """A canonical pair Vienna CAN model, but which fights the preferred fold,
    must come back with low probability rather than being dropped."""
    seq = "AAAAAAAAAAAAAAAGCGCU"  # A1 pairs U20 canonically, but nothing else does
    p, _, _, dropped = motif_boltzmann_probability(seq, [(0, 19)])
    assert dropped == []
    assert p < 0.5


@requires_vienna
def test_probability_never_exceeds_one():
    seq = "GCGCGCAAAAGCGCGC"
    p, _, _, _ = motif_boltzmann_probability(seq, [(0, 15)])
    assert 0.0 <= p <= 1.0


def test_sequence_extraction_skips_non_rna():
    """The Mg in the fixture must not enter the sequence, or every index
    downstream of it would be shifted."""
    seq, index = rna_sequence_and_index(parse_to_atom_array(RNA_FIXTURE))
    assert seq == "GACUAC"
    assert len(index) == len(seq)
    assert ("Z", 99) not in index  # the Mg


def test_thermo_block_reports_na_without_pairs(tmp_path):
    handler = PipelineDataHandler(
        [make_row(name="design", path=RNA_FIXTURE, metadata={}, metrics={})]
    )
    out = RNAThermoBlock(required_pairs=[])(handler, sandbox_dir=tmp_path)
    assert _metrics(out, "rna_thermo")["status"] == "na"


@requires_vienna
def test_thermo_block_resolves_pairs_through_index_map(tmp_path):
    """Pairs addressed by reference id must resolve via diffused_index_map."""
    index_map = {"A1": "A1", "A3": "A3"}
    handler = PipelineDataHandler(
        [
            make_row(
                name="design",
                path=RNA_FIXTURE,
                metadata={"diffused_index_map": json.dumps(index_map)},
                metrics={},
            )
        ]
    )
    block = RNAThermoBlock(required_pairs=[["A1", "A3"]])
    value = _metrics(block(handler, sandbox_dir=tmp_path), "rna_thermo")
    assert value["n_pairs_scored"] == 1
    assert value["n_pairs_unresolved"] == 0
    # A1 and A3 are sequence indices 0 and 2 here, inside the minimum hairpin
    # loop, so Vienna cannot model the pair and the block says so.
    assert value["status"] == "na"
    assert value["n_pairs_dropped"] == 1
    assert value["dropped_reasons"] == "separation"


@requires_vienna
def test_pseudoknot_pairs_are_excluded_and_flagged(tmp_path):
    handler = PipelineDataHandler(
        [make_row(name="design", path=RNA_FIXTURE, metadata={}, metrics={})]
    )
    block = RNAThermoBlock(required_pairs=[["0", "5"]], pseudoknot_pairs=[["1", "4"]])
    value = _metrics(block(handler, sandbox_dir=tmp_path), "rna_thermo")
    assert value["status"] == "partial"
    assert value["pk_pairs_excluded"] == 1
    assert value["n_pairs_scored"] == 1


def test_self_pair_is_rejected(tmp_path):
    handler = PipelineDataHandler(
        [make_row(name="design", path=RNA_FIXTURE, metadata={}, metrics={})]
    )
    with pytest.raises(ValueError, match="pair a residue with itself"):
        RNAThermoBlock(required_pairs=[["2", "2"]])(handler, sandbox_dir=tmp_path)


# ── T5: novelty gate ───────────────────────────────────────────────────


def test_identity_of_a_sequence_with_itself_is_one():
    assert nw_identity("GGGGAAAACCCC", "GGGGAAAACCCC") == pytest.approx(1.0)


def test_gaps_count_in_the_denominator():
    """Global identity must penalise length mismatch, unlike local identity."""
    short, long = "GGGGAAAACCCC", "GGGGAAAACCCC" + "A" * 24
    assert nw_identity(short, long) < 0.55


def test_gate_passes_a_novel_sequence(tmp_path):
    fasta = tmp_path / "refs.fasta"
    fasta.write_text(">native\nAAAAAAAAAAAAAAAAAAAAAAAA\n")
    handler = PipelineDataHandler(
        [make_row(name="design", path=RNA_FIXTURE, metadata={}, metrics={})]
    )
    block = SequenceIdentityGateBlock(references=str(fasta), threshold=0.40)
    value = _metrics(block(handler, sandbox_dir=tmp_path), "sequence_identity_gate")
    assert bool(value["passed"]) is True
    assert value["nearest_natural_id"] == "native"


def test_gate_fails_a_copy_of_the_native(tmp_path):
    """The design's own sequence as the reference must fail at any threshold."""
    fasta = tmp_path / "refs.fasta"
    fasta.write_text(">native\nGACUAC\n")
    handler = PipelineDataHandler(
        [make_row(name="design", path=RNA_FIXTURE, metadata={}, metrics={})]
    )
    block = SequenceIdentityGateBlock(references=str(fasta), threshold=0.40)
    value = _metrics(block(handler, sandbox_dir=tmp_path), "sequence_identity_gate")
    assert value["max_identity"] == pytest.approx(1.0)
    assert bool(value["passed"]) is False


def test_single_design_has_zero_intra_submission_identity(tmp_path):
    """A singleton has zero peers and a finite, neutral maximum identity."""
    fasta = tmp_path / "refs.fasta"
    fasta.write_text(">native\nAAAAAAAAAAAAAAAAAAAAAAAA\n")
    handler = PipelineDataHandler(
        [make_row(name="design", path=RNA_FIXTURE, metadata={}, metrics={})]
    )
    block = SequenceIdentityGateBlock(
        references=str(fasta), compute_intra_identity=True
    )

    value = _metrics(block(handler, sandbox_dir=tmp_path), "sequence_identity_gate")

    assert value["n_intra_submission_comparisons"] == 0
    assert value["max_intra_submission_identity"] == pytest.approx(0.0)
    assert np.isfinite(value["max_intra_submission_identity"])


@pytest.mark.parametrize(
    ("sequences", "expected_comparisons"),
    [
        (["AAAAAA", "AAAAAC"], 1),
        (["AAAAAA", "AAAAAC", "CCCCCC"], 2),
    ],
)
def test_intra_submission_identity_excludes_self(
    tmp_path, sequences, expected_comparisons
):
    """Every row is compared with each peer, but never with itself."""

    class SyntheticSequenceIdentityGateBlock(SequenceIdentityGateBlock):
        def _scaffold(self, loaded):
            return loaded.metadata["test_sequence"], 0

    fasta = tmp_path / "refs.fasta"
    fasta.write_text(">native\nGGGGGG\n")
    handler = PipelineDataHandler(
        [
            make_row(
                name=f"design-{i}",
                path=RNA_FIXTURE,
                metadata={"test_sequence": sequence},
                metrics={},
            )
            for i, sequence in enumerate(sequences)
        ]
    )
    block = SyntheticSequenceIdentityGateBlock(
        references=str(fasta), compute_intra_identity=True
    )

    result = block(handler, sandbox_dir=tmp_path)
    for i, sequence in enumerate(sequences):
        value = _metrics(result, "sequence_identity_gate", row=i)
        expected_max = max(
            nw_identity(sequence, other) for j, other in enumerate(sequences) if i != j
        )
        assert value["n_intra_submission_comparisons"] == expected_comparisons
        assert value["max_intra_submission_identity"] == pytest.approx(expected_max)


def test_threshold_is_user_settable(tmp_path):
    """The same design flips verdict with the threshold, and nothing else."""
    fasta = tmp_path / "refs.fasta"
    fasta.write_text(">partial\nGACAAA\n")  # 4/6 identical to the design
    handler = PipelineDataHandler(
        [make_row(name="design", path=RNA_FIXTURE, metadata={}, metrics={})]
    )
    strict = SequenceIdentityGateBlock(references=str(fasta), threshold=0.40)
    loose = SequenceIdentityGateBlock(references=str(fasta), threshold=0.90)

    def get(b):
        return _metrics(b(handler, sandbox_dir=tmp_path), "sequence_identity_gate")

    assert bool(get(strict)["passed"]) is False
    assert bool(get(loose)["passed"]) is True


def test_motif_residues_are_excluded_from_the_scaffold(tmp_path):
    """With every residue marked as motif, the scaffold must be empty."""
    fasta = tmp_path / "refs.fasta"
    fasta.write_text(">native\nGACUAC\n")
    motif = ["A1", "A2", "A3", "A4", "B1", "B2"]
    handler = PipelineDataHandler(
        [
            make_row(
                name="design",
                path=RNA_FIXTURE,
                metadata={"motif": motif},
                metrics={},
            )
        ]
    )
    block = SequenceIdentityGateBlock(references=str(fasta))
    value = _metrics(block(handler, sandbox_dir=tmp_path), "sequence_identity_gate")
    assert value["n_motif_residues_excluded"] == 6
    assert value["scaffold_length"] == 0
    assert value["status"] == "na"


def test_fasta_reader_uppercases_and_converts_t(tmp_path):
    f = tmp_path / "x.fasta"
    f.write_text(">a\nacgt\nACGT\n>b\nGGGG\n")
    assert read_fasta(f) == [("a", "ACGUACGU"), ("b", "GGGG")]


# ── T4: TM-score ───────────────────────────────────────────────────────


@requires_tmtools
def test_tm_score_against_itself_is_one(tmp_path):
    handler = PipelineDataHandler(
        [make_row(name="design", path=RNA_FIXTURE, metadata={}, metrics={})]
    )
    block = TMScoreBlock(reference=RNA_FIXTURE)
    value = _metrics(block(handler, sandbox_dir=tmp_path), "tm_score")
    assert value["tm_ref"] == pytest.approx(1.0, abs=1e-3)
    assert value["tm_design"] == pytest.approx(1.0, abs=1e-3)
    assert value["rmsd_aligned"] == pytest.approx(0.0, abs=1e-3)


def test_anchors_skip_ions_and_pick_one_atom_per_residue():
    coords, seq = polymer_anchors(parse_to_atom_array(RNA_FIXTURE))
    assert len(coords) == 6  # 6 nucleotides, Mg excluded
    assert seq == "GACUAC"
    assert coords.dtype == np.float64


def test_reference_chain_selection(tmp_path):
    """Restricting the reference to chain B must leave only its 2 residues."""
    coords, seq = polymer_anchors(parse_to_atom_array(RNA_FIXTURE), chain="B")
    assert len(coords) == 2
    assert seq == "AC"
    with pytest.raises(ValueError, match="at least 3"):
        TMScoreBlock(reference=RNA_FIXTURE, reference_chain="B").setup(
            None, tmp_path, 0
        )


@requires_vienna
def test_pairs_vienna_cannot_model_are_dropped_not_scored():
    """The silent failure this guards: hc_add_bp ignores a constraint on bases
    closer than the minimum hairpin loop, and the identical partition functions
    then read as p = 1.0 -- a perfect score for a pair never considered."""
    p, g0, g1, dropped = motif_boltzmann_probability("GGGGAAAACCCC", [(3, 4)])
    assert p is None and g1 is None
    assert dropped == [(3, 4, "separation")]

    p, _, _, dropped = motif_boltzmann_probability("GGGGAAAACCCC", [(4, 9)])
    assert p is None
    assert dropped == [(4, 9, "noncanonical")]


def test_identity_accepts_rna_alphabet():
    """biotite's NucleotideSequence is a DNA alphabet, so U must be relabelled
    before alignment or every RNA comparison raises."""
    assert nw_identity("ACGUACGU", "ACGUACGU") == pytest.approx(1.0)
    assert nw_identity("UUUUUUUU", "AAAAAAAA") == pytest.approx(0.0)
    # The relabel must not make U and T look different from each other.
    assert nw_identity("ACGU", "ACGT") == pytest.approx(1.0)


def test_gate_on_a_uracil_containing_design(tmp_path):
    fasta = tmp_path / "refs.fasta"
    fasta.write_text(">native\nGGGGUUUUCCCC\n")
    handler = PipelineDataHandler(
        [make_row(name="design", path=RNA_FIXTURE, metadata={}, metrics={})]
    )
    block = SequenceIdentityGateBlock(references=str(fasta), threshold=0.40)
    value = _metrics(block(handler, sandbox_dir=tmp_path), "sequence_identity_gate")
    assert value["status"] == "ok"
