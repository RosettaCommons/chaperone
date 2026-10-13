"""Tests for shared monomer sequence and MSA helpers."""

import gzip
from types import SimpleNamespace

import pytest

from blocks.networks.prediction.monomer_reference import (
    normalize_msa_dirs,
    residue_sequence,
    stage_monomer_manifest,
    stage_query_a3m,
    standard_residue_sequence,
)
from conftest import make_protein as protein


def test_stage_monomer_manifest_records_sequence_without_validation_cropping(tmp_path):
    source = tmp_path / "input.cif"
    loaded = SimpleNamespace(
        atom_array=protein(),
        basename="target",
        path=str(source),
        metrics={},
    )

    manifest = stage_monomer_manifest(
        loaded,
        manifest_dir=tmp_path / "manifest",
    )

    assert manifest["sequence"] == "AGS"
    assert manifest["source_path"] == str(source)
    assert "reference_path" not in manifest


def test_stage_monomer_manifest_rejects_multichain_input(tmp_path):
    loaded = SimpleNamespace(
        atom_array=protein() + protein(chain_id="B"),
        basename="target",
        path=str(tmp_path / "input.cif"),
        metrics={},
    )

    with pytest.raises(ValueError, match="monomer predictor"):
        stage_monomer_manifest(loaded, manifest_dir=tmp_path / "manifest")


def test_sequence_uses_ccd_parent_for_modified_residue():
    modified = protein()
    modified.res_name[modified.res_id == 1] = "MSE"
    modified.res_name[modified.res_id == 2] = "SEP"
    modified.res_name[modified.res_id == 3] = "A1EJS"

    assert residue_sequence(modified) == "MSK"
    # The MSA stores are keyed on the X-for-nonstandard encoding instead.
    assert standard_residue_sequence(modified) == "XXX"


def test_normalize_msa_dirs_requires_absolute_paths():
    assert normalize_msa_dirs(["/msa/store"]) == [
        {"dir": "/msa/store", "extension": ".a3m.gz", "directory_depth": 2}
    ]
    with pytest.raises(ValueError, match="absolute"):
        normalize_msa_dirs(["relative/store"])
    with pytest.raises(ValueError, match="at least one"):
        normalize_msa_dirs([])


def test_stage_query_a3m_strips_comments_and_verifies_query(tmp_path):
    source = tmp_path / "msa.a3m.gz"
    with gzip.open(source, "wt") as handle:
        handle.write("#189\t1\n>query\nAGS\n>hit1\nAG-S\n>hit2\nAaGS\n")
    staged = tmp_path / "target.a3m"

    depth = stage_query_a3m(source, staged, sequence="AGS")

    assert depth == 3
    assert staged.read_text().startswith(">query\nAGS\n")
    assert not staged.read_text().startswith("#")


def test_stage_query_a3m_rewrites_store_x_to_canonical_residue(tmp_path):
    source = tmp_path / "msa.a3m"
    source.write_text(">query\nXGS\n>hit\nAGS\n")
    staged = tmp_path / "target.a3m"

    stage_query_a3m(source, staged, sequence="MGS")

    assert staged.read_text().splitlines()[1] == "MGS"


def test_stage_query_a3m_rejects_a_different_sequence(tmp_path):
    source = tmp_path / "msa.a3m"
    source.write_text(">query\nAGS\n")

    with pytest.raises(ValueError, match="does not match the folded sequence"):
        stage_query_a3m(source, tmp_path / "target.a3m", sequence="AGA")


def test_stage_query_a3m_truncates_to_max_sequences(tmp_path):
    source = tmp_path / "msa.a3m"
    source.write_text(">query\nAGS\n>hit1\nAGS\n>hit2\nAGS\n")

    depth = stage_query_a3m(
        source, tmp_path / "target.a3m", sequence="AGS", max_sequences=2
    )

    assert depth == 2
