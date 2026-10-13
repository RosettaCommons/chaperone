"""Focused tests for the ColabFold (AlphaFold2) block's benchmark path."""

import gzip

import pytest

from blocks import Blocks
from blocks.networks.prediction.colabfold import (
    ColabFoldBlock,
    colabfold_fasta,
    colabfold_multimer_a3m,
    protein_chain_sequences,
    unique_chain_sequences,
)
from conftest import make_protein as _protein


def test_colabfold_is_discoverable():
    assert Blocks.get("colabfold") is ColabFoldBlock


def test_local_msas_and_server_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        ColabFoldBlock(use_local_msas=True, use_msa_server=True)


def test_msa_dirs_without_use_local_msas_is_rejected():
    with pytest.raises(ValueError, match="no effect"):
        ColabFoldBlock(msa_dirs=["/msa/store"])


def test_missing_msa_gate_requires_local_msas():
    with pytest.raises(ValueError, match="requires use_local_msas"):
        ColabFoldBlock(raise_if_missing_msa_for_protein_of_length_n=0)


def _staged_block(tmp_path, **kwargs) -> ColabFoldBlock:
    return ColabFoldBlock(command_template=["true"], **kwargs)


def _write_msa(store, sequence, hit):
    from atomworks.ml.utils.misc import hash_sequence

    with gzip.open(store / f"{hash_sequence(sequence)}.a3m.gz", "wt") as handle:
        handle.write(f">query\n{sequence}\n>hit\n{hit}\n")


def test_multimer_helpers_preserve_chain_order_and_homomer_cardinality():
    chains = protein_chain_sequences(_protein(chain_id="B") + _protein(chain_id="A"))

    assert [chain_id for chain_id, _, _ in chains] == ["B", "A"]
    assert colabfold_fasta("toy", chains) == ">toy\nAGS:AGS\n"
    assert unique_chain_sequences(chains) == [
        {
            "sequence": "AGS",
            "cardinality": 2,
            "chain_ids": ["B", "A"],
            "msa_lookup_sequences": [],
        }
    ]


def test_multimer_a3m_serializes_colabfold_header_and_padded_msas():
    unique = [
        {
            "sequence": "AGS",
            "cardinality": 1,
            "chain_ids": ["A"],
            "msa_lookup_sequences": [],
        },
        {
            "sequence": "GG",
            "cardinality": 1,
            "chain_ids": ["B"],
            "msa_lookup_sequences": [],
        },
    ]
    a3m = colabfold_multimer_a3m(
        unique,
        [([">query", ">a_hit"], ["AGS", "A-S"]), ([">query"], ["GG"])],
    )

    assert a3m == (
        "#3,2\t1,1\n"
        ">query\nAGSGG\n"
        ">query\nAGS--\n"
        ">a_hit\nA-S--\n"
        ">query\n---GG\n"
    )


def test_stage_multimer_uses_each_unique_chains_local_msa(tmp_path):
    chain_a = _protein(chain_id="A")
    chain_b = _protein(chain_id="B")
    chain_b.res_name[chain_b.res_id == 3] = "GLY"
    store = tmp_path / "store"
    store.mkdir()
    _write_msa(store, "AGS", "A-S")
    _write_msa(store, "AGG", "AG-")
    inputdir = tmp_path / "inputs"
    inputdir.mkdir()

    block = _staged_block(
        tmp_path,
        use_local_msas=True,
        msa_dirs=[{"dir": str(store), "extension": ".a3m.gz", "directory_depth": 0}],
    )
    is_multimer = block._stage_input(str(inputdir), "toy", chain_a + chain_b)

    assert is_multimer
    assert (inputdir / "toy.a3m").is_file()
    assert not (inputdir / "toy.fasta").exists()
    staged = (inputdir / "toy.a3m").read_text()
    assert staged.startswith("#3,3\t1,1\n>query\nAGSAGG\n")
    assert ">hit\nA-S---\n" in staged
    assert ">hit\n---AG-\n" in staged


def test_stage_homomer_collapses_identical_chains(tmp_path):
    inputdir = tmp_path / "inputs"
    inputdir.mkdir()
    store = tmp_path / "store"
    store.mkdir()
    _write_msa(store, "AGS", "A-S")

    block = _staged_block(
        tmp_path,
        use_local_msas=True,
        msa_dirs=[{"dir": str(store), "extension": ".a3m.gz", "directory_depth": 0}],
    )
    block._stage_input(
        str(inputdir), "toy", _protein(chain_id="A") + _protein(chain_id="B")
    )

    assert (inputdir / "toy.a3m").read_text().startswith("#3\t2\n>query\nAGS\n")


def test_stage_multimer_gate_fails_on_a_missing_chain_msa(tmp_path):
    inputdir = tmp_path / "inputs"
    inputdir.mkdir()
    empty = tmp_path / "empty"
    empty.mkdir()

    block = _staged_block(
        tmp_path,
        use_local_msas=True,
        msa_dirs=[str(empty)],
        raise_if_missing_msa_for_protein_of_length_n=0,
    )
    with pytest.raises(ValueError, match="No on-disk MSA"):
        block._stage_input(
            str(inputdir), "toy", _protein(chain_id="A") + _protein(chain_id="B")
        )


def test_create_command_stages_complex_fasta_and_rejects_monomer_model(tmp_path):
    from pipelines import PipelineDataHandler
    from pipelines.data import make_row
    from atomworks.io.utils.io_utils import to_cif_file
    from pipelines.io import ensure_bonds

    structure = tmp_path / "toy.cif"
    to_cif_file(
        ensure_bonds(_protein(chain_id="A") + _protein(chain_id="B")),
        structure,
    )
    handler = PipelineDataHandler([make_row(name="toy", path=structure)])

    sandbox = tmp_path / "auto"
    cmd = _staged_block(tmp_path).create_command(handler, sandbox, 0)
    assert (sandbox / "inputs" / "toy.fasta").read_text() == ">toy\nAGS:AGS\n"
    assert cmd[cmd.index("--model-type") + 1] == "auto"

    with pytest.raises(ValueError, match="cannot predict a multimer"):
        _staged_block(tmp_path, model_type="alphafold2_ptm").create_command(
            handler, tmp_path / "ptm", 0
        )


def test_compile_does_not_require_an_input_manifest(tmp_path):
    from atomworks.io.utils.io_utils import to_pdb_string

    outputdir = tmp_path / "outputs"
    outputdir.mkdir()
    (
        outputdir / "toy_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb"
    ).write_text(to_pdb_string(_protein()))

    block = _staged_block(tmp_path)
    result = block.compile(tmp_path).unpack()

    assert result.loc[0, "colabfold.rank"] == 1
    assert "validation_reference_path" not in result.columns
