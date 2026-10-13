"""Multi-character chain ids (RFD3 unindexed residues) are renamed for MPNN and restored."""

import gzip
import json
import shutil

import pytest
from atomworks.io.utils.io_utils import to_cif_file
from biotite.structure import Atom, array

from blocks.networks.mpnn import ChainRename, ClassicMPNNBlock
from chaperone import PipelineDataHandler
from chaperone.data import make_row
from chaperone.io import ensure_bonds, parse_to_atom_array


@pytest.fixture(autouse=True)
def use_legacy_runtime(monkeypatch):
    import blocks.networks.mpnn as module
    monkeypatch.setattr(module, "MODELHUB_CONTAINER", module._MODELHUB_SIF_CONTAINER)


def _atoms(chains):
    atoms = []
    for chain, res_ids in chains.items():
        for res_id in res_ids:
            for atom_name, offset in (("N", 0.0), ("CA", 1.4), ("C", 2.4)):
                atoms.append(
                    Atom(
                        [len(atoms) * 1.5 + offset, 0.0, 0.0],
                        chain_id=chain,
                        res_id=res_id,
                        res_name="GLY",
                        atom_name=atom_name,
                        element=atom_name[0],
                        hetero=False,
                    )
                )
    return array(atoms)


def _write(path, chains):
    to_cif_file(ensure_bonds(_atoms(chains)), str(path))
    return path


def _chain_ids(path):
    return set(parse_to_atom_array(str(path)).chain_id)


def _structure_block():
    # The rename only matters when MPNN writes structures: the default sequence mode
    # writes no PDB files, so multi-character chain ids are never a problem there.
    return ClassicMPNNBlock(
        num_seq_per_pdb=1, output_mode="structures", args={"pack_side_chains": 1}
    )


# --- ChainRename -----------------------------------------------------------


def test_multi_character_chains_get_unused_single_letters():
    rename = ChainRename.for_atoms(_atoms({"A": [1, 2], "X1": [5], "X2": [9]}))

    assert rename.mapping == {"X1": "B", "X2": "C"}
    assert bool(rename)


def test_single_character_chains_need_no_rename():
    rename = ChainRename.for_atoms(_atoms({"A": [1], "B": [2]}))

    assert not rename
    assert rename.mapping == {}


def test_running_out_of_single_letters_is_an_error():
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    chains = {letter: [1] for letter in letters[:-1]}  # only Z is left
    chains.update({"X1": [1], "X2": [1]})

    with pytest.raises(ValueError, match="not enough unused"):
        ChainRename.for_atoms(_atoms(chains))


def test_rename_returns_a_copy_and_leaves_the_input_alone():
    atoms = _atoms({"A": [1], "X1": [5, 6]})
    rename = ChainRename.for_atoms(atoms)

    renamed = rename.rename(atoms)

    assert set(renamed.chain_id) == {"A", "B"}
    assert set(renamed.res_id[renamed.chain_id == "B"]) == {5, 6}
    assert set(atoms.chain_id) == {"A", "X1"}


def test_positions_follow_the_rename():
    rename = ChainRename.for_atoms(_atoms({"A": [1], "X1": [5], "X2": [9]}))

    assert rename.apply("X15") == "B5"
    assert rename.apply("X29") == "C9"
    assert rename.apply("A1") == "A1"
    assert (
        rename.apply("_suppress_cif_motif_inference") == "_suppress_cif_motif_inference"
    )
    assert ChainRename().apply("A7") == "A7"


def test_a_position_naming_a_real_residue_of_a_kept_chain_is_not_renamed():
    # "X15" is chain X residue 15 here, not chain X1 residue 5.
    rename = ChainRename.for_atoms(_atoms({"A": [1], "X": [15], "X1": [5]}))

    assert rename.mapping == {"X1": "B"}
    assert rename.apply("X15") == "X15"
    assert rename.apply("X14") == "B4"  # chain X has no residue 14


def test_only_files_that_look_like_this_rename_are_restorable():
    rename = ChainRename({"X1": "B"})

    assert rename.restorable_in(["A", "B"])
    assert not rename.restorable_in(["A"])  # the stand-in is missing
    assert not rename.restorable_in(["A", "B", "X1"])  # an original is present


# --- restoring a CIF -------------------------------------------------------


def test_restore_cif_gives_back_the_original_chain_ids(tmp_path):
    atoms = _atoms({"A": [1, 2], "X1": [5, 6], "X2": [9]})
    rename = ChainRename.for_atoms(atoms)
    path = _write(tmp_path / "out.cif", {"A": [1, 2], "B": [5, 6], "C": [9]})

    rename.restore_cif(path)

    restored = parse_to_atom_array(str(path))
    assert set(restored.chain_id) == {"A", "X1", "X2"}
    assert set(restored.res_id[restored.chain_id == "X1"]) == {5, 6}
    assert len(restored) == len(atoms)


def test_restore_cif_handles_gzipped_files(tmp_path):
    rename = ChainRename({"X1": "B"})
    plain = _write(tmp_path / "out.cif", {"A": [1], "B": [5]})
    zipped = tmp_path / "out.cif.gz"
    with plain.open("rb") as source, gzip.open(zipped, "wb") as target:
        shutil.copyfileobj(source, target)

    rename.restore_cif(zipped)

    with gzip.open(zipped, "rt") as handle:
        assert "X1" in handle.read()


# --- the block, end to end -------------------------------------------------


def test_block_renames_for_mpnn_and_restores_the_outputs(tmp_path):
    source = _write(tmp_path / "design.cif", {"A": [1, 2, 3], "X1": [5, 6], "X2": [9]})
    row = make_row("rfd3", str(source), metadata={"motif": ["A1", "X15", "X29"]})
    block = _structure_block()
    sandbox = tmp_path / "chunk_0"

    block.create_command(PipelineDataHandler([row]), sandbox, chunk_id=0)

    # MPNN is handed a renamed copy, and the fixed positions follow the rename.
    staged = json.loads(
        (sandbox / "inputs" / "pdb_position_fixed_chunk0.json").read_text()
    )
    ((renamed_path, positions),) = staged.items()
    assert renamed_path.endswith("chain_ids_renamed/design.cif")
    assert positions == ["A1", "B5", "C9"]
    assert _chain_ids(renamed_path) == {"A", "B", "C"}
    assert _chain_ids(source) == {"A", "X1", "X2"}  # the input is untouched

    # What MPNN would write: structures named after the input, with the stand-ins.
    packed = sandbox / "packed"
    packed.mkdir()
    shutil.copy(renamed_path, packed / "design_packed_1_1.cif")
    unrelated = _write(packed / "design2_packed_1_1.cif", {"A": [1], "B": [2]})

    handler = block.compile(sandbox)

    paths = sorted(
        entry["path"]
        for structures in handler._df.get_column("structures").to_list()
        for entry in structures
    )
    assert len(paths) == 2
    by_name = {path.rsplit("/", 1)[-1]: path for path in paths}
    assert _chain_ids(by_name["design_packed_1_1.cif"]) == {"A", "X1", "X2"}
    # An output that only shares a name prefix is left alone.
    assert _chain_ids(by_name["design2_packed_1_1.cif"]) == {"A", "B"}
    assert unrelated.is_file()


def test_block_with_only_single_letter_chains_writes_no_copy_and_no_sidecar(tmp_path):
    source = _write(tmp_path / "design.cif", {"A": [1, 2], "B": [3]})
    row = make_row("rfd3", str(source), metadata={"motif": ["A1"]})
    sandbox = tmp_path / "chunk_0"

    _structure_block().create_command(PipelineDataHandler([row]), sandbox, chunk_id=0)

    assert not (sandbox / "inputs" / "chain_ids_renamed").exists()
    assert not list((sandbox / "inputs").glob("chain_renames_chunk*.json"))


def test_sequence_mode_leaves_chain_ids_alone(tmp_path):
    source = _write(tmp_path / "design.cif", {"A": [1, 2], "X1": [5]})
    row = make_row("rfd3", str(source), metadata={"motif": ["X15"]})
    sandbox = tmp_path / "chunk_0"

    ClassicMPNNBlock(num_seq_per_pdb=1).create_command(
        PipelineDataHandler([row]), sandbox, chunk_id=0
    )

    staged = json.loads(
        (sandbox / "inputs" / "pdb_position_fixed_chunk0.json").read_text()
    )
    ((path, positions),) = staged.items()
    assert path.endswith("design.cif") and "chain_ids_renamed" not in path
    assert positions == ["X15"]


def test_portable_cif_staging_and_sequence_backlinks(tmp_path, monkeypatch):
    import blocks.networks.mpnn as module
    from chaperone.data.schema import get_sequences_from_row

    monkeypatch.setattr(module, "MODELHUB_CONTAINER", module.MODELHUB_PORTABLE_CONTAINER)
    source = _write(tmp_path / "design.cif", {"Z": [1, 2], "X1": [5]})
    row = make_row("rfd3", str(source), design_id="parent", metadata={"motif": ["X15"]})
    sandbox = tmp_path / "chunk_0"
    block = ClassicMPNNBlock(num_seq_per_pdb=1)
    block.create_command(PipelineDataHandler([row]), sandbox, chunk_id=0)
    ((path, positions),) = json.loads(
        (sandbox / "inputs/pdb_position_fixed_chunk0.json").read_text()
    ).items()
    assert path.endswith("portable_pdb/design.pdb")
    assert positions == ["A5"]
    assert _chain_ids(path) == {"Z", "A"}
    assert _chain_ids(source) == {"Z", "X1"}
    # The native runner sorts A before Z, even though the input begins with Z.
    seqs = sandbox / "seqs"
    seqs.mkdir()
    (seqs / "design.fa").write_text(
        ">design, T=0.1\nG/GG\n>design, id=1, overall_confidence=0.8\nW/AA\n"
    )
    compiled = block.compile(sandbox)._df.to_dicts()
    assert len(compiled) == 1
    assert compiled[0]["parent_design_id"] == "parent"
    assert get_sequences_from_row(compiled[0]) == {"X1": "W", "Z": "AA"}


def test_portable_packed_pdb_restores_multichar_chains(tmp_path, monkeypatch):
    import blocks.networks.mpnn as module
    monkeypatch.setattr(module, "MODELHUB_CONTAINER", module.MODELHUB_PORTABLE_CONTAINER)
    source = _write(tmp_path / "design.cif", {"A": [1, 2], "X1": [5]})
    row = make_row("rfd3", str(source), metadata={"motif": ["X15"]})
    sandbox = tmp_path / "chunk_0"
    block = _structure_block()
    block.create_command(PipelineDataHandler([row]), sandbox, chunk_id=0)
    packed = sandbox / "packed"
    packed.mkdir()
    shutil.copy(sandbox / "inputs/portable_pdb/design.pdb", packed / "design_packed_1_1.pdb")
    for _ in range(2):  # recompilation must not double-count the restored CIF.
        handler = block.compile(sandbox)
        assert len(handler) == 1
        path = handler._df["structures"].to_list()[0][0]["path"]
        assert path.endswith(".cif")
        assert _chain_ids(path) == {"A", "X1"}


@pytest.mark.parametrize(
    "preference,backend,portable_exists,legacy_exists,expected",
    [
        ("auto", "apptainer", False, True, "legacy"),
        ("auto", "apptainer", True, True, "portable"),
        ("auto", "apptainer", False, False, "portable"),
        ("auto", "docker", False, True, "portable"),
        ("portable", "apptainer", False, True, "portable"),
        ("legacy", "apptainer", True, True, "legacy"),
        ("legacy", "docker", True, True, "error"),
    ],
)
def test_container_selection_preference(
    tmp_path, monkeypatch, preference, backend, portable_exists, legacy_exists, expected
):
    import blocks.networks.mpnn as module
    from chaperone import get_settings

    monkeypatch.setattr(get_settings(), "CHAPERONE_MPNN_CONTAINER", preference)
    monkeypatch.setattr(get_settings(), "CHAPERONE_CONTAINER_BACKEND", backend)
    monkeypatch.setattr(get_settings(), "MPNN_WEIGHTS_DIR", str(tmp_path / "site-profile-weights"))
    monkeypatch.setattr(module, "get_container_dirs", lambda: [tmp_path])
    legacy = tmp_path / "legacy.sif"
    monkeypatch.setattr(module, "_ONPREM_MODELHUB_SIF", str(legacy))
    if portable_exists:
        (tmp_path / "modelhub_portable.sif").touch()
    if legacy_exists:
        legacy.touch()
    if expected == "error":
        with pytest.raises(ValueError, match="requires Apptainer"):
            module._pick_modelhub_container()
    else:
        wanted = (
            module.MODELHUB_PORTABLE_CONTAINER if expected == "portable"
            else module._MODELHUB_SIF_CONTAINER
        )
        assert module._pick_modelhub_container() is wanted
