"""Public LigandMPNN adapter contracts, independent of model installation."""

import json
import shutil

import pytest
from atomworks.io.utils.io_utils import to_cif_file
from biotite.structure import Atom, array

import blocks.networks.mpnn as mpnn
from chaperone import ExecutionEngine, PipelineDataHandler, get_settings
from chaperone.data import make_row
from chaperone.data.schema import get_sequences_from_row
from chaperone.io import ensure_bonds, parse_to_atom_array


def input_handler(tmp_path):
    atoms = []
    for chain, residues in {"Z": [1, 2], "X1": [5]}.items():
        for residue in residues:
            for name in ("N", "CA", "C", "O"):
                atoms.append(Atom([len(atoms) * 1.5, 0, 0], chain_id=chain, res_id=residue,
                                  res_name="GLY", atom_name=name, element=name[0], hetero=False))
    path = tmp_path / "design.cif"
    to_cif_file(ensure_bonds(array(atoms)), str(path))
    return PipelineDataHandler([make_row("rfd3", str(path), design_id="parent", metadata={"motif": ["X15"]})])


def option(command, name):
    return command[command.index(f"--{name}") + 1]


def run_py_arguments(command):
    """Everything after the interpreter and ``run.py`` (container prefix dropped)."""
    return [str(part) for part in command[command.index(f"{mpnn._PUBLIC_MPNN_ROOT}/run.py") + 1 :]]


def params(filename):
    return f"{mpnn._PUBLIC_MPNN_ROOT}/model_params/{filename}"


# Every checkpoint of the pinned upstream revision (get_model_params.sh), by
# (model_type, noise_level). Spelled out here, not derived, so an edit to the
# table in the block cannot silently change what a request means.
EXPECTED_CHECKPOINTS = {
    ("protein_mpnn", 0.02): "proteinmpnn_v_48_002.pt",
    ("protein_mpnn", 0.1): "proteinmpnn_v_48_010.pt",
    ("protein_mpnn", 0.2): "proteinmpnn_v_48_020.pt",
    ("protein_mpnn", 0.3): "proteinmpnn_v_48_030.pt",
    ("ligand_mpnn", 0.05): "ligandmpnn_v_32_005_25.pt",
    ("ligand_mpnn", 0.1): "ligandmpnn_v_32_010_25.pt",
    ("ligand_mpnn", 0.2): "ligandmpnn_v_32_020_25.pt",
    ("ligand_mpnn", 0.3): "ligandmpnn_v_32_030_25.pt",
    ("soluble_mpnn", 0.02): "solublempnn_v_48_002.pt",
    ("soluble_mpnn", 0.1): "solublempnn_v_48_010.pt",
    ("soluble_mpnn", 0.2): "solublempnn_v_48_020.pt",
    ("soluble_mpnn", 0.3): "solublempnn_v_48_030.pt",
    ("per_residue_label_membrane_mpnn", 0.2): "per_residue_label_membrane_mpnn_v_48_020.pt",
    ("global_label_membrane_mpnn", 0.2): "global_label_membrane_mpnn_v_48_020.pt",
}
UPSTREAM_DEFAULTS = {
    "protein_mpnn": "proteinmpnn_v_48_020.pt",
    "ligand_mpnn": "ligandmpnn_v_32_010_25.pt",
    "soluble_mpnn": "solublempnn_v_48_020.pt",
    "per_residue_label_membrane_mpnn": "per_residue_label_membrane_mpnn_v_48_020.pt",
    "global_label_membrane_mpnn": "global_label_membrane_mpnn_v_48_020.pt",
}


def test_recipe_pins_source_and_bakes_every_official_checkpoint():
    recipe = mpnn.LIGANDMPNN_PUBLIC_CONTAINER.to_singularity()
    assert mpnn._PUBLIC_MPNN_REVISION in recipe
    assert "torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128" in recipe
    filenames = {name for name, _ in mpnn._PUBLIC_MPNN_ALL_CHECKPOINTS}
    assert filenames == set(EXPECTED_CHECKPOINTS.values()) | {"ligandmpnn_sc_v_32_002_16.pt"}
    for filename, checksum in mpnn._PUBLIC_MPNN_ALL_CHECKPOINTS:
        assert f"https://files.ipd.uw.edu/pub/ligandmpnn/{filename} " in recipe
        assert f"echo '{checksum}  {mpnn._PUBLIC_MPNN_ROOT}/model_params/{filename}' | sha256sum -c -" in recipe
    assert recipe.count("sha256sum -c -") == len(EXPECTED_CHECKPOINTS) + 1 == 15
    assert len({checksum for _, checksum in mpnn._PUBLIC_MPNN_ALL_CHECKPOINTS}) == 15
    assert "pip check" in recipe and "import run, data_utils, model_utils, sc_utils" in recipe


def test_block_image_is_the_public_image():
    assert mpnn.ClassicMPNNBlock._get_image() is mpnn.LIGANDMPNN_PUBLIC_CONTAINER
    assert mpnn.LIGANDMPNN_PUBLIC_CONTAINER.name == "ligandmpnn_public"


def test_public_numpy_patch_replaces_only_removed_aliases_and_checks_pinned_source(tmp_path):
    sources = {
        "np/residue_constants.py": "np.int, np.int32, np.int, np.int64, np.int\n",
        "data/templates.py": "np.object, np.object_, np.object\n",
        "np/relax/utils.py": "np.bool, np.bool_\n",
    }
    for name, source in sources.items():
        path = tmp_path / "openfold" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    patch = mpnn._PUBLIC_MPNN_NUMPY_COMPATIBILITY_PATCH.replace(mpnn._PUBLIC_MPNN_ROOT, str(tmp_path))
    exec(compile(patch, "numpy_compatibility_patch", "exec"), {})
    assert (tmp_path / "openfold/np/residue_constants.py").read_text() == "int, np.int32, int, np.int64, int\n"
    assert (tmp_path / "openfold/data/templates.py").read_text() == "object, np.object_, object\n"
    assert (tmp_path / "openfold/np/relax/utils.py").read_text() == "bool, np.bool_\n"
    with pytest.raises(RuntimeError, match="Unexpected NumPy alias count"):
        exec(compile(patch, "numpy_compatibility_patch", "exec"), {})


def test_model_choice_has_distinct_cache_identity():
    fingerprints = {
        ExecutionEngine._cache_fingerprint([mpnn.ClassicMPNNBlock(**kwargs)])
        for kwargs in (
            {},
            {"noise_level": 0.2},
            {"model_type": "protein_mpnn"},
            {"model_type": "soluble_mpnn"},
            {"model_type": "soluble_mpnn", "noise_level": 0.02},
        )
    }
    assert len(fingerprints) == 5


@pytest.mark.parametrize(
    "backend,env_flag",
    [("apptainer", "--env"), ("docker", "-e")],
)
def test_cpu_device_is_masked_inside_each_container_runtime(monkeypatch, tmp_path, backend, env_flag):
    monkeypatch.setattr(get_settings(), "CHAPERONE_CONTAINER_BACKEND", backend)
    monkeypatch.setattr("chaperone.containers._supports_private_mount_namespace", lambda: False)
    block = mpnn.ClassicMPNNBlock(device="cpu")
    command = block.create_command(input_handler(tmp_path), tmp_path / "chunk", 0)
    assert command[0] == backend
    mask_index = command.index("CUDA_VISIBLE_DEVICES=")
    assert command[mask_index - 1] == env_flag
    assert mask_index < command.index("python")
    assert command.count("CUDA_VISIBLE_DEVICES=") == 1
    assert "CUDA_VISIBLE_DEVICES=" not in mpnn.ClassicMPNNBlock().command_template


# ── routing: model_type x noise_level ──────────────────────────────────────


@pytest.mark.parametrize(("model_type", "noise_level"), list(EXPECTED_CHECKPOINTS))
def test_every_official_checkpoint_is_routable(model_type, noise_level):
    block = mpnn.ClassicMPNNBlock(model_type=model_type, noise_level=noise_level)
    assert option(block.command_template, f"checkpoint_{model_type}") == params(
        EXPECTED_CHECKPOINTS[(model_type, noise_level)]
    )


@pytest.mark.parametrize("model_type", UPSTREAM_DEFAULTS)
def test_default_noise_level_is_the_upstream_default(model_type):
    block = mpnn.ClassicMPNNBlock(model_type=model_type)
    assert block.noise_level is None
    assert option(block.command_template, f"checkpoint_{model_type}") == params(UPSTREAM_DEFAULTS[model_type])


def test_noise_level_changes_only_the_selected_model_checkpoint():
    default = mpnn.ClassicMPNNBlock().command_template
    block = mpnn.ClassicMPNNBlock(model_type="soluble_mpnn", noise_level=0.3)
    changed = [i for i, (a, b) in enumerate(zip(default, block.command_template)) if a != b]
    assert [block.command_template[i] for i in changed] == [params("solublempnn_v_48_030.pt")]
    assert default[changed[0] - 1] == "--checkpoint_soluble_mpnn"


@pytest.mark.parametrize("model_type", list(UPSTREAM_DEFAULTS))
def test_create_command_routes_model_type_to_run_py(tmp_path, model_type):
    block = mpnn.ClassicMPNNBlock(model_type=model_type, num_seq_per_pdb=1)
    command = block.create_command(input_handler(tmp_path), tmp_path / "chunk", 0)
    assert option(command, "model_type") == model_type
    assert option(command, f"checkpoint_{model_type}") == params(UPSTREAM_DEFAULTS[model_type])


def test_soluble_mpnn_end_to_end_with_version(tmp_path):
    block = mpnn.ClassicMPNNBlock(model_type="soluble_mpnn", noise_level=0.02, num_seq_per_pdb=1)
    sandbox = tmp_path / "chunk"
    command = block.create_command(input_handler(tmp_path), sandbox, 0)
    assert option(command, "model_type") == "soluble_mpnn"
    assert option(command, "checkpoint_soluble_mpnn") == params("solublempnn_v_48_002.pt")
    assert option(command, "pack_side_chains") == "0"
    seqs = sandbox / "seqs"
    seqs.mkdir()
    (seqs / "design.fa").write_text(
        ">design, T=0.1, seed=0, num_res=3, model_path=x\nG/GG\n"
        ">design, id=1, T=0.1, seed=0, overall_confidence=0.5, ligand_confidence=0.0, seq_rec=0.3\nW/AA\n"
    )
    row, = block.compile(sandbox)._df.to_dicts()
    assert get_sequences_from_row(row) == {"X1": "W", "Z": "AA"}


@pytest.mark.parametrize(
    ("model_type", "noise_level"),
    [
        ("ligand_mpnn", 0.02),
        ("protein_mpnn", 0.05),
        ("soluble_mpnn", 0.05),
        ("per_residue_label_membrane_mpnn", 0.1),
        ("global_label_membrane_mpnn", 0.3),
    ],
)
def test_unavailable_model_version_lists_the_valid_ones(model_type, noise_level):
    with pytest.raises(ValueError) as error:
        mpnn.ClassicMPNNBlock(model_type=model_type, noise_level=noise_level)
    message = str(error.value)
    assert f"model_type='{model_type}' has no checkpoint with noise_level={noise_level}" in message
    for valid in (
        "protein_mpnn: 0.02, 0.1, 0.2, 0.3",
        "ligand_mpnn: 0.05, 0.1, 0.2, 0.3",
        "soluble_mpnn: 0.02, 0.1, 0.2, 0.3",
        "per_residue_label_membrane_mpnn: 0.2",
        "global_label_membrane_mpnn: 0.2",
    ):
        assert valid in message


def test_unknown_model_type_and_noise_level_are_rejected():
    with pytest.raises(ValueError, match="model_type"):
        mpnn.ClassicMPNNBlock(model_type="mpnn_v3")
    with pytest.raises(ValueError, match="noise_level"):
        mpnn.ClassicMPNNBlock(noise_level=0.15)


def test_an_option_the_block_does_not_have_is_an_error():
    # Options of removed backends must not be silently ignored.
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        mpnn.ClassicMPNNBlock(removed_option=True)


@pytest.mark.parametrize(
    ("model_type", "args"),
    [
        ("ligand_mpnn", {"transmembrane_buried": "Z1"}),
        ("soluble_mpnn", {"transmembrane_interface": "Z1"}),
        ("global_label_membrane_mpnn", {"transmembrane_buried": "Z1"}),
        ("per_residue_label_membrane_mpnn", {"global_transmembrane_label": 1}),
    ],
)
def test_membrane_options_need_their_membrane_model(model_type, args):
    with pytest.raises(ValueError, match="only applies to model_type"):
        mpnn.ClassicMPNNBlock(model_type=model_type, args=args)


def test_membrane_options_reach_run_py_with_renamed_chains(tmp_path):
    block = mpnn.ClassicMPNNBlock(
        model_type="per_residue_label_membrane_mpnn",
        args={"transmembrane_buried": "Z1 X15", "transmembrane_interface": "Z2"},
    )
    command = block.create_command(input_handler(tmp_path), tmp_path / "chunk", 0)
    assert option(command, "transmembrane_buried") == "Z1 A5"
    assert option(command, "transmembrane_interface") == "Z2"
    block = mpnn.ClassicMPNNBlock(model_type="global_label_membrane_mpnn", args={"global_transmembrane_label": 1})
    command = block.create_command(input_handler(tmp_path), tmp_path / "chunk2", 0)
    assert option(command, "global_transmembrane_label") == "1"


def test_default_public_command_is_unchanged(tmp_path):
    """Golden: the default ligand_mpnn command, as before the routing rework."""
    block = mpnn.ClassicMPNNBlock()
    sandbox = tmp_path / "chunk"
    command = block.create_command(input_handler(tmp_path), sandbox, 0)
    inputs = f"{sandbox}/inputs/pdb_position_fixed_chunk0.json"
    assert run_py_arguments(command) == [
        "--checkpoint_protein_mpnn", params("proteinmpnn_v_48_020.pt"),
        "--checkpoint_ligand_mpnn", params("ligandmpnn_v_32_010_25.pt"),
        "--checkpoint_soluble_mpnn", params("solublempnn_v_48_020.pt"),
        "--checkpoint_per_residue_label_membrane_mpnn", params("per_residue_label_membrane_mpnn_v_48_020.pt"),
        "--checkpoint_global_label_membrane_mpnn", params("global_label_membrane_mpnn_v_48_020.pt"),
        "--out_folder", str(sandbox),
        "--model_type", "ligand_mpnn",
        "--pdb_path_multi", inputs,
        "--fixed_residues_multi", inputs,
        "--batch_size", "8",
        "--temperature", "0.1",
        "--omit_AA", "X",
        "--ligand_mpnn_use_side_chain_context", "1",
        "--pack_side_chains", "0",
        "--checkpoint_path_sc", params("ligandmpnn_sc_v_32_002_16.pt"),
        "--repack_everything", "0",
        "--packed_suffix", "_",
        "--zero_indexed", "1",
        "--force_hetatm", "1",
        "--fasta_seq_separation", "/",
        "--number_of_packs_per_design", "4",
        "--pack_with_ligand_context", "1",
    ]
    assert command[command.index("python") :][:3] == ["python", "-u", f"{mpnn._PUBLIC_MPNN_ROOT}/run.py"]


def test_public_sequence_contract_and_original_chain_lineage(tmp_path):
    block = mpnn.ClassicMPNNBlock(num_seq_per_pdb=1, args={"chains_to_design": "X1", "symmetry_residues": "X15,Z1"})
    sandbox = tmp_path / "chunk"
    command = block.create_command(input_handler(tmp_path), sandbox, 0)
    assert option(command, "fasta_seq_separation") == "/"
    assert option(command, "pack_side_chains") == "0"
    assert option(command, "chains_to_design") == "A"
    assert option(command, "symmetry_residues") == "A5,Z1"
    assert option(command, "checkpoint_ligand_mpnn") == params("ligandmpnn_v_32_010_25.pt")
    ((path, positions),) = json.loads((sandbox / "inputs/pdb_position_fixed_chunk0.json").read_text()).items()
    assert path.endswith("portable_pdb/design.pdb")
    assert positions == "A5"  # run.py calls value.split() on the fixed-residue string
    assert set(parse_to_atom_array(path).chain_id) == {"A", "Z"}
    seqs = sandbox / "seqs"
    seqs.mkdir()
    (seqs / "design.fa").write_text(
        ">design, T=0.1\nG/GG\n>design, id=0, overall_confidence=0.8\nW/AA\n"
    )
    row, = block.compile(sandbox)._df.to_dicts()
    assert row["parent_design_id"] == "parent"
    assert get_sequences_from_row(row) == {"X1": "W", "Z": "AA"}


def test_public_chain_filter_updates_fasta_contract(tmp_path):
    block = mpnn.ClassicMPNNBlock(args={"parse_these_chains_only": "X1"})
    sandbox = tmp_path / "chunk"
    command = block.create_command(input_handler(tmp_path), sandbox, 0)
    assert option(command, "parse_these_chains_only") == "A"
    seqs = sandbox / "seqs"
    seqs.mkdir()
    (seqs / "design.fa").write_text(">design, id=0\nW\n")
    row, = block.compile(sandbox)._df.to_dicts()
    assert get_sequences_from_row(row) == {"X1": "W"}


def test_public_packing_uses_public_weights_and_restores_chain_ids(tmp_path):
    block = mpnn.ClassicMPNNBlock(output_mode="structures", num_seq_per_pdb=1)
    sandbox = tmp_path / "chunk"
    command = block.create_command(input_handler(tmp_path), sandbox, 0)
    assert option(command, "pack_side_chains") == "1"
    assert option(command, "number_of_packs_per_design") == "4"
    assert option(command, "checkpoint_path_sc") == params("ligandmpnn_sc_v_32_002_16.pt")
    assert option(command, "pack_with_ligand_context") == "1"
    packed = sandbox / "packed"
    packed.mkdir()
    shutil.copy(sandbox / "inputs/portable_pdb/design.pdb", packed / "design__0_1.pdb")
    row, = block.compile(sandbox)._df.to_dicts()
    assert set(parse_to_atom_array(row["structures"][0]["path"]).chain_id) == {"X1", "Z"}


def test_custom_packer_is_bound_read_only_and_must_exist(tmp_path):
    packer = tmp_path / "packer.pt"
    packer.write_bytes(b"x")
    block = mpnn.ClassicMPNNBlock(output_mode="structures", args={"checkpoint_path_sc": str(packer)})
    assert f"{packer}:{packer}:ro" in block.command_template
    command = block.create_command(input_handler(tmp_path), tmp_path / "chunk", 0)
    assert option(command, "checkpoint_path_sc") == str(packer)
    with pytest.raises(FileNotFoundError, match="custom packer checkpoint does not exist"):
        mpnn.ClassicMPNNBlock(output_mode="structures", args={"checkpoint_path_sc": str(tmp_path / "missing.pt")})


def test_public_rejects_ambiguous_residue_bias_before_execution(tmp_path):
    bias = tmp_path / "bias.json"
    bias.write_text('{"X15": {"A": 1.0}}')
    block = mpnn.ClassicMPNNBlock(args={"bias_AA_per_residue": str(bias)})
    with pytest.raises(ValueError, match="single-character input chain IDs"):
        block.create_command(input_handler(tmp_path), tmp_path / "chunk", 0)


def test_portable_pdb_missing_optional_fields_preserves_source_and_finite_values(tmp_path):
    import numpy as np
    from biotite.structure.io.pdb import PDBFile

    source = next(input_handler(tmp_path).get_data(load=True)).atom_array
    source.set_annotation("b_factor", np.array([np.nan, 12.5] * (len(source) // 2)))
    source.set_annotation("occupancy", np.array([0.5, np.nan] * (len(source) // 2)))
    before = source.copy()
    path = mpnn.ClassicMPNNBlock._write_portable_pdb(
        "source.cif", source, mpnn.ChainRename.for_atoms(source), tmp_path / "staged",
    )
    written = PDBFile.read(path).get_structure(model=1, extra_fields=["b_factor", "occupancy"])
    np.testing.assert_allclose(written.coord, source.coord, atol=1e-3)
    np.testing.assert_allclose(written.b_factor, [0.0, 12.5] * (len(source) // 2))
    np.testing.assert_allclose(written.occupancy, [0.5, 1.0] * (len(source) // 2))
    np.testing.assert_array_equal(source.b_factor, before.b_factor)
    np.testing.assert_array_equal(source.occupancy, before.occupancy)
    np.testing.assert_array_equal(source.chain_id, before.chain_id)
    source.coord[0, 0] = np.nan
    with pytest.raises(ValueError, match="PDB-representable"):
        mpnn.ClassicMPNNBlock._write_portable_pdb(
            "invalid.cif", source, mpnn.ChainRename.for_atoms(source), tmp_path / "invalid",
        )
