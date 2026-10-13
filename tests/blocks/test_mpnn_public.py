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


@pytest.fixture(autouse=True)
def public_backend(monkeypatch):
    monkeypatch.setattr(mpnn, "MODELHUB_CONTAINER", mpnn.LIGANDMPNN_PUBLIC_CONTAINER)
    # A supplied private layout must not leak into the explicit public backend.
    monkeypatch.setattr(get_settings(), "MPNN_WEIGHTS_DIR", "/nonexistent/private/weights")


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


@pytest.mark.parametrize("backend", ["apptainer", "docker"])
def test_unconfigured_auto_uses_public_source(monkeypatch, backend):
    monkeypatch.setattr(get_settings(), "CHAPERONE_MPNN_CONTAINER", "auto")
    monkeypatch.setattr(get_settings(), "CHAPERONE_CONTAINER_BACKEND", backend)
    monkeypatch.setattr(get_settings(), "MPNN_WEIGHTS_DIR", None)
    assert mpnn._pick_modelhub_container() is mpnn.LIGANDMPNN_PUBLIC_CONTAINER
    monkeypatch.setattr(get_settings(), "CHAPERONE_MPNN_CONTAINER", "public")
    monkeypatch.setattr(get_settings(), "MPNN_WEIGHTS_DIR", "/private/weights")
    assert mpnn._pick_modelhub_container() is mpnn.LIGANDMPNN_PUBLIC_CONTAINER


def test_public_recipe_pins_source_and_all_public_checkpoint_bytes():
    recipe = mpnn.LIGANDMPNN_PUBLIC_CONTAINER.to_singularity()
    assert mpnn._PUBLIC_MPNN_REVISION in recipe
    assert "baker-laboratory/fused_mpnn" not in recipe
    assert "torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128" in recipe
    for filename, checksum in mpnn._PUBLIC_MPNN_CHECKPOINTS.values():
        assert f"https://files.ipd.uw.edu/pub/ligandmpnn/{filename}" in recipe
        assert checksum in recipe
    assert recipe.count("sha256sum -c -") == 6
    assert "pip check" in recipe and "import run, data_utils, model_utils, sc_utils" in recipe


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


def test_backend_switch_has_distinct_cache_identity_without_invalidating_site_caches(monkeypatch):
    monkeypatch.setattr(get_settings(), "MPNN_WEIGHTS_DIR", None)
    fingerprints = []
    for image in (mpnn._MODELHUB_SIF_CONTAINER, mpnn.MODELHUB_PORTABLE_CONTAINER, mpnn.LIGANDMPNN_PUBLIC_CONTAINER):
        monkeypatch.setattr(mpnn, "MODELHUB_CONTAINER", image)
        # Equalize even the unused packer option: the selected command alone
        # must distinguish the scientific implementation used for sequence output.
        block = mpnn.ClassicMPNNBlock(args={"checkpoint_path_sc": "/same/unused/packer.pt"})
        config = block.model_dump(mode="json")
        assert config["command_template"] == block.command_template
        fingerprints.append(ExecutionEngine._cache_fingerprint([block]))
        assert fingerprints[-1] == ExecutionEngine._cache_fingerprint([
            mpnn.ClassicMPNNBlock(args={"checkpoint_path_sc": "/same/unused/packer.pt"})
        ])
    assert len(set(fingerprints)) == 3


@pytest.mark.parametrize("backend,env_flag", [("apptainer", "--env"), ("docker", "-e")])
@pytest.mark.parametrize("image", [mpnn._MODELHUB_SIF_CONTAINER, mpnn.MODELHUB_PORTABLE_CONTAINER, mpnn.LIGANDMPNN_PUBLIC_CONTAINER])
def test_cpu_device_is_masked_inside_each_container_runtime(monkeypatch, tmp_path, backend, env_flag, image):
    monkeypatch.setattr(mpnn, "MODELHUB_CONTAINER", image)
    monkeypatch.setattr(get_settings(), "MPNN_WEIGHTS_DIR", None)
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


def test_public_sequence_contract_and_original_chain_lineage(tmp_path):
    block = mpnn.ClassicMPNNBlock(num_seq_per_pdb=1, args={"chains_to_design": "X1", "symmetry_residues": "X15,Z1"})
    sandbox = tmp_path / "chunk"
    command = block.create_command(input_handler(tmp_path), sandbox, 0)
    assert option(command, "fasta_seq_separation") == "/"
    assert option(command, "pack_side_chains") == "0"
    assert option(command, "chains_to_design") == "A"
    assert option(command, "symmetry_residues") == "A5,Z1"
    assert option(command, "checkpoint_ligand_mpnn") == mpnn._public_checkpoint("ligand_mpnn")
    assert "/private/weights" not in " ".join(map(str, command))
    ((path, positions),) = json.loads((sandbox / "inputs/pdb_position_fixed_chunk0.json").read_text()).items()
    assert path.endswith("portable_pdb/design.pdb")
    assert positions == "A5"  # public runner calls value.split(), unlike private fork
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
    assert option(command, "checkpoint_path_sc") == mpnn._public_checkpoint("path_sc")
    assert option(command, "pack_with_ligand_context") == "1"
    packed = sandbox / "packed"
    packed.mkdir()
    shutil.copy(sandbox / "inputs/portable_pdb/design.pdb", packed / "design__0_1.pdb")
    row, = block.compile(sandbox)._df.to_dicts()
    assert set(parse_to_atom_array(row["structures"][0]["path"]).chain_id) == {"X1", "Z"}


def test_public_rejects_unavailable_private_model():
    with pytest.raises(ValueError, match="does not support lddt_enhanced"):
        mpnn.ClassicMPNNBlock(lddt_enhanced=True)


def test_public_rejects_ambiguous_residue_bias_before_execution(tmp_path):
    bias = tmp_path / "bias.json"
    bias.write_text('{"X15": {"A": 1.0}}')
    block = mpnn.ClassicMPNNBlock(args={"bias_AA_per_residue": str(bias)})
    with pytest.raises(ValueError, match="single-character input chain IDs"):
        block.create_command(input_handler(tmp_path), tmp_path / "chunk", 0)
