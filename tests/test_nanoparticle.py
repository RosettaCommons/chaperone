"""Geometry, motif, and score provenance contracts without external models."""

import json

import numpy as np
import pytest
from biotite.structure import Atom, array, concatenate
from scipy.spatial.transform import Rotation

from blocks.networks.rpxdock import RPXDockBlock
from blocks.scripts.rpxdock_run import c3_frames, ranked_models
from blocks.util_blocks.nanoparticle import prepare_c3, NanoparticlePrepareBlock
from chaperone import PipelineDataHandler
from chaperone.data import make_row
from chaperone.io import write_pdb


def trimer():
    # Three non-collinear motif residues and two designed residues per protomer.
    xyz = np.array([[9, 0, -2], [10, 1, 0], [9, -1, 2], [12, 0, 3], [11, 2, 4]])
    subunits = []
    for i, chain in enumerate("ABC"):
        coords = Rotation.from_euler("z", 120 * i, degrees=True).apply(xyz)
        subunits.append(
            array(
                [
                    Atom(
                        p,
                        chain_id=chain,
                        res_id=10 + j,
                        res_name="GLY" if j == 0 else "ALA",
                        atom_name="CA",
                        element="C",
                    )
                    for j, p in enumerate(coords)
                ]
            )
        )
    return concatenate(subunits)


def references(atoms):
    motif = atoms[atoms.res_id < 13]
    # One stalk CA overlaps a newly designed residue; exclude scaffold motif contacts.
    stalk = atoms[(atoms.chain_id == "A") & np.isin(atoms.res_id, [10, 14])]
    mapping = {f"{c}{r}": f"{c}{r}" for c, r in zip(motif.chain_id, motif.res_id)}
    return motif, stalk, mapping


def test_c3_preparation_is_rigid_transform_invariant_and_retains_radial_offset():
    atoms = trimer()
    motif, stalk, mapping = references(atoms)
    expected, fixed, metrics = prepare_c3(atoms, motif, stalk, mapping)
    moved = atoms.copy()
    moved.coord = Rotation.from_euler("xyz", [23, 61, -15], degrees=True).apply(
        atoms.coord
    ) + [100, -20, 40]
    actual, moved_fixed, moved_metrics = prepare_c3(moved, motif, stalk, mapping)
    np.testing.assert_allclose(actual.coord, expected.coord, atol=1e-5)
    assert np.linalg.norm(actual.coord.mean(axis=0)[:2]) > 9
    assert metrics["n_clashes"] == moved_metrics["n_clashes"] == 1
    assert fixed == moved_fixed == ["A1", "A2", "A3"]
    assert actual.res_name[0] == "GLY"
    assert actual.res_id.tolist() == [1, 2, 3, 4, 5]
    assert moved_metrics["c3_ca_rmsd"] < 1e-4


def test_asymmetric_input_and_missing_motif_fail():
    atoms = trimer()
    motif, stalk, mapping = references(atoms)
    with pytest.raises(ValueError, match="correspondence"):
        prepare_c3(atoms, motif, stalk, {})
    atoms.coord[-1] += [30, 0, 0]
    with pytest.raises(ValueError, match="not C3"):
        prepare_c3(atoms, motif, stalk, mapping)


def test_preparation_preserves_lineage_and_filters_clashes(tmp_path):
    atoms = trimer()
    motif, stalk, mapping = references(atoms)
    paths = [tmp_path / f"{name}.pdb" for name in ["input", "motif", "stalk"]]
    for path, value in zip(paths, [atoms, motif, stalk]):
        write_pdb(value, path)
    handler = PipelineDataHandler(
        [
            make_row(
                "rfd1",
                paths[0],
                design_id="parent",
                metadata={"diffused_index_map": json.dumps(mapping)},
            )
        ]
    )
    block = NanoparticlePrepareBlock(
        motif_pdb=str(paths[1]), clash_ref_pdb=str(paths[2]), max_clashes=1
    )
    result = block(handler, tmp_path / "pass")
    assert len(result) == 1
    assert result.names == ["rfd1", "nanoparticle_prepare"]
    item = next(result.get_data(which="last", load=True))
    assert item.metadata["motif"] == ["A1", "A2", "A3"]
    assert result.unpack()["parent_design_id"][0] == "parent"
    result.write_parquet(tmp_path / "prepared.parquet")
    assert len(PipelineDataHandler(tmp_path / "prepared.parquet")) == 1
    block.max_clashes = 0
    assert len(block(handler, tmp_path / "filtered")) == 0


def test_model_ranking_uses_model_index_not_rank():
    assert ranked_models([10, 90, 80], 2) == [1, 2]
    assert ranked_models([90, 90, 10], 2) == [0, 1]
    for invalid in ([], [1, np.nan], [np.inf], [[1, 2]]):
        with pytest.raises(ValueError):
            ranked_models(invalid, 1)


def test_frames_form_c3_subgroup_about_docking_axis():
    axis = np.array([1.0, 2.0, 3.0])
    axis /= np.linalg.norm(axis)
    frames = c3_frames(axis)
    np.testing.assert_allclose(frames[1] @ frames[1], frames[2], atol=1e-12)
    np.testing.assert_allclose(frames[1] @ frames[2], np.eye(4), atol=1e-12)
    for frame in frames:
        np.testing.assert_allclose(frame[:3, :3] @ axis, axis)


def scored_outputs(tmp_path, atoms=None, score=90):
    atoms = trimer() if atoms is None else atoms
    sources = []
    for index in range(2):
        directory = tmp_path / f"dock_{index}"
        directory.mkdir()
        path = directory / "trimer_000.pdb"
        write_pdb(atoms, path)
        (directory / "poses.json").write_text(
            json.dumps([{"path": str(path), "score": score, "model_index": 7}])
        )
        sources.append(
            {
                "path": f"/input/parent{index}.pdb",
                "design_id": f"parent{index}",
                "output": str(directory),
                "motif_ordinals": [0, 1],
                "residue_count": 5,
            }
        )
    (tmp_path / "inputs.json").write_text(json.dumps(sources))


def test_docked_outputs_have_unique_identity_and_all_chain_motifs(tmp_path):
    scored_outputs(tmp_path)
    result = RPXDockBlock().compile(tmp_path)
    assert len(result) == result.unpack()["design_id"].nunique() == 2
    assert result.unpack()["parent_design_id"].to_list() == ["parent0", "parent1"]
    assert next(result.get_data(which="last")).metadata["motif"] == [
        "A10",
        "A11",
        "B10",
        "B11",
        "C10",
        "C11",
    ]
    assert result.unpack()["rpxdock.rpx_score"].to_list() == [90, 90]
    assert result.unpack()["rpxdock.model_index"].to_list() == [7, 7]
    result.write_parquet(tmp_path / "docked.parquet")
    assert len(PipelineDataHandler(tmp_path / "docked.parquet")) == 2


@pytest.mark.parametrize("kind", ["monomer", "score", "numbering"])
def test_invalid_docked_outputs_fail_closed(tmp_path, kind):
    atoms = trimer()
    if kind == "monomer":
        atoms = atoms[atoms.chain_id == "A"]
    if kind == "numbering":
        atoms.res_id[atoms.chain_id == "B"] += 20
    scored_outputs(tmp_path, atoms, float("nan") if kind == "score" else 90)
    with pytest.raises(ValueError):
        RPXDockBlock().compile(tmp_path)


def test_single_pose_upstream_patch_preserves_dimension_and_is_guarded(tmp_path):
    from blocks.networks.rpxdock import _RPX_COMPATIBILITY_PATCH, RPXDOCK_CONTAINER

    path = tmp_path / "rpxdock/search/onecomp.py"
    path.parent.mkdir(parents=True)
    path.write_text(
        "disp = np.sum(xforms[:, :3, 3] * spec.axis[None, :3], axis=-1).squeeze()\n"
    )
    (path.parent / "result.py").write_text(
        "if symframes is None: pass\nelif symframes == 'primary': pass\nelse: assert symframes.shape == (3, 4, 4)\n"
    )
    patch = _RPX_COMPATIBILITY_PATCH.replace("/opt/rpxdock", str(tmp_path))
    exec(compile(patch, "single_pose_patch", "exec"), {})
    exec((path.parent / "result.py").read_text(), {"symframes": c3_frames([0, 0, 1])})
    from types import SimpleNamespace

    for count in (1, 3):
        scope = {
            "np": np,
            "xforms": np.tile(np.eye(4), (count, 1, 1)),
            "spec": SimpleNamespace(axis=np.array([0.0, 0.0, 1.0, 0.0])),
        }
        exec(path.read_text(), scope)
        assert scope["disp"].shape == (count,)
    with pytest.raises(RuntimeError, match="Unexpected RPXdock"):
        exec(compile(patch, "single_pose_patch", "exec"), {})
    recipe = RPXDOCK_CONTAINER.to_dockerfile()
    assert "numpy==1.26.4" in recipe
    assert "Unexpected RPXdock" in recipe
    assert "PYTHONPATH=/opt/rpxdock:/opt/willutil" in recipe


def test_rpx_command_preserves_empty_residue_exclusion(tmp_path):
    import shlex

    atoms = trimer()
    monomer = atoms[atoms.chain_id == "A"]
    path = tmp_path / "monomer.pdb"
    write_pdb(monomer, path)
    handler = PipelineDataHandler(
        [make_row("prepare", path, metadata={"motif": ["A10", "A11", "A12"]})]
    )
    cache = tmp_path / "cache"
    block = RPXDockBlock(ignored_aas="", cache_dir=str(cache))
    assert not cache.exists()
    assert f"{cache}:{cache}" in block.command_template
    assert f"HOME={cache}" in block.command_template
    assert f"TMPDIR={cache / 'tmp'}" in block.command_template
    command = block.create_command(handler, tmp_path / "sandbox", 0)
    assert (cache / "tmp").is_dir()
    argv = shlex.split(command[2])
    assert argv[argv.index("--ignored-aas") + 1] == ""
    staged = json.loads((tmp_path / "sandbox/inputs.json").read_text())
    assert staged[0]["motif_ordinals"] == [0, 1, 2]
