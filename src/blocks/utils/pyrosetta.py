import logging
import os
from pathlib import Path
import sys
from io import StringIO

import biotite.structure.io.pdb as pdb

logger = logging.getLogger(__name__)


def _get_configured_pyrosetta_path() -> str | None:
    configured = os.environ.get("PYROSETTA_PATH") or os.environ.get(
        "PIPELINES_PYROSETTA_PATH"
    )
    try:
        from pipelines import get_setting_field

        configured = configured or get_setting_field("PYROSETTA_PATH", default=None)
    except Exception:
        # Keep PyRosetta usable in standalone contexts where pipelines settings
        # are unavailable; the direct environment-variable fallback still works.
        pass

    if not configured:
        return None

    path = Path(configured).expanduser()
    if not path.is_dir():
        logger.warning("Configured PYROSETTA_PATH does not exist: %s", path)
        return None
    return str(path)


def _ensure_pyrosetta_on_syspath() -> str | None:
    configured_path = _get_configured_pyrosetta_path()
    if configured_path and configured_path not in sys.path:
        sys.path.append(configured_path)
        logger.info("Added PYROSETTA_PATH to sys.path: %s", configured_path)
    return configured_path


def _pyrosetta_install_instructions() -> str:
    return (
        "pyrosetta is not installed. "
        "If you already have PyRosetta, set PYROSETTA_PATH to its 'setup' "
        "directory (for example in .env.local). "
        "Otherwise install the helper with "
        "uv pip install -e '.[pyrosetta]' --override overrides.txt "
        "and then run "
        'python -c "import pyrosetta_installer; '
        'pyrosetta_installer.install_pyrosetta()".'
    )


_ensure_pyrosetta_on_syspath()

try:
    import pyrosetta as pyr

    logger.info("PyRosetta imported.")
    PYROSETTA_AVAILABLE = True
except ImportError:
    PYROSETTA_AVAILABLE = False
    pyr = None


def atom_array_to_pose(atom_array, params_files: list[str] | None = None):
    """Convert a biotite AtomArray to a PyRosetta Pose.

    When ``params_files`` is given, the residue type set is extended with
    those .params files before PDB parsing — required for ligands not in
    Rosetta's database. Uses the lower-level
    ``core.import_pose.pose_from_pdbstring()`` path since the distributed
    helper doesn't accept a custom residue type set.
    """
    if not PYROSETTA_AVAILABLE:
        raise ImportError(_pyrosetta_install_instructions())

    pdb_file = pdb.PDBFile()
    pdb_file.set_structure(atom_array)

    buf = StringIO()
    pdb_file.write(buf)
    pdb_string = buf.getvalue()
    buf.close()

    if params_files:
        pose = pyr.Pose()

        rts = pose.conformation().modifiable_residue_type_set_for_conf(
            pyr.rosetta.core.chemical.FULL_ATOM_t
        )
        rts.read_files_for_base_residue_types(pyr.Vector1(list(params_files)))
        pose.conformation().reset_residue_type_set_for_conf(rts)

        pyr.rosetta.core.import_pose.pose_from_pdbstring(
            pose,
            pdb_string,
            pose.residue_type_set_for_pose(),
            "temp.pdb",
        )

        return pose

    # Low-level path (same as the params branch / ppi_ddg): fills an existing Pose
    # directly. Avoids pyrosetta.distributed's PackedPose, which pickles the pose and
    # therefore requires a `--serialization` pyrosetta build (not present on many
    # standard installs, e.g. /net/software/pyrosetta3.12).
    pose = pyr.Pose()
    pyr.io.pose_from_pdbstring(
        pose, pdb_string, pose.residue_type_set_for_pose(), "temp.pdb"
    )
    return pose


_PYROSETTA_INITIALIZED = False


def init_pyrosetta():
    global _PYROSETTA_INITIALIZED
    if not PYROSETTA_AVAILABLE:
        raise ImportError(_pyrosetta_install_instructions())
    if _PYROSETTA_INITIALIZED:
        return True
    pyr_init_opts = [
        "-run:preserve_header",
        "-use_input_sc",
        "-fast_restyping",
        "-ignore_waters true",
        "-beta",
        "-corrections::beta_nov16",
        "-skip_connect_info",
        "-mute all",
    ]

    pyr.init(" ".join(pyr_init_opts))
    _PYROSETTA_INITIALIZED = True
    return True


# --------------------------------------------------------------------------- #
# ddG helpers — ports from ddg_soft_sm.py
# --------------------------------------------------------------------------- #


def move_chain_far_away(pose, chain: str = "A", dist: float = 10000.0):
    """Return a clone of ``pose`` with ``chain`` translated ``dist`` Å along +x.

    Port of ddg_soft_sm.py:569-579. Used to compute the unbound score in
    ``calc_ddg_norepack`` without deleting residues.
    """
    ros = pyr.rosetta
    pose = pose.clone()
    sel = ros.core.select.residue_selector.ChainSelector(chain)
    subset = sel.apply(pose)
    x_unit = ros.numeric.xyzVector_double_t(1, 0, 0)
    far = ros.numeric.xyzVector_double_t(dist, 0, 0)
    ros.protocols.toolbox.pose_manipulation.rigid_body_move(
        x_unit, 0, far, pose, subset
    )
    return pose


def calc_ddg_norepack(pose, scorefxn, move_chain: str = "A") -> float:
    """score(bound) − score(unbound), no repacking.

    Port of ddg_soft_sm.py:603-610. The ``move_chain`` is translated far
    away to break all inter-chain contacts; the difference in total score
    is the binding energy.
    """
    pose = pose.clone()
    close_score = scorefxn(pose)
    far_pose = move_chain_far_away(pose, chain=move_chain)
    far_score = scorefxn(far_pose)
    return close_score - far_score


def beta_pack(pose, lock_chain: str = "B", soft: bool = False) -> None:
    """Repack sidechains in-place with beta_nov16 (or _soft).

    Minimal port of RosettaPacker.beta_pack (ddg_soft_sm.py:484-499).
    Locks ``lock_chain`` entirely; repacks every other residue within
    10 Å of it; leaves the rest of the protein alone. Used to relax the
    binder interface around a fixed ligand before scoring.
    """
    ros = pyr.rosetta

    lock_sel = ros.core.select.residue_selector.ChainSelector(lock_chain)
    # interface = residues within 10 Å of lock_chain, excluding lock_chain itself
    interface_sel = ros.core.select.residue_selector.NeighborhoodResidueSelector(
        lock_sel, 10.0, False
    )
    # prevent repack on: lock_chain + everything NOT at the interface
    not_interface = ros.core.select.residue_selector.NotResidueSelector(interface_sel)
    prevent_sel = ros.core.select.residue_selector.OrResidueSelector(
        lock_sel, not_interface
    )

    tf = ros.core.pack.task.TaskFactory()
    tf.push_back(ros.core.pack.task.operation.RestrictToRepacking())
    tf.push_back(
        ros.core.pack.task.operation.OperateOnResidueSubset(
            ros.core.pack.task.operation.PreventRepackingRLT(), prevent_sel, False
        )
    )
    tf.push_back(ros.core.pack.task.operation.IncludeCurrent())

    if soft:
        sfxn = ros.core.scoring.ScoreFunctionFactory.create_score_function(
            "beta_nov16_soft"
        )
    else:
        sfxn = pyr.get_fa_scorefxn()

    packer = ros.protocols.minimization_packing.PackRotamersMover()
    packer.score_function(sfxn)
    packer.task_factory(tf)
    packer.apply(pose)
