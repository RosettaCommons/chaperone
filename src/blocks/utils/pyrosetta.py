import logging
import os
from pathlib import Path
import sys
from io import StringIO

import biotite.structure.io.pdb as pdb

logger = logging.getLogger(__name__)


def _get_configured_pyrosetta_path() -> str | None:
    configured = os.environ.get("PYROSETTA_PATH") or os.environ.get(
        "CHAPERONE_PYROSETTA_PATH"
    )
    try:
        from chaperone import get_setting_field

        configured = configured or get_setting_field("PYROSETTA_PATH", default=None)
    except Exception:
        # Keep PyRosetta usable in standalone contexts where Chaperone settings
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
        "uv pip install -e '.[pyrosetta]' "
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

    # Low-level path (same as the params branch): fills an existing Pose
    # directly. Avoids pyrosetta.distributed's PackedPose, which pickles the pose and
    # therefore requires a `--serialization` pyrosetta build (not present on many
    # standard installs).
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
