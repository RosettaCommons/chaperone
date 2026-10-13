"""Rosetta ligand params file discovery, resolution, and generation.

Dynamically scans a params directory for ``*.params`` files and provides
lookup by ligand 3-letter code.  The directory is resolved in order:

1. ``PIPELINES_LIGAND_PARAMS_DIR`` environment variable
2. Bundled ``blocks/utils/params/`` directory (ships with the repo)

Usage::

    from blocks.utils.params import resolve_param_file, list_available_ligands

    path = resolve_param_file("FAD")          # -> "/…/params/FAD.params"
    all_ligands = list_available_ligands()     # -> ["FAD", ...]

For dynamic generation from an AtomArray (protonates if needed, runs
molfile_to_params)::

    params_path, protonated_ligand = generate_params(ligand_array, cache_dir)
"""

import logging
import os
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_BUNDLED_PARAMS_DIR = Path(__file__).parent / "params"


def _get_ligand_params_dir() -> Path:
    from chaperone import get_setting_field

    # default="" suppresses the raise when LIGAND_PARAMS_DIR is None in
    # settings — we fall through to the bundled dir instead.
    d = get_setting_field("LIGAND_PARAMS_DIR", default="")
    return Path(d) if d else _BUNDLED_PARAMS_DIR


LIGAND_PARAMS_DIR = _get_ligand_params_dir()


@lru_cache(maxsize=1)
def _scan_params_dir(params_dir: str | None = None) -> dict[str, Path]:
    """Scan the params directory and return {ligand_name: path} mapping.

    The ligand name is derived from the filename stem, e.g.
    ``FAD.params`` -> ``"FAD"``, ``Boltzdesign_FAD.params`` -> ``"Boltzdesign_FAD"``.
    """
    d = Path(params_dir) if params_dir else LIGAND_PARAMS_DIR
    registry: dict[str, Path] = {}
    if not d.is_dir():
        logger.warning("Params directory does not exist: %s", d)
        return registry
    for p in sorted(d.glob("*.params")):
        registry[p.stem] = p
    logger.debug("Discovered %d params files in %s", len(registry), d)
    return registry


def list_available_ligands(params_dir: str | None = None) -> list[str]:
    """Return sorted list of available ligand names."""
    return sorted(_scan_params_dir(params_dir).keys())


def resolve_param_file(ligand_name: str, params_dir: str | None = None) -> str:
    """Resolve a ligand name to its ``.params`` file path.

    Parameters
    ----------
    ligand_name : str
        Ligand 3-letter code (e.g. ``"FAD"``) or prefixed variant
    params_dir : str, optional
        Override the params directory to scan.

    Returns
    -------
    str
        Absolute path to the ``.params`` file.

    Raises
    ------
    KeyError
        If the ligand name is not found in the scanned directory.
    """
    registry = _scan_params_dir(params_dir)
    if ligand_name not in registry:
        available = ", ".join(sorted(registry.keys())) or "(none)"
        raise KeyError(
            f"No params file found for ligand '{ligand_name}'. "
            f"Available: {available}. "
            f"Either add a .params file to {LIGAND_PARAMS_DIR} or pass "
            f"param_file= directly."
        )
    return str(registry[ligand_name])


# --------------------------------------------------------------------------- #
# Dynamic params generation (port of ddg_soft_sm.py:atom_array_to_params_file)
# --------------------------------------------------------------------------- #


def _patch_sdf_names(atom_names, sdf_path: Path) -> None:
    """Overwrite atom-name columns in an SDF written by RDKit.

    Direct port of ddg_soft_sm.py:246-271. RDKit's SDWriter truncates the
    atom-alias field to 3 chars; this patches columns 30:35 of each atom
    record with the full (≤5-char) name so molfile_to_params --keep-names
    sees the real names.
    """
    sdf_lines = sdf_path.read_text().splitlines(keepends=True)
    atom_start = None
    for i, line in enumerate(sdf_lines):
        if atom_start is None:
            if "V2000" in line:
                atom_start = i + 1
            continue
        iatom = i - atom_start
        if iatom < len(atom_names):
            sdf_lines[i] = line[:30] + "%5s" % (atom_names[iatom]) + line[35:]
    sdf_path.write_text("".join(sdf_lines))


def generate_params(
    ligand_array,
    cache_dir,
    ph: float = 7.4,
    ref_smiles: str | None = None,
    force: bool = False,
):
    """Generate a Rosetta .params file for a ligand, with filesystem caching.

    If the ligand has no hydrogens, it is protonated first via Dimorphite-DL
    (pH-aware). The protonated array is returned alongside the params path so
    the caller can rebuild the complex with the correct H before pose
    construction.

    Caching is filesystem-based (keyed by 3-letter res_name) since chunks run
    as separate processes and share no module state. Generation happens in a
    per-PID temp subdir, then the params file is atomically moved into place
    with os.replace — if two chunks race, the second overwrite is a harmless
    identical-bytes replacement.

    Parameters
    ----------
    ligand_array : AtomArray
        Single-residue ligand. May or may not have hydrogens.
    cache_dir : str | Path
        Directory for cached .params files. Shared across chunks.
    ph : float
        Target pH for protonation (only used if ligand lacks H).
    ref_smiles : str, optional
        Reference SMILES for geometric bond-graph rebuild when the input
        ligand's bonds are disconnected (CCD-name mismatch in diffusion
        outputs). Falls back to CCD lookup on res_name. See
        ``blocks.utils.ligand.protonate_ligand``.
    force : bool
        Regenerate even if a cached params file exists.

    Returns
    -------
    (str, AtomArray)
        Path to .params file, and the (possibly protonated) ligand array.
    """
    import numpy as np
    from atomworks.io.tools.rdkit import atom_array_to_rdkit
    from rdkit import Chem

    from chaperone import get_setting_field

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    name3 = str(ligand_array.res_name[0])[:3]
    params_path = cache_dir / f"{name3}.params"

    # Protonate if needed — even on cache hit, the caller needs the H-bearing
    # array to build a correct pose.
    if (np.asarray(ligand_array.element) == "H").sum() == 0:
        from blocks.utils.ligand import protonate_ligand

        ligand_array = protonate_ligand(ligand_array, ph=ph, ref_smiles=ref_smiles)

    # Normalize the ligand res_name to the 3-char code used for the params
    # residue type. Diffusion/cofold outputs carry names like "LIG_B" (>3
    # chars), which both break biotite's PDB writer (residue names must be
    # ≤3 chars) and fail to match the generated params residue ("LIG"). Doing
    # this here keeps the returned array, the PDB the pose is built from, and
    # the params file all in agreement.
    if str(ligand_array.res_name[0]) != name3:
        ligand_array = ligand_array.copy()
        ligand_array.res_name[:] = name3

    if params_path.exists() and not force:
        logger.debug("generate_params: cache hit %s", params_path)
        return str(params_path), ligand_array

    # --- cache miss: generate into a per-PID temp subdir -------------------
    tmp_dir = cache_dir / f".{name3}_{os.getpid()}_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        sdf_path = tmp_dir / f"{name3}.sdf"

        # AtomArray → RDKit → SDF
        mol = atom_array_to_rdkit(ligand_array, hydrogen_policy="keep")
        writer = Chem.SDWriter(str(sdf_path))
        writer.write(mol)
        writer.close()
        _patch_sdf_names(mol._annotations["atom_name"], sdf_path)

        # subprocess molfile_to_params
        script = get_setting_field("MOLFILE_TO_PARAMS_SCRIPT")
        if not script:
            raise FileNotFoundError(
                "MOLFILE_TO_PARAMS_SCRIPT is not set. Set it to Rosetta's "
                "main/source/scripts/python/public/molfile_to_params.py."
            )
        script = Path(script)
        # A relative value is repo-relative, not CWD-relative: SLURM workers and
        # sandboxed chunk dirs run from elsewhere.
        if not script.is_absolute():
            repo_root = Path(__file__).resolve().parents[3]
            script = repo_root / script
        if not script.exists():
            raise FileNotFoundError(
                f"MOLFILE_TO_PARAMS_SCRIPT does not exist: {script}. Set it to "
                "Rosetta's main/source/scripts/python/public/molfile_to_params.py."
            )

        # cwd = script's parent dir — molfile_to_params imports rosetta_py
        # which lives one level up; cwd=public/ puts python/ on sys.path[0]'s
        # parent via the script's own sys.path insertion.
        cmd = [
            sys.executable,
            str(script),
            "--name",
            name3,
            "-p",
            str(tmp_dir / name3),
            str(sdf_path),
            "--no-pdb",
            "--clobber",
            "--keep-names",
        ]
        proc = subprocess.run(
            cmd,
            cwd=script.parent,
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"molfile_to_params failed for {name3} "
                f"(exit {proc.returncode}):\n{proc.stderr}"
            )

        tmp_params = tmp_dir / f"{name3}.params"
        if not tmp_params.exists():
            raise RuntimeError(
                f"molfile_to_params did not produce {tmp_params}. "
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )

        # atomic publish
        os.replace(tmp_params, params_path)
        logger.info("Generated params for %s → %s", name3, params_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return str(params_path), ligand_array
