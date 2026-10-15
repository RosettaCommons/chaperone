"""Discover user-supplied Rosetta ligand params via CHAPERONE_LIGAND_PARAMS_DIR.

No ligand params or automatic generation protocol is bundled. Pass a directory
to the lookup helpers or set CHAPERONE_LIGAND_PARAMS_DIR before importing them.
"""

import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)


def _get_ligand_params_dir() -> Path | None:
    from chaperone import get_setting_field

    d = get_setting_field("LIGAND_PARAMS_DIR", default="")
    return Path(d) if d else None


LIGAND_PARAMS_DIR = _get_ligand_params_dir()


@lru_cache(maxsize=1)
def _scan_params_dir(params_dir: str | None = None) -> dict[str, Path]:
    """Scan the params directory and return {ligand_name: path} mapping.

    The ligand name is derived from the filename stem, e.g.
    ``FAD.params`` -> ``"FAD"``, ``Boltzdesign_FAD.params`` -> ``"Boltzdesign_FAD"``.
    """
    d = Path(params_dir) if params_dir else LIGAND_PARAMS_DIR
    registry: dict[str, Path] = {}
    if d is None:
        return registry
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
            "Set CHAPERONE_LIGAND_PARAMS_DIR to your params directory or pass "
            "override_params_file= to the ligand_ddg block."
        )
    return str(registry[ligand_name])
