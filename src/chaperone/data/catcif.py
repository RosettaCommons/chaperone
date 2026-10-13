"""Catcif (concatenated-CIF archive) storage backend for ``PipelineDataHandler``.

A catcif archive is a single file holding many CIF entries. Each entry is
addressed as ``archive.catcif:<entry_id>`` — that string is a valid path for
:py:func:`chaperone.io.parse_to_atom_array` and for ``get_data`` workers,
since ``catcif_tools`` resolves it transparently.

The archive is readable outside of pipelines and uses one self-describing
``archive.catcif:<entry_id>`` string per structure.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from chaperone.data.handler import PipelineDataHandler

from chaperone.data.schema import Column, StorageFormat, StructureColumn

logger = logging.getLogger(__name__)


CATCIF_SUFFIX = ".catcif"
CATCIF_TAG_SEP = ":"


def is_catcif_path(path: Optional[str]) -> bool:
    """Return True if ``path`` is an ``archive.catcif:entry_id`` reference."""
    if not path:
        return False
    sep_idx = path.find(CATCIF_TAG_SEP)
    if sep_idx <= 0:
        return False
    return path[:sep_idx].endswith(CATCIF_SUFFIX)


def split_catcif_path(path: str) -> tuple[str, str]:
    """Split ``archive.catcif:entry_id`` into ``(archive_path, entry_id)``."""
    sep_idx = path.find(CATCIF_TAG_SEP)
    if sep_idx <= 0 or not path[:sep_idx].endswith(CATCIF_SUFFIX):
        raise ValueError(f"Not a catcif path: {path!r}")
    return path[:sep_idx], path[sep_idx + 1 :]


def list_catcif_entries(archive_path: str | Path) -> list[str]:
    """List all entry ids in a catcif archive."""
    import catcif_tools

    return list(catcif_tools.get_tags(str(archive_path)))


def read_catcif_cif_bytes(path: str) -> bytes:
    """Read an ``archive.catcif:entry_id`` tag and return its CIF text as bytes."""
    import catcif_tools

    cif_str = catcif_tools.get_structure(path)
    return cif_str.encode("utf-8")


def _canonicalize_to_cif(path: str) -> Optional[str]:
    """Read a structure file (any supported format) and return CIF text.

    Returns None if the file is missing or unreadable.
    """
    if is_catcif_path(path):
        import catcif_tools

        try:
            return catcif_tools.get_structure(path)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to read catcif entry %s: %s", path, e)
            return None

    p = Path(path)
    if not p.exists():
        logger.warning("Structure file not found, skipping: %s", path)
        return None

    lower = path.lower()
    if lower.endswith(".cif") or lower.endswith(".cif.gz") or lower.endswith(".mmcif"):
        try:
            if lower.endswith(".cif.gz"):
                import gzip

                with gzip.open(path, "rt") as f:
                    return f.read()
            return p.read_text()
        except Exception as e:  # noqa: BLE001
            logger.warning("Direct read failed for %s: %s — falling back", path, e)

    try:
        from atomworks.io.utils.io_utils import to_cif_file

        from chaperone.io import ensure_bonds, parse_to_atom_array

        atom_array = ensure_bonds(parse_to_atom_array(path))
        with tempfile.NamedTemporaryFile(suffix=".cif", delete=True) as tmp:
            to_cif_file(atom_array, tmp.name)
            return Path(tmp.name).read_text()
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to canonicalize %s: %s", path, e)
        return None


def _unique_entry_id(stem: str, counts: dict[str, int]) -> str:
    """Return a unique entry id for a given stem, tracking counts in-place."""
    if stem not in counts:
        counts[stem] = 0
        return stem
    counts[stem] += 1
    return f"{stem}_{counts[stem]}"


def write_handler_step_to_catcif(
    handler: "PipelineDataHandler",
    step: str,
    archive_path: str | Path,
    *,
    cleanup: bool = False,
    compress: bool = False,
) -> "PipelineDataHandler":
    """Consolidate all structure files for ``step`` into a single catcif archive.

    The handler's structure paths for ``step`` are rewritten in-place to
    ``archive_path:entry_id`` tags. Row count is preserved exactly; the only
    mutation is to the structure dicts matching ``step``.

    Args:
        handler: Input handler. Mutated in place (also returned for chaining).
        step: Structure name to consolidate. Structures with other names are
            left untouched.
        archive_path: Output ``.catcif`` file. Parent directories are created.
        cleanup: If True, delete the original files after successful archival.
            Default False — original files are left in place.
        compress: If True, compress individual entries inside the archive.

    Returns:
        The same handler, with rewritten paths for ``step``.

    Raises:
        ValueError: If the handler is empty or has no structures for ``step``.
    """
    import catcif_tools

    from chaperone.data.handler import PipelineDataHandler

    if len(handler) == 0:
        raise ValueError("No outputs to store: handler is empty.")

    archive_path = Path(archive_path)
    if archive_path.suffix != CATCIF_SUFFIX:
        archive_path = archive_path.with_suffix(CATCIF_SUFFIX)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_str = str(archive_path)

    # Reuse entry IDs from upstream archives to preserve lineage.
    path_to_entry: dict[str, str] = {}
    counts: dict[str, int] = {}
    for row in handler._df.iter_rows(named=True):
        for struct in row[Column.STRUCTURES]:
            if struct[StructureColumn.NAME] != step:
                continue
            orig = struct[StructureColumn.PATH]
            if not orig or orig in path_to_entry:
                continue
            if is_catcif_path(orig):
                src_archive, entry_id = split_catcif_path(orig)
                if str(Path(src_archive).resolve()) == str(archive_path.resolve()):
                    continue
                path_to_entry[orig] = _unique_entry_id(entry_id, counts)
                continue
            stem = Path(orig).stem
            if orig.lower().endswith(".cif.gz"):
                stem = Path(stem).stem
            path_to_entry[orig] = _unique_entry_id(stem, counts)

    if not path_to_entry:
        raise ValueError(
            f"No structure paths found for step '{step}'. "
            f"Available steps: {handler.names}"
        )

    written: dict[str, str] = {}  # orig_path -> catcif tag
    for orig, entry_id in path_to_entry.items():
        cif_str = _canonicalize_to_cif(orig)
        if cif_str is None:
            continue
        catcif_tools.append_to_catcif_file(
            archive_str,
            cif_str,
            entry_id,
            scores=None,
            compress=compress,
        )
        written[orig] = f"{archive_str}{CATCIF_TAG_SEP}{entry_id}"

    if not written:
        raise ValueError(
            f"No structures could be read into catcif archive {archive_str}. "
            f"All {len(path_to_entry)} source files were missing or unreadable."
        )

    new_rows = []
    for row in handler._df.iter_rows(named=True):
        new_structures = []
        for struct in row[Column.STRUCTURES]:
            if struct[StructureColumn.NAME] == step:
                orig = struct[StructureColumn.PATH]
                new_path = written.get(orig)
                if new_path is not None:
                    struct = dict(struct)
                    struct[StructureColumn.PATH] = new_path
                    struct[StructureColumn.STORE_PATH] = archive_str
                    struct[StructureColumn.STORAGE_FORMAT] = StorageFormat.CATCIF
            new_structures.append(struct)
        new_rows.append(
            {
                Column.STRUCTURES: new_structures,
                Column.METRICS: row[Column.METRICS],
            }
        )
    handler._df = PipelineDataHandler(new_rows)._df

    if cleanup:
        for orig in written:
            try:
                Path(orig).unlink()
            except OSError as e:
                logger.warning("Failed to remove %s: %s", orig, e)

    logger.info(
        "Wrote %d/%d structures for step '%s' -> %s",
        len(written),
        len(path_to_entry),
        step,
        archive_str,
    )
    return handler


def extract_catcif_to_files(
    archive_path: str | Path,
    output_dir: str | Path,
) -> dict[str, str]:
    """Extract every entry in a catcif archive to ``output_dir/<entry_id>.cif``.

    Returns a mapping ``{catcif_tag: written_file_path}`` so callers can
    rewrite ``StructureColumn.PATH`` values.
    """
    import catcif_tools

    archive_path = Path(archive_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_str = str(archive_path)

    mapping: dict[str, str] = {}
    for entry_id in catcif_tools.get_tags(archive_str):
        tag = f"{archive_str}{CATCIF_TAG_SEP}{entry_id}"
        cif_str = catcif_tools.get_structure(tag)
        out_path = output_dir / f"{entry_id}.cif"
        out_path.write_text(cif_str)
        mapping[tag] = str(out_path)
    return mapping
