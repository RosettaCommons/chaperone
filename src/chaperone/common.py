"""Common utility functions for Chaperone."""

import glob
import logging
import os
import platform
from functools import lru_cache
from os import PathLike
from pathlib import Path

import sys
import tqdm
import rootutils
from rich.console import Console

logger = logging.getLogger(__name__)


def mkdir_p(path: str | PathLike) -> Path:
    """``Path.mkdir(parents=True, exist_ok=True)`` that survives NFS attribute caching.

    ``exist_ok`` is not atomic. After ``os.mkdir`` raises EEXIST, pathlib re-checks
    with ``is_dir()``; on NFS with long directory-attribute caching that stat can
    return a stale negative, so the FileExistsError escapes. Concurrent chunk jobs
    racing to create a shared step directory then abort the whole run.
    """
    p = Path(path)
    try:
        p.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        pass
    return p


def is_macos() -> bool:
    """True if running on a macOS host (Intel or Apple Silicon).

    Used by container/runtime code that needs to relax Linux-only
    assumptions (e.g. skipping ``--gpus all`` on Docker, filtering
    out bind mounts whose source paths don't exist on dev laptops).
    """
    return platform.system() == "Darwin"


def is_apple_silicon() -> bool:
    """True if running on an ARM64 macOS host (Apple Silicon).

    Used by container/runtime selection code that needs to fall back
    from CUDA-only images on dev boxes that can't build them locally.
    """
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def get_console() -> Console:
    """Get a rich Console configured based on settings.

    If CHAPERONE_DISABLE_RICH env var is set, returns a Console with
    colors/markup disabled.
    """
    # Check env var directly to avoid circular imports at module load time
    disable_rich = os.environ.get("CHAPERONE_DISABLE_RICH", "").lower() in (
        "1",
        "true",
        "yes",
    )
    if disable_rich:
        return Console(force_terminal=False, no_color=True, markup=False)
    return Console()


@lru_cache(maxsize=1)
def find_project_root() -> Path:
    """Find the project root directory.

    Searches for marker files (pyproject.toml, .git) starting from this file's
    location and walking up. Falls back to dotenv if available.

    Returns:
        Path to the project root directory.
    """
    return rootutils.setup_root(
        __file__,
        indicator=[".git", "pyproject.toml"],
        dotenv=True,
        pythonpath=True,
        cwd=False,
    )


def progress_disabled() -> bool:
    """True when a progress bar would be written to a file rather than a terminal.

    tqdm redraws by emitting the whole bar again, so a redirected bar becomes
    one line per update. A measured chunk log was 421 of 426 lines of progress
    bar — reading its tail returned pages of "836/840", "838/840" and nothing
    useful. That matters beyond tidiness here: an agent tailing a log pays for
    every one of those lines in context.
    """
    if os.environ.get("CHAPERONE_PROGRESS") == "always":
        return False
    return not sys.stderr.isatty()


def iterrows(df, **kwargs):
    kwargs.setdefault("disable", progress_disabled())
    try:
        return tqdm.tqdm(
            df.iter_rows(named=True),
            colour="#4FB9AF",  # paper teal
            total=len(df),
            leave=False,
            **kwargs,
        )
    except AttributeError:
        return tqdm.tqdm(
            df.iterrows(),
            colour="#4FB9AF",  # paper teal
            total=len(df),
            leave=False,
            **kwargs,
        )


def setup_logging():
    import warnings
    import logging as _logging
    from absl import logging as absl_logging

    find_project_root()
    warnings.simplefilter("ignore")

    # Remove verbose atomworks errors
    _logging.getLogger("atomworks.constants").setLevel(_logging.ERROR)
    _logging.getLogger("transforms").setLevel(_logging.WARNING)

    # Suppress noisy INFO messages from chaperone.data.handler
    _logging.getLogger("chaperone.data.handler").setLevel(_logging.WARNING)

    class _SilenceAtomworksFilter:
        def filter(self, record):
            return "atomworks" not in (record.pathname or "")

    # Filter out noisy atomworks.io warnings about formal charges and missing fields
    class _SilenceAtomworksIOWarnings(_logging.Filter):
        _suppressed_messages = [
            "We can't fix formal charges without building from templates",
            "Field charge not found in file",
            "Failed to build `_build_entity_poly`",
        ]

        def filter(self, record):
            msg = record.getMessage()
            return not any(
                suppressed in msg for suppressed in self._suppressed_messages
            )

    _logging.getLogger("atomworks.io").addFilter(_SilenceAtomworksIOWarnings())

    absl_logging.get_absl_handler().addFilter(_SilenceAtomworksFilter())
    _logging.getLogger("httpx").setLevel(_logging.WARNING)
    _logging.getLogger("httpcore").setLevel(_logging.WARNING)


def strip_name(
    p, ignore_punctuated_suffixes=False, ignore_capitalization=False, ignore_dots=False
):
    """Strip file extension and optional formatting from a path.

    Args:
        p: File path or filename
        ignore_punctuated_suffixes: If True, keep only the first part before '.'
        ignore_capitalization: If True, convert to lowercase
        ignore_dots: If True, replace '.' with '_'

    Returns:
        Basename without extension and optional formatting removed
    """
    p = str(p)
    # Catcif tag ``archive.catcif:entry_id`` → use the entry_id as the identity;
    # otherwise splitext splits on ``.catcif`` and returns the archive's stem,
    # collapsing every entry to the same key and breaking backlinking.
    if ".catcif:" in p:
        name = p.rsplit(":", 1)[1]
        if ignore_dots:
            name = name.replace(".", "_")
        if ignore_punctuated_suffixes:
            name = name.split(".")[0]
        if ignore_capitalization:
            name = name.lower()
        return name
    if p.endswith(".cif.gz"):
        p = p.replace(".cif.gz", ".cif")
    name = os.path.basename(os.path.splitext(p)[0])
    if ignore_dots:
        name = name.replace(".", "_")
    if ignore_punctuated_suffixes:
        name = name.split(".")[0]
    if ignore_capitalization:
        name = name.lower()
    return name


STRUCTURE_EXTENSIONS = ("*.pdb", "*.cif", "*.cif.gz")


def glob_designs_in_dir(
    dir: str | PathLike,
    recursive: bool = False,
    extensions: tuple[str, ...] = STRUCTURE_EXTENSIONS,
):
    """
    Fetches designs in run directory.
    Includes both cif files and pdb files by default.

    Args:
        dir: Directory to search in
        recursive: Whether to search recursively
        extensions: Glob patterns for file extensions to match.
            Defaults to ("*.pdb", "*.cif", "*.cif.gz").

    Returns:
        List of paths to design files
    """
    paths = []
    for ext in extensions:
        if recursive:
            paths.extend(glob.glob(os.path.join(dir, "**", ext), recursive=True))
        else:
            paths.extend(glob.glob(os.path.join(dir, ext)))
    return paths


def tail_block_log(sandbox_dir: str | PathLike, n_lines: int = 25) -> str:
    """Return a block sandbox's log tail for inclusion in an error.

        <rundir>/<step>/chunk_N        <- sandbox_dir
        <rundir>/logs/<step>/chunk_N.log

    Returns an empty string when no log is available.
    """
    sandbox = Path(sandbox_dir)
    chunk, step = sandbox.name, sandbox.parent.name
    candidates = [
        sandbox.parent.parent / "logs" / step / f"{chunk}.log",
        sandbox.parent.parent / "logs" / f"{step}.log",
        *sorted(sandbox.glob("*.log")),
    ]
    for path in candidates:
        try:
            if not path.is_file():
                continue
            lines = path.read_text(errors="replace").splitlines()
        except OSError:
            continue
        if not lines:
            continue
        tail = "\n".join(lines[-n_lines:])
        return f"\n--- last {min(n_lines, len(lines))} lines of {path} ---\n{tail}\n"
    return ""


def try_float(v):
    """Convert to float, preserving nonnumeric values and normalizing blanks."""
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return v
