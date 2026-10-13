"""Shared tee writer for chunk logs.

Used by both the legacy in-process path (``chaperone.engine.chunking._tee_output``)
and the execvp-finisher path (``chaperone.block_config.runner``).

Stdlib-only at module load so that ``runner.py``'s wait phase can load this
file via ``importlib.util`` without triggering ``chaperone/__init__.py``
(which would pull in polars + the block registry and blow the wait-mode
~16 MiB memory budget).
"""

from __future__ import annotations

import re

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


class _TeeWriter:
    """Mirrors writes to two streams; strips ANSI codes from the file stream.

    The primary stream receives the data verbatim (so terminals keep their
    coloring); the secondary (file) stream gets ANSI-stripped text so the
    on-disk chunk log is readable in plain editors.
    """

    def __init__(self, primary, secondary):
        self._primary = primary
        self._secondary = secondary

    def write(self, data):
        self._primary.write(data)
        self._secondary.write(_ANSI_ESCAPE_RE.sub("", data))

    def flush(self):
        self._primary.flush()
        self._secondary.flush()

    def __getattr__(self, name):
        return getattr(self._primary, name)
