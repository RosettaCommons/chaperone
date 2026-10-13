"""InlineMetricBlock: shared scaffold for per-row metric blocks.

Most metric blocks follow the same shape:

    rows_out = []
    for loaded in handler.get_data(which="last", load=True):
        metrics = <compute something on loaded.atom_array>
        rows_out.append(loaded.to_row_with_updates(metrics={self.name: metrics}))
    return PipelineDataHandler(rows_out)

``InlineMetricBlock`` captures that iterate→compute→``to_row_with_updates``
scaffold once. Subclasses implement :meth:`compute_metrics`, which returns the
value to store under ``self.name`` (a metrics dict, or a bare scalar). The
emitted metrics JSON is byte-identical to the hand-rolled loop.

Hooks (all optional):
  - :meth:`get_data_kwargs` — customize the ``handler.get_data`` call
    (e.g. ``load=False``, ``select=[...]``). Default ``which="last", load=True``.
  - :meth:`setup` — runs once before the loop; its return value (``context``)
    is threaded into every :meth:`compute_metrics` call. Use for expensive
    shared state (reference parsing, cache dirs, binary checks).
  - :meth:`teardown` — runs once after the loop in a ``finally`` block, even on
    error. Use for cleanup paired with :meth:`setup`.
"""

import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any

from chaperone.block_config.inline import InlineBlockFunction
from chaperone.data import PipelineDataHandler

logger = logging.getLogger(__name__)


class InlineMetricBlock(InlineBlockFunction):
    """Base class for per-row metric blocks.

    Subclasses implement :meth:`compute_metrics`. The base owns the row
    iteration, the ``to_row_with_updates(metrics={self.name: ...})`` wrapping,
    and the final ``PipelineDataHandler`` construction.
    """

    _creates_new_step = False

    def get_data_kwargs(self) -> dict:
        """Keyword arguments for ``handler.get_data``.

        Override to change which structures are loaded (e.g. ``load=False`` for
        path-only blocks, or ``select=[SELECTIONS.MOTIF_ALLATOM]`` for mask
        access). Defaults to the most common case: the last structure of each
        row, fully loaded.
        """
        return {"which": "last", "load": True}

    def setup(
        self,
        handler: PipelineDataHandler,
        sandbox_dir: Path | None,
        chunk_id: int | None,
    ) -> Any:
        """Run once before iterating rows; return a context object.

        The returned value is passed as ``context`` to every
        :meth:`compute_metrics` call (and to :meth:`teardown`). Default returns
        ``None``.
        """
        return None

    @abstractmethod
    def compute_metrics(self, loaded, *, context: Any = None, index: int = 0):
        """Compute the metric value for one loaded structure.

        Returns the value to store under ``self.name`` — typically a ``dict`` of
        named metrics, but a bare scalar is also allowed (e.g. ``contact_ms``).

        Args:
            loaded: The item yielded by ``handler.get_data`` for this row. With
                the default ``which="last"`` this is a single ``LoadedStructure``;
                when :meth:`get_data_kwargs` requests several structures via
                ``which=<list>`` or ``"all"`` it is a ``tuple`` of them, which
                comparison metrics unpack (e.g. ``first, last = loaded`` for
                RMSD / peppr). The metrics are stored on the row of the LAST
                structure (see :meth:`_row_anchor`).
            context: The object returned by :meth:`setup`.
            index: The 0-based row index within this chunk.
        """

    def teardown(self, context: Any) -> None:
        """Run once after iterating rows, in a ``finally`` block. Default no-op."""
        return None

    @staticmethod
    def _row_anchor(item):
        """The structure whose row receives the computed metrics.

        ``get_data`` yields a single ``LoadedStructure`` for ``which="last"/
        "first"`` and a ``tuple`` of them for ``which=<list>``/``"all"``.
        Single-structure metrics anchor on the item itself; comparison metrics
        anchor on the LAST structure of the tuple (the prediction/design).
        """
        return item[-1] if isinstance(item, tuple) else item

    def execute(self, handler, sandbox_dir=None, chunk_id=None):
        context = self.setup(handler, sandbox_dir, chunk_id)
        rows_out = []
        try:
            for index, item in enumerate(handler.get_data(**self.get_data_kwargs())):
                value = self.compute_metrics(item, context=context, index=index)
                rows_out.append(
                    self._row_anchor(item).to_row_with_updates(
                        metrics={self.name: value}
                    )
                )
        finally:
            self.teardown(context)

        return PipelineDataHandler(rows_out)
