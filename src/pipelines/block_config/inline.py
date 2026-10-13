"""
CompiledBlockFunction: Block execution split into prepare→execute→compile steps.
"""

from typing import ClassVar
from abc import abstractmethod
from pathlib import Path
import logging
from pipelines.block_config.base import BlockFunction
from pipelines.data import PipelineDataHandler
from pipelines.common import get_console

logger = logging.getLogger(__name__)
console = get_console()


############################################################
# Simple class for writing and executing prepared jobs
############################################################


class InlineBlockFunction(BlockFunction):
    """Block variant with Python execution and chunking support.

    Resource requirements are defined via _resources ClassVar on each block.
    """

    _inline: ClassVar[bool] = True  # Python execution
    _chunked: ClassVar[bool] = True  # Supports chunking

    @abstractmethod
    def execute(
        self,
        handler: PipelineDataHandler,
        sandbox_dir: Path,
        chunk_id: int,
    ) -> "PipelineDataHandler":
        """Run process inline"""

    def __call__(
        self,
        handler: PipelineDataHandler | None,
        sandbox_dir: Path | str,
        chunk_id=0,
        recompile=False,
        force_exec=False,
    ):
        """Execute block: prepare → execute → compile."""
        sandbox_dir = Path(sandbox_dir)

        output = self.execute(handler, sandbox_dir, chunk_id)
        handler = PipelineDataHandler.merge(handler, output)
        return handler
