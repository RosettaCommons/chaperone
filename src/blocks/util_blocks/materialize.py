"""MaterializeBlock: Extract archived structures back to disk."""

import logging
from pathlib import Path
from typing import ClassVar, Optional

from pydantic import Field

from pipelines import BlockFunction, PipelineDataHandler

logger = logging.getLogger(__name__)


class MaterializeBlock(BlockFunction):
    """Write stored structures to disk.

    Use after filtering to materialize only surviving rows, avoiding
    unnecessary disk writes for pruned intermediates.
    """

    _name: ClassVar[str] = "materialize"
    _description: ClassVar[str] = "Write stored structures to disk"
    _inline: ClassVar[bool] = True
    _chunked: ClassVar[bool] = False
    _creates_new_step: ClassVar[bool] = False

    output_dir: Optional[str] = Field(
        default=None,
        description="Target directory for materialized files. Defaults to sandbox_dir/materialized.",
    )
    nest_by_step: bool = Field(
        default=True,
        description="Organize files into subdirectories by step name (e.g. output_dir/rfd3/, output_dir/af3/).",
    )

    def __call__(self, handler: PipelineDataHandler, sandbox_dir=None, **kwargs):
        if not handler.is_stored:
            logger.info("No stored structures to materialize; returning handler as-is.")
            return handler

        if self.output_dir:
            target = Path(self.output_dir)
        elif sandbox_dir:
            target = Path(sandbox_dir) / "materialized"
        else:
            raise ValueError(
                "MaterializeBlock requires output_dir or sandbox_dir to write files."
            )

        target.mkdir(parents=True, exist_ok=True)
        logger.info(f"Materializing stored structures to {target}")
        return handler.materialize(target, nest_by_step=self.nest_by_step)
