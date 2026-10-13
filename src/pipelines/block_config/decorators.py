"""
Decorators for creating block classes from simple functions.

Provides a convenient way to define blocks without writing full class definitions.
The decorated function becomes the core logic, wrapped in a proper BlockFunction subclass.
"""

from pathlib import Path
from typing import Callable, ClassVar, List, Literal, Optional, TYPE_CHECKING


from pipelines.block_config.base import BlockFunction
from pipelines.block_config.inline import InlineBlockFunction
from pipelines.data import PipelineDataHandler

if TYPE_CHECKING:
    from pipelines.data.structure import LoadedStructure
    from pipelines.data.schema import SELECTIONS


def loaded_block(
    chunked: bool = False,
    which: Literal["last", "first", "all"] = "last",
    load: bool = False,
    select: Optional[List["SELECTIONS"]] = None,
    creates_new_step: bool = True,
) -> BlockFunction:
    """Decorator to create a block that operates on LoadedStructure objects.

    The decorated function receives a LoadedStructure and should return a LoadedStructure
    (or modify it in place). The decorator handles iteration over rows and conversion
    back to PipelineDataHandler.

    Args:
        chunked: If True, block supports chunked execution on SLURM (default False).
            Dynamically created classes are not picklable, so chunked dispatch is
            disabled by default. Use a full class for SLURM-chunked blocks.
        which: Which structure to load from each row - "last", "first", or "all"
        load: If True, load atom arrays into LoadedStructure
        select: List of SELECTIONS to compute masks for (requires load=True)
        creates_new_step: If False, update existing step instead of creating new one

    Example:
        @loaded_block(
            load=True,
            select=[SELECTIONS.MOTIF_CA],
        )
        def compute_motif_com(loaded: LoadedStructure) -> LoadedStructure:
            '''Compute center of mass for motif residues.'''
            mask = loaded.masks[SELECTIONS.MOTIF_CA]
            com = loaded.atom_array.coord[mask].mean(axis=0)
            loaded.metrics['motif_com'] = com.tolist()
            return loaded

        # Use in pipeline:
        engine.run(compute_motif_com)
    """

    def decorator(func: Callable[["LoadedStructure"], "LoadedStructure"]):
        # Create a new class dynamically that inherits from InlineBlockFunction
        func_name = func.__name__
        func_doc = func.__doc__ or f"Block created from {func_name}"

        class DecoratedBlock(InlineBlockFunction):
            """Dynamically created block from @loaded_block decorator."""

            _name: ClassVar[str] = func_name
            _description: ClassVar[str] = func_doc
            _chunked: ClassVar[bool] = chunked
            _inline: ClassVar[bool] = True
            _creates_new_step: ClassVar[bool] = creates_new_step

            # Store decorator config as class attributes
            _which: ClassVar[str] = which
            _load: ClassVar[bool] = load
            _select: ClassVar[Optional[List["SELECTIONS"]]] = select
            _func: ClassVar[Callable] = staticmethod(func)

            def execute(
                self,
                handler: PipelineDataHandler,
                sandbox_dir: Path,
                chunk_id: int,
            ) -> PipelineDataHandler:
                """Execute the decorated function on each row."""
                rows = []
                for loaded in handler.get_data(
                    which=self._which,
                    load=self._load,
                    select=self._select,
                ):
                    # Run the decorated function
                    result = self._func(loaded)
                    rows.append(result.to_row())

                return PipelineDataHandler(rows)

        # Set the class name to match the function
        DecoratedBlock.__name__ = func_name
        DecoratedBlock.__qualname__ = func_name

        # Return an instance of the block (not the class)
        # This allows direct use: engine.run(my_block)
        return DecoratedBlock()

    return decorator


def metric_block(
    chunked: bool = False,
    which: Literal["last", "first"] = "last",
    load: bool = True,
    select: Optional[List["SELECTIONS"]] = None,
    creates_new_step: bool = False,
):
    """Decorator for simple metric computation blocks.

    Similar to @loaded_block but specialized for computing metrics.
    The decorated function receives a LoadedStructure and should return
    a dict of metrics. The returned dict is automatically nested under
    the block name in the row's metrics (matching the standard
    ``{block_name: metrics_dict}`` convention used across all blocks).

    Args:
        chunked: If True, block supports chunked execution (default False).
            Dynamically created classes are not picklable, so chunked dispatch is
            disabled by default. Use a full class for SLURM-chunked blocks.
        which: Which structure to load - "last" or "first"
        load: If True, load atom arrays (default True for metrics)
        select: List of SELECTIONS to compute masks for
        creates_new_step: If False, update existing step instead of creating new one

    Example:
        @metric_block(select=[SELECTIONS.MOTIF_ALLATOM])
        def motif_sasa(loaded: LoadedStructure) -> dict:
            '''Compute solvent-accessible surface area for motif.'''
            # ... computation ...
            return {"motif_sasa": sasa_value}

        # Metrics stored as: row.metrics["motif_sasa"] = {"motif_sasa": sasa_value}
        engine.run(motif_sasa)
    """

    def decorator(func: Callable[["LoadedStructure"], dict]):
        func_name = func.__name__
        func_doc = func.__doc__ or f"Metric block: {func_name}"

        class MetricBlock(InlineBlockFunction):
            """Dynamically created metric block from @metric_block decorator."""

            _name: ClassVar[str] = func_name
            _description: ClassVar[str] = func_doc
            _chunked: ClassVar[bool] = chunked
            _inline: ClassVar[bool] = True
            _creates_new_step: ClassVar[bool] = creates_new_step

            _which: ClassVar[str] = which
            _load: ClassVar[bool] = load
            _select: ClassVar[Optional[List]] = select
            _func: ClassVar[Callable] = staticmethod(func)

            def execute(
                self,
                handler: PipelineDataHandler,
                sandbox_dir: Path,
                chunk_id: int,
            ) -> PipelineDataHandler:
                """Execute the metric function on each row."""
                rows = []
                for loaded in handler.get_data(
                    which=self._which,
                    load=self._load,
                    select=self._select,
                ):
                    new_metrics = self._func(loaded)
                    loaded.metrics[self._name] = new_metrics
                    rows.append(loaded.to_row())

                return PipelineDataHandler(rows)

        MetricBlock.__name__ = func_name
        MetricBlock.__qualname__ = func_name

        return MetricBlock()

    return decorator


structure_wise_block = loaded_block
