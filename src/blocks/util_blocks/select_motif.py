"""Block for applying user-defined functions to select motif residues."""

import logging
from typing import Callable, Optional

from pydantic import Field

from pipelines import BlockFunction, PipelineDataHandler
from pipelines.data.structure import LoadedStructure

logger = logging.getLogger(__name__)


class SelectMotifBlock(BlockFunction):
    """Apply a user-defined function across structures to update motif annotations.

    The selector function receives a LoadedStructure and returns a list of
    residue identifiers (e.g. ["A10", "B25"]) to be stored as the "motif"
    metadata field. This is consumed downstream by MPNN's
    keep_motif_sequences_fixed=True (default).

    Example usage::

        def select_interface_residues(loaded: LoadedStructure) -> list[str]:
            atom_array = loaded.atom_array
            # ... compute interface residues ...
            return ["A190", "A195", "B3", "B5"]

        SelectMotif = Blocks.get("select_motif")
        block = SelectMotif(selector=select_interface_residues)
    """

    _name = "select_motif"
    _display_name = "Select Motif"
    _description = "Apply a function to select motif residues for downstream fixing"
    _inline = True
    _chunked = False

    selector: Optional[Callable[[LoadedStructure], list[str]]] = Field(
        default=None,
        description="Function mapping LoadedStructure -> list of residue IDs to fix",
        json_schema_extra={"cli_type": None},
    )
    metadata_key: str = Field(
        default="motif",
        description="Metadata key to write selected residues into",
    )
    append: bool = Field(
        default=True,
        description="Append to existing metadata key (True) or overwrite (False)",
    )

    def __call__(self, handler, sandbox_dir=None, **kwargs):
        if self.selector is None:
            raise ValueError(
                "SelectMotifBlock requires a 'selector' function. "
                "Pass a callable(LoadedStructure) -> list[str]."
            )

        rows_out = []
        for loaded in handler.get_data(which="last", load=True):
            selected = self.selector(loaded)
            if not isinstance(selected, list):
                raise TypeError(
                    f"selector must return list[str], got {type(selected).__name__}"
                )

            existing = (loaded.metadata or {}).get(self.metadata_key) or []
            if self.append:
                merged = sorted(set(existing + selected))
            else:
                merged = sorted(set(selected))

            row = loaded.to_row_with_updates(metadata={self.metadata_key: merged})
            rows_out.append(row)

        logger.info(
            f"SelectMotifBlock: annotated {len(rows_out)} structures "
            f"with '{self.metadata_key}' metadata"
        )
        return PipelineDataHandler(rows_out)
