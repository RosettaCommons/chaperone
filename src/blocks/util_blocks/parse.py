import logging
from pathlib import Path

from atomworks.io.utils.io_utils import to_cif_file, to_pdb_string

from chaperone import PipelineDataHandler
from chaperone.block_config import BlockFunction
from chaperone.io import ensure_bonds

logger = logging.getLogger(__name__)


class ParseToCifBlock(BlockFunction):
    """Parse input structures and re-save as clean CIF files."""

    _name = "to_cif"
    _display_name = "CIF Converter"
    _description = "Parse structures via get_data and re-save as CIF"
    _creates_new_step = True
    _inline = True
    _chunked = False

    def __call__(
        self,
        handler: "PipelineDataHandler",
        sandbox_dir: Path,
    ) -> "PipelineDataHandler":
        all_rows = []

        for loaded in handler.get_data(load=True, which="last"):
            output_path = sandbox_dir / f"{loaded.basename}.cif"
            to_cif_file(ensure_bonds(loaded.atom_array), str(output_path))
            all_rows.append(
                loaded.to_row_with_updates(
                    name=self._name,
                    path=str(output_path),
                )
            )

        return PipelineDataHandler(all_rows)


class ParseToPdbBlock(BlockFunction):
    """Parse input structures and re-save as clean PDB files."""

    _name = "to_pdb"
    _display_name = "PDB Converter"
    _description = "Parse structures via get_data and re-save as PDB"
    _creates_new_step = True
    _inline = True
    _chunked = False

    def __call__(
        self,
        handler: "PipelineDataHandler",
        sandbox_dir: Path,
    ) -> "PipelineDataHandler":
        all_rows = []

        for loaded in handler.get_data(load=True, which="last"):
            output_path = sandbox_dir / f"{loaded.basename}.pdb"
            output_path.write_text(to_pdb_string(loaded.atom_array))
            all_rows.append(
                loaded.to_row_with_updates(
                    name=self._name,
                    path=str(output_path),
                )
            )

        return PipelineDataHandler(all_rows)
