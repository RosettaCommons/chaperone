"""
Copy block for making duplicates of designs.

This block creates n copies of each incoming design file in the sandbox directory.
Useful for expanding a dataset or creating replicate runs.
"""

from pathlib import Path

from chaperone.block_config import CompiledBlockFunction
from chaperone import PipelineDataHandler
from pydantic import Field


class CopyBlock(CompiledBlockFunction):
    """Copy designs for lineage testing and expansion."""

    _name = "copy"
    _display_name = "Copy"
    _description = "Copy designs for lineage testing"
    _inline = False
    _chunked = True

    n_multiply: int = Field(
        default=1, description="Number of copies to create for each design"
    )

    def get_command_template(self) -> list[str]:
        return []

    def init_handler(self):
        """Create a single dummy input for testing."""
        import tempfile

        tmpdir = Path(tempfile.mkdtemp())
        input_path = tmpdir / "design.pdb"
        input_path.write_text(
            "ATOM      1  N   ALA A   1      0.000   0.000   0.000  1.00  0.00           N"
        )
        return PipelineDataHandler.from_paths([input_path], name="input")

    def create_command(self, handler, sandbox_dir, chunk_id):
        """Create bash command to copy files."""

        # Build copy commands - iterate over rows directly
        cmd = []
        for loaded in handler.get_data():
            # Route through stage_file so catcif-tagged upstream paths get
            # materialized to a real file before `cp` sees them.
            staged = Path(self.stage_file(loaded.path, sandbox_dir, subdir="inputs"))
            input_stem = loaded.basename

            # Get all suffixes (e.g., .cif.gz -> ['.cif', '.gz'])
            all_suffixes = "".join(staged.suffixes)

            for n in range(self.n_multiply):
                output_path = Path(sandbox_dir) / f"{input_stem}_copy{n}{all_suffixes}"
                cmd.extend(["cp", str(staged), str(output_path), "&&"])

        # Remove last &&
        return cmd[:-1] if cmd else None

    def compile(self, sandbox_dir):
        return PipelineDataHandler.from_dir(sandbox_dir, name=self.name)


class Copy2Block(CopyBlock):
    _name = "copy2"
    _display_name = "Copy"
    n_multiply: int = Field(
        default=2, description="Number of copies to create for each design"
    )
