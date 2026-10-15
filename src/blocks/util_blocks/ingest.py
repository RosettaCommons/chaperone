from pathlib import Path
import tempfile
import logging
from glob import glob
from chaperone import PipelineDataHandler
from typing import Union, Optional
from chaperone.block_config import BlockFunction
from pydantic import Field, field_validator, model_validator
import polars as pl

logger = logging.getLogger(__name__)

###########################################
# Helper functions
###########################################


def expand_source_paths(source: str) -> list[Path]:
    """Expand source string into list of directory paths.

    Supports:
    - Comma-separated paths: "dir1,dir2,dir3"
    - Glob patterns: "path/to/*/subdir" or "path/**/subdir"
    - Mixed: "dir1,path/to/*"

    Args:
        source: Source string with paths/patterns

    Returns:
        List of Path objects pointing to directories
    """
    all_paths = []

    # Split by comma first
    parts = [p.strip() for p in source.split(",")]

    for part in parts:
        # Check if this part contains glob patterns
        if "*" in part:
            # Use glob to expand the pattern
            matched_paths = glob(part, recursive=True)
            if not matched_paths:
                logger.warning(f"Glob pattern '{part}' matched no paths")
            all_paths.extend(matched_paths)
        else:
            # Direct path
            all_paths.append(part)

    # Convert to Path objects and filter to directories only
    dir_paths = []
    for p in all_paths:
        path = Path(p)
        if path.is_dir():
            dir_paths.append(path)
        elif path.exists():
            logger.warning(f"Skipping non-directory path: {path}")
        else:
            logger.warning(f"Skipping non-existent path: {path}")

    return dir_paths


def compile_dirs(dirs: list[Path], compile_as: str, name: str) -> PipelineDataHandler:
    """Compile directories using a block's compile method.

    Handles failed compilations gracefully by skipping directories that fail.

    Args:
        dirs: List of directory paths to compile
        compile_as: Name of the block to use for compilation

    Returns:
        PipelineDataHandler with compiled data from all successful directories

    Raises:
        RuntimeError: If no directories were successfully compiled
    """
    from blocks import Blocks

    block_cls = Blocks.get(compile_as)
    # Construct the block normally so field defaults are applied. compile()
    # methods on prediction oracles read configured fields (compile_as_metric,
    # read_most_confident_model, store, ...); model_construct leaves them UNSET,
    # raising AttributeError. Use the real constructor so defaults are present.
    block = block_cls(name=name)
    compile_fn = block.compile

    # Single directory - compile directly (no error handling needed)
    if len(dirs) == 1:
        logger.info(f"Compiling single directory: {dirs[0]}")
        return compile_fn(dirs[0])

    # Multiple directories - compile each and concatenate
    output_parquets = []
    failed_dirs = []

    for dir_path in dirs:
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            logger.info(f"Compiling directory: {dir_path}")
            handler = compile_fn(dir_path)
            handler.to_parquet(tmp_path)
            output_parquets.append(tmp_path)
        except Exception as e:
            logger.warning(f"Failed to compile {dir_path}: {e}")
            failed_dirs.append(dir_path)
            # Clean up the temp file for this failed compilation
            tmp_path.unlink(missing_ok=True)

    # Check if we have any successful compilations
    if not output_parquets:
        error_msg = (
            f"Failed to compile any directories. Attempted {len(dirs)} directories."
        )
        if failed_dirs:
            error_msg += f" Failed: {', '.join(str(d) for d in failed_dirs)}"
        raise RuntimeError(error_msg)

    if failed_dirs:
        logger.warning(
            f"Successfully compiled {len(output_parquets)}/{len(dirs)} directories. "
            f"Failed: {', '.join(str(d) for d in failed_dirs)}"
        )

    # Concatenate all parquets
    try:
        dfs = [pl.read_parquet(p) for p in output_parquets]
        df = pl.concat(dfs, how="diagonal_relaxed") if dfs else None
        return PipelineDataHandler(df)
    finally:
        # Clean up temp files
        for p in output_parquets:
            p.unlink(missing_ok=True)


###########################################
# Block definition
###########################################


class IngestBlock(BlockFunction):
    """Ingest data from various sources (dir, csv, dataframe, sequences, or compile)."""

    _name = "ingest"
    _display_name = "Ingest"
    _description = "Ingest data from various sources"
    _requires_input = False
    _inline = True
    _chunked = False

    source: Union[str, PipelineDataHandler] = Field(
        ...,
        description=(
            "Data source to ingest. When compile_as is provided, this should be "
            "a comma-separated list of directory paths. Otherwise, may be a pandas "
            "DataFrame, PipelineDataHandler, CSV/Parquet file, or directory. "
            "Files with sequence columns (e.g., 'sequence') will be auto-detected "
            "and converted to CIF files with empty coordinates."
        ),
        json_schema_extra={"cli_type": str},
    )
    compile_as: Optional[str] = Field(
        default=None,
        description=(
            "If provided, treat source as comma-separated directories and compile "
            "using the specified block (e.g., 'rfd3', 'mpnn')"
        ),
    )

    @field_validator("compile_as")
    @classmethod
    def validate_compile_as(cls, v):
        """Validate that compile_as refers to a valid CompiledBlockFunction."""
        if v is not None:
            from blocks import Blocks
            from chaperone import CompiledBlockFunction

            block_cls = Blocks.get(v)
            if block_cls is None:
                raise ValueError(f"Unknown block name for compile_as: {v}")
            elif not issubclass(block_cls, CompiledBlockFunction):
                raise ValueError(
                    f"Cannot compile as non-CompiledBlockFunction block: "
                    f"(got {block_cls})"
                )
        return v

    @model_validator(mode="after")
    def validate_source_with_compile_as(self):
        """Validate that source is appropriate for the compile_as setting."""
        if self.compile_as is not None:
            # When compile_as is set, source must be a string
            if not isinstance(self.source, str):
                raise ValueError(
                    f"When compile_as is provided, source must be a string of "
                    f"comma-separated directories (got {type(self.source)})"
                )
        return self

    def __call__(self, handler, sandbox_dir) -> PipelineDataHandler:
        """Execute the ingest operation."""
        if self.compile_as is not None:
            dir_paths = expand_source_paths(self.source)
            if not dir_paths:
                raise ValueError(
                    f"No valid directories found from source: {self.source}. "
                    f"Check that paths exist and are directories."
                )
            logger.info(f"Found {len(dir_paths)} directories to compile: {dir_paths}")
            return compile_dirs(dir_paths, self.compile_as, name=self.name)
        else:
            return PipelineDataHandler.load(self.source, allow_sequences=True)
