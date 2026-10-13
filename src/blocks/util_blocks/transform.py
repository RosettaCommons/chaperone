"""Block for applying atomworks Transform objects to pipeline rows."""

import logging
import tempfile
from pathlib import Path
from typing import Any, Optional

from pydantic import Field

from chaperone import BlockFunction, PipelineDataHandler

logger = logging.getLogger(__name__)


class TransformBlock(BlockFunction):
    """Apply an atomworks Transform to each row of the pipeline.

    Uses LoadedStructure.to_transform_dict() / from_transform_dict() for
    seamless interop with atomworks' dict-based Transform interface.

    Example usage::

        from atomworks.ml.transforms.base import Transform

        class MySASA(Transform):
            def forward(self, data):
                from biotite.structure import sasa
                data["sasa"] = sasa(data["atom_array"]).mean()
                return data

        TransformBlock = Blocks.get("transform")
        block = TransformBlock(
            transform=MySASA(),
            extract_metrics=["sasa"],
        )

    To write a modified atom_array back as a new structure::

        block = TransformBlock(
            transform=my_cropping_transform,
            write_structure=True,
        )
    """

    _name = "transform"
    _display_name = "Atomworks Transform"
    _description = "Apply an atomworks Transform to each pipeline row"
    _inline = True
    _chunked = False

    transform: Any = Field(
        default=None,
        description="An atomworks Transform instance to apply",
        json_schema_extra={"cli_type": None},
    )
    extract_metrics: Optional[list[str]] = Field(
        default=None,
        description="Keys to extract from the transform output as metrics",
    )
    extract_metadata: Optional[list[str]] = Field(
        default=None,
        description="Keys to extract from the transform output as metadata updates",
    )
    write_structure: bool = Field(
        default=False,
        description="If True, write result['atom_array'] back as a new CIF file",
    )

    def __call__(self, handler, sandbox_dir=None, **kwargs):
        from atomworks.ml.transforms.base import Transform

        if self.transform is None:
            raise ValueError("TransformBlock requires a 'transform' instance.")
        if not isinstance(self.transform, Transform):
            raise TypeError(
                f"Expected an atomworks Transform, got {type(self.transform).__name__}"
            )

        sandbox = Path(sandbox_dir) if sandbox_dir else Path(tempfile.mkdtemp())

        rows_out = []
        for loaded in handler.get_data(which="last", load=True):
            data = loaded.to_transform_dict()
            result = self.transform(data)

            updated = loaded.from_transform_dict(
                result,
                extract_metrics=self.extract_metrics,
                extract_metadata=self.extract_metadata,
                metric_namespace=self.name,
            )

            path = None
            if self.write_structure and "atom_array" in result:
                import biotite.structure.io.pdbx as pdbx

                out_path = sandbox / f"{updated.basename}_{self.name}.cif"
                cif = pdbx.CIFFile()
                pdbx.set_structure(cif, result["atom_array"])
                cif.write(out_path)
                path = str(out_path)

            rows_out.append(
                updated.to_row_with_updates(
                    name=self.name if self.write_structure else None,
                    path=path,
                )
            )

        logger.info(f"TransformBlock: processed {len(rows_out)} rows")
        return PipelineDataHandler(rows_out)
