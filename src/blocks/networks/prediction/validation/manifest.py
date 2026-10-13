"""Manifest PDB assemblies and scoring metadata for prediction validation."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import Field

from pipelines import BlockFunction, PipelineDataHandler, get_setting_field
from pipelines.data import make_row
from pipelines.io import parse_pdb_assembly

logger = logging.getLogger(__name__)

VALIDATION_COLUMNS: tuple[str, ...] = (
    "pdb_id",
    "assembly_id",
    "interfaces_to_score",
    "pn_units_to_score",
    "example_id",
    "bucket",
)
VALIDATION_BUCKETS: tuple[str, ...] = (
    "protein",
    "protein-protein",
    "protein-nucleic",
    "protein-ligand",
    "misc",
)


def _serialized_pn_unit_id_map(atom_array) -> dict[str, str]:
    """Map assembly PN-unit IDs to the IDs recovered after CIF serialization."""
    if "chain_iid" not in atom_array.get_annotation_categories():
        return {}
    mapping: dict[str, str] = {}
    for source, chain_id in zip(atom_array.chain_iid, atom_array.chain_id, strict=True):
        source = str(source)
        target = f"{chain_id}_1"
        if source in mapping and mapping[source] != target:
            raise ValueError(
                f"PN unit {source!r} maps to both {mapping[source]!r} and {target!r}"
            )
        mapping[source] = target
    return mapping


class ManifestValStructuresBlock(BlockFunction):
    """Materialize the parquet's PDB assemblies and carry its scoring schema."""

    _name = "manifest_val_structures"
    _display_name = "ManifestValStructures"
    _description = "Manifest reference PDB assemblies for prediction validation"
    _requires_input = False
    _inline = True
    _chunked = False

    parquet_path: str = Field(
        ...,
        description="Entry-level validation table (.parquet or .csv).",
    )
    pdb_mirror: str = Field(
        default_factory=lambda: get_setting_field("PDB_MIRROR_PATH"),
        description="Local PDB mirror root, sharded by the middle two PDB-ID characters.",
    )

    bucket: str = Field(
        "all",
        description=(
            "Validation composition bucket to select. 'all' keeps every row; "
            f"individual buckets are: {', '.join(VALIDATION_BUCKETS)}."
        ),
    )

    def init_handler(self) -> PipelineDataHandler:
        import pandas as pd

        from pipelines.data.schema import Column

        if not (source := Path(self.parquet_path)).is_file():
            raise FileNotFoundError(source)
        df = pd.read_csv(source) if source.suffix == ".csv" else pd.read_parquet(source)
        missing_columns = [column for column in VALIDATION_COLUMNS if column not in df]
        if missing_columns:
            raise ValueError(
                f"validation table {source} is missing columns {missing_columns}; "
                f"available columns: {list(df.columns)}"
            )
        if self.bucket not in ("all", *VALIDATION_BUCKETS):
            raise ValueError(
                f"unknown validation bucket {self.bucket!r}; expected 'all' or "
                f"one of {list(VALIDATION_BUCKETS)}"
            )
        if self.bucket != "all":
            df = df[df["bucket"] == self.bucket]

        rows = []
        for record in df[list(VALIDATION_COLUMNS)].to_dict("records"):
            validation = dict(record)
            validation["pdb_id"] = str(validation["pdb_id"]).lower()
            validation["assembly_id"] = str(validation["assembly_id"])
            rows.append(
                {
                    Column.STRUCTURES: [],
                    Column.METRICS: json.dumps({"validation": validation}, default=str),
                }
            )
        if not rows:
            raise ValueError(f"validation table {source} contains no rows")
        return PipelineDataHandler(rows)

    def __call__(self, handler, sandbox_dir) -> PipelineDataHandler:
        from atomworks.io.utils.io_utils import to_cif_file

        handler = self.init_handler() if handler is None else handler
        output_dir = Path(sandbox_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for _, row in handler.unpack().iterrows():
            validation = {
                column.removeprefix("validation."): row[column]
                for column in row.index
                if column.startswith("validation.")
            }
            pdb_id = str(validation["pdb_id"]).lower()
            assembly_id = str(validation["assembly_id"])
            atom_array = parse_pdb_assembly(
                pdb_id=pdb_id,
                assembly_id=assembly_id,
                pdb_mirror=self.pdb_mirror,
                clean_atom_array=True,
                disambiguate_assembly_chain_ids=True,
            )
            validation["pn_unit_id_map"] = _serialized_pn_unit_id_map(atom_array)
            target = f"{pdb_id}_{assembly_id}"
            reference_path = output_dir / f"{target}.cif"
            to_cif_file(atom_array, reference_path)
            rows.append(
                make_row(
                    name="validation_target",
                    path=reference_path,
                    metadata={"task": target},
                    metrics={"validation": validation},
                )
            )

        if not rows:
            raise ValueError("manifest_val_structures produced no reference assemblies")
        logger.info("manifest_val_structures built %d reference assemblies", len(rows))
        return PipelineDataHandler(rows)
