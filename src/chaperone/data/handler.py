"""
PipelineDataHandler: Lightweight wrapper for pipeline DataFrames.

SCHEMA (from schema.py):
    Each row represents a design lineage with:
    - structures: List of structure dicts, each with {name, path, metadata}
    - metrics: JSON string with accumulated metrics from all steps

KEY DESIGN DECISIONS:
    1. Single DataFrame wrapping - no complex state management
    2. merge_rows() in schema.py handles all merge logic (APPEND vs UPDATE)
    3. Backlinking uses strip_name() to match child paths to parent paths
    4. LoadedStructure provides lazy loading of atom arrays

USAGE PATTERN:
    # Create from block output
    handler = PipelineDataHandler(rows)

    # Merge with parent (backlinking happens automatically)
    merged = PipelineDataHandler.merge(parent_handler, child_handler)

    # Iterate over loaded structures
    for structures in handler.get_data():
        first, last = structures[0], structures[-1]
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import re
from functools import lru_cache
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import polars as pl

if TYPE_CHECKING:
    import pandas as pd

from chaperone.common import glob_designs_in_dir, strip_name
from chaperone.data.schema import (
    ArtifactKind,
    SELECTIONS,
    Column,
    StorageFormat,
    StructureColumn,
    _ensure_metadata_fields,
    create_path_entries,
    dump_metrics,
    dump_sequences,
    get_deserialized,
    get_sequences_from_row,
    make_row,
    make_sequence_row,
    merge_rows,
)
from chaperone.data.structure import LoadedStructure

logger = logging.getLogger(__name__)


@lru_cache(maxsize=2048)
def _artifact_sha256(path: str, size: int, mtime_ns: int) -> str:
    """Hash one immutable artifact, memoized by its observable file version."""
    del size, mtime_ns  # Included in the cache key; the digest only reads bytes.
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_pandas_dataframe(value) -> bool:
    """Recognize pandas inputs without importing pandas for normal Polars paths."""
    if type(value).__module__.split(".", 1)[0] != "pandas":
        return False
    import pandas as pd

    return isinstance(value, pd.DataFrame)


def _pandas_notna(value) -> bool:
    """Use pandas' scalar missing-value semantics only on the conversion path."""
    import pandas as pd

    return bool(pd.notna(value))


###########################################
# Native sequence-table detection
###########################################

AA_ONE_TO_THREE = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
    "X": "UNK",
}

# Minimal coordinate-free structure encoding for non-protein sequence inputs.
# The default protein path below remains sequence-native; RNA/DNA still need
# residue and anchor-atom typing so prediction builders create the right entity.
_POLYMER_ANCHORS = {
    "protein": (AA_ONE_TO_THREE, "CA", "C"),
    "rna": ({base: base for base in "ACGU"}, "C3'", "C"),
    "dna": ({base: f"D{base}" for base in "ACGT"}, "C3'", "C"),
}

SEQUENCE_COLUMN_PATTERNS = {
    r"^sequence$",
    r"^seq$",
    r"_sequence$",
    r"_seq$",
    r"^protein_sequence$",
    r"^amino_acid_sequence$",
    r"^aa_sequence$",
    r"^chain_sequence$",
}

CHAIN_SEQUENCE_COLUMN = re.compile(r"^sequence_([A-Za-z][A-Za-z0-9]*)$")


class SequenceOnlyInputError(ValueError):
    """Raised when a structure-only consumer receives a sequence artifact."""


@dataclass(frozen=True)
class PredictionInput:
    """Sequence-first view consumed by structure prediction blocks.

    ``context`` is the optional parent structure used for ligands, nucleic
    acids, fixed-chain context, and templates.  ``sequences`` is authoritative
    for protein chains and may differ from the proteins in that context.
    """

    design_id: str
    sequences: dict[str, str]
    metrics: dict
    context: LoadedStructure | None = None
    parent_design_id: str | None = None

    @property
    def basename(self) -> str:
        return self.design_id

    @property
    def path(self) -> str | None:
        return self.context.path if self.context is not None else None

    @property
    def atom_array(self):
        return self.context.atom_array if self.context is not None else None

    @property
    def metadata(self) -> dict:
        return self.context.metadata if self.context is not None else {}

    @property
    def sequence(self) -> dict[str, str]:
        """Compatibility alias for existing prediction builders."""
        return self.sequences


def _sequence_to_cif(
    sequence: str,
    output_path: Path,
    chain_id: str = "A",
    molecule_type: str = "protein",
) -> Path:
    """Write a one-anchor-per-residue, coordinate-free polymer CIF.

    This compatibility helper is used for RNA/DNA sequence inputs, whose
    residue names determine how structure-prediction blocks type the chain.
    Protein sequence tables continue to use native sequence rows and therefore
    do not pay the cost of materializing placeholder structures.
    """
    from atomworks.io.utils.io_utils import to_cif_file
    from biotite.structure import AtomArray

    from chaperone.io import ensure_bonds

    if molecule_type not in _POLYMER_ANCHORS:
        raise ValueError(
            f"molecule_type must be one of {sorted(_POLYMER_ANCHORS)}, "
            f"got {molecule_type!r}"
        )
    residue_names, anchor_atom, element = _POLYMER_ANCHORS[molecule_type]
    sequence = sequence.strip().upper()
    invalid = sorted(set(sequence) - set(residue_names))
    if invalid:
        raise ValueError(
            f"Sequence contains invalid {molecule_type} residues {invalid}: "
            f"{sequence[:50]}..."
        )
    if not sequence:
        raise ValueError(f"Empty {molecule_type} sequence")

    atom_array = AtomArray(len(sequence))
    for i, residue in enumerate(sequence):
        atom_array.coord[i] = [np.nan, np.nan, np.nan]
        atom_array.chain_id[i] = chain_id
        atom_array.res_id[i] = i + 1
        atom_array.res_name[i] = residue_names[residue]
        atom_array.atom_name[i] = anchor_atom
        atom_array.element[i] = element
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    to_cif_file(ensure_bonds(atom_array), str(output_path))
    return output_path


def _get_sequence_cif_dir() -> Path:
    """Return the checkout run cache used for typed RNA/DNA sequence CIFs."""
    from chaperone import get_settings

    rundir = get_settings().RUNDIR
    if rundir is None:
        raise ValueError(
            "Cannot create RNA/DNA CIF files from sequences: RUNDIR is not set. "
            "Set CHAPERONE_RUNDIR or initialize the Chaperone CLI first."
        )
    cif_dir = Path(rundir) / ".cache" / "sequence_cifs"
    cif_dir.mkdir(parents=True, exist_ok=True)
    return cif_dir


def _detect_sequence_column(df: pl.DataFrame, assert_match=True) -> str | None:
    """Auto-detect columns that contain sequence data."""
    for col in df.columns:
        col_lower = col.lower()
        name_match = any(
            re.match(pattern, col_lower) for pattern in SEQUENCE_COLUMN_PATTERNS
        )
        if name_match:
            logger.info(f"Detected sequence column by name: {col}")
            return col
    if assert_match:
        raise ValueError(
            f"DataFrame missing '{Column.STRUCTURES}' column and no sequence "
            f"columns detected. Available columns: {df.columns}. "
            f"Sequence columns should be named like: 'sequence', 'seq', '*_sequence', etc."
        )
    return None


def _detect_sequence_columns(df: pl.DataFrame) -> dict[str, str]:
    """Return ``chain_id -> column`` for a sequence table.

    A bare ``sequence``/``seq`` column is the single-chain alias for chain A.
    Chain-qualified columns always win so ambiguous tables fail explicitly.
    """
    qualified = {}
    for column in df.columns:
        if column == "sequence_length":
            continue
        match = CHAIN_SEQUENCE_COLUMN.fullmatch(column)
        if match:
            qualified[match.group(1)] = column
    aliases = [c for c in ("sequence", "seq") if c in df.columns]
    if qualified and aliases:
        raise ValueError(
            "Sequence table must use either chain-qualified sequence_<CHAIN> "
            "columns or a bare sequence/seq column, not both."
        )
    if qualified:
        return qualified
    if len(aliases) > 1:
        raise ValueError(
            "Sequence table must use either 'sequence' or 'seq', not both."
        )
    return {"A": aliases[0]} if aliases else {}


def _parse_json_object(value, *, field: str) -> dict:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON in {field}: {error}") from error
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"{field} must be a JSON object")


def _row_with_state(row: dict, *, structures: list, metrics: dict | str) -> dict:
    """Replace structures/metrics while preserving logical-design columns."""
    return {
        str(Column.STRUCTURES): structures,
        str(Column.METRICS): (
            metrics if isinstance(metrics, str) else dump_metrics(metrics)
        ),
        str(Column.DESIGN_ID): row.get(str(Column.DESIGN_ID)),
        str(Column.PARENT_DESIGN_ID): row.get(str(Column.PARENT_DESIGN_ID)),
        str(Column.ARTIFACT_KIND): row.get(str(Column.ARTIFACT_KIND)),
        str(Column.SEQUENCES): row.get(str(Column.SEQUENCES)),
    }


def _convert_sequences_to_pipeline_df(
    df: pl.DataFrame,
    sequence_columns: dict[str, str] | str,
    *,
    base_dir: Path | None = None,
) -> pl.DataFrame:
    """Convert a sequence table without materializing placeholder structures."""
    if isinstance(sequence_columns, str):
        sequence_columns = {"A": sequence_columns}
    if not sequence_columns:
        raise ValueError("No sequence columns found")

    rows = []
    from chaperone.common import iterrows

    for idx, row in enumerate(iterrows(df)):
        sequences = {}
        for chain, column in sequence_columns.items():
            sequence = row.get(column)
            if not isinstance(sequence, str) or not sequence:
                raise ValueError(f"Row {idx} has no sequence for chain {chain}")
            sequence = sequence.strip().upper()
            invalid = sorted(set(sequence) - set(AA_ONE_TO_THREE))
            if invalid:
                raise ValueError(
                    f"Row {idx} chain {chain} contains invalid amino acids: {invalid}"
                )
            sequences[chain] = sequence

        fingerprint = "|".join(f"{c}:{sequences[c]}" for c in sorted(sequences))
        design_id = (
            row.get("design_id")
            or hashlib.sha256(fingerprint.encode()).hexdigest()[:16]
        )
        parent_design_id = row.get("parent_design_id")
        context_path = row.get("context_path")
        if context_path and base_dir is not None:
            context_text = str(context_path)
            if ".catcif:" in context_text:
                archive, entry = context_text.split(":", 1)
                archive_path = Path(archive)
                if not archive_path.is_absolute():
                    context_text = f"{(base_dir / archive_path).resolve()}:{entry}"
                context_path = context_text
            else:
                context_file = Path(context_text)
                if not context_file.is_absolute():
                    context_path = str((base_dir / context_file).resolve())

        reserved = {
            "design_id",
            "parent_design_id",
            "context_path",
            "context_name",
            "context_metadata",
            *sequence_columns.values(),
        }
        metrics = {k: v for k, v in row.items() if k not in reserved and v is not None}
        metrics["sequence_length"] = sum(len(s) for s in sequences.values())
        pipeline_row = make_sequence_row(
            design_id=str(design_id),
            parent_design_id=(
                str(parent_design_id) if parent_design_id is not None else None
            ),
            sequences=sequences,
            context_path=context_path,
            context_name=str(row.get("context_name") or "context"),
            context_metadata=_parse_json_object(
                row.get("context_metadata"), field="context_metadata"
            ),
            metrics=metrics,
        )
        rows.append(pipeline_row)

    if not rows:
        raise ValueError("No valid sequences found in input data")

    ids = [row[str(Column.DESIGN_ID)] for row in rows]
    seen = set()
    duplicates = set()
    for value in ids:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    duplicates = sorted(duplicates)
    if duplicates:
        raise ValueError(f"Duplicate design_id values in sequence table: {duplicates}")

    logger.info("Loaded %d native sequence rows", len(rows))
    return pl.DataFrame(rows, infer_schema_length=None)


def _convert_nucleic_sequences_to_pipeline_df(
    df: pl.DataFrame,
    sequence_column: str,
    molecule_type: str,
) -> pl.DataFrame:
    """Materialize typed RNA/DNA anchors without changing protein ingestion."""
    cif_dir = _get_sequence_cif_dir()
    rows = []
    from chaperone.common import iterrows

    for idx, row in enumerate(iterrows(df)):
        sequence = row.get(sequence_column)
        if not isinstance(sequence, str) or not sequence.strip():
            raise ValueError(f"Row {idx} has no sequence in {sequence_column!r}")
        sequence = sequence.strip().upper()
        digest = hashlib.sha256(f"{molecule_type}:{sequence}".encode()).hexdigest()[:24]
        cif_path = cif_dir / f"{digest}.cif"
        if not cif_path.exists():
            _sequence_to_cif(sequence, cif_path, molecule_type=molecule_type)
        metrics = {key: value for key, value in row.items() if value is not None}
        metrics["sequence_length"] = len(sequence)
        rows.append(make_row(name="ingest", path=str(cif_path), metrics=metrics))

    if not rows:
        raise ValueError("No valid sequences found in input data")
    logger.info("Loaded %d typed %s sequence rows", len(rows), molecule_type)
    return pl.DataFrame(rows, infer_schema_length=None)


def _read_df(path: str | Path) -> pl.DataFrame:
    """Single disk-read dispatch for tabular handler files (.parquet / .csv).

    Uses the final ``Path.suffix`` (case-insensitive) so double-suffixed paths
    (e.g. ``x.cif.gz``) are handled identically everywhere — the trailing
    suffix decides the reader. The previous loaders diverged here
    (``str.endswith`` in ``_to_polars`` vs ``Path.suffix`` in ``from_file``).
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pl.read_parquet(path)
    if suffix == ".csv":
        return pl.read_csv(path)
    raise ValueError(f"Unsupported file format: {path.suffix}. Use .parquet or .csv")


def _to_polars(data) -> pl.DataFrame:
    """Canonicalize input data to polars DataFrame."""
    if isinstance(data, pl.DataFrame):
        return data
    if _is_pandas_dataframe(data):
        return pl.from_pandas(data)
    if isinstance(data, list):
        if not data:
            raise ValueError("Cannot create PipelineDataHandler from empty list")
        # Optional annotations can first appear after Polars' default 100 rows.
        return pl.DataFrame(data, infer_schema_length=None)
    if isinstance(data, (str, Path)):
        return _read_df(data)
    raise TypeError(f"Expected DataFrame, list[dict], or path, got {type(data)}")


class PipelineDataHandler:
    """Thin wrapper around polars DataFrame with pipeline-specific utilities.

    Responsibilities:
        - DataFrame I/O (parquet, CSV, from paths/dir)
        - Chunking for parallel execution
        - Merging child handlers with parent (backlinking)
        - Iterating over LoadedStructure objects

    NOT responsible for:
        - Metadata inheritance logic (handled by merge_rows in schema.py)
        - Structure matching logic (handled by _match_row_in_lookup)
    """

    # ---------------------------------------------------------------------
    # Init / save options
    # ---------------------------------------------------------------------

    def __init__(
        self,
        data: Union[
            pl.DataFrame, pd.DataFrame, list[dict], PathLike, "PipelineDataHandler"
        ],
        allow_sequences: bool = False,
    ):
        """Initialize PipelineDataHandler.

        Args:
            data: Input data - DataFrame, list of dicts, or path to parquet/csv file.
                  Unpacked DataFrames (from unpack()) are automatically re-packed.
            allow_sequences: If True and data lacks 'structures' column but has a
                sequence-like column, load it as a native sequence artifact.
        """
        if isinstance(data, PipelineDataHandler):
            self._df = data._df
            # Preserve lineage stamp when re-wrapping (engine.run(data=handler)).
            self._producing_steps = list(getattr(data, "_producing_steps", []))
            return

        df = _to_polars(data)
        self._df = self._normalize_schema(df, allow_sequences)
        # Step name(s) that produced this handler — set by the engine when a
        # step finishes (see ``stamp_producing_step``). Empty for raw/external
        # inputs. Drives branch- and passthrough-aware DAG parentage.
        self._producing_steps: list[str] = []

    def __repr__(self) -> str:
        return f"PipelineDataHandler({len(self._df)} rows)\n{self._df.head()}"

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return PipelineDataHandler(self._df[idx])
        import json

        structures = self._df["structures"][idx]
        metrics = self._df["metrics"][idx]
        if isinstance(metrics, str):
            metrics = json.loads(metrics)
        return structures, metrics

    def __add__(self, other: "PipelineDataHandler") -> "PipelineDataHandler":
        """Concatenate two handlers vertically (row-wise)."""
        if not isinstance(other, PipelineDataHandler):
            return NotImplemented
        result = PipelineDataHandler(
            pl.concat([self._df, other._df], how="diagonal_relaxed")
        )
        # Union producers so a fan-in (handler_a + handler_b) records BOTH
        # parents, preserving the full DAG for downstream lineage.
        result._producing_steps = list(
            dict.fromkeys([*self._producing_steps, *other._producing_steps])
        )
        return result

    def content_fingerprint(self) -> str:
        """Return a stable digest of rows and the artifacts they reference.

        Cache directories are often reused while a campaign rewrites inputs at
        the same paths. Row counts, block parameters, and filenames therefore
        do not identify the actual computation. Include the complete logical
        rows plus the bytes of every referenced loose file or CatCIF archive.

        File digests are memoized by path, size, and nanosecond mtime so a
        multi-step campaign hashes an unchanged archive only once per process.
        Missing paths are represented explicitly in the digest; they are not
        silently treated as empty files.
        """
        digest = hashlib.sha256()
        artifacts: set[str] = set()

        for row in self._df.iter_rows(named=True):
            encoded = json.dumps(
                row, sort_keys=True, separators=(",", ":"), default=str
            ).encode()
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
            for structure in row.get(str(Column.STRUCTURES)) or []:
                for field in (StructureColumn.PATH, StructureColumn.STORE_PATH):
                    value = structure.get(str(field))
                    if not value:
                        continue
                    text = str(value)
                    # CatCIF handles address an entry after the final colon;
                    # the archive bytes plus the serialized row's entry name
                    # identify that input without materializing it.
                    physical = text.split(":", 1)[0] if ".catcif:" in text else text
                    artifacts.add(physical)

        for artifact in sorted(artifacts):
            digest.update(artifact.encode())
            path = Path(artifact)
            try:
                stat = path.stat()
                if not path.is_file():
                    digest.update(b"\0not-a-file")
                    continue
                digest.update(
                    _artifact_sha256(
                        str(path.resolve()), stat.st_size, stat.st_mtime_ns
                    ).encode()
                )
            except OSError as error:
                digest.update(f"\0missing:{type(error).__name__}".encode())

        return digest.hexdigest()

    @property
    def producing_steps(self) -> list[str]:
        """Step name(s) that produced this handler's data (lineage parents)."""
        return list(self._producing_steps)

    def stamp_producing_step(self, step_name: str) -> "PipelineDataHandler":
        """Record which pipeline step produced this handler.

        The engine calls this when a step finishes so that a later
        ``engine.run(next_block, data=this_handler)`` can register the *true*
        data parent (this step) rather than whichever step happened to be the
        manifest tail. This is what makes branches (one producer feeding several
        consumers) and passthrough metric blocks resolve correctly. Returns
        ``self`` for chaining.
        """
        self._producing_steps = [step_name]
        return self

    @property
    def is_stored(self) -> bool:
        """True if any structure in this handler has a store_path set."""
        for row in self._df.iter_rows(named=True):
            for s in row[Column.STRUCTURES]:
                if s.get(StructureColumn.STORE_PATH):
                    return True
        return False

    @property
    def shape(self):
        return self._df.shape

    def subset_by_name(self, name: str) -> "PipelineDataHandler":
        """Return a new handler keeping only structures matching the given name."""
        col = str(Column.STRUCTURES)
        df = self._df.with_columns(
            pl.col(col).list.eval(
                pl.element().filter(
                    pl.element().struct.field(str(StructureColumn.NAME)) == name
                )
            )
        ).filter(pl.col(col).list.len() > 0)
        return PipelineDataHandler(df)

    @staticmethod
    def _normalize_schema(df: pl.DataFrame, allow_sequences: bool) -> pl.DataFrame:
        """Normalize DataFrame to pipeline schema (pack if needed, convert sequences)."""
        cols = set(df.columns)
        has_structures = str(Column.STRUCTURES) in cols
        has_metrics = str(Column.METRICS) in cols

        # Unpacked DataFrame (has structures but not metrics) - repack
        if has_structures and not has_metrics:
            return PipelineDataHandler._ensure_design_columns(
                PipelineDataHandler.pack(df)._df
            )

        # Valid packed schema - use as-is
        if has_structures:
            return PipelineDataHandler._ensure_design_columns(df)

        # No structures column — check for unpacked structure columns (<name>.path)
        if _detect_structure_names(df.columns):
            return PipelineDataHandler._ensure_design_columns(
                PipelineDataHandler.pack(df)._df
            )

        # No structures at all — try sequence conversion
        if not allow_sequences:
            raise ValueError(
                f"DataFrame missing '{Column.STRUCTURES}' column. "
                f"Available: {df.columns}. Use allow_sequences=True to auto-detect."
            )
        sequence_columns = _detect_sequence_columns(df)
        if not sequence_columns:
            _detect_sequence_column(df)  # raises the established diagnostic
        return PipelineDataHandler._ensure_design_columns(
            _convert_sequences_to_pipeline_df(df, sequence_columns)
        )

    @staticmethod
    def _ensure_design_columns(df: pl.DataFrame) -> pl.DataFrame:
        """Backfill sequence-era columns on legacy packed handlers."""
        additions = []
        for column in (
            Column.DESIGN_ID,
            Column.PARENT_DESIGN_ID,
            Column.ARTIFACT_KIND,
            Column.SEQUENCES,
        ):
            if str(column) not in df.columns:
                additions.append(pl.lit(None, dtype=pl.Utf8).alias(str(column)))
        if additions:
            df = df.with_columns(additions)

        def _infer_design_id(structures):
            if structures is None or len(structures) == 0:
                return None
            struct = structures[-1]
            if hasattr(struct, "to_dict"):
                struct = struct.to_dict()
            path = struct.get(str(StructureColumn.PATH)) or struct.get(
                str(StructureColumn.PATH_HANDLE)
            )
            return strip_name(path) if path else None

        inferred_id = pl.col(str(Column.STRUCTURES)).map_elements(
            _infer_design_id, return_dtype=pl.Utf8
        )
        return df.with_columns(
            pl.coalesce([pl.col(str(Column.DESIGN_ID)), inferred_id]).alias(
                str(Column.DESIGN_ID)
            ),
            pl.when(pl.col(str(Column.ARTIFACT_KIND)).is_not_null())
            .then(pl.col(str(Column.ARTIFACT_KIND)))
            .when(pl.col(str(Column.SEQUENCES)).is_not_null())
            .then(pl.lit(str(ArtifactKind.SEQUENCE)))
            .otherwise(pl.lit(str(ArtifactKind.STRUCTURE)))
            .alias(str(Column.ARTIFACT_KIND)),
        )

    @classmethod
    def from_file(
        cls, path: str | Path, allow_sequences: bool = False
    ) -> "PipelineDataHandler":
        """Load a handler from a parquet or CSV file.

        Args:
            path: Path to a .parquet or .csv file.
            allow_sequences: If True, auto-detect sequence columns for data without structures.
        """
        path = Path(path)
        df = _read_df(path)
        if (
            allow_sequences
            and str(Column.STRUCTURES) not in df.columns
            and not _detect_structure_names(df.columns)
        ):
            columns = _detect_sequence_columns(df)
            if columns:
                df = _convert_sequences_to_pipeline_df(
                    df, columns, base_dir=path.parent
                )
                return cls(df)
        return cls(df, allow_sequences=allow_sequences)

    @classmethod
    def from_sequences(
        cls,
        data: Union[pl.DataFrame, pd.DataFrame, str, Path],
        molecule_type: str = "protein",
    ) -> "PipelineDataHandler":
        """Create a handler from data containing sequence columns.

        Auto-detects a single ``sequence``/``seq`` column or chain-qualified
        ``sequence_<CHAIN>`` columns. Protein inputs remain native sequence
        artifacts. RNA/DNA inputs are written as coordinate-free typed CIFs so
        prediction builders can distinguish nucleic acids from proteins.

        Args:
            data: DataFrame or path to CSV/parquet with a sequence column.
            molecule_type: ``"protein"``, ``"rna"`` or ``"dna"``.
        """
        if molecule_type not in _POLYMER_ANCHORS:
            raise ValueError(
                f"molecule_type must be one of {sorted(_POLYMER_ANCHORS)}, "
                f"got {molecule_type!r}"
            )
        if isinstance(data, (str, Path)) and molecule_type == "protein":
            return cls.from_file(data, allow_sequences=True)
        df = _read_df(data) if isinstance(data, (str, Path)) else _to_polars(data)
        if molecule_type != "protein":
            sequence_column = _detect_sequence_column(df)
            return cls(
                _convert_nucleic_sequences_to_pipeline_df(
                    df, sequence_column, molecule_type
                )
            )
        sequence_columns = _detect_sequence_columns(df)
        if not sequence_columns:
            _detect_sequence_column(df)
        result_df = _convert_sequences_to_pipeline_df(df, sequence_columns)
        return cls(result_df)

    @classmethod
    def load(cls, source, allow_sequences: bool = False) -> "PipelineDataHandler":
        """Smart loader: accepts handler, file path, or directory and returns a handler.

        Supports:
        - PipelineDataHandler (returned as-is)
        - Path to .parquet or .csv file
        - Directory containing .parquet or .csv files (uses first found)
        - DataFrame (polars or pandas)

        Args:
            source: Data source to load.
            allow_sequences: If True, auto-detect sequence columns for data without structures.
        """
        if isinstance(source, PipelineDataHandler):
            return source

        if isinstance(source, (str, PathLike)):
            path = Path(source)
            if path.is_file():
                return cls.from_file(path, allow_sequences=allow_sequences)
            if path.is_dir():
                # Look for parquet/csv files in the directory
                for suffix in [".parquet", ".csv"]:
                    candidates = [p for p in path.iterdir() if p.suffix == suffix]
                    if candidates:
                        # Make selection deterministic by sorting candidates by name
                        candidates = sorted(candidates, key=lambda p: p.name)
                        return cls.from_file(
                            candidates[0], allow_sequences=allow_sequences
                        )
                raise ValueError(
                    f"No .parquet or .csv files found in directory: {path}"
                )

        if isinstance(source, pl.DataFrame) or _is_pandas_dataframe(source):
            return cls(source, allow_sequences=allow_sequences)

        raise TypeError(
            f"Cannot load from {type(source)}. Expected PipelineDataHandler, "
            f"file path, directory, or DataFrame."
        )

    @classmethod
    def from_dir(
        cls, dir: str | PathLike, name: str, recursive=False
    ) -> "PipelineDataHandler":
        paths = glob_designs_in_dir(dir, recursive=recursive)
        if not paths:
            raise ValueError(f"No design files found in directory: {dir}")
        return cls.from_paths(paths, name)

    @classmethod
    def from_paths(cls, paths: Sequence[PathLike], name: str) -> "PipelineDataHandler":
        """Create handler from paths."""
        if paths is None or len(paths) == 0:
            raise ValueError("No paths provided to from_paths")

        data = create_path_entries(paths, name)

        return cls(data)

    def write_parquet(self, path, *, debug_csv: bool | None = None, **kwargs):
        """Write the handler DataFrame to parquet.

        Args:
            debug_csv: Whether to also emit the side-car ``.csv`` debug dump
                (an O(rows×metrics) Python-level ``unpack()``). The side-car is
                **opt-in**: only emitted when ``debug_csv=True``. Leaving it
                ``None``/``False`` skips the dump — this avoids paying the
                unpack cost on the many intermediate writes (shared input
                parquet, per-chunk output, per-step output). Final-step or
                user-facing writes should pass ``debug_csv=True`` explicitly.
                When ``debug_csv=True`` the existing ``DEBUG_CSV_MAX_ROWS``
                setting still gates very large DataFrames.
        """
        kwargs.pop("index", None)
        self._df.write_parquet(path, **kwargs)

        if not debug_csv:
            return

        # WRITE DEBUG (skip for large DataFrames to avoid O(n) Python-level serialization)
        from chaperone import get_settings

        max_rows = get_settings().DEBUG_CSV_MAX_ROWS
        if max_rows == -1 or (max_rows > 0 and len(self._df) <= max_rows):
            try:
                self.unpack().to_csv(str(path).replace(".parquet", ".csv"), index=False)
            except Exception:
                _write_debug_csv(self._df, str(path).replace(".parquet", ".csv"))
            logger.info(
                "Wrote debug csv {}".format(str(path).replace(".parquet", ".csv"))
            )

    to_parquet = write_parquet

    def save_csv(self, path, **kwargs):
        """Save handler as unpacked CSV; fall back to raw DataFrame CSV on failure."""
        path = Path(path)
        try:
            self.unpack().to_csv(path, index=False, **kwargs)
        except Exception:
            _write_debug_csv(self._df, path)
        logger.info(f"Wrote csv {path}")

    @property
    def has_sequence_artifacts(self) -> bool:
        """Whether any row's current logical artifact is sequence-only."""
        return bool(
            self._df.select(
                (pl.col(str(Column.ARTIFACT_KIND)) == str(ArtifactKind.SEQUENCE)).any()
            ).item()
        )

    @property
    def has_sequence_data(self) -> bool:
        """Whether rows carry chain sequences, including folded descendants."""
        return bool(
            self._df.select(pl.col(str(Column.SEQUENCES)).is_not_null().any()).item()
        )

    def sequence_dataframe(self) -> pl.DataFrame:
        """Return the stable, user-facing flat sequence table."""
        rows = []
        for row in self._df.iter_rows(named=True):
            sequences = get_sequences_from_row(row)
            if not sequences:
                continue
            structures = row.get(str(Column.STRUCTURES)) or []
            context = structures[-1] if structures else None
            out = {
                "design_id": row.get(str(Column.DESIGN_ID)),
                "parent_design_id": row.get(str(Column.PARENT_DESIGN_ID)),
                "context_path": (
                    context.get(str(StructureColumn.PATH)) if context else None
                ),
                "context_name": (
                    context.get(str(StructureColumn.NAME)) if context else None
                ),
                "context_metadata": json.dumps(
                    (context.get(str(StructureColumn.METADATA)) or {})
                    if context
                    else {},
                    sort_keys=True,
                ),
            }
            out.update({f"sequence_{chain}": seq for chain, seq in sequences.items()})
            out.update(
                {
                    key: _csv_safe_value(value)
                    for key, value in _flatten_dict(
                        get_deserialized(row, Column.METRICS)
                    ).items()
                }
            )
            rows.append(out)
        if not rows:
            return pl.DataFrame(
                schema={
                    "design_id": pl.Utf8,
                    "parent_design_id": pl.Utf8,
                    "context_path": pl.Utf8,
                    "context_name": pl.Utf8,
                    "context_metadata": pl.Utf8,
                }
            )
        return pl.DataFrame(rows, infer_schema_length=None)

    def write_sequences_csv(self, path: str | Path) -> None:
        """Write :meth:`sequence_dataframe` to CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.sequence_dataframe().write_csv(path)
        logger.info("Wrote sequence table %s", path)

    def store(
        self,
        step: str,
        store_path: str | Path,
        *,
        cleanup: bool = False,
    ) -> "PipelineDataHandler":
        """Consolidate a step's structures into a catcif archive.

        The ``.catcif`` suffix is added to ``store_path`` when missing. Each
        structure becomes one archive entry, and handler paths are rewritten
        to self-describing ``archive.catcif:entry_id`` tags.

        Row count is preserved exactly; only structure dicts matching
        ``step`` are modified.

        Args:
            step: Step name whose structures to consolidate. Three sentinels
                are also accepted:

                - ``"last"`` — store only the most recent step's structures.
                - ``"all"`` — store every distinct step with structures.
                - ``"pending"`` — store only steps whose structures don't yet
                  have a ``store_path`` set.

                When a sentinel is used, ``store_path`` is treated as a
                directory and per-step archives are written as
                ``<store_path>/<step>.catcif``.
            store_path: Output path. A file for a single named step; a
                directory for sentinel modes.
            cleanup: If True, delete the original structure files after
                successful archival.

        Returns:
            self, with updated structure entries for the stored step(s).
        """
        sentinels = {"last", "all", "pending"}
        if step in sentinels:
            return self._store_multi(step, store_path, cleanup=cleanup)

        from chaperone.data.catcif import write_handler_step_to_catcif

        return write_handler_step_to_catcif(self, step, store_path, cleanup=cleanup)

    def _store_multi(
        self,
        which: str,
        store_dir: str | Path,
        *,
        cleanup: bool,
    ) -> "PipelineDataHandler":
        """Store several steps in one call under ``store_dir``.

        See :py:meth:`store` for the meaning of ``which``
        (``"last" | "all" | "pending"``).
        """
        # Walk the handler to determine per-step status:
        #   - order: insertion order of step names in the lineage
        #   - has_structs: step has at least one structure dict
        #   - stored: every structure for that step already has a store_path
        step_order: list[str] = []
        has_structs: set[str] = set()
        stored: dict[str, bool] = {}
        for row in self._df.iter_rows(named=True):
            for struct in row[Column.STRUCTURES]:
                name = struct[StructureColumn.NAME]
                if name not in has_structs:
                    has_structs.add(name)
                    step_order.append(name)
                    stored[name] = True
                if not struct.get(StructureColumn.STORE_PATH):
                    stored[name] = False

        if not step_order:
            raise ValueError("Handler has no structures to store.")

        if which == "last":
            targets = [step_order[-1]]
        elif which == "all":
            targets = list(step_order)
        elif which == "pending":
            targets = [s for s in step_order if not stored[s]]
        else:
            raise ValueError(f"Unknown which: {which!r}")

        if not targets:
            logger.info(
                "store(which=%r): no steps to store; all already stored.", which
            )
            return self

        store_dir = Path(store_dir)
        store_dir.mkdir(parents=True, exist_ok=True)
        for step_name in targets:
            path = store_dir / f"{step_name}.catcif"
            self.store(step_name, path, cleanup=cleanup)
        return self

    def materialize(
        self,
        output_dir: str | Path,
        nest_by_step: bool = True,
        *,
        expand_catcif: bool = True,
    ) -> "PipelineDataHandler":
        """Write stored structures to disk and return handler with rewritten paths.

        Only rewrites paths that have a store_path set; leaves other paths as-is.

        Args:
            output_dir: Directory to write materialized files into.
            nest_by_step: If True, organize files into subdirectories by step name
                (e.g. ``output_dir/rfd3/design_<digest>.cif``). Filenames include
                archive/entry identity to avoid collisions; design IDs are unchanged.
            expand_catcif: If True (default), extract catcif-stored entries to
                individual files. If False, leave catcif paths as self-describing
                ``archive.catcif:entry_id`` tags (readers resolve them transparently).

        Returns:
            A new handler with paths rewritten to the materialized files.

        Raises:
            ValueError: If no structures have a store_path set.
        """
        if not self.is_stored:
            raise ValueError("Handler has no store; call store() first.")

        from chaperone.data.catcif import catcif_entry_filename

        output_dir = Path(output_dir)

        # Discover catcif stores and preserve step names for nested output.
        catcif_stores: set[str] = set()
        path_to_step: dict[str, str] = {}
        for row in self._df.iter_rows(named=True):
            for s in row[Column.STRUCTURES]:
                sp = s.get(StructureColumn.STORE_PATH)
                if not sp:
                    continue
                fmt = s.get(StructureColumn.STORAGE_FORMAT)
                if fmt == StorageFormat.CATCIF:
                    catcif_stores.add(sp)
                orig = s.get(StructureColumn.PATH)
                if nest_by_step and orig:
                    path_to_step[orig] = s[StructureColumn.NAME]

        path_map: dict[str, str] = {}

        # Optionally expand archive entries to files.
        if catcif_stores and expand_catcif:
            import catcif_tools

            # Iterate actual in-use catcif tags; do not expand unused entries.
            seen_tags: set[str] = set()
            for row in self._df.iter_rows(named=True):
                for s in row[Column.STRUCTURES]:
                    if s.get(StructureColumn.STORAGE_FORMAT) != StorageFormat.CATCIF:
                        continue
                    tag = s.get(StructureColumn.PATH)
                    if tag in seen_tags:
                        continue
                    seen_tags.add(tag)
                    filename = catcif_entry_filename(tag)
                    if nest_by_step and tag in path_to_step:
                        out_path = output_dir / path_to_step[tag] / filename
                    else:
                        out_path = output_dir / filename
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(catcif_tools.get_structure(tag))
                    path_map[tag] = str(out_path)

        rows = []
        for row in self._df.iter_rows(named=True):
            new_structures = []
            for struct in row[Column.STRUCTURES]:
                old_path = struct[StructureColumn.PATH]
                new_path = path_map.get(old_path, old_path)
                was_rewritten = new_path != old_path
                new_structures.append(
                    {
                        StructureColumn.NAME: struct[StructureColumn.NAME],
                        StructureColumn.PATH: new_path,
                        StructureColumn.PATH_HANDLE: struct.get(
                            StructureColumn.PATH_HANDLE
                        ),
                        # Clear store pointers only when the path was materialized
                        # (otherwise catcif tags still need them for retrieval).
                        StructureColumn.STORE_PATH: None
                        if was_rewritten
                        else struct.get(StructureColumn.STORE_PATH),
                        StructureColumn.STORAGE_FORMAT: None
                        if was_rewritten
                        else struct.get(StructureColumn.STORAGE_FORMAT),
                        StructureColumn.METADATA: struct.get(StructureColumn.METADATA),
                    }
                )
            rows.append(
                _row_with_state(
                    row,
                    structures=new_structures,
                    metrics=row[Column.METRICS],
                )
            )
        return PipelineDataHandler(rows)

    def to_pandas(self):
        return self._df.to_pandas()

    def unpack(
        self, all: bool = True, to_pandas: bool = True
    ) -> "pl.DataFrame | pd.DataFrame":
        """Unpack metrics into a flat DataFrame for filtering and aggregations.

        Flattens the nested metrics JSON into separate columns with dot-separated names.

        Args:
            all: If True, also unpack structures column into <name>.path and
                 <name>.metadata.* columns. The structures column is removed.
                 If False (default), structures column is preserved as-is.
            to_pandas: If True (default), return a pandas DataFrame.
                 If False, return a polars DataFrame.

        Returns:
            DataFrame with flattened columns (pandas by default, polars if to_pandas=False)

        Example:
            With all=False (default):
                Input metrics: '{"rmsd": {"bkbn.rmsd": 1.2}, "mpnn": {"score": -5.0}}'
                Output columns: ["structures", "rmsd.bkbn.rmsd", "mpnn.score"]

            With all=True:
                Input: structures=[{name: "rfd3", path: "/x.pdb", metadata: {task: "enzyme"}}]
                Output columns: ["rfd3.path", "rfd3.metadata.task", "rmsd.bkbn.rmsd", ...]
        """
        from chaperone.data.schema import get_deserialized

        rows = []
        for row in self._df.iter_rows(named=True):
            flat_row = {
                str(Column.DESIGN_ID): row.get(str(Column.DESIGN_ID)),
                str(Column.PARENT_DESIGN_ID): row.get(str(Column.PARENT_DESIGN_ID)),
                str(Column.ARTIFACT_KIND): row.get(str(Column.ARTIFACT_KIND)),
            }
            for chain_id, sequence in get_sequences_from_row(row).items():
                flat_row[f"sequence_{chain_id}"] = sequence

            # Unpack structures if all=True, otherwise preserve as-is
            if all:
                structures = row[Column.STRUCTURES]
                for struct in structures:
                    name = struct[StructureColumn.NAME]
                    flat_row[f"{name}.path"] = struct[StructureColumn.PATH]
                    store_path = struct.get(StructureColumn.STORE_PATH)
                    flat_row[f"{name}.stored"] = bool(store_path)
                    flat_row[f"{name}.store_path"] = store_path
                    flat_row[f"{name}.storage_format"] = struct.get(
                        StructureColumn.STORAGE_FORMAT
                    )
                    metadata = struct.get(StructureColumn.METADATA) or {}
                    for meta_key, meta_val in metadata.items():
                        flat_row[f"{name}.metadata.{meta_key}"] = meta_val
            else:
                flat_row[Column.STRUCTURES] = row[Column.STRUCTURES]

            # Unpack metrics
            metrics = get_deserialized(row, Column.METRICS)
            flat_row.update(_flatten_dict(metrics))
            rows.append(flat_row)

        result = pl.DataFrame(rows, infer_schema_length=None)

        # Drop all-null metadata columns (e.g. protein_hunter.metadata.aromatics_near_ligand)
        metadata_cols = [c for c in result.columns if ".metadata." in c]
        null_metadata_cols = [c for c in metadata_cols if result[c].is_null().all()]
        if null_metadata_cols:
            result = result.drop(null_metadata_cols)

        return result.to_pandas() if to_pandas else result

    @classmethod
    def pack(cls, df: pl.DataFrame | pd.DataFrame) -> "PipelineDataHandler":
        """Pack a flattened DataFrame back into PipelineDataHandler format.

        Takes a pandas or Polars DataFrame (typically from ``unpack()``) and
        compresses columns back into the pipeline schema. Handles both cases:
        - DataFrame with 'structures' column: pack metrics only
        - DataFrame with unpacked structures (<name>.path, <name>.metadata.*): reconstruct structures

        Args:
            df: pandas or Polars DataFrame with either a 'structures' column or
                unpacked structure columns, plus flattened metric columns

        Returns:
            PipelineDataHandler with structures and metrics re-packed

        Example:
            With structures column:
                Input columns: ["structures", "rmsd.bkbn.rmsd", "mpnn.score"]
                Output metrics: '{"rmsd": {"bkbn.rmsd": 1.2}, "mpnn": {"score": -5.0}}'

            With unpacked structures:
                Input columns: ["rfd3.path", "rfd3.metadata.task", "rmsd.bkbn.rmsd"]
                Output: structures reconstructed from <name>.* columns
        """

        df = _to_polars(df)
        has_structures_col = Column.STRUCTURES in df.columns

        # Detect unpacked structure columns (pattern: <name>.path)
        structure_names = []
        if not has_structures_col:
            structure_names = _detect_structure_names(df.columns)

        state_columns = {
            str(Column.DESIGN_ID),
            str(Column.PARENT_DESIGN_ID),
            str(Column.ARTIFACT_KIND),
            str(Column.SEQUENCES),
        }
        sequence_columns = _detect_sequence_columns(df)
        rows = []
        for row in df.iter_rows(named=True):
            if has_structures_col:
                structures = row[Column.STRUCTURES]
                # Collect all non-structures columns into metrics
                metrics_flat = {
                    k: v
                    for k, v in row.items()
                    if k != Column.STRUCTURES
                    and k not in state_columns
                    and k not in sequence_columns.values()
                    and _pandas_notna(v)
                }
            else:
                # Reconstruct structures from unpacked columns
                structures = _reconstruct_structures(row, structure_names)
                # Collect non-structure columns into metrics
                # Structure columns are specifically: <name>.path and <name>.metadata.*
                structure_cols = _get_structure_column_names(
                    structure_names, df.columns
                )
                metrics_flat = {
                    k: v
                    for k, v in row.items()
                    if k not in structure_cols
                    and k not in state_columns
                    and k not in sequence_columns.values()
                    and v is not None
                }

            # Unflatten the metrics dict
            metrics = _unflatten_dict(metrics_flat)

            sequences = {
                chain: str(row[column])
                for chain, column in sequence_columns.items()
                if row.get(column) not in (None, "")
            }
            if not sequences:
                sequences = get_sequences_from_row(row)
            rows.append(
                {
                    str(Column.STRUCTURES): structures,
                    str(Column.METRICS): dump_metrics(metrics),
                    str(Column.DESIGN_ID): row.get(str(Column.DESIGN_ID)),
                    str(Column.PARENT_DESIGN_ID): row.get(str(Column.PARENT_DESIGN_ID)),
                    str(Column.ARTIFACT_KIND): row.get(str(Column.ARTIFACT_KIND)),
                    str(Column.SEQUENCES): dump_sequences(sequences or None),
                }
            )

        # Bypass __init__ to avoid re-normalization
        handler = cls.__new__(cls)
        handler._df = cls._ensure_design_columns(
            pl.DataFrame(rows, infer_schema_length=None)
        )
        handler._producing_steps = []
        return handler

    def get_slices(
        self,
        rows_per_chunk: int,
    ) -> Dict[int, Tuple[int, int]]:
        n = len(self)
        return {
            i: (
                i * rows_per_chunk,
                min(rows_per_chunk, n - i * rows_per_chunk),
            )
            for i in range((n + rows_per_chunk - 1) // rows_per_chunk)
        }

    # ---------------------------------------------------------------------
    # Data Loading API
    # ---------------------------------------------------------------------

    def get_data(
        self,
        load: bool = False,
        select: Optional[List[SELECTIONS]] = None,
        which: "str | list[int]" = "last",
        num_workers: int = 0,
    ):
        """
        Primary data entrypoint for blocks.

        Args:
            load: If True, load atom arrays from paths
            select: List of SELECTIONS to compute masks for
            which: Which structure(s) to yield per row:
                - "last": Yield only the last LoadedStructure (default, most common)
                - "first": Yield only the first LoadedStructure
                - "all": Yield tuple of LoadedStructures at indices [0, -1]
                - list[int]: Yield tuple of LoadedStructures at the given indices
                  into the row's structures list (e.g., [-2, -1] for penultimate
                  and last)
            num_workers: Deprecated / no-op. A previous "threaded" loader used
                ``ThreadPoolExecutor`` here on the assumption that
                ``parse_to_atom_array`` releases the GIL. It does not in
                practice — the atomworks parse path is GIL-bound, so the
                threaded loader measured ~5× *slower* than sequential (see
                memory ``atomworks_gil_threaded_slow.md``). ``num_workers > 0``
                is therefore treated identically to ``num_workers == 0``
                (sequential). For genuine parallelism, use a
                ``multiprocessing.Pool`` over ``parse_to_atom_array`` at the
                call site.

        Yields:
            LoadedStructure (if which="last" or "first")
            tuple[LoadedStructure, ...] (if which="all" or list[int])
        """
        if num_workers > 0:
            logger.warning(
                "get_data(num_workers=%d) is GIL-bound and not faster than "
                "sequential; falling through to sequential loading. Use a "
                "multiprocessing.Pool over parse_to_atom_array for real "
                "parallelism.",
                num_workers,
            )

        from tqdm import tqdm

        for row in tqdm(
            self._df.iter_rows(named=True), total=len(self._df), desc="get_data"
        ):
            yield self._yield_for_row(row, which=which, load=load, select=select)

    def get_prediction_inputs(
        self,
        *,
        which: "str | list[int]" = "last",
        load_context: bool = True,
        select: Optional[List[SELECTIONS]] = None,
    ):
        """Yield sequence-authoritative inputs for prediction/oracle blocks.

        Protein sequences come from the row when present.  If absent, they are
        read from the context structure for backward compatibility.  The last
        structure in the row is exposed only as context; it is never treated as
        the designed sequence artifact itself.
        """
        from tqdm import tqdm

        for row in tqdm(
            self._df.iter_rows(named=True),
            total=len(self._df),
            desc="get_prediction_inputs",
        ):
            structures = row.get(str(Column.STRUCTURES)) or []
            metrics = get_deserialized(row, Column.METRICS)
            context = None
            if structures:
                if which == "first":
                    context_index = 0
                elif which == "last":
                    context_index = -1
                elif isinstance(which, list) and len(which) == 1:
                    context_index = which[0]
                else:
                    raise ValueError(
                        "Prediction sequence inputs require which='first', "
                        "which='last', or a one-element index list."
                    )
                context = LoadedStructure.from_single(
                    structures[context_index],
                    metrics=metrics,
                    load=load_context,
                    select=select,
                    design_id=row.get(str(Column.DESIGN_ID)),
                    parent_design_id=row.get(str(Column.PARENT_DESIGN_ID)),
                )
            sequences = get_sequences_from_row(row)
            if not sequences and context is not None:
                if context.sequence is None:
                    context = context.load_array(select=select)
                sequences = {
                    str(chain): str(sequence)
                    for chain, sequence in (context.sequence or {}).items()
                    if sequence
                }
            if not sequences:
                raise ValueError(
                    f"Prediction input {row.get(str(Column.DESIGN_ID))!r} has "
                    "neither protein sequences nor a readable context structure."
                )
            design_id = row.get(str(Column.DESIGN_ID))
            if not design_id:
                design_id = context.basename if context is not None else None
            if not design_id:
                raise ValueError("Prediction input is missing design_id")
            yield PredictionInput(
                design_id=str(design_id),
                parent_design_id=row.get(str(Column.PARENT_DESIGN_ID)),
                sequences=sequences,
                metrics=metrics,
                context=context,
            )

    def _yield_for_row(
        self,
        row: dict,
        which: "str | list[int]",
        load: bool,
        select: Optional[List[SELECTIONS]],
    ):
        """Resolve the ``which`` selection for a single row into LoadedStructure(s).

        Single source of truth for the ``which`` dispatch shared by every
        ``get_data`` consumer (sequential and any parallel variant).
        """
        struct_list = row[Column.STRUCTURES]
        metrics = get_deserialized(row, Column.METRICS)

        sequence_only = row.get(str(Column.ARTIFACT_KIND)) == str(ArtifactKind.SEQUENCE)
        requests_logical_last = (
            which in ("last", "all")
            or isinstance(which, list)
            and any(i == -1 or i == len(struct_list) - 1 for i in which)
        )
        if sequence_only and requests_logical_last:
            raise SequenceOnlyInputError(
                f"Design {row.get(str(Column.DESIGN_ID))!r} is sequence-only. "
                "This block requires coordinates; fold it with a prediction "
                "block or run MPNN with output_mode='structures'. The retained "
                "parent structure is context, not the designed output."
            )
        if not struct_list:
            raise SequenceOnlyInputError(
                f"Design {row.get(str(Column.DESIGN_ID))!r} has no structure "
                "coordinates. Fold it with a prediction block first."
            )

        def _load(i):
            return LoadedStructure.from_single(
                struct_list[i],
                metrics=metrics,
                load=load,
                select=select,
                design_id=row.get(str(Column.DESIGN_ID)),
                parent_design_id=row.get(str(Column.PARENT_DESIGN_ID)),
            )

        if isinstance(which, list):
            return tuple(_load(i) for i in which)
        elif which == "last":
            return _load(-1)
        elif which == "first":
            return _load(0)
        elif which == "all":
            return tuple(_load(i) for i in [0, -1])
        else:
            raise ValueError(
                f"Unknown which value: {which}. Use 'last', 'first', 'all', "
                f"or a list of int indices (e.g., [0, -1])"
            )

    def filter(self, predicate: Callable[[dict], bool]) -> "PipelineDataHandler":
        """Filter rows by predicate on metrics.

        Args:
            predicate: Function taking metrics dict, returning bool (True = keep)

        Returns:
            New PipelineDataHandler with filtered rows
        """
        from chaperone.data.schema import get_deserialized

        kept_rows = []
        for row in self._df.iter_rows(named=True):
            metrics = get_deserialized(row, Column.METRICS)
            if predicate(metrics):
                kept_rows.append(row)

        return PipelineDataHandler(kept_rows)

    @property
    def names(self):
        """Get step names from the row with the longest structures list."""
        if self._df.height == 0:
            return []
        # Find row with longest structures list using Polars expressions
        idx = self._df.select(pl.col(Column.STRUCTURES).list.len().arg_max())[0, 0]
        longest = self._df[Column.STRUCTURES][idx]
        return [s[StructureColumn.NAME] for s in longest]

    def resolve_name(self, step):
        """Resolve a step reference against the structures present in the handler."""
        if step == "last":
            if not self.names:
                raise ValueError("Cannot resolve 'last' on an empty handler")
            return self.names[-1]
        if step == "first":
            if not self.names:
                raise ValueError("Cannot resolve 'first' on an empty handler")
            return self.names[0]
        if step not in self.names:
            raise ValueError(
                f"Step '{step}' not found in handler lineage. Available: {self.names}"
            )
        return step

    @property
    def latest_step_name(self) -> str | None:
        """Return the last structure name in the handler lineage, if any."""
        return self.names[-1] if self.names else None

    def as_single_step(self, step):
        """Return new handler with structures list pruned to only the specified step.

        Args:
            step: Step name, or "first"/"last" for first/last structure

        Returns:
            PipelineDataHandler with structures column containing only the matching structure
        """
        name = self.resolve_name(step)

        # Filter structures list to keep only the structure with matching name
        pruned_df = self._df.with_columns(
            pl.col(Column.STRUCTURES)
            .list.eval(
                pl.element().filter(
                    pl.element().struct.field(StructureColumn.NAME) == name
                )
            )
            .alias(Column.STRUCTURES)
        )
        pruned_df = pruned_df.filter(pl.col(Column.STRUCTURES).list.len() > 0)
        return PipelineDataHandler(pruned_df)

    def stash_structure_in_metrics(self, step) -> "PipelineDataHandler":
        """Move a structure from the lineage into metrics, preserving its path.

        Removes the named structure from each row's structures list and stores
        its path in the metrics dict under ``{step}.path``. This is useful when
        you need to remove an intermediate prediction from the lineage (so the
        next block doesn't pick it up as input) while retaining a reference to
        the file for later analysis.

        Args:
            step: Step name, or "first"/"last" for first/last structure.

        Returns:
            New PipelineDataHandler with the structure removed from lineage and
            its path saved in metrics.

        Example::

            # After af3_monomer + RMSD + filter, stash af3_monomer so that
            # af3_complex picks up the mpnn design as input instead:
            handler = handler.stash_structure_in_metrics("af3_monomer")
        """

        name = self.resolve_name(step)
        rows = []
        for row in self._df.iter_rows(named=True):
            structures = row[Column.STRUCTURES]
            metrics = get_deserialized(row, Column.METRICS) or {}

            stashed_path = None
            remaining = []
            for s in structures:
                if s[StructureColumn.NAME] == name:
                    stashed_path = s[StructureColumn.PATH]
                else:
                    remaining.append(s)

            if stashed_path is not None:
                metrics.setdefault(name, {})
                metrics[name]["path"] = stashed_path

            rows.append(_row_with_state(row, structures=remaining, metrics=metrics))

        return PipelineDataHandler(pl.DataFrame(rows, infer_schema_length=None))

    def with_metrics(
        self, values_by_key: dict, *, key: str = "design_id"
    ) -> "PipelineDataHandler":
        """Attach externally-computed metrics to each row, joined on ``key``.

        ``values_by_key`` maps each row's ``key`` value (``design_id`` by default,
        read from that row's metrics) to a metrics fragment merged into its
        metrics dict — e.g. ``{"d0": {"motif": {"rmsd": 1.2}}}`` makes
        ``motif.rmsd`` a first-class metric addressable by the filter DSL and by
        ``unpack``. Joining on ``key`` rather than row position is deliberate: the
        rows are not guaranteed to be in the caller's order.

        Raises if a row's key is missing from ``values_by_key`` (every row must be
        measured) so a partial map can never silently leave some rows un-updated.
        """
        rows = []
        for row in self._df.iter_rows(named=True):
            metrics = get_deserialized(row, Column.METRICS) or {}
            row_key = (
                row.get(str(Column.DESIGN_ID))
                if key == str(Column.DESIGN_ID)
                else metrics.get(key)
            )
            if row_key not in values_by_key:
                raise KeyError(
                    f"with_metrics: no value for {key}={row_key!r}; every row must "
                    "be measured (join is by key, never by position)"
                )
            merged = {**metrics, **values_by_key[row_key]}
            rows.append(
                _row_with_state(row, structures=row[Column.STRUCTURES], metrics=merged)
            )
        return PipelineDataHandler(pl.DataFrame(rows, infer_schema_length=None))

    def rewind_to(self, step) -> "PipelineDataHandler":
        """Truncate lineage back to a step, stashing everything after it as metrics.

        Keeps all structures up to and including ``step`` in the lineage.
        Every structure *after* ``step`` is removed from the lineage and its
        path is saved in metrics under ``{name}.path`` (same behaviour as
        :meth:`stash_structure_in_metrics`).  Metrics and row count are
        preserved.

        This is the primary tool for "evaluation sidecar" pipelines where
        you predict + measure + filter, then continue from an earlier
        design step.

        Args:
            step: Step name, or "first"/"last" for positional lookup.

        Returns:
            New PipelineDataHandler with truncated lineage and stashed paths.

        Example::

            h_designed = engine.run("mpnn")
            h_monomer  = engine.run([af3_monomer, rmsd, filter])

            # Continue from mpnn lineage; monomer metrics are kept
            engine.run(af3_complex, data=h_monomer.rewind_to("mpnn"))
        """
        import json

        name = self.resolve_name(step)

        rows = []
        for row in self._df.iter_rows(named=True):
            structures = row[Column.STRUCTURES]
            metrics = json.loads(row[Column.METRICS]) if row[Column.METRICS] else {}

            keep = []
            stash = []
            found = False
            for s in structures:
                if not found:
                    keep.append(s)
                    if s[StructureColumn.NAME] == name:
                        found = True
                else:
                    stash.append(s)

            if not found:
                # Step not in this row — keep as-is
                keep = structures
                stash = []

            for s in stash:
                s_name = s[StructureColumn.NAME]
                s_path = s[StructureColumn.PATH]
                if s_path is not None:
                    metrics.setdefault(s_name, {})
                    metrics[s_name]["path"] = s_path

            rows.append(_row_with_state(row, structures=keep, metrics=metrics))

        return PipelineDataHandler(pl.DataFrame(rows, infer_schema_length=None))

    # ---------------------------------------------------------------------
    # Merge Operations
    # ---------------------------------------------------------------------

    @classmethod
    def merge(
        cls,
        base: Optional["PipelineDataHandler"],
        new: "PipelineDataHandler",
        ignore_capitalization=False,
        how: str = "inner",
    ) -> "PipelineDataHandler":
        """Merge of two data handlers.

        Args:
            base: Parent handler (left side of join)
            new: Child handler (right side of join)
            ignore_capitalization: If True, ignore case when matching paths
            how: Join type - "inner" (default, backlinking merge) or
                 "left" (preserves all base rows)

        Merged based on the paths of the last structure in both dataframes
        """
        if new is None or new._df is None or new._df.height == 0:
            if how == "left":
                return base
            # An inner join against an empty child yields nothing. Returning
            # ``base`` would pass parent rows through as if the block had
            # produced them, so a block that emitted no output is recorded as a
            # success and its chunk is never retried.
            if base is None or base._df is None:
                return new
            if base._df.height:
                logger.warning(
                    "Inner merge: child handler is empty; dropping %d parent row(s). "
                    "The upstream block produced no output for these inputs.",
                    base._df.height,
                )
            # ``clear()`` keeps the schema, which concat_outputs needs.
            return cls(base._df.clear())
        if base is None or base._df is None or base._df.height == 0:
            return new
        assert (
            Column.STRUCTURES in new._df.columns
        ), "New handler missing STRUCTURES column."
        if Column.STRUCTURES not in base._df.columns:
            # Dummy base provided
            return new

        if how == "inner":
            assert (
                Column.STRUCTURES in new._df.columns
            ), "New handler missing STRUCTURES column."
            if Column.STRUCTURES not in base._df.columns:
                return new

        # Fast path: Polars-native merge for large legacy structure DataFrames.
        # Logical design lineage requires exact parent_design_id handling that
        # the path-only native merge intentionally does not implement.
        # Skip native merge when path_handle backlinking is needed
        sequence_lineage = base.has_sequence_data or new.has_sequence_data
        if len(base._df) > 100 and len(new._df) > 100 and not sequence_lineage:
            has_path_handles = False
            try:
                ph_expr = (
                    pl.col(Column.STRUCTURES)
                    .list.last()
                    .struct.field(StructureColumn.PATH_HANDLE)
                )
                if new._df.select(ph_expr.is_not_null().any()).item():
                    has_path_handles = True
            except Exception:
                pass
            if not has_path_handles:
                result = cls._try_native_merge(base, new, how=how)
                if result is not None:
                    return result

        strip_fn = functools.partial(
            strip_name,
            ignore_punctuated_suffixes=True,
            ignore_capitalization=ignore_capitalization,
        )

        if how == "left":
            # Overlay: iterate base rows, lookup in overlay (left join)
            overlay_lookup: Dict[str, dict] = {}
            overlay_ids: Dict[str, dict] = {}
            for row in new._df.iter_rows(named=True):
                row_id = row.get(str(Column.DESIGN_ID))
                if row_id:
                    overlay_ids[str(row_id)] = row
                structs = row.get(Column.STRUCTURES, [])
                if structs:
                    key = _merge_key_for_struct(structs[-1], strip_fn)
                    overlay_lookup[key] = row

            data = []
            for row in base._df.iter_rows(named=True):
                row_id = row.get(str(Column.DESIGN_ID))
                overlay_row = overlay_ids.get(str(row_id)) if row_id else None
                structs = row.get(Column.STRUCTURES, [])
                if overlay_row is None and not structs:
                    data.append(row)
                    continue
                if overlay_row is None:
                    key = _merge_key_for_struct(structs[-1], strip_fn)
                    overlay_row = overlay_lookup.get(key)
                if overlay_row is not None:
                    data.append(merge_rows(row, overlay_row))
                else:
                    data.append(row)
            return cls(data)

        # Inner join: iterate new rows, backlink to base
        base_rows = list(base._df.iter_rows(named=True))
        lookup = {
            strip_fn(row[Column.STRUCTURES][-1][StructureColumn.PATH]): row
            for row in base_rows
            if row[Column.STRUCTURES]
        }
        exact_path_groups: Dict[str, list[dict]] = {}
        id_groups: Dict[str, list[dict]] = {}
        for base_row in base_rows:
            base_structures = base_row.get(str(Column.STRUCTURES)) or []
            if base_structures:
                path = base_structures[-1].get(str(StructureColumn.PATH))
                if path:
                    exact_path_groups.setdefault(str(path), []).append(base_row)
            row_id = base_row.get(str(Column.DESIGN_ID))
            if row_id:
                id_groups.setdefault(str(row_id), []).append(base_row)
        exact_path_lookup = {
            key: rows[0] for key, rows in exact_path_groups.items() if len(rows) == 1
        }
        id_lookup = {key: rows[0] for key, rows in id_groups.items() if len(rows) == 1}

        data = []
        for row in new._df.iter_rows(named=True):
            parent_row = None
            design_id = row.get(str(Column.DESIGN_ID))
            parent_id = row.get(str(Column.PARENT_DESIGN_ID))

            # Compiled blocks may rename inputs to avoid basename collisions.
            # Their path_handle records the exact source artifact and is more
            # specific than either a basename or a campaign-local design ID.
            new_structures = row.get(str(Column.STRUCTURES)) or []
            path_handle = (
                new_structures[-1].get(str(StructureColumn.PATH_HANDLE))
                if new_structures
                else None
            )
            if path_handle:
                parent_row = exact_path_lookup.get(str(path_handle))

            # An inline metric/update serializes the current logical design
            # with both its design_id and its older lineage parent.  Overlay
            # that exact current row first.  Only a genuinely new child (no
            # exact current design in base) backlinks through parent_design_id.
            if parent_row is None and design_id:
                parent_row = id_lookup.get(str(design_id))

            if parent_row is None and parent_id:
                parent_row = id_lookup.get(str(parent_id))
                if parent_row is None:
                    if str(parent_id) in id_groups:
                        raise ValueError(
                            f"Cannot backlink child {design_id!r}: parent_design_id "
                            f"{parent_id!r} matches {len(id_groups[str(parent_id)])} "
                            "rows and no unique path_handle was supplied."
                        )
                    raise ValueError(
                        f"No parent design {parent_id!r} found for child "
                        f"{design_id!r}."
                    )

            # Preserve the legacy name-based fallback for outputs that do not
            # declare lineage explicitly.  This intentionally comes after the
            # exact-id and declared-parent contracts above.
            if parent_row is None and design_id:
                try:
                    parent_row = _match_row_in_lookup(
                        key=str(design_id),
                        lookup=id_lookup,
                        ensure_single=True,
                    )
                except ValueError:
                    pass

            if parent_row is None:
                structures = row.get(Column.STRUCTURES) or []
                if not structures or not lookup:
                    raise ValueError(
                        f"Cannot backlink child {row.get(str(Column.DESIGN_ID))!r}: "
                        "it has no matching parent_design_id or structure path."
                    )
                key = _merge_key_for_struct(structures[-1], strip_fn)
                try:
                    parent_row = _match_row_in_lookup(
                        key=key, lookup=lookup, ensure_single=True
                    )
                except ValueError as e:
                    if ignore_capitalization:
                        raise e
                    return cls.merge(base, new, ignore_capitalization=True)

            merged_row = merge_rows(parent_row, row)
            data.append(merged_row)

        return cls(data)

    def has_metric(self, metric_name: str) -> bool:
        """Check if a metric key exists in the metrics column of any row.

        Used for cache detection: if a block's name already appears as a
        metric key, the block has already been run on this data.
        """
        import json

        if Column.METRICS not in self._df.columns:
            return False
        for metrics_str in self._df[str(Column.METRICS)].to_list():
            if metrics_str:
                metrics = json.loads(metrics_str)
                if metric_name in metrics:
                    return True
        return False

    @classmethod
    def _try_native_merge(cls, base, new, how="inner"):
        """Unified Polars-native merge for both inner and left join styles.

        Uses Polars join + string ops instead of Python iteration.
        Returns None if not applicable (triggers Python fallback).
        """
        try:
            # NaN/Infinity guard: the native merge concatenates the two metric
            # JSON blobs as raw strings, so a literal ``NaN``/``Infinity`` token
            # in either blob (from an unsanitized upstream write) would survive
            # into the merged blob and permanently invalidate the row's JSON
            # (polars ``str.json_path_match`` then silently returns null → the
            # row silently fails every filter). Fall back to the Python merge,
            # which re-serializes through ``dump_metrics`` and sanitizes.
            if _metrics_blobs_have_nonfinite(base._df) or _metrics_blobs_have_nonfinite(
                new._df
            ):
                logger.debug(
                    "Native merge (%s) skipped: metrics blob contains NaN/Infinity; "
                    "using Python merge to re-sanitize",
                    how,
                )
                return None

            key_expr = (
                pl.col(Column.STRUCTURES).list.last().struct.field(StructureColumn.PATH)
            )
            base_df = base._df.with_columns(key_expr.alias("_mk"))
            new_df = new._df.with_columns(key_expr.alias("_mk"))

            if how == "left":
                new_df = new_df.unique(subset=["_mk"], keep="last")
                new_df = new_df.select(
                    "_mk", str(Column.METRICS), str(Column.STRUCTURES)
                )

            joined = base_df.join(new_df, on="_mk", how=how, suffix="_new")

            # This shortcut only combines metrics. Structure/annotation changes
            # need merge_rows() even when the paths match (e.g. annotate).
            if not joined.select(
                pl.col(str(Column.STRUCTURES))
                .eq_missing(pl.col(f"{Column.STRUCTURES}_new"))
                .all()
            ).item():
                return None

            if how == "inner" and len(joined) != len(new._df):
                return None
            if how == "left" and len(joined) > len(base_df):
                logger.debug(
                    "Overlay join produced more rows than base, falling back to Python"
                )
                return None

            m = str(Column.METRICS)
            mn = f"{m}_new"

            if how == "left":
                merged_metrics = (
                    pl.when(pl.col(mn).is_null())
                    .then(pl.col(m))
                    .when((pl.col(m) == "{}") | pl.col(m).is_null())
                    .then(pl.col(mn))
                    .when((pl.col(mn) == "{}") | pl.col(mn).is_null())
                    .then(pl.col(m))
                    .otherwise(
                        pl.col(m).str.strip_suffix("}")
                        + pl.lit(", ")
                        + pl.col(mn).str.strip_prefix("{")
                    )
                    .alias(m)
                )
            else:
                merged_metrics = (
                    pl.when((pl.col(m) == "{}") | pl.col(m).is_null())
                    .then(pl.col(mn))
                    .when((pl.col(mn) == "{}") | pl.col(mn).is_null())
                    .then(pl.col(m))
                    .otherwise(
                        pl.col(m).str.strip_suffix("}")
                        + pl.lit(", ")
                        + pl.col(mn).str.strip_prefix("{")
                    )
                    .alias(m)
                )

            result_df = joined.select(
                merged_metrics if column == m else pl.col(column)
                for column in base._df.columns
            )

            logger.info(
                f"Native merge ({how}): {len(result_df)} rows merged via Polars join"
            )
            return cls(result_df)
        except Exception as e:
            logger.debug(
                f"Native merge ({how}) not applicable, falling back to Python: {e}"
            )
            return None


_NONFINITE_JSON_RE = re.compile(r"-?\b(?:NaN|Infinity)\b")


def _metrics_blobs_have_nonfinite(df: pl.DataFrame) -> bool:
    """Return True if any metrics JSON blob contains a non-finite JSON token.

    ``json.dumps(..., allow_nan=True)`` (the historical default at unsanitized
    sites) emits bare ``NaN``/``Infinity``/``-Infinity`` tokens, which are
    invalid JSON. Used to gate the string-concatenating native merge.
    """
    m = str(Column.METRICS)
    if m not in df.columns:
        return False
    for blob in df[m].to_list():
        if blob and _NONFINITE_JSON_RE.search(blob):
            return True
    return False


def _merge_key_for_struct(struct: dict, strip_fn) -> str:
    """Extract the merge key from a structure dict.

    Uses ``path_handle`` as fallback when ``path`` is absent, allowing rows
    to backlink to a parent even when the output path doesn't follow the
    parent naming convention.
    """
    path = struct.get(StructureColumn.PATH)
    if not path:
        path = struct.get(StructureColumn.PATH_HANDLE)
    if not path:
        raise ValueError(f"Structure has neither path nor path_handle: {struct}")
    return strip_fn(path)


def _match_row_in_lookup(key, lookup, ensure_single=True):
    # Fast path: exact match (O(1)) — handles update blocks where paths are identical
    exact = lookup.get(key)
    if exact is not None:
        return exact

    # Slow path: prefix matching for child→parent (e.g. design1_seq0 → design1)
    # Track (match_length, matched_row) so we can disambiguate by specificity.
    matches = []
    for match, matched_row in lookup.items():
        if key.startswith(match) and (
            len(match) == len(key) or not key[len(match) : len(match) + 1].isdigit()
        ):
            if not ensure_single:
                return matched_row
            matches.append((len(match), matched_row))
    if len(matches) == 1:
        return matches[0][1]
    if len(matches) > 1:
        # Multiple prefix matches: pick the longest (most specific) parent.
        # This handles cases where one parent name is a prefix of another
        # (e.g. "...-keep_boundary" vs "...-keep_boundary-partial").
        return max(matches, key=lambda x: x[0])[1]

    # Suffix matching: parent key is a suffix of child key at a word boundary
    # Handles blocks that prepend a prefix (e.g. partial_diffusion_chunk0_design1 → design1)
    matches = []
    for match, matched_row in lookup.items():
        sep = len(key) - len(match)
        if sep > 0 and key.endswith(match) and key[sep - 1] == "_":
            if not ensure_single:
                return matched_row
            matches.append((len(match), matched_row))
    if len(matches) == 1:
        return matches[0][1]
    if len(matches) > 1:
        return max(matches, key=lambda x: x[0])[1]

    raise ValueError(
        f"Expected single match for key {key}, found {len(matches)} matches. Source keys (len({len(lookup.keys())})): {list(lookup.keys())}"
    )


def _write_debug_csv(df: pl.DataFrame, path: Path) -> None:
    """Write DataFrame to CSV, serializing nested columns to JSON strings."""
    import json

    rows = []
    for row in df.iter_rows(named=True):
        new_row = {}
        for k, v in row.items():
            # Serialize any nested types (dict, list) to JSON
            if isinstance(v, (dict, list)):
                new_row[k] = json.dumps(v)
            else:
                new_row[k] = v
        rows.append(new_row)
    pl.DataFrame(rows, infer_schema_length=None).write_csv(path)


def _csv_safe_value(value):
    """Return a scalar suitable for Polars CSV, JSON-encoding nested values."""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def _flatten_dict(d: dict, sep: str = ".") -> dict:
    """Flatten a nested dict into dot-separated keys (one level deep).

    Only flattens the first level to preserve dots in original metric keys.
    This matches the metrics structure where top-level keys are block names
    and nested keys can contain dots (e.g., "bkbn_align.bkbn_rmsd").

    Args:
        d: Nested dictionary to flatten (expected to be one level deep)
        sep: Separator between nested keys

    Returns:
        Flat dictionary with dot-separated keys

    Example:
        {"rmsd": {"bkbn_align.bkbn_rmsd": 1.2}} -> {"rmsd.bkbn_align.bkbn_rmsd": 1.2}
    """
    items = {}
    for k, v in d.items():
        if isinstance(v, dict):
            for inner_k, inner_v in v.items():
                items[f"{k}{sep}{inner_k}"] = inner_v
        else:
            items[k] = v
    return items


def _unflatten_dict(d: dict, sep: str = ".") -> dict:
    """Unflatten a dot-separated dict back into nested structure.

    Only splits on the first separator to preserve dots in original metric keys.
    This matches the metrics structure where top-level keys are block names
    and nested keys can contain dots (e.g., "rmsd.bkbn_align.bkbn_rmsd").

    Args:
        d: Flat dictionary with dot-separated keys
        sep: Separator used in keys

    Returns:
        Nested dictionary (one level deep)

    Example:
        {"rmsd.bkbn_align.bkbn_rmsd": 1.2} -> {"rmsd": {"bkbn_align.bkbn_rmsd": 1.2}}
    """
    result = {}
    for key, value in d.items():
        if sep in key:
            # Only split on first separator
            first_part, rest = key.split(sep, 1)
            if first_part not in result:
                result[first_part] = {}
            result[first_part][rest] = value
        else:
            result[key] = value
    return result


def _detect_structure_names(columns: list) -> list:
    """Detect structure names from unpacked column names.

    Looks for columns matching the pattern <name>.path to identify structure names.
    Returns names in the order they appear (preserving structure list order).

    Args:
        columns: List of column names

    Returns:
        List of structure names in order

    Example:
        ["rfd3.path", "rfd3.metadata.task", "mpnn.path"] -> ["rfd3", "mpnn"]
    """
    names = []
    seen = set()
    for col in columns:
        if col.endswith(".path"):
            name = col[:-5]  # Remove ".path"
            if name not in seen:
                names.append(name)
                seen.add(name)
    return names


def _get_structure_column_names(structure_names: list, columns: list[str]) -> set:
    """Get all structure-related column names for the given structure names.

    Args:
        structure_names: List of structure names

    Returns:
        Set of column names that belong to structures (path and metadata.*)
    """
    cols = set()
    for name in structure_names:
        cols.add(f"{name}.path")
        cols.add(f"{name}.stored")
        cols.add(f"{name}.store_path")
        cols.add(f"{name}.storage_format")
        cols.update(c for c in columns if c.startswith(f"{name}.metadata."))
    return cols


def _reconstruct_structures(row: dict, structure_names: list) -> list:
    """Reconstruct the structures list from unpacked columns.

    Args:
        row: Dict of column name -> value (from iter_rows(named=True))
        structure_names: List of structure names in order

    Returns:
        List of structure dicts matching the schema

    Example:
        Row with columns: rfd3.path, rfd3.metadata.task, rfd3.metadata.motif
        Returns: [{"name": "rfd3", "path": ..., "metadata": {"task": ..., "motif": ...}}]
    """
    structures = []
    for name in structure_names:
        path_col = f"{name}.path"
        path_val = row.get(path_col)
        if path_val is None:
            # Skip structures that don't exist in this row
            continue

        # Extract metadata from <name>.metadata.* columns
        metadata_prefix = f"{name}.metadata."
        metadata = {}
        for col, val in row.items():
            if col.startswith(metadata_prefix):
                meta_key = col[len(metadata_prefix) :]
                # Preserve None as None for metadata fields
                # Handle both scalar and array values (lists are valid metadata)
                if isinstance(val, (list, np.ndarray)):
                    metadata[meta_key] = val if len(val) > 0 else None
                else:
                    metadata[meta_key] = val if val is not None else None

        structures.append(
            {
                StructureColumn.NAME: name,
                StructureColumn.PATH: path_val,
                StructureColumn.PATH_HANDLE: None,
                StructureColumn.STORE_PATH: row.get(f"{name}.store_path"),
                StructureColumn.STORAGE_FORMAT: row.get(f"{name}.storage_format"),
                StructureColumn.METADATA: _ensure_metadata_fields(metadata),
            }
        )

    return structures
