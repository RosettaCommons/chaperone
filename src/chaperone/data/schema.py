"""
Defines SCHEMA and related enums for data processing.
Everything atomworks.
"""

from enum import StrEnum
import json
import math
from pathlib import Path

import numpy as np
import polars as pl
from biotite.structure import AtomArray
from chaperone.data.components import (
    fetch_mask_from_component,
    ComponentValidationError,
)
import logging

logger = logging.getLogger(__name__)

###########################################################################
# Schema definition
###########################################################################


class DesignMetadata(StrEnum):
    """Standard metadata field names for residue annotations."""

    TASK = "task"
    """
    Design task type; when running sweeps of parameters this field can be used to track the original task. 
    - For RFD3 these are equal to the keys in your input json
    """

    ORIGINAL_INPUT = "original_input"
    """Path to the original input structure for the design. Used for tracking source atoms."""

    DIFFUSED_INDEX_MAP = "diffused_index_map"
    """Mapping from diffused to original residue indices in the original_input."""

    MOTIF = "motif"
    """Residue indices that belong to motifs. 
    *Motif* just means anything that is of interest from diffusion, could include the indices for the fixed motif residues, the guideposts or the target binder
    """

    LIGAND = "ligand"
    """Residue indices or name for the ligand in this structure."""

    # Calculated by annotate:
    INTERFACE = "interface"
    """Residue indices belonging to the interface"""

    HBOND = "hbond"
    """Residue indices involved in hydrogen bonds"""

    AROMATICS_NEAR_LIGAND = "aromatics_near_ligand"
    """Aromatic residues near ligand binding site"""
    # ... add more as needed ...


# Allowed selections
SELECTABLE_METADATA_FIELDS = {
    DesignMetadata.MOTIF,
    DesignMetadata.INTERFACE,
    DesignMetadata.LIGAND,
    DesignMetadata.HBOND,
    DesignMetadata.AROMATICS_NEAR_LIGAND,
}


class StorageFormat(StrEnum):
    """How a structure's files are persisted."""

    RAW = "raw"
    CATCIF = "catcif"  # Concatenated-CIF archive (single .catcif file)


class ArtifactKind(StrEnum):
    """Logical representation carried by a pipeline row."""

    STRUCTURE = "structure"
    SEQUENCE = "sequence"


class StructureColumn(StrEnum):
    NAME = "name"
    PATH = "path"
    PATH_HANDLE = "path_handle"
    STORE_PATH = "store_path"
    STORAGE_FORMAT = "storage_format"
    METADATA = "metadata"


class Column(StrEnum):
    """Standard column names for data processing schema."""

    STRUCTURES = "structures"
    PARENT_STRUCTURE = "parent_structure"
    METRICS = "metrics"
    DESIGN_ID = "design_id"
    PARENT_DESIGN_ID = "parent_design_id"
    ARTIFACT_KIND = "artifact_kind"
    SEQUENCES = "sequences"


STRUCTURE_METADATA_TYPE = pl.Struct(
    {
        # High-level features (nullable strings)
        str(DesignMetadata.TASK): pl.Utf8,
        str(DesignMetadata.ORIGINAL_INPUT): pl.Utf8,
        str(DesignMetadata.DIFFUSED_INDEX_MAP): pl.Utf8,  # json string
        # Residue annotations all like ["A10", "B5"] (list of components)
        str(DesignMetadata.MOTIF): pl.List(pl.Utf8),
        str(DesignMetadata.LIGAND): pl.List(pl.Utf8),  # e.g. ["LG1", "LG2"]
        str(DesignMetadata.INTERFACE): pl.List(pl.Utf8),
        str(DesignMetadata.HBOND): pl.List(pl.Utf8),
        str(DesignMetadata.AROMATICS_NEAR_LIGAND): pl.List(pl.Utf8),
    }
)
STRUCTURE_TYPE = pl.Struct(
    {
        str(StructureColumn.NAME): pl.Utf8,  # Step name: "rfd3", "mpnn", "af3"
        str(StructureColumn.PATH): pl.Utf8,  # Single file path
        str(
            StructureColumn.PATH_HANDLE
        ): pl.Utf8,  # Temporary backlinking key (stripped on merge)
        str(
            StructureColumn.STORE_PATH
        ): pl.Utf8,  # Archive path (set after store())
        str(
            StructureColumn.STORAGE_FORMAT
        ): pl.Utf8,  # "raw" | "catcif" (see StorageFormat)
        str(StructureColumn.METADATA): STRUCTURE_METADATA_TYPE,
    }
)

SCHEMA = {
    str(Column.STRUCTURES): pl.List(STRUCTURE_TYPE),
    str(Column.METRICS): pl.Utf8,
    str(Column.DESIGN_ID): pl.Utf8,
    str(Column.PARENT_DESIGN_ID): pl.Utf8,
    str(Column.ARTIFACT_KIND): pl.Utf8,
    # JSON mapping of chain id -> one-letter protein sequence. JSON keeps the
    # schema stable when different rows carry different chain identifiers.
    str(Column.SEQUENCES): pl.Utf8,
}

# Example entry:
# example = {
#     "structures": [{
#             "name": "rfd3",
#             "path": "/path/to/rfd3/design1.pdb",
#             "metadata": {
#                 "task": "enzyme",
#                 "original_input": "path/to/input/input1.pdb",
#                 "motif": ["A10", "A11", "A12"],
#                 "ligand": ["LG1"],
#                 "interface": ["A10", "B5", "B6"],
#                 "hbond": ["A11", "B5"],
#                 "aromatics_near_ligand": [],
#             },
#         }],
#     "metrics": '{"rmsd": 1.2, "binding_energy": -7.5}',
# }


###########################################################################
# Helpers for creating dataframes
###########################################################################


def create_path_entries(paths: list, name: str):
    return [make_row(name=name, path=path) for path in paths]


def _ensure_metadata_fields(metadata: dict | None) -> dict:
    """Ensure metadata dict has all expected fields (with None for missing).

    Also normalizes list-typed fields (SELECTABLE_METADATA_FIELDS) so that
    bare strings are wrapped in a list. This prevents polars schema conflicts
    when concatenating handlers from different blocks.
    """
    base = {str(field): None for field in DesignMetadata}
    if metadata:
        base.update(metadata)
    # Normalize: list-typed fields that are bare strings -> [string]
    for field in SELECTABLE_METADATA_FIELDS:
        v = base.get(str(field))
        if isinstance(v, str):
            base[str(field)] = [v]
    return base


def _sanitize_for_json(obj):
    """Replace NaN/Infinity with None so json.dumps produces valid JSON."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def dump_metrics(metrics: dict | None) -> str:
    """Single serialization seam for the metrics JSON blob.

    Always sanitizes NaN/Infinity to ``None`` before ``json.dumps`` so the
    resulting string is valid JSON (invalid JSON makes polars
    ``str.json_path_match`` silently return null, which makes a row silently
    fail every filter). All metric serialization sites should route through
    this function.
    """
    return json.dumps(_sanitize_for_json(metrics or {}))


def dump_sequences(sequences: dict[str, str] | None) -> str | None:
    """Serialize a chain-qualified protein sequence map."""
    if sequences is None:
        return None
    return json.dumps({str(k): str(v) for k, v in sequences.items()}, sort_keys=True)


def get_sequences_from_row(row: dict) -> dict[str, str]:
    """Return the row's chain-qualified sequence map, if present."""
    value = row.get(Column.SEQUENCES)
    if isinstance(value, str):
        return json.loads(value)
    return dict(value or {})


def make_row(
    name: str,
    path: str | Path,
    metadata: dict = None,
    metrics: dict = None,
    path_handle: str | Path | None = None,
    as_metric: bool = False,
    design_id: str | None = None,
    parent_design_id: str | None = None,
    artifact_kind: ArtifactKind | str | None = None,
    sequences: dict[str, str] | None = None,
):
    """Create a single row dict for the schema.

    Args:
        path_handle: Temporary backlinking key. When set, merge will use this
            to match the row to a parent even if *path* doesn't follow the
            parent naming convention.  Stripped after merge.
        as_metric: If True, store the path as a path_handle only and embed it
            in the metrics dict under ``name``.  Useful for oracle blocks run
            in "metric-only" mode (``compile_as_metric``).
    """
    if as_metric:
        metrics = metrics or {}
        block_metrics = metrics.get(name, {})
        block_metrics["path"] = str(path)
        metrics[name] = block_metrics
        path_handle = str(path) if path_handle is None else path_handle
        path = None

    return {
        str(Column.STRUCTURES): [
            {
                str(StructureColumn.NAME): name,
                str(StructureColumn.PATH): str(path)
                if path is not None
                else None,  # Ensure path is string for Polars
                str(StructureColumn.PATH_HANDLE): str(path_handle)
                if path_handle is not None
                else None,
                str(StructureColumn.STORE_PATH): None,
                str(StructureColumn.STORAGE_FORMAT): None,
                str(StructureColumn.METADATA): _ensure_metadata_fields(metadata),
            }
        ],
        str(Column.METRICS): dump_metrics(metrics),
        str(Column.DESIGN_ID): design_id,
        str(Column.PARENT_DESIGN_ID): parent_design_id,
        str(Column.ARTIFACT_KIND): (
            str(artifact_kind)
            if artifact_kind is not None
            else (None if as_metric else str(ArtifactKind.STRUCTURE))
        ),
        str(Column.SEQUENCES): dump_sequences(sequences),
    }


def make_sequence_row(
    *,
    design_id: str,
    sequences: dict[str, str],
    parent_design_id: str | None = None,
    context_path: str | Path | None = None,
    context_name: str = "context",
    context_metadata: dict | None = None,
    metrics: dict | None = None,
) -> dict:
    """Create a sequence-native row with optional structure context."""
    structures = []
    if context_path is not None:
        structures = make_row(
            name=context_name,
            path=context_path,
            metadata=context_metadata,
        )[str(Column.STRUCTURES)]
    return {
        str(Column.STRUCTURES): structures,
        str(Column.METRICS): dump_metrics(metrics),
        str(Column.DESIGN_ID): str(design_id),
        str(Column.PARENT_DESIGN_ID): (
            str(parent_design_id) if parent_design_id is not None else None
        ),
        str(Column.ARTIFACT_KIND): str(ArtifactKind.SEQUENCE),
        str(Column.SEQUENCES): dump_sequences(sequences),
    }


def get_deserialized(row, key) -> dict:
    """Retrieve key from row and deserialize if json string"""
    value = row.get(key)
    if isinstance(value, str):
        return json.loads(value)
    return value or {}


def _merge_metadata(parent_meta: dict, child_meta: dict) -> dict:
    """Merge metadata dicts, inheriting parent values where child has None."""
    parent_meta = parent_meta or {}
    child_meta = child_meta or {}
    merged = {}
    for field in DesignMetadata:
        child_val = child_meta.get(str(field))
        parent_val = parent_meta.get(str(field))
        merged[str(field)] = child_val if child_val is not None else parent_val
    return merged


def _find_structure_index(structures: list, name: str, path: str) -> int:
    """Find index of structure with matching name and path, or -1 if not found."""
    for i, s in enumerate(structures):
        if s.get(StructureColumn.NAME) == name and s.get(StructureColumn.PATH) == path:
            return i
    return -1


def _strip_path_handle(struct: dict) -> dict:
    """Return a copy of *struct* with ``path_handle`` removed (set to None)."""
    out = dict(struct)
    out[StructureColumn.PATH_HANDLE] = None
    return out


def merge_rows(
    *rows: dict,
):
    """
    Merge two or more rows by folding left. Handles two cases per pair:

    1. APPEND: Child has different (name, path) than parent structures
       → Child structure is added to the structures list

    2. UPDATE: Child has same (name, path) as an existing parent structure
       → Existing structure is updated in place (metadata merged, not duplicated)

    In both cases:
    - Child metadata inherits None fields from parent's last structure
    - Metrics from both rows are merged

    Example (APPEND case):
        parent: structures=[{name: "rfd3", path: "design1.pdb", ...}]
        child:  structures=[{name: "mpnn", path: "design1_seq0.pdb", ...}]
        output: structures=[{name: "rfd3", ...}, {name: "mpnn", ...}]

    Example (UPDATE case):
        parent: structures=[{name: "rfd3", path: "design1.pdb", metadata: {task: "enzyme"}}]
        child:  structures=[{name: "rfd3", path: "design1.pdb", metadata: {motif: ["A10"]}}]
        output: structures=[{name: "rfd3", path: "design1.pdb", metadata: {task: "enzyme", motif: ["A10"]}}]
    """
    if len(rows) < 2:
        raise ValueError("merge_rows requires at least 2 rows")

    result = rows[0]
    for child_row in rows[1:]:
        result = _merge_two_rows(result, child_row)
    return result


def _merge_two_rows(parent_row: dict, child_row: dict) -> dict:
    """Merge a single parent-child pair."""
    parent_structures = list(parent_row.get(Column.STRUCTURES) or [])
    child_structures = child_row.get(Column.STRUCTURES) or []

    # Get last parent's metadata for inheritance
    parent_metadata = {}
    if parent_structures:
        parent_metadata = parent_structures[-1].get(StructureColumn.METADATA) or {}

    # Process each child structure; strip path_handle from inherited parents
    merged_structures = [_strip_path_handle(s) for s in parent_structures]

    for child_struct in child_structures:
        child_name = child_struct.get(StructureColumn.NAME)
        child_path = child_struct.get(StructureColumn.PATH)
        child_meta = child_struct.get(StructureColumn.METADATA) or {}

        # METRIC-ONLY case: path is None → only contribute metrics, no structure entry
        if child_path is None:
            continue

        # Check if this structure already exists in parent
        existing_idx = _find_structure_index(merged_structures, child_name, child_path)

        if existing_idx >= 0:
            # UPDATE case: same (name, path) exists - merge metadata in place
            existing_meta = (
                merged_structures[existing_idx].get(StructureColumn.METADATA) or {}
            )
            merged_meta = _merge_metadata(existing_meta, child_meta)
            merged_structures[existing_idx] = {
                StructureColumn.NAME: child_name,
                StructureColumn.PATH: child_path,
                StructureColumn.PATH_HANDLE: None,
                StructureColumn.STORE_PATH: child_struct.get(StructureColumn.STORE_PATH)
                or merged_structures[existing_idx].get(StructureColumn.STORE_PATH),
                StructureColumn.STORAGE_FORMAT: child_struct.get(
                    StructureColumn.STORAGE_FORMAT
                )
                or merged_structures[existing_idx].get(StructureColumn.STORAGE_FORMAT),
                StructureColumn.METADATA: merged_meta,
            }
        else:
            # APPEND case: new structure - inherit from parent and append
            merged_meta = _merge_metadata(parent_metadata, child_meta)
            merged_structures.append(
                {
                    StructureColumn.NAME: child_name,
                    StructureColumn.PATH: child_path,
                    StructureColumn.PATH_HANDLE: None,
                    StructureColumn.STORE_PATH: child_struct.get(
                        StructureColumn.STORE_PATH
                    ),
                    StructureColumn.STORAGE_FORMAT: child_struct.get(
                        StructureColumn.STORAGE_FORMAT
                    ),
                    StructureColumn.METADATA: merged_meta,
                }
            )

    # Merge metrics from both rows
    merged_metrics = {
        **get_deserialized(parent_row, Column.METRICS),
        **get_deserialized(child_row, Column.METRICS),
    }

    child_kind = child_row.get(Column.ARTIFACT_KIND)
    child_design_id = child_row.get(Column.DESIGN_ID)
    child_sequences = child_row.get(Column.SEQUENCES)
    if child_kind is None:
        artifact_kind = parent_row.get(Column.ARTIFACT_KIND)
        design_id = parent_row.get(Column.DESIGN_ID)
        parent_design_id = parent_row.get(Column.PARENT_DESIGN_ID)
        sequences = parent_row.get(Column.SEQUENCES)
    elif child_design_id is None:
        # Structure-producing blocks written before the sequence schema do not
        # stamp a design id.  They advance the representation of the same
        # logical design, so preserve its lineage and sequence payload.
        artifact_kind = child_kind
        design_id = parent_row.get(Column.DESIGN_ID)
        parent_design_id = parent_row.get(Column.PARENT_DESIGN_ID)
        sequences = (
            child_sequences
            if child_sequences is not None
            else parent_row.get(Column.SEQUENCES)
        )
    else:
        artifact_kind = child_kind
        design_id = child_design_id
        parent_design_id = child_row.get(Column.PARENT_DESIGN_ID)
        if parent_design_id is None:
            if design_id != parent_row.get(Column.DESIGN_ID):
                parent_design_id = parent_row.get(Column.DESIGN_ID)
            else:
                parent_design_id = parent_row.get(Column.PARENT_DESIGN_ID)
        sequences = (
            child_sequences
            if child_sequences is not None
            else parent_row.get(Column.SEQUENCES)
        )

    return {
        Column.STRUCTURES: merged_structures,
        Column.METRICS: dump_metrics(merged_metrics),
        Column.DESIGN_ID: design_id,
        Column.PARENT_DESIGN_ID: parent_design_id,
        Column.ARTIFACT_KIND: artifact_kind,
        Column.SEQUENCES: sequences,
    }


###########################################################################
# Selection enum for both alignment and RMSD targets
###########################################################################

_NUCLEOTIDE_RES_NAMES = ["DA", "DT", "DG", "DC", "A", "U", "G", "C"]


class SELECTIONS(StrEnum):
    """
    Selections within an atom array for metrics or otherwise.
    """

    CA = "ca"
    BKBN = "bkbn"
    BKBN_NO_OXYGEN = "bkbn_no_o"
    ALLATOM = "allatom"
    MOTIF_CA = "motif_ca"
    MOTIF_BKBN = "motif_bkbn"
    MOTIF_BKBN_NO_OXYGEN = "motif_bkbn_no_o"
    MOTIF_ALLATOM = "motif_allatom"
    NUCLEOTIDE = "nucleotide"
    NUCLEOTIDE_BKBN = "nucleotide_bkbn"
    NUCLEOTIDE_P = "nucleotide_p"
    LIGANDS = "ligand"
    INTERFACES = "interface"

    def apply(self, **kwargs) -> np.ndarray[bool]:
        return get_selection_mask(sele=self, **kwargs)


def get_selection_mask(
    sele: SELECTIONS, atom_array: AtomArray, metadata: dict, split_by_chain=False
) -> np.ndarray[bool] | dict[str, np.ndarray[bool]]:
    mask = _get_selection_mask(sele, atom_array, metadata=metadata)
    if split_by_chain:
        # Break apart based on chain
        chain_id = atom_array.chain_id
        masks = {cid: mask & (chain_id == cid) for cid in np.unique(chain_id)}
        return masks
    else:
        assert not isinstance(mask, list), "Enum returns list, did not request list!"
        return mask


def _get_selection_mask(
    sele: SELECTIONS, atom_array: AtomArray, metadata: dict
) -> np.ndarray[bool]:
    """
    Get boolean mask for selected atom_array.
    Recursive function for composed selections
    """

    ##############################################################
    # Global selections
    ##############################################################

    if sele == SELECTIONS.CA:
        return np.isin(atom_array.atom_name, ["CA"])

    elif sele == SELECTIONS.BKBN:
        return np.isin(atom_array.atom_name, ["N", "CA", "C", "O"])

    elif sele == SELECTIONS.BKBN_NO_OXYGEN:
        return np.isin(atom_array.atom_name, ["N", "CA", "C"])

    elif sele == SELECTIONS.ALLATOM:
        return np.ones(len(atom_array), dtype=bool)

    ##############################################################
    # Nucleic acid selections
    ##############################################################

    elif sele == SELECTIONS.NUCLEOTIDE:
        return np.isin(atom_array.res_name, _NUCLEOTIDE_RES_NAMES)

    elif sele == SELECTIONS.NUCLEOTIDE_BKBN:
        return np.isin(atom_array.res_name, _NUCLEOTIDE_RES_NAMES) & np.isin(
            atom_array.atom_name, ["P", "O5'", "C5'", "C4'", "C3'", "O3'"]
        )

    elif sele == SELECTIONS.NUCLEOTIDE_P:
        return np.isin(atom_array.res_name, _NUCLEOTIDE_RES_NAMES) & np.isin(
            atom_array.atom_name, ["P"]
        )

    ##############################################################
    # Motif selections
    ##############################################################

    elif sele == SELECTIONS.MOTIF_ALLATOM:
        return get_mask_for_metadata("motif", atom_array, metadata, exclude_hetero=True)

    elif sele == SELECTIONS.MOTIF_CA:
        return _get_selection_mask(
            SELECTIONS.MOTIF_ALLATOM, atom_array, metadata
        ) & _get_selection_mask(SELECTIONS.CA, atom_array, metadata)

    elif sele == SELECTIONS.MOTIF_BKBN:
        return _get_selection_mask(
            SELECTIONS.MOTIF_ALLATOM, atom_array, metadata
        ) & _get_selection_mask(SELECTIONS.BKBN, atom_array, metadata)

    elif sele == SELECTIONS.MOTIF_BKBN_NO_OXYGEN:
        return _get_selection_mask(
            SELECTIONS.MOTIF_ALLATOM, atom_array, metadata
        ) & _get_selection_mask(SELECTIONS.BKBN_NO_OXYGEN, atom_array, metadata)

    ##############################################################
    # Based on design metadata:
    ##############################################################

    elif str(sele) in SELECTABLE_METADATA_FIELDS:
        return get_mask_for_metadata(str(sele), atom_array, metadata)

    else:
        raise ValueError(f"Unknown selection: {sele}, available: {list(SELECTIONS)}")


def get_mask_for_metadata(enum, atom_array, metadata, *, exclude_hetero=False):
    """Fetch mask based on a design metadata field.

    Parameters
    ----------
    exclude_hetero : bool
        If True, skip components that resolve entirely to hetero (ligand) atoms.
        This prevents ligand residues in the motif list from breaking selections
        when the ligand has been removed (e.g. ``remove_ligands=True`` in AF3).
    """
    assert enum in SELECTABLE_METADATA_FIELDS, (
        f"Enum {enum} not in design metadata fields {SELECTABLE_METADATA_FIELDS}"
    )
    components = metadata.get(str(enum))
    if components is None or len(components) == 0:
        raise ValueError(
            f"No components found for selection '{enum}'. "
            f"The upstream block did not populate metadata['{enum}']. "
            f"Available metadata keys: {[k for k, v in metadata.items() if v]}. "
            f"Ensure the producing block sets DesignMetadata.{enum.upper()} in make_row(metadata=...)."
        )
    mask = []
    for component in components:
        try:
            component_mask = fetch_mask_from_component(component, atom_array=atom_array)
        except ComponentValidationError:
            if exclude_hetero:
                continue
            raise
        if exclude_hetero and np.all(atom_array.hetero[component_mask]):
            continue
        mask.append(component_mask)
    if len(mask) == 0:
        raise ValueError(
            f"No non-hetero components found for selection '{enum}' after "
            f"filtering. Components: {components}"
        )
    mask = np.any(np.stack(mask, axis=0), axis=0)
    return mask
