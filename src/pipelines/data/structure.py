"""
LoadedStructure: Container for loaded design data from pipeline rows.

Provides lazy loading of atom arrays and selection mask computation.
"""

import tempfile
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Dict, List, Optional

from pipelines.common import strip_name
from pipelines.data.schema import (
    SELECTIONS,
    Column,
    StructureColumn,
    get_deserialized,
    get_selection_mask,
    make_row,
)
from pipelines.io import get_sequences, parse_to_atom_array

logger = logging.getLogger(__name__)


@dataclass
class LoadedStructure:
    """Container for loaded design data from a single pipeline step.

    Attributes:
        name: Step name (e.g., "rfd3", "mpnn", "af3")
        path: File path for this step
        metadata: Design-level metadata (task, motif, ligand, etc.)
        metrics: Step-specific metrics dict
        atom_array: Loaded atom array (only when load=True)
        masks: Dict mapping SELECTION enum to boolean mask (only when select is provided)
    """

    name: str
    path: str
    metadata: dict
    metrics: dict

    # Optionally loaded
    basename: str
    design_id: Optional[str] = None
    parent_design_id: Optional[str] = None
    store_path: Optional[str] = None
    storage_format: Optional[str] = None
    atom_array: Optional[object] = None
    sequence: Optional[dict[str, str | None]] = None
    masks: Optional[Dict[SELECTIONS, object]] = None

    def to_transform_dict(self, **extra) -> dict:
        """Convert to a dict compatible with atomworks Transform.forward().

        The returned dict always contains ``atom_array`` (the standard key
        consumed by virtually every atomworks transform), plus ``metadata``,
        ``metrics``, ``path``, and ``name``.  Pass extra keyword arguments to
        inject additional keys.

        Returns:
            Dictionary suitable for passing to an atomworks Transform.
        """
        d = {
            "atom_array": self.atom_array,
            "metadata": self.metadata,
            "metrics": self.metrics,
            "path": self.path,
            "name": self.name,
        }
        d.update(extra)
        return d

    def from_transform_dict(
        self,
        result: dict,
        *,
        extract_metrics: list[str] | None = None,
        extract_metadata: list[str] | None = None,
        metric_namespace: str | None = None,
    ) -> "LoadedStructure":
        """Apply a transform result dict back onto this LoadedStructure.

        Optionally pulls named keys from *result* into ``metrics`` and/or
        ``metadata``.  If *result* contains an ``atom_array`` that differs from
        the original, it is stored on the returned copy.

        Args:
            result: Dict returned by an atomworks Transform.
            extract_metrics: Keys whose values become metrics.
            extract_metadata: Keys whose values become metadata entries.
            metric_namespace: If set, metrics are nested under this key
                (e.g. ``{"transform": {"sasa": 0.5}}``).

        Returns:
            A **new** LoadedStructure with the updates applied.
        """
        from dataclasses import replace as dc_replace

        new_metrics = dict(self.metrics)
        if extract_metrics:
            extracted = {}
            for key in extract_metrics:
                if key in result:
                    val = result[key]
                    if hasattr(val, "item"):
                        val = val.item()
                    extracted[key] = val
            if metric_namespace:
                new_metrics[metric_namespace] = {
                    **new_metrics.get(metric_namespace, {}),
                    **extracted,
                }
            else:
                new_metrics.update(extracted)

        new_metadata = dict(self.metadata)
        if extract_metadata:
            for key in extract_metadata:
                if key in result:
                    new_metadata[key] = result[key]

        atom_array = result.get("atom_array", self.atom_array)

        return dc_replace(
            self,
            atom_array=atom_array,
            metadata=new_metadata,
            metrics=new_metrics,
        )

    def to_row_with_updates(
        self,
        name=None,
        path=None,
        metadata=None,
        metrics=None,
    ) -> dict:
        """Serialize back to dict with optional overrides"""
        if name is not None:
            self.name = name
        if path is not None:
            self.path = path
        if metadata is not None:
            # TODO: Check metadata
            self.metadata = {**self.metadata, **metadata}
        if metrics is not None:
            self.metrics = {**self.metrics, **metrics}
        return self.to_row()

    def to_row(self) -> dict:
        return make_row(
            name=self.name,
            path=self.path,
            metadata=self.metadata,
            metrics=self.metrics,
            design_id=self.design_id,
            parent_design_id=self.parent_design_id,
        )

    def load_array(
        self,
        select: Optional[List[SELECTIONS]] = None,
    ) -> "LoadedStructure":
        """Return a new LoadedStructure with the atom array loaded from disk or store.

        Args:
            select: List of SELECTIONS to compute masks for.

        Returns:
            New LoadedStructure with atom_array, sequence, and optionally masks populated.
        """
        from dataclasses import replace as dc_replace

        load_path = _resolve_path(self.path, self.storage_format)
        atom_array = parse_to_atom_array(
            load_path,
            assign_types=True,
            remove_ccds=[],
        )
        sequence = get_sequences(atom_array)

        masks = None
        if select and atom_array is not None:
            masks = {}
            for selection in select:
                try:
                    mask = get_selection_mask(
                        selection,
                        atom_array=atom_array,
                        metadata=self.metadata,
                    )
                    masks[selection] = mask
                except ValueError as e:
                    if "No components found" in str(e) or "Component not found" in str(
                        e
                    ):
                        logger.warning(
                            f"Skipping selection {selection} for {self.name}: {e}"
                        )
                        continue
                    raise e

        return dc_replace(
            self,
            atom_array=atom_array,
            sequence=sequence,
            masks=masks,
        )

    @classmethod
    def load_all_from_row(cls, row, load=False, select=None) -> List["LoadedStructure"]:
        metrics = get_deserialized(row, Column.METRICS)
        structures = row[Column.STRUCTURES]
        outs = []
        for i, struct in enumerate(structures):
            try:
                outs.append(
                    cls.from_single(
                        structure=struct, metrics=metrics, load=load, select=select
                    )
                )
            except Exception:
                paths = [s.get(StructureColumn.PATH, "?") for s in structures]
                logger.error(
                    f"Failed loading structure {i}/{len(structures)} "
                    f"(name={struct.get(StructureColumn.NAME)}, "
                    f"path={struct.get(StructureColumn.PATH)}). "
                    f"All paths in row: {paths}"
                )
                raise
        return outs

    @classmethod
    def from_single(
        cls,
        structure: dict,
        metrics: dict,
        load: bool = False,
        select: Optional[List[SELECTIONS]] = None,
        design_id: str | None = None,
        parent_design_id: str | None = None,
    ) -> "LoadedStructure":
        """Create LoadedStructure from a structure dict.

        Args:
            structure: Structure dict with name, path, store_path, metadata
            metrics: Metrics dict for this row
            load: If True, load atom array from path
            select: List of SELECTIONS to compute masks for (requires load=True)

        Returns:
            LoadedStructure instance
        """
        path = structure[StructureColumn.PATH]
        name = structure[StructureColumn.NAME]
        metadata = structure.get(StructureColumn.METADATA)
        store_path = structure.get(StructureColumn.STORE_PATH)
        storage_format = structure.get(StructureColumn.STORAGE_FORMAT)

        if select and not load:
            raise ValueError(
                "Cannot compute selection masks without loading atom array (load=True required)."
            )

        instance = cls(
            name=name,
            path=path,
            metadata=metadata,
            metrics=metrics,
            basename=strip_name(path),
            design_id=design_id,
            parent_design_id=parent_design_id,
            store_path=store_path,
            storage_format=storage_format,
        )

        if load:
            instance = instance.load_array(select=select)

        return instance


def _resolve_path(path: str, storage_format: Optional[str] = None) -> str:
    """Resolve a structure path for loading.

    Resolution order:
    1. Catcif tag (``archive.catcif:entry_id``): fetch via ``catcif_tools`` to
       a temp file. Recognised by ``storage_format == "catcif"`` or a path
       matching the catcif tag pattern.
    2. If the path exists on disk, use it directly.
    3. Fall through to the original path.
    """
    from pipelines.data.catcif import is_catcif_path

    if storage_format == "catcif" or is_catcif_path(path):
        import catcif_tools

        cif_str = catcif_tools.get_structure(path)
        tmp = tempfile.NamedTemporaryFile(suffix=".cif", delete=False)
        tmp.write(cif_str.encode("utf-8"))
        tmp.flush()
        return tmp.name

    if Path(path).exists():
        return path
    return path
