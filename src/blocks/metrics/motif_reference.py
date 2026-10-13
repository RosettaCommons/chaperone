"""Metrics scored against the design's own input motif, not against another step.

Motif-scaffolding campaigns need to ask "does this structure reproduce the native
site?" — a question about the reference structure, not about agreement between
two pipeline steps. Both blocks here resolve residue identity through RFD3's
``diffused_index_map`` (original residue id -> designed residue id), so they can
score an oracle refold without ever comparing it to the diffusion output.

``reference_motif_rmsd`` superimposes the mapped motif atoms on the reference and
reports their RMSD. ``motif_geometry`` measures named distances and angles between
mapped atoms, which needs no reference at all — useful for catalytic geometry that
has a target value rather than a target structure.
"""

import json
import logging
from typing import Dict, List, Optional

import numpy as np
from pydantic import Field

from blocks.metrics.utils import get_rmsd, weighted_rigid_align
from pipelines import InlineBlockFunction, JobArgs, PipelineDataHandler

logger = logging.getLogger(__name__)

LIGAND_PREFIX = "L:"


def _index_map(loaded) -> dict:
    raw = loaded.metadata.get("diffused_index_map")
    if not raw:
        return {}
    return json.loads(raw) if isinstance(raw, str) else raw


def _split_resid(resid: str):
    """'A27' -> ('A', 27). Chain ids may be multi-character."""
    i = len(resid)
    while i > 0 and resid[i - 1].isdigit():
        i -= 1
    return resid[:i], int(resid[i:])


def _residue_mask(atom_array, resid: str):
    """Mask for a residue id, or for a ligand when the id carries the L: prefix."""
    if resid.startswith(LIGAND_PREFIX):
        return atom_array.res_name == resid[len(LIGAND_PREFIX) :]
    chain, num = _split_resid(resid)
    return (atom_array.chain_id == chain) & (atom_array.res_id == num)


def _paired_coords(ref_atoms, tgt_atoms, ref_id: str, tgt_id: str):
    """Coordinates of atoms present under both ids, in a common order."""
    ref_mask, tgt_mask = (
        _residue_mask(ref_atoms, ref_id),
        _residue_mask(tgt_atoms, tgt_id),
    )
    if not ref_mask.any() or not tgt_mask.any():
        return None, None
    tgt_by_name = {
        str(n): i for i, n in zip(np.where(tgt_mask)[0], tgt_atoms.atom_name[tgt_mask])
    }
    ref_idx, tgt_idx = [], []
    for i, name in zip(np.where(ref_mask)[0], ref_atoms.atom_name[ref_mask]):
        j = tgt_by_name.get(str(name))
        if j is not None:
            ref_idx.append(i)
            tgt_idx.append(j)
    if not ref_idx:
        return None, None
    return ref_atoms.coord[ref_idx], tgt_atoms.coord[tgt_idx]


class ReferenceMotifRMSDBlock(InlineBlockFunction):
    """RMSD of the designed motif against the structure it was taken from."""

    _name = "reference_motif_rmsd"
    _display_name = "Reference motif RMSD"
    _description = "Motif RMSD against the input reference, via diffused_index_map"
    _creates_new_step = False
    _resources = JobArgs(partition="cpu", rows_per_job=500)

    reference: str = Field(
        description="Path to the canonical structure the motif was taken from."
    )
    which: List[int] = Field(
        default=[0, -1],
        description="Two indices into each row's structures list: the structure "
        "carrying diffused_index_map, and the structure to score. Default [0, -1] "
        "reads the map off the diffusion output and scores the final prediction; "
        "[0, 0] scores the diffusion output itself.",
    )
    residues: Optional[List[str]] = Field(
        default=None,
        description="Reference residue ids to score, e.g. ['A10-12', 'B4']. "
        "Defaults to every entry in diffused_index_map.",
    )
    ligands: List[str] = Field(
        default_factory=list,
        description="Ligand/ion residue names to include, e.g. ['MN']. RFD3NA "
        "leaves non-polymer components out of diffused_index_map, so they are "
        "matched by residue name in both structures instead.",
    )

    def execute(self, handler, sandbox_dir=None, chunk_id=None):
        from pipelines.io import parse_to_atom_array

        ref_atoms = parse_to_atom_array(self.reference)
        wanted = _expand_residues(self.residues) if self.residues else None

        rows_out = []
        for src, tgt in handler.get_data(which=self.which, load=True):
            metrics = self._compute(ref_atoms, src, tgt, wanted)
            rows_out.append(tgt.to_row_with_updates(metrics={self.name: metrics}))
        return PipelineDataHandler(rows_out)

    def _compute(self, ref_atoms, src, tgt, wanted) -> dict:
        metrics = {"motif_rmsd": None, "n_atoms": 0, "n_residues": 0}
        index_map = _index_map(src)
        if not index_map:
            logger.warning("No diffused_index_map on %s; skipping", src.path)
            return metrics

        pairs = [
            (k, str(v))
            for k, v in index_map.items()
            if wanted is None or k in wanted or k.startswith(LIGAND_PREFIX)
        ]
        pairs += [(f"{LIGAND_PREFIX}{n}", f"{LIGAND_PREFIX}{n}") for n in self.ligands]

        ref_xyz, tgt_xyz, n_res = [], [], 0
        for ref_id, tgt_id in pairs:
            a, b = _paired_coords(ref_atoms, tgt.atom_array, ref_id, tgt_id)
            if a is None:
                logger.warning("No paired atoms for %s -> %s", ref_id, tgt_id)
                continue
            ref_xyz.append(a)
            tgt_xyz.append(b)
            n_res += 1

        if not ref_xyz:
            return metrics
        ref_xyz, tgt_xyz = np.concatenate(ref_xyz), np.concatenate(tgt_xyz)

        # align_fn maps the mobile side (target) onto the static side (reference),
        # so it is applied to tgt_xyz -- never to ref_xyz.
        align_fn = weighted_rigid_align(
            ref_xyz[None], tgt_xyz[None], return_alignment_function=True
        )
        metrics["motif_rmsd"] = float(get_rmsd(ref_xyz, align_fn(tgt_xyz)))
        metrics["n_atoms"] = len(ref_xyz)
        metrics["n_residues"] = n_res
        return metrics


class MotifGeometryBlock(InlineBlockFunction):
    """Named distances and angles between motif atoms, addressed by reference id."""

    _name = "motif_geometry"
    _display_name = "Motif geometry"
    _description = "Distances/angles between motif atoms, via diffused_index_map"
    _creates_new_step = False
    _resources = JobArgs(partition="cpu", rows_per_job=500)

    measurements: Dict[str, List[str]] = Field(
        description="Name -> atom specs as '<reference residue id>.<atom name>', "
        'e.g. {\'contact\': ["A24.N1", "B4.O2\'"]}. Two specs give a '
        "distance in A, three give the angle in degrees at the middle atom. "
        "Ligand atoms use the index map's L: prefix, e.g. 'L:MN.MN'.",
    )
    which: List[int] = Field(
        default=[0, -1],
        description="Two indices into each row's structures list: the structure "
        "carrying diffused_index_map, and the structure to measure.",
    )

    def execute(self, handler, sandbox_dir=None, chunk_id=None):
        rows_out = []
        for src, tgt in handler.get_data(which=self.which, load=True):
            metrics = self._compute(src, tgt)
            rows_out.append(tgt.to_row_with_updates(metrics={self.name: metrics}))
        return PipelineDataHandler(rows_out)

    def _compute(self, src, tgt) -> dict:
        index_map = _index_map(src)
        metrics = {name: None for name in self.measurements}
        if not index_map:
            logger.warning("No diffused_index_map on %s; skipping", src.path)
            return metrics

        for name, specs in self.measurements.items():
            if len(specs) not in (2, 3):
                raise ValueError(
                    f"measurement {name!r} needs 2 (distance) or 3 (angle) atoms, "
                    f"got {len(specs)}"
                )
            pts = [self._atom_coord(index_map, tgt.atom_array, s) for s in specs]
            if any(p is None for p in pts):
                continue
            if len(pts) == 2:
                metrics[name] = float(np.linalg.norm(pts[0] - pts[1]))
            else:
                u, v = pts[0] - pts[1], pts[2] - pts[1]
                cos = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
                metrics[name] = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
        return metrics

    def _atom_coord(self, index_map, atom_array, spec: str):
        ref_id, _, atom_name = spec.rpartition(".")
        if not ref_id:
            raise ValueError(f"atom spec {spec!r} must be '<residue id>.<atom name>'")
        # Non-polymer components are absent from the index map, so an id that is
        # not in it (e.g. "L:MN") is used against the target directly.
        tgt_id = str(index_map.get(ref_id, ref_id))
        mask = _residue_mask(atom_array, tgt_id) & (atom_array.atom_name == atom_name)
        if not mask.any():
            logger.warning("Atom %s (-> %s) not found", spec, tgt_id)
            return None
        return atom_array.coord[mask][0]


def _expand_residues(residues: List[str]) -> set:
    """['A10-12', 'B4'] -> {'A10', 'A11', 'A12', 'B4'}."""
    out = set()
    for item in residues:
        chain, _, span = item.partition("-")
        if not span:
            out.add(item)
            continue
        prefix, start = _split_resid(chain)
        for n in range(start, int(span) + 1):
            out.add(f"{prefix}{n}")
    return out
