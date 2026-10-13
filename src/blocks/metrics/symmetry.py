"""Cyclic (Cn) point-group symmetry metrics for homo-oligomeric assemblies.

Every other geometry block in the repo measures a *local* property — a contact
area, a clash, an interface energy. This one measures the **global** geometry of
an assembly: whether the protomers of a designed homo-oligomer are genuinely
related by a single rotation of 360/N about a common axis.

Geometry
========
For a perfect ``Cn`` assembly the N protomers are identical copies related by
rotations of ``k * 360/N`` about one shared axis, with no translation along that
axis. So for every ordered pair of protomers ``(i, j)`` we superimpose ``j``
onto ``i`` on backbone atoms and decompose the fitted rigid transform
``x -> R x + t``:

- **RMSD after superposition** — if the protomers are not the same fold, no
  rotation can bring them together and this is large. ``subunit_rmsd`` reports
  the mean over protomers 2..N superimposed onto protomer 1.
- **Rotation angle** ``theta = arccos((tr(R) - 1) / 2)`` — the magnitude of the
  fitted rotation, always in ``[0, 180]``. For a ``Cn`` the multiset of the
  ``C(N, 2)`` pairwise angles is fixed by N alone (for C3: ``{120, 120, 120}``;
  for C4: ``{90, 90, 90, 90, 180, 180}``), *independently of how the chains are
  labelled*. ``angle_deviation_deg`` is the largest deviation after matching the
  sorted observed angles against the sorted ideal ones — a matching that, in one
  dimension, minimises the largest mismatch, so the metric never depends on
  guessing the cyclic order of the chains.
- **Rotation axis** — the eigenvector of ``R`` with eigenvalue 1. Every pair must
  share one axis. ``axis_rmsd`` is the RMS Euclidean distance between the
  sign-aligned unit axes and their mean direction (dimensionless, in ``[0, 2]``;
  0 is perfectly collinear), and ``axis_deviation_deg`` reports the same thing in
  the more readable unit of degrees.
- **Screw translation** ``t . axis`` — a proper cyclic symmetry has none. A
  non-zero value means the "assembly" is a helix, not a closed ring, which the
  angle and axis checks alone would happily accept.

Only ``Cn`` point groups are handled. Dihedral and polyhedral groups contain
improper-order subgroups whose pairwise-angle multiset is not determined by the
chain count alone, and would need the chain-to-subunit assignment supplied.
"""

import logging
import re

import biotite.structure as struc
import numpy as np
from pydantic import Field

from pipelines import InlineMetricBlock, JobArgs

logger = logging.getLogger(__name__)

BACKBONE_ATOMS = ("N", "CA", "C", "O")

_POINT_GROUP_PATTERN = re.compile(r"^[Cc]([2-9]|[1-9][0-9]+)$")

_NULL_METRICS = {
    "n_chains": None,
    "n_residues_per_subunit": None,
    "subunit_rmsd": None,
    "subunit_rmsd_max": None,
    "angle_deviation_deg": None,
    "mean_rotation_deg": None,
    "axis_rmsd": None,
    "axis_deviation_deg": None,
    "max_screw_translation": None,
}


def cyclic_pair_angles(n_copies: int) -> np.ndarray:
    """Sorted ideal pairwise rotation angles (deg) of a ``Cn`` assembly.

    Protomers ``i`` and ``j`` of a ``Cn`` are related by a rotation of
    ``k * 360/n`` with ``k = (j - i) mod n``; a rotation magnitude is only ever
    read in ``[0, 180]``, so ``k`` and ``n - k`` are indistinguishable.
    """
    angles = []
    for i in range(n_copies):
        for j in range(i + 1, n_copies):
            angle = 360.0 * ((j - i) % n_copies) / n_copies
            angles.append(min(angle, 360.0 - angle))
    return np.sort(np.asarray(angles, dtype=float))


def rotation_angle_and_axis(rotation: np.ndarray) -> tuple[float, np.ndarray]:
    """Angle (deg, in ``[0, 180]``) and unit axis of a 3x3 rotation matrix."""
    rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
    cosine = (np.trace(rotation) - 1.0) / 2.0
    angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

    # The skew part gives the axis directly, but degenerates as the angle
    # approaches 0 or 180, where the eigenvector for eigenvalue 1 is exact.
    skew = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
    norm = float(np.linalg.norm(skew))
    if norm > 1e-6:
        return angle, skew / norm

    values, vectors = np.linalg.eig(rotation)
    index = int(np.argmin(np.abs(values - 1.0)))
    axis = np.real(vectors[:, index])
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-9:
        raise ValueError("rotation matrix has no unit eigenvector")
    return angle, axis / axis_norm


class SymmetryMetricsBlock(InlineMetricBlock):
    """Measure how closely an assembly obeys its declared cyclic point group.

    Reports the fold identity of the protomers (``subunit_rmsd``), the deviation
    of the fitted inter-protomer rotations from the ideal ``360/N``
    (``angle_deviation_deg``), the collinearity of the fitted rotation axes
    (``axis_rmsd`` / ``axis_deviation_deg``), and the residual screw translation
    along the axis. See the module docstring for the geometry.

    All-zero deviations mean an ideal assembly; the metrics grow monotonically as
    the assembly departs from its point group.
    """

    _name = "symmetry"
    _display_name = "Symmetry Metrics"
    _description = "Cyclic point-group deviation of a homo-oligomeric assembly"
    _creates_new_step = False
    _resources = JobArgs(partition="cpu", rows_per_job=500)

    point_group: str = Field(
        default="C3",
        description="Expected cyclic point group, e.g. 'C3'. Only Cn is supported; "
        "dihedral and polyhedral groups need an explicit subunit assignment.",
    )
    chains: list[str] | None = Field(
        default=None,
        description="Protomer chain IDs. Defaults to every protein chain in the "
        "structure, alphabetically. Name them explicitly to score one oligomer "
        "inside a larger assembly.",
    )
    backbone_only: bool = Field(
        default=True,
        description="Superimpose on backbone N/CA/C/O only. Set False to include "
        "side chains — only comparable when every protomer is fully packed.",
    )
    exclude_hetero: bool = Field(
        default=True,
        description="Drop hetero (ligand/cofactor) atoms before scoring, so a "
        "bound cofactor never enters the protomer superposition.",
    )

    @property
    def n_copies(self) -> int:
        match = _POINT_GROUP_PATTERN.fullmatch(self.point_group)
        if match is None:
            raise ValueError(
                f"point_group must be cyclic ('C2', 'C3', ...), got "
                f"{self.point_group!r}"
            )
        return int(match.group(1))

    def get_data_kwargs(self):
        return {"which": "last", "load": True}

    def compute_metrics(self, loaded, *, context=None, index=0):
        try:
            coords = self._subunit_coordinates(loaded.atom_array)
            return self._symmetry_metrics(coords)
        except Exception as exc:  # noqa: BLE001 — one bad row must not kill the chunk
            logger.warning(
                "%s: symmetry metric failed on index %d: %s", self.name, index, exc
            )
            return {**_NULL_METRICS, "error": str(exc)}

    # ── geometry ────────────────────────────────────────────────────────────

    def _subunit_coordinates(self, atom_array) -> np.ndarray:
        """``(n_copies, n_atoms, 3)`` of matched backbone atoms, one row/chain.

        Atoms are matched by ``(residue ordinal, atom name)`` within each chain,
        so a superposition can never pair atoms that do not correspond.
        """
        keep = struc.filter_amino_acids(atom_array)
        if self.exclude_hetero and hasattr(atom_array, "hetero"):
            keep = keep & ~np.asarray(atom_array.hetero, dtype=bool)
        protein = atom_array[keep]
        protein = protein[protein.element != "H"]
        if self.backbone_only:
            protein = protein[np.isin(protein.atom_name, BACKBONE_ATOMS)]
        if not len(protein):
            raise ValueError("structure has no protein backbone atoms")

        chains = self.chains or sorted(set(np.unique(protein.chain_id).tolist()))
        if len(chains) != self.n_copies:
            raise ValueError(
                f"{self.point_group} needs {self.n_copies} protomer chains, found "
                f"{len(chains)}: {list(chains)}"
            )

        per_chain: list[dict[tuple[int, str], np.ndarray]] = []
        for chain in chains:
            mask = protein.chain_id == chain
            if not mask.any():
                raise ValueError(f"structure has no atoms for chain {chain}")
            subunit = protein[mask]
            # Residue ordinal, not res_id: chains may be numbered independently.
            _, ordinal = np.unique(subunit.res_id, return_inverse=True)
            per_chain.append(
                {
                    (int(ordinal[i]), str(subunit.atom_name[i])): subunit.coord[i]
                    for i in range(len(subunit))
                }
            )

        shared = set(per_chain[0])
        for atoms in per_chain[1:]:
            shared &= set(atoms)
        if not shared:
            raise ValueError("protomer chains share no comparable backbone atoms")
        keys = sorted(shared)

        dropped = max(len(atoms) - len(keys) for atoms in per_chain)
        if dropped:
            logger.warning(
                "%s: %d atom(s) present in some protomers but not all; scoring the "
                "%d shared atoms",
                self.name,
                dropped,
                len(keys),
            )
        return np.stack(
            [np.stack([atoms[key] for key in keys]) for atoms in per_chain]
        ).astype(float)

    def _symmetry_metrics(self, coords: np.ndarray) -> dict:
        from biotite.structure import superimpose

        n_copies, n_atoms, _ = coords.shape
        n_backbone = len(BACKBONE_ATOMS) if self.backbone_only else 1

        rmsds: dict[tuple[int, int], float] = {}
        angles: list[float] = []
        axes: list[np.ndarray] = []
        screws: list[float] = []

        for i in range(n_copies):
            for j in range(i + 1, n_copies):
                fitted, transform = superimpose(coords[i], coords[j])
                deviation = np.asarray(fitted) - coords[i]
                rmsds[(i, j)] = float(np.sqrt((deviation**2).sum(axis=-1).mean()))

                rotation = np.asarray(transform.rotation).reshape(3, 3)
                angle, axis = rotation_angle_and_axis(rotation)
                angles.append(angle)
                axes.append(axis)

                # The full map is x -> R (x - c_mobile) + c_fixed, so the screw
                # component is the part of its translation along the axis.
                centre_mobile = coords[j].mean(axis=0)
                centre_fixed = coords[i].mean(axis=0)
                translation = centre_fixed - rotation @ centre_mobile
                screws.append(abs(float(np.dot(translation, axis))))

        observed = np.sort(np.asarray(angles))
        ideal = cyclic_pair_angles(n_copies)
        angle_deviation = float(np.max(np.abs(observed - ideal)))

        axis_stack = np.stack(axes)
        # A rotation about +n by theta is the same as about -n by -theta, so the
        # sign of a fitted axis is arbitrary; align every axis to the first.
        signs = np.sign(axis_stack @ axis_stack[0])
        signs[signs == 0] = 1.0
        aligned = axis_stack * signs[:, None]
        mean_axis = aligned.mean(axis=0)
        mean_norm = float(np.linalg.norm(mean_axis))
        if mean_norm < 1e-9:
            raise ValueError("fitted rotation axes cancel; assembly has no shared axis")
        mean_axis = mean_axis / mean_norm

        axis_rmsd = float(np.sqrt(((aligned - mean_axis) ** 2).sum(axis=1).mean()))
        axis_deviation = float(
            np.degrees(np.arccos(np.clip(aligned @ mean_axis, -1.0, 1.0))).max()
        )

        onto_first = [rmsds[(0, j)] for j in range(1, n_copies)]
        return {
            "n_chains": int(n_copies),
            "n_residues_per_subunit": int(n_atoms // n_backbone),
            "subunit_rmsd": float(np.mean(onto_first)),
            "subunit_rmsd_max": float(np.max(list(rmsds.values()))),
            "angle_deviation_deg": angle_deviation,
            "mean_rotation_deg": float(np.mean(angles)),
            "axis_rmsd": axis_rmsd,
            "axis_deviation_deg": axis_deviation,
            "max_screw_translation": float(np.max(screws)),
        }
