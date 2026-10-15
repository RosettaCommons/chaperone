"""Ori-token sweep helpers for RFD3.

RFD3 takes an ``ori_token`` per design spec; its ``set_com`` step subtracts that
token from input coords before diffusion. Different shifts land the ligand in
different bf16 quantization buckets, so a structured sweep over ori_token values
produces a genuinely different design distribution per shell position — not just
the noise you get from ``jitter_ori_sigma``.

Typical flow (single-conformer)::

    from blocks.utils.ori_sweep import (
        generate_ori_points, write_merged_json,
    )

    entries = generate_ori_points(
        input_pdb="files/target.pdb",
        ligand_name="HPD",
        exposed_atoms=["O1P", "O2P"],
        surface_offsets=[5.0, 6.0, 7.0],
        n_points=20,
        cone_half_angle=60.0,
    )
    sweep_json = write_merged_json(
        "files/base.json", entries, "files/ori_sweep.json"
    )
    engine.run(RFD3(inputs=sweep_json, n_batches=10, batch_size=10))

Each entry in the merged JSON becomes an independent RFD3 task. For a
single-entry base JSON, task names follow ``ori_t{tier}_{fib_idx:02d}``.

Multi-conformer flow (one base JSON, many PDB conformers)::

    import json
    base_specs = json.load(open("files/multi_conf.json"))
    per_base_entries = {
        k: generate_ori_points(
            input_pdb=v["input"],
            ligand_name="HPD",
            exposed_atoms=["O1P", "O2P"],
            surface_offsets=[5.0, 7.0],
            n_points=20,
            cone_half_angle=60.0,
        )
        for k, v in base_specs.items()
    }
    sweep_json = write_merged_json(
        "files/multi_conf.json", per_base_entries, "files/ori_sweep.json"
    )

In multi-entry mode, keys become ``{base_key}_ori_t{tier}_{fib_idx:02d}``
and surface in ``rfd3.metadata.task`` for grouping by (conformer, ori-tier).
"""

from __future__ import annotations

import json
import os
from typing import Sequence

import numpy as np
from scipy.optimize import brentq
from scipy.spatial.distance import cdist

from chaperone.io import parse_to_atom_array


def fibonacci_sphere(n: int) -> np.ndarray:
    """Near-uniform points on a unit sphere. Returns shape ``(n, 3)``."""
    i = np.arange(n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5**0.5) * i
    return np.column_stack(
        [
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi),
        ]
    )


def _sphere_angles(unit_vec: np.ndarray) -> tuple[float, float]:
    x, y, z = unit_vec
    theta = float(np.degrees(np.arctan2(y, x)) % 360)
    phi = float(np.degrees(np.arccos(np.clip(z, -1, 1))))
    return theta, phi


def load_ligand_coords(pdb_path: str, res_name: str) -> dict[str, np.ndarray]:
    """Return ``{atom_name: xyz}`` for one ligand residue.

    Validates that the ligand has exactly one residue with the given
    ``res_name`` and that its atom names are unique — duplicate atom names
    would silently collide in the returned dict.
    """
    aa = parse_to_atom_array(pdb_path)
    lig = aa[aa.res_name == res_name]
    if len(lig) == 0:
        raise ValueError(f"no atoms with res_name={res_name!r} in {pdb_path}")
    unique_residues = {(str(c), int(r)) for c, r in zip(lig.chain_id, lig.res_id)}
    if len(unique_residues) != 1:
        raise ValueError(
            f"expected exactly one residue with res_name={res_name!r} in "
            f"{pdb_path}, found {len(unique_residues)}: {sorted(unique_residues)}"
        )
    atom_names = [str(n) for n in lig.atom_name]
    if len(set(atom_names)) != len(atom_names):
        from collections import Counter

        dups = [n for n, c in Counter(atom_names).items() if c > 1]
        raise ValueError(
            f"duplicate atom names in res_name={res_name!r} of {pdb_path}: {dups}"
        )
    return {n: c.astype(float) for n, c in zip(atom_names, lig.coord)}


def _offset_surface_radius(unit_dir, centroid, lig_xyz, target):
    """Radial t such that ``centroid + t*u`` sits exactly ``target`` Å from the
    nearest ligand atom (offset-surface, not fixed-radius)."""

    def f(t):
        return float(cdist((centroid + t * unit_dir)[None], lig_xyz).min() - target)

    hi = target + float(np.linalg.norm(lig_xyz - centroid, axis=1).max()) + 1.0
    return brentq(f, 0.0, hi)


def generate_ori_points(
    *,
    input_pdb: str,
    ligand_name: str,
    exposed_atoms: Sequence[str],
    surface_offsets: Sequence[float],
    n_points: int,
    cone_half_angle: float,
) -> list[dict]:
    """Fibonacci directions × radial tiers → offset-surface lattice of ori_tokens.

    Cone exclusion prunes directions near the linker-exit rays defined by
    ``exposed_atoms`` (kept directions are at least ``cone_half_angle`` degrees
    away from every exposed-atom ray). Each kept direction is placed at every
    ``surface_offset`` distance from the nearest ligand atom.

    Returns one dict per (kept direction, tier) pair with keys:
    ``idx``, ``fib_idx``, ``tier``, ``ori_token`` (``[x, y, z]``), ``theta``,
    ``phi``, ``radial_dist``, ``surface_dist``, ``cone_angle``.
    """
    coords = load_ligand_coords(input_pdb, ligand_name)
    lig_xyz = np.stack(list(coords.values()))
    centroid = lig_xyz.mean(axis=0)
    unit = fibonacci_sphere(n_points)

    if len(exposed_atoms) == 0:
        # Fully buried ligand (no linker/handle) — no exclusion cone, sweep is omnidirectional.
        keep = np.ones(n_points, dtype=bool)
        min_ang = np.full(n_points, 180.0)
    else:
        exp_xyz = np.stack([coords[a] for a in exposed_atoms])
        exp_dir = exp_xyz - centroid
        exp_dir /= np.linalg.norm(exp_dir, axis=1, keepdims=True)
        dots = unit @ exp_dir.T
        cos_thr = np.cos(np.radians(cone_half_angle))
        keep = ~(dots > cos_thr).any(axis=1)
        min_ang = np.degrees(np.arccos(np.clip(dots.max(axis=1), -1, 1)))

    out: list[dict] = []
    idx = 0
    for fi in np.where(keep)[0]:
        theta, phi = _sphere_angles(unit[fi])
        for tier, off in enumerate(surface_offsets):
            t = _offset_surface_radius(unit[fi], centroid, lig_xyz, off)
            xyz = centroid + t * unit[fi]
            out.append(
                {
                    "idx": idx,
                    "fib_idx": int(fi),
                    "tier": tier,
                    "ori_token": xyz.tolist(),
                    "theta": theta,
                    "phi": phi,
                    "radial_dist": float(t),
                    "surface_dist": float(off),
                    "cone_angle": float(min_ang[fi]),
                }
            )
            idx += 1
    return out


def write_merged_json(
    base_json_path: str,
    entries: list[dict] | dict[str, list[dict]],
    out_path: str,
) -> str:
    """Write a single multi-key RFD3 JSON merging ori_token entries into bases.

    Modes:

    Single-entry base (legacy):
        keys = ``ori_t{tier}_{fib_idx:02d}``.

    Multi-entry base with ``entries: list[dict]``:
        Cross-product. Keys = ``{base_key}_ori_t{tier}_{fib_idx:02d}``.
        WARNING: ori_tokens from ``generate_ori_points`` live in the input
        frame of the PDB they were computed on. Reusing the same list across
        bases whose ligands sit at different coordinates is geometrically
        suspect — the shells will land on the wrong frame for non-source
        bases. Prefer the dict form below for multi-conformer sweeps.

    Multi-entry base with ``entries: dict[base_key -> list[dict]]``:
        Per-base. Keys = ``{base_key}_ori_t{tier}_{fib_idx:02d}`` for each
        (base_key, ori_entry). Bases whose key is absent from the dict
        contribute nothing.

    Returns the output path.
    """
    base_specs = json.load(open(base_json_path))
    spec: dict[str, dict] = {}

    if len(base_specs) == 1:
        only_key, base = next(iter(base_specs.items()))
        es = entries[only_key] if isinstance(entries, dict) else entries
        for e in es:
            key = f"ori_t{e['tier']}_{e['fib_idx']:02d}"
            spec[key] = {**base, "ori_token": e["ori_token"]}
    else:
        for base_key, base in base_specs.items():
            if isinstance(entries, dict):
                if base_key not in entries:
                    continue
                es = entries[base_key]
            else:
                es = entries
            for e in es:
                key = f"{base_key}_ori_t{e['tier']}_{e['fib_idx']:02d}"
                spec[key] = {**base, "ori_token": e["ori_token"]}

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    json.dump(spec, open(out_path, "w"), indent=2)
    return out_path
