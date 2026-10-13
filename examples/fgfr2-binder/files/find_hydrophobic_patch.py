#!/usr/bin/env -S bash -c '"${PIPELINES_REPO:-.}/.venv/bin/python" "$0" "$@"'
"""Find the largest solvent-exposed hydrophobic patch on each half of the receptor.

Splits chain B in two by residue ID (default: at B244, the FGF-binding patch),
then for each half identifies the largest cluster of solvent-exposed hydrophobic
residues. A "patch" = connected component where every pair of hydrophobic CAs is
within --cluster-distance Å of another in the same set.

Run:
    ./files/find_hydrophobic_patch.py [--cut-resi 244] [--sasa-min 30] [--cluster-distance 8.0]

Prints two RFD3 contig fragments (one per half, covering ±5 residues around each
patch) and the patch residue lists.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from chaperone.io import parse_to_atom_array  # noqa: E402

# Sidechain-driven hydrophobics (canonical aliphatic + aromatic + Met).
# Excludes Tyr (often polar via OH), Cys (S can H-bond), Gly/Pro (no clear side chain).
HYDROPHOBIC = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP"}


def compute_sasa(atoms):
    """Per-residue total SASA (Å²) using biotite's Shrake-Rupley implementation."""
    from biotite.structure import sasa

    atom_sasa = sasa(atoms, vdw_radii="Single", probe_radius=1.4)
    # Aggregate to per-residue
    starts = []
    chain_ids = atoms.chain_id
    res_ids = atoms.res_id
    last = None
    for i, (c, r) in enumerate(zip(chain_ids, res_ids)):
        if last != (c, r):
            starts.append(i)
            last = (c, r)
    starts.append(len(atoms))
    rec = []
    for i in range(len(starts) - 1):
        lo, hi = starts[i], starts[i + 1]
        rec.append(
            (
                str(atoms.chain_id[lo]),
                int(atoms.res_id[lo]),
                str(atoms.res_name[lo]),
                float(atom_sasa[lo:hi].sum()),
            )
        )
    return rec  # list of (chain_id, res_id, res_name, sasa)


def cluster_by_distance(coords, cutoff):
    """Greedy connected-component clustering: edge if pairwise distance ≤ cutoff."""
    from scipy.spatial.distance import cdist

    if len(coords) == 0:
        return []
    d = cdist(coords, coords)
    n = len(coords)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if d[i, j] <= cutoff:
                union(i, j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def patch_for_half(atoms, sasa_per_res, lo, hi, sasa_min, cluster_distance):
    """Return (cluster of (resid, resname), centroid, ca_coords) for the largest
    hydrophobic patch in chain B residues [lo, hi]. (lo, hi) inclusive."""
    cb_mask = (atoms.chain_id == "B") & (atoms.atom_name == "CA")
    cb_atoms = atoms[cb_mask]
    keep = []
    for res in sasa_per_res:
        cid, resid, rname, area = res
        if cid != "B":
            continue
        if not (lo <= resid <= hi):
            continue
        if rname not in HYDROPHOBIC:
            continue
        if area < sasa_min:
            continue
        keep.append((resid, rname, area))
    if not keep:
        return [], None, None
    coords = np.array(
        [
            cb_atoms.coord[(cb_atoms.res_id == r[0])][0]
            for r in keep
            if (cb_atoms.res_id == r[0]).any()
        ]
    )
    # filter `keep` to those that actually had a CA (skip mismatches, e.g. atomworks splits)
    keep = [k for k in keep if (cb_atoms.res_id == k[0]).any()]
    clusters = cluster_by_distance(coords, cluster_distance)
    clusters.sort(key=len, reverse=True)
    if not clusters:
        return [], None, None
    top = clusters[0]
    cluster_resids = [keep[i] for i in top]
    cluster_coords = coords[top]
    return cluster_resids, cluster_coords.mean(axis=0), cluster_coords


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--target-pdb",
        default=str(SCRIPT_DIR / "1djs_target.pdb"),
        help="PDB to analyse (default: 1djs_target.pdb).",
    )
    p.add_argument(
        "--cut-resi",
        type=int,
        default=244,
        help="Residue at which to split chain B (default: 244, the FGF binding patch).",
    )
    p.add_argument(
        "--sasa-min",
        type=float,
        default=30.0,
        help="Minimum residue SASA in Å² to count a residue as exposed (default: 30).",
    )
    p.add_argument(
        "--cluster-distance",
        type=float,
        default=8.0,
        help="Max Cα-Cα distance to be in the same patch (default: 8.0 Å).",
    )
    p.add_argument(
        "--padding",
        type=int,
        default=5,
        help="Residues of context to include in the suggested contig (default: ±5).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    atoms = parse_to_atom_array(args.target_pdb, remove_hydrogens=True)
    sasa_per_res = compute_sasa(atoms)

    b_atoms = atoms[atoms.chain_id == "B"]
    b_res_ids = sorted(int(r) for r in np.unique(b_atoms.res_id))
    lo_global, hi_global = b_res_ids[0], b_res_ids[-1]

    halves = [
        ("Half A (N-side)", lo_global, args.cut_resi),
        ("Half B (C-side)", args.cut_resi + 1, hi_global),
    ]

    print(f"Receptor chain B: residues {lo_global}-{hi_global}")
    print(
        f"Cut at B{args.cut_resi}; SASA threshold ≥ {args.sasa_min} Å²; cluster ≤ {args.cluster_distance} Å.\n"
    )

    for name, lo, hi in halves:
        cluster, centroid, _ = patch_for_half(
            atoms, sasa_per_res, lo, hi, args.sasa_min, args.cluster_distance
        )
        if not cluster:
            print(f"{name} (B{lo}-B{hi}): no hydrophobic patch found.")
            continue
        cluster.sort()
        rids = [c[0] for c in cluster]
        c_lo, c_hi = min(rids), max(rids)
        sug_lo = max(lo, c_lo - args.padding)
        sug_hi = min(hi, c_hi + args.padding)
        print(f"{name} (B{lo}-B{hi}):")
        print(
            f"  Largest patch ({len(cluster)} residues, centroid={centroid.round(1).tolist()}):"
        )
        for resid, rname, area in cluster:
            print(f"    B{resid:3d}  {rname}  SASA={area:5.1f} Å²")
        print(f"  Suggested contig segment (±{args.padding} pad): B{sug_lo}-B{sug_hi}")
        print(f"  Hotspot residues: {','.join(f'B{r}' for r in rids)}")
        print()


if __name__ == "__main__":
    main()
