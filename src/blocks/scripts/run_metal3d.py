#!/usr/bin/env python
"""Batch wrapper around Metal3D's metal3d.py.

Runs Metal3D inference on a list of input structures and writes the
predicted metal-binding probes (PDB) and a single combined JSON summary
that the Metal3DBlock parses in compile().

The upstream script (Metal3D/metal3d.py) hardcodes ``weights/...``
relative to the current working directory and pins ``torch.empty(...,
device='cuda')`` inside processStructures, which fails on CPU.  This
wrapper:

  * resolves the weights path from ``METAL3D_REPO`` (or the script's
    default install location ``/opt/metal-site-prediction``);
  * patches the CUDA-only voxel allocation so the block works on CPU
    boxes too (e.g. macOS dev runs);
  * loops over inputs, writes one probefile per input, and aggregates
    site/max-probability statistics into ``metal3d_results.json``.
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(os.environ.get("METAL3D_REPO", "/opt/metal-site-prediction"))
METAL3D_DIR = REPO_ROOT / "Metal3D"
sys.path.insert(0, str(METAL3D_DIR))

# helpers.py imports py3Dmol at module level (only used by interactive
# viewers like show_map). Stub it so the import doesn't fail in headless
# containers that omit it.
import types as _types  # noqa: E402

if "py3Dmol" not in sys.modules:
    sys.modules["py3Dmol"] = _types.ModuleType("py3Dmol")

from utils.helpers import (  # noqa: E402
    create_grid_fromBB,
    find_unique_sites,
    get_all_metalbinding_resids,
    get_all_protein_resids,
    get_bb,
    get_probability_mean,
    write_cubefile,
)
from utils.model import Model  # noqa: E402
from utils import voxelization as _vox  # noqa: E402
from utils.voxelization import processStructures  # noqa: E402


def _patch_voxelization_for_cpu() -> None:
    """Replace processStructures with a CPU-safe, fork-free implementation.

    Upstream uses ``Pool(cpu_count())`` to parallelise voxelisation and
    hardcodes ``torch.empty(..., device='cuda')``. Both break for us:

    * The ``device='cuda'`` path crashes on CPU-only hosts.
    * ``Pool`` uses fork by default, which deadlocks inside apptainer
      once torch's CUDA libs are loaded in the parent.

    Voxelisation is fast enough per residue that serial execution is
    fine even for a few hundred metal-binding sites.
    """
    import time as _time

    def processStructures_cpu_safe(pdb_file, resids, clean=True):
        from moleculekit.molecule import Molecule

        start = _time.time()
        try:
            prot = Molecule(pdb_file)
        except Exception as exc:  # pragma: no cover - input error path
            raise RuntimeError(f"could not read {pdb_file}: {exc}") from exc

        if clean:
            prot.filter("protein and not hydrogen")

        results = []
        for idx in resids:
            try:
                env = (prot.copy(), idx)
            except Exception:
                print(f"ignore {idx}")
                continue
            r = _vox.voxelize_single_notcentered(env)
            if r is not None:
                results.append(r)

        if not results:
            raise RuntimeError(
                "voxelization produced no boxes — check for chain breaks or "
                "missing metal-binding residues"
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        voxels = torch.empty(len(results), 8, 32, 32, 32, device=device)
        vox_env, prot_centers_list, prot_n_list, envs = zip(*results)
        for i, v in enumerate(vox_env):
            voxels[i] = v.to(device)

        print(
            f"-----  Voxelization  -----  "
            f"{_time.time() - start:.3f}s ({len(results)} boxes)"
        )
        return voxels, prot_centers_list, prot_n_list, envs

    _vox.processStructures = processStructures_cpu_safe


_patch_voxelization_for_cpu()
from utils.voxelization import processStructures  # noqa: E402,F811  (re-import patched)


def _patch_probability_mean_serial() -> None:
    """helpers.get_probability_mean uses a forking Pool too — same deadlock.

    Replace it with a vectorised KDTree query that's actually faster than
    the per-point ``Pool(cpu_count()).map`` on the protein-sized grids
    Metal3D produces.
    """
    from scipy.spatial import KDTree
    from utils import helpers as _helpers

    def get_probability_mean_serial(grid, prot_centers, pvalues):
        prot_v = np.asarray(prot_centers)
        pvalues = np.asarray(pvalues, dtype=np.float32)
        tree = KDTree(prot_v)
        # Vectorised k=20 ball query (matches upstream compute_average_p_fast):
        # query returns (n_grid, k) distance + index arrays with inf for empty
        # slots, which we mask out before averaging.
        dists, idxs = tree.query(grid, k=20, distance_upper_bound=0.25, workers=-1)
        valid = np.isfinite(dists)
        # When there are fewer than n_protein voxel centres, idxs uses len(prot_v)
        # as a sentinel — clip so np.take doesn't go OOB; we mask it out below.
        idxs_clipped = np.clip(idxs, 0, len(pvalues) - 1)
        gathered = pvalues[idxs_clipped]
        gathered = np.where(valid, gathered, 0.0)
        counts = valid.sum(axis=1)
        sums = gathered.sum(axis=1)
        return np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)

    _helpers.get_probability_mean = get_probability_mean_serial


_patch_probability_mean_serial()
from utils.helpers import get_probability_mean  # noqa: E402,F811  (re-import patched)


def _patch_create_grid_vectorised() -> None:
    """Replace the upstream Python triple-loop with np.meshgrid.

    For typical protein bounding boxes (~50³ = 125k points) the original
    Python loop dominates runtime; meshgrid is ~1000× faster.
    """
    from utils import helpers as _helpers

    def create_grid_fromBB(boundingBox, voxelSize=1):
        xrange = np.arange(boundingBox[0][0], boundingBox[1][0] + 0.5, step=voxelSize)
        yrange = np.arange(boundingBox[0][1], boundingBox[1][1] + 0.5, step=voxelSize)
        zrange = np.arange(boundingBox[0][2], boundingBox[1][2] + 0.5, step=voxelSize)
        # 'ij' + reshape preserves the upstream's
        # outer=x / middle=y / inner=z ordering used by write_cubefile.
        xx, yy, zz = np.meshgrid(xrange, yrange, zrange, indexing="ij")
        gridpoints = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(
            np.float64
        )
        return gridpoints, (xrange.shape[0], yrange.shape[0], zrange.shape[0])

    _helpers.create_grid_fromBB = create_grid_fromBB


_patch_create_grid_vectorised()
from utils.helpers import create_grid_fromBB  # noqa: E402,F811  (re-import patched)


def _read_probes(probefile: Path) -> list[dict]:
    sites = []
    with open(probefile) as f:
        for line in f:
            if not line.startswith(("HETATM", "ATOM")):
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                p = float(line[54:60])
            except ValueError:
                continue
            sites.append({"x": x, "y": y, "z": z, "p": p})
    return sites


def run_one(
    pdb: Path,
    output_dir: Path,
    model: torch.nn.Module,
    device: torch.device,
    *,
    metalbinding: bool,
    cluster_threshold: float,
    pthreshold: float,
    batch_size: int,
    write_cube: bool,
) -> dict:
    """Run Metal3D on a single structure and return a summary dict."""
    pdb = Path(pdb)
    stem = pdb.stem
    if stem.endswith(".cif"):
        stem = stem[:-4]
    probefile = output_dir / f"{stem}.probes.pdb"
    cubefile = output_dir / f"{stem}.cube"

    if metalbinding:
        ids = get_all_metalbinding_resids(str(pdb))
    else:
        ids = get_all_protein_resids(str(pdb))

    if len(ids) == 0:
        return {
            "input": str(pdb),
            "n_sites": 0,
            "max_p": None,
            "max_xyz": None,
            "sites": [],
            "probefile": None,
            "warning": "no metal-binding residues found",
        }

    voxels, prot_centers, prot_N, _ = processStructures(str(pdb), ids)
    voxels = voxels.to(device)

    outputs = torch.zeros([voxels.size(0), 1, 32, 32, 32])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        for i in range(0, voxels.size(0), batch_size):
            o = model(voxels[i : i + batch_size])
            outputs[i : i + batch_size] = o.cpu().detach()

    prot_v = np.vstack(prot_centers)
    output_v = outputs.flatten().numpy()
    bb = get_bb(prot_v)
    grid, box_N = create_grid_fromBB(bb)
    pvalues = get_probability_mean(grid, prot_v, output_v)

    if write_cube:
        write_cubefile(bb, pvalues, box_N, outname=str(cubefile), gridres=1)

    find_unique_sites(
        pvalues,
        grid,
        writeprobes=True,
        probefile=str(probefile),
        threshold=cluster_threshold,
        p=pthreshold,
    )

    sites = _read_probes(probefile) if probefile.exists() else []
    sites.sort(key=lambda s: s["p"], reverse=True)

    if pvalues.size:
        max_idx = int(np.argmax(pvalues))
        max_p = float(np.max(pvalues))
        max_xyz = [float(v) for v in grid[max_idx]]
    else:
        max_p = None
        max_xyz = None

    return {
        "input": str(pdb),
        "n_sites": len(sites),
        "max_p": max_p,
        "max_xyz": max_xyz,
        "sites": sites,
        "probefile": str(probefile) if sites else None,
        "cubefile": str(cubefile) if write_cube and cubefile.exists() else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Batch Metal3D runner")
    parser.add_argument(
        "--pdb",
        action="append",
        default=[],
        help="Input PDB/CIF path. Pass --pdb multiple times for batches.",
    )
    parser.add_argument(
        "--pdb-list",
        type=str,
        default=None,
        help="Optional file with one input path per line (in addition to --pdb).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where probefiles and metal3d_results.json are written.",
    )
    parser.add_argument(
        "--metalbinding",
        type=int,
        default=1,
        help="1 = scan only metal-binding sidechains (default); 0 = whole protein",
    )
    parser.add_argument("--cluster-threshold", type=float, default=7.0)
    parser.add_argument("--pthreshold", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--write-cube", type=int, default=0)
    parser.add_argument(
        "--weights",
        type=str,
        default=str(METAL3D_DIR / "weights" / "metal_0.5A_v3_d0.2_16Abox.pth"),
    )
    args = parser.parse_args()

    inputs = list(args.pdb)
    if args.pdb_list:
        with open(args.pdb_list) as f:
            inputs.extend(line.strip() for line in f if line.strip())
    if not inputs:
        parser.error("No inputs provided. Pass --pdb and/or --pdb-list.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Metal3D weights from {args.weights} on {device}")
    model = Model().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    start = time.time()
    results = {}
    for pdb in inputs:
        pdb_path = Path(pdb)
        print(f"\n=== Metal3D: {pdb_path.name} ===")
        try:
            res = run_one(
                pdb_path,
                output_dir,
                model,
                device,
                metalbinding=bool(args.metalbinding),
                cluster_threshold=args.cluster_threshold,
                pthreshold=args.pthreshold,
                batch_size=args.batch_size,
                write_cube=bool(args.write_cube),
            )
        except Exception as exc:
            print(f"  FAILED: {exc}")
            res = {"input": str(pdb_path), "error": str(exc)}
        results[pdb_path.name] = res

    out_json = output_dir / "metal3d_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_json} ({time.time() - start:.1f}s, {len(results)} inputs)")


if __name__ == "__main__":
    main()
