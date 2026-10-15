"""Pinned public RPXdock adapter: scores and C3 trimers refer to the same model."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np


def ranked_models(scores, count):
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("RPXdock returned empty or non-finite scores")
    return np.argsort(-values, kind="stable")[:count].tolist()


def c3_frames(axis):
    from scipy.spatial.transform import Rotation

    axis = np.asarray(axis, dtype=float)[:3]
    axis /= np.linalg.norm(axis)
    frames = []
    for angle in (0, 2 * np.pi / 3, 4 * np.pi / 3):
        frame = np.eye(4)
        frame[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
        frames.append(frame)
    return np.stack(frames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--nout", type=int, default=1)
    parser.add_argument("--beam-size", type=int, default=10000)
    parser.add_argument("--ignored-aas", default="CGP")
    args = parser.parse_args()
    if args.nout < 1 or args.beam_size < 1:
        parser.error("nout and beam-size must be positive")
    import rpxdock as rp
    from rpxdock.app import dock

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Source checkout contains scoring archives omitted by the upstream wheel.
    sys.argv = [
        sys.argv[0],
        "--architecture",
        "I3",
        "--inputs1",
        args.input,
        "--hscore_files",
        "small_ilv_h",
        "--hscore_data_dir",
        str(Path(rp.__file__).parent / "data/hscore"),
        "--cart_bounds",
        "0",
        "200",
        "--use_orig_coords",
        "--score_only_ss",
        "EH",
        "--function",
        "stnd",
        "--beam_size",
        str(args.beam_size),
        "--ncpu",
        "1",
        "--ignored_aas",
        args.ignored_aas,
        "--output_prefix",
        str(args.output_dir / "search"),
    ]
    options = dock.get_rpxdock_args()
    hscore = rp.RpxHier(options.hscore_files, **options)
    result = dock.dock_onecomp(hscore, **options)
    frames = c3_frames(rp.search.dockspec.DockSpec1CompCage("I3").axis)
    records = []
    for rank, model in enumerate(ranked_models(result.scores.values, args.nout)):
        path = args.output_dir / f"trimer_{rank:03d}.pdb"
        # Construct the C3 subgroup explicitly; the first three icosahedral
        # symmetry frames are not guaranteed to be the desired trimer.
        result.dump_pdb(model, fname=str(path), symframes=frames, use_orig_coords=True)
        records.append(
            {
                "path": str(path.resolve()),
                "model_index": model,
                "score": float(result.scores.values[model]),
            }
        )
    (args.output_dir / "poses.json").write_text(
        json.dumps(records, indent=2, allow_nan=False)
    )


if __name__ == "__main__":
    main()
