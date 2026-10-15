#!/usr/bin/env python
"""Run SimpleFold while keeping its large reusable downloads in one cache."""

from __future__ import annotations

import argparse
import fcntl
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path


CCD_URL = "https://huggingface.co/boltz-community/boltz-1/resolve/main/ccd.pkl"


def _download_once(url: str, destination: Path) -> None:
    """Download *url* atomically, with a process lock for shared filesystems."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.with_suffix(f"{destination.suffix}.lock")
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if destination.is_file() and destination.stat().st_size > 0:
            return
        partial = destination.with_suffix(f"{destination.suffix}.partial.{os.getpid()}")
        try:
            for attempt in range(5):
                try:
                    request = urllib.request.Request(
                        url, headers={"User-Agent": "chaperone-simplefold/1"}
                    )
                    with urllib.request.urlopen(request, timeout=60) as response:
                        with partial.open("wb") as output:
                            shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
                    if partial.stat().st_size == 0:
                        raise IOError(f"downloaded an empty file from {url}")
                    os.replace(partial, destination)
                    break
                except Exception:
                    partial.unlink(missing_ok=True)
                    if attempt == 4:
                        raise
                    time.sleep(5 * (attempt + 1))
        finally:
            partial.unlink(missing_ok=True)


def _link_shared_ccd(cache: Path, weights_dir: Path) -> None:
    """Provide SimpleFold's required CCD without its unused Boltz checkpoint."""
    shared_ccd = weights_dir / "data" / "ccd.pkl"
    _download_once(CCD_URL, shared_ccd)
    cache.mkdir(parents=True, exist_ok=True)
    local_ccd = cache / "ccd.pkl"
    if local_ccd.is_symlink() and local_ccd.resolve() == shared_ccd.resolve():
        return
    if local_ccd.exists() or local_ccd.is_symlink():
        local_ccd.unlink()
    local_ccd.symlink_to(shared_ccd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simplefold-model", default="simplefold_100M")
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--fasta-path", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--predict-plddt", action="store_true")
    parser.add_argument("--output-format", choices=("pdb", "mmcif"), default="pdb")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Upstream uses package-internal absolute imports (``from model ...``).
    # Its console script works because it starts inside the package directory;
    # add that same directory when invoking the module through this wrapper.
    import simplefold

    sys.path.append(str(Path(simplefold.__file__).resolve().parent))
    from simplefold import inference
    import esm

    # Upstream uses torch.hub for both fair-esm's code and weights. The image
    # already carries the audited fair-esm package, so avoid a runtime GitHub
    # clone and retain only the canonical checkpoint download/cache behavior.
    inference.esm_registry["esm2_3B"] = esm.pretrained.esm2_t36_3B_UR50D

    output_dir = args.output_dir.resolve()
    weights_dir = args.weights_dir.resolve()
    cache = output_dir / "cache"
    _link_shared_ccd(cache, weights_dir)
    checkpoint = weights_dir / "checkpoints" / f"{args.simplefold_model}.ckpt"
    _download_once(inference.ckpt_url_dict[args.simplefold_model], checkpoint)

    # Upstream downloads both the CCD and an unused Boltz confidence checkpoint.
    # The inference path only reads ccd.pkl, so replace that helper with the
    # already-populated shared-cache implementation.
    inference.download_fasta_utilities = lambda requested_cache: _link_shared_ccd(
        Path(requested_cache), weights_dir
    )

    namespace = argparse.Namespace(
        simplefold_model=args.simplefold_model,
        ckpt_dir=str(weights_dir / "checkpoints"),
        output_dir=str(output_dir),
        num_steps=args.num_steps,
        tau=args.tau,
        no_log_timesteps=False,
        fasta_path=str(args.fasta_path.resolve()),
        nsample_per_protein=args.num_samples,
        plddt=args.predict_plddt,
        output_format=args.output_format,
        backend="torch",
        seed=args.seed,
    )
    inference.predict_structures_from_fastas(namespace)


if __name__ == "__main__":
    main()
