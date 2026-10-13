#!/usr/bin/env python
"""Inner runner for the ESMFold2 pipelines block.

ESMFold2 ships no CLI, so this script is the command the ``esmfold2`` block
invokes inside its container. It reads one spec JSON per design from
``--inputs`` and writes a predicted CIF per design plus a ``summary.csv`` of
confidence metrics to ``--outdir``.

Spec JSON schema (written by ``ESMFold2Block.create_command``)::

    {
      "name": "design_0",
      "chains":  [{"id": "A", "sequence": "MSEQ..."}],
      "ligands": [{"id": "B", "smiles": "N1C(=O)..."}]
    }

Folds via the Biohub ESMFold2 Python API:
``ESMFold2InputBuilder().fold(model, StructurePredictionInput(...), ...)``.
Requires a GPU. Inside the block container the weights live under
``HF_HUB_CACHE=/projects/ml/esm-weights``.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time
from pathlib import Path

# Weights + CCD cache. The container sets HF_HUB_CACHE; keep a fallback so the
# script also runs against a local install. ESMCFOLD_CCD_PATH points the input
# builder at the component dictionary shipped alongside the ESMFold2 snapshot.
os.environ.setdefault("HF_HUB_CACHE", "/projects/ml/esm-weights")
if "ESMCFOLD_CCD_PATH" not in os.environ:
    _ccd = glob.glob(
        os.path.join(
            os.environ["HF_HUB_CACHE"],
            "models--biohub--ESMFold2",
            "snapshots",
            "*",
            "ccd.pkl",
        )
    )
    if _ccd:
        os.environ["ESMCFOLD_CCD_PATH"] = _ccd[0]


def _build_input(spec, builder_types):
    """Build a StructurePredictionInput from a spec dict."""
    StructurePredictionInput, ProteinInput, LigandInput = builder_types
    sequences = []
    for chain in spec.get("chains", []):
        sequences.append(
            ProteinInput(id=str(chain["id"]), sequence=chain["sequence"], msa=None)
        )
    for lig in spec.get("ligands", []):
        sequences.append(LigandInput(id=str(lig["id"]), smiles=lig["smiles"]))
    return StructurePredictionInput(sequences=sequences)


def _seq_length(spec) -> int:
    return sum(len(c["sequence"]) for c in spec.get("chains", []))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=Path, required=True, help="dir of spec JSONs")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--model", default="biohub/ESMFold2")
    ap.add_argument("--num-loops", type=int, default=3)
    ap.add_argument("--num-sampling-steps", type=int, default=200)
    ap.add_argument("--num-diffusion-samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--all-samples",
        action="store_true",
        help="write every diffusion sample (default: best by pLDDT only)",
    )
    args = ap.parse_args()

    import torch

    if not torch.cuda.is_available():
        sys.exit("ERROR: no CUDA device. ESMFold2 requires a GPU.")

    from esm.models.esmfold2 import (
        ESMFold2InputBuilder,
        LigandInput,
        ProteinInput,
        StructurePredictionInput,
    )
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    builder_types = (StructurePredictionInput, ProteinInput, LigandInput)

    spec_files = sorted(args.inputs.glob("*.json"))
    if not spec_files:
        sys.exit(f"no spec JSONs found in {args.inputs}")
    args.outdir.mkdir(parents=True, exist_ok=True)

    print(
        f"[setup] {len(spec_files)} designs | loading {args.model} on cuda ...",
        flush=True,
    )
    t0 = time.time()
    model = ESMFold2Model.from_pretrained(args.model).to("cuda").eval()
    builder = ESMFold2InputBuilder()  # loads ccd.pkl
    print(f"[setup] model ready in {time.time() - t0:.0f}s", flush=True)

    rows = []
    for k, spec_file in enumerate(spec_files, 1):
        spec = json.loads(spec_file.read_text())
        name = spec.get("name", spec_file.stem)
        length = _seq_length(spec)
        try:
            spi = _build_input(spec, builder_types)
            t = time.time()
            result = builder.fold(
                model,
                spi,
                num_loops=args.num_loops,
                num_sampling_steps=args.num_sampling_steps,
                num_diffusion_samples=args.num_diffusion_samples,
                seed=args.seed,
                complex_id=name,
            )
            results = result if isinstance(result, list) else [result]
            # Sort best (highest mean pLDDT) first.
            results.sort(key=lambda r: float(r.plddt.mean()), reverse=True)

            if args.all_samples and len(results) > 1:
                emitted = [(f"{name}_sample{i}", r) for i, r in enumerate(results)]
            else:
                emitted = [(name, results[0])]

            for out_name, r in emitted:
                (args.outdir / f"{out_name}.cif").write_text(r.complex.to_mmcif())
                plddt = float(r.plddt.mean())
                ptm = float(r.ptm) if r.ptm is not None else float("nan")
                iptm = float(r.iptm) if r.iptm is not None else float("nan")
                rows.append((out_name, length, plddt, ptm, iptm))
            best = results[0]
            print(
                f"[{k}/{len(spec_files)}] {name}: len={length} "
                f"plddt={float(best.plddt.mean()):.3f} "
                f"({time.time() - t:.0f}s)",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001 - record failure row, keep going
            print(f"[{k}/{len(spec_files)}] {name}: FAILED -> {e!r}", flush=True)
            rows.append((name, -1, float("nan"), float("nan"), float("nan")))

    summary = args.outdir / "summary.csv"
    with open(summary, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "length", "mean_plddt", "ptm", "iptm"])
        for name, length, plddt, ptm, iptm in rows:
            w.writerow([name, length, f"{plddt:.4f}", f"{ptm:.4f}", f"{iptm:.4f}"])

    n_ok = len([r for r in rows if r[1] > 0])
    print(f"[done] wrote {n_ok} structures + {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
