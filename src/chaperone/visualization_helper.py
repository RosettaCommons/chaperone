"""Plot and inspect rows already selected by the pipeline's FilterBlock.

Selection belongs to the pipeline: pass its filtered handler to these helpers.
They do not implement independent thresholds, groups, or quantile rankings.
"""

from __future__ import annotations

import gzip
import math
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from chaperone import ExecutionEngine, PipelineDataHandler
from chaperone.data.catcif import (
    catcif_entry_filename,
    is_catcif_path,
    read_catcif_cif_bytes,
)

__all__ = ["plot_metrics", "select_for_viewing", "print_funnel"]


def plot_metrics(
    handler: PipelineDataHandler,
    metrics_col: str | list[str] | None = None,
    figsize: tuple[float, float] = (16, 12),
    bins: int = 50,
    color: str = "#4FB9AF",
    sample_size: int | None = 50000,
) -> plt.Figure:
    """Plot numeric metrics in an already selected handler.

    ``metrics_col`` selects exact columns or step prefixes (e.g. ``"af3"``).
    None plots all numeric metrics. Missing or nonfinite values raise before
    sampling, so a small display sample cannot hide a failed measurement.
    """
    frame = handler.unpack(all=True, to_pandas=False)
    numeric = [name for name, dtype in frame.schema.items() if dtype.is_numeric()]
    if metrics_col is None:
        columns = numeric
    else:
        queries = [metrics_col] if isinstance(metrics_col, str) else metrics_col
        columns = []
        for query in queries:
            matches = [c for c in numeric if c == query or c.startswith(query + ".")]
            if not matches:
                raise ValueError(f"No numeric metrics found for {query!r}")
            columns.extend(matches)
        columns = list(dict.fromkeys(columns))
    if not columns or frame.is_empty():
        raise ValueError("No numeric metric rows to plot")
    invalid = {
        name: int((~np.isfinite(frame[name].cast(pl.Float64).to_numpy())).sum())
        for name in columns
    }
    invalid = {name: count for name, count in invalid.items() if count}
    if invalid:
        raise ValueError(f"Cannot plot missing/nonfinite metrics (counts): {invalid}")
    if sample_size is not None:
        if sample_size < 1:
            raise ValueError("sample_size must be positive or None")
        frame = frame.sample(n=min(sample_size, frame.height), seed=0)

    n_cols = min(4, len(columns))
    fig, axs = plt.subplots(math.ceil(len(columns) / n_cols), n_cols, figsize=figsize)
    axes = iter(np.asarray(axs).reshape(-1))
    for name in columns:
        plt.sca(next(axes))
        plt.hist(frame[name].to_numpy(), bins=bins, color=color, edgecolor="white")
        plt.title(name, fontsize=9)
        plt.tick_params(labelsize=8)
        plt.gca().spines[["top", "right"]].set_visible(False)
    for ax in axes:
        ax.set_visible(False)
    fig.suptitle(f"Metrics (n={len(handler):,}; plotted {frame.height:,})", fontsize=12)
    fig.tight_layout()
    return fig


def select_for_viewing(
    handler: PipelineDataHandler,
    output_dir: Path | str,
    n_samples: int = 10,
    path_col: str = "rfd3.path",
) -> pl.DataFrame:
    """Copy a sample of already selected structures, including CatCIF entries.

    Use a new directory for each selection; existing files are never removed.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    frame = handler.unpack(all=True, to_pandas=False)
    if path_col not in frame.columns or frame[path_col].null_count():
        raise ValueError(f"Missing structure paths in {path_col!r}")
    selected = frame.sample(n=min(n_samples, frame.height), seed=0)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    paths = []
    for index, raw_path in enumerate(selected[path_col]):
        if is_catcif_path(raw_path):
            destination = output_dir / f"{index:03d}_{catcif_entry_filename(raw_path)}"
            destination.write_bytes(read_catcif_cif_bytes(raw_path))
        else:
            source = Path(raw_path)
            destination = (
                output_dir
                / f"{index:03d}_{source.stem if source.suffix == '.gz' else source.name}"
            )
            if source.suffix == ".gz":
                with gzip.open(source, "rb") as fin, destination.open("wb") as fout:
                    shutil.copyfileobj(fin, fout)
            else:
                shutil.copyfile(source, destination)
        paths.append(str(destination))
    return selected.with_columns(pl.Series("copied_to", paths, dtype=pl.Utf8))


def print_funnel(engine: ExecutionEngine) -> None:
    """Print row counts from the engine's existing step outputs."""
    for step in engine.manifest.steps:
        path = engine.rundir / step.name / "output.parquet"
        count = pl.read_parquet(path).height if path.exists() else None
        print(f"  {step.name:<28} {count if count is not None else '?':>6}")
