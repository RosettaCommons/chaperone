"""Visualization and notebook helpers for pipeline benchmark analysis.

Uses unpacked metric columns (``handler.unpack(all=True)``) with fully qualified
keys (e.g. ``af3.plddt``). :class:`~blocks.util_blocks.filter.MetricFilter` is
imported from ``blocks.util_blocks.filter``; mask logic here stays on flat
columns because ``FilterBlock`` operates on the packed ``METRICS`` JSON column.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import random
import shutil
import warnings
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# Imported only for the ``GroupFilters`` type alias below — the canonical
# import path is ``blocks.util_blocks.filter.MetricFilter``. Deliberately NOT
# re-exported in ``__all__`` to avoid a second, divergent import path (the
# historical collision with the other ``MetricFilter`` class).
from blocks.util_blocks.filter import MetricFilter
from pipelines.data.catcif import (
    is_catcif_path,
    read_catcif_cif_bytes,
    split_catcif_path,
)
from pipelines import ExecutionEngine, PipelineDataHandler

__all__ = [
    "GroupFilters",
    "plot_metrics",
    "select_for_viewing",
    "_build_metrics_df",
    "_build_combined_mask_numpy",
    "print_funnel",
    "per_atom_conjunction",
    "ExecutionEngine",
    "PipelineDataHandler",
    "pl",
    "np",
    "plt",
]

GroupFilters = Mapping[str, Mapping[str, MetricFilter]]

_NUMERIC_DTYPES = frozenset(
    (
        pl.Float64,
        pl.Float32,
        pl.Int64,
        pl.Int32,
        pl.Int16,
        pl.Int8,
        pl.UInt64,
        pl.UInt32,
        pl.UInt16,
        pl.UInt8,
    )
)


def _unpack_to_polars(
    handler: PipelineDataHandler,
    *,
    drop_empty_structures: bool,
) -> pl.DataFrame:
    raw = handler._df
    if drop_empty_structures and "structures" in raw.columns:
        raw = raw.filter(
            pl.col("structures").is_not_null() & (pl.col("structures").list.len() > 0)
        )
    unpacked = PipelineDataHandler(raw).unpack(all=True)
    return unpacked if isinstance(unpacked, pl.DataFrame) else pl.from_pandas(unpacked)


def _resolve_metrics_col_list(
    metrics_col: str | list[str] | None,
    individual_filters: Mapping[str, MetricFilter] | None,
    group_filters: GroupFilters | None,
) -> list[str]:
    if metrics_col is None:
        inferred = _infer_metrics_col_prefixes(individual_filters, group_filters)
        return inferred if inferred else ["rfd3"]
    if isinstance(metrics_col, str):
        return [metrics_col]
    return list(metrics_col)


def _filter_key_order(
    metric_columns: set[str],
    individual_filters: Mapping[str, MetricFilter] | None,
    group_filters: GroupFilters | None,
) -> list[str]:
    """Keys that appear in filters, in mapping order, intersected with ``metric_columns``."""
    out: list[str] = []
    for key in individual_filters or ():
        if key in metric_columns and key not in out:
            out.append(key)
    for grp in (group_filters or {}).values():
        for key in grp:
            if key in metric_columns and key not in out:
                out.append(key)
    return out


def _iter_all_filters(
    individual_filters: Mapping[str, MetricFilter] | None,
    group_filters: GroupFilters | None,
) -> Iterator[MetricFilter]:
    for filt in (individual_filters or {}).values():
        yield filt
    for grp in (group_filters or {}).values():
        yield from grp.values()


def _filters_include_quantile(
    individual_filters: Mapping[str, MetricFilter] | None,
    group_filters: GroupFilters | None,
) -> bool:
    return any(
        f.quantile is not None
        for f in _iter_all_filters(individual_filters, group_filters)
    )


def _build_metrics_df(df: pl.DataFrame, metrics_col: str | list[str]) -> pl.DataFrame:
    """Numeric metric columns from JSON string columns and/or ``prefix.*`` unpacked columns."""
    cols = [metrics_col] if isinstance(metrics_col, str) else list(metrics_col)
    frames: list[pl.DataFrame] = []

    for col in cols:
        if col in df.columns and df.schema[col] == pl.Utf8:
            series = df.get_column(col)
            if series.is_empty():
                continue
            parsed = [
                json.loads(v) if isinstance(v, str) else v for v in series.to_list()
            ]
            if not any(parsed):
                continue
            mf = pl.DataFrame(parsed)
            if mf.width == 0:
                continue
            scalar_cols = [
                c for c, dt in mf.schema.items() if not isinstance(dt, pl.List)
            ]
            if not scalar_cols:
                continue
            frames.append(
                mf.select(
                    [
                        pl.col(c).cast(pl.Float64, strict=False).alias(c)
                        for c in scalar_cols
                    ]
                )
            )
        else:
            prefix = col + "."
            matching = [
                c
                for c in df.columns
                if c.startswith(prefix) and df.schema[c] in _NUMERIC_DTYPES
            ]
            if not matching and col in df.columns and df.schema[col] in _NUMERIC_DTYPES:
                matching = [col]
            if not matching:
                raise ValueError(
                    f"'{col}' is not a JSON column and no '{prefix}*' numeric columns found"
                )
            frames.append(
                df.select(
                    [
                        pl.col(c).cast(pl.Float64, strict=False).alias(c)
                        for c in matching
                    ]
                )
            )

    if not frames:
        return pl.DataFrame()
    out = frames[0]
    for f in frames[1:]:
        out = out.hstack(f)
    return out


def _merge_metric_filters(prev: MetricFilter, filt: MetricFilter) -> MetricFilter:
    new_min = (
        max(v for v in (prev.min, filt.min) if v is not None)
        if (prev.min is not None or filt.min is not None)
        else None
    )
    new_max = (
        min(v for v in (prev.max, filt.max) if v is not None)
        if (prev.max is not None or filt.max is not None)
        else None
    )
    qopts = [q for q in (prev.quantile, filt.quantile) if q is not None]
    new_q = min(qopts) if qopts else None
    d0, d1 = prev.quantile_direction, filt.quantile_direction
    new_dir = d0 if d0 == d1 else d1
    return MetricFilter(
        min=new_min,
        max=new_max,
        quantile=new_q,
        quantile_direction=new_dir,
        parent_based=prev.parent_based or filt.parent_based,
    )


def _collect_all_filters(
    individual_filters: Mapping[str, MetricFilter] | None,
    group_filters: GroupFilters | None,
) -> dict[str, MetricFilter]:
    """One entry per metric key; recurring keys get tightest min/max and strictest quantile."""
    merged: dict[str, MetricFilter] = {}
    sources: list[Mapping[str, MetricFilter]] = []
    if individual_filters:
        sources.append(individual_filters)
    sources.extend((group_filters or {}).values())

    for src in sources:
        for key, filt in src.items():
            if key not in merged:
                merged[key] = filt
            else:
                merged[key] = _merge_metric_filters(merged[key], filt)
    return merged


def _eligible_finite_values(
    metric_df: pl.DataFrame,
    key: str,
    eligible: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Row indices and values where ``eligible`` and value is finite."""
    if key not in metric_df.columns:
        return None
    v = metric_df.get_column(key).to_numpy(allow_copy=True)
    idx = np.nonzero(eligible)[0]
    if idx.size == 0:
        return None
    vs = v[idx]
    ok = ~np.isnan(vs)
    if not np.any(ok):
        return None
    return idx[ok], vs[ok]


def _quantile_pass_mask_numpy(
    metric_df: pl.DataFrame,
    key: str,
    quantile: float,
    *,
    higher_is_better: bool,
    eligible: np.ndarray,
) -> np.ndarray:
    out = np.zeros(metric_df.height, dtype=bool)
    pair = _eligible_finite_values(metric_df, key, eligible)
    if pair is None:
        return out
    row_idx, vals = pair
    n_keep = max(1, math.ceil(vals.size * quantile))
    order = np.argsort(-vals if higher_is_better else vals, kind="mergesort")
    out[row_idx[order[:n_keep]]] = True
    return out


def _quantile_cutoff_for_plot(
    metric_df: pl.DataFrame,
    key: str,
    quantile: float,
    *,
    higher_is_better: bool,
    eligible: np.ndarray,
) -> float | None:
    pair = _eligible_finite_values(metric_df, key, eligible)
    if pair is None:
        return None
    _, vals = pair
    vals = np.sort(vals)
    n_keep = max(1, math.ceil(vals.size * quantile))
    if higher_is_better:
        return float(vals[-n_keep])
    return float(vals[n_keep - 1])


def _apply_minmax_row_mask(
    metric_df: pl.DataFrame,
    mask: np.ndarray,
    key: str,
    filt: MetricFilter,
) -> np.ndarray:
    if key not in metric_df.columns:
        return mask
    v = metric_df.get_column(key).to_numpy(allow_copy=True)
    valid = ~np.isnan(v)
    m = valid.copy()
    if filt.min is not None:
        m &= v >= filt.min
    if filt.max is not None:
        m &= v <= filt.max
    out = mask & m
    out[~valid] = False
    return out


def _mask_minmax_and_groups_numpy(
    metric_df: pl.DataFrame,
    individual_filters: Mapping[str, MetricFilter] | None,
    group_filters: GroupFilters | None,
) -> np.ndarray:
    """Individual AND (group1 OR group2 OR …); min/max only (before quantile)."""
    n = metric_df.height
    combined = np.ones(n, dtype=bool)
    for key, filt in (individual_filters or {}).items():
        combined = _apply_minmax_row_mask(metric_df, combined, key, filt)
    if not group_filters:
        return combined
    group_or = np.zeros(n, dtype=bool)
    for grp in group_filters.values():
        grp_mask = np.ones(n, dtype=bool)
        for key, filt in grp.items():
            grp_mask = _apply_minmax_row_mask(metric_df, grp_mask, key, filt)
        group_or |= grp_mask
    return combined & group_or


def _build_combined_mask_numpy(
    metric_df: pl.DataFrame,
    individual_filters: Mapping[str, MetricFilter] | None,
    group_filters: GroupFilters | None,
) -> np.ndarray:
    """Min/max + groups, then AND quantile masks (each ranks on the same pre-quantile set)."""
    base = _mask_minmax_and_groups_numpy(metric_df, individual_filters, group_filters)
    merged = _collect_all_filters(individual_filters, group_filters)
    q_masks: list[np.ndarray] = []
    for key, filt in merged.items():
        if filt.quantile is None:
            continue
        if filt.parent_based:
            warnings.warn(
                f"MetricFilter '{key}': parent_based=True unsupported here; "
                "using global quantile rank.",
                stacklevel=2,
            )
        q_masks.append(
            _quantile_pass_mask_numpy(
                metric_df,
                key,
                filt.quantile,
                higher_is_better=filt.quantile_direction == "higher",
                eligible=base,
            )
        )
    if not q_masks:
        return base
    return base & np.logical_and.reduce(q_masks)


def _ordered_metric_keys(
    metric_df: pl.DataFrame,
    individual_filters: Mapping[str, MetricFilter] | None,
    group_filters: GroupFilters | None,
) -> list[str]:
    names = set(metric_df.columns)
    ordered = _filter_key_order(names, individual_filters, group_filters)
    return ordered + sorted(names - set(ordered))


def _infer_metrics_col_prefixes(
    individual_filters: Mapping[str, MetricFilter] | None,
    group_filters: GroupFilters | None,
) -> list[str]:
    prefixes: set[str] = set()
    for key in individual_filters or ():
        prefixes.add(key.split(".", 1)[0])
    for grp in (group_filters or {}).values():
        for key in grp:
            prefixes.add(key.split(".", 1)[0])
    return sorted(prefixes)


def _marginal_minmax_mask_for_key(
    metric_df: pl.DataFrame, key: str, filt: MetricFilter
) -> np.ndarray:
    """Valid rows for ``key`` that satisfy only this filter's min/max (ignores other columns)."""
    n = metric_df.height
    if key not in metric_df.columns:
        return np.zeros(n, dtype=bool)
    v = metric_df.get_column(key).to_numpy(allow_copy=True)
    valid = ~np.isnan(v)
    m = valid.copy()
    if filt.min is not None:
        m &= v >= filt.min
    if filt.max is not None:
        m &= v <= filt.max
    m[~valid] = False
    return m


def _marginal_pass_mask_for_column(
    metric_df: pl.DataFrame, key: str, filt: MetricFilter
) -> np.ndarray:
    """This column's min/max, then quantile on that slice only (matches single-metric FilterBlock)."""
    m = _marginal_minmax_mask_for_key(metric_df, key, filt)
    if filt.quantile is None:
        return m
    if filt.parent_based:
        warnings.warn(
            f"MetricFilter '{key}': parent_based=True unsupported in plot_metrics; "
            "using global rank for this column.",
            stacklevel=2,
        )
    return _quantile_pass_mask_numpy(
        metric_df,
        key,
        filt.quantile,
        higher_is_better=filt.quantile_direction == "higher",
        eligible=m,
    )


def _draw_filter_guides(
    ax: plt.Axes,
    filt: MetricFilter,
    metric_df: pl.DataFrame,
    key: str,
    *,
    quantile_eligible: np.ndarray,
) -> None:
    if filt.min is not None:
        ax.axvline(
            filt.min,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"min={filt.min}",
        )
    if filt.max is not None:
        ax.axvline(
            filt.max,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"max={filt.max}",
        )
    if filt.quantile is None:
        return
    xcut = _quantile_cutoff_for_plot(
        metric_df,
        key,
        filt.quantile,
        higher_is_better=filt.quantile_direction == "higher",
        eligible=quantile_eligible,
    )
    if xcut is not None:
        ax.axvline(
            xcut,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"q={filt.quantile} cutoff",
        )


def plot_metrics(
    handler: PipelineDataHandler,
    individual_filters: Mapping[str, MetricFilter] | None = None,
    group_filters: GroupFilters | None = None,
    metrics_col: str | list[str] | None = None,
    figsize: tuple[float, float] = (16, 12),
    bins: int = 50,
    color: str = "#4FB9AF",
    sample_size: int | None = 50000,
    remove_missing_structures: bool = True,
) -> plt.Figure:
    """Histogram grid: gray = full sample; teal = rows passing **this** metric's gate only.

    Panel ``[% Kept]`` is therefore the marginal pass rate for that metric (quantile
    ranks only within rows that pass that column's min/max). The suptitle still
    reports the **joint** pass rate (all individual AND group OR AND quantiles).

    Infers ``metrics_col`` from filter keys when omitted (else ``rfd3``).
    If any filter uses ``quantile``, subsampling is disabled for the joint mask.
    """
    all_filters = _collect_all_filters(individual_filters, group_filters)
    df = _unpack_to_polars(handler, drop_empty_structures=remove_missing_structures)
    n_total = df.height
    if (
        sample_size
        and n_total > sample_size
        and not _filters_include_quantile(individual_filters, group_filters)
    ):
        df = df.sample(n=sample_size, seed=random.randint(0, 1_000_000))
        n_total = df.height

    resolved = _resolve_metrics_col_list(metrics_col, individual_filters, group_filters)
    metric_df = _build_metrics_df(df, resolved)
    if metric_df.width == 0:
        raise ValueError(f"No metrics found for {resolved!r}")

    scalar_keys = _ordered_metric_keys(metric_df, individual_filters, group_filters)
    n_cols = 4
    n_rows = math.ceil(len(scalar_keys) / n_cols) or 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=100)
    axes = np.array(axes).flatten()

    combined_mask = _build_combined_mask_numpy(
        metric_df, individual_filters, group_filters
    )

    for i, key in enumerate(scalar_keys):
        ax = axes[i]
        v = metric_df.get_column(key).to_numpy(allow_copy=True)
        valid = ~np.isnan(v)
        data = v[valid]
        n_valid = int(data.size)
        filt = all_filters.get(key)
        has_gate = filt is not None and (
            filt.min is not None or filt.max is not None or filt.quantile is not None
        )
        if has_gate:
            row_mask = _marginal_pass_mask_for_column(metric_df, key, filt)
            pct = 100 * row_mask.sum() / n_valid if n_valid else 0.0
            if n_valid:
                _, bin_edges, _ = ax.hist(
                    data,
                    bins=bins,
                    alpha=0.3,
                    color="gray",
                    edgecolor="white",
                    linewidth=0.3,
                )
                ax.hist(
                    v[row_mask],
                    bins=bin_edges,
                    alpha=0.7,
                    color=color,
                    edgecolor="white",
                    linewidth=0.3,
                )
            q_elig = _marginal_minmax_mask_for_key(metric_df, key, filt)
            _draw_filter_guides(ax, filt, metric_df, key, quantile_eligible=q_elig)
            title = f"{key}\n[{pct:.1f}% Kept]"
        else:
            ax.hist(
                data,
                bins=bins,
                alpha=0.7,
                color=color,
                edgecolor="white",
                linewidth=0.3,
            )
            title = key

        ax.set_title(title.replace("_", " "), fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[len(scalar_keys) :]:
        ax.set_visible(False)

    n_pass = int(combined_mask.sum())
    pct_pass = 100 * n_pass / n_total if n_total else 0.0
    n_f = len(individual_filters or {}) + sum(
        len(g) for g in (group_filters or {}).values()
    )
    st = f"Metrics (n={n_total:,})"
    if n_f:
        st += f" | Filters: {n_f} | Pass all: {n_pass:,} ({pct_pass:.1f}%)"
        if group_filters:
            st += f" | Groups: {len(group_filters)} (OR)"
    plt.suptitle(st, fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig


def select_for_viewing(
    handler: PipelineDataHandler,
    output_dir: Path | str,
    individual_filters: Mapping[str, MetricFilter] | None = None,
    group_filters: GroupFilters | None = None,
    n_samples: int = 10,
    metrics_col: str | list[str] | None = None,
    path_col: str = "rfd3.path",
    include_failing: bool = False,
) -> pl.DataFrame:
    """Copy up to ``n_samples`` passing (and optionally failing) structures to ``output_dir``."""
    output_dir = Path(output_dir)
    df = _unpack_to_polars(handler, drop_empty_structures=True)
    n_total = df.height
    resolved = _resolve_metrics_col_list(metrics_col, individual_filters, group_filters)
    metric_df = _build_metrics_df(df, resolved)
    if metric_df.width == 0:
        print(f"No metrics found for {resolved!r}")
        return pl.DataFrame()

    metric_names = set(metric_df.columns)
    new_cols = [c for c in metric_df.columns if c not in df.columns]
    if new_cols:
        df = df.hstack(metric_df.select(new_cols))

    filter_metric_order = _filter_key_order(
        metric_names, individual_filters, group_filters
    )
    mask = pl.Series(
        _build_combined_mask_numpy(metric_df, individual_filters, group_filters)
    )
    passing_df = df.filter(mask)
    failing_df = df.filter(~mask)

    if passing_df.height == 0:
        print(f"No structures pass all filters (0 / {n_total:,})")
        if not include_failing:
            return pl.DataFrame()

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    yes_dir, no_dir = output_dir / "yes", output_dir / "no"
    yes_dir.mkdir()
    if include_failing:
        no_dir.mkdir()

    def safe_entry_filename(entry_id: str) -> str:
        return (
            entry_id.replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
            .replace(" ", "_")
        )

    def copy_sample(
        src: pl.DataFrame, dst: Path, n: int
    ) -> tuple[pl.DataFrame, list[str]]:
        if src.height == 0:
            return pl.DataFrame(), []
        sel = src.sample(
            n=min(n, src.height), seed=int(np.random.randint(0, 1_000_000))
        )
        paths: list[str] = []
        for idx, row in enumerate(sel.iter_rows(named=True)):
            raw_path = str(row[path_col])
            if is_catcif_path(raw_path):
                _, entry_id = split_catcif_path(raw_path)
                dp = dst / f"{idx:03d}_{safe_entry_filename(entry_id)}.cif"
                dp.write_bytes(read_catcif_cif_bytes(raw_path))
            else:
                sp = Path(raw_path)
                if sp.suffix == ".gz":
                    dp = dst / f"{idx:03d}_{sp.stem}"
                    with gzip.open(sp, "rb") as fin, open(dp, "wb") as fout:
                        shutil.copyfileobj(fin, fout)
                else:
                    dp = dst / f"{idx:03d}_{sp.name}"
                    shutil.copy(sp, dp)
            paths.append(str(dp))
        return sel, paths

    sel_pass, p_pass = copy_sample(passing_df, yes_dir, n_samples)
    if not sel_pass.is_empty():
        sel_pass = sel_pass.with_columns(pl.Series("copied_to", p_pass))

    sel_fail = pl.DataFrame()
    if include_failing and failing_df.height > 0:
        sel_fail, p_fail = copy_sample(failing_df, no_dir, n_samples)
        if not sel_fail.is_empty():
            sel_fail = sel_fail.with_columns(pl.Series("copied_to", p_fail))

    print(
        f"Selected {sel_pass.height} passing from {passing_df.height:,} "
        f"(of {n_total:,}) → {yes_dir}"
    )
    if include_failing:
        print(
            f"Selected {sel_fail.height} failing from {failing_df.height:,} → {no_dir}"
        )

    out = (
        pl.concat([sel_pass, sel_fail])
        if (include_failing and not sel_fail.is_empty())
        else sel_pass
    )
    if out.is_empty():
        return pl.DataFrame()

    remaining = sorted(metric_names - set(filter_metric_order))
    ret_cols = [path_col, "copied_to"] + filter_metric_order + remaining
    if "rfd3.metadata.task" in out.columns:
        ret_cols.insert(1, "rfd3.metadata.task")
    return out.select([c for c in ret_cols if c in out.columns])


def print_funnel(engine: ExecutionEngine) -> None:
    """Print row counts per manifest stage (reads parquets under ``engine.rundir``)."""
    for s in engine.manifest.steps:
        p = f"{engine.rundir}/{s.name}/output.parquet"
        n = pl.read_parquet(p).height if os.path.exists(p) else None
        marker = "✓" if n else ("—" if n == 0 else "?")
        print(f"  {marker} {s.name:<28} {n if n is not None else '':>6}")


def per_atom_conjunction(
    handler: PipelineDataHandler,
    step_prefix: str,
    atoms: list[str],
    threshold: float,
    direction: str = "min",
) -> pl.DataFrame:
    """Per-atom and cumulative pass rates for a conjunction of RASA atom columns."""
    unpacked = handler.unpack(all=True)
    df = unpacked if isinstance(unpacked, pl.DataFrame) else pl.from_pandas(unpacked)
    n = df.height
    rows: list[dict[str, Any]] = []
    cum = np.ones(n, dtype=bool)
    for a in atoms:
        col = f"{step_prefix}.ligand_atom_rasa.{a}"
        if col not in df.columns:
            rows.append(
                {
                    "atom": a,
                    "n_pass": None,
                    "pct_pass": None,
                    "n_cumulative": None,
                    "pct_cumulative": None,
                }
            )
            continue
        v = df[col].to_numpy()
        m = (v >= threshold) if direction == "min" else (v <= threshold)
        cum &= m
        rows.append(
            {
                "atom": a,
                "n_pass": int(m.sum()),
                "pct_pass": 100 * m.sum() / n,
                "n_cumulative": int(cum.sum()),
                "pct_cumulative": 100 * cum.sum() / n,
            }
        )
    return pl.DataFrame(rows)


