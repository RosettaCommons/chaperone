#!/usr/bin/env python3
"""Draw the two figures of the guidance-cycling benchmark from its bundled table.

Reads files/cycling_af3_metrics.csv (800 AF3 predictions: 20 fusion designs x 5
routes x 2 MPNN modes x 4 sequences) and writes, at 300 dpi:

    .assets/metric_distributions.png   per-prediction AF3 metrics, route x MPNN mode
    .assets/pass_rates.png             best-of-4 pass rate per design, at two gates

The routes are the uncycled baseline, which is the MPNN sequence threaded back onto
the generator backbone, and four cycling stages at classifier-free guidance 1.0, 0.5,
0.3 and 0. Guidance 0 switches guidance off rather than weakening it. A design counts
as passing a cell when any of its four sequences clears the gate.

    python examples/benchmarks/guidance-cycling/plot_comparison.py
"""

import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
SRC = HERE / "files" / "cycling_af3_metrics.csv"
OUT = HERE / ".assets"
INK, AXIS, GRID = "#2c2820", "#555c59", "#e1e6e4"
TEAL, BLUE, GOLD = "#1c8273", "#6585c4", "#DA9D67"

for path in fm.findSystemFonts():                      # optional; falls back silently
    if os.path.basename(path).startswith("NotoSans"):
        fm.fontManager.addfont(path)
family = "Noto Sans" if "Noto Sans" in {f.name for f in fm.fontManager.ttflist} else "sans-serif"
plt.rcParams.update({"font.family": family, "font.size": 8, "axes.edgecolor": AXIS,
                     "axes.labelcolor": INK, "text.color": INK, "xtick.color": AXIS,
                     "ytick.color": AXIS, "axes.linewidth": 0.6, "pdf.fonttype": 42})

ROUTES = [("baseline_mpnn_asu", "baseline"), ("cfg100", "cfg 1.0"), ("cfg050", "cfg 0.5"),
          ("cfg030", "cfg 0.3"), ("cfg000", "cfg 0")]
MODES = [("soluble_mpnn", "soluble", BLUE), ("enhanced_plddt", "enhanced pLDDT", GOLD)]
PANELS = [("mean_plddt", "AF3 mean pLDDT", "mean pLDDT", [(80, "80"), (85, "85")], None),
          ("plddt_q10", "AF3 pLDDT q10", "pLDDT q10", [(80, "80")], None),
          ("af3_fixed_ca_rmsd_to_previous_mpnn_asu", "Fixed-residue RMSD to the design",
           "fixed CA RMSD (A)", [(2, "2 A")], (0, 12)),
          ("af3_full_ca_rmsd_to_route_mpnn_asu", "Full-chain RMSD to its own backbone",
           "full CA RMSD (A)", [], (0, 16)),
          ("pae_mean", "AF3 PAE mean", "PAE mean", [(7, "7")], None),
          ("pae_q90", "AF3 PAE q90", "PAE q90", [], None)]

rows = list(csv.DictReader(open(SRC)))


def val(row, key):
    try:
        return float(row[key])
    except (TypeError, ValueError):
        return np.nan


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_xticks(range(len(ROUTES)))
    ax.set_xticklabels([label for _, label in ROUTES], fontsize=7)


fig, axes = plt.subplots(3, 2, figsize=(7.2, 7.6), dpi=300)
for ax, (col, title, ylabel, gates, ylim) in zip(axes.ravel(), PANELS):
    data, pos, colors = [], [], []
    for i, (route, _) in enumerate(ROUTES):
        for j, (mode, _, color) in enumerate(MODES):
            values = [v for v in (val(r, col) for r in rows
                                  if r["route"] == route and r["mpnn_mode"] == mode)
                      if np.isfinite(v)]
            data.append(values)
            pos.append(i + (j - 0.5) * 0.34)
            colors.append(color)
    box = ax.boxplot(data, positions=pos, widths=0.30, patch_artist=True, showfliers=False,
                     medianprops=dict(color=INK, lw=1.0), whiskerprops=dict(color=AXIS, lw=0.6),
                     capprops=dict(color=AXIS, lw=0.6), boxprops=dict(lw=0.6))
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
        patch.set_edgecolor(color)
    for p, values, color in zip(pos, data, colors):          # one dot per prediction
        if values:
            ax.plot(np.random.default_rng(0).normal(p, 0.035, len(values)), values, ".",
                    ms=1.6, color=color, alpha=0.45, zorder=1)
    for gate, label in gates:
        ax.axhline(gate, color=TEAL, lw=0.7, ls=(0, (3, 2)), zorder=0)
        ax.text(len(ROUTES) - 0.42, gate, label, color=TEAL, fontsize=6, va="bottom", ha="right")
    style(ax)
    ax.set_title(title, fontsize=8.5, color=INK, pad=4)
    ax.set_ylabel(ylabel, fontsize=7.5)
    if ylim:
        ax.set_ylim(*ylim)
axes[0, 0].legend(handles=[Patch(facecolor=c, alpha=0.45, edgecolor=c, label=label)
                           for _, label, c in MODES], fontsize=7, frameon=False,
                  loc="lower right")
fig.suptitle("Does guidance cycling improve an already-good backbone?  "
             "20 designs, 800 AF3 predictions", fontsize=9.5, color=INK, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.985])
fig.savefig(OUT / "metric_distributions.png", dpi=300)
plt.close(fig)

cases = sorted({r["case"] for r in rows})


def any_pass(case, route, mode, plddt_gate):
    return any(val(r, "mean_plddt") > plddt_gate
               and val(r, "af3_fixed_ca_rmsd_to_previous_mpnn_asu") < 2.0
               for r in rows
               if r["case"] == case and r["route"] == route and r["mpnn_mode"] == mode)


fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), dpi=300)
for ax, (gate, title) in zip(axes, [(80, "pLDDT > 80 and fixed RMSD < 2 A"),
                                    (85, "pLDDT > 85 and fixed RMSD < 2 A")]):
    for j, (mode, label, color) in enumerate(MODES):
        fractions = [sum(any_pass(c, route, mode, gate) for c in cases) / len(cases)
                     for route, _ in ROUTES]
        x = np.arange(len(ROUTES)) + (j - 0.5) * 0.34
        ax.bar(x, fractions, width=0.30, color=color, edgecolor="none", label=label)
        for xi, v in zip(x, fractions):
            if v > 0:
                ax.text(xi, v + 0.012, f"{v * 100:.0f}%", ha="center", fontsize=6, color=AXIS)
    style(ax)
    ax.set_title(title, fontsize=8.5, color=INK, pad=4)
    ax.set_ylabel("designs passing (best of 4)", fontsize=7.5)
    ax.set_ylim(0, 0.45)
axes[0].legend(fontsize=7, frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "pass_rates.png", dpi=300)
print(f"wrote {OUT / 'metric_distributions.png'} and {OUT / 'pass_rates.png'}")
