#!/usr/bin/env python3
"""Summarise Harbor eval runs: agent outcome, held-out score, and cost.

Reads only committed run artifacts under jobs/results/, so it is safe to call
while runs are still in flight — incomplete runs simply report what exists.

    ./scripts/eval_status.py                 # every run
    ./scripts/eval_status.py pdl1-baseline # only matching run names
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "jobs" / "results"


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def summarise(run_dir: Path) -> dict:
    """One row per run: what the agent produced and what the scorer said."""
    name = run_dir.name
    effort = next(
        (e for e in ("low", "medium", "high") if re.search(rf"\b{e}\b|-{e}-", name)),
        "?",
    )

    # The trial directory is the single child that is not a bookkeeping file.
    trials = [p for p in run_dir.iterdir() if p.is_dir()]
    trial = trials[0] if trials else None

    row = {
        "run": name,
        "effort": effort,
        "designs": None,
        "n_pass": None,
        "pass_rate": None,
        "contract_ok": None,
        "gpu_hours": None,
        "wallclock_min": None,
        "status": "no trial",
        "void": False,
    }
    if trial is None:
        return row

    if (trial / "exception.txt").is_file():
        head = (trial / "exception.txt").read_text().strip().splitlines()
        row["status"] = next(
            (ln for ln in reversed(head) if ln.strip() and not ln.startswith(" ")),
            "exception",
        )[:70]
    else:
        row["status"] = "agent ok"

    designs = next(iter(trial.glob("**/submission/designs.csv")), None)
    if designs is not None:
        try:
            row["designs"] = max(sum(1 for _ in designs.open()) - 1, 0)
        except OSError:
            pass

    meta = trial / "controller" / "run_meta.json"
    if not meta.is_file():
        # Compatibility for trials produced before controller-owned accounting.
        meta = next(iter(trial.glob("**/submission/run_meta.json")), None)
    if meta is not None:
        m = _load(meta)
        row["gpu_hours"] = m.get("gpu_hours")
        row["wallclock_min"] = m.get("wallclock_min")

    score = trial / "scoring" / "score.json"
    if score.is_file():
        s = _load(score)
        row["n_pass"] = s.get("n_pass")
        row["pass_rate"] = s.get("pass_rate")
        row["contract_ok"] = s.get("contract_ok")
        # A void run scores 0 for harness reasons; it must never be averaged in.
        row["void"] = bool(s.get("harness_fault"))
        row["status"] = "VOID (harness fault)" if row["void"] else "SCORED"
    elif (trial / "scoring").is_dir():
        row["status"] = "scoring"
    return row


def main() -> None:
    pattern = sys.argv[1] if len(sys.argv) > 1 else ""
    if not RESULTS.is_dir():
        raise SystemExit(f"No results directory: {RESULTS}")
    runs = sorted(p for p in RESULTS.iterdir() if p.is_dir() and pattern in p.name)
    if not runs:
        raise SystemExit(f"No runs matching {pattern!r}")

    rows = [summarise(r) for r in runs]
    head = f"{'run':<42} {'eff':<7} {'des':>4} {'pass':>5} {'rate':>6} {'gpu_h':>7}  status"
    print(head)
    print("-" * len(head))
    for r in rows:
        rate = (
            f"{r['pass_rate']:.3f}" if isinstance(r["pass_rate"], (int, float)) else "-"
        )
        gpu = (
            f"{r['gpu_hours']:.2f}" if isinstance(r["gpu_hours"], (int, float)) else "-"
        )
        print(
            f"{r['run']:<42} {r['effort']:<7} "
            f"{str(r['designs'] or '-'):>4} {str(r['n_pass'] if r['n_pass'] is not None else '-'):>5} "
            f"{rate:>6} {gpu:>7}  {r['status']}"
        )

    scored = [r for r in rows if isinstance(r["n_pass"], int) and not r["void"]]
    voided = [r for r in rows if r["void"]]
    if voided:
        print(f"\n  {len(voided)} run(s) excluded as harness faults, not results.")
    if scored:
        print()
        for eff in ("low", "medium", "high"):
            vals = [r["n_pass"] for r in scored if r["effort"] == eff]
            if vals:
                mean = sum(vals) / len(vals)
                print(
                    f"  {eff:<7} n={len(vals)}  n_pass mean={mean:.1f}  values={vals}"
                )


if __name__ == "__main__":
    main()
