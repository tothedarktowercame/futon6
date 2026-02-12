#!/usr/bin/env python3
"""Run post-Direction-D regime-split diagnostics for GPL-H.

Consumes Direction D step rows and evaluates a split strategy:
  - low-gap branch: p90_gap <= 1 + delta_low
  - high-gap branch: mean_gap >= 1 + delta_high
  - trivial branch: min_score <= zero_tol

Outputs summary metrics and optional JSON for wiring/reporting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


def load_rows(path: Path):
    data = json.loads(path.read_text())
    rows = data.get("rows", [])
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return data, rows


def frac(mask: np.ndarray) -> float:
    return float(np.mean(mask))


def summarize(rows, delta_low: float, delta_high: float, zero_tol: float):
    p90 = np.array([r["p90_gap"] for r in rows], dtype=float)
    mean_gap = np.array([r["mean_gap"] for r in rows], dtype=float)
    min_score = np.array([r["min_score"] for r in rows], dtype=float)

    low = p90 <= (1.0 + delta_low)
    high = mean_gap >= (1.0 + delta_high)
    trivial = min_score <= zero_tol

    split_only = low | high
    covered = split_only | trivial

    miss_split = ~split_only
    miss_all = ~covered

    by_family = {}
    families = sorted({r["family"] for r in rows})
    for fam in families:
        idx = np.array([r["family"] == fam for r in rows], dtype=bool)
        by_family[fam] = {
            "rows": int(np.sum(idx)),
            "low_frac": frac(low[idx]),
            "high_frac": frac(high[idx]),
            "trivial_frac": frac(trivial[idx]),
            "split_covered_frac": frac(split_only[idx]),
            "covered_frac": frac(covered[idx]),
            "max_min_score": float(np.max(min_score[idx])),
            "max_mean_gap": float(np.max(mean_gap[idx])),
            "max_p90_gap": float(np.max(p90[idx])),
        }

    hardest_split_misses = []
    if np.any(miss_split):
        miss_rows = [rows[i] for i, m in enumerate(miss_split) if m]
        miss_rows.sort(key=lambda r: (r["min_score"], r["mean_gap"], r["p90_gap"]), reverse=True)
        hardest_split_misses = miss_rows[:12]

    hardest_all_misses = []
    if np.any(miss_all):
        miss_rows = [rows[i] for i, m in enumerate(miss_all) if m]
        miss_rows.sort(key=lambda r: (r["min_score"], r["mean_gap"], r["p90_gap"]), reverse=True)
        hardest_all_misses = miss_rows[:12]

    return {
        "params": {
            "delta_low": delta_low,
            "delta_high": delta_high,
            "zero_tol": zero_tol,
        },
        "counts": {
            "rows": int(len(rows)),
            "low_rows": int(np.sum(low)),
            "high_rows": int(np.sum(high)),
            "trivial_rows": int(np.sum(trivial)),
            "split_covered_rows": int(np.sum(split_only)),
            "covered_rows": int(np.sum(covered)),
            "split_miss_rows": int(np.sum(miss_split)),
            "miss_rows": int(np.sum(miss_all)),
        },
        "fractions": {
            "low": frac(low),
            "high": frac(high),
            "trivial": frac(trivial),
            "split_covered": frac(split_only),
            "covered": frac(covered),
            "split_miss": frac(miss_split),
            "miss": frac(miss_all),
        },
        "miss_stats": {
            "split_miss_max_min_score": float(np.max(min_score[miss_split])) if np.any(miss_split) else None,
            "split_miss_p95_min_score": float(np.quantile(min_score[miss_split], 0.95)) if np.any(miss_split) else None,
            "miss_max_min_score": float(np.max(min_score[miss_all])) if np.any(miss_all) else None,
            "miss_p95_min_score": float(np.quantile(min_score[miss_all], 0.95)) if np.any(miss_all) else None,
        },
        "hardest_split_misses": hardest_split_misses,
        "hardest_misses": hardest_all_misses,
        "by_family": by_family,
    }


def best_grid(rows, low_grid, high_grid):
    p90 = np.array([r["p90_gap"] for r in rows], dtype=float)
    mean_gap = np.array([r["mean_gap"] for r in rows], dtype=float)
    min_score = np.array([r["min_score"] for r in rows], dtype=float)

    candidates = []
    for dl in low_grid:
        low = p90 <= (1.0 + dl)
        for dh in high_grid:
            high = mean_gap >= (1.0 + dh)
            split = low | high
            miss = ~split
            candidates.append({
                "delta_low": float(dl),
                "delta_high": float(dh),
                "split_covered_frac": float(np.mean(split)),
                "low_frac": float(np.mean(low)),
                "high_frac": float(np.mean(high)),
                "split_miss_rows": int(np.sum(miss)),
                "split_miss_max_min_score": float(np.max(min_score[miss])) if np.any(miss) else None,
            })

    candidates.sort(
        key=lambda c: (
            c["split_covered_frac"],
            -(abs(c["low_frac"] - 0.5) + abs(c["high_frac"] - 0.5)),
            -c["delta_low"],
            -c["delta_high"],
        ),
        reverse=True,
    )
    return candidates[:15]


def main() -> None:
    ap = argparse.ArgumentParser(description="Post-D regime split diagnostics")
    ap.add_argument("--input", default="data/first-proof/problem6-direction-d-results.json")
    ap.add_argument("--delta-low", type=float, default=0.10)
    ap.add_argument("--delta-high", type=float, default=0.10)
    ap.add_argument("--zero-tol", type=float, default=1e-12)
    ap.add_argument("--output", default="data/first-proof/problem6-post-d-regime-split-results.json")
    ap.add_argument("--scan", action="store_true", help="Also scan a small threshold grid")
    args = ap.parse_args()

    in_path = Path(args.input)
    data, rows = load_rows(in_path)

    summary = summarize(rows, args.delta_low, args.delta_high, args.zero_tol)
    out = {
        "source": str(in_path),
        "source_case2b_instances": data.get("case2b_instances"),
        "summary": summary,
    }

    if args.scan:
        out["grid_top"] = best_grid(
            rows,
            low_grid=[0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20],
            high_grid=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    s = summary
    print("=" * 84)
    print("Post-D Regime Split Summary")
    print("=" * 84)
    print(f"Source rows:      {s['counts']['rows']}")
    print(f"low branch rows:  {s['counts']['low_rows']} ({s['fractions']['low']:.4f})")
    print(f"high branch rows: {s['counts']['high_rows']} ({s['fractions']['high']:.4f})")
    print(f"split coverage:   {s['counts']['split_covered_rows']} ({s['fractions']['split_covered']:.4f})")
    print(f"trivial rows:     {s['counts']['trivial_rows']} ({s['fractions']['trivial']:.4f})")
    print(f"total coverage:   {s['counts']['covered_rows']} ({s['fractions']['covered']:.4f})")
    print(f"split misses:     {s['counts']['split_miss_rows']} (max min_score = {s['miss_stats']['split_miss_max_min_score']})")
    print(f"total misses:     {s['counts']['miss_rows']}")

    if args.scan and out.get("grid_top"):
        print("\nTop grid candidates (split coverage only):")
        for c in out["grid_top"][:8]:
            print(
                f"  dl={c['delta_low']:.2f} dh={c['delta_high']:.2f} "
                f"split={c['split_covered_frac']:.4f} low={c['low_frac']:.4f} "
                f"high={c['high_frac']:.4f} miss_rows={c['split_miss_rows']}"
            )

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
