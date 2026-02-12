#!/usr/bin/env python3
"""Post-D diagnostics with a theorem-grade ratio certificate.

Key deterministic lemma used by this script:
Let s_v = ||Y_t(v)||, d_v = tr(Y_t(v)), g_v = d_v/s_v on active vertices.
Define m = min_v s_v, dbar = avg_v d_v, gbar = avg_v g_v.
Then m <= dbar / gbar.
Hence dbar/gbar < 1 certifies a good step.
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
    required = {"mean_gap", "mean_drift", "min_score", "p90_gap", "family"}
    missing = [k for k in required if k not in rows[0]]
    if missing:
        raise ValueError(
            f"Input rows missing fields {missing}. Re-run verify-p6-gpl-h-direction-d.py "
            "with the current version."
        )
    return data, rows


def frac(mask: np.ndarray) -> float:
    return float(np.mean(mask))


def summarize(rows, delta_low: float, delta_high: float, zero_tol: float, ratio_tol: float):
    p90 = np.array([r["p90_gap"] for r in rows], dtype=float)
    mean_gap = np.array([r["mean_gap"] for r in rows], dtype=float)
    mean_drift = np.array([r["mean_drift"] for r in rows], dtype=float)
    min_score = np.array([r["min_score"] for r in rows], dtype=float)

    ratio = mean_drift / np.maximum(mean_gap, 1e-15)

    low = p90 <= (1.0 + delta_low)
    high = mean_gap >= (1.0 + delta_high)
    trivial = min_score <= zero_tol

    ratio_cert = ratio < (1.0 - ratio_tol)
    ratio_or_trivial = ratio_cert | trivial

    # Legacy split coverage (for continuity with previous post-D runs)
    split_only = low | high
    split_plus_trivial = split_only | trivial

    miss_ratio = ~ratio_or_trivial
    miss_split = ~split_plus_trivial

    nontrivial = ~trivial

    by_family = {}
    families = sorted({r["family"] for r in rows})
    for fam in families:
        idx = np.array([r["family"] == fam for r in rows], dtype=bool)
        by_family[fam] = {
            "rows": int(np.sum(idx)),
            "low_frac": frac(low[idx]),
            "high_frac": frac(high[idx]),
            "trivial_frac": frac(trivial[idx]),
            "ratio_cert_frac": frac(ratio_cert[idx]),
            "ratio_or_trivial_frac": frac(ratio_or_trivial[idx]),
            "split_plus_trivial_frac": frac(split_plus_trivial[idx]),
            "max_min_score": float(np.max(min_score[idx])),
            "max_ratio": float(np.max(ratio[idx])),
            "p95_ratio": float(np.quantile(ratio[idx], 0.95)),
        }

    hardest_ratio_misses = []
    if np.any(miss_ratio):
        miss_rows = [rows[i] for i, m in enumerate(miss_ratio) if m]
        miss_rows.sort(
            key=lambda r: (
                r["min_score"],
                r["mean_drift"] / max(r["mean_gap"], 1e-15),
                r["mean_gap"],
            ),
            reverse=True,
        )
        hardest_ratio_misses = miss_rows[:12]

    hardest_split_misses = []
    if np.any(miss_split):
        miss_rows = [rows[i] for i, m in enumerate(miss_split) if m]
        miss_rows.sort(key=lambda r: (r["min_score"], r["mean_gap"], r["p90_gap"]), reverse=True)
        hardest_split_misses = miss_rows[:12]

    return {
        "params": {
            "delta_low": delta_low,
            "delta_high": delta_high,
            "zero_tol": zero_tol,
            "ratio_tol": ratio_tol,
        },
        "counts": {
            "rows": int(len(rows)),
            "nontrivial_rows": int(np.sum(nontrivial)),
            "low_rows": int(np.sum(low)),
            "high_rows": int(np.sum(high)),
            "trivial_rows": int(np.sum(trivial)),
            "ratio_cert_rows": int(np.sum(ratio_cert)),
            "ratio_or_trivial_rows": int(np.sum(ratio_or_trivial)),
            "split_plus_trivial_rows": int(np.sum(split_plus_trivial)),
            "ratio_miss_rows": int(np.sum(miss_ratio)),
            "split_miss_rows": int(np.sum(miss_split)),
            "nontrivial_ratio_fail_rows": int(np.sum(nontrivial & ~ratio_cert)),
        },
        "fractions": {
            "nontrivial": frac(nontrivial),
            "low": frac(low),
            "high": frac(high),
            "trivial": frac(trivial),
            "ratio_cert": frac(ratio_cert),
            "ratio_or_trivial": frac(ratio_or_trivial),
            "split_plus_trivial": frac(split_plus_trivial),
            "ratio_miss": frac(miss_ratio),
            "split_miss": frac(miss_split),
            "nontrivial_ratio_fail": float(np.mean((nontrivial & ~ratio_cert)[nontrivial])) if np.any(nontrivial) else 0.0,
        },
        "ratio_stats": {
            "ratio_mean": float(np.mean(ratio)),
            "ratio_p95": float(np.quantile(ratio, 0.95)),
            "ratio_max": float(np.max(ratio)),
            "nontrivial_ratio_max": float(np.max(ratio[nontrivial])) if np.any(nontrivial) else None,
            "nontrivial_ratio_p95": float(np.quantile(ratio[nontrivial], 0.95)) if np.any(nontrivial) else None,
        },
        "miss_stats": {
            "ratio_miss_max_min_score": float(np.max(min_score[miss_ratio])) if np.any(miss_ratio) else None,
            "split_miss_max_min_score": float(np.max(min_score[miss_split])) if np.any(miss_split) else None,
        },
        "hardest_ratio_misses": hardest_ratio_misses,
        "hardest_split_misses": hardest_split_misses,
        "by_family": by_family,
    }


def best_grid(rows, low_grid, high_grid):
    p90 = np.array([r["p90_gap"] for r in rows], dtype=float)
    mean_gap = np.array([r["mean_gap"] for r in rows], dtype=float)

    candidates = []
    for dl in low_grid:
        low = p90 <= (1.0 + dl)
        for dh in high_grid:
            high = mean_gap >= (1.0 + dh)
            split = low | high
            candidates.append({
                "delta_low": float(dl),
                "delta_high": float(dh),
                "split_covered_frac": float(np.mean(split)),
                "low_frac": float(np.mean(low)),
                "high_frac": float(np.mean(high)),
                "split_miss_rows": int(np.sum(~split)),
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
    ap.add_argument("--ratio-tol", type=float, default=1e-12)
    ap.add_argument("--output", default="data/first-proof/problem6-post-d-regime-split-results.json")
    ap.add_argument("--scan", action="store_true", help="Also scan a small threshold grid")
    args = ap.parse_args()

    in_path = Path(args.input)
    data, rows = load_rows(in_path)

    summary = summarize(rows, args.delta_low, args.delta_high, args.zero_tol, args.ratio_tol)
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
    print(f"Source rows:              {s['counts']['rows']}")
    print(f"Nontrivial rows:          {s['counts']['nontrivial_rows']} ({s['fractions']['nontrivial']:.4f})")
    print(f"low branch rows:          {s['counts']['low_rows']} ({s['fractions']['low']:.4f})")
    print(f"high branch rows:         {s['counts']['high_rows']} ({s['fractions']['high']:.4f})")
    print(f"trivial rows:             {s['counts']['trivial_rows']} ({s['fractions']['trivial']:.4f})")
    print(f"ratio cert rows:          {s['counts']['ratio_cert_rows']} ({s['fractions']['ratio_cert']:.4f})")
    print(f"ratio or trivial coverage:{s['counts']['ratio_or_trivial_rows']} ({s['fractions']['ratio_or_trivial']:.4f})")
    print(f"split+trivial coverage:   {s['counts']['split_plus_trivial_rows']} ({s['fractions']['split_plus_trivial']:.4f})")
    print(f"ratio misses:             {s['counts']['ratio_miss_rows']} (max min_score = {s['miss_stats']['ratio_miss_max_min_score']})")
    print(f"nontrivial ratio fails:   {s['counts']['nontrivial_ratio_fail_rows']} ({s['fractions']['nontrivial_ratio_fail']:.4f} of nontrivial)")
    print(f"ratio max (all/nontrivial): {s['ratio_stats']['ratio_max']:.6f} / {s['ratio_stats']['nontrivial_ratio_max']:.6f}")

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
