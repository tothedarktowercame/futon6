#!/usr/bin/env python3
"""Side-by-side: heuristic eval vs Bayesian-posterior eval.

Consumes an existing report produced by `eval-grounding-gold.py`
(its `strategy_table` has TP and FP_on_gold_symbols counters per
strategy). Computes per-strategy reliability posteriors as
Beta(1+TP, 1+FP) and prints them alongside the heuristic point
estimates.

This is the first artifact for M-bayesian-structure-learning.md §8.
No new gold runs needed — it re-frames the data we already have so
Joe can decide whether the Bayesian path is worth pursuing further.

Usage:
    python scripts/eval-grounding-bayesian.py \\
        --eval-report data/grounding-gold-eval-p3-full.json \\
        --out data/grounding-bayesian-report.json \\
        --simulated-extra-bindings 5000   # expected info gain projection
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from futon6 import bayesian_grounding as _bg


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-report", type=Path, required=True,
        help="JSON produced by scripts/eval-grounding-gold.py "
             "(must carry `strategy_table` with TP / FP counts).",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("grounding-bayesian-report.json"),
        help="Where to write the bayesian summary",
    )
    parser.add_argument(
        "--ci-level", type=float, default=0.95,
        help="Credible interval level (default 0.95)",
    )
    parser.add_argument(
        "--simulated-extra-bindings", type=int, default=5000,
        help="Hypothetical additional bindings to project expected "
             "info gain over (per strategy). Used to answer 'how "
             "much does the next batch tighten the posterior?'",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    eval_report = json.loads(args.eval_report.read_text(encoding="utf-8"))
    strategy_table = eval_report.get("strategy_table") or {}
    if not strategy_table:
        raise SystemExit(
            f"eval report at {args.eval_report} has no strategy_table; "
            "did you point at the right file?"
        )

    reliabilities = _bg.fit_reliabilities_from_eval_report(strategy_table)

    # Expected info gain projection — same N for every strategy as
    # rough headline; real per-strategy emit volumes would refine.
    proj = _bg.expected_batch_info_gain(
        reliabilities,
        {name: args.simulated_extra_bindings for name in reliabilities},
    )

    rows = []
    for strat, rel in sorted(reliabilities.items(), key=lambda kv: -kv[1].n_observations):
        info = strategy_table[strat]
        heuristic_precision = info.get("precision_on_gold_symbols", 0.0)
        lo, hi = rel.credible_interval(level=args.ci_level)
        rows.append({
            "strategy": strat,
            "n_observations": rel.n_observations,
            "tp": int(info.get("tp", 0)),
            "fp_on_gold_symbols": int(info.get("fp_on_gold_symbols", 0)),
            "heuristic_precision": heuristic_precision,
            "bayesian_alpha": rel.alpha,
            "bayesian_beta": rel.beta,
            "bayesian_mean": rel.mean,
            "ci_low": lo,
            "ci_high": hi,
            "expected_info_gain_per_5k_obs": proj[strat],
        })

    out = {
        "source_eval_report": str(args.eval_report),
        "ci_level": args.ci_level,
        "simulated_extra_bindings": args.simulated_extra_bindings,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    # Printed report
    print(f"[bayesian-eval] from {args.eval_report}")
    print(f"[bayesian-eval] CI level: {int(args.ci_level*100)}%")
    print(f"[bayesian-eval] hypothetical N obs/strategy for info-gain "
          f"projection: {args.simulated_extra_bindings}")
    print()
    header = (
        f"{'strategy':16s}  {'n_obs':>6s}  {'heur P':>7s}  "
        f"{'Bayes mean':>10s}  {f'{int(args.ci_level*100)}% CI':>16s}  "
        f"{'ΔVar per N':>10s}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        ci_str = f"[{r['ci_low']*100:4.1f}%, {r['ci_high']*100:4.1f}%]"
        delta_var = r["expected_info_gain_per_5k_obs"]
        print(
            f"  {r['strategy']:14s}  {r['n_observations']:>6d}  "
            f"{r['heuristic_precision']*100:>6.1f}%  "
            f"{r['bayesian_mean']*100:>9.1f}%  "
            f"{ci_str:>16s}  "
            f"{delta_var:>10.2e}"
        )
    print()
    # Headline: total expected info gain (sum of variance deltas)
    total = sum(r["expected_info_gain_per_5k_obs"] for r in rows)
    print(f"[bayesian-eval] Total expected variance reduction "
          f"({args.simulated_extra_bindings} obs/strategy): {total:.2e}")
    print(f"[bayesian-eval] wrote {args.out}")
    return out


if __name__ == "__main__":
    main()
