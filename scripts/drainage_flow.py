#!/usr/bin/env python3
"""Drainage / flow layer — ON TOP of the paleo-topography (paleo_topography.py).

Reads data/paleo-topography.json (the commit-pinned terrain, NEVER live HEAD —
trap 3 is already discharged by that layer). Overlays, per drained basin:

  1. FLOW (discharge): how much actually drained — the verified closing artifacts
     the realized route carried from have to want (the retrodictive witness).

  2. COUNTERFACTUAL ROUTES: alternative drainage networks reachable in the SAME
     paleo terrain — the realized routes of sibling basins that (a) share >=1
     pattern with this basin (reachable in the same channel system) and (b)
     drained AT-OR-BEFORE this basin (existed in the terrain at the time; a route
     minted later was unreachable then — trap 3 again, at route granularity).

  3. CALIBRATION FLAGS (trap 1, the load-bearing discipline). The realized route
     has an OUTCOME; counterfactuals have only metric (curvature) scores. So this
     emits CALIBRATION FLAGS, never training labels, never a :better-route:
       :metric-disconfirmed-by-drainage  the metric scored the realized route
            POORLY (net-bottlenecked channel, median kappa < 0) yet it DRAINED
            -> THE METRIC owes an update. This is the one honest training signal.
       :metric-prefers-alt  some reachable sibling route scores higher under the
            metric. This does NOT mean history was suboptimal — the alt is
            untested for THIS hole; retrodiction cannot say whether the metric or
            the route is wrong. A flag for a human/metric review, not a label.
       :metric-concordant  metric likes the realized route (kappa >= 0) and it
            drained — terrain and history agree.
       :unroutable  no realized pattern-cite chain mined (skeletal basin).

  CH1 SELF-REFERENCE GUARD: metric-preferred counterfactuals are explicitly NOT
  emitted as better outcomes and must never be fed into a prior as if they were
  (that would teach the prior the metric's counterfactual tastes, not reality's).
  Only :metric-disconfirmed-by-drainage carries a learning signal, and it points
  AT THE METRIC.

Trap 2 (censored training set): this scores DRAINED basins only — won games. The
honest negative class (dry basins / never-closed holes) lives in data/dry-basins/
and is REPORTED here (counts) but not modelled this pass (bounded excursion).

Usage: python3 scripts/drainage_flow.py [--paleo FILE] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

FUTON6 = Path("/home/joe/code/futon6")
KAPPA_BOTTLENECK = 0.0    # median kappa < 0 => net-bottlenecked channel
PREFERS_EPS = 0.10        # a sibling must beat the realized kappa by this margin


def pattern_set(basin: dict) -> set:
    return {r["ident"] for r in basin["realized_route"] if r.get("ident")}


def realized_kappa(basin: dict):
    k = basin["terrain"].get("kappa")
    return k["median"] if k else None


def classify(basin: dict, siblings: list[dict]) -> dict:
    """Emit the calibration flag for one drained basin. siblings = reachable
    counterfactual basins (pattern-overlap, drained at-or-before)."""
    rk = realized_kappa(basin)
    drained = (basin["outcome"]["verified_frac"] or 0.0) >= 0.5
    if not basin["realized_route"]:
        return {"flag": "unroutable", "realized_kappa": rk,
                "note": "no realized pattern-cite chain to score"}

    # best reachable counterfactual by the metric (curvature median)
    scored = [{"basin": s["basin"], "kappa": realized_kappa(s),
               "shared": sorted(pattern_set(basin) & pattern_set(s)),
               "drained_at": s["drainage_pin"]["at"]}
              for s in siblings if realized_kappa(s) is not None]
    scored.sort(key=lambda s: s["kappa"], reverse=True)
    best = scored[0] if scored else None

    # (1) the honest training signal: metric disconfirmed by the drainage fact
    if rk is not None and rk < KAPPA_BOTTLENECK and drained:
        return {"flag": "metric-disconfirmed-by-drainage", "realized_kappa": rk,
                "owes": "metric",
                "note": f"realized route is a net-bottlenecked channel "
                        f"(median kappa {rk:.3f} < 0) yet it drained "
                        f"(verified {basin['outcome']['verified_frac']}). "
                        f"The METRIC owes an update.",
                "counterfactuals": scored[:5]}

    # (2) a reachable sibling the metric prefers — CALIBRATION FLAG ONLY
    if best and rk is not None and best["kappa"] > rk + PREFERS_EPS:
        return {"flag": "metric-prefers-alt", "realized_kappa": rk,
                "metric_prefers": best,
                "note": "a reachable sibling route scores higher under the metric. "
                        "NOT a better-route label: the alt is untested for this "
                        "hole; retrodiction cannot say if the metric or the route "
                        "is wrong. Review flag, never a training label.",
                "counterfactuals": scored[:5]}

    # (3) metric and drainage agree
    return {"flag": "metric-concordant", "realized_kappa": rk,
            "note": "metric likes the realized route (kappa >= 0) and it drained.",
            "counterfactuals": scored[:5]}


def negative_class_report() -> dict:
    """Trap 2: the censored complement. Report it; do not model it this pass."""
    summ = FUTON6 / "data" / "dry-basins" / "_summary.json"
    if summ.exists():
        s = json.loads(summ.read_text())
        return {"present": True,
                "dry_basins": s.get("dry-basins"),
                "skipped_not_completed": s.get("skipped-not-completed"),
                "closure_failure_seeds": s.get("closure-failure-seeds"),
                "note": "the honest negative class (never-drained basins) — "
                        "REPORTED, not modelled this excursion (trap 2, bounded)."}
    return {"present": False,
            "note": "data/dry-basins/_summary.json absent; run dry_basin_miner.py"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paleo", default=str(FUTON6 / "data" / "paleo-topography.json"))
    ap.add_argument("--out", default=str(FUTON6 / "data" / "drainage-flow.json"))
    args = ap.parse_args()

    paleo = json.loads(Path(args.paleo).read_text())
    basins = paleo["basins"]

    flows = []
    for b in basins:
        # reachable counterfactuals: share a pattern AND drained at-or-before
        # (a route minted later was not in this paleo terrain — trap 3 at route grain)
        bt = b["drainage_pin"]["at"]
        bp = pattern_set(b)
        siblings = [s for s in basins
                    if s["basin"] != b["basin"] and pattern_set(s) & bp
                    and (s["drainage_pin"]["at"] and bt
                         and s["drainage_pin"]["at"] <= bt)]
        cls = classify(b, siblings)
        flows.append({
            "basin": b["basin"],
            "drained_at": bt,
            "hole": b["hole"],
            "realized_route": [r["ident"] for r in b["realized_route"]],
            "discharge": {                       # FLOW: what actually drained
                "verified_artifacts": b["outcome"]["verified"],
                "total_artifacts": b["outcome"]["artifacts"],
                "verified_frac": b["outcome"]["verified_frac"],
                "route_len": b["route_len"],
            },
            "n_reachable_counterfactuals": len(siblings),
            **cls,
        })

    counts: dict[str, int] = {}
    for f in flows:
        counts[f["flag"]] = counts.get(f["flag"], 0) + 1
    report = {
        "schema": "drainage-flow-v1",
        "reads": "data/paleo-topography.json (paleo terrain, not live HEAD)",
        "flag_discipline": "trap 1: calibration flags, NEVER a :better-route "
                          "label. Only :metric-disconfirmed-by-drainage carries a "
                          "learning signal, and it points at the metric. "
                          "metric-preferred counterfactuals are NOT better outcomes "
                          "(CH1 self-reference guard).",
        "metric": "Ollivier-Ricci curvature median over the cascade channel "
                  "(higher = smoother/less bottlenecked); thresholds "
                  f"bottleneck<{KAPPA_BOTTLENECK}, prefers-margin {PREFERS_EPS}.",
        "n_basins": len(flows),
        "flag_counts": counts,
        "total_discharge_verified": sum(f["discharge"]["verified_artifacts"] for f in flows),
        "negative_class": negative_class_report(),
        "flows": flows,
    }
    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print(f"drainage-flow: {len(flows)} basins; flags {counts}")
    print(f"  total verified discharge: {report['total_discharge_verified']} artifacts")
    nc = report["negative_class"]
    if nc["present"]:
        print(f"  negative class (trap 2, reported not modelled): "
              f"{nc['dry_basins']} dry basins / {nc['skipped_not_completed']} skipped")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
