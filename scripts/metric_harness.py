#!/usr/bin/env python3
"""Metric harness — M-metric-harness INSTANTIATE (1), CPU side.

Computes the PROGRESS slope of accretion metrics via the DERIVE leave-one-out
mechanism: hold out a fixed set of papers, sweep substrate size k=1..N, measure each
registered metric, and emit a SlopeReport per metric (points, rise@1→10, rising?,
attribution stage). Pluggable metric registry — seeded with the two accretion metrics
computable on data in hand (concept-coverage, encyclopedia-defined). New metrics
(comprehension, expository coverage, …) register with their stage + a fn reading their
data from `ctx`. The slope report is the progress artifact (and the superpod go/no-go
input). See holes/missions/M-metric-harness.md (DERIVE/VERIFY).

  futon6/.venv/bin/python scripts/metric_harness.py            # run on data/warp/concept-index.json
  futon6/.venv/bin/python scripts/metric_harness.py --self-test
"""
import argparse
import json
import os
import statistics
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- metric registry --------------------------------------------------------
# each metric: {name, stage, axis, kind: 'held-out'|'corpus', fn}
#   held-out  fn(held_paper, subctx, ctx) -> float   (mean over the held-out set)
#   corpus    fn(subctx, ctx) -> float
# subctx = {'concepts': set, 'defined': set, 'papers': list, 'n': int} for substrate size k
METRICS = []


def metric(name, stage, axis, kind):
    def deco(fn):
        METRICS.append({"name": name, "stage": stage, "axis": axis, "kind": kind, "fn": fn})
        return fn
    return deco


@metric("concept-coverage", "S2", "accretion", "held-out")
def _coverage(h, subctx, ctx):
    ch = ctx["paper2c"][h]
    return len(ch & subctx["concepts"]) / max(1, len(ch))


@metric("encyclopedia-defined", "S2", "accretion", "corpus")
def _encyclopedia(subctx, ctx):
    return float(len(subctx["defined"]))


# --- context loaders (pluggable; more stages add keys to ctx later) ---------
def load_concept_index(path):
    ci = json.load(open(path))
    paper2c, paper2def = defaultdict(set), defaultdict(set)
    for c, rec in ci.items():
        for p in rec.get("papers", []):
            paper2c[p].add(c)
            if rec.get("defined"):
                paper2def[p].add(c)
    return {"paper2c": paper2c, "paper2def": paper2def, "n_concepts": len(ci)}


# --- leave-one-out slope replay ---------------------------------------------
def kgrid(n):
    return sorted({k for k in (1, 2, 5, 10, 20, 50, 100, 200, 400, n) if 1 <= k <= n})


def held_out(papers, n_held):
    step = max(1, len(papers) // max(1, n_held))
    return papers[::step][:n_held]


def run(ctx, n_held=20):
    papers = sorted(ctx["paper2c"])
    held = held_out(papers, n_held)
    heldset = set(held)
    pool = [p for p in papers if p not in heldset]
    grid = kgrid(len(pool))
    # build each substrate context once per k (shared across held-out papers + metrics)
    subctx = {}
    for k in grid:
        sub = pool[:k]
        conc, dfn = set(), set()
        for p in sub:
            conc |= ctx["paper2c"][p]
            dfn |= ctx["paper2def"][p]
        subctx[k] = {"concepts": conc, "defined": dfn, "papers": sub, "n": k}
    reports = []
    for m in METRICS:
        pts = []
        for k in grid:
            sc = subctx[k]
            if m["kind"] == "held-out":
                v = statistics.mean(m["fn"](h, sc, ctx) for h in held)
            else:
                v = m["fn"](sc, ctx)
            pts.append([k, v])
        v1 = pts[0][1]
        v10 = next((v for k, v in pts if k >= 10), pts[-1][1])
        vmax = pts[-1][1]
        rising = all(pts[i][1] <= pts[i + 1][1] + 1e-9 for i in range(len(pts) - 1))
        reports.append({
            "metric": m["name"], "stage": m["stage"], "axis": m["axis"], "kind": m["kind"],
            "points": pts, "v@1": round(v1, 4), "v@10": round(v10, 4), "v@max": round(vmax, 4),
            "rise_1_to_10": round(v10 - v1, 4), "rising": rising,
            "attribution_stage": m["stage"], "n_held": len(held), "n_pool": len(pool),
        })
    return reports


def print_reports(reports):
    for r in reports:
        print(f"\n[{r['stage']} · {r['axis']}] {r['metric']}  "
              f"(rising={r['rising']}, rise@1→10={r['rise_1_to_10']}, attribution={r['attribution_stage']})")
        print("  k:   " + " ".join(f"{k:>7}" for k, _ in r["points"]))
        print("  val: " + " ".join(f"{v:>7.3f}" for _, v in r["points"]))


def self_test():
    # synthetic: 30 concepts, each spanning 5 consecutive papers -> coverage must rise with k
    ci = {f"c{i}": {"defined": True, "papers": [f"p{j:03d}" for j in range(i, i + 5)]} for i in range(30)}
    import tempfile
    p = tempfile.mktemp(suffix=".json")
    json.dump(ci, open(p, "w"))
    reps = run(load_concept_index(p), n_held=5)
    cov = next(r for r in reps if r["metric"] == "concept-coverage")
    enc = next(r for r in reps if r["metric"] == "encyclopedia-defined")
    assert cov["v@max"] > cov["v@1"], f"coverage should rise overall, got {cov['v@1']}->{cov['v@max']}"
    assert enc["v@max"] > enc["v@1"], "encyclopedia count should accrete"
    # round-trip: SlopeReport -> JSON -> back
    reps2 = json.loads(json.dumps({"reports": reps}))["reports"]
    assert reps2[0]["metric"] == reps[0]["metric"] and reps2[0]["points"] == reps[0]["points"]
    os.unlink(p)
    print(f"self-test PASS: coverage rose {cov['v@1']}→{cov['v@max']}, "
          f"encyclopedia {enc['v@1']:.0f}→{enc['v@max']:.0f}, round-trip ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept-index", default="data/warp/concept-index.json")
    ap.add_argument("--n-held", type=int, default=20)
    ap.add_argument("--out", default="data/metric-harness-report.json")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    cip = a.concept_index if os.path.isabs(a.concept_index) else os.path.join(ROOT, a.concept_index)
    reps = run(load_concept_index(cip), a.n_held)
    print_reports(reps)
    outp = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    json.dump({"reports": reps}, open(outp, "w"), indent=1)
    print(f"\nwrote {a.out}  ({len(reps)} metrics, {reps[0]['n_pool']} pool / {reps[0]['n_held']} held-out)")


if __name__ == "__main__":
    main()
