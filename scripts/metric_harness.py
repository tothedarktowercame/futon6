#!/usr/bin/env python3
"""Metric harness — M-metric-harness INSTANTIATE, CPU side.

Computes the PROGRESS slope of accretion metrics via the DERIVE leave-one-out
mechanism: hold out a fixed set, sweep substrate size k=1..N, measure each registered
metric, emit a SlopeReport per metric (points, rise@1→10, rising?, attribution stage).

Two paper universes, as the pipeline really has (MAP finding):
  - SUBSTRATE corpus — the concept-index (~9.7k CT papers); grounding metrics sweep it.
  - MINED corpus — the CLeans we've actually extracted (~100); mined-corpus metrics
    (recurring holes, …) sweep it, grounding concepts against the substrate.
Metrics declare `requires` (ctx keys) + `corpus`; run() computes only those whose data
is present and sweeps the matching universe. New metrics = `@metric` + (maybe) a loader.

  futon6/.venv/bin/python scripts/metric_harness.py            # both corpora if present
  futon6/.venv/bin/python scripts/metric_harness.py --self-test
"""
import argparse
import glob
import json
import os
import re
import statistics
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- metric registry --------------------------------------------------------
# {name, stage, axis, kind:'held-out'|'corpus', corpus:'substrate'|'mined', requires:(keys), fn}
#   held-out fn(held_paper, subctx, ctx) -> float   (mean over held-out)
#   corpus   fn(subctx, ctx) -> float
# subctx always has {'papers':list,'n':int}; +{'concepts','defined'} (substrate),
#   +{'holekey_df':Counter} (mined) when the ctx provides the source data.
METRICS = []


def metric(name, stage, axis, kind, corpus, requires):
    def deco(fn):
        METRICS.append({"name": name, "stage": stage, "axis": axis, "kind": kind,
                        "corpus": corpus, "requires": tuple(requires), "fn": fn})
        return fn
    return deco


@metric("concept-coverage", "S2", "accretion", "held-out", "substrate", ["paper2c"])
def _coverage(h, subctx, ctx):
    ch = ctx["paper2c"][h]
    return len(ch & subctx["concepts"]) / max(1, len(ch))


@metric("encyclopedia-defined", "S2", "accretion", "corpus", "substrate", ["paper2def"])
def _encyclopedia(subctx, ctx):
    return float(len(subctx["defined"]))


@metric("recurring-holes", "S5", "accretion", "corpus", "mined", ["paper2holekeys"])
def _recurring_holes(subctx, ctx):
    # (type,concept) hole-keys appearing in >= 2 papers of the substrate (df>=2).
    return float(sum(1 for c in subctx["holekey_df"].values() if c >= 2))


# --- loaders ----------------------------------------------------------------
def load_concept_index(path, ctx=None):
    if ctx is None:
        ctx = {}
    ci = json.load(open(path))
    paper2c, paper2def = defaultdict(set), defaultdict(set)
    for c, rec in ci.items():
        for p in rec.get("papers", []):
            paper2c[p].add(c)
            if rec.get("defined"):
                paper2def[p].add(c)
    ctx["paper2c"] = paper2c
    ctx["paper2def"] = paper2def
    ctx["_concept_index"] = ci
    ctx["substrate_universe"] = sorted(paper2c)
    return ctx


def _concept_matcher(ci):
    # one regex over genuine concept names (longest-first), for grounding hole-box text
    names = sorted((c for c, r in ci.items() if r.get("genuine")), key=len, reverse=True)
    names = [n for n in names if len(n) >= 4][:6000]
    return re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\b") if names else None


def load_clean_holes(clean_dir, ctx):
    import edn_format as e
    rx = _concept_matcher(ctx.get("_concept_index", {})) if ctx.get("_concept_index") else None
    paper2holekeys = {}
    for f in sorted(glob.glob(os.path.join(clean_dir, "*.clean.edn"))):
        pid = os.path.basename(f)[:-len(".clean.edn")]
        try:
            d = {str(k): v for k, v in dict(e.loads(open(f).read())).items()}
        except Exception:
            continue
        boxes = d.get(":clean/boxes") or []
        keys = set()
        for b in boxes:
            bb = {str(k): v for k, v in dict(b).items()}
            hole = bb.get(":hole")
            if hole is None:
                continue
            hd = {str(k): v for k, v in dict(hole).items()}
            sat = str(hd.get(":satiety", "?")).lstrip(":")
            text = str(bb.get(":text", "")).lower()
            concepts = set(rx.findall(text)) if rx else set()
            if concepts:
                for c in concepts:
                    keys.add((sat, c))
            else:  # fallback: type + method (coarser, still cross-paper)
                keys.add((sat, "method:" + str(bb.get(":method", "?")).lstrip(":")))
        paper2holekeys[pid] = keys
    ctx["paper2holekeys"] = paper2holekeys
    ctx["mined_universe"] = sorted(paper2holekeys)
    return ctx


# --- leave-one-out slope replay ---------------------------------------------
def kgrid(n):
    return sorted({k for k in (1, 2, 5, 10, 20, 50, 100, 200, 400, n) if 1 <= k <= n})


def held_out(papers, n_held):
    step = max(1, len(papers) // max(1, n_held))
    return papers[::step][:n_held]


def _subctx(sub, ctx):
    sc = {"papers": sub, "n": len(sub)}
    if "paper2c" in ctx:
        u = set()
        for p in sub:
            u |= ctx["paper2c"][p]
        sc["concepts"] = u
    if "paper2def" in ctx:
        u = set()
        for p in sub:
            u |= ctx["paper2def"][p]
        sc["defined"] = u
    if "paper2holekeys" in ctx:
        df = Counter()
        for p in sub:
            for key in ctx["paper2holekeys"].get(p, ()):
                df[key] += 1
        sc["holekey_df"] = df
    return sc


def run(ctx, n_held=20):
    reports = []
    for cdom, ukey in (("substrate", "substrate_universe"), ("mined", "mined_universe")):
        if ukey not in ctx:
            continue
        ms = [m for m in METRICS if m["corpus"] == cdom and all(r in ctx for r in m["requires"])]
        if not ms:
            continue
        universe = ctx[ukey]
        held = held_out(universe, n_held)
        heldset = set(held)
        pool = [p for p in universe if p not in heldset]
        grid = kgrid(len(pool))
        subctx = {k: _subctx(pool[:k], ctx) for k in grid}
        for m in ms:
            pts = []
            for k in grid:
                sc = subctx[k]
                v = (statistics.mean(m["fn"](h, sc, ctx) for h in held)
                     if m["kind"] == "held-out" else m["fn"](sc, ctx))
                pts.append([k, v])
            v1, vmax = pts[0][1], pts[-1][1]
            v10 = next((v for k, v in pts if k >= 10), vmax)
            rising = all(pts[i][1] <= pts[i + 1][1] + 1e-9 for i in range(len(pts) - 1))
            reports.append({"metric": m["name"], "stage": m["stage"], "axis": m["axis"],
                            "corpus": cdom, "points": pts, "v@1": round(v1, 4),
                            "v@10": round(v10, 4), "v@max": round(vmax, 4),
                            "rise_1_to_10": round(v10 - v1, 4), "rising": rising,
                            "attribution_stage": m["stage"], "n_held": len(held), "n_pool": len(pool)})
    return reports


def print_reports(reports):
    for r in reports:
        print(f"\n[{r['stage']} · {r['axis']} · {r['corpus']}] {r['metric']}  "
              f"(rising={r['rising']}, rise@1→10={r['rise_1_to_10']}, attribution={r['attribution_stage']})")
        print("  k:   " + " ".join(f"{k:>7}" for k, _ in r["points"]))
        print("  val: " + " ".join(f"{v:>7.3f}" for _, v in r["points"]))


def self_test():
    ci = {f"c{i}": {"defined": True, "genuine": True, "papers": [f"p{j:03d}" for j in range(i, i + 5)]}
          for i in range(30)}
    import tempfile
    p = tempfile.mktemp(suffix=".json")
    json.dump(ci, open(p, "w"))
    reps = run(load_concept_index(p), n_held=5)
    cov = next(r for r in reps if r["metric"] == "concept-coverage")
    enc = next(r for r in reps if r["metric"] == "encyclopedia-defined")
    assert cov["v@max"] > cov["v@1"], "coverage should rise"
    assert enc["v@max"] > enc["v@1"], "encyclopedia should accrete"
    reps2 = json.loads(json.dumps({"reports": reps}))["reports"]
    assert reps2[0]["points"] == reps[0]["points"], "round-trip"
    os.unlink(p)
    print(f"self-test PASS: coverage {cov['v@1']}→{cov['v@max']}, encyclopedia {enc['v@1']:.0f}→{enc['v@max']:.0f}, round-trip ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept-index", default="data/warp/concept-index.json")
    ap.add_argument("--clean-dir", default="data/mark5-ct100-run/holes/clean-ct200")
    ap.add_argument("--n-held", type=int, default=20)
    ap.add_argument("--out", default="data/metric-harness-report.json")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    ctx = {}
    cip = a.concept_index if os.path.isabs(a.concept_index) else os.path.join(ROOT, a.concept_index)
    if os.path.exists(cip):
        load_concept_index(cip, ctx)
    cd = a.clean_dir if os.path.isabs(a.clean_dir) else os.path.join(ROOT, a.clean_dir)
    if os.path.isdir(cd):
        load_clean_holes(cd, ctx)
    reps = run(ctx, a.n_held)
    print_reports(reps)
    outp = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    json.dump({"reports": reps}, open(outp, "w"), indent=1)
    print(f"\nwrote {a.out}  ({len(reps)} metric(s))")


if __name__ == "__main__":
    main()
