#!/usr/bin/env python3
"""Pass-3 v0: harvest typed holes from the existing IATC argument-graphs.

Pure CPU, zero LLM calls — runs on the graphs already produced (GPU already
spent). Demonstrates the Pass-3 value layer Joe/Rob want:
  - WEAK PROOFS: proofs with a high unfilled-warrant ratio (reasoning the author
    elided) — candidates for reconstruction.
  - HOLE INVENTORY: the distinct :wanted missing-warrants across the corpus —
    the things a fill-by-retrieval step would target.
  - RECURRING GAPS: a :wanted that appears in >=2 different papers (Joe's
    "two usages" bar, applied at the REASONING layer = demand-side df). Filling
    one of these helps multiple proofs — the highest-value fill targets.

Usage:
  futon6/.venv/bin/python scripts/clean_hole_harvest.py \
      [--graphs data/iatc-argument-graphs] [--out data/showcases/clean-demo/pass3-holes.json]
"""
import argparse
import glob
import json
import os
import edn_format as edn
from clean_structure_embed import kw


def load(path):
    m = edn.loads(open(path).read())
    d = {kw(k): v for k, v in dict(m).items()}
    edges = []
    for e in d.get("edges", []):
        ed = {kw(k): v for k, v in dict(e).items()}
        if kw(ed.get("kind")) != "infer":
            continue
        w = ed.get("warrant")
        wd = {kw(k): v for k, v in dict(w).items()} if w is not None else {}
        edges.append({"missing": kw(wd.get("kind")) == "missing-warrant",
                      "wanted": kw(wd.get("wanted")) if wd.get("wanted") is not None else None})
    n_claims = len(d.get("nodes", []))
    return n_claims, edges


def main():
    ap = argparse.ArgumentParser()
    # Default was the GLOBAL graph tree (recursive), so this read every run ever
    # made rather than the run in hand. The stepper now passes the run dir.
    ap.add_argument("--graphs", default="data/iatc-argument-graphs")
    # Default was a shared demo path outside any run directory, so the product
    # was not a run artifact and RETRIEVE never collected it.
    ap.add_argument("--out", default="data/showcases/clean-demo/pass3-holes.json")
    args = ap.parse_args()

    files = [f for f in glob.glob(os.path.join(args.graphs, "**", "*.edn"), recursive=True)
             if "/.attempts/" not in f and "/by-pid/" not in f]
    # Exclude the sidecar reports the IATC loop writes beside each graph; without
    # this a run of N proofs is read as 2N (run_artifacts, 2026-08-07).
    try:
        import sys as _sys
        _h = os.path.dirname(os.path.abspath(__file__))
        if _h not in _sys.path:
            _sys.path.insert(0, _h)
        from run_artifacts import is_sidecar
        files = [f for f in files if not is_sidecar(f)]
    except Exception:
        files = [f for f in files if not f.endswith(".rung2.edn")]

    # dedup by paper id, prefer the richer runs
    pref = ["loop-run-70b", "gh200", "linode-stageA", "loop-run-dpdemo-final", "loop-run"]
    def rank(f):
        for i, p in enumerate(pref):
            if p in f:
                return i
        return len(pref)
    by_pid = {}
    for f in sorted(files, key=rank):
        pid = os.path.basename(f).replace(".edn", "")
        by_pid.setdefault(pid, f)

    papers, wanted_to_papers = [], {}
    for pid, f in sorted(by_pid.items()):
        try:
            n_claims, edges = load(f)
        except Exception as e:
            papers.append({"pid": pid, "error": str(e)[:60]})
            continue
        n_edges = len(edges)
        n_missing = sum(1 for e in edges if e["missing"])
        ratio = (n_missing / n_edges) if n_edges else None
        for e in edges:
            if e["missing"] and e["wanted"]:
                wanted_to_papers.setdefault(e["wanted"], set()).add(pid)
        papers.append({"pid": pid, "claims": n_claims, "edges": n_edges,
                       "missing_warrants": n_missing, "unfilled_ratio": ratio})

    assessable = [p for p in papers if p.get("edges")]
    weak = sorted(assessable, key=lambda p: (-p["unfilled_ratio"], -p["missing_warrants"]))
    recurring = sorted(((w, sorted(ps)) for w, ps in wanted_to_papers.items() if len(ps) >= 2),
                       key=lambda x: -len(x[1]))

    out = {
        "n_papers": len(by_pid),
        "n_assessable": len(assessable),
        "n_degenerate": len(by_pid) - len(assessable),
        "total_missing_warrants": sum(p.get("missing_warrants", 0) for p in papers),
        "distinct_wanted": len(wanted_to_papers),
        "recurring_gaps": [{"wanted": w, "papers": ps} for w, ps in recurring],
        "weak_proofs": weak,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"{out['n_papers']} papers ({out['n_assessable']} assessable, "
          f"{out['n_degenerate']} degenerate/no-edges)")
    print(f"{out['total_missing_warrants']} missing-warrant holes, "
          f"{out['distinct_wanted']} distinct :wanted")
    print(f"\nRECURRING GAPS (:wanted in >=2 papers — highest-value fill targets):")
    for r in recurring[:12]:
        print(f"  {len(r[1])}x  {r[0]:42s} {r[1]}")
    if not recurring:
        print("  (none — every missing warrant is paper-local at this corpus size)")
    print(f"\nWEAKEST PROOFS (highest unfilled-warrant ratio):")
    for p in weak[:10]:
        print(f"  {p['unfilled_ratio']:.2f}  {p['pid']:16s} "
              f"{p['missing_warrants']}/{p['edges']} warrants missing")


if __name__ == "__main__":
    main()
