#!/usr/bin/env python3
"""OR-curvature (Ollivier-Ricci) over the citation graph — improvement #1: the
REAL metric terrain for the paper landscape, replacing the NLP scalar tension.

Per claude-3's drainage-citation interface: reuse the curvature core
(substrate_metric_e1_curvature), project citations -> edges, compute kappa per
edge (1 - W1/d), aggregate to per-paper (mean incident edge kappa).
  kappa < 0  : the paper bridges otherwise-separate communities (frontier /
               interdisciplinary positioning) — negatively curved idea-space.
  kappa > 0  : embedded in a dense, well-connected community (mainstream core).

    warp_or_curvature.py -> data/warp/or-curvature.json {paper: kappa}
"""
import json
import statistics
import sys
import time
from collections import defaultdict

sys.path.insert(0, "/home/joe/code/futon3c/scripts")
import substrate_metric_e1_curvature as eng

W = "/home/joe/code/futon6/data/warp"


def main():
    cit = json.load(open(W + "/citations.json")).get("edges", [])
    edges = [(e["from"], e["to"]) for e in cit
             if e.get("from") and e.get("to") and e["from"] != e["to"]]
    multi, simple = defaultdict(list), defaultdict(set)
    for a, b in edges:
        multi[a].append(b); multi[b].append(a)
        simple[a].add(b); simple[b].add(a)
    node_k = defaultdict(list)
    t0 = time.time(); ok = 0
    for a, b in edges:
        try:
            r = eng.curvature_for_edge(multi, simple,
                                       {"edge": [a, b], "relation": "cites"}, legacy=True)
            node_k[a].append(r.kappa); node_k[b].append(r.kappa); ok += 1
        except Exception:
            pass
    out = {p: round(statistics.fmean(ks), 4) for p, ks in node_k.items() if ks}
    json.dump({"schema": "or-curvature-v1", "n_papers": len(out),
               "edges_done": ok, "paper_kappa": out}, open(W + "/or-curvature.json", "w"))
    ks = list(out.values())
    print(f"OR-curvature: {len(out)} papers from {ok} edges in {time.time()-t0:.0f}s; "
          f"kappa min {min(ks):.2f} median {statistics.median(ks):.2f} max {max(ks):.2f}")


if __name__ == "__main__":
    raise SystemExit(main())
