#!/usr/bin/env python3
"""Definition-dependency graph + PageRank authority (Joe's plan, step 3).

Edge C -> D when concept C's DEFINITION SNIPPET mentions concept D (C depends on
D). PageRank on this graph: authority flows to depended-upon concepts, so the
foundational core (category, functor, ...) ranks highest, derived concepts
downstream — that ranking is the grounding curriculum AND the prioritization for
the GPU canonical-definition pass (ground high-authority concepts most carefully;
their errors propagate furthest). The graph is also the EFE-landscape substrate
(embeddable directly with src/futon6/graph_embed.py, like the mission graph).

    warp_concept_graph.py  ->  data/warp/concept-graph.json
"""
import json
import re
from collections import defaultdict
from pathlib import Path

W = Path("/home/joe/code/futon6/data/warp")
DASH = re.compile(r"[‐-―−-]")


def canon(t):
    t = DASH.sub(" ", t.lower())
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def ngrams(text, concepts):
    w = canon(text).split()
    g = set()
    for n in (1, 2, 3):
        for i in range(len(w) - n + 1):
            cand = " ".join(w[i:i + n])
            if cand in concepts:
                g.add(cand)
    return g


def pagerank(nodes, edges, d=0.85, iters=50):
    N = len(nodes)
    pr = {n: 1.0 / N for n in nodes}
    incoming = defaultdict(list)
    outdeg = {n: len(edges.get(n, ())) for n in nodes}
    for c, deps in edges.items():
        for dep in deps:
            incoming[dep].append(c)
    for _ in range(iters):
        dangling = sum(pr[n] for n in nodes if outdeg[n] == 0)
        pr = {n: (1 - d) / N + d * (dangling / N +
              sum(pr[c] / outdeg[c] for c in incoming.get(n, ()) if outdeg[c]))
              for n in nodes}
    return pr


def main():
    hl = json.load(open(W / "hitlist.json"))
    concepts = {h["concept"] for h in hl["hitlist"]}
    used = {h["concept"]: h["used_papers"] for h in hl["hitlist"]}
    snips = json.load(open(W / "def-snippets.json"))["snippets"]

    edges = {}
    for c in concepts:
        deps = set()
        for s in snips.get(c, []):
            deps |= ngrams(s["snippet"], concepts)
        deps.discard(c)
        edges[c] = deps
    n_edges = sum(len(v) for v in edges.values())

    pr = pagerank(concepts, edges)
    authority = sorted(concepts, key=lambda n: -pr[n])
    # grounding order: low-out-degree (leaf / few deps) + high-authority first
    indeg = defaultdict(int)
    for c, deps in edges.items():
        for dep in deps:
            indeg[dep] += 1

    (W / "concept-graph.json").write_text(json.dumps({
        "schema": "concept-graph-v1",
        "n_nodes": len(concepts), "n_edges": n_edges,
        "authority": [{"concept": c, "pagerank": round(pr[c], 6),
                       "depended_on_by": indeg[c], "depends_on": len(edges[c]),
                       "used_papers": used.get(c, 0)} for c in authority[:120]],
    }))
    print(f"nodes {len(concepts)}  edges {n_edges}  "
          f"(avg out-degree {n_edges/max(1,len(concepts)):.1f})")
    print("=== top 20 by PageRank authority (the foundational core) ===")
    for c in authority[:20]:
        print(f"  pr={pr[c]:.5f}  in={indeg[c]:4} out={len(edges[c]):3} "
              f"used={used.get(c,0):4}  {c}")
    print("=== sample leaf concepts (depend on nothing in-corpus = primitives/atoms) ===")
    leaves = [c for c in authority if not edges[c]][:12]
    print("  " + ", ".join(leaves))


if __name__ == "__main__":
    raise SystemExit(main())
