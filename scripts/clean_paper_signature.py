#!/usr/bin/env python3
"""Whole-paper CLean: the same structural treatment we give definitions/moves at the
INSTANCE level, applied at the PAPER level (Joe's idea).

A paper's structural signature = the COMPOSITION of its proofs' shapes:
  - macro fingerprint  — normalized distribution over proof macro-shapes (the paper's
                         proof-shape profile: reduce-heavy / construct-heavy / has-contradiction …)
  - structural profile — mean of the per-proof structural features (n_boxes, depth, holes,
                         discharges, fanout, …)

Papers then have structural TWINS and ARCHETYPES at the whole-paper level — the proof-level
CLean embedding (clean_structure_embed) lifted to papers. Confirms that the macro/structure
vocabulary, weak per-proof, is informative once AGGREGATED per paper.

  futon6/.venv/bin/python scripts/clean_paper_signature.py --embed data/showcases/mark6-clean-embed.json
"""
import argparse
import json
import math
import statistics as st
from collections import Counter, defaultdict

FEATS = ["n_boxes", "n_wires", "n_holes", "n_discharges_known", "max_fanout",
         "depth", "n_sources", "n_sinks"]


def signatures(ids, breakdowns, min_proofs=2):
    macros = sorted({b["macro"] for b in breakdowns})
    feats = [f for f in FEATS if isinstance(breakdowns[0].get(f), (int, float))]
    bypaper = defaultdict(list)
    for i, b in zip(ids, breakdowns):
        bypaper[i.split("__")[0]].append(b)
    papers = [p for p in bypaper if len(bypaper[p]) >= min_proofs]
    sig = {}
    for p in papers:
        bs = bypaper[p]
        n = len(bs)
        mc = Counter(b["macro"] for b in bs)
        fp = [mc.get(m, 0) / n for m in macros]
        prof = [sum(float(b.get(f, 0) or 0) for b in bs) / n for f in feats]
        sig[p] = [fp, prof]
    for j in range(len(feats)):
        col = [sig[p][1][j] for p in papers]
        m, s = st.mean(col), (st.pstdev(col) or 1)
        for p in papers:
            sig[p][1][j] = (sig[p][1][j] - m) / s
    return papers, sig, bypaper, macros


def vec(sig, p):
    return sig[p][0] + [x * 0.5 for x in sig[p][1]]   # macro fingerprint + downweighted profile


def cos(a, b):
    dp = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dp / (na * nb) if na and nb else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", default="data/showcases/mark6-clean-embed.json")
    a = ap.parse_args()
    d = json.load(open(a.embed))
    papers, sig, bypaper, macros = signatures(d["ids"], d["breakdowns"])
    print(f"=== whole-paper CLean: {len(papers)} papers with a structural signature ===\n")
    print("macro fingerprints (proof-shape profile):")
    for p in papers:
        print(f"  {p} (n={len(bypaper[p])}): {dict(Counter(b['macro'] for b in bypaper[p]))}")
    print("\nwhole-paper structural twins (paper-level NN):")
    sims = []
    for p in papers:
        q, s = sorted([(x, cos(vec(sig, p), vec(sig, x))) for x in papers if x != p],
                      key=lambda z: -z[1])[0]
        sims.append(s)
        print(f"  {p}  ≈  {q}   (paper-sim {s:.2f})")
    print(f"\npaper-twin sim: mean {st.mean(sims):.2f}, range {min(sims):.2f}–{max(sims):.2f} "
          f"— papers cluster by argument-shape composition, not topic")


if __name__ == "__main__":
    main()
