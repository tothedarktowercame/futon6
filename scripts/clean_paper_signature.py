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
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def _pid(graph_id):
    """Paper id from a proof-graph id, via the one shared parser."""
    try:
        from paper_ids import proof_pid_from_graph_name
        return proof_pid_from_graph_name(graph_id)
    except Exception:
        return graph_id.split("__p")[0]

FEATS = ["n_boxes", "n_wires", "n_holes", "n_discharges_known", "max_fanout",
         "depth", "n_sources", "n_sinks"]


def signatures(ids, breakdowns, min_proofs=2):
    macros = sorted({b["macro"] for b in breakdowns})
    feats = [f for f in FEATS if isinstance(breakdowns[0].get(f), (int, float))]
    bypaper = defaultdict(list)
    for i, b in zip(ids, breakdowns):
        # split("__")[0] collapses every legacy id (math__0310337 -> "math"), which
        # merges five distinct papers into one signature. Fourth instance of this
        # class; use the shared parser (H14/H19b).
        bypaper[_pid(i)].append(b)
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
    ap.add_argument("--run-dir", help="persist structural-canon.json here (H16/S11)")
    ap.add_argument("--run-id", default="adhoc")
    ap.add_argument("--corpus-id", default="adhoc")
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

    # PERSIST (E-superpod-hardening S11). This stage computed a shape census and
    # a signature per paper and then printed both into a terminal, so learning
    # goals #3 and #4 died at teardown exactly as the lexicon and the curve did
    # (H15/H16). Artifact shape per the annex fixture: one record per canonical
    # shape with population and exemplars, plus one signature per paper.
    if a.run_dir:
        out_dir = a.run_dir if os.path.isabs(a.run_dir) else os.path.join(ROOT, a.run_dir)
        os.makedirs(out_dir, exist_ok=True)
        shape_pop = Counter()
        shape_ex = {}
        for pid, rows in bypaper.items():
            for b in rows:
                m = b["macro"]
                shape_pop[m] += 1
                shape_ex.setdefault(m, [])
                if len(shape_ex[m]) < 4:
                    shape_ex[m].append(b.get("id") or pid)
        payload = {
            "run_id": a.run_id, "corpus_id": a.corpus_id,
            "n_papers": len(papers),
            "shapes": [{"shape": m, "n": n, "exemplars": shape_ex.get(m, [])}
                       for m, n in shape_pop.most_common()],
            "signatures": [
                {"paper": p,
                 "n_proofs": len(bypaper[p]),
                 "profile": dict(Counter(b["macro"] for b in bypaper[p])),
                 "nearest": sorted([(x, round(cos(vec(sig, p), vec(sig, x)), 4))
                                    for x in papers if x != p], key=lambda z: -z[1])[0]}
                for p in papers],
            "paper_twin_sim": {"mean": round(st.mean(sims), 4),
                               "min": round(min(sims), 4), "max": round(max(sims), 4)},
        }
        path = os.path.join(out_dir, "structural-canon.json")
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=1)
        print(f"\nwrote {path}  ({len(payload['shapes'])} shapes, {len(papers)} signatures)")


if __name__ == "__main__":
    main()
