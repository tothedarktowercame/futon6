#!/usr/bin/env python3
"""Warrant normalization — the warrant/hole analogue of SFC-NORM.

A FOUNDATION step (Phase 1/2), not Phase-3 repair: it maintains a canonical hole
vocabulary so that conceptually-identical missing-warrants from different papers
collapse to one entry. Without it, every Phase-3 technique (weak-proof detection,
conjecture-filling, fill-by-retrieval) inherits a fractured vocabulary.

Mirrors the discipline the stack already uses for CONCEPTS (term-prior df,
consolidate_scope_votes' ">=N papers" mint) — just applied to the hole layer.

Matching: token-normalize the free-text :wanted slugs, then cluster by MiniLM
cosine (catches the paraphrase/subset cases pure stemming misses, e.g.
"verification-of-2-group-properties" ~ "verification-of-2-group-axioms").

In production this runs INCREMENTALLY (each new paper's :wanted matched against
the running vocab; mint a canonical hole only when a novel slug recurs, df>=2);
this batch form is the consolidation that incremental maintenance converges to.

Usage:
  futon6/.venv/bin/python scripts/warrant_normalize.py \
      [--graphs data/iatc-argument-graphs] [--thresh 0.72] \
      [--out data/showcases/clean-demo/hole-vocabulary.json]
"""
import argparse
import glob
import json
import os
import re
import numpy as np
import edn_format as edn
from clean_structure_embed import kw


def collect_wanted(path):
    """All (:wanted, pid) from edge warrants AND the top-level :holes list."""
    m = edn.loads(open(path).read())
    d = {kw(k): v for k, v in dict(m).items()}
    out = []
    for e in d.get("edges", []):
        ed = {kw(k): v for k, v in dict(e).items()}
        w = ed.get("warrant")
        if w is not None:
            wd = {kw(k): v for k, v in dict(w).items()}
            if wd.get("wanted") is not None:
                out.append(kw(wd["wanted"]))
    for h in d.get("holes", []):
        hd = {kw(k): v for k, v in dict(h).items()}
        if hd.get("wanted") is not None:
            out.append(kw(hd["wanted"]))
    return out


def norm_text(slug):
    return re.sub(r"[^a-z0-9 ]+", " ", slug.replace("-", " ").lower()).strip()


def embed(texts):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return np.asarray(model.encode(texts, normalize_embeddings=True))


def cluster(slugs, embs, thresh):
    """Union-find over pairs with cosine >= thresh."""
    n = len(slugs)
    parent = list(range(n))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    def union(a, b):
        parent[find(a)] = find(b)
    sim = embs @ embs.T
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= thresh:
                union(i, j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", default="data/iatc-argument-graphs")
    ap.add_argument("--thresh", type=float, default=0.72)
    ap.add_argument("--out", default="data/showcases/clean-demo/hole-vocabulary.json")
    args = ap.parse_args()

    files = [f for f in glob.glob(os.path.join(args.graphs, "**", "*.edn"), recursive=True)
             if "/.attempts/" not in f and "/by-pid/" not in f]
    by_pid = {}
    for f in files:
        by_pid.setdefault(os.path.basename(f).replace(".edn", ""), f)

    slug_papers = {}
    for pid, f in by_pid.items():
        try:
            for w in collect_wanted(f):
                slug_papers.setdefault(w, set()).add(pid)
        except Exception:
            pass

    slugs = sorted(slug_papers)
    if not slugs:
        print("no :wanted found"); return
    embs = embed([norm_text(s) for s in slugs])
    clusters = cluster(slugs, embs, args.thresh)

    vocab = []
    for cl in clusters:
        members = [slugs[i] for i in cl]
        papers = set()
        for s in members:
            papers |= slug_papers[s]
        canonical = sorted(members, key=lambda s: (-len(slug_papers[s]), len(s)))[0]
        vocab.append({"canonical": canonical, "variants": sorted(members),
                      "df_papers": len(papers), "papers": sorted(papers),
                      "n_variants": len(members)})
    vocab.sort(key=lambda v: (-v["df_papers"], -v["n_variants"]))

    recurring = [v for v in vocab if v["df_papers"] >= 2]
    merged = [v for v in vocab if v["n_variants"] >= 2]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"thresh": args.thresh, "n_slugs": len(slugs),
                   "n_canonical": len(vocab), "vocabulary": vocab}, fh, indent=2)

    print(f"{len(slugs)} raw :wanted slugs -> {len(vocab)} canonical holes "
          f"(thresh={args.thresh})")
    print(f"  {len(merged)} canonical holes merged >=2 paraphrase variants")
    print(f"  {len(recurring)} canonical holes recur across >=2 papers (df>=2 fill targets)\n")
    print("MERGED PARAPHRASES (the foundation defect, now resolved):")
    shown = 0
    for v in merged:
        if v["n_variants"] >= 2:
            print(f"  [{v['df_papers']}p] {v['canonical']}")
            for var in v["variants"]:
                if var != v["canonical"]:
                    print(f"          ~ {var}")
            shown += 1
        if shown >= 10:
            break
    print("\nRECURRING GAPS that EMERGE after normalization (df>=2):")
    for v in recurring[:12]:
        print(f"  {v['df_papers']}x  {v['canonical']:46s} {v['papers']}")
    if not recurring:
        print("  (still none — gaps are genuinely paper-local at this corpus size)")


if __name__ == "__main__":
    main()
