#!/usr/bin/env python3
"""Paper landscape (first cut, Joe's plan): a REAL metric over papers from the
multiplicity concept-embedding, + a de-confounded TENSION field.

Geometry (the metric): each paper = the (tf-weighted) mean of the embedding
vectors of the hitlist concepts it mentions -> papers near each other genuinely
use the same concept families. 2D via PCA for the carpet layout.

Tension (the reading): TEMPORAL establishment-reaching, NOT raw grounding (which
today's test showed is confounded by expository density). For paper P:
  reaching(P) = fraction of P's concepts whose EARLIEST corpus definition is not
  strictly before P's arXiv date -> P leans on concepts the corpus hadn't
  settled yet = grasping at the new. Incremental papers (all concepts long
  established) score low.

Outputs: data/warp/paper-landscape.json + data/warp/paper-landscape.html
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

W = Path("/home/joe/code/futon6/data/warp")
DASH = re.compile(r"[‐-―−-]")


def canon(t):
    t = DASH.sub(" ", t.lower())
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def adate(pid):                       # arXiv YYMM.NNNNN -> sortable int (YYMM)
    m = re.match(r"(\d{4})\.", pid)
    return int(m.group(1)) if m else 9999


def main():
    hl = json.load(open(W / "hitlist.json"))["hitlist"]
    concepts = [h["concept"] for h in hl]
    cidx = {c: i for i, c in enumerate(concepts)}
    emb = np.load(W / "concept-embed.npy")
    defidx = json.load(open(W / "defined-index.json"))["concept_to_papers"]
    conc = json.load(open(W / "concordance.json"))["terms"]

    # concept establishment: earliest corpus definition date (min arXiv date of
    # defining papers), canon-keyed.
    first_def = {}
    for term, papers in defidx.items():
        c = canon(term)
        if c in cidx and papers:
            d = min(adate(p) for p in papers)
            first_def[c] = min(first_def.get(c, 9999), d)

    # paper -> set of hitlist concepts it mentions (canon)
    paper_concepts = defaultdict(set)
    cset = set(concepts)
    for term, rows in conc.items():
        c = canon(term)
        if c in cset:
            for r in rows:
                paper_concepts[r.get("paper")].add(c)

    papers, vecs, tension, ncon = [], [], [], []
    for p, cs in paper_concepts.items():
        cs = [c for c in cs if c in cidx]
        if len(cs) < 4:               # need a few concepts to place a paper
            continue
        v = emb[[cidx[c] for c in cs]].mean(axis=0)
        pd = adate(p)
        reaching = sum(1 for c in cs if first_def.get(c, 9999) >= pd) / len(cs)
        papers.append(p); vecs.append(v); tension.append(reaching); ncon.append(len(cs))
    X = np.array(vecs, dtype=np.float32)
    tension = np.array(tension)
    # 2D layout: PCA (SVD of centered paper matrix)
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    xy = U[:, :2] * S[:2]

    out = [{"paper": papers[i], "x": float(xy[i, 0]), "y": float(xy[i, 1]),
            "tension": round(float(tension[i]), 4), "n_concepts": ncon[i]}
           for i in range(len(papers))]
    (W / "paper-landscape.json").write_text(json.dumps(
        {"schema": "paper-landscape-v1", "n_papers": len(papers), "papers": out}))

    # --- simple HTML scatter (color = tension blue->red) ---
    xs, ys = xy[:, 0], xy[:, 1]
    def sc(v, lo, hi, a, b): return a + (b - a) * (v - lo) / (hi - lo + 1e-9)
    dots = []
    for i in range(len(papers)):
        px = sc(xs[i], xs.min(), xs.max(), 40, 1160)
        py = sc(ys[i], ys.min(), ys.max(), 760, 40)
        t = tension[i]
        col = f"rgb({int(40+215*t)},{int(60+40*(1-t))},{int(220*(1-t))})"
        dots.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{2+ncon[i]**0.5*0.6:.1f}" '
                    f'fill="{col}" fill-opacity="0.6"><title>{papers[i]} '
                    f't={t:.2f} n={ncon[i]}</title></circle>')
    html = ('<!doctype html><meta charset=utf8><title>CT paper landscape</title>'
            '<body style="background:#0d1117;color:#ccc;font:13px sans-serif">'
            f'<h3>math.CT paper landscape — {len(papers)} papers · geometry=concept-multiplicity '
            'embedding · color=temporal reaching-tension (red=reaching, blue=incremental)</h3>'
            f'<svg width=1200 height=800>{"".join(dots)}</svg></body>')
    (W / "paper-landscape.html").write_text(html)

    print(f"{len(papers)} papers placed; tension mean {tension.mean():.2f}")
    order = np.argsort(-tension)
    print("=== highest reaching-tension (de-confounded) ===")
    for i in order[:6]:
        print(f"  t={tension[i]:.2f} n={ncon[i]:3} {papers[i]}")
    print("=== lowest (incremental) ===")
    for i in order[-6:]:
        print(f"  t={tension[i]:.2f} n={ncon[i]:3} {papers[i]}")


if __name__ == "__main__":
    raise SystemExit(main())
