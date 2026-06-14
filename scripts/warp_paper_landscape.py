#!/usr/bin/env python3
"""Paper landscape v2 (Joe): ALL papers + a topographic TENSION field.

Geometry (real metric): each paper = tf-mean of its used-concept multiplicity-
embedding vectors (IDF-weighted) -> 2D via t-SNE (sklearn). Now places EVERY concordance paper that uses
>=2 embedded concepts (classical coverage is the whole 9745-paper corpus).

Field (the reading): TEMPORAL establishment-reaching tension, rendered as a
topographic surface — Gaussian scatter-add on a grid + banded terrain fill +
marching-squares contour level-sets (the mission-efe-field technique) so it
reads as a landscape, not a scatter.

Outputs: data/warp/paper-landscape.json + data/warp/paper-landscape.html
"""
import json
import math
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


def adate(pid):
    m = re.match(r"(\d{4})\.", pid)
    return int(m.group(1)) if m else 9999



def main():
    hl = json.load(open(W / "hitlist.json"))["hitlist"]
    concepts = [h["concept"] for h in hl]
    cidx = {c: i for i, c in enumerate(concepts)}
    emb = np.load(W / "concept-embed.npy")
    defidx = json.load(open(W / "defined-index.json"))["concept_to_papers"]
    conc = json.load(open(W / "concordance.json"))["terms"]

    first_def = {}
    for term, papers in defidx.items():
        c = canon(term)
        if c in cidx and papers:
            first_def[c] = min(first_def.get(c, 9999), min(adate(p) for p in papers))

    # paper -> concepts: prefer the all-corpus usage scan (all 9742 papers);
    # fall back to the concordance (phrase-rich only for the DP-marked papers).
    cset = set(concepts)
    paper_concepts = defaultdict(set)
    usage_f = W / "concept-usage.json"
    if usage_f.exists():
        for p, cs in json.load(open(usage_f))["paper_concepts"].items():
            paper_concepts[p] = {c for c in cs if c in cset}
    else:
        for term, rows in conc.items():
            c = canon(term)
            if c in cset:
                for r in rows:
                    paper_concepts[r.get("paper")].add(c)

    papers, placed_cs, tension, ncon = [], [], [], []
    for p, cs in paper_concepts.items():
        cs = [c for c in cs if c in cidx]
        if len(cs) < 2:
            continue
        pd = adate(p)
        tension.append(sum(1 for c in cs if first_def.get(c, 9999) >= pd) / len(cs))
        papers.append(p); placed_cs.append(cs); ncon.append(len(cs))
    tension = np.array(tension)
    # IDF-weighted paper vectors (specific/rare concepts dominate = attestation),
    # then t-SNE for spread + clusters (real DR, not hand-rolled force).
    from collections import Counter
    df = Counter(c for cs in placed_cs for c in cs)
    Np = len(papers)
    PV = np.zeros((Np, emb.shape[1]), dtype=np.float32)
    for i, cs in enumerate(placed_cs):
        wsum = 0.0
        for c in cs:
            w = math.log(Np / df[c])
            PV[i] += w * emb[cidx[c]]; wsum += w
        if wsum:
            PV[i] /= wsum
    from sklearn.manifold import TSNE
    xy = TSNE(n_components=2, init="pca", perplexity=30, metric="cosine",
              random_state=7).fit_transform(PV)

    (W / "paper-landscape.json").write_text(json.dumps({
        "schema": "paper-landscape-v2", "n_papers": len(papers),
        "papers": [{"paper": papers[i], "x": float(xy[i, 0]), "y": float(xy[i, 1]),
                    "tension": round(float(tension[i]), 4), "n_concepts": ncon[i]}
                   for i in range(len(papers))]}))

    # ---- topographic render (mission-efe-field technique) ----
    VW, VH, PAD = 1600, 1000, 60
    xs, ys = xy[:, 0], xy[:, 1]
    def mapx(v): return PAD + (VW - 2 * PAD) * (v - xs.min()) / (np.ptp(xs) + 1e-9)
    def mapy(v): return PAD + (VH - 2 * PAD) * (v - ys.min()) / (np.ptp(ys) + 1e-9)
    px = [mapx(v) for v in xs]; py = [mapy(v) for v in ys]

    STEP, SIGMA = 16, 34.0
    gw, gh = VW // STEP + 1, VH // STEP + 1
    grid = [[0.0] * gw for _ in range(gh)]
    rc = int(3 * SIGMA / STEP)
    for i in range(len(papers)):
        cgx, cgy = int(round(px[i] / STEP)), int(round(py[i] / STEP))
        for vy in range(max(0, cgy - rc), min(gh, cgy + rc + 1)):
            for vx in range(max(0, cgx - rc), min(gw, cgx + rc + 1)):
                d2 = (vx * STEP - px[i]) ** 2 + (vy * STEP - py[i]) ** 2
                grid[vy][vx] += tension[i] * math.exp(-d2 / (2 * SIGMA * SIGMA))
    fmax = max(max(r) for r in grid) or 1.0
    NB = 7
    TERR = ["#0a0e1a", "#10243a", "#16374a", "#1f5a44", "#3f7a34", "#9c8a2c", "#c87a28"]
    fill = []
    for gy in range(gh - 1):
        for gx in range(gw - 1):
            val = grid[gy][gx] / fmax
            if val < 0.04:
                continue
            fill.append(f'<rect x="{gx*STEP}" y="{gy*STEP}" width="{STEP}" height="{STEP}" '
                        f'fill="{TERR[min(NB-1,int(val*NB))]}" opacity="0.55"/>')

    def interp(p1, p2, v1, v2, lv):
        t = (lv - v1) / (v2 - v1) if v2 != v1 else 0.5
        return (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))
    contour = []
    for li in range(1, NB):
        lv = li / NB * fmax
        for gy in range(gh - 1):
            for gx in range(gw - 1):
                f00, f10 = grid[gy][gx], grid[gy][gx + 1]
                f01, f11 = grid[gy + 1][gx], grid[gy + 1][gx + 1]
                x0, y0, x1, y1 = gx * STEP, gy * STEP, (gx + 1) * STEP, (gy + 1) * STEP
                cr = []
                if (f00 > lv) != (f10 > lv): cr.append(interp((x0, y0), (x1, y0), f00, f10, lv))
                if (f10 > lv) != (f11 > lv): cr.append(interp((x1, y0), (x1, y1), f10, f11, lv))
                if (f11 > lv) != (f01 > lv): cr.append(interp((x1, y1), (x0, y1), f11, f01, lv))
                if (f01 > lv) != (f00 > lv): cr.append(interp((x0, y1), (x0, y0), f01, f00, lv))
                for k in range(0, len(cr) - 1, 2):
                    (ax, ay), (bx, by) = cr[k], cr[k + 1]
                    contour.append(f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{bx:.1f}" y2="{by:.1f}" '
                                   f'stroke="#e6edff" stroke-width="1" opacity="{0.22+0.07*li:.2f}"/>')
    dots = []
    for i in range(len(papers)):
        col = f"rgb({int(60+195*tension[i])},{int(90+30*(1-tension[i]))},{int(210*(1-tension[i]))})"
        dots.append(f'<circle cx="{px[i]:.1f}" cy="{py[i]:.1f}" r="{1.5+ncon[i]**0.5*0.4:.1f}" '
                    f'fill="{col}" fill-opacity="0.7"><title>{papers[i]} '
                    f't={tension[i]:.2f} n={ncon[i]}</title></circle>')
    html = ('<!doctype html><meta charset=utf8><title>math.CT paper landscape</title>'
            '<body style="margin:0;background:#0a0e1a;color:#ccd;font:13px sans-serif">'
            f'<div style="padding:8px 14px">math.CT paper landscape — <b>{len(papers)} papers</b> · '
            'geometry = concept-multiplicity embedding (real metric, superpod-free) · '
            'terrain = temporal reaching-tension (ridges = frontier/new-idea regions, '
            'valleys = incremental)</div>'
            f'<svg width={VW} height={VH}>{"".join(fill)}{"".join(contour)}{"".join(dots)}</svg></body>')
    (W / "paper-landscape.html").write_text(html)
    print(f"placed {len(papers)} / 9745 papers (>=2 embedded concepts); "
          f"tension mean {tension.mean():.2f}; wrote paper-landscape.html")


if __name__ == "__main__":
    raise SystemExit(main())
