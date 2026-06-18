#!/usr/bin/env python3
"""Greatest-hits landscape (Joe): the 200 most-cited IN-CORPUS papers rendered as
DISTRICTS OF THEIR DETECTED SCOPES — the apples-to-apples comparison to the
mission-EFE portrait (papers <-> missions, DP scopes <-> mission scopes), so we
see the same level of local structure.

Hubs laid out by concept-multiplicity (t-SNE, ~comparable count to missions);
each paper's scopes spiral around its hub (golden-angle, like mission_efe_field),
colored by anatomy kind; a per-scope local-incompleteness metric (ungrounded
fraction within the scope) scatter-adds into a topographic terrain.

    warp_greatest_hits.py -> data/warp/greatest-hits.html

Scopes are AGGREGATED per (paper, anatomy-kind-class): one glyph per kind, sized
~ scope count, so each paper yields <=6 glyphs (<~1.2k total).

History (perf / OOM hazard, measured & fixed 2026-06-14): the original emitted
ONE <circle> (each with a nested <title>) AND one <line> per scope-MARK across
all ~194 papers. Real papers carry thousands of marks each, so output exploded
to ~793k circles + ~795k lines + 6k rects on a single HTML line (189 MB,
~1.6M DOM nodes) — which OOMs Firefox and trips librsvg's 1M-element cap; 60% of
the dots were one kind (teal "math", ~478k), saturating the canvas. The per-kind
aggregation below replaced that. If you ever need per-mark detail again, don't
render it as DOM at this scale: rasterise the flat geometry directly to PNG
(circles/lines/rects); a browser is the wrong tool at that node count.
"""
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from viz_budget import guard_svg

W = Path("/home/joe/code/futon6/data/warp")
GOLD = Path("/home/joe/code/futon6/data/showcases/ct-anatomy/golden")


def kindcol(k):
    if k == "let-binder" or k.startswith("bind"):
        return "#3b6fd4"
    if k.startswith("constrain"):
        return "#9a55c8"
    if k == "math":
        return "#3a8ea0"
    if k == "proof-move":
        return "#e0a33a"
    if k.startswith("assume") or k.startswith("quant"):
        return "#2f9f6a"
    return "#8893a8"


# six anatomy-kind classes (legend order); raw marks bucket into these so one
# paper yields <=6 aggregate glyphs instead of one dot per scope-mark.
CLASSES = ["bind", "constrain", "math", "proof-move", "assume", "other"]


def classof(k):
    if k == "let-binder" or k.startswith("bind"):
        return "bind"
    if k.startswith("constrain"):
        return "constrain"
    if k == "math":
        return "math"
    if k == "proof-move":
        return "proof-move"
    if k.startswith("assume") or k.startswith("quant"):
        return "assume"
    return "other"


def scope_points(marks):
    syms = [(m["start"], m["end"], m["kind"]) for m in marks
            if m.get("kind") in ("symbol", "symbol-grounded")]
    pts = []
    for m in marks:
        k = m.get("kind", "")
        if not (m.get("layer") == "scope" or k in ("let-binder", "math", "proof-move")):
            continue
        if m.get("end", 0) <= m.get("start", 0):
            continue
        inside = [kk for s, e, kk in syms if m["start"] <= s and e <= m["end"]]
        ung = sum(1 for kk in inside if kk == "symbol")
        frac = ung / len(inside) if inside else 0.0
        pts.append((k, 0.2 + 0.8 * frac))
    return pts


def main():
    ids = [l.strip() for l in open("/tmp/gh200.txt") if l.strip()]
    paper_scopes = {}
    for pid in ids:
        f = GOLD / f"fable-{pid}-dp-emacs.json"
        if not f.exists():
            continue
        sp = scope_points(json.load(open(f))["marks"])
        if sp:
            paper_scopes[pid] = sp
    papers = list(paper_scopes)
    if len(papers) < 10:
        print(f"only {len(papers)} marked — wait for DP-marking"); return 1

    # hub layout: t-SNE on concept-multiplicity vectors
    usage = json.load(open(W / "concept-usage.json"))["paper_concepts"]
    hl = json.load(open(W / "hitlist.json"))["hitlist"]
    cidx = {h["concept"]: i for i, h in enumerate(hl)}
    emb = np.load(W / "concept-embed.npy")
    df = Counter(c for cs in usage.values() for c in cs)
    Nall = len(usage)
    PV = np.zeros((len(papers), emb.shape[1]), np.float32)
    for i, p in enumerate(papers):
        cs = [c for c in usage.get(p, []) if c in cidx]
        ws = 0.0
        for c in cs:
            w = math.log(Nall / df[c]); PV[i] += w * emb[cidx[c]]; ws += w
        if ws:
            PV[i] /= ws
    from sklearn.manifold import TSNE
    hub = TSNE(n_components=2, init="pca", perplexity=min(15, len(papers) - 1),
               metric="cosine", random_state=7).fit_transform(PV)

    VW, VH, PAD = 1600, 1000, 70
    hx = PAD + (VW - 2 * PAD) * (hub[:, 0] - hub[:, 0].min()) / (np.ptp(hub[:, 0]) + 1e-9)
    hy = PAD + (VH - 2 * PAD) * (hub[:, 1] - hub[:, 1].min()) / (np.ptp(hub[:, 1]) + 1e-9)

    # Aggregate per (paper, anatomy-kind-class): one glyph per kind, ringed
    # around the hub at a fixed slot, area ~ scope count (log-scaled so the
    # ~10^4-scope papers stay bounded), tooltip carries the count + mean
    # incompleteness. Collapses ~793k dots to <=6 per paper.
    ROFF = 24.0
    scope_pts, hub_lines, dots, hub_dots = [], [], [], []
    for i, p in enumerate(papers):
        cx, cy = hx[i], hy[i]
        agg = {}  # class -> [count, sum_metric]
        for k, metric in paper_scopes[p]:
            a = agg.setdefault(classof(k), [0, 0.0])
            a[0] += 1
            a[1] += metric
        # pale hub-center marker: the paper itself, spokes radiate to its kinds
        hub_dots.append(f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="1.4" '
                        f'fill="#cdd6ee" fill-opacity="0.9"/>')
        for c, (n, msum) in agg.items():
            inc = msum / n
            ang = CLASSES.index(c) / len(CLASSES) * 2 * math.pi
            x, y = cx + ROFF * math.cos(ang), cy + ROFF * math.sin(ang)
            # small dots so per-paper rosettes stay distinct; the spoke ties each
            # kind-dot back to its hub so the paper<->scopes grouping reads.
            r = 1.2 + 1.6 * math.log10(1 + n)
            scope_pts.append((x, y, inc, n))
            hub_lines.append(f'<line x1="{cx:.0f}" y1="{cy:.0f}" x2="{x:.1f}" y2="{y:.1f}" '
                             f'stroke="#6f80a8" stroke-width="0.6" opacity="0.55"/>')
            dots.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{kindcol(c)}" '
                        f'fill-opacity="0.85"><title>{p}: {c} ×{n} '
                        f'(mean incompleteness {inc:.2f})</title></circle>')

    # inter-paper citation roads: in-corpus edges among these papers, drawn as a
    # distinct violet layer above the intra-paper spokes (mirrors the mission-EFE
    # landscape's citation roads, tuned brighter/heavier for this hub density).
    hubpos = {p: (hx[i], hy[i]) for i, p in enumerate(papers)}
    cw = Counter()
    for ed in json.load(open(W / "citations.json"))["edges"]:
        a, b = ed.get("from"), ed.get("to")
        if a in hubpos and b in hubpos and a != b:
            cw[tuple(sorted((a, b)))] += 1
    roads = []
    for (a, b), c in cw.items():
        (x1, y1), (x2, y2) = hubpos[a], hubpos[b]
        op = min(0.6, 0.22 + 0.08 * (c - 1))
        wd = min(2.6, 0.7 + 0.35 * (c - 1))
        roads.append(f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" '
                     f'stroke="#c9a8ff" stroke-width="{wd:.1f}" opacity="{op:.2f}"/>')

    # terrain field (per-scope incompleteness), mission-efe technique
    STEP, SIG = 16, 30.0
    gw, gh = VW // STEP + 1, VH // STEP + 1
    grid = [[0.0] * gw for _ in range(gh)]
    rc = int(3 * SIG / STEP)
    for x, y, mtr, n in scope_pts:
        cgx, cgy = int(round(x / STEP)), int(round(y / STEP))
        wt = mtr * math.log10(1 + n)  # count-weighted, log-scaled like the glyph
        for vy in range(max(0, cgy - rc), min(gh, cgy + rc + 1)):
            for vx in range(max(0, cgx - rc), min(gw, cgx + rc + 1)):
                d2 = (vx * STEP - x) ** 2 + (vy * STEP - y) ** 2
                grid[vy][vx] += wt * math.exp(-d2 / (2 * SIG * SIG))
    fmax = max(max(r) for r in grid) or 1.0
    NB = 7
    TERR = ["#0a0e1a", "#10243a", "#16374a", "#1f5a44", "#3f7a34", "#9c8a2c", "#c87a28"]
    fill = []
    for gy in range(gh - 1):
        for gx in range(gw - 1):
            v = grid[gy][gx] / fmax
            if v >= 0.05:
                fill.append(f'<rect x="{gx*STEP}" y="{gy*STEP}" width="{STEP}" height="{STEP}" '
                            f'fill="{TERR[min(NB-1,int(v*NB))]}" opacity="0.5"/>')

    def interp(p1, p2, v1, v2, lv):
        t = (lv - v1) / (v2 - v1) if v2 != v1 else 0.5
        return (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))
    contour = []
    for li in range(1, NB):
        lv = li / NB * fmax
        for gy in range(gh - 1):
            for gx in range(gw - 1):
                f00, f10, f01, f11 = grid[gy][gx], grid[gy][gx+1], grid[gy+1][gx], grid[gy+1][gx+1]
                x0, y0, x1, y1 = gx*STEP, gy*STEP, (gx+1)*STEP, (gy+1)*STEP
                cr = []
                if (f00 > lv) != (f10 > lv): cr.append(interp((x0, y0), (x1, y0), f00, f10, lv))
                if (f10 > lv) != (f11 > lv): cr.append(interp((x1, y0), (x1, y1), f10, f11, lv))
                if (f11 > lv) != (f01 > lv): cr.append(interp((x1, y1), (x0, y1), f11, f01, lv))
                if (f01 > lv) != (f00 > lv): cr.append(interp((x0, y1), (x0, y0), f01, f00, lv))
                for kk in range(0, len(cr) - 1, 2):
                    (ax, ay), (bx, by) = cr[kk], cr[kk + 1]
                    contour.append(f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{bx:.1f}" y2="{by:.1f}" '
                                   f'stroke="#e6edff" stroke-width="0.9" opacity="{0.2+0.07*li:.2f}"/>')
    html = ('<!doctype html><meta charset=utf8><title>CT greatest-hits (scope districts)</title>'
            '<body style="margin:0;background:#0a0e1a;color:#ccd;font:13px sans-serif">'
            f'<div style="padding:8px 14px">math.CT <b>greatest hits</b> — {len(papers)} most-cited '
            'in-corpus papers as DISTRICTS OF THEIR DETECTED SCOPES (apples-to-apples with the '
            'mission-EFE portrait) · hubs = concept-multiplicity t-SNE · one glyph per anatomy kind, '
            'area ~ scope count (blue binder, purple constrain, teal math, amber proof-move, '
            'green assume/quant, grey other) · violet roads = in-corpus citations '
            'between papers · terrain = count-weighted incompleteness</div>'
            f'<svg width={VW} height={VH}>{"".join(fill)}{"".join(contour)}'
            f'{"".join(hub_lines)}{"".join(roads)}{"".join(dots)}{"".join(hub_dots)}</svg></body>')
    n_elem = guard_svg(html, "greatest-hits")
    (W / "greatest-hits.html").write_text(html)
    print(f"greatest-hits: {len(papers)} papers, {len(scope_pts)} scope-points, "
          f"{n_elem} svg elements -> greatest-hits.html")


if __name__ == "__main__":
    raise SystemExit(main())
