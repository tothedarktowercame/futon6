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
"""
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

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

    GOLDA = math.pi * (3 - math.sqrt(5))
    scope_pts, hub_lines, dots = [], [], []
    for i, p in enumerate(papers):
        scs = paper_scopes[p]
        n = len(scs)
        R = 14 + 3.4 * math.sqrt(n)
        cx, cy = hx[i], hy[i]
        for j, (k, metric) in enumerate(scs):
            ang = j * GOLDA
            rad = R * math.sqrt((j + 0.5) / n)
            x, y = cx + rad * math.cos(ang), cy + rad * math.sin(ang)
            scope_pts.append((x, y, metric))
            hub_lines.append(f'<line x1="{cx:.0f}" y1="{cy:.0f}" x2="{x:.1f}" y2="{y:.1f}" '
                             f'stroke="#33415a" stroke-width="0.4" opacity="0.35"/>')
            dots.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.1" fill="{kindcol(k)}" '
                        f'fill-opacity="0.85"><title>{p}: {k} (incompleteness {metric:.2f})</title></circle>')

    # terrain field (per-scope incompleteness), mission-efe technique
    STEP, SIG = 16, 30.0
    gw, gh = VW // STEP + 1, VH // STEP + 1
    grid = [[0.0] * gw for _ in range(gh)]
    rc = int(3 * SIG / STEP)
    for x, y, mtr in scope_pts:
        cgx, cgy = int(round(x / STEP)), int(round(y / STEP))
        for vy in range(max(0, cgy - rc), min(gh, cgy + rc + 1)):
            for vx in range(max(0, cgx - rc), min(gw, cgx + rc + 1)):
                d2 = (vx * STEP - x) ** 2 + (vy * STEP - y) ** 2
                grid[vy][vx] += mtr * math.exp(-d2 / (2 * SIG * SIG))
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
            'mission-EFE portrait) · hubs = concept-multiplicity t-SNE · scope color = anatomy kind '
            '(blue binder, purple constrain, teal math, amber proof-move, green assume/quant) · '
            'terrain = per-scope incompleteness</div>'
            f'<svg width={VW} height={VH}>{"".join(fill)}{"".join(contour)}'
            f'{"".join(hub_lines)}{"".join(dots)}</svg></body>')
    (W / "greatest-hits.html").write_text(html)
    print(f"greatest-hits: {len(papers)} papers, {len(scope_pts)} scope-points -> greatest-hits.html")


if __name__ == "__main__":
    raise SystemExit(main())
