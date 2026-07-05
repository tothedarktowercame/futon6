#!/usr/bin/env python3
# mission_carpet.py — phylo-mandala spatialized into a Turkish Carpet (Joe, 2026-06-08).
# v2: edges reweighted by TURN-ATTESTATION. A shared pattern-scope's spring strength scales with how
# often turns actually retrieve it (the evidence log) — so THICK ROADS (heavily-enacted patterns) are
# STRONGER SPRINGS and pull the layout; nominal co-mentions stay faint and don't deform it. Descent
# (citations) = the complementary web (strong). Node = mandala, colour = L, size = generativity,
# gold ring = compounding frontier.
import re, math, sys, json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

ROOT = Path("/home/joe/code")
OUT = ROOT / "futon6" / "data" / "mission-carpet.html"
sys.path.insert(0, str(Path(__file__).parent))
from mission_fold import load_sip, load_tree, build  # noqa: E402
SIP = load_sip()
ATT = json.load(open(ROOT / "futon6/data/pattern-attestation.json")).get("by_name", {})

W = {}
for m, cls, L, T in re.findall(r':mission "([^"]+)" :class :(\w+) :L ([\d.]+) :T (\d+)',
                               (ROOT / "futon6/data/mission-wholeness.edn").read_text()):
    W[m] = dict(cls=cls, L=float(L), T=int(T))
def colour(s):
    w = W.get(s)
    if not w:
        return "hsl(0,0%,30%)"
    hue = {"alive": 130, "mess": 9, "pipeline": 205, "stub": 0}[w["cls"]]
    return f"hsl({hue},{0 if w['cls']=='stub' else 72}%,{28 + 44 * min(1, w['L'] / 90):.0f}%)"
def mini_data(stem):
    try:
        tree, _ = load_tree(stem); nodes, roots = build(tree, SIP)
    except Exception:
        return []
    kids = {}
    for r in roots:
        for c in nodes[r]["children"]:
            kids.setdefault(nodes[c]["title"].strip().lower(), []).append(c)
    return sorted((sum(nodes[i]["sub_count"] for i in g),
                   len({nodes[cc]["title"].strip().lower() for i in g for cc in nodes[i]["children"]}))
                  for g in kids.values())[::-1]

paths = {p.stem: p for p in ROOT.glob("futon*/holes/**/M-*.md")}
stems = sorted(paths)
idx = {s: i for i, s in enumerate(stems)}
N = len(stems)
texts = {s: paths[s].read_text(errors="ignore") for s in stems}

refs = defaultdict(set); indeg = Counter(); field = {}
dre = re.compile(r'Date:?\*{0,2}\s*(\d{4})-(\d{2})-(\d{2})')
for s in stems:
    m = dre.search(texts[s])
    if m:
        field[s] = "".join(m.groups())
    for r in set(re.findall(r'\bM-[a-z0-9][a-z0-9-]+', texts[s])):
        if r in paths and r != s:
            refs[s].add(r); indeg[r] += 1
parent = {s: (max(refs[s], key=lambda r: (indeg[r], r)) if refs[s] else None) for s in stems}
frontier = {s for s in stems if field.get(s, "") >= "20260501" and indeg[s] >= 4}

# HGT: shared distinctive pattern-scopes, weighted by TURN-ATTESTATION
flex = {Path(f).stem for f in __import__("glob").glob(str(ROOT / "futon*/library/**/*.flexiarg"), recursive=True)}
flex = {b for b in flex if b.count("-") >= 2 and len(b) >= 12}
applied = {s: {b for b in flex if b in texts[s]} for s in stems}
pm = defaultdict(set)
for s in stems:
    for b in applied[s]:
        pm[b].add(s)
pair_att = defaultdict(int)          # edge -> attestation of its strongest shared road
for g, ms in pm.items():
    a = ATT.get(g, 0)
    ms = sorted(ms)
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            pair_att[(ms[i], ms[j])] = max(pair_att[(ms[i], ms[j])], a)

# --roads-only: refresh the attestation-weighted road list WITHOUT re-solving
# the spring layout. The daily job uses this so road ink tracks live
# attestation while district positions stay put — geometry re-solve (springs
# moving districts) is a deliberate operator decision, not a cron default.
if "--roads-only" in sys.argv:
    json.dump([[a, b, int(w)] for (a, b), w in pair_att.items()],
              open(ROOT / "futon6" / "data" / "mission-carpet-roads.json", "w"))
    print(f"roads-only: {len(pair_att)} HGT roads (layout untouched)")
    raise SystemExit(0)

# springs: citations (complementary web, strong) + HGT (strength ∝ attestation)
E = []
for s in stems:
    if parent[s]:
        E.append((idx[s], idx[parent[s]], 0.9))
for (a, b), w in pair_att.items():
    E.append((idx[a], idx[b], 0.02 + 0.006 * min(w, 200)))   # THICK ROADS = STRONGER SPRINGS
ei = np.array([e[0] for e in E]); ej = np.array([e[1] for e in E]); ek = np.array([e[2] for e in E])
deg = np.bincount(np.concatenate([ei, ej]), minlength=N).astype(float)   # connectivity → un-attached pull in

rng = np.random.default_rng(7)
P = rng.standard_normal((N, 2)) * 280
for it in range(460):
    diff = P[:, None, :] - P[None, :, :]
    d2 = (diff ** 2).sum(-1) + 1.0
    rep = (diff / d2[..., None]).sum(1) * 1300.0   # looser core (was 900) — declump the dense centre
    att = np.zeros((N, 2))
    f = (P[ej] - P[ei]) * ek[:, None] * 0.036          # softer springs → looser core (was 0.05)
    np.add.at(att, ei, f); np.add.at(att, ej, -f)
    grav = (-P) * (0.022 / (1.0 + deg))[:, None]        # un-attached clusters drift toward centre
    P += (rep + att + grav) * (0.85 ** (it / 60))
    P -= P.mean(0)
P -= P.min(0); P = P * (3200 / max(P.max(0))) + 200

# --- GEOMETRIC neighbourhoods + Salingaros at the region scale ---
# T = geometric density (tightly-bound -> dense); H = topological citation-coherence (interconnected).
patlib = {}
for f in __import__("glob").glob(str(ROOT / "futon*/library/**/*.flexiarg"), recursive=True):
    pr = Path(f).parts
    if "library" in pr:
        patlib[Path(f).stem] = pr[pr.index("library") + 1]
cite = {frozenset((idx[s], idx[r])) for s in stems for r in refs[s]}
# GEOMETRIC grid over the carpet — forces real sub-regions; per-cell count = density = T.
GRID = 7
x0, x1 = P[:, 0].min(), P[:, 0].max(); y0, y1 = P[:, 1].min(), P[:, 1].max()
cw = (x1 - x0) / GRID; chh = (y1 - y0) / GRID
cell = defaultdict(list)
for i in range(N):
    cx = min(GRID - 1, int((P[i, 0] - x0) / (x1 - x0 + 1e-9) * GRID))
    cy = min(GRID - 1, int((P[i, 1] - y0) / (y1 - y0 + 1e-9) * GRID))
    cell[(cx, cy)].append(i)
clu = {k: v for k, v in cell.items() if len(v) >= 4}
maxd = max(len(v) for v in clu.values()) if clu else 1
regions = []
for (cx, cy), mem in clu.items():
    cen = P[mem].mean(0)
    T = round(10 * len(mem) / maxd, 1)                       # equal-area cells -> count IS density
    internal = {a for a in mem for b in mem if a != b and frozenset((a, b)) in cite}
    H = round(10 * len(internal) / len(mem), 1)              # fraction with an internal citation
    names = [stems[i] for i in mem]
    cls = Counter(W.get(n, {}).get("cls", "?") for n in names)
    libs = Counter(patlib.get(g) for n in names for g in applied[n] if patlib.get(g))
    fm, fa, fp = cls.get("mess", 0) / len(mem), cls.get("alive", 0) / len(mem), cls.get("pipeline", 0) / len(mem)
    if fm > 0.3 and H < 5:
        char, col = "slum", "#c0392b"
    elif fp > 0.3:
        char, col = "condos", "#2d6aa8"
    elif fa >= 0.45 and len(libs) >= 4:
        char, col = "village", "#3a8a4a"
    else:
        char, col = "quarter", "#6b6b6b"
    regions.append(dict(cen=cen, rx=x0 + cx * cw, ry=y0 + cy * chh, cw=cw, chh=chh, T=T, H=H,
                        L=round(T * H / 10, 1), char=char, col=col, n=len(mem), libs=libs,
                        mess=cls.get("mess", 0), alive=cls.get("alive", 0)))
region_svg = "".join(
    f'<rect x="{r["rx"]:.0f}" y="{r["ry"]:.0f}" width="{r["cw"]:.0f}" height="{r["chh"]:.0f}" '
    f'fill="{r["col"]}" opacity="0.09" stroke="{r["col"]}" stroke-opacity="0.3" stroke-width="1.5"/>'
    for r in regions)
region_labels = "".join(
    f'<text x="{r["rx"]+r["cw"]/2:.0f}" y="{r["ry"]+16:.0f}" fill="{r["col"]}" font-size="17" '
    f'font-weight="bold" text-anchor="middle">{r["char"]} L{r["L"]} (T{r["T"]} H{r["H"]}, '
    f'{r["n"]}m {len(r["libs"])}lib)</text>' for r in regions)
front_labels = "".join(
    f'<text x="{P[idx[s]][0]+9:.0f}" y="{P[idx[s]][1]+4:.0f}" fill="#ffe08a" font-size="11">{s[2:]}</text>'
    for s in frontier)

MAXC = max((c for s in stems for c, _ in mini_data(s)), default=1)
hgt_lines = []
for (a, b), w in pair_att.items():
    x1, y1 = P[idx[a]]; x2, y2 = P[idx[b]]
    wid = 0.3 + 0.02 * min(w, 200); op = 0.05 + 0.004 * min(w, 120)
    hgt_lines.append(f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" '
                     f'stroke="#b08fd0" stroke-width="{wid:.1f}" opacity="{op:.2f}"/>')
desc_lines = [f'<line x1="{P[idx[s]][0]:.0f}" y1="{P[idx[s]][1]:.0f}" '
              f'x2="{P[idx[parent[s]]][0]:.0f}" y2="{P[idx[parent[s]]][1]:.0f}" '
              f'stroke="#5d6e90" stroke-width="0.8" opacity="0.22"/>'
              for s in stems if parent[s]]
minis = []
for s in sorted(stems, key=lambda s: indeg[s]):
    x, y = P[idx[s]]; col = colour(s); R = 7 + 2.2 * math.sqrt(indeg[s])
    secs = mini_data(s) or [(0, 0)]; n = len(secs); petals = []
    for j, (cnt, nsub) in enumerate(secs):
        ang = j / n * 2 * math.pi - math.pi / 2
        ln = 3 + R * min(1, (cnt / MAXC) ** 0.5)
        x2, y2 = x + ln * math.cos(ang), y + ln * math.sin(ang)
        petals.append(f'<line x1="{x:.0f}" y1="{y:.0f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{col}" '
                      f'stroke-width="1" opacity="0.8"/><circle cx="{x2:.1f}" cy="{y2:.1f}" '
                      f'r="{1.2+0.4*min(nsub,7):.1f}" fill="{col}"/>')
    ring = (f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{R+5:.0f}" fill="none" stroke="#ffe08a" '
            f'stroke-width="2.5"/>') if s in frontier else ""
    minis.append(f'<g><title>{s}  L={W.get(s,{}).get("L","?")} {W.get(s,{}).get("cls","?")} '
                 f'backlinks={indeg[s]} genes={len(applied[s])}</title>'
                 f'<circle cx="{x:.0f}" cy="{y:.0f}" r="2.2" fill="{col}"/>{"".join(petals)}{ring}</g>')
labels = [f'<text x="{P[idx[s]][0]+8:.0f}" y="{P[idx[s]][1]:.0f}" fill="#eee" font-size="15" '
          f'font-weight="bold">{s[2:]}</text>' for s, _ in indeg.most_common(8)]

doc = f"""<!doctype html><meta charset=utf-8><title>Mission carpet</title>
<style>body{{margin:0;background:#0a0c11;color:#cdd3df;font:13px sans-serif}}header{{padding:12px 20px}}
h1{{font-size:16px;margin:0 0 4px}}p{{margin:0;color:#8b95a7;font-size:12px}}</style>
<header><h1>Mission carpet — attestation-weighted ({N} missions)</h1>
<p>Springs: citations (complementary web) + HGT roads weighted by TURN-attestation (thick = enacted). Tinted
NEIGHBOURHOODS carry Salingaros at the region scale: <b>T = GEOMETRIC density</b> (tightly-bound → dense),
<b>H = TOPOLOGICAL citation-coherence</b> (interconnected), L=T·H. Red=slum (dense, disconnected) · blue=condos
(pipeline) · green=village (alive + library-diverse). Gold rings + labels = the compounding frontier. Zoom in.</p></header>
<svg width="3600" height="3600" viewBox="0 0 3600 3600">
<g>{region_svg}</g>
<g>{''.join(hgt_lines)}</g><g>{''.join(desc_lines)}</g><g>{''.join(minis)}</g>
<g>{front_labels}</g><g>{region_labels}</g><g>{''.join(labels)}</g></svg>"""
OUT.write_text(doc)
# Emit positions so the EFE-field render (D2) can paint onto the real city layout.
import json as _json
_json.dump({s: [round(float(P[idx[s]][0]), 1), round(float(P[idx[s]][1]), 1)] for s in stems},
           open(ROOT / "futon6" / "data" / "mission-carpet-pos.json", "w"))
# Emit the HGT pattern-roads (attestation-weighted) so the EFE-field render can show the backdrop.
_json.dump([[a, b, int(w)] for (a, b), w in pair_att.items()],
           open(ROOT / "futon6" / "data" / "mission-carpet-roads.json", "w"))
print(f"wrote {OUT}")
print(f"{N} missions, {len(desc_lines)} citation springs, {len(pair_att)} HGT roads, "
      f"{len(regions)} neighbourhoods, {len(frontier)} gold rings (labelled)")
print("neighbourhoods (T=geometric density · H=citation-coherence):")
for r in sorted(regions, key=lambda r: -r["L"]):
    print(f"  {r['char']:8} T{r['T']:4.1f} H{r['H']:4.1f} L{r['L']:4.1f}  {r['n']:2d} missions, "
          f"{len(r['libs'])} libs, mess={r['mess']} alive={r['alive']}")
