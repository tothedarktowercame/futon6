#!/usr/bin/env python3
# mission_phylo_mandala.py — the SYNTHESIS (Joe, 2026-06-08): per-mission MANDALAS placed in
# the futonic PHYLOGENY. Each node is its own detailed scope-tree mandala (sections as petals,
# sub-scopes as petal-size), SIZED by generativity (trunks render big + detailed, frontier tips
# small), COLOURED by L (mana), laid out by DESCENT (radial tree, trunk near centre), over the
# descent edges, frontier gold-ringed. Shows how each mission's anatomy sits in the landscape.
# Big canvas — meant to be zoomed (SVG scales).
import re, math, sys
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path("/home/joe/code")
OUT = ROOT / "futon6" / "data" / "mission-phylo-mandala.html"
sys.path.insert(0, str(Path(__file__).parent))
from mission_fold import load_sip, load_tree, build  # noqa: E402
SIP = load_sip()

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
    out = []
    for grp in kids.values():
        cnt = sum(nodes[i]["sub_count"] for i in grp)
        nsub = len({nodes[cc]["title"].strip().lower() for i in grp for cc in nodes[i]["children"]})
        out.append((cnt, nsub))
    return sorted(out, reverse=True)

# --- phylogeny ---
paths = {p.stem: p for p in ROOT.glob("futon*/holes/**/M-*.md")}
stems = set(paths)
refs = defaultdict(set); indeg = Counter(); field = {}
dre = re.compile(r'Date:?\*{0,2}\s*(\d{4})-(\d{2})-(\d{2})')
for s, p in paths.items():
    t = p.read_text(errors="ignore")
    m = dre.search(t)
    if m:
        field[s] = "".join(m.groups())
    for r in set(re.findall(r'\bM-[a-z0-9][a-z0-9-]+', t)):
        if r in stems and r != s:
            refs[s].add(r); indeg[r] += 1
parent = {}; children = defaultdict(list); roots = []
for s in stems:
    cand = max(refs[s], key=lambda r: (indeg[r], r)) if refs[s] else None
    if cand and indeg[cand] > indeg.get(s, 0):
        parent[s] = cand; children[cand].append(s)
    else:
        parent[s] = None; roots.append(s)
VR = "__root__"; children[VR] = sorted(roots, key=lambda s: -indeg[s])
depth = {}; leaves = {}
def dfs(s, d, seen):
    depth[s] = d
    ch = [c for c in children.get(s, []) if c not in seen]
    if not ch:
        leaves[s] = 1; return 1
    leaves[s] = sum(dfs(c, d + 1, seen | {s}) for c in ch); return leaves[s]
dfs(VR, 0, set())
CX = CY = 2000; RING = 235
pos = {}
def place(s, a0, a1):
    a = (a0 + a1) / 2
    pos[s] = (CX + depth[s] * RING * math.cos(a), CY + depth[s] * RING * math.sin(a))
    ch = children.get(s, []); tot = sum(leaves[c] for c in ch) or 1
    cur = a0
    for c in ch:
        w = (a1 - a0) * leaves[c] / tot
        place(c, cur, cur + w); cur += w
place(VR, -math.pi / 2, 3 * math.pi / 2)
frontier = {s for s in stems if field.get(s, "") >= "20260501" and indeg[s] >= 4}
MAXC = max((c for s in stems for c, _ in mini_data(s)), default=1)

# --- render: descent edges + a detailed mandala per node ---
edges = [f'<line x1="{pos[parent[s]][0]:.0f}" y1="{pos[parent[s]][1]:.0f}" '
         f'x2="{pos[s][0]:.0f}" y2="{pos[s][1]:.0f}"/>'
         for s in stems if parent[s] and parent[s] in pos]
mandalas = []
for s in sorted(stems, key=lambda s: indeg[s]):   # generative last = drawn on top
    x, y = pos[s]; col = colour(s)
    R = 7 + 2.4 * math.sqrt(indeg[s])              # mandala size by generativity
    secs = mini_data(s) or [(0, 0)]
    n = len(secs)
    petals = []
    for j, (cnt, nsub) in enumerate(secs):
        ang = j / n * 2 * math.pi - math.pi / 2
        ln = 3 + R * min(1, (cnt / MAXC) ** 0.5)
        x2, y2 = x + ln * math.cos(ang), y + ln * math.sin(ang)
        pr = 1.2 + 0.45 * min(nsub, 7)
        petals.append(f'<line x1="{x:.0f}" y1="{y:.0f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                      f'stroke="{col}" stroke-width="{1.0 if indeg[s]<8 else 1.6}" opacity="0.8"/>'
                      f'<circle cx="{x2:.1f}" cy="{y2:.1f}" r="{pr:.1f}" fill="{col}"/>')
    ring = (' <circle cx="%.0f" cy="%.0f" r="%.1f" fill="none" stroke="#ffe08a" '
            'stroke-width="2.5"/>' % (x, y, R + 4)) if s in frontier else ""
    mandalas.append(f'<g><title>{s}  L={W.get(s,{}).get("L","?")} {W.get(s,{}).get("cls","?")} '
                    f'backlinks={indeg[s]} sections={n}</title>'
                    f'<circle cx="{x:.0f}" cy="{y:.0f}" r="2.4" fill="{col}" stroke="#000" '
                    f'stroke-width="0.4"/>{"".join(petals)}{ring}</g>')
labels = [f'<text x="{pos[s][0]+7+2.4*math.sqrt(indeg[s]):.0f}" y="{pos[s][1]:.0f}" fill="#eee" '
          f'font-size="15" font-weight="bold">{s[2:]}</text>'
          for s, _ in indeg.most_common(8) if s in pos]

doc = f"""<!doctype html><meta charset=utf-8><title>Phylo-mandala — mandalas in the phylogeny</title>
<style>body{{margin:0;background:#0a0c11;color:#cdd3df;font:13px sans-serif}}
header{{padding:13px 20px}} h1{{font-size:16px;margin:0 0 4px}} p{{margin:2px 0;color:#8b95a7;font-size:12px}}
line.d{{stroke:#5d6e90;stroke-width:0.8;opacity:0.30}}</style>
<header><h1>Mandalas in the phylogeny — the micro in the macro ({len(stems)} missions)</h1>
<p>Each mission is its detailed scope-tree mandala (petals = sections, petal-size = sub-scopes), placed by
DESCENT (trunk = self-representing-stack near centre, generations outward), SIZED by generativity (backlinks),
COLOURED by L. Gold rings = the compounding frontier. <b>Zoom in</b> (it's detailed by design).</p></header>
<svg width="4000" height="4000" viewBox="0 0 4000 4000">
<g>{''.join(f'<line class=d {e[6:]}' for e in edges)}</g>
<g>{''.join(mandalas)}</g><g>{''.join(labels)}</g></svg>"""
OUT.write_text(doc)
print(f"wrote {OUT}")
print(f"{len(stems)} mission-mandalas in the phylogeny, max depth {max(depth.values())}, "
      f"{len(frontier)} frontier")
print("biggest (trunk):", ", ".join(f"{s}({indeg[s]})" for s, _ in indeg.most_common(4)))
