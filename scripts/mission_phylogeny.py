#!/usr/bin/env python3
# mission_phylogeny.py — the mission corpus as a PHYLOGENY (Joe, 2026-06-08).
# Citation = descent edge (you cite what came before). Generativity = backlinks (how many
# descend from you). Parent = the most-generative ancestor a mission cites. Radial tree:
# trunk (self-representing-stack) near centre, descent outward, the COMPOUNDING FRONTIER
# (recent missions already generative) as the live growing tips. Node colour = L (mana),
# size = generativity. Emits the GRAPH (mission-phylogeny.edn) for M-futon-forward-model;
# historical phylogeny = the Piano Roll (Urðr), the frontier = Verðandi, the projection = Skuld.
import re, math
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path("/home/joe/code")
OUT_HTML = ROOT / "futon6" / "data" / "mission-phylogeny.html"
OUT_EDN = ROOT / "futon6" / "data" / "mission-phylogeny.edn"

W = {}
for m, cls, L, T in re.findall(r':mission "([^"]+)" :class :(\w+) :L ([\d.]+) :T (\d+)',
                               (ROOT / "futon6/data/mission-wholeness.edn").read_text()):
    W[m] = dict(cls=cls, L=float(L), T=int(T))

def colour(s):
    w = W.get(s)
    if not w:
        return "hsl(0,0%,30%)"
    hue = {"alive": 130, "mess": 9, "pipeline": 205, "stub": 0}[w["cls"]]
    sat = 0 if w["cls"] == "stub" else 72
    return f"hsl({hue},{sat}%,{28 + 44 * min(1, w['L'] / 90):.0f}%)"

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

# parent = most-generative cited ancestor; attach only "up" in generativity (breaks cycles)
parent = {}
children = defaultdict(list); roots = []
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
    leaves[s] = sum(dfs(c, d + 1, seen | {s}) for c in ch)
    return leaves[s]
dfs(VR, 0, set())

CX = CY = 1040; RING = 100
pos = {}
def place(s, a0, a1):
    a = (a0 + a1) / 2
    pos[s] = (CX + depth[s] * RING * math.cos(a), CY + depth[s] * RING * math.sin(a))
    ch = children.get(s, [])
    tot = sum(leaves[c] for c in ch) or 1
    cur = a0
    for c in ch:
        w = (a1 - a0) * leaves[c] / tot
        place(c, cur, cur + w); cur += w
place(VR, -math.pi / 2, 3 * math.pi / 2)

frontier = {s for s in stems if field.get(s, "") >= "20260501" and indeg[s] >= 4}

# --- emit the graph ---
def edn():
    gi = " ".join(f'"{s}" {indeg[s]}' for s in sorted(stems, key=lambda s: -indeg[s]))
    out = ['{:trunk "M-self-representing-stack"',
           f' :compounding-frontier [{" ".join(chr(34)+s+chr(34) for s in sorted(frontier, key=lambda s:-indeg[s]))}]',
           f' :generativity-index {{{gi}}}',   # flat {mission -> generativity} for claude-1's pre-witness prior
           " :nodes ["]
    for s in sorted(stems, key=lambda s: -indeg[s]):
        w = W.get(s, {})
        out.append(f'  {{:mission "{s}" :generativity {indeg[s]} :children {len(children.get(s,[]))} '
                   f':depth {depth.get(s,0)} :parent {chr(34)+parent[s]+chr(34) if parent[s] else "nil"} '
                   f':class :{w.get("cls","unknown")} :L {w.get("L",0)} '
                   f':frontier? {"true" if s in frontier else "false"}}}')
    return "\n".join(out) + " ]}\n"
OUT_EDN.write_text(edn())

# --- render ---
edges = []
for s in stems:
    if parent[s] and parent[s] in pos:
        x1, y1 = pos[parent[s]]; x2, y2 = pos[s]
        edges.append(f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}"/>')
nodes = []
for s in sorted(stems, key=lambda s: indeg[s]):
    x, y = pos[s]; r = 3 + 2.0 * math.sqrt(indeg[s])
    ring = ' stroke="#ffe08a" stroke-width="2.5"' if s in frontier else ' stroke="#0008" stroke-width="0.5"'
    nodes.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{r:.1f}" fill="{colour(s)}"{ring}>'
                 f'<title>{s}  L={W.get(s,{}).get("L","?")} {W.get(s,{}).get("cls","?")} '
                 f'backlinks={indeg[s]} children={len(children.get(s,[]))}</title></circle>')
labels = [f'<text x="{pos[s][0]+r0:.0f}" y="{pos[s][1]:.0f}" fill="#eee" font-size="12">{s[2:]}</text>'
          for s, _ in indeg.most_common(10) if s in pos for r0 in [3 + 2 * math.sqrt(indeg[s])]]
fr_labels = [f'<text x="{pos[s][0]+6:.0f}" y="{pos[s][1]+4:.0f}" fill="#ffe08a" font-size="10">{s[2:]}</text>'
             for s in frontier if s in pos]

doc = f"""<!doctype html><meta charset=utf-8><title>Mission phylogeny</title>
<style>body{{margin:0;background:#0a0c11;color:#cdd3df;font:13px sans-serif}}
header{{padding:13px 20px}} h1{{font-size:16px;margin:0 0 4px}} p{{margin:2px 0;color:#8b95a7;font-size:12px}}
line.d{{stroke:#5d6e90;stroke-width:0.7;opacity:0.32}}</style>
<header><h1>Mission phylogeny — descent &amp; generativity ({len(stems)} missions)</h1>
<p>Citation = descent. Node size = generativity (backlinks = #descendants); colour = L (mana) by class.
Trunk = self-representing-stack near centre, descent outward; <b style="color:#ffe08a">gold-ringed = the
compounding frontier</b> (recent missions already generative — where growth attaches next).</p></header>
<svg width="2080" height="2080" viewBox="0 0 2080 2080">
<g>{''.join(f'<line class=d {e[6:]}' for e in edges)}</g>
<g>{''.join(nodes)}</g><g>{''.join(labels)}</g><g>{''.join(fr_labels)}</g></svg>"""
OUT_HTML.write_text(doc)
print(f"wrote {OUT_EDN}\nwrote {OUT_HTML}")
print(f"{len(stems)} missions, {len(roots)} roots, max depth {max(depth.values())}, "
      f"{len(frontier)} compounding-frontier nodes")
print("trunk subtree (self-representing-stack leaves):", leaves.get("M-self-representing-stack"))
print("frontier:", ", ".join(sorted(frontier, key=lambda s: -indeg[s])))
