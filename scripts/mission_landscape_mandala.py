#!/usr/bin/env python3
# mission_landscape_mandala.py — the LANDSCAPE as a MANDALA OF MANDALAS (Joe, 2026-06-08).
# Each mission is itself a mini-mandala: the eightfold rendered radially — a spoke per phase
# PRESENT (length = how filled), MISSING spokes = the holes — coloured by L (Salingaros mana).
# The minis are arranged in the big mandala: centre = "first in" (oldest), spiraling outward
# (phyllotaxis), over a faint cross-ref web. The whole stack's aliveness-terrain, in one figure.
import re, glob, math, subprocess, sys
from pathlib import Path
from collections import Counter

ROOT = Path("/home/joe/code")
OUT = ROOT / "futon6" / "data" / "mission-landscape-mandala.html"
GOLDEN = math.radians(137.508)
sys.path.insert(0, str(Path(__file__).parent))
from mission_fold import load_sip, load_tree, build  # noqa: E402

SIP = load_sip()
PHASES = ["head", "identify", "map", "derive", "argue", "verify", "instantiate", "document"]
def mini_data(stem):
    # the mission's radial scope-tree: each root-child (section) -> (concept-count, n-subscopes),
    # deduped by title. Drawn as a petal: spoke length = richness, petal size = #sub-scopes.
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

# wholeness: L, class, T per mission
W = {}
for m, cls, L, T in re.findall(r':mission "([^"]+)" :class :(\w+) :L ([\d.]+) :T (\d+)',
                               (ROOT / "futon6/data/mission-wholeness.edn").read_text()):
    W[m] = dict(cls=cls, L=float(L), T=int(T))

def ymd(y, mo, d):
    return int(y) * 372 + int(mo) * 31 + int(d)

paths = {p.stem: p for p in ROOT.glob("futon*/holes/**/M-*.md")}
stems = set(paths)

def git_add_dates():
    out = {}
    for repo in sorted({p.relative_to(ROOT).parts[0] for p in paths.values()}):
        try:
            r = subprocess.run(["git", "-C", str(ROOT / repo), "log", "--diff-filter=A",
                                "--reverse", "--format=%aI", "--name-only"],
                               capture_output=True, text=True, timeout=90)
        except Exception:
            continue
        cur = None
        for line in r.stdout.splitlines():
            md = re.match(r'(\d{4})-(\d{2})-(\d{2})T', line)
            if md:
                cur = ymd(*md.groups())
            elif line.endswith(".md") and cur:
                st = Path(line).stem
                if st.startswith("M-") and st not in out:
                    out[st] = cur
    return out

git_dates = git_add_dates()
date_re = re.compile(r'\*{0,2}Date:?\*{0,2}\s*(\d{4})-(\d{2})-(\d{2})')
info = {}
outdeg = Counter(); indeg = Counter()
for s, p in paths.items():
    t = p.read_text(errors="ignore")
    m = date_re.search(t)
    ord_ = (ymd(*m.groups()) if m else None) or git_dates.get(s)   # semantic date primary, git fills gaps
    refs = {r for r in re.findall(r'\bM-[a-z0-9][a-z0-9-]+', t) if r in stems and r != s}
    info[s] = dict(ord=ord_, refs=refs)
    for r in refs:
        outdeg[s] += 1; indeg[r] += 1

dated = [s for s in stems if info[s]["ord"] is not None]
order = sorted(dated, key=lambda s: info[s]["ord"]) + sorted(s for s in stems if info[s]["ord"] is None)
SCALE, CX, CY = 72, 1050, 1050
pos = {s: (CX + SCALE * math.sqrt(i) * math.cos(i * GOLDEN),
           CY + SCALE * math.sqrt(i) * math.sin(i * GOLDEN)) for i, s in enumerate(order)}

def colour(s, light_boost=0):
    w = W.get(s)
    if not w:
        return "hsl(0,0%,28%)"
    hue = {"alive": 130, "mess": 9, "pipeline": 205, "stub": 0}[w["cls"]]
    sat = 0 if w["cls"] == "stub" else 72
    return f"hsl({hue},{sat}%,{28 + 44 * min(1, w['L'] / 90) + light_boost:.0f}%)"

# faint cross-ref web
edges = []
for s in order:
    x1, y1 = pos[s]
    for r in info[s]["refs"]:
        if r in pos:
            x2, y2 = pos[r]
            edges.append(f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}"/>')

# each mission = a mini-mandala: its full radial scope-tree (sections as petals)
MINI = {s: mini_data(s) for s in order}
MAXC = max((c for v in MINI.values() for c, _ in v), default=1)
minis = []
for s in order:
    x, y = pos[s]
    w = W.get(s, {})
    col = colour(s)
    R = 6 + 1.6 * w.get("T", 2)
    secs = MINI[s] or [(0, 0)]
    n = len(secs)
    parts = []
    for j, (cnt, nsub) in enumerate(secs):
        ang = j / n * 2 * math.pi - math.pi / 2
        ln = 3 + R * min(1, (cnt / MAXC) ** 0.5)
        x2, y2 = x + ln * math.cos(ang), y + ln * math.sin(ang)
        pr = 1.3 + 0.5 * min(nsub, 6)                  # petal size = #sub-scopes
        parts.append(f'<line x1="{x:.0f}" y1="{y:.0f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                     f'stroke="{col}" stroke-width="1.1" opacity="0.8"/>'
                     f'<circle cx="{x2:.1f}" cy="{y2:.1f}" r="{pr:.1f}" fill="{col}"/>')
    deg = indeg[s] + outdeg[s]
    minis.append(f'<g><title>{s}  L={w.get("L","?")} {w.get("cls","?")} '
                 f'{n} sections, refs={deg}</title>'
                 f'<circle cx="{x:.0f}" cy="{y:.0f}" r="2.4" fill="{col}" stroke="#000" '
                 f'stroke-width="0.4"/>{"".join(parts)}</g>')

labels = [f'<text x="{pos[s][0]+8:.0f}" y="{pos[s][1]:.0f}" fill="#e6e6e6" font-size="11">{s[2:]}</text>'
          for s, _ in indeg.most_common(8) if s in pos]

tally = Counter(W.get(s, {}).get("cls", "?") for s in order)
doc = f"""<!doctype html><meta charset=utf-8><title>Mission landscape — mandala of mandalas</title>
<style>body{{margin:0;background:#0a0c11;color:#cdd3df;font:13px sans-serif}}
header{{padding:14px 22px}} h1{{font-size:16px;margin:0 0 4px}} p{{margin:2px 0;color:#8b95a7;font-size:12px}}
.dot{{display:inline-block;width:10px;height:10px;border-radius:50%;vertical-align:middle;margin:0 4px}}
line.web{{stroke:#5a6a8a;stroke-width:0.4;opacity:0.05}}</style>
<header><h1>Mission landscape — a mandala of mandalas ({len(order)} missions)</h1>
<p>Each mission is its own eightfold mandala: a spoke per phase present (length = how filled),
<b>missing spokes = holes</b>; colour = L (mana) by class, brighter = more alive. Centre = "first in"
(oldest), spiraling out; faint web = cross-refs. Hover for L/class/refs. classes: {dict(tally)}</p>
<p><span class=dot style="background:hsl(130,72%,58%)"></span>alive
<span class=dot style="background:hsl(9,72%,48%)"></span>mess
<span class=dot style="background:hsl(205,72%,58%)"></span>pipeline
<span class=dot style="background:hsl(0,0%,42%)"></span>stub</p></header>
<svg width="2100" height="2100" viewBox="0 0 2100 2100">
<g>{''.join(f'<line class=web {e[6:]}' for e in edges)}</g>
<g>{''.join(minis)}</g><g>{''.join(labels)}</g></svg>"""
OUT.write_text(doc)
print(f"wrote {OUT}")
print(f"{len(order)} mini-mandalas, {len(edges)} cross-ref edges, classes {dict(tally)}")
print(f"centre (oldest): {', '.join(order[:5])}")
