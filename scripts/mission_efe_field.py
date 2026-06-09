#!/usr/bin/env python3
# mission_efe_field.py — D2 (claude-3): the per-step-cost METRIC field g(s) over Futon
# City at per-scope granularity — missions are DISTRICTS of scope-points spiralling
# around their HEAD hub; the metric topography is drawn as SMOOTH level sets (marching
# squares), over the faint pattern-road backdrop.
#   g(s) = per-step cost / local metric (epistemic pole). NOT the EFE. EFE = G(π) = the
#   geodesic over this; drawn later as policy streamlines. 🌟 claimed cap at its minting
#   mission; ⭐ unclaimed = registered goal w/ no minting mission (endpoint, no terrain).
import json, re, math, subprocess, time
from pathlib import Path
from collections import defaultdict

ROOT = Path("/home/joe/code")
POS = json.load(open(ROOT / "futon6/data/mission-carpet-pos.json"))
SCOPES = json.load(open("/tmp/scopes.json"))
CAPS = json.load(open(ROOT / "futon6/data/capability-graph.json"))
ROADS = json.load(open(ROOT / "futon6/data/mission-carpet-roads.json"))
OUT = ROOT / "futon6/data/mission-efe-field.html"
bare = lambda k: k[2:] if k.startswith("M-") else k

# Salingaros class (red=mess / green=alive / blue=pipeline / grey=stub) + phylogeny generativity
CLS = dict(re.findall(r':mission "M-([^"]+)" :class :(\w+)', (ROOT / "futon6/data/mission-wholeness.edn").read_text()))
_gblock = re.search(r':generativity-index \{([^}]*)\}', (ROOT / "futon6/data/mission-phylogeny.edn").read_text())
GEN = {bare(m): int(g) for m, g in re.findall(r'"(M-[^"]+)" (\d+)', _gblock.group(1) if _gblock else "")}
CLSCOL = {"alive": "#3a9a4a", "mess": "#c0392b", "pipeline": "#3a7ad0", "stub": "#777777"}
def ccol(m): return CLSCOL.get(CLS.get(m, ""), "#888888")

# --- districts: each mission's scopes spiral around its HEAD hub ---
by_m = defaultdict(list)
for sc in SCOPES:
    by_m[sc["m"]].append(sc)
ORDER = {"eightfold-phase": 0, "loose-section": 1, "mission-scope-in": 2, "mission-scope-out": 2,
         "map-item": 3, "source-material": 4, "relates-to": 5, "capability-scope": 6,
         "pattern": 7, "psr": 8, "pur": 8}
GOLD = math.pi * (3 - math.sqrt(5))
FRONTIER = {"capability-scope", "pattern", "psr", "pur"}
scope_pts, hub_lines, hubs = [], [], []
for m, scs in by_m.items():
    key = "M-" + m
    if key not in POS:
        continue
    cx, cy = POS[key]
    n = len(scs)
    R = 16 + 3.6 * math.sqrt(n)
    scs = sorted(scs, key=lambda s: ORDER.get(s["binder"], 9))
    hubs.append((cx, cy, m, n))
    for i, sc in enumerate(scs):
        ang = i * GOLD
        rad = R * math.sqrt((i + 0.5) / n)
        x, y = cx + rad * math.cos(ang), cy + rad * math.sin(ang)
        metric = 0.18 + (1.0 if sc["det"] else 0.0) + (0.30 if sc["binder"] in FRONTIER else 0.0)
        scope_pts.append((x, y, metric, sc["det"], ccol(m)))
        hub_lines.append((cx, cy, x, y))

# --- metric field on a vertex grid via scatter-add ---
W = H = 3600
STEP = 40
SIGMA = 70.0
gw, gh = W // STEP + 1, H // STEP + 1
grid = [[0.0] * gw for _ in range(gh)]
rc = int(3 * SIGMA / STEP)
for x, y, mtr, _det, _col in scope_pts:
    cgx, cgy = int(round(x / STEP)), int(round(y / STEP))
    for vy in range(max(0, cgy - rc), min(gh, cgy + rc + 1)):
        for vx in range(max(0, cgx - rc), min(gw, cgx + rc + 1)):
            d2 = (vx * STEP - x) ** 2 + (vy * STEP - y) ** 2
            grid[vy][vx] += mtr * math.exp(-d2 / (2 * SIGMA * SIGMA))
fmax = max(max(r) for r in grid) or 1.0
NB = 7
TERR = ["#0a0e1a", "#0f2236", "#143447", "#1d5347", "#3a7338", "#94862e", "#c2792a"]

# subtle banded fill (low opacity; the smooth contours carry the topo)
fill = []
for gy in range(gh - 1):
    for gx in range(gw - 1):
        b = min(NB - 1, int(grid[gy][gx] / fmax * NB))
        if grid[gy][gx] / fmax < 0.03:
            continue
        fill.append(f'<rect x="{gx*STEP}" y="{gy*STEP}" width="{STEP}" height="{STEP}" fill="{TERR[b]}" opacity="0.5"/>')

# smooth contour lines via marching squares, one set per band level
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
            op = 0.25 + 0.07 * li
            for k in range(0, len(cr) - 1, 2):
                (ax, ay), (bx, by) = cr[k], cr[k + 1]
                contour.append(f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{bx:.1f}" y2="{by:.1f}" '
                               f'stroke="#e6edff" stroke-width="1.1" opacity="{op:.2f}" stroke-linecap="round"/>')

# --- MOMENTUM overlay: "Joe's territory" — missions worked recently (git, last ~3 weeks) ---
# A warm LASSO (one bold dashed contour of a recency-weighted activity field), distinct from
# the white metric level-sets: it shows WHERE THE WORK HAS BEEN, the momentum/exploit baseline
# the EFE recommendation either confirms (inside) or breaks (outside).
def march(grid, lv):  # marching-squares segment list at level lv (reused for the lasso)
    segs = []
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
                segs.append((cr[k], cr[k + 1]))
    return segs

NOW = time.time()
MOM = defaultdict(float)
for repo in sorted({p.parents[2] for p in ROOT.glob("futon*/holes/missions/M-*.md")}):
    try:
        out = subprocess.run(["git", "-C", str(repo), "log", "--since=21 days ago",
                              "--pretty=format:%x01%ct", "--name-only"],
                             capture_output=True, text=True, timeout=25).stdout
    except Exception:
        continue
    t = None
    for ln in out.splitlines():
        if ln.startswith("\x01"):
            t = int(ln[1:]) if ln[1:].strip().isdigit() else None
        elif t and ln.endswith(".md"):
            base = ln.rsplit("/", 1)[-1]
            if base.startswith("M-"):
                MOM["M-" + base[2:-3]] += math.exp(-((NOW - t) / 86400) / 10.0)  # ~10-day decay
mgrid = [[0.0] * gw for _ in range(gh)]
MSIG = 125.0
mrc = int(3 * MSIG / STEP)
for k, w in MOM.items():
    if k not in POS:
        continue
    x, y = POS[k]
    cgx, cgy = int(round(x / STEP)), int(round(y / STEP))
    for vy in range(max(0, cgy - mrc), min(gh, cgy + mrc + 1)):
        for vx in range(max(0, cgx - mrc), min(gw, cgx + mrc + 1)):
            d2 = (vx * STEP - x) ** 2 + (vy * STEP - y) ** 2
            mgrid[vy][vx] += w * math.exp(-d2 / (2 * MSIG * MSIG))
mmax = max(max(r) for r in mgrid) or 1.0
LV = 0.30 * mmax
lasso_fill = [f'<rect x="{gx*STEP}" y="{gy*STEP}" width="{STEP}" height="{STEP}" fill="#ffae3b" opacity="0.045"/>'
              for gy in range(gh - 1) for gx in range(gw - 1) if mgrid[gy][gx] > LV]
lasso = "".join(f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" stroke="#ffb43c" '
                f'stroke-width="3.4" opacity="0.82" stroke-dasharray="11,8" stroke-linecap="round"/>'
                for a, b in march(mgrid, LV))
def in_territory(px, py):
    gx, gy = int(round(px / STEP)), int(round(py / STEP))
    return (0 <= gy < gh and 0 <= gx < gw and mgrid[gy][gx] > LV)

# DARK MATTER: momentum missions with NO scope-district — recently worked, not (yet) in
# substrate-2. They make empty lasso loops (gravity, no light); a ghost marker names them.
darkm = sorted((k for k in MOM if k in POS and k[2:] not in by_m and MOM[k] > 1.0),
               key=lambda k: -MOM[k])
ghosts = []
for k in darkm:
    gx0, gy0 = POS[k]; stem = k[2:]
    tt = (f"{stem} — DARK MATTER: recent git momentum ({MOM[k]:.1f}) but NO substrate-2 "
          f"scope-district — a recently-worked mission D1 hasn't ingested yet. Present on "
          f"momentum (the empty lasso loop), invisible to the metric.")
    ghosts.append(
        f'<g><title>{tt}</title>'
        f'<circle cx="{gx0:.0f}" cy="{gy0:.0f}" r="24" fill="#ffb43c" opacity="0.05" pointer-events="all"/>'
        f'<circle cx="{gx0:.0f}" cy="{gy0:.0f}" r="24" fill="none" stroke="#d8b066" stroke-width="1.3" '
        f'stroke-dasharray="4,4" opacity="0.85"/>'
        f'<text x="{gx0:.0f}" y="{gy0+5:.0f}" text-anchor="middle" font-size="16" fill="#d8b066" '
        f'pointer-events="all">⬡</text>'
        f'<text x="{gx0+28:.0f}" y="{gy0+4:.0f}" fill="#d8b066" font-size="12">{stem} · no substrate-2 district (dark matter)</text>'
        f'</g>')

# faint pattern-road backdrop (attestation-weighted)
roads = []
for a, b, w in ROADS:
    if a in POS and b in POS:
        x1, y1 = POS[a]; x2, y2 = POS[b]
        op = 0.04 + 0.006 * min(w, 60)
        roads.append(f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" '
                     f'stroke="#9a7fd0" stroke-width="{0.4+0.02*min(w,80):.1f}" opacity="{op:.2f}"/>')

hubline_svg = "".join(f'<line x1="{a:.0f}" y1="{b:.0f}" x2="{c:.1f}" y2="{d:.1f}" stroke="#54627f" stroke-width="0.4" opacity="0.22"/>'
                      for a, b, c, d in hub_lines)
scope_svg = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{2.4 if det else 1.4}" fill="{"#ffb454" if det else col}" opacity="{0.9 if det else 0.6}"/>'
                    for x, y, mtr, det, col in scope_pts)
# HEAD hubs: colour = Salingaros class (red/green/blue/grey), size = phylogeny generativity
hub_svg = "".join(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{2.6+1.5*math.sqrt(GEN.get(m,0)):.1f}" fill="{ccol(m)}" '
                  f'stroke="#04060c" stroke-width="0.9"><title>{m} · {CLS.get(m,"?")} · generativity {GEN.get(m,0)} · {n} scopes</title></circle>'
                  for x, y, m, n in hubs)

def starpoly(cx, cy, r, fill, stroke, label, title):
    p = []
    for i in range(10):
        a = math.pi / 2 + i * math.pi / 5
        rr = r if i % 2 == 0 else r * 0.42
        p.append(f"{cx+rr*math.cos(a):.1f},{cy-rr*math.sin(a):.1f}")
    return (f'<g><title>{title}</title>'
            f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="{r+4:.0f}" fill="#000" opacity="0" pointer-events="all"/>'
            f'<polygon points="{" ".join(p)}" fill="{fill}" stroke="{stroke}" stroke-width="1.7" pointer-events="all"/>'
            f'<text x="{cx+r+4:.0f}" y="{cy+5:.0f}" fill="#ffe08a" font-size="13">{label}</text></g>')

claimed = []
for cap, info in CAPS.items():
    if info["claimed"]:
        mp = [POS[mm] for mm in info["minted_by"] if mm in POS]
        if mp:
            cx = sum(p[0] for p in mp) / len(mp); cy = sum(p[1] for p in mp) / len(mp)
            t = f"{cap} — CLAIMED ({info['status']}). {info.get('title','')[:120]} · minted by: {', '.join(info['minted_by'])}"
            claimed.append(starpoly(cx, cy, 10, "#ffe08a", "#a8801f", cap, t.replace('"', "'")))  # FILLED = claimed
def cap_anchor(cap):  # centroid of the claimed ascent-parents' minting missions (a graph foothold)
    mp = []
    for p in CAPS[cap]["scope"]:
        pv = CAPS.get(p, {})
        if pv.get("claimed"):
            mp += [POS[m] for m in pv["minted_by"] if m in POS]
    return (sum(q[0] for q in mp) / len(mp), sum(q[1] for q in mp) / len(mp)) if mp else None

unclaimed = sorted((c for c, v in CAPS.items() if not v["claimed"]), key=str)
summit_svg, islands = [], []
for c in unclaimed:
    info = CAPS[c]; a = cap_anchor(c)
    if a:                                                        # SUMMIT — place above its claimed substrate
        cx, cy = a[0], a[1] - 46
        t = f"{c} — UNCLAIMED SUMMIT ({info['status']}): builds on claimed {info['scope']}. {info.get('title','')[:100]}"
        summit_svg.append(f'<line x1="{cx:.0f}" y1="{cy+46:.0f}" x2="{cx:.0f}" y2="{cy:.0f}" stroke="#ffd24a" '
                          f'stroke-width="0.9" stroke-dasharray="3,3" opacity="0.55"/>'
                          + starpoly(cx, cy, 12, "none", "#ffd24a", c, t.replace('"', "'")))
    else:
        islands.append(c)                                       # ISLAND — no terrain, needs constructing
sky = []
for i, c in enumerate(islands):
    info = CAPS[c]
    t = f"{c} — UNCLAIMED ISLAND ({info['status']}): NO ascent-anchor, no terrain — needs a constructed foothold. {info.get('title','')[:100]}"
    sky.append(starpoly(180 + i * 470, 80, 13, "none", "#ff8a6a", c, t.replace('"', "'")))  # red-ish = truly off-map

# --- specially MARKED missions (e.g. the WM's current recommendation) — 🚀 + explainer ---
MARKED = {
    "emacs-cursor-peripheral": (
        "War Machine recommendation",
        "Read-only Emacs peripheral — a visible agent cursor 'body' inside the existing Emacs "
        "session via the futon3c peripheral registry + WS transport (futon3 mission, in progress)."),
}
marks = []
for stem, (why, desc) in MARKED.items():
    key = "M-" + stem
    if key not in POS:
        continue
    mx, my = POS[key]
    mscs = by_m.get(stem, [])
    mdet = sum(1 for s in mscs if s["det"])
    mfront = sum(1 for s in mscs if s["binder"] in FRONTIER)
    title = (f"🚀 M-{stem} — {why}.  {desc}  "
             f"METRIC HERE: class={CLS.get(stem,'?')} · generativity {GEN.get(stem,0)} · "
             f"{len(mscs)} scopes, {mdet} open/:detached, {mfront} frontier.  "
             f"DIAGNOSTIC: low open-signal ({mdet}/{len(mscs)}) and no frontier scopes ⇒ a low "
             f"epistemic peak — so a WM pick here is likely pragmatic/where-driven, not terrain-driven. "
             f"Look at WHERE it sits: which class-neighbourhood, near which roads/stars.")
    t = title.replace('"', "'")
    marks.append(
        f'<g><title>{t}</title>'
        f'<circle cx="{mx:.0f}" cy="{my:.0f}" r="46" fill="none" stroke="#7fe0ff" stroke-width="1.0" opacity="0.40"/>'
        f'<circle cx="{mx:.0f}" cy="{my:.0f}" r="34" fill="#7fe0ff" opacity="0.07" pointer-events="all"/>'
        f'<circle cx="{mx:.0f}" cy="{my:.0f}" r="34" fill="none" stroke="#7fe0ff" stroke-width="2.4" opacity="0.95"/>'
        f'<text x="{mx:.0f}" y="{my+11:.0f}" text-anchor="middle" font-size="32" pointer-events="all">🚀</text>'
        f'<text x="{mx+42:.0f}" y="{my+5:.0f}" fill="#bdeaff" font-size="15" font-weight="bold">M-{stem} ◄ WM</text>'
        f'</g>')

doc = f"""<!doctype html><meta charset=utf-8><title>Futon City — per-scope metric field</title>
<style>body{{margin:0;background:#05060a;color:#cdd3df;font:13px sans-serif}}header{{padding:11px 20px}}
h1{{font-size:16px;margin:0 0 4px}}p{{margin:0;color:#8b95a7;font-size:12px;max-width:1180px}}
text{{cursor:default}}</style>
<header><h1>Futon City — per-step-cost <b>METRIC field</b> g(s), per-scope ({len(scope_pts)} scopes / {len(hubs)} districts) · 🌟{len(claimed)} claimed · ⭐{len(unclaimed)} unclaimed</h1>
<p><b>This is the metric (terrain), NOT the EFE</b> (EFE = G(π) = the geodesic over it, drawn later as policy
streamlines). Each mission is a DISTRICT — scopes spiral around the HEAD hub, <b>coloured by Salingaros class</b>
(<span style="color:#3a9a4a">green=alive</span> · <span style="color:#c0392b">red=mess</span> ·
<span style="color:#3a7ad0">blue=pipeline</span> · grey=stub) and <b>sized by generativity</b> (backlinks).
Orange points = open <b>:detached</b> holes (high ground); smooth level sets = topography; faint purple = pattern
roads. <b>★ filled = claimed</b> capability (at its minting mission) · <b>☆ empty = unclaimed</b> (sky: a registered
goal with no minting mission). <b>🚀 = a specially-marked mission (the WM's current recommendation)</b> —
cyan ring, hover for its story + metric diagnostic. <b><span style="color:#ffb43c">amber dashed lasso = YOUR
territory</span></b> (missions worked in git's last ~3 weeks — the momentum/exploit baseline; inside = the WM
confirms, outside = it breaks trend). <b><span style="color:#d8b066">⬡ = dark matter</span></b> (a lasso loop with
momentum but no substrate-2 district — a mission worked but not yet ingested). <b>Hover any star or hub for its story.</b></p></header>
<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<rect x="0" y="0" width="{W}" height="150" fill="#0c0f18"/>
<g>{''.join(fill)}</g><g>{''.join(lasso_fill)}</g><g>{''.join(roads)}</g><g>{''.join(contour)}</g>
<g>{hubline_svg}</g><g>{scope_svg}</g><g>{hub_svg}</g>
<g>{lasso}</g><g>{''.join(ghosts)}</g>
<g>{''.join(claimed)}</g><g>{''.join(summit_svg)}</g><g>{''.join(sky)}</g>
<g>{''.join(marks)}</g></svg>"""
OUT.write_text(doc)
print(f"wrote {OUT}")
print(f"{len(scope_pts)} scopes / {len(hubs)} districts · {sum(1 for p in scope_pts if p[3])} holes · "
      f"{len(contour)} contour segs · {len(roads)} roads · 🌟{len(claimed)} ⭐{len(unclaimed)}")
_top = sorted(MOM.items(), key=lambda kv: -kv[1])[:10]
print("momentum (recent-git, top): " + ", ".join(f"{k[2:]}={v:.2f}" for k, v in _top))
_p = POS.get("M-emacs-cursor-peripheral")
print(f"lasso level={LV:.3f}/mmax={mmax:.3f} · 🚀 emacs-cursor-peripheral inside YOUR territory? "
      f"{in_territory(*_p) if _p else 'n/a'}")
