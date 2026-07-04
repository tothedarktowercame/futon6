#!/usr/bin/env python3
# mission_efe_field.py — D2 (claude-3): the per-step-cost METRIC field g(s) over Futon
# City at per-scope granularity — missions are DISTRICTS of scope-points spiralling
# around their HEAD hub; the metric topography is drawn as SMOOTH level sets (marching
# squares), over the faint pattern-road backdrop.
#   g(s) = per-step cost / local metric (epistemic pole). NOT the EFE. EFE = G(π) = the
#   geodesic over this; drawn later as policy streamlines. 🌟 claimed cap at its minting
#   mission; ⭐ unclaimed = registered goal w/ no minting mission (endpoint, no terrain).
import json, re, math, subprocess, time, sys
from pathlib import Path
from collections import defaultdict

ROOT = Path("/home/joe/code")
# Optional variant arg (force|embed|springs|seed) selects an alternate mission layout from
# mission_carpet_variants.py — the projection FAMILY. No arg = the canonical carpet, unchanged.
_VARIANT = sys.argv[1] if len(sys.argv) > 1 else None
POS = json.load(open(ROOT / "futon6/data" /
                     (f"mission-carpet-pos-{_VARIANT}.json" if _VARIANT else "mission-carpet-pos.json")))
SCOPES = json.load(open(ROOT / "futon6/data/efe-scopes.json"))  # reproducible: scripts/mission_efe_scope_dump.py
CAPS = json.load(open(ROOT / "futon6/data/capability-graph.json"))
ROADS = json.load(open(ROOT / "futon6/data/mission-carpet-roads.json"))
OUT = ROOT / "futon6/data" / (f"mission-efe-field-{_VARIANT}.html" if _VARIANT else "mission-efe-field.html")
bare = lambda k: k[2:] if k.startswith("M-") else k

# Off-map minters: builder/* and external/* are NOT missions, so a claimed cap
# minted only by them has no carpet position and would silently not render
# (kit-observables and 10 others). Anchor a builder-minted claimed cap at the
# mission that OWNS the builder — the faithful extension of the field semantic
# "claimed cap at its minting mission". Only FINDABLE owners are mapped here;
# ambiguous (builder/wm-*: four builders, no single owner) and external/* caps
# (no host mission by nature — off-map terrain) are left unplaced and FLAGGED
# for an operator semantics decision rather than guessed.
BUILDER_HOST_MISSION = {
    "builder/pudding-prover":    "M-pudding-peradams",
    "builder/futon7-daily-scan": "M-daily-scan",
    "builder/ct-prototype":      "M-symbol-grounding",
}

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
ORDER = {"eightfold-phase": 0, "loose-section": 1, "plain-argument": 1, "mission-scope-in": 2,
         "mission-scope-out": 2, "map-item": 3, "source-material": 4, "relates-to": 5,
         "capability-scope": 6, "pattern": 7, "psr": 8, "pur": 8,
         "verify-gate": 9, "certificate": 10}
GOLD = math.pi * (3 - math.sqrt(5))
FRONTIER = {"capability-scope", "pattern", "psr", "pur", "verify-gate"}
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
        vac = bool(sc.get("vacuous"))
        verdict = sc.get("verdict")
        metric = 0.18 + (1.0 if sc["det"] else 0.0) + (0.30 if sc["binder"] in FRONTIER else 0.0)
        # anatomy terms (2026-06-12 redraw): a vacuous scope (binder with no named
        # entities inside) is suspect terrain; a certificate re-grades its district
        # by its verdict — verified ground is LOW cost, known-broken ground is high.
        if vac:
            metric += 0.5
        if sc["binder"] == "certificate":
            metric = max(0.05, metric - 0.45) if verdict == "pass" else metric + 0.8
        scope_pts.append((x, y, metric, sc["det"], ccol(m), vac, verdict))
        hub_lines.append((cx, cy, x, y))

# --- metric field on a vertex grid via scatter-add ---
W = H = 3600
STEP = 40
SIGMA = 70.0
gw, gh = W // STEP + 1, H // STEP + 1
grid = [[0.0] * gw for _ in range(gh)]
rc = int(3 * SIGMA / STEP)
for x, y, mtr, _det, _col, _vac, _ver in scope_pts:
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
def scope_mark(x, y, mtr, det, col, vac, verdict):
    if verdict is not None:  # certificate: verdict diamond, green pass / red fail
        c = "#4ade80" if verdict == "pass" else "#ef4444"
        return (f'<path d="M {x:.1f} {y-4.4:.1f} L {x+4.4:.1f} {y:.1f} L {x:.1f} {y+4.4:.1f} '
                f'L {x-4.4:.1f} {y:.1f} Z" fill="{c}" stroke="#04060c" stroke-width="0.6" opacity="0.95"/>')
    if vac:  # vacuous scope: hollow ring — a binder with nothing bound inside
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" fill="none" stroke="{col}" stroke-width="1.0" opacity="0.85"/>'
    return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{2.4 if det else 1.4}" '
            f'fill="{"#ffb454" if det else col}" opacity="{0.9 if det else 0.6}"/>')
scope_svg = "".join(scope_mark(*pt) for pt in scope_pts)
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
offmap_unplaced = []                                             # claimed caps with no anchor — flagged, not silently dropped
_placements = []                                                 # (cap, cx, cy, title) before de-overlap
for cap, info in CAPS.items():
    if info["claimed"]:
        mp = [POS[mm] for mm in info["minted_by"] if mm in POS]
        anchored_via = None
        if not mp:                                              # off-map minter (builder/* or external/*)
            hosts = [BUILDER_HOST_MISSION.get(mm) for mm in info["minted_by"]]
            hosts = [h for h in hosts if h and h in POS]
            mp = [POS[h] for h in hosts]
            if mp:
                anchored_via = "owning mission: " + ", ".join(hosts)
        if mp:
            cx = sum(p[0] for p in mp) / len(mp); cy = sum(p[1] for p in mp) / len(mp)
            via = f" · anchored at {anchored_via}" if anchored_via else ""
            t = f"{cap} — CLAIMED ({info['status']}). {info.get('title','')[:120]} · minted by: {', '.join(info['minted_by'])}{via}"
            _placements.append((cap, cx, cy, t.replace('"', "'")))
        else:
            offmap_unplaced.append(cap)
# De-overlap: caps sharing one anchor (the kit-* family all anchor at
# M-pudding-peradams — the first co-location the field has, since a builder mints
# several caps per mission) fan onto a small ring so each star LANDS distinctly
# instead of stacking invisibly. Single-occupant points are unchanged.
_by_pt = defaultdict(list)
for pl in _placements:
    _by_pt[(round(pl[1]), round(pl[2]))].append(pl)
for grp in _by_pt.values():
    if len(grp) == 1:
        cap, cx, cy, t = grp[0]
        claimed.append(starpoly(cx, cy, 10, "#ffe08a", "#a8801f", cap, t))      # FILLED = claimed
    else:
        for i, (cap, cx, cy, t) in enumerate(sorted(grp, key=lambda g: g[0])):  # ring fan-out, deterministic by name
            a = 2 * math.pi * i / len(grp)
            claimed.append(starpoly(cx + 18*math.cos(a), cy - 18*math.sin(a), 10, "#ffe08a", "#a8801f", cap, t))
def cap_anchor(cap):  # centroid of the claimed ascent-parents' minting missions (a graph foothold)
    mp = []
    for p in CAPS[cap]["scope"]:
        pv = CAPS.get(p, {})
        if pv.get("claimed"):
            mp += [POS[m] for m in pv["minted_by"] if m in POS]
    return (sum(q[0] for q in mp) / len(mp), sum(q[1] for q in mp) / len(mp)) if mp else None

# Projection-layer grounding (same status as BUILDER_HOST_MISSION / the pudding-kit
# coalescing in starmap_to_capability_graph.bb): a mission that GROUNDS an unclaimed
# cap without minting it. The curated EDN stays untouched — M-cold-chain exit cond. 4
# reserves the minted-by flip for the curators' channel; this map is display-only and
# claims nothing. Warrants:
#   cold-*: futon7/holes/M-cold-chain.md ("four cold-* stars = one ladder", rung table)
#   kit-*:  pudding-prover-registry.edn — held kits of the family whose claimed
#           siblings (pudding-kit cluster) already anchor at M-pudding-peradams
# Value = (grounding mission, rung) — rung orders a multi-cap ladder bottom-up.
GROUNDED_BY = {
    "cold-eoi-authored-outbox": ("M-cold-chain", 1),
    "cold-eoi-sent":            ("M-cold-chain", 2),
    "cold-send-response":       ("M-cold-chain", 3),
    "cold-response-conversion": ("M-cold-chain", 4),
    "kit-outbox":               ("M-pudding-peradams", 1),
    "kit-intake":               ("M-pudding-peradams", 2),
    "kit-cadence":              ("M-pudding-peradams", 3),
}

unclaimed = sorted((c for c, v in CAPS.items() if not v["claimed"]), key=str)
# Anchor resolution for unclaimed caps, strongest foothold first:
#   1. own minting mission on the map (in flight: minted-by recorded in the curated
#      EDN while the cap is still :held) — incl. the builder-host fallback, which the
#      claimed branch already gets;
#   2. GROUNDED_BY (projection-layer, warrants above);
#   3. claimed scope-parents' centroid (the original SUMMIT);
#   4. an already-anchored unclaimed scope-parent (transitive — summit on a summit).
# Only a cap with no foothold under all four lands in the sky.
def _own_minter(info):
    via = [m for m in info["minted_by"] if m in POS]
    if not via:
        via = [h for h in (BUILDER_HOST_MISSION.get(m) for m in info["minted_by"]) if h and h in POS]
    if via:
        pts = [POS[m] for m in via]
        return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts), via)
anchors = {}                                                     # cap -> (x, y, kind, via)
for c in unclaimed:
    info = CAPS[c]
    om = _own_minter(info)
    if om:
        anchors[c] = (om[0], om[1], "minting", ", ".join(om[2])); continue
    if c in GROUNDED_BY and GROUNDED_BY[c][0] in POS:
        gm = GROUNDED_BY[c][0]
        anchors[c] = (POS[gm][0], POS[gm][1], "grounded", gm); continue
    a = cap_anchor(c)
    if a:
        anchors[c] = (a[0], a[1], "summit", "claimed " + ", ".join(info["scope"]))
for _ in range(3):                                               # transitive pass (4)
    for c in unclaimed:
        if c in anchors:
            continue
        pa = [anchors[p] for p in CAPS[c]["scope"] if p in anchors]
        if pa:
            via = ", ".join(p for p in CAPS[c]["scope"] if p in anchors)
            anchors[c] = (sum(p[0] for p in pa) / len(pa), sum(p[1] for p in pa) / len(pa),
                          "summit-chain", "unclaimed " + via)

KINDLBL = {"minting":      "MINTING IN PROGRESS at",
           "grounded":     "GROUNDED (not minted) by",
           "summit":       "SUMMIT: builds on",
           "summit-chain": "SUMMIT-CHAIN: stacks on"}
summit_svg, islands = [], []
_anchor_groups = defaultdict(list)
for c in unclaimed:
    if c in anchors:
        _anchor_groups[(round(anchors[c][0]), round(anchors[c][1]))].append(c)
    else:
        islands.append(c)                                       # ISLAND — no terrain, needs constructing
for grp in _anchor_groups.values():
    grp.sort(key=lambda c: (GROUNDED_BY.get(c, ("", 99))[1], c))  # ladder rung order where known
    for j, c in enumerate(grp):                                  # co-anchored caps stack as rungs
        x, y, kind, via = anchors[c]
        cx, cy = x, y - 46 - 42 * j
        info = CAPS[c]
        t = f"{c} — UNCLAIMED ({info['status']}) · {KINDLBL[kind]} {via}. {info.get('title','')[:100]}"
        summit_svg.append(f'<line x1="{cx:.0f}" y1="{y:.0f}" x2="{cx:.0f}" y2="{cy:.0f}" stroke="#ffd24a" '
                          f'stroke-width="0.9" stroke-dasharray="3,3" opacity="0.55"/>'
                          + starpoly(cx, cy, 12, "none", "#ffd24a", c, t.replace('"', "'")))
# --- the SKY: strictly the OFF-MAP registry — anything with no place in the terrain.
#   hollow red  = UNCLAIMED island (a registered goal with no foothold anywhere);
#   filled, red-rimmed = CLAIMED cap whose minters have no carpet position (external/*
#   or ambiguous builder/* owners) — real inventory with no address; previously these
#   were dropped to a stdout warning and never rendered at all.
sky = []
_per_row = max(1, (W - 360) // 470)                              # wrap: keep every star inside the viewBox
_sky_items = [(c, False) for c in islands] + [(c, True) for c in sorted(offmap_unplaced)]
SKY_H = max(150, 80 + 52 * ((max(len(_sky_items), 1) - 1) // _per_row) + 60)
for i, (c, is_claimed) in enumerate(_sky_items):
    info = CAPS[c]
    sx, sy = 180 + (i % _per_row) * 470, 80 + 52 * (i // _per_row)
    if is_claimed:
        t = (f"{c} — CLAIMED but OFF-MAP ({info['status']}): minted by {', '.join(info['minted_by'])} — "
             f"no carpet position for any minter; operator semantics decision pending. {info.get('title','')[:100]}")
        sky.append(starpoly(sx, sy, 13, "#ffe08a", "#c0392b", c, t.replace('"', "'")))
    else:
        t = (f"{c} — UNCLAIMED ISLAND ({info['status']}): NO foothold (own minter, grounding mission, "
             f"scope-parents all came up empty) — needs a constructed foothold. {info.get('title','')[:100]}")
        sky.append(starpoly(sx, sy, 13, "none", "#ff8a6a", c, t.replace('"', "'")))  # red-ish = truly off-map

# --- specially MARKED missions (kept empty for this page's live v1 contract) ---
MARKED = {}
marks = []
for stem, (why, desc) in MARKED.items():
    key = "M-" + stem
    if key not in POS:
        continue
    mx, my = POS[key]
    mscs = by_m.get(stem, [])
    mdet = sum(1 for s in mscs if s["det"])
    mfront = sum(1 for s in mscs if s["binder"] in FRONTIER)
    mvac = sum(1 for s in mscs if s.get("vacuous"))
    mcert = [s.get("verdict") for s in mscs if s["binder"] == "certificate"]
    title = (f"🚀 M-{stem} — {why}.  {desc}  "
             f"METRIC HERE: class={CLS.get(stem,'?')} · generativity {GEN.get(stem,0)} · "
             f"{len(mscs)} scopes, {mdet} open/:detached, {mfront} frontier, {mvac} vacuous, certs={mcert or 'none'}.  "
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

LIVE_OVERLAY_STYLE = """
#live-status{display:inline-block;margin-left:10px;padding:2px 7px;border:1px solid #334155;border-radius:999px;color:#94a3b8;background:#0b1020;font-size:11px}
#live-status.live{color:#bbf7d0;border-color:#22c55e;background:#052e16}
#live-status.offline{color:#fecaca;border-color:#ef4444;background:#300b12}
#live-overlay text{font-family:ui-sans-serif,system-ui,sans-serif;paint-order:stroke;stroke:#05060a;stroke-width:3px;stroke-linejoin:round}
.live-frontier-label{fill:#f8e7a0;font-size:16px;font-weight:700}
.live-frontier-item{fill:#f8e7a0;font-size:12px}
.live-agent-label{fill:#dbeafe;font-size:12px;font-weight:700}
.live-agent-ring{stroke-width:1.7}
.live-agent-idle{opacity:.72}
.live-agent-invoking{opacity:1;animation:liveAgentPulse 1.4s ease-in-out infinite}
.live-session-glyph{font-size:18px;text-anchor:middle;dominant-baseline:central;stroke:none}
.live-approx{fill:none;stroke-dasharray:5 4}
.live-wm-label{fill:#fff7ad;font-size:13px;font-weight:800}
.live-wm-target{stroke:#fb7185}
.live-wm-enacted{stroke:#facc15}
.live-wm-ring{fill:none;stroke-width:2.5;opacity:.9}
.live-wm-pulse{animation:liveWmPulse 1.8s ease-out infinite;transform-box:fill-box;transform-origin:center}
.live-ship-halo{fill:none;stroke:#ff9f1c;stroke-width:2.2;opacity:.95}
.live-ship-pulse{animation:liveWmPulse 1.7s ease-out infinite;transform-box:fill-box;transform-origin:center}
.live-ship-glyph{font-size:30px;text-anchor:middle;dominant-baseline:central;stroke:none}
.live-ship-label{fill:#ffd89a;font-size:13px;font-weight:800}
.live-offline-badge{fill:#fecaca;font-size:22px;font-weight:800}
@keyframes liveAgentPulse{0%,100%{opacity:.65}50%{opacity:1}}
@keyframes liveWmPulse{0%{opacity:.95;transform:scale(.82)}70%{opacity:.08;transform:scale(1.45)}100%{opacity:0;transform:scale(1.55)}}
"""

LIVE_OVERLAY_SCRIPT = """
<script>
(() => {
  const NS = "http://www.w3.org/2000/svg";
  const ENDPOINT = "http://localhost:7070/api/alpha/live-efe-map";
  const REFRESH_MS = 10000;
  const FRONTIER_Y = 3410;
  const FRONTIER_H = 175;
  const svg = document.getElementById("efe-field");
  const badge = document.getElementById("live-status");
  if (!svg) return;

  function el(name, attrs = {}, text = null) {
    const node = document.createElementNS(NS, name);
    for (const [k, v] of Object.entries(attrs)) {
      if (v !== null && v !== undefined) node.setAttribute(k, String(v));
    }
    if (text !== null) node.textContent = text;
    return node;
  }

  const layer = el("g", {id: "live-overlay"});
  svg.appendChild(layer);

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function setBadge(kind, text) {
    if (!badge) return;
    badge.className = kind;
    badge.textContent = text;
  }

  function validPlacement(p) {
    return p && Number.isFinite(Number(p.x)) && Number.isFinite(Number(p.y));
  }

  function placementKind(p) {
    return (p && p.placement) || "unknown";
  }

  function isShelf(p) {
    return placementKind(p) === "frontier-shelf";
  }

  function isEmbedded(p) {
    return placementKind(p) === "embedded";
  }

  function isExcursion(id) {
    return String(id || "").startsWith("E-");
  }

  function fmt(n) {
    return Number.isFinite(Number(n)) ? Number(n).toFixed(2) : "n/a";
  }

  function anchorsText(p) {
    const anchors = (p && p.anchors) || [];
    if (!anchors.length) return "none";
    return anchors.map((a) => {
      if (typeof a === "string") return a;
      return a["mission-id"] || a.id || a.label || JSON.stringify(a);
    }).join(", ");
  }

  function placementTitle(p) {
    if (!p) return "placement: none";
    return `placement=${placementKind(p)} method=${p.method || "unknown"} confidence=${p.confidence ?? "?"} anchors=${anchorsText(p)}`;
  }

  function title(parent, text) {
    parent.appendChild(el("title", {}, text));
  }

  function drawFrontierBand(data) {
    const g = el("g", {"data-layer": "frontier-band"});
    g.appendChild(el("rect", {x: 0, y: FRONTIER_Y, width: 3600, height: FRONTIER_H, fill: "#09111f", opacity: 0.88}));
    g.appendChild(el("line", {x1: 0, y1: FRONTIER_Y, x2: 3600, y2: FRONTIER_Y, stroke: "#f8e7a0", "stroke-width": 1.3, "stroke-dasharray": "9 7", opacity: 0.8}));
    g.appendChild(el("text", {x: 24, y: FRONTIER_Y + 28, class: "live-frontier-label"}, "frontier shelf — off-map live placements"));
    for (const item of ((data.frontier && data.frontier.items) || [])) {
      if (!validPlacement(item)) continue;
      const missionId = item["mission-id"] || "unknown";
      const itemG = el("g", {"data-live-kind": "frontier", "data-mission-id": missionId});
      title(itemG, `${missionId}${isExcursion(missionId) ? " — excursion" : ""}\\n${placementTitle(item)}`);
      itemG.appendChild(el("circle", {cx: item.x, cy: item.y, r: 7, fill: "none", stroke: "#f8e7a0", "stroke-width": 1.8, "stroke-dasharray": "4 3"}));
      itemG.appendChild(el("text", {x: Number(item.x) + 12, y: Number(item.y) + 4, class: "live-frontier-item"}, `${missionId}${isExcursion(missionId) ? " · excursion" : ""}`));
      g.appendChild(itemG);
    }
    layer.appendChild(g);
  }

  function drawPlacementMarker(parent, p, attrs) {
    const cx = Number(p.x);
    const cy = Number(p.y);
    const r = attrs.r || 7;
    const glyph = attrs.glyph || null;
    const ringClass = `live-agent-ring ${attrs.className || ""}`;
    if (isEmbedded(p)) {
      parent.appendChild(el("circle", {cx, cy, r: r + 4, fill: attrs.fill, opacity: 0.18, stroke: attrs.fill, "stroke-width": attrs.strokeWidth || 1.7, class: ringClass}));
      if (glyph) parent.appendChild(el("text", {x: cx, y: cy + 1, class: `live-session-glyph ${attrs.className || ""}`, style: "stroke:none;paint-order:normal"}, glyph));
      return;
    }
    parent.appendChild(el("circle", {cx, cy, r: r + 4, fill: "none", stroke: attrs.fill, "stroke-width": attrs.strokeWidth || 1.8, "stroke-dasharray": isShelf(p) ? "6 4" : "4 3", class: `live-approx ${ringClass}`}));
    if (glyph) parent.appendChild(el("text", {x: cx, y: cy + 1, class: `live-session-glyph ${attrs.className || ""}`, style: "stroke:none;paint-order:normal"}, glyph));
  }

  function agentActive(agent) {
    const status = String(agent.status || "").toLowerCase();
    const activity = String(agent["invoke-activity"] || "").toLowerCase();
    return status === "invoking" || activity === "invoking" || activity === "active" || Boolean(agent["running-job-id"]);
  }

  function drawAgents(data) {
    for (const agent of ((data.agents && data.agents.items) || [])) {
      if (!agent["mission-id"] || !validPlacement(agent.placement)) continue;
      const p = agent.placement;
      const active = agentActive(agent);
      const g = el("g", {"data-live-kind": "agent", "data-agent-id": agent["agent-id"], "data-mission-id": agent["mission-id"]});
      title(g, `${agent["agent-id"]}\\nmission=${agent["mission-id"]}${isExcursion(agent["mission-id"]) ? " (excursion)" : ""}\\nclock-source=${agent["clock-source"] || "unknown"}\\nstatus=${agent.status || "unknown"}\\n${placementTitle(p)}`);
      drawPlacementMarker(g, p, {
        r: active ? 8 : 6,
        fill: active ? "#67e8f9" : "#7a879b",
        glyph: "🛸",
        className: active ? "live-agent-invoking" : "live-agent-idle"
      });
      g.appendChild(el("text", {x: Number(p.x) + 11, y: Number(p.y) - 9, class: "live-agent-label"}, agent["agent-id"] || "agent"));
      if (isShelf(p) || isExcursion(agent["mission-id"])) {
        g.appendChild(el("text", {x: Number(p.x) + 11, y: Number(p.y) + 8, class: "live-frontier-item"}, "excursion"));
      }
      layer.appendChild(g);
    }
  }

  function shipContributors(ship) {
    return ((ship && ship.contributing) || []).map((c) => `${c["agent-id"] || "agent"} -> ${c["mission-id"] || "unknown"}`).join("\\n");
  }

  function drawShip(data) {
    const ship = data.ship;
    if (!validPlacement(ship)) return;
    const cx = Number(ship.x);
    const cy = Number(ship.y);
    const n = ship["session-count"] || ((ship.contributing || []).length);
    const g = el("g", {"data-live-kind": "ship", "data-method": ship.method || "unknown"});
    title(g, `operator centroid of ${n} active sessions\\nmethod=${ship.method || "unknown"}\\n${shipContributors(ship)}`);
    g.appendChild(el("circle", {cx, cy, r: 34, class: "live-ship-halo live-ship-pulse"}));
    g.appendChild(el("circle", {cx, cy, r: 20, class: "live-ship-halo"}));
    g.appendChild(el("text", {x: cx, y: cy + 1, class: "live-ship-glyph", style: "stroke:none;paint-order:normal"}, "🚀"));
    g.appendChild(el("text", {x: cx + 25, y: cy - 18, class: "live-ship-label"}, `operator centroid · ${n}`));
    layer.appendChild(g);
  }

  function drawWmPoint(item, slot, color, dx, dy) {
    const wrapped = item[slot];
    const p = wrapped && wrapped.placement;
    if (!validPlacement(p)) return;
    const missionId = wrapped["mission-id"] || (p && p["mission-id"]) || "unknown";
    const cx = Number(p.x);
    const cy = Number(p.y);
    const g = el("g", {"data-live-kind": `wm-${slot}`, "data-mission-id": missionId});
    title(g, `WM ${slot}: ${missionId}${isExcursion(missionId) ? " (excursion)" : ""}\\ndecision=${item.decision || "unknown"} G=${fmt(item.G)} expected-G=${fmt(item["expected-G"])} realized-G=${fmt(item["realized-G"])}\\ntrigger=${item.trigger || "unknown"}\\n${placementTitle(p)}`);
    g.appendChild(el("circle", {cx, cy, r: 28, fill: "none", stroke: color, "stroke-width": 1.4, opacity: 0.28}));
    g.appendChild(el("circle", {cx, cy, r: 17, fill: "none", stroke: color, "stroke-width": 2.4, "stroke-dasharray": isEmbedded(p) ? null : "6 4", class: `live-wm-ring live-wm-${slot}`}));
    g.appendChild(el("circle", {cx, cy, r: 23, fill: "none", stroke: color, "stroke-width": 2.0, class: `live-wm-ring live-wm-${slot} live-wm-pulse`}));
    g.appendChild(el("text", {x: cx + dx, y: cy + dy, class: "live-wm-label"}, `${slot} G ${fmt(item.G)}`));
    layer.appendChild(g);
  }

  function drawWarMachine(data) {
    // Ticks arrive newest-first; repeat ticks on the same mission would stack
    // dozens of identical pulses. Draw only the latest tick per mission+slot.
    const seen = new Set();
    for (const item of ((data["war-machine"] && data["war-machine"].items) || [])) {
      for (const [slot, color, dx, dy] of [["enacted", "#facc15", 24, -20], ["target", "#fb7185", 24, 22]]) {
        const wrapped = item[slot];
        const missionId = wrapped && wrapped["mission-id"];
        const key = `${slot}:${missionId}`;
        if (!missionId || seen.has(key)) continue;
        seen.add(key);
        drawWmPoint(item, slot, color, dx, dy);
      }
    }
  }

  function draw(data) {
    clear(layer);
    drawFrontierBand(data);
    drawWarMachine(data);
    drawAgents(data);
    drawShip(data);
  }

  function drawOffline(error) {
    clear(layer);
    const g = el("g", {"data-live-kind": "offline"});
    g.appendChild(el("rect", {x: 18, y: 54, width: 205, height: 34, rx: 6, fill: "#300b12", stroke: "#ef4444", opacity: 0.92}));
    g.appendChild(el("text", {x: 32, y: 77, class: "live-offline-badge"}, "live layer offline"));
    title(g, error ? String(error) : "live endpoint unreachable");
    layer.appendChild(g);
  }

  async function refresh() {
    try {
      const resp = await fetch(ENDPOINT, {cache: "no-store"});
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      if (!data.ok) throw new Error("endpoint returned ok=false");
      draw(data);
      const agents = data.agents ? data.agents["with-placement"] : 0;
      const wm = data["war-machine"] ? data["war-machine"].count : 0;
      setBadge("live", `live layer on · ${agents} agents · ${wm} WM`);
    } catch (err) {
      drawOffline(err);
      setBadge("offline", "live layer offline");
    }
  }

  refresh();
  window.setInterval(refresh, REFRESH_MS);
})();
</script>
"""

doc = f"""<!doctype html><meta charset=utf-8><title>Futon City — per-scope metric field</title>
<style>body{{margin:0;background:#05060a;color:#cdd3df;font:13px sans-serif}}header{{padding:11px 20px}}
h1{{font-size:16px;margin:0 0 4px}}p{{margin:0;color:#8b95a7;font-size:12px;max-width:1180px}}
text{{cursor:default}}{LIVE_OVERLAY_STYLE}</style>
<header><h1>Futon City — per-step-cost <b>METRIC field</b> g(s), per-scope ({len(scope_pts)} scopes / {len(hubs)} districts) · 🌟{len(claimed)} claimed · ⭐{len(unclaimed)} unclaimed <span id="live-status">live layer loading</span></h1>
<p><b>This is the metric (terrain), NOT the EFE</b> (EFE = G(π) = the geodesic over it, drawn later as policy
streamlines). Each mission is a DISTRICT — scopes spiral around the HEAD hub, <b>coloured by Salingaros class</b>
(<span style="color:#3a9a4a">green=alive</span> · <span style="color:#c0392b">red=mess</span> ·
<span style="color:#3a7ad0">blue=pipeline</span> · grey=stub) and <b>sized by generativity</b> (backlinks).
Orange points = open <b>:detached</b> holes (high ground); smooth level sets = topography; faint purple = pattern
roads. <b>Anatomy marks (2026-06-12)</b>: hollow rings = <b>vacuous scopes</b> (a binder with no named
entities inside — suspect terrain, +cost); <span style="color:#4ade80">◆ green diamond = certificate
PASS</span> (verified ground, −cost) · <span style="color:#ef4444">◆ red = FAIL</span> (+cost); verify-gates
count as frontier. <b>★ filled = claimed</b> capability (at its minting mission) · <b>☆ empty gold = unclaimed but
anchored</b> (tethered to its foothold — an in-flight minting mission, a grounding mission, or its claimed substrate;
hover for which; co-anchored stars stack as ladder rungs). <b>The sky holds only what is OFF-MAP</b>:
<b><span style="color:#ff8a6a">☆ red = unclaimed, no foothold anywhere</span></b> (a registered goal with no path
built) · <b><span style="color:#c0392b">★ red-rimmed filled = claimed but off-map</span></b> (its minting missions
have no district on the carpet — operator semantics decision pending).
<b><span style="color:#ffb43c">amber dashed lasso = YOUR
territory</span></b> (missions worked in git's last ~3 weeks — the momentum/exploit baseline; inside = the WM
confirms, outside = it breaks trend). <b><span style="color:#d8b066">⬡ = dark matter</span></b> (a lasso loop with
momentum but no substrate-2 district — a mission worked but not yet ingested). <b>Live overlay:</b> WM attention and agent telemetry on the EFE landscape. <b>Hover any star, hub, or live marker for its story.</b></p></header>
<svg id="efe-field" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<rect x="0" y="0" width="{W}" height="{SKY_H}" fill="#0c0f18"/>
<g>{''.join(fill)}</g><g>{''.join(lasso_fill)}</g><g>{''.join(roads)}</g><g>{''.join(contour)}</g>
<g>{hubline_svg}</g><g>{scope_svg}</g><g>{hub_svg}</g>
<g>{lasso}</g><g>{''.join(ghosts)}</g>
<g>{''.join(claimed)}</g><g>{''.join(summit_svg)}</g><g>{''.join(sky)}</g>
<g>{''.join(marks)}</g></svg>{LIVE_OVERLAY_SCRIPT}"""
OUT.write_text(doc)
print(f"wrote {OUT}")
print(f"{len(scope_pts)} scopes / {len(hubs)} districts · {sum(1 for p in scope_pts if p[3])} holes · "
      f"{len(contour)} contour segs · {len(roads)} roads · 🌟{len(claimed)} ⭐{len(unclaimed)}")
if offmap_unplaced:
    print(f"⚠ {len(offmap_unplaced)} claimed cap(s) still off-map (no owning-mission anchor; "
          f"operator semantics decision pending): {', '.join(sorted(offmap_unplaced))}")
_kinds = defaultdict(list)
for c in unclaimed:
    _kinds[anchors[c][2] if c in anchors else "SKY"].append(c)
for k in ("minting", "grounded", "summit", "summit-chain", "SKY"):
    if _kinds[k]:
        print(f"unclaimed/{k}: {', '.join(sorted(_kinds[k]))}")
_top = sorted(MOM.items(), key=lambda kv: -kv[1])[:10]
print("momentum (recent-git, top): " + ", ".join(f"{k[2:]}={v:.2f}" for k, v in _top))
print(f"lasso level={LV:.3f}/mmax={mmax:.3f}")
