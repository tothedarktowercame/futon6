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
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import futon6_config as config

ROOT = config.code_root()
# Optional variant arg (force|embed|springs|seed) selects an alternate mission layout from
# mission_carpet_variants.py — the projection FAMILY. No arg = the canonical carpet, unchanged.
_VARIANT = sys.argv[1] if len(sys.argv) > 1 else None
POS = json.load(open(ROOT / "futon6/data" /
                     (f"mission-carpet-pos-{_VARIANT}.json" if _VARIANT else "mission-carpet-pos.json")))
SCOPES = json.load(open(ROOT / "futon6/data/efe-scopes.json"))  # reproducible: scripts/mission_efe_scope_dump.py
CAPS = json.load(open(ROOT / "futon6/data/capability-graph.json"))
ROADS = json.load(open(ROOT / "futon6/data/mission-carpet-roads.json"))
OUT = ROOT / "futon6/data" / (f"mission-efe-field-{_VARIANT}.html" if _VARIANT else "mission-efe-field.html")
CAPABILITY_ZONES = json.load(open(ROOT / "futon3c/resources/capability_zones/live-map-pca3-v1.json"))
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
        scope_pts.append((x, y, metric, sc["det"], ccol(m), vac, verdict, m))
        hub_lines.append((cx, cy, x, y, m))

# --- metric field on a vertex grid via scatter-add ---
W = H = 3600
STEP = 40
SIGMA = 70.0
gw, gh = W // STEP + 1, H // STEP + 1
grid = [[0.0] * gw for _ in range(gh)]
rc = int(3 * SIGMA / STEP)
for x, y, mtr, _det, _col, _vac, _ver, _m in scope_pts:
    cgx, cgy = int(round(x / STEP)), int(round(y / STEP))
    for vy in range(max(0, cgy - rc), min(gh, cgy + rc + 1)):
        for vx in range(max(0, cgx - rc), min(gw, cgx + rc + 1)):
            d2 = (vx * STEP - x) ** 2 + (vy * STEP - y) ** 2
            grid[vy][vx] += mtr * math.exp(-d2 / (2 * SIGMA * SIGMA))
fmax = max(max(r) for r in grid) or 1.0
NB = 7
import efe_carpet_controls as _ctl
# Per-mission level-set band of the metric field at the hub's grid cell — the global
# control panel's "band floor" hides districts below a band (E-kimi-task-45).
BAND = {m: _ctl.hub_band(grid, fmax, NB, STEP, x, y) for x, y, m, n in hubs}
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
# --- MISSION-DOC ACTIVITY overlay (M-the-perfect-crime, 2026-09-25) ---
# Warrant: claude-12-turn-95 (Joe) — "minimal visual improvement … that will let me see that
# the Tornhill information is being considered". Relabelled per claude-12-turn-100 (reviewer
# feedback relayed by Joe): the ring counts commits to the mission's OWN DOC — activity on a
# planning document — NOT Tornhill churn, which is change-frequency in the code under study.
# Nothing links a mission to its code yet (M-the-perfect-crime plan layers 1–2), so code
# churn cannot be measured; the label now says exactly what is measured, and says on hover
# what is not. (The first version's "Tornhill churn" label was itself the overclaim class
# this mission exists to catch — recorded in the mission.)
# One unweighted count per mission, 180-day window, deduped across worktree repos by commit
# hash. Districts with zero commits in the window get a thin dashed grey ring — an explicit
# no-data state, never silent absence (the mission's own crime).
CHURN = defaultdict(int)
_seen_commits = set()  # futon* includes worktree repos (futon3c-*, futon2-*) sharing history;
                       # count each commit once, not once per worktree (M-the-perfect-crime
                       # 2026-09-24 checkpoint flagged exactly this un-checked caveat).
for repo in sorted({p.parents[2] for p in ROOT.glob("futon*/holes/missions/M-*.md")}):
    try:
        out = subprocess.run(["git", "-C", str(repo), "log", "--since=180 days ago",
                              "--pretty=format:%x01%H", "--name-only"],
                             capture_output=True, text=True, timeout=25).stdout
    except Exception:
        continue
    sha = None
    for ln in out.splitlines():
        if ln.startswith("\x01"):
            sha = ln[1:].strip() or None
        elif sha and ln.endswith(".md"):
            base = ln.rsplit("/", 1)[-1]
            if base.startswith("M-") and (sha, base) not in _seen_commits:
                _seen_commits.add((sha, base))
                CHURN["M-" + base[2:-3]] += 1
_cmax = max(CHURN.values(), default=0)
def churn_ring(x, y, m, n, attrs=""):
    c = CHURN.get("M-" + m, 0)
    r = 2.6 + 1.5 * math.sqrt(GEN.get(m, 0)) + 4.5  # just outside the hub disc
    if c == 0:  # explicit no-data: no commits touched this mission doc in the window
        return (f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{r:.1f}" fill="none" stroke="#5a6372" '
                f'stroke-width="0.7" stroke-dasharray="2,3" opacity="0.7" {attrs}>'
                f'<title>{m} · mission-doc activity: 0 commits to this mission doc in 180d '
                f'(explicit no-data ring; code churn not measurable — no mission→code link yet)</title></circle>')
    w = 0.8 + 2.6 * (math.log1p(c) / math.log1p(_cmax)) if _cmax else 0.8
    return (f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{r:.1f}" fill="none" stroke="#eab308" '
            f'stroke-width="{w:.1f}" opacity="0.85" {attrs}>'
            f'<title>{m} · mission-doc activity: {c} commits to this mission doc in 180d '
            f'(git log, thickness ∝ log count, max {_cmax}). This is activity on a planning '
            f'document, NOT Tornhill churn (change-frequency in the code under study) — '
            f'no mission→code link exists yet, M-the-perfect-crime layers 1–2.</title></circle>')

# --- CODE RING overlay: reads the file-level Tornhill report (M-the-perfect-crime) ---
# Warrant: packet 1b (E-kimi-task-44, 2026-09-26) — the pink ring now shows the Tornhill
# report looked up per mission via its code.files link, never the old per-mission git pass.
# States, never silent: solid pink = measured (files in the report), dashed pink = linked
# files but none changed in the report's window, faint dotted grey = no mission→code link,
# and when the report is absent no rings at all plus a legend line saying so.
import efe_tornhill_ring as _tring
ACTIVITY = json.load(open(ROOT / "futon6/data/mission-activity.json"))
ACT = {}
for _row in ACTIVITY["missions"]:
    ACT[_row["mission"]] = _row
    if _row["mission"].startswith("M-"):
        ACT[_row["mission"][2:]] = _row
# Global-control attributes (E-kimi-task-45): every per-mission element carries its
# district's band + status class so the panel can hide/show without a reload.
STATUS = {m: _ctl.status_class((ACT.get(m) or {}).get("status_line")) for _, _, m, _ in hubs}
def data_attrs(m):
    return f'data-m="{m}" data-band="{BAND[m]}" data-status="{STATUS[m]}"'
# Hover band (claude-12, 2026-09-26; Joe: hovering a ring gave nothing to follow). The
# rings are thin unfilled strokes, so the pointer mostly misses them. Each ring gets an
# invisible stroke-only twin, `width` units wide, carrying the same <title> and data-*
# attributes (so the carpet controls hide it with its ring). Widths keep the doc ring
# (r+4.5) and code ring (r+7.5) bands from covering each other.
_RING_RE = re.compile(r'<circle cx="([^"]+)" cy="([^"]+)" r="([^"]+)"([^>]*)>(<title>.*?</title>)</circle>', re.S)
def with_hover_band(svg, width):
    def band(mo):
        data = " ".join(re.findall(r'data-[\w-]+="[^"]*"', mo.group(4)))
        return (mo.group(0) + f'<circle cx="{mo.group(1)}" cy="{mo.group(2)}" r="{mo.group(3)}" '
                f'fill="none" stroke="#000" stroke-opacity="0" stroke-width="{width}" '
                f'pointer-events="stroke" {data}>{mo.group(5)}</circle>')
    return _RING_RE.sub(band, svg)
activity_svg = with_hover_band("".join(churn_ring(x, y, m, n, data_attrs(m)) for x, y, m, n in hubs), 3)
_TREP = _tring.load_report(_tring.REPORT_DIR)
_TCHAT = _tring.load_chat(_tring.REPORT_DIR) if _TREP else None
_TIDX = _tring.index(_TREP, _TCHAT) if _TREP else None
_TMETA = ({"filename": _TREP.get("_filename"), "generated": _TREP.get("generated"),
           "window_days": _TREP.get("window_days")} if _TREP else None)
def code_churn_ring(x, y, m, n):
    r = 2.6 + 1.5 * math.sqrt(GEN.get(m, 0)) + 7.5  # outside the doc-activity ring
    ring = _tring.mission_ring(ACT.get(m), _TIDX)
    return f'<g {data_attrs(m)}>' + _tring.ring_svg(x, y, r, m, ring, _TMETA, _hotspot_max) + "</g>"
_RINGS = {m: _tring.mission_ring(ACT.get(m), _TIDX) for _, _, m, _ in hubs}
_hotspot_max = max((r["hotspot"] for r in _RINGS.values() if r["state"] == "measured"),
                   default=0) or 1
code_churn_svg = with_hover_band("".join(code_churn_ring(x, y, m, n) for x, y, m, n in hubs), 4)
_n_measured = sum(1 for r in _RINGS.values() if r["state"] == "measured")
_n_unmeasured = sum(1 for r in _RINGS.values() if r["state"] == "unmeasured")
_n_nolink = sum(1 for r in _RINGS.values() if r["state"] == "no-link")
if _TREP:
    _code_ring_legend = (f"Measured for <b>{_n_measured}/{len(hubs)} districts</b> "
                         f"(unmeasured {_n_unmeasured} · no-link {_n_nolink}); source "
                         f"{_TMETA['filename']} generated {_TMETA['generated']} "
                         f"({_TMETA['window_days']}d window).")
else:
    _code_ring_legend = "<b>No Tornhill report — code ring not drawn</b>."

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

# faint pattern-road backdrop (attestation-weighted). Ink is RELATIVE to the
# current economy's strongest road: attestation now refreshes on a rolling
# 60-day window (refresh_pattern_attestation.sh), so absolute counts are not
# comparable across refreshes — the old absolute scale (calibrated to counts
# ~100-236) would render a quiet week as a roadless map.
roads = []
_wmax = max((w for _, _, w in ROADS), default=1) or 1
for a, b, w in ROADS:
    if a in POS and b in POS:
        x1, y1 = POS[a]; x2, y2 = POS[b]
        f = min(w, _wmax) / _wmax
        # Road ends carry bare mission ids (data-m form) so the carpet controls
        # can remove a road with either of its missions.
        roads.append(f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" '
                     f'stroke="#9a7fd0" stroke-width="{0.4+1.6*f:.1f}" opacity="{0.04+0.36*f:.2f}" '
                     f'data-road-a="{a.removeprefix("M-")}" data-road-b="{b.removeprefix("M-")}"/>')

hubline_svg = "".join(f'<line x1="{a:.0f}" y1="{b:.0f}" x2="{c:.1f}" y2="{d:.1f}" stroke="#54627f" stroke-width="0.4" opacity="0.22" {data_attrs(m)}/>'
                      for a, b, c, d, m in hub_lines)
def scope_mark(x, y, mtr, det, col, vac, verdict, attrs=""):
    if verdict is not None:  # certificate: verdict diamond, green pass / red fail
        c = "#4ade80" if verdict == "pass" else "#ef4444"
        return (f'<path d="M {x:.1f} {y-4.4:.1f} L {x+4.4:.1f} {y:.1f} L {x:.1f} {y+4.4:.1f} '
                f'L {x-4.4:.1f} {y:.1f} Z" fill="{c}" stroke="#04060c" stroke-width="0.6" opacity="0.95" {attrs}/>')
    if vac:  # vacuous scope: hollow ring — a binder with nothing bound inside
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" fill="none" stroke="{col}" stroke-width="1.0" opacity="0.85" {attrs}/>'
    return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{2.4 if det else 1.4}" '
            f'fill="{"#ffb454" if det else col}" opacity="{0.9 if det else 0.6}" {attrs}/>')
scope_svg = "".join(scope_mark(*pt[:7], data_attrs(pt[7])) for pt in scope_pts)
# HEAD hubs: colour = Salingaros class (red/green/blue/grey), size = phylogeny generativity
hub_svg = "".join(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{2.6+1.5*math.sqrt(GEN.get(m,0)):.1f}" fill="{ccol(m)}" '
                  f'stroke="#04060c" stroke-width="0.9" {data_attrs(m)}><title>{m} · {CLS.get(m,"?")} · generativity {GEN.get(m,0)} · {n} scopes · band {BAND[m]} · status {STATUS[m]}</title></circle>'
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
#capability-zones-toggle,#capability-disagreement-toggle{margin-left:10px;padding:3px 8px;border:1px solid #64748b;border-radius:5px;color:#e2e8f0;background:#172033;font:11px ui-sans-serif,system-ui,sans-serif;cursor:pointer}
#capability-zones-toggle[aria-pressed="true"],#capability-disagreement-toggle[aria-pressed="true"]{border-color:#67e8f9;color:#cffafe;background:#164e63}
#capability-zones-layer[data-hide-disagreement="true"] .capability-zone-disagreement{display:none}
#capability-zones-help{margin:2px 0 6px;color:#8b95a7;font-size:12px;max-width:1180px}
#capability-zones-help summary{cursor:pointer;color:#67e8f9}
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
.live-coord-thread{fill:none;stroke:#67e8f9;stroke-width:1.4;stroke-linecap:round;pointer-events:stroke}
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
.capability-zone-entity{stroke:#05060a;stroke-width:2.2;paint-order:stroke;vector-effect:non-scaling-stroke}
.capability-zone-mixed{fill-opacity:.22;stroke:#f8fafc;stroke-width:2.6;stroke-dasharray:4 3}
.capability-zone-disagreement{fill:none;stroke:#ffffff;stroke-width:2.4;vector-effect:non-scaling-stroke}
.capability-zone-legend text{font-family:ui-sans-serif,system-ui,sans-serif;fill:#f8fafc;font-size:12px;paint-order:stroke;stroke:#05060a;stroke-width:3px}
.capability-zone-legend-bg{fill:#07101d;fill-opacity:.92;stroke:#94a3b8;stroke-width:1.2}
@keyframes liveAgentPulse{0%,100%{opacity:.65}50%{opacity:1}}
@keyframes liveWmPulse{0%{opacity:.95;transform:scale(.82)}70%{opacity:.08;transform:scale(1.45)}100%{opacity:0;transform:scale(1.55)}}
"""

LIVE_OVERLAY_SCRIPT = """
<script>
(() => {
  const NS = "http://www.w3.org/2000/svg";
  const ENDPOINT = "http://localhost:7070/api/alpha/live-efe-map";
  const REFRESH_MS = 10000;
  const STATIC_CAPABILITY_ZONES = __CAPABILITY_ZONES_JSON__;
  const FRONTIER_Y = 3410;
  const FRONTIER_H = 175;
  const svg = document.getElementById("efe-field");
  const badge = document.getElementById("live-status");
  const zonesToggle = document.getElementById("capability-zones-toggle");
  if (!svg) return;

  function el(name, attrs = {}, text = null) {
    const node = document.createElementNS(NS, name);
    for (const [k, v] of Object.entries(attrs)) {
      if (v !== null && v !== undefined) node.setAttribute(k, String(v));
    }
    if (text !== null) node.textContent = text;
    return node;
  }

  const zoneLayer = el("g", {id: "capability-zones-layer", "data-reduction-version": "pending"});
  svg.appendChild(zoneLayer);
  const layer = el("g", {id: "live-overlay"});
  svg.appendChild(layer);
  let zonesVisible = true;

  if (zonesToggle) zonesToggle.addEventListener("click", () => {
    zonesVisible = !zonesVisible;
    zoneLayer.style.display = zonesVisible ? "" : "none";
    zonesToggle.setAttribute("aria-pressed", String(zonesVisible));
    zonesToggle.textContent = zonesVisible ? "capability zones: on" : "capability zones: off";
  });

  const disagreementToggle = document.getElementById("capability-disagreement-toggle");
  let disagreementVisible = true;
  if (disagreementToggle) disagreementToggle.addEventListener("click", () => {
    disagreementVisible = !disagreementVisible;
    zoneLayer.setAttribute("data-hide-disagreement", String(!disagreementVisible));
    disagreementToggle.setAttribute("aria-pressed", String(disagreementVisible));
    disagreementToggle.textContent = disagreementVisible ? "disagreement ×: shown" : "disagreement ×: hidden";
  });

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

  function drawCapabilityZones(data) {
    clear(zoneLayer);
    const zones = data["capability-zones"] || STATIC_CAPABILITY_ZONES;
    zoneLayer.setAttribute("data-reduction-version", zones["reduction-version"] || "unavailable");
    const items = zones.items || [];
    for (const item of items) {
      if (!Number.isFinite(Number(item.x)) || !Number.isFinite(Number(item.y))) continue;
      const g = el("g", {
        "data-capability-mission-id": item["mission-id"],
        "data-capability-zone": item.class,
        "data-capability-mixed": Boolean(item["mixed?"]),
        "data-capability-disagreement": Boolean(item["disagreement?"])
      });
      const color = item.color || ((zones.legend || []).find((z) => z.class === item.class) || {}).color || "#94a3b8";
      title(g, `${item["mission-id"]}\n3-D zone=${item.class} margin=${fmt(item.margin)}${item["mixed?"] ? " (mixed)" : ""}\nraw high-D diagnostic=${item["high-d-class"]} margin=${fmt(item["high-d-margin"])}${item["disagreement?"] ? " · DISAGREES" : ""}`);
      g.appendChild(el("circle", {cx: item.x, cy: item.y, r: 11, fill: color, opacity: item["mixed?"] ? .55 : .82,
        class: `capability-zone-entity ${item["mixed?"] ? "capability-zone-mixed" : ""}`}));
      if (item["disagreement?"]) {
        const x = Number(item.x), y = Number(item.y);
        g.appendChild(el("path", {d: `M ${x-5} ${y-5} L ${x+5} ${y+5} M ${x+5} ${y-5} L ${x-5} ${y+5}`,
          class: "capability-zone-disagreement"}));
      }
      zoneLayer.appendChild(g);
    }
    const legend = el("g", {class: "capability-zone-legend", "data-capability-legend": "true", transform: "translate(2800 55)"});
    legend.appendChild(el("rect", {x: 0, y: 0, width: 330, height: 265, rx: 8, class: "capability-zone-legend-bg"}));
    legend.appendChild(el("text", {x: 14, y: 22, style: "font-weight:800"}, `capability zones · ${zones["reduction-version"] || "unavailable"}`));
    for (const [i, row] of (zones.legend || []).entries()) {
      const col = i >= 7 ? 1 : 0, line = i % 7;
      const x = 14 + col * 158, y = 47 + line * 27;
      legend.appendChild(el("circle", {cx: x + 6, cy: y - 4, r: 6, fill: row.color, stroke: "#fff", "stroke-width": .7}));
      legend.appendChild(el("text", {x: x + 18, y}, `${row.class} · ${row["mission-count"] || 0}`));
    }
    legend.appendChild(el("text", {x: 14, y: 244}, "dashed/dim = mixed · × = high-D disagreement"));
    zoneLayer.appendChild(legend);
    const counts = document.getElementById("cz-counts");
    if (counts) {
      const shown = items.filter((i) => Number.isFinite(Number(i.x)) && Number.isFinite(Number(i.y)));
      const disagree = shown.filter((i) => i["disagreement?"]).length;
      const mixed = shown.filter((i) => i["mixed?"]).length;
      counts.textContent = `Currently rendered: ${shown.length} missions, ${disagree} disagreements, ${mixed} mixed.`;
    }
    zoneLayer.setAttribute("data-hide-disagreement", String(!disagreementVisible));
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
    // Co-located sessions (same mission → same coords) would overprint their
    // glyphs and labels; fan the 2nd, 3rd… out in a small ring around the hub.
    const crowd = new Map();
    const positions = new Map();
    // Warrant: claude-12-turn-97 (Joe) — "the live annotations should be up to date with
    // the state of Agency". Applies client-side the same witnessed-presence predicate the
    // server already has at futon3c HEAD (5b14a6f6, turn-c12-saucers F5): an agent is drawn
    // only if its registry status is alive this server epoch (invoking/idle). The running
    // :7070 JVM (started 2026-09-21) predates that commit and still emits "restored"
    // durable-lineage agents with missions — the phantom flying sources Joe saw. Withheld
    // agents are counted and surfaced in the badge, never silently dropped (the page's own
    // no-silent-absence rule). Remove this filter once :7070 runs >= 5b14a6f6.
    let withheldStale = 0;
    const staleIds = [];
    for (const agent of ((data.agents && data.agents.items) || [])) {
      if (!agent["mission-id"] || !validPlacement(agent.placement)) continue;
      const st = String(agent.status || "").toLowerCase();
      if (st !== "invoking" && st !== "idle") {
        withheldStale += 1;
        staleIds.push(`${agent["agent-id"]}(${agent.status})`);
        continue;
      }
      const spot = `${agent.placement.x},${agent.placement.y}`;
      const k = crowd.get(spot) || 0;
      crowd.set(spot, k + 1);
      const p = k === 0 ? agent.placement : Object.assign({}, agent.placement, {
        x: Number(agent.placement.x) + 14 * k,
        y: Number(agent.placement.y) + 30 * k
      });
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
      positions.set(String(agent["agent-id"]), {x: Number(p.x), y: Number(p.y)});
    }
    if (withheldStale > 0) {
      const g = el("g", {"data-live-kind": "stale-withheld"});
      g.appendChild(el("rect", {x: 18, y: 92, width: 320, height: 34, rx: 6, fill: "#1a2a12", stroke: "#a3e635", opacity: 0.92}));
      g.appendChild(el("text", {x: 32, y: 115, class: "live-offline-badge", style: "fill:#d9f99d"},
        `${withheldStale} stale agent annotation(s) withheld — not alive in Agency this epoch`));
      title(g, `Withheld (registry status not invoking/idle — phantoms from before futon3c 5b14a6f6 reaches the live JVM):\\n${staleIds.join("\\n")}`);
      layer.appendChild(g);
    }
    return positions;
  }

  function coordAge(edge) {
    return Number(edge["age-ms"] ?? edge.age_ms ?? 0);
  }

  function drawCoordination(data, agentPositions) {
    const edges = (data.coordination && data.coordination.items) || [];
    if (!edges.length || !agentPositions || !agentPositions.size) return;
    const g = el("g", {"data-layer": "coordination"});
    for (const edge of edges) {
      const from = String(edge.from || "");
      const to = String(edge.to || "");
      const a = agentPositions.get(from);
      const b = agentPositions.get(to);
      if (!a || !b) continue;
      const ageMs = coordAge(edge);
      const fade = Math.max(0.1, 1 - ageMs / 3600000);
      const ageMin = Math.max(0, ageMs / 60000).toFixed(1);
      const midX = (a.x + b.x) / 2;
      const midY = (a.y + b.y) / 2 - Math.min(42, Math.hypot(b.x - a.x, b.y - a.y) / 8);
      const path = el("path", {
        d: `M ${a.x} ${a.y} Q ${midX} ${midY} ${b.x} ${b.y}`,
        class: "live-coord-thread",
        opacity: fade,
        "data-live-kind": "coord",
        "data-from": from,
        "data-to": to
      });
      title(path, `${from} → ${to} · ${edge.kind || "coord"} · ${ageMin} age-min`);
      g.appendChild(path);
    }
    if (g.childNodes.length) layer.appendChild(g);
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
    drawCapabilityZones(data);
    clear(layer);
    drawFrontierBand(data);
    drawWarMachine(data);
    const agentPositions = drawAgents(data);
    drawCoordination(data, agentPositions);
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

  drawCapabilityZones({"capability-zones": STATIC_CAPABILITY_ZONES});
  refresh();
  window.setInterval(refresh, REFRESH_MS);
})();
</script>
"""

# --- GLOBAL CONTROL PANEL (M-the-perfect-crime, E-kimi-task-45, 2026-09-26) ---
# Joe (2026-09-26): "a global controller for the carpet that would turn off N of the
# level sets … quite a lot of missions are in level set 0 which may be basically useless
# for me now". Band floor + status filter hide districts (display:none, no reload);
# layer toggles hide the doc ring / code ring / scope dots / momentum lasso. Defaults
# show everything — the page is unchanged until a control is touched.
from collections import Counter as _Counter
_BAND_COUNTS = _Counter(BAND.values())
_STATUS_COUNTS = _Counter(STATUS.values())
CONTROLS_CSS = """
#efe-controls{position:fixed;left:10px;top:120px;z-index:20;width:320px;max-height:calc(100vh - 20px);overflow:auto;padding:8px 10px;border:1px solid #334155;border-radius:8px;background:rgba(7,12,22,.93);color:#cdd3df;font:12px ui-sans-serif,system-ui,sans-serif}
#efe-controls h2{margin:0 0 6px;font-size:12px;color:#e2e8f0}
#efe-controls label{display:block;margin:3px 0;cursor:pointer}
#efe-controls .ctl-section{margin:7px 0 3px;color:#8b95a7;font-weight:700;text-transform:uppercase;font-size:10px;letter-spacing:.06em}
#efe-controls input[type=range]{width:150px;vertical-align:middle}
#efe-controls .ctl-count{color:#8b95a7}
#efe-controls .ctl-legend{margin:7px 0 0;color:#8b95a7;font-size:11px;line-height:1.35}
#ctl-band-floor-label{color:#e2e8f0}
"""
_panel_status = "".join(
    f'<label><input type="checkbox" data-ctl-status="{s}" checked> {s} '
    f'<span class="ctl-count">({_STATUS_COUNTS.get(s, 0)})</span></label>'
    for s in ("done", "open", "unknown"))
_panel_html = (
    '<div id="efe-controls">'
    '<h2>carpet controls</h2>'
    '<div class="ctl-section">band floor</div>'
    f'<label><input type="range" id="ctl-band-floor" min="0" max="{NB - 1}" step="1" value="0"> '
    f'<span id="ctl-band-floor-label">hide missions below band 0 — showing {len(hubs)} / hiding 0</span></label>'
    f'<div class="ctl-count">districts per band: '
    + " · ".join(f"b{b} {_BAND_COUNTS.get(b, 0)}" for b in range(NB)) + '</div>'
    '<div class="ctl-section">status</div>' + _panel_status +
    '<label><input type="checkbox" id="ctl-scopes-only"> hide scopes only</label>' +
    '<div class="ctl-section">layers</div>'
    '<label><input type="checkbox" data-ctl-layer="#layer-doc-ring" checked> mission-doc ring</label>'
    '<label><input type="checkbox" data-ctl-layer="#layer-code-ring" checked> code ring</label>'
    '<label><input type="checkbox" data-ctl-layer="#layer-scope-dots" checked> scope dots</label>'
    '<label><input type="checkbox" data-ctl-layer=".layer-lasso" checked> momentum lasso</label>'
    '<p class="ctl-legend">band = density of scopes around the mission (its scopes weighted '
    'by determined / frontier / vacuous, blurred with its neighbours) — not a judgement of value. '
    'A mission removed by band or status is gone: hub, rings, scopes, capability-zone mark, '
    'live markers and its pattern roads. Tick “hide scopes only” to keep the mission and drop just its scopes.</p>'
    '</div>')
# --- Click-for-details (Joe 2026-09-27: hover titles are not enough; a click should
# explain the mission's marks). Everything the panel says about a mission comes from
# the same inputs its marks were drawn from; ring/zone/live text is read from the
# drawn elements' own <title>s at click time, so the panel cannot drift from them.
_roads_of = defaultdict(list)
for _a, _b, _w in ROADS:
    if _a in POS and _b in POS:
        _roads_of[_a.removeprefix("M-")].append((_b.removeprefix("M-"), _w))
        _roads_of[_b.removeprefix("M-")].append((_a.removeprefix("M-"), _w))
def _mission_info(m, n):
    scs = by_m.get(m, [])
    act = ACT.get(m) or ACT.get("M-" + m) or {}
    return {
        "cls": CLS.get(m, "?"), "clsColor": ccol(m), "gen": GEN.get(m, 0), "scopes": n,
        "band": BAND[m], "status": STATUS[m], "statusLine": act.get("status_line"), "doc": act.get("doc"),
        "binders": dict(_Counter(sc["binder"] for sc in scs).most_common()),
        "holes": sum(1 for sc in scs if sc["det"]), "vacuous": sum(1 for sc in scs if sc.get("vacuous")),
        "certPass": sum(1 for sc in scs if sc["binder"] == "certificate" and sc.get("verdict") == "pass"),
        "certFail": sum(1 for sc in scs if sc["binder"] == "certificate" and sc.get("verdict") not in (None, "pass")),
        "momentum": round(MOM.get("M-" + m, 0.0), 2),
        "roads": sorted(_roads_of.get(m, []), key=lambda r: -r[1])[:25], "roadCount": len(_roads_of.get(m, [])),
    }
MISSION_INFO = {m: _mission_info(m, n) for _, _, m, n in hubs}
DETAILS_CSS = """
#efe-details{position:fixed;right:12px;top:12px;bottom:12px;z-index:30;width:min(460px,92vw);overflow:auto;padding:14px 16px;border:1px solid #475569;border-radius:10px;background:rgba(7,12,22,.97);color:#dde3ee;font:16px/1.45 ui-sans-serif,system-ui,sans-serif}
#efe-details[hidden]{display:none}
#efe-details h2{margin:0 26px 4px 0;font-size:20px;color:#f8fafc;overflow-wrap:anywhere}
#efe-details h3{margin:14px 0 4px;font-size:15px;color:#e2e8f0;text-transform:uppercase;letter-spacing:.05em}
#efe-details p,#efe-details li{margin:3px 0;color:#cbd5e1;font-size:16px}
#efe-details ul{margin:2px 0;padding-left:20px}
#efe-details .why{color:#94a3b8;font-size:14px}
#efe-details .sw{display:inline-block;width:.9em;height:.9em;border-radius:50%;vertical-align:-.1em;margin-right:4px}
#efe-details button.close{position:absolute;right:10px;top:8px;font-size:20px;background:none;border:0;color:#cbd5e1;cursor:pointer}
#efe-details button.go{background:none;border:0;padding:0;color:#93c5fd;cursor:pointer;font:inherit;text-decoration:underline}
#efe-field [data-m],#efe-field [data-capability-mission-id]{cursor:pointer}
"""
DETAILS_SCRIPT = """
<div id="efe-details" hidden role="dialog" aria-label="Mission details"></div>
<script>
(() => {
  const INFO = __MISSION_INFO_JSON__;
  const svg = document.getElementById("efe-field");
  const box = document.getElementById("efe-details");
  if (!svg || !box) return;
  const NS = "http://www.w3.org/2000/svg";
  const bare = (v) => (v && v.startsWith("M-") && !INFO[v] ? v.slice(2) : v);
  const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
  const titleOf = (e) => { const t = e && e.querySelector(":scope > title, title"); return t ? t.textContent.trim() : ""; };
  // Wording follows mission_wholeness.py (Salingaros L = T·H over the scope tree).
  const CLASS_WHY = {alive: "alive — well differentiated (branches at several depths) and harmonious: organised complexity", mess: "mess — well differentiated but low harmony: disorganised complexity, wants a centring pass", pipeline: "pipeline — enough centres but flat, branching only at the root: a line of phases", stub: "stub — few centres yet: wants developing"};
  let marker = null;
  function missionOf(target) {
    const e = target.closest("[data-m],[data-capability-mission-id],[data-mission-id]");
    if (!e) return null;
    const m = bare(e.getAttribute("data-m") || e.getAttribute("data-capability-mission-id") || e.getAttribute("data-mission-id"));
    return INFO[m] ? m : null;
  }
  // Describe only the rings actually drawn for this mission, by their drawn
  // stroke (hover twins have stroke-opacity 0 and are skipped), with a small
  // swatch in the same stroke so the text matches what is on the map.
  const RING_KIND = {"#eab308": "Yellow ring — mission-doc activity: commits to the planning doc in 180 days (thickness ∝ log count)", "#5a6372": "Thin dashed grey ring — mission-doc activity: no commits to the planning doc in 180 days", "#3a4252": "Faint dotted grey ring — no mission→code link yet, so code churn cannot be shown"};
  function ring(els, layer) {
    const c = els.flatMap((e) => (e.tagName === "circle" ? [e] : Array.from(e.querySelectorAll("circle")))).find((e) => e.getAttribute("stroke-opacity") !== "0" && e.getAttribute("stroke"));
    if (!c) return null;
    const stroke = c.getAttribute("stroke"), dash = c.getAttribute("stroke-dasharray") || "";
    const kind = stroke === "#ec4899" ? (dash ? "Dashed pink ring — code linked, but none of its files changed in the Tornhill report's window" : "Pink ring — code churn from the Tornhill report (thickness ∝ log summed hotspot)") : (RING_KIND[stroke] || "Ring");
    const sw = '<svg width="22" height="22" style="vertical-align:-5px;margin-right:6px"><circle cx="11" cy="11" r="8" fill="none" stroke="' + stroke + '" stroke-width="' + Math.max(1.5, Number(c.getAttribute("stroke-width")) || 1) + '"' + (dash ? ' stroke-dasharray="' + dash + '"' : "") + "/></svg>";
    const off = document.getElementById(layer)?.style.display === "none" ? ' <span class="why">(layer switched off in the controls)</span>' : "";
    return "<p>" + sw + "<b>" + esc(kind) + "</b>" + off + "</p><p class=why>" + esc(titleOf(c) || titleOf(c.parentNode)) + "</p>";
  }
  function hubOf(m) { return Array.from(svg.querySelectorAll("circle[data-m]")).find((c) => c.getAttribute("data-m") === m && /generativity/.test(titleOf(c))); }
  function show(m) {
    const d = INFO[m];
    const q = (sel) => Array.from(svg.querySelectorAll(sel)).filter((e) => bare(e.getAttribute("data-m") || e.getAttribute("data-capability-mission-id") || e.getAttribute("data-mission-id")) === m);
    const rings = [ring(q("#layer-doc-ring circle[data-m]"), "layer-doc-ring"), ring(q("#layer-code-ring [data-m]"), "layer-code-ring")].filter(Boolean);
    const zone = q("[data-capability-mission-id]")[0];
    const live = q("#live-overlay [data-mission-id]").map(titleOf).filter(Boolean);
    const h = [];
    h.push('<button class="close" aria-label="Close">×</button><h2>M-' + esc(m) + "</h2>");
    if (d.statusLine) h.push("<p><b>Status line:</b> " + esc(d.statusLine) + "</p>");
    if (d.doc) h.push('<p class="why">' + esc(d.doc) + "</p>");
    h.push("<p><b>Filter status:</b> " + esc(d.status) + ' <span class="why">(done only when the status line opens with a done word; no status line = unknown)</span></p>');
    h.push("<p><b>Band:</b> " + d.band + ' of 0–6 <span class="why">— density of scopes around the mission, weighted determined / frontier / vacuous and blurred with its neighbours; not a judgement of value</span></p>');
    h.push("<h3>Hub</h3><p><span class=sw style=background:" + d.clsColor + "></span><b>Colour</b> = Salingaros class: " + esc(CLASS_WHY[d.cls] || d.cls) + "</p>");
    h.push("<p><b>Size</b> = generativity " + d.gen + ' <span class="why">(citation backlinks: how many other missions cite this one; citations are not drawn as lines)</span></p>');
    h.push("<h3>Scopes · " + d.scopes + "</h3><p class=why>The small marks spiralling round the hub, one per scope, ordered by binder kind from the centre out.</p><ul>");
    if (d.holes) h.push("<li>" + d.holes + ' <span style="color:#ffb454">● orange</span> = open :detached holes (high ground: work named but not attached)</li>');
    if (d.vacuous) h.push("<li>" + d.vacuous + " ○ hollow = vacuous scopes (a binder with no named entities inside: suspect terrain)</li>");
    if (d.certPass || d.certFail) h.push("<li>" + d.certPass + ' <span style="color:#4ade80">◆</span> certificate PASS (verified ground) · ' + d.certFail + ' <span style="color:#ef4444">◆</span> FAIL</li>');
    h.push("<li>" + (d.holes || d.vacuous || d.certPass || d.certFail ? "the rest are" : "all are") + " small dots in the hub's class colour</li></ul>");
    h.push("<p class=why>By binder: " + Object.entries(d.binders).map(([k, v]) => esc(k) + " " + v).join(" · ") + "</p>");
    if (rings.length) h.push("<h3>Rings</h3>" + rings.join(""));
    if (zone) {
      h.push("<h3>Capability zone</h3><p><b>" + esc(zone.getAttribute("data-capability-zone")) + "</b>" + (zone.getAttribute("data-capability-mixed") === "true" ? " · <b>mixed</b> (dashed/dim: the two nearest zone seeds are almost equally close, so the call is ambiguous)" : "") + "</p>");
      if (zone.getAttribute("data-capability-disagreement") === "true") h.push("<p>× <b>disagreement</b>: the raw high-dimensional reading names a different class than this zone. A diagnostic of boundary distortion, not proof the zone is wrong.</p>");
      const zt = titleOf(zone); if (zt) h.push('<p class="why">' + esc(zt) + "</p>");
    }
    if (live.length) h.push("<h3>Live now</h3><ul>" + live.map((t) => "<li>" + esc(t) + "</li>").join("") + "</ul>");
    if (d.momentum > 0) h.push("<h3>Momentum</h3><p>" + d.momentum + ' <span class="why">— recent git activity on the mission doc (10-day decay); high momentum puts it inside the amber dashed lasso, your territory</span></p>');
    if (d.roadCount) h.push("<h3>Pattern roads · " + d.roadCount + '</h3><p class="why">Purple lines: missions that apply the same library pattern; ink ∝ how often the strongest shared pattern was enacted in logged turns over the last 60 days.</p>');
    if (d.roads.length) h.push("<ul>" + d.roads.map(([o, w]) => '<li><button class="go" data-go="' + esc(o) + '">M-' + esc(o) + "</button> · " + w + "</li>").join("") + "</ul>" + (d.roadCount > d.roads.length ? "<p class=why>… and " + (d.roadCount - d.roads.length) + " more</p>" : ""));
    box.innerHTML = h.join("");
    box.hidden = false;
    box.scrollTop = 0;
    const hub = hubOf(m);
    if (marker) marker.remove();
    if (hub) {
      marker = document.createElementNS(NS, "circle");
      for (const [k, v] of [["cx", hub.getAttribute("cx")], ["cy", hub.getAttribute("cy")], ["r", Number(hub.getAttribute("r")) + 16], ["fill", "none"], ["stroke", "#f8fafc"], ["stroke-width", "3"], ["stroke-dasharray", "6,4"], ["pointer-events", "none"]]) marker.setAttribute(k, v);
      svg.appendChild(marker);
    }
    return hub;
  }
  function close() { box.hidden = true; if (marker) { marker.remove(); marker = null; } }
  svg.addEventListener("click", (ev) => { const m = missionOf(ev.target); if (m) show(m); });
  box.addEventListener("click", (ev) => {
    if (ev.target.closest("button.close")) { close(); return; }
    const go = ev.target.closest("button.go");
    if (go && INFO[go.dataset.go]) { const hub = show(go.dataset.go); if (hub) hub.scrollIntoView({block: "center", inline: "center", behavior: "smooth"}); }
  });
  document.addEventListener("keydown", (ev) => { if (ev.key === "Escape") close(); });
  window.efeDetails = {show, close};
})();
</script>
"""
# Inline JS: every string literal is single-line (a literal newline inside a JS string
# broke the page on 2026-09-25). Plain string (not f-string); node --check the extract.
CONTROLS_SCRIPT = """
<script>
(() => {
  const svg = document.getElementById("efe-field");
  const panel = document.getElementById("efe-controls");
  if (!svg || !panel) return;
  const header = document.querySelector("header");
  if (header) panel.style.top = (header.offsetHeight + 8) + "px";
  const els = Array.from(svg.querySelectorAll("[data-m]"));
  const missions = new Map();
  for (const e of els) {
    const m = e.getAttribute("data-m");
    if (!missions.has(m)) {
      missions.set(m, {band: Number(e.getAttribute("data-band")), status: e.getAttribute("data-status")});
    }
  }
  const floor = document.getElementById("ctl-band-floor");
  const floorLabel = document.getElementById("ctl-band-floor-label");
  const statusBoxes = Array.from(panel.querySelectorAll("input[data-ctl-status]"));
  const layerBoxes = Array.from(panel.querySelectorAll("input[data-ctl-layer]"));
  const scopesOnly = document.getElementById("ctl-scopes-only");
  // Hiding is a generated stylesheet keyed on mission ids, not per-element
  // display, so layers the live overlay draws or redraws later (capability
  // zones, agent markers) are hidden too. Ids appear bare (data-m, roads) or
  // M-prefixed (capability zones, live overlay); both forms are matched.
  const hideStyle = document.createElement("style");
  document.head.appendChild(hideStyle);
  function selectors(m, only) {
    const q = (attr, v) => '[' + attr + '="' + CSS.escape(v) + '"]';
    if (only) return ["#layer-scope-dots " + q("data-m", m), "line" + q("data-m", m)];
    const out = [q("data-m", m), q("data-road-a", m), q("data-road-b", m)];
    for (const v of [m, "M-" + m]) out.push(q("data-capability-mission-id", v), q("data-mission-id", v));
    return out;
  }
  function apply() {
    const f = Number(floor.value);
    const hiddenStatus = new Set(statusBoxes.filter((b) => !b.checked).map((b) => b.getAttribute("data-ctl-status")));
    const gone = [];
    for (const [m, info] of missions) if (info.band < f || hiddenStatus.has(info.status)) gone.push(m);
    const sel = gone.flatMap((m) => selectors(m, scopesOnly.checked));
    hideStyle.textContent = sel.length ? sel.join(",") + "{display:none !important}" : "";
    floorLabel.textContent = "hide missions below band " + f + " — showing " + (missions.size - gone.length) + " / hiding " + gone.length;
  }
  function applyLayers() {
    for (const b of layerBoxes) {
      const sel = b.getAttribute("data-ctl-layer");
      for (const node of document.querySelectorAll(sel)) {
        node.style.display = b.checked ? "" : "none";
      }
    }
  }
  floor.addEventListener("input", apply);
  for (const b of statusBoxes) b.addEventListener("change", apply);
  scopesOnly.addEventListener("change", apply);
  for (const b of layerBoxes) b.addEventListener("change", applyLayers);
})();
</script>
"""

doc = f"""<!doctype html><meta charset=utf-8><title>Futon City — per-scope metric field</title>
<style>body{{margin:0;background:#05060a;color:#cdd3df;font:13px sans-serif}}header{{padding:11px 20px}}
h1{{font-size:16px;margin:0 0 4px}}p{{margin:0;color:#8b95a7;font-size:12px;max-width:1180px}}
text{{cursor:default}}{LIVE_OVERLAY_STYLE}{CONTROLS_CSS}{DETAILS_CSS}</style>
{_panel_html}
<header><h1>Futon City — per-step-cost <b>METRIC field</b> g(s), per-scope ({len(scope_pts)} scopes / {len(hubs)} districts) · 🌟{len(claimed)} claimed · ⭐{len(unclaimed)} unclaimed <span id="live-status">live layer loading</span><button id="capability-zones-toggle" type="button" aria-pressed="true">capability zones: on</button><button id="capability-disagreement-toggle" type="button" aria-pressed="true">disagreement ×: shown</button></h1>
<details id="capability-zones-help"><summary>capability zones — what am I looking at?</summary>
<p>Every mission is coloured by its <b>capability zone</b>: the action-class whose seed it sits
nearest in a 3-D PCA reduction (<code>pca3-v1</code>) of the BGE embedding space. The zone is
computed in 3-D and only <i>displayed</i> here — nothing is decided on this 2-D picture, so what
you accept is the same object the War Machine's preferences will read.
<b>dashed/dim = mixed</b>: the two nearest seeds are so close (thinnest global decile of margins)
that the call is honestly ambiguous.
<b>× = disagreement</b>: the raw 1024-dimensional cosine reading — kept only as a
boundary-distortion <i>diagnostic</i> — names a <i>different</i> class than the operative 3-D
zone. A × does not mean the zone is wrong; thin high-D margins flip easily under projection.
But where ×s <i>cluster inside one zone</i>, treat that zone's boundary as distortion-suspect
and record a complaint: e.g. the <b>no-op</b> zone currently holds 51 missions of which 25 are
disagreements (raw high-D mostly read them as <code>close</code> or <code>apply-cascade</code>) —
exactly the kind of boundary question the walk exists to catch. <span id="cz-counts"></span></p>
</details>
<p><b>This is the metric (terrain), NOT the EFE</b> (EFE = G(π) = the geodesic over it, drawn later as policy
streamlines). Each mission is a DISTRICT — scopes spiral around the HEAD hub, <b>coloured by Salingaros class</b>
(<span style="color:#3a9a4a">green=alive</span> · <span style="color:#c0392b">red=mess</span> ·
<span style="color:#3a7ad0">blue=pipeline</span> · grey=stub) and <b>sized by generativity</b> — hub SIZE is
the citation-backlink count; citations are <i>not</i> drawn as lines.
Orange points = open <b>:detached</b> holes (high ground); smooth level sets = topography.
<b><span style="color:#9a7fd0">Purple lines = shared-PATTERN roads</span></b> (the two missions apply the same
library pattern), <b>ink ∝ turn-attestation</b> of the strongest shared pattern over a <b>rolling 60-day
window</b>, scaled to the current strongest road — bold = heavily <i>enacted</i> in recent logged turns,
near-invisible = shared but not recently retrieved. A big hub with a faint fan is
therefore "much-cited, patterns not yet exercised"; a bold fan is enacted structure. <b>Anatomy marks (2026-06-12)</b>: hollow rings = <b>vacuous scopes</b> (a binder with no named
entities inside — suspect terrain, +cost); <span style="color:#4ade80">◆ green diamond = certificate
PASS</span> (verified ground, −cost) · <span style="color:#ef4444">◆ red = FAIL</span> (+cost); verify-gates
count as frontier. <b>★ filled = claimed</b> capability (at its minting mission) · <b>☆ empty gold = unclaimed but
anchored</b> (tethered to its foothold — an in-flight minting mission, a grounding mission, or its claimed substrate;
hover for which; co-anchored stars stack as ladder rungs). <b>The sky holds only what is OFF-MAP</b>:
<b><span style="color:#ff8a6a">☆ red = unclaimed, no foothold anywhere</span></b> (a registered goal with no path
built) · <b><span style="color:#c0392b">★ red-rimmed filled = claimed but off-map</span></b> (its minting missions
have no district on the carpet — operator semantics decision pending).
<b><span style="color:#67e8f9">cyan threads = recent bells between placed sessions</span></b>, fading with age.
<b><span style="color:#ffb43c">amber dashed lasso = YOUR
territory</span></b> (missions worked in git's last ~3 weeks — the momentum/exploit baseline; inside = the WM
confirms, outside = it breaks trend). <b><span style="color:#d8b066">⬡ = dark matter</span></b> (a lasso loop with
momentum but no substrate-2 district — a mission worked but not yet ingested). <b><span style="color:#eab308">yellow ring = mission-doc activity</span></b> (M-the-perfect-crime):
thickness ∝ log of commits touching the mission's <b>own planning doc</b> in the last <b>180 days</b>
(git log); <b><span style="color:#5a6372">thin dashed grey ring = zero commits in window</span></b> —
an explicit no-data mark. This is <b>not Tornhill churn</b>: churn is change-frequency in the code
under study, and nothing links a mission to its code yet (the mission's plan layers 1–2), so code
churn cannot be shown here and the ring does not claim it.
<b><span style="color:#ec4899">pink ring = CODE ring (Tornhill report)</span></b>:
each mission's <code>code.files</code> link looked up in the file-level Tornhill report
(thickness ∝ log of summed hotspot). Hover gives summed <b>revs</b> and <b>hotspot</b>, the
top files with their trend, seat count when chat attribution exists, and the report's
provenance. <b><span style="color:#ec4899">dashed pink</span> = linked files, none changed in
the report's window · <b><span style="color:#3a4252">faint dotted grey</span> = no mission→code
link yet</b> (coverage gap, stated not hidden). {_code_ring_legend} <b>Live overlay:</b> WM attention and agent telemetry on the EFE landscape — agents are drawn only when their Agency registry status is alive this server epoch (invoking/idle); anything merely restored from durable lineage is withheld and counted in a notice, per claude-12-turn-97 (the annotation layer must track the state of Agency even while the layout lags). <b>Hover any star, hub, or live marker for its story.</b></p></header>
<svg id="efe-field" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<rect x="0" y="0" width="{W}" height="{SKY_H}" fill="#0c0f18"/>
<g>{''.join(fill)}</g><g class="layer-lasso">{''.join(lasso_fill)}</g><g>{''.join(roads)}</g><g>{''.join(contour)}</g>
<g>{hubline_svg}</g><g id="layer-scope-dots">{scope_svg}</g><g id="layer-doc-ring">{activity_svg}</g><g id="layer-code-ring">{code_churn_svg}</g><g>{hub_svg}</g>
<g class="layer-lasso">{lasso}</g><g>{''.join(ghosts)}</g>
<g>{''.join(claimed)}</g><g>{''.join(summit_svg)}</g><g>{''.join(sky)}</g>
<g>{''.join(marks)}</g></svg>{LIVE_OVERLAY_SCRIPT.replace("__CAPABILITY_ZONES_JSON__", json.dumps(CAPABILITY_ZONES, separators=(",", ":"))) }{CONTROLS_SCRIPT}{DETAILS_SCRIPT.replace("__MISSION_INFO_JSON__", json.dumps(MISSION_INFO, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"))}"""
# Readable type (Joe 2026-09-27: too small to read). One scale for every size on
# the page — header, controls, SVG labels, live-overlay classes — so the relative
# hierarchy is kept; nothing smaller than 15px. The details panel is already sized.
def _bigger(n): return str(max(15, round(float(n) * 1.35)))
_detail_at = doc.index('<div id="efe-details"')
_style_end = doc.index("</style>")
doc = (re.sub(r'(font-size:|font:)(\d+(?:\.\d+)?)px', lambda mo: mo.group(1) + _bigger(mo.group(2)) + "px", doc[:_style_end].replace(DETAILS_CSS, "\x00DETAILS\x00"))
       .replace("\x00DETAILS\x00", DETAILS_CSS)
       + re.sub(r'font-size="(\d+(?:\.\d+)?)"', lambda mo: 'font-size="' + _bigger(mo.group(1)) + '"', doc[_style_end:_detail_at])
       + doc[_detail_at:])
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
