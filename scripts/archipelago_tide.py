#!/usr/bin/env python3
"""archipelago_tide.py — Prototype A (inundation) page variant.

Reads the baseline page data/mission-efe-field-embed.html and writes
data/archipelago-A-tide.html with a tide (water-level) control added. The
tide is a SEPARATE scalar field from the terrain: terrain height is scope
density; the tide field counts scopes by WORK DATE (mission-dating.json,
git blame of anchor passages). Brief §3/§10; semantics doc §2/§3/§8.8/§9.

Field: L_c(x) = sum over scopes with dated >= c of exp(-d^2/2 sigma^2),
sigma=70, unit weight, on the baseline's 40-unit vertex grid. Undated
scopes always count (never submerged, and listed in the readout). Land
where L_c >= theta, theta ONE absolute constant for all cutoffs, chosen so
the lowest-tide land best matches the baseline's drawn terrain cells
(grid/fmax >= 0.03, recomputed from data/efe-scopes.json). Islands are
4-connected components of land cells.

Scope dots are placed with the baseline's golden-angle spiral (same R,
same binder ORDER, tie-break by scope id), but their identities come from
mission-dating.json, so positions differ slightly from the baseline's
dots; the page legend says so.

The page only displays precomputed results: per week, run-length water
masks and marching-squares shoreline segments, swapped by the slider. No
animation loop.
"""
import base64
import json
import math
import re
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import futon6_config as config

ROOT = config.ROOT
BASELINE_HTML = ROOT / "data" / "mission-efe-field-embed.html"
POS_EMBED = ROOT / "data" / "mission-carpet-pos-embed.json"
DATING = ROOT / "data" / "mission-dating.json"
EFE_SCOPES = ROOT / "data" / "efe-scopes.json"
OUT = ROOT / "data" / "archipelago-A-tide.html"

W = H = 3600
STEP = 40
SIGMA = 70.0
GW, GH = W // STEP + 1, H // STEP + 1  # vertices; cells are 0..GW-2 x 0..GH-2
GOLD = math.pi * (3 - math.sqrt(5))
# Binder spiral order, copied from mission_efe_field.py (lines 88-91).
ORDER = {"eightfold-phase": 0, "loose-section": 1, "plain-argument": 1,
         "mission-scope-in": 2, "mission-scope-out": 2, "map-item": 3,
         "source-material": 4, "relates-to": 5, "capability-scope": 6,
         "pattern": 7, "psr": 8, "pur": 8, "verify-gate": 9, "certificate": 10}
FRONTIER = {"capability-scope", "pattern", "psr", "pur", "verify-gate"}

WATER_FILL = "#020409"  # darker than the lowest terrain band, so islands read as land
WATER_OPACITY = 0.82
SHORE_STROKE = "#7fa7c9"
SHORE_WIDTH = 1.2

# Tide level: a cell of the low-tide land stays land at cutoff c while the
# dated-or-undated scope density there is at least TAU, about one scope
# within 80 units (a single scope peaks near 0.9 at its nearest vertex).
# Absolute, one constant for all cutoffs. It is NOT theta: theta decides
# where the map has land at all (matching the baseline terrain); TAU decides
# whether that land has recent work. Using theta for both made small
# missions water at the lowest tide, i.e. size read as age.
TAU = 0.5

LEGEND = ("Water = no dated work on or after the cutoff nearby. "
          "Not failure, not irrelevance. Terrain height is scope density, "
          "not age.")


# ---------------------------------------------------------------- geometry

def fan_out(pos):
    """The baseline's coincident-point fan-out (mission_efe_field.py):
    missions sharing a point are spread on a 90-unit circle, sorted."""
    pos = {k: list(v) for k, v in pos.items()}
    at = defaultdict(list)
    for k, xy in pos.items():
        at[tuple(xy)].append(k)
    for (x, y), ks in at.items():
        if len(ks) > 1:
            for i, k in enumerate(sorted(ks)):
                a = 2 * math.pi * i / len(ks) - math.pi / 2
                pos[k] = [x + 90 * math.cos(a), y + 90 * math.sin(a)]
    return pos


def spiral(n):
    """Golden-angle spiral offsets for n scopes round a hub (baseline)."""
    R = 16 + 3.6 * math.sqrt(n)
    out = []
    for i in range(n):
        ang = i * GOLD
        rad = R * math.sqrt((i + 0.5) / n)
        out.append((rad * math.cos(ang), rad * math.sin(ang)))
    return out


def dating_points(dating_scopes, pos):
    """Tide scope points from mission-dating.json: sorted by (binder ORDER,
    scope id) per mission, placed on the baseline spiral. Returns dicts."""
    by_m = defaultdict(list)
    for sid, s in dating_scopes.items():
        by_m[s["mission"]].append(s | {"id": sid})
    pts = []
    for m, scs in by_m.items():
        if m not in pos:
            continue
        cx, cy = pos[m]
        scs.sort(key=lambda s: (ORDER.get(s["binder"], 9), s["id"]))
        for (dx, dy), s in zip(spiral(len(scs)), scs):
            pts.append({"x": cx + dx, "y": cy + dy, "dated": s["dated"],
                        "id": s["id"], "mission": m, "binder": s["binder"]})
    return pts


def terrain_points(efe_scopes, pos):
    """Baseline scope points from efe-scopes.json: stable sort by binder
    ORDER (dump order within a binder), baseline metric weights."""
    by_m = defaultdict(list)
    for sc in efe_scopes:
        by_m["M-" + sc["m"]].append(sc)
    pts = []
    for m, scs in by_m.items():
        if m not in pos:
            continue
        cx, cy = pos[m]
        scs = sorted(scs, key=lambda s: ORDER.get(s["binder"], 9))
        for (dx, dy), sc in zip(spiral(len(scs)), scs):
            metric = (0.18 + (1.0 if sc["det"] else 0.0)
                      + (0.30 if sc["binder"] in FRONTIER else 0.0))
            if sc.get("vacuous"):
                metric += 0.5
            if sc["binder"] == "certificate":
                metric = (max(0.05, metric - 0.45)
                          if sc.get("verdict") == "pass" else metric + 0.8)
            pts.append((cx + dx, cy + dy, metric))
    return pts


# ------------------------------------------------------------------- field

def gaussian_scatter(field, x, y, w, step=STEP, sigma=SIGMA):
    """Add w * exp(-d^2/2 sigma^2) round (x, y) into a vertex grid."""
    gw, gh = len(field[0]), len(field)
    rc = int(3 * sigma / step)
    cgx, cgy = int(round(x / step)), int(round(y / step))
    two_s2 = 2 * sigma * sigma
    for vy in range(max(0, cgy - rc), min(gh, cgy + rc + 1)):
        dy2 = (vy * step - y) ** 2
        row = field[vy]
        for vx in range(max(0, cgx - rc), min(gw, cgx + rc + 1)):
            row[vx] += w * math.exp(-((vx * step - x) ** 2 + dy2) / two_s2)


def empty_grid(gw=GW, gh=GH):
    return [[0.0] * gw for _ in range(gh)]


def land_cells(field, theta):
    """Cell (gx, gy) is land when the vertex value >= theta, on the same
    90x90 cell domain the baseline fill draws (vertices 0..gw-2)."""
    out = set()
    for gy in range(len(field) - 1):
        row = field[gy]
        for gx in range(len(row) - 1):
            if row[gx] >= theta:
                out.add((gx, gy))
    return out


def components(cells):
    """4-connected components of a set of (gx, gy) cells."""
    seen = set()
    comps = []
    for c in cells:
        if c in seen:
            continue
        comp = set()
        stack = [c]
        seen.add(c)
        while stack:
            gx, gy = stack.pop()
            comp.add((gx, gy))
            for nb in ((gx-1, gy), (gx+1, gy), (gx, gy-1), (gx, gy+1)):
                if nb in cells and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        comps.append(comp)
    return comps


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def choose_theta(low_field, baseline_cells, n_candidates=300):
    """ONE absolute theta for every cutoff: the level whose low-tide land
    best matches the baseline's drawn terrain cells (max Jaccard)."""
    vals = sorted(low_field[gy][gx]
                  for gy in range(len(low_field) - 1)
                  for gx in range(len(low_field[0]) - 1))
    vals = [v for v in vals if v > 0]
    if not vals:
        return 0.0, 1.0 if not baseline_cells else 0.0
    cands = sorted({vals[min(len(vals) - 1, int(i * len(vals) / n_candidates))]
                    for i in range(n_candidates)})
    best, best_j = 0.0, -1.0
    for t in cands:
        j = jaccard(land_cells(low_field, t), baseline_cells)
        if j > best_j:
            best, best_j = t, j
    return best, best_j


def fields_by_cutoff(pts, cutoffs, gw=GW, gh=GH, step=STEP, sigma=SIGMA):
    """Yield (cutoff, field, n_expired) per cutoff. The field holds every
    placed scope with dated >= cutoff, plus undated scopes (dated None),
    which never expire. Built incrementally: each dated scope is added once
    and subtracted once. The field is mutated in place between yields."""
    field = [[0.0] * gw for _ in range(gh)]
    for p in pts:
        gaussian_scatter(field, p["x"], p["y"], 1.0, step, sigma)
    dated = sorted((p for p in pts if p["dated"]), key=lambda p: p["dated"])
    k = 0
    for c in cutoffs:
        while k < len(dated) and dated[k]["dated"][:10] < c:
            p = dated[k]
            gaussian_scatter(field, p["x"], p["y"], -1.0, step, sigma)
            k += 1
        yield c, field, k


def tide_land(field, low_land, tau=TAU):
    """Land at a cutoff: low-tide land cells whose density is still >= tau."""
    return {(gx, gy) for gx, gy in low_land if field[gy][gx] >= tau}


def mission_submerged(scope_dates, cutoff):
    """A hub is submerged at cutoff when the mission has no scope dated on or
    after it. An undated scope (None) keeps it above water at every cutoff;
    so does having no scopes at all (nothing known, not old)."""
    if not scope_dates or any(d is None for d in scope_dates):
        return False
    return max(d[:10] for d in scope_dates) < cutoff


def mission_dates(pts, missions):
    """Per mission, the dates that decide whether its hub is above water: its
    scopes' dates; for a mission with no scopes, its mission-doc date
    (last_work, else last_admin) when it has one. A mission with neither stays
    [] (unknown: never submerged)."""
    out = defaultdict(list)
    for p in pts:
        out[p["mission"]].append(p["dated"])
    for m, info in missions.items():
        if m not in out:
            d = info.get("last_work") or info.get("last_admin")
            if d:
                out[m].append(d)
    return out


def weekly_cutoffs(first_day, last_day):
    """Mondays from the week of first_day to the week of last_day."""
    d0 = date.fromisoformat(first_day[:10])
    d1 = date.fromisoformat(last_day[:10])
    monday = d0 - timedelta(days=d0.weekday())
    out = []
    while monday <= d1:
        out.append(monday.isoformat())
        monday += timedelta(days=7)
    return out


def water_runs(water):
    """Run-length encode water cells: [[gy, gx0, gx1], ...] merged rows."""
    rows = defaultdict(list)
    for gx, gy in water:
        rows[gy].append(gx)
    out = []
    for gy in sorted(rows):
        xs = sorted(rows[gy])
        x0 = prev = xs[0]
        for x in xs[1:]:
            if x == prev + 1:
                prev = x
            else:
                out.append([gy, x0, prev])
                x0 = prev = x
        out.append([gy, x0, prev])
    return out


def shoreline_segments(field, theta, step=STEP, cells=None):
    """Marching squares at level theta; flat [x1, y1, x2, y2, ...]. With
    cells, only those grid cells are traced."""
    def interp(p1, p2, v1, v2):
        t = (theta - v1) / (v2 - v1) if v2 != v1 else 0.5
        return (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))
    segs = []
    for gy in range(len(field) - 1):
        for gx in range(len(field[0]) - 1):
            if cells is not None and (gx, gy) not in cells:
                continue
            f00, f10 = field[gy][gx], field[gy][gx + 1]
            f01, f11 = field[gy + 1][gx], field[gy + 1][gx + 1]
            x0, y0 = gx * step, gy * step
            x1, y1 = x0 + step, y0 + step
            cr = []
            if (f00 > theta) != (f10 > theta):
                cr.append(interp((x0, y0), (x1, y0), f00, f10))
            if (f10 > theta) != (f11 > theta):
                cr.append(interp((x1, y0), (x1, y1), f10, f11))
            if (f11 > theta) != (f01 > theta):
                cr.append(interp((x1, y1), (x0, y1), f11, f01))
            if (f01 > theta) != (f00 > theta):
                cr.append(interp((x0, y1), (x0, y0), f01, f00))
            for k in range(0, len(cr) - 1, 2):
                segs.extend((round(cr[k][0], 1), round(cr[k][1], 1),
                             round(cr[k+1][0], 1), round(cr[k+1][1], 1)))
    return segs


# ------------------------------------------------------------- build (live)

def build(theta_report=None):
    t0 = time.time()
    pos = fan_out(json.loads(POS_EMBED.read_text()))
    dating = json.loads(DATING.read_text())
    missions = dating["missions"]
    pts = dating_points(dating["scopes"], pos)
    n_undated = sum(1 for p in pts if p["dated"] is None)

    # Baseline terrain grid -> drawn cells (grid/fmax >= 0.03).
    tgrid = empty_grid()
    for x, y, w in terrain_points(json.loads(EFE_SCOPES.read_text()), pos):
        gaussian_scatter(tgrid, x, y, w)
    fmax = max(max(r) for r in tgrid) or 1.0
    baseline_cells = {(gx, gy)
                      for gy in range(GH - 1) for gx in range(GW - 1)
                      if tgrid[gy][gx] / fmax >= 0.03}

    # Low-tide field: every placed scope, unit weight. theta fixes where the
    # map has land at all (matched to the baseline terrain cells).
    low = empty_grid()
    for p in pts:
        gaussian_scatter(low, p["x"], p["y"], 1.0)
    theta, theta_j = choose_theta(low, baseline_cells)
    low_land = land_cells(low, theta)

    dated_days = sorted(p["dated"][:10] for p in pts if p["dated"])
    first_day = dated_days[0] if dated_days else date.today().isoformat()
    cutoffs = weekly_cutoffs(first_day, date.today().isoformat())

    m_dates = mission_dates(pts, missions)

    weeks = []
    for c, field, expired in fields_by_cutoff(pts, cutoffs):
        land = tide_land(field, low_land)
        water = low_land - land          # only terrain is flooded; open sea
        comps = components(land)         # beyond the terrain stays undrawn
        submerged = {}
        for m in pos:
            if not mission_submerged(m_dates.get(m), c):
                continue
            info = missions.get(m, {})
            if info.get("last_work"):
                tip = f"last dated work {info['last_work'][:10]}"
            elif info.get("n_admin_commits"):
                tip = ("no non-administrative commits in git history "
                       f"(administrative only, last {info['last_admin'][:10]})")
            else:
                tip = f"last dated scope {max(m_dates[m])[:10]}"
            submerged[m] = tip
        weeks.append({
            "cutoff": c,
            "water": water_runs(water),
            "shore": shoreline_segments(field, TAU, cells=low_land),
            "islands": len(comps),
            "exposed_missions": len(pos) - len(submerged),
            "exposed_scopes": len(pts) - expired,
            "submerged": submerged,
        })
    meta = {
        "theta": theta, "theta_jaccard": theta_j, "tau": TAU,
        "cutoffs": cutoffs, "step": STEP,
        "n_missions": len(pos), "n_scopes": len(pts),
        "n_undated": n_undated,
        "build_seconds": round(time.time() - t0, 1),
    }
    return meta, weeks


# -------------------------------------------------------------------- page

CONTROLS_HTML = """
<div id="tide-controls" style="position:fixed;right:12px;top:12px;z-index:50;
background:#0c1220ee;border:1px solid #2a3a55;border-radius:8px;
padding:10px 14px;max-width:380px;color:#cfd8ea;font:15px/1.4 sans-serif">
<b>Tide (prototype A)</b> — temporal inundation<br>
<input id="tide-slider" type="range" min="0" max="0" value="0" step="1"
 style="width:340px">
<div id="tide-readout">…</div>
<div style="margin-top:6px;font-size:14px;color:#9fb0c8">__LEGEND__
Undated scopes (__N_UNDATED__) always count and are never submerged.
Scope dots here come from the dating table's scopes, so their positions
differ slightly from the baseline's dots. A mission with no scope dated on
or after the cutoff is drawn at 30% of its opacity, not hidden.</div>
</div>
"""

# Shared with archipelago_c.py: dims every element of a submerged mission and
# the capability-zone rings. Call setTideDim(week) with {cutoff, submerged}.
HUB_DIM_JS = """// A mission is drawn as several elements carrying data-m: its core disc,
// the doc ring and its hit target, the code-ring group, and its scope dots.
// A submerged mission dims all of them to 0.3 of their own opacity.
const hubs = {};
svg.querySelectorAll('[data-m]').forEach(el => {
  const m = el.getAttribute('data-m');
  const op = parseFloat(el.getAttribute('opacity') || '1');
  (hubs[m] = hubs[m] || []).push({el, op, titles: [...el.querySelectorAll('title')]
    .map(t => [t, t.textContent])});
});
// Capability-zone rings are drawn by the baseline's own script at load (and
// redrawn on live refresh), keyed by data-capability-mission-id; they are
// re-dimmed whenever the SVG's children change. Event-driven, no loop.
let current = null;
function setTideDim(w){ current = w; applyDim(); }
let observer = null;
function applyDim(){
  if (!current) return;
  if (observer) observer.disconnect();  // our own title edits are mutations
  const w = current;
  let dimmed = 0;
  for (const m in hubs) {
    const sub = w.submerged['M-' + m] || w.submerged[m];
    if (sub) dimmed++;
    for (const h of hubs[m]) {
      h.el.setAttribute('opacity', sub ? 0.3 * h.op : h.op);
      for (const [t, orig] of h.titles)
        t.textContent = orig + (sub ? ' · submerged at ' + w.cutoff + ': ' + sub : '');
    }
  }
  svg.querySelectorAll('[data-capability-mission-id]').forEach(g => {
    const m = g.getAttribute('data-capability-mission-id');
    const sub = w.submerged[m] || w.submerged['M-' + m];
    g.setAttribute('opacity', sub ? 0.3 : 1);
  });
  svg.setAttribute('data-tide-dimmed', dimmed);
  observer.observe(svg, {childList: true, subtree: true});
}
observer = new MutationObserver(applyDim);
"""

SCRIPT = """
<script>
(function(){
const TIDE = __TIDE_JSON__;
const slider = document.getElementById('tide-slider');
const readout = document.getElementById('tide-readout');
slider.max = TIDE.weeks.length - 1;
slider.value = 0;  // open at the lowest tide: the baseline's land
const STEP = TIDE.step;
const svg = document.getElementById('efe-field');
const NS = 'http://www.w3.org/2000/svg';
const gWater = document.getElementById('tide-water');
const gShore = document.getElementById('tide-shore');
__HUB_DIM_JS__
const cache = {};
function render(i){
  const w = TIDE.weeks[i];
  if (!cache[i]) {
    const wf = document.createDocumentFragment();
    for (const [gy, x0, x1] of w.water) {
      const r = document.createElementNS(NS, 'rect');
      r.setAttribute('x', x0*STEP); r.setAttribute('y', gy*STEP);
      r.setAttribute('width', (x1-x0+1)*STEP); r.setAttribute('height', STEP);
      r.setAttribute('fill', TIDE.waterFill); r.setAttribute('opacity', TIDE.waterOpacity);
      wf.appendChild(r);
    }
    const sf = document.createDocumentFragment();
    const s = w.shore;
    for (let k = 0; k + 3 < s.length; k += 4) {
      const l = document.createElementNS(NS, 'line');
      l.setAttribute('x1', s[k]); l.setAttribute('y1', s[k+1]);
      l.setAttribute('x2', s[k+2]); l.setAttribute('y2', s[k+3]);
      l.setAttribute('stroke', TIDE.shoreStroke);
      l.setAttribute('stroke-width', TIDE.shoreWidth);
      sf.appendChild(l);
    }
    cache[i] = {water: wf, shore: sf};
  }
  gWater.replaceChildren(cache[i].water.cloneNode(true));
  gShore.replaceChildren(cache[i].shore.cloneNode(true));
  setTideDim(w);
  readout.textContent = 'cutoff ' + w.cutoff + ' · ' + w.islands + ' islands · ' +
    'exposed missions ' + w.exposed_missions + '/' + TIDE.nMissions + ' · ' +
    'exposed scopes ' + w.exposed_scopes + '/' + TIDE.nScopes + ' · ' +
    'undated ' + TIDE.nUndated;
}
slider.addEventListener('input', () => render(+slider.value));
const m = (location.hash || '').match(/cutoff=(\\d{4}-\\d{2}-\\d{2})/);
if (m) {
  let best = 0;
  TIDE.weeks.forEach((w, i) => { if (w.cutoff <= m[1]) best = i; });
  slider.value = best;
}
render(+slider.value);
})();
</script>
"""


def compose_page(meta, weeks):
    doc = BASELINE_HTML.read_text()
    tide = {
        "step": STEP, "waterFill": WATER_FILL, "waterOpacity": WATER_OPACITY,
        "shoreStroke": SHORE_STROKE, "shoreWidth": SHORE_WIDTH,
        "nMissions": meta["n_missions"], "nScopes": meta["n_scopes"],
        "nUndated": meta["n_undated"], "weeks": weeks,
    }
    # Water layer right after the baseline's contour group: covers terrain
    # fill, lasso, roads and contours; sits below hubs, scopes, stars.
    pat = re.compile(r'(<g>(?:<line[^>]*stroke="#e6edff"[^>]*/>)*</g>)')
    m = pat.search(doc)
    if not m:
        raise SystemExit("could not locate the baseline contour group")
    doc = (doc[:m.end()] + '<g id="tide-water"></g><g id="tide-shore"></g>'
           + doc[m.end():])
    controls = (CONTROLS_HTML
                .replace("__LEGEND__", LEGEND)
                .replace("__N_UNDATED__", str(meta["n_undated"])))
    script = SCRIPT.replace("__HUB_DIM_JS__", HUB_DIM_JS).replace(
        "__TIDE_JSON__",
        json.dumps(tide, separators=(",", ":")).replace("</", "<\\/"))
    doc += controls + script
    return doc


def main():
    meta, weeks = build()
    doc = compose_page(meta, weeks)
    OUT.write_text(doc)
    base_size = BASELINE_HTML.stat().st_size
    added = len(doc.encode()) - base_size
    print(f"theta={meta['theta']:.4f}  jaccard={meta['theta_jaccard']:.3f}")
    print(f"cutoffs {meta['cutoffs'][0]}..{meta['cutoffs'][-1]} "
          f"({len(meta['cutoffs'])} weeks), build {meta['build_seconds']}s")
    print(f"wrote {OUT} (+{added/1e6:.2f} MB over baseline "
          f"{base_size/1e6:.2f} MB)")
    for w in weeks[:: max(1, len(weeks) // 6)]:
        print(f"  {w['cutoff']}: {w['islands']} islands, "
              f"{w['exposed_missions']}/{meta['n_missions']} missions, "
              f"{w['exposed_scopes']}/{meta['n_scopes']} scopes")


if __name__ == "__main__":
    main()
