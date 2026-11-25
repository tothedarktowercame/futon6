#!/usr/bin/env python3
"""archipelago_c.py — Prototype C (archipelago): tide (A) + roads (B).

Combines Prototype A's inundation with Prototype B's illuminated pattern
roads under ONE time semantics: everything is "since the cutoff c".

  * Tide: exactly as A (imported from archipelago_tide: fields_by_cutoff,
    tide_land, mission_submerged, TAU, components, dating_points, fan_out,
    water/shore layers).
  * Brightness: exposure rate per 1,000 turns over the KNOWN days from
    max(c, exposure first_day) to exposure last_day; gap days unknown, as
    B. One absolute ceiling over all (pattern, cutoff) pairs.
  * Connection classes per road row at c:
      interior — both ends exposed, same island (hidden by default);
      ferry    — both exposed, different islands, AGGREGATED to one line
                 per island pair (width 1 + 1.5*sqrt(k distinct patterns),
                 opacity from the brightest pattern's rate);
      tether   — one end exposed, one submerged (dotted, fixed opacity:
                 "live work shares a pattern with older work", no more);
      sunken   — both submerged (not drawn, counted).
  * An exposed mission whose hub cell is water gets its own islet id
    "islet:<mission>" — never dropped, never merged.
  * Pattern focus overrides classes: every road of the pattern as
    individual strands (#ffd166), the rest at 0.05.

Answers B's evaluation (semantics §13): the unfocused view of 1,700
individual roads was an even haze; C aggregates the inter-island
structure into ferries and tethers and hides the interior web.
"""
import json
import math
import re
import sys
import time
from bisect import bisect_left
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import futon6_config as config
import archipelago_tide as tide
import archipelago_illumination as illum

ROOT = config.ROOT
BASELINE_HTML = ROOT / "data" / "mission-efe-field-embed.html"
OUT = ROOT / "data" / "archipelago-C.html"

ROAD_COLOR = illum.ROAD_COLOR
FOCUS_COLOR = illum.FOCUS_COLOR
TETHER_COLOR = "#9fb0c8"
TETHER_OPACITY = 0.35
BASELINE_ROAD_STROKE = illum.BASELINE_ROAD_STROKE

LEGEND = ("Ferry = exposed islands that scope a common pattern; width = "
          "how many patterns, brightness = how often retrieval surfaced "
          "the brightest of them since the cutoff. Dotted tether = live "
          "work that shares a pattern with work under water. Interior "
          "roads (within one island) are hidden by default; sunken roads "
          "(both ends under water) are not drawn, only counted. Surfaced "
          "is not used, and not validated. Water = no dated work on or "
          "after the cutoff nearby. Not failure, not irrelevance. Terrain "
          "height is scope density, not age.")


# --------------------------------------------------------------- time rates

def rates_since(exposure, turns, gaps, cutoffs, first_day, last_day):
    """rate per 1,000 turns over the KNOWN days from max(c, first_day) to
    last_day, per cutoff. Returns ({cutoff: {pattern: (rate, raw)}},
    {cutoff: window_start}, {cutoff: gap_count})."""
    gaps = set(gaps)
    known = sorted(d for d in set(turns) | set(first_day and [first_day] or [])
                   if first_day <= d <= last_day and d not in gaps)
    by_day = defaultdict(Counter)
    for p, per in exposure.items():
        for d, n in per.items():
            by_day[d][p] += n
    rates, starts, gap_counts = {}, {}, {}
    for c in cutoffs:
        start = max(c, first_day)
        days = [d for d in known if start <= d <= last_day]
        t = sum(turns.get(d, 0) for d in days)
        raw = Counter()
        for d in days:
            raw.update(by_day.get(d, {}))
        rates[c] = {p: (1000.0 * n / t if t else 0.0, n)
                    for p, n in raw.items() if n}
        starts[c] = start
        gap_counts[c] = sum(1 for d in gaps if start <= d <= last_day)
    return rates, starts, gap_counts


# ----------------------------------------------------------- classification

def island_membership(land, hub_cells, submerged, cutoff):
    """island id per EXPOSED mission: the index of the 4-connected land
    component containing its hub cell, or "islet:<mission>" when the hub
    cell is water. Returns (island_of, comps, n_islets)."""
    comps = tide.components(land)
    cell_to_comp = {}
    for i, comp in enumerate(comps):
        for cell in comp:
            cell_to_comp[cell] = i
    island_of = {}
    n_islets = 0
    for m, cell in hub_cells.items():
        if submerged(m, cutoff):
            continue
        if cell in cell_to_comp:
            island_of[m] = cell_to_comp[cell]
        else:
            island_of[m] = f"islet:{m}"
            n_islets += 1
    return island_of, comps, n_islets


def classify_roads(roads, island_of, submerged, cutoff):
    """Per road row: interior / ferry / tether / sunken at the cutoff."""
    out = {"interior": [], "ferry": [], "tether": [], "sunken": []}
    for r in roads:
        sa, sb = submerged(r["a"], cutoff), submerged(r["b"], cutoff)
        if sa and sb:
            out["sunken"].append(r)
        elif sa or sb:
            out["tether"].append(r)
        elif island_of.get(r["a"]) == island_of.get(r["b"]):
            out["interior"].append(r)
        else:
            out["ferry"].append(r)
    return out


def aggregate_ferries(ferry_rows):
    """One ferry per island pair: k = number of DISTINCT patterns (not
    rows); mission pairs kept for the tooltip."""
    agg = {}
    for r in ferry_rows:
        key = tuple(sorted((r["_ia"], r["_ib"]), key=str))
        a = agg.setdefault(key, {"patterns": set(), "pairs": set(),
                                 "ia": key[0], "ib": key[1]})
        a["patterns"].add(r["pattern"])
        a["pairs"].add(tuple(sorted((r["a"], r["b"]))))
    return list(agg.values())


def aggregate_tethers(tether_rows):
    """One line per (exposed mission, submerged mission); patterns listed."""
    agg = {}
    for r in tether_rows:
        exp, sub = (r["a"], r["b"]) if r["_a_exposed"] else (r["b"], r["a"])
        a = agg.setdefault((exp, sub), {"exposed": exp, "submerged": sub,
                                        "patterns": set()})
        a["patterns"].add(r["pattern"])
    return list(agg.values())


# -------------------------------------------------------------------- build

def build():
    t0 = time.time()
    pos = tide.fan_out(json.loads(tide.POS_EMBED.read_text()))
    dating = json.loads(tide.DATING.read_text())
    pts = tide.dating_points(dating["scopes"], pos)
    rj = json.loads(illum.ROADS_JSON.read_text())
    roads, patterns = rj["roads"], rj["patterns"]
    ex = json.loads(illum.EXPOSURE_JSON.read_text())
    exposure, turns = ex["exposure"], ex["turns"]
    gaps, first_day_e, last_day_e = ex.get("gaps", []), ex["first_day"], ex["last_day"]

    # Tide fields (as A): low-tide land and per-cutoff land.
    tgrid = tide.empty_grid()
    for x, y, w in tide.terrain_points(json.loads(tide.EFE_SCOPES.read_text()), pos):
        tide.gaussian_scatter(tgrid, x, y, w)
    fmax = max(max(r) for r in tgrid) or 1.0
    baseline_cells = {(gx, gy)
                      for gy in range(tide.GH - 1) for gx in range(tide.GW - 1)
                      if tgrid[gy][gx] / fmax >= 0.03}
    low = tide.empty_grid()
    for p in pts:
        tide.gaussian_scatter(low, p["x"], p["y"], 1.0)
    theta, theta_j = tide.choose_theta(low, baseline_cells)
    low_land = tide.land_cells(low, theta)

    dated_days = sorted(p["dated"][:10] for p in pts if p["dated"])
    first_day = dated_days[0] if dated_days else date.today().isoformat()
    cutoffs = tide.weekly_cutoffs(first_day, date.today().isoformat())

    m_dates = tide.mission_dates(pts, dating["missions"])

    def submerged(m, c):
        return tide.mission_submerged(m_dates.get(m), c)

    hub_cells = {m: (min(tide.GW - 2, max(0, int(round(x / tide.STEP)))),
                     min(tide.GH - 2, max(0, int(round(y / tide.STEP)))))
                 for m, (x, y) in pos.items()}

    rates, rate_starts, rate_gaps = rates_since(
        exposure, turns, gaps, cutoffs, first_day_e, last_day_e)
    ceiling = illum.scale_max(rates, "rate")

    pat_names = sorted(patterns, key=lambda p: (-patterns[p]["n_missions"], p))
    pat_idx = {p: i for i, p in enumerate(pat_names)}
    pat_meta = [{"key": p, "n": patterns[p]["n_missions"],
                 "off": len(patterns[p]["off_map_missions"]),
                 "missions": patterns[p]["missions"]} for p in pat_names]
    unknown_pats = sorted(p for p in pat_names if p not in exposure)

    offs = illum.strand_offsets(roads)
    geoms = []
    for i, r in enumerate(roads):
        x1, y1 = pos[r["a"]]
        x2, y2 = pos[r["b"]]
        dx, dy = x2 - x1, y2 - y1
        ln = math.hypot(dx, dy) or 1.0
        ox, oy = -dy / ln * offs[i], dx / ln * offs[i]
        geoms.append([round(x1 + ox, 1), round(y1 + oy, 1),
                      round(x2 + ox, 1), round(y2 + oy, 1)])

    def centroid(members):
        xs = [pos[m][0] for m in members]
        ys = [pos[m][1] for m in members]
        return [round(sum(xs) / len(xs), 1), round(sum(ys) / len(ys), 1)]

    weeks = []
    for c, field, _expired in tide.fields_by_cutoff(pts, cutoffs):
        land = tide.tide_land(field, low_land)
        water = low_land - land
        island_of, comps, n_islets = island_membership(
            land, hub_cells, submerged, c)
        rows = [dict(r, _ia=island_of.get(r["a"]), _ib=island_of.get(r["b"]),
                     _a_exposed=not submerged(r["a"], c)) for r in roads]
        cls = classify_roads(rows, island_of, submerged, c)
        ferries = []
        # island -> exposed member missions (for centroids)
        members = defaultdict(list)
        for m, iid in island_of.items():
            members[iid].append(m)
        for a in aggregate_ferries(cls["ferry"]):
            k = len(a["patterns"])
            top = sorted(a["patterns"],
                         key=lambda p: rates[c].get(p, (0, 0))[0],
                         reverse=True)[:5]
            best = rates[c].get(top[0], (0, 0))[0] if top else 0
            ferries.append({
                "g": centroid(members[a["ia"]]) + centroid(members[a["ib"]]),
                "k": k,
                "w": round(1 + 1.5 * math.sqrt(k), 2),
                "o": round(illum.opacity(best, ceiling), 3),
                "top": [[p, round(rates[c].get(p, (0, 0))[0], 1)]
                        for p in top],
                "pairs": sorted(a["pairs"]),
                "rows": sum(1 for r in cls["ferry"]
                            if {r["_ia"], r["_ib"]} == {a["ia"], a["ib"]}),
            })
        tethers = [{
            "g": [pos[t["exposed"]][0], pos[t["exposed"]][1],
                  pos[t["submerged"]][0], pos[t["submerged"]][1]],
            "exposed": t["exposed"], "submerged": t["submerged"],
            "patterns": sorted(t["patterns"]),
        } for t in aggregate_tethers(cls["tether"])]
        weeks.append({
            "cutoff": c,
            "water": tide.water_runs(water),
            "shore": tide.shoreline_segments(field, tide.TAU, cells=low_land),
            "islands": len(comps), "islets": n_islets,
            "exposed_missions": sum(1 for m in pos if not submerged(m, c)),
            "ferries": ferries, "tethers": tethers,
            "n_interior": len(cls["interior"]),
            "interior_idx": [i for i, r in enumerate(roads)
                             if not submerged(r["a"], c)
                             and not submerged(r["b"], c)
                             and island_of.get(r["a"]) == island_of.get(r["b"])],
            "n_ferry_rows": len(cls["ferry"]),
            "n_sunken": len(cls["sunken"]),
            "submerged": {m: (f"last dated scope {max(m_dates[m])[:10]}"
                              if m_dates.get(m) else "no dated work")
                          for m in pos if submerged(m, c)},
            "rates": {str(pat_idx[p]): [round(v[0], 3), v[1]]
                      for p, v in rates[c].items() if p in pat_idx},
            "rate_start": rate_starts[c], "rate_gaps": rate_gaps[c],
        })
    data = {
        "weeks": weeks, "patterns": pat_meta,
        "roads": [{"g": geoms[i], "p": pat_idx[r["pattern"]],
                   "a": r["a"], "b": r["b"],
                   "c": (r["constructed"] or "")[:10] or None,
                   "cs": r["constructed_source"]}
                  for i, r in enumerate(roads)],
        "unknownPatterns": [pat_idx[p] for p in unknown_pats],
        "ceilRate": round(ceiling, 3),
        "opMin": illum.OPACITY_MIN, "opMax": illum.OPACITY_MAX,
        "step": tide.STEP, "waterFill": tide.WATER_FILL,
        "waterOpacity": tide.WATER_OPACITY, "shoreStroke": tide.SHORE_STROKE,
        "shoreWidth": tide.SHORE_WIDTH,
        "roadColor": ROAD_COLOR, "focusColor": FOCUS_COLOR,
        "tetherColor": TETHER_COLOR, "tetherOpacity": TETHER_OPACITY,
        "nMissions": len(pos), "nScopes": len(pts),
        "nUndated": sum(1 for p in pts if p["dated"] is None),
    }
    meta = {"build_seconds": round(time.time() - t0, 1), "cutoffs": cutoffs,
            "theta": theta, "theta_jaccard": theta_j, "ceiling": ceiling}
    return meta, data


# -------------------------------------------------------------------- page

CONTROLS_HTML = """
<div id="c-controls" style="position:fixed;right:12px;top:12px;z-index:50;
background:#0c1220ee;border:1px solid #2a3a55;border-radius:8px;
padding:10px 14px;max-width:420px;color:#cfd8ea;font:15px/1.4 sans-serif">
<b>Archipelago (prototype C)</b><br>
<input id="c-slider" type="range" min="0" max="0" value="0" step="1"
 style="width:380px">
<div id="c-readout">…</div>
<label style="font-size:14px"><input type="checkbox" id="c-interior">
interior roads</label>
<label style="font-size:14px"><input type="checkbox" id="c-tethers" checked>
tethers</label>
<label style="font-size:14px">focus
<select id="c-focus" style="max-width:210px"></select></label>
<button id="c-clear" style="font-size:14px">clear</button>
<div id="c-focus-info" style="font-size:14px;color:#ffd166"></div>
<div style="margin-top:6px;font-size:13px;color:#9fb0c8">__LEGEND__</div>
</div>
"""

SCRIPT = """
<script>
(function(){
const D = __C_JSON__;
const NS = 'http://www.w3.org/2000/svg';
const svg = document.getElementById('efe-field');
const el = id => document.getElementById(id);
const gWater = el('c-water'), gShore = el('c-shore');
const gFerry = el('c-ferries'), gTether = el('c-tethers');
const gInterior = el('c-interior-g'), gFocus = el('c-focus-g');
const slider = el('c-slider'), readout = el('c-readout');
const focusSel = el('c-focus'), focusInfo = el('c-focus-info');
const unknownSet = new Set(D.unknownPatterns);
D.patterns.forEach((p, i) => {
  const o = document.createElement('option');
  o.value = i; o.textContent = p.key + ' (' + p.n + ')';
  focusSel.appendChild(o);
});
focusSel.insertBefore(Object.assign(document.createElement('option'),
  {value: '', textContent: '(none)'}), focusSel.firstChild);
__HUB_DIM_JS__
function line(x1, y1, x2, y2, attrs) {
  const l = document.createElementNS(NS, 'line');
  l.setAttribute('x1', x1); l.setAttribute('y1', y1);
  l.setAttribute('x2', x2); l.setAttribute('y2', y2);
  for (const k in attrs) l.setAttribute(k, attrs[k]);
  return l;
}
let focus = null;
function op(v){
  if (v <= 0) return D.opMin;
  return D.opMin + (D.opMax - D.opMin) *
    Math.min(1, Math.log1p(v) / Math.log1p(Math.max(D.ceilRate, 1e-9)));
}
function render(){
  const w = D.weeks[+slider.value];
  const showInterior = el('c-interior').checked;
  const showTethers = el('c-tethers').checked;
  const wf = document.createDocumentFragment();
  for (const [gy, x0, x1] of w.water) {
    const r = document.createElementNS(NS, 'rect');
    r.setAttribute('x', x0*D.step); r.setAttribute('y', gy*D.step);
    r.setAttribute('width', (x1-x0+1)*D.step); r.setAttribute('height', D.step);
    r.setAttribute('fill', D.waterFill); r.setAttribute('opacity', D.waterOpacity);
    wf.appendChild(r);
  }
  gWater.replaceChildren(wf);
  const sf = document.createDocumentFragment();
  for (let k = 0; k + 3 < w.shore.length; k += 4)
    sf.appendChild(line(w.shore[k], w.shore[k+1], w.shore[k+2], w.shore[k+3],
      {stroke: D.shoreStroke, 'stroke-width': D.shoreWidth}));
  gShore.replaceChildren(sf);
  const ff = document.createDocumentFragment();
  for (const f of w.ferries) {
    const l = line(f.g[0], f.g[1], f.g[2], f.g[3],
      {stroke: D.roadColor, 'stroke-width': f.w, opacity: f.o,
       'stroke-linecap': 'round'});
    const t = document.createElementNS(NS, 'title');
    t.textContent = 'ferry · ' + f.k + ' pattern(s) · top: ' +
      f.top.map(x => x[0] + ' ' + x[1] + '/1k').join(', ') +
      ' · pairs: ' + f.pairs.map(p => p[0] + '↔' + p[1]).join(', ');
    l.appendChild(t); ff.appendChild(l);
  }
  gFerry.replaceChildren(ff);
  const tf = document.createDocumentFragment();
  if (showTethers) for (const t of w.tethers) {
    const l = line(t.g[0], t.g[1], t.g[2], t.g[3],
      {stroke: D.tetherColor, 'stroke-width': 1, opacity: D.tetherOpacity,
       'stroke-dasharray': '2,3'});
    const tt = document.createElementNS(NS, 'title');
    tt.textContent = 'tether · ' + t.exposed + ' (exposed) shares with ' +
      t.submerged + ' (under water): ' + t.patterns.join(', ');
    l.appendChild(tt); tf.appendChild(l);
  }
  gTether.replaceChildren(tf);
  const inf = document.createDocumentFragment();
  if (showInterior && focus === null) for (const i of w.interior_idx) {
    const r = D.roads[i];
    const rec = w.rates[String(r.p)];
    const l = line(r.g[0], r.g[1], r.g[2], r.g[3],
      {stroke: D.roadColor, 'stroke-width': 1,
       opacity: rec ? op(rec[0]) : D.opMin});
    const t = document.createElementNS(NS, 'title');
    t.textContent = D.patterns[r.p].key + ' · ' + r.a + ' ↔ ' + r.b +
      (rec ? ' · ' + rec[0].toFixed(1) + '/1k since ' + w.rate_start
           : ' · no exposure since ' + w.rate_start);
    l.appendChild(t); inf.appendChild(l);
  }
  gInterior.replaceChildren(inf);
  const ff2 = document.createDocumentFragment();
  if (focus !== null) for (const r of D.roads) {
    const focused = r.p === focus;
    const l = line(r.g[0], r.g[1], r.g[2], r.g[3], focused
      ? {stroke: D.focusColor, 'stroke-width': 2, opacity: 1}
      : {stroke: D.roadColor, 'stroke-width': 1, opacity: 0.05});
    const t = document.createElementNS(NS, 'title');
    const rec = w.rates[String(r.p)];
    t.textContent = D.patterns[r.p].key + ' · ' + r.a + ' ↔ ' + r.b +
      ' · constructed ' + (r.c || 'unknown') + ' (' + r.cs + ')' +
      (rec ? ' · ' + rec[0].toFixed(1) + '/1k since ' + w.rate_start
           : (unknownSet.has(r.p) ? ' · no retrieval record'
                                  : ' · no exposure since ' + w.rate_start));
    l.appendChild(t);
    if (focused) l.addEventListener('click', () => setFocus(''));
    ff2.appendChild(l);
  }
  gFocus.replaceChildren(ff2);
  gFerry.style.display = focus !== null ? 'none' : '';
  gTether.style.display = focus !== null ? 'none' : '';
  setTideDim(w);
  readout.textContent = 'cutoff ' + w.cutoff + ' · ' + w.islands + ' islands · ' +
    w.islets + ' islets · ' + w.exposed_missions + '/' + D.nMissions +
    ' missions exposed · interior ' + w.n_interior +
    ' · ferries ' + w.ferries.length + ' (of ' + w.n_ferry_rows + ' rows)' +
    ' · tethers ' + w.tethers.length + ' · sunken ' + w.n_sunken +
    ' · exposure since ' + w.rate_start +
    (w.rate_gaps ? ' (' + w.rate_gaps + ' gap days)' : '');
  if (focus !== null) {
    const p = D.patterns[focus];
    const rec = w.rates[String(focus)];
    focusInfo.textContent = p.key + ' · ' + p.n + ' missions on-map · ' +
      p.off + ' off-map · ' +
      (rec ? rec[0].toFixed(1) + '/1k since ' + w.rate_start
           : 'no exposure since ' + w.rate_start);
  } else focusInfo.textContent = '';
}
function setFocus(v){
  focus = v === '' ? null : +v;
  focusSel.value = v;
  render();
}
focusSel.addEventListener('change', () => setFocus(focusSel.value));
el('c-clear').addEventListener('click', () => setFocus(''));
document.addEventListener('keydown', e => { if (e.key === 'Escape') setFocus(''); });
slider.addEventListener('input', render);
el('c-interior').addEventListener('change', render);
el('c-tethers').addEventListener('change', render);
slider.max = D.weeks.length - 1;
slider.value = D.weeks.length - 1;
const h = new URLSearchParams((location.hash || '').slice(1));
if (h.get('cutoff')) {
  let best = 0;
  D.weeks.forEach((w, i) => { if (w.cutoff <= h.get('cutoff')) best = i; });
  slider.value = best;
}
if (h.get('interior') === '1') el('c-interior').checked = true;
if (h.get('focus')) {
  const i = D.patterns.findIndex(p => p.key === h.get('focus'));
  if (i >= 0) focus = i, focusSel.value = String(i);
}
render();
})();
</script>
"""


def compose_page(data):
    doc = BASELINE_HTML.read_text()
    pat = re.compile(
        r'(<g>)((?:<line[^>]*stroke="' + BASELINE_ROAD_STROKE + r'"[^>]*/>)*</g>)')
    m = pat.search(doc)
    if not m:
        raise SystemExit("could not locate the baseline road group")
    layers = ('<g style="display:none">' + m.group(2)
              + '<g id="c-water"></g><g id="c-shore"></g>'
              + '<g id="c-ferries"></g><g id="c-tethers"></g>'
              + '<g id="c-interior-g"></g><g id="c-focus-g"></g>')
    doc = doc[:m.start()] + layers + doc[m.end():]
    script = SCRIPT.replace("__HUB_DIM_JS__", tide.HUB_DIM_JS).replace(
        "__C_JSON__",
        json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    return doc + CONTROLS_HTML.replace("__LEGEND__", LEGEND) + script


def main():
    meta, data = build()
    doc = compose_page(data)
    OUT.write_text(doc)
    added = len(doc.encode()) - BASELINE_HTML.stat().st_size
    print(f"theta={meta['theta']:.4f} jaccard={meta['theta_jaccard']:.3f} "
          f"ceiling={meta['ceiling']:.1f}/1k, build {meta['build_seconds']}s")
    print(f"wrote {OUT} (+{added/1e6:.2f} MB)")
    for w in data["weeks"]:
        if w["cutoff"] in ("2026-06-01", "2026-08-31", "2026-10-26"):
            print(f"  {w['cutoff']}: {w['islands']} islands, "
                  f"{w['islets']} islets, {w['exposed_missions']} exposed, "
                  f"ferries {len(w['ferries'])} (rows {w['n_ferry_rows']}), "
                  f"tethers {len(w['tethers'])}, sunken {w['n_sunken']}")


if __name__ == "__main__":
    main()
