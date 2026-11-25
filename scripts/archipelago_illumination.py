#!/usr/bin/env python3
"""archipelago_illumination.py — Prototype B (illumination) page variant.

Reads the baseline page data/mission-efe-field-embed.html and writes
data/archipelago-B-illumination.html: the pattern-road layer from
data/pattern-roads.json (CONSTRUCTIVE channel: a pattern both missions
scope) with brightness from data/pattern-exposure-daily.json (DIAGNOSTIC
channel: how often futon3a retrieval SURFACED the pattern — an offer, not
a use, not validation; brief §4/§5/§10, semantics §4-§6).

  * Roads: one straight line per (pair, pattern) row between fanned hub
    positions; k rows of one pair are offset sideways 2 units apart,
    centred, so k shared patterns read as k strands. The baseline's own
    text co-mention road group is hidden (display:none, kept in the DOM)
    and the legend says so.
  * Windows: a slider over 30-day windows ending each Monday. rate =
    1000 * exposures / turns over the window's KNOWN days; gap days are
    reported, never zeroed. ONE absolute opacity scale for all windows
    (log1p, max = 99th percentile over all (pattern, window) pairs; raw
    count gets its own fixed scale behind the toggle).
  * Unknown vs zero: a pattern absent from the exposure table is dashed
    at minimum opacity ("no retrieval record"); a pattern present with
    zero exposure in a known window is solid at minimum opacity. The two
    are always distinguishable.
  * A road whose constructed date is after the window end is not drawn
    for that window. Null constructed (undated scope ends) is drawn and
    its tooltip says the date is unknown — unknown is not excluded.
"""
import json
import math
import re
import sys
import time
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import futon6_config as config
from archipelago_tide import fan_out  # same frame as every variant

ROOT = config.ROOT
BASELINE_HTML = ROOT / "data" / "mission-efe-field-embed.html"
POS_EMBED = ROOT / "data" / "mission-carpet-pos-embed.json"
ROADS_JSON = ROOT / "data" / "pattern-roads.json"
EXPOSURE_JSON = ROOT / "data" / "pattern-exposure-daily.json"
OUT = ROOT / "data" / "archipelago-B-illumination.html"

WINDOW_DAYS = 30
OPACITY_MIN, OPACITY_MAX = 0.12, 0.95
STRAND_OFFSET = 2.0
ROAD_COLOR = "#8fb3d9"        # one neutral hue; brightness is the only encoding
FOCUS_COLOR = "#ffd166"
BASELINE_ROAD_STROKE = "#9a7fd0"

LEGEND = ("Road = a pattern both missions scope (constructive). "
          "Brightness = how often retrieval surfaced that pattern in the "
          "window, per 1,000 turns. Surfaced is not used, and not "
          "validated.")


# ----------------------------------------------------------------- windows

def window_ends(first_day, last_day, window_days=WINDOW_DAYS):
    """Mondays from the first Monday >= first_day + window_days to the last
    Monday <= last_day (inclusive)."""
    d0 = date.fromisoformat(first_day) + timedelta(days=window_days)
    d0 += timedelta(days=(7 - d0.weekday()) % 7)
    d1 = date.fromisoformat(last_day)
    d1 -= timedelta(days=d1.weekday())
    out = []
    while d0 <= d1:
        out.append(d0.isoformat())
        d0 += timedelta(days=7)
    return out


def window_rates(exposure, turns, gaps, ends, window_days=WINDOW_DAYS):
    """Per window: {pattern: (rate, raw)} over the window's KNOWN days.

    Gap days are unknown: they contribute neither turns nor exposure (the
    rate is over known days only) and are counted per window. A day
    entirely absent from turns is a no-data day, treated as 0 turns.
    Returns (rates, gap_counts)."""
    gaps = set(gaps)
    rates, gap_counts = {}, {}
    for end in ends:
        e = date.fromisoformat(end)
        days = [(e - timedelta(days=i)).isoformat()
                for i in range(window_days - 1, -1, -1)]
        known = [d for d in days if d not in gaps]
        gap_counts[end] = len(days) - len(known)
        t = sum(turns.get(d, 0) for d in known)
        per = {}
        for p, per_day in exposure.items():
            raw = sum(per_day.get(d, 0) for d in known)
            if raw:
                per[p] = (1000.0 * raw / t if t else 0.0, raw)
        rates[end] = per
    return rates, gap_counts


def scale_max(rates, kind="rate", percentile=99):
    """ONE absolute scale ceiling for all windows: the percentile over all
    (pattern, window) pairs. kind: 'rate' or 'raw'."""
    vals = sorted(v[0 if kind == "rate" else 1]
                  for per in rates.values() for v in per.values())
    if not vals:
        return 1.0
    k = min(len(vals) - 1, max(0, math.ceil(len(vals) * percentile / 100) - 1))
    return vals[k] or 1.0


def opacity(value, ceiling):
    """log1p map onto OPACITY_MIN..OPACITY_MAX; zero exposure in a known
    window stays drawn at the minimum."""
    if value <= 0:
        return OPACITY_MIN
    ceiling = max(ceiling, 1e-9)
    f = min(1.0, math.log1p(value) / math.log1p(ceiling))
    return OPACITY_MIN + (OPACITY_MAX - OPACITY_MIN) * f


def road_style(pattern, rates_in_window, ceiling, exposure_keys):
    """(opacity, dashed, unknown). A pattern absent from the exposure table
    entirely is unknown: dashed, minimum opacity. A pattern present with
    zero exposure in a known window is solid minimum — the two are always
    distinguishable."""
    rec = rates_in_window.get(pattern)
    if rec is None:
        if pattern not in exposure_keys:
            return OPACITY_MIN, True, True
        return OPACITY_MIN, False, False
    return opacity(rec[0], ceiling), False, False


def strand_offsets(rows):
    """Rows of one (a, b) pair are offset sideways 2 units apart, centred.
    Returns {road_index: offset_units}."""
    by_pair = defaultdict(list)
    for i, r in enumerate(rows):
        by_pair[(r["a"], r["b"])].append(i)
    off = {}
    for pair, idxs in by_pair.items():
        n = len(idxs)
        for j, i in enumerate(sorted(idxs, key=lambda i: rows[i]["pattern"])):
            off[i] = (j - (n - 1) / 2) * STRAND_OFFSET
    return off


def road_drawable(constructed, window_end):
    """A road constructed after the window end is not drawn for that
    window. Unknown construction (None) is drawn, never excluded."""
    if constructed is None:
        return True
    return constructed[:10] <= window_end


# ------------------------------------------------------------------- build

def build():
    t0 = time.time()
    pos = fan_out(json.loads(POS_EMBED.read_text()))
    rj = json.loads(ROADS_JSON.read_text())
    roads, patterns = rj["roads"], rj["patterns"]
    ex = json.loads(EXPOSURE_JSON.read_text())
    exposure, turns, gaps = ex["exposure"], ex["turns"], ex.get("gaps", [])

    ends = window_ends(ex["first_day"], ex["last_day"])
    rates, gap_counts = window_rates(exposure, turns, gaps, ends)
    ceil_rate = scale_max(rates, "rate")
    ceil_raw = scale_max(rates, "raw")

    offs = strand_offsets(roads)
    geoms = []
    for i, r in enumerate(roads):
        x1, y1 = pos[r["a"]]
        x2, y2 = pos[r["b"]]
        dx, dy = x2 - x1, y2 - y1
        ln = math.hypot(dx, dy) or 1.0
        ox, oy = -dy / ln * offs[i], dx / ln * offs[i]
        geoms.append([round(x1 + ox, 1), round(y1 + oy, 1),
                      round(x2 + ox, 1), round(y2 + oy, 1)])

    pat_names = sorted(patterns, key=lambda p: (-patterns[p]["n_missions"], p))
    pat_idx = {p: i for i, p in enumerate(pat_names)}
    pat_meta = [{
        "key": p, "n": patterns[p]["n_missions"],
        "off": len(patterns[p]["off_map_missions"]),
        "missions": patterns[p]["missions"],
    } for p in pat_names]
    road_rows = [{
        "g": geoms[i], "p": pat_idx[r["pattern"]],
        "a": r["a"], "b": r["b"], "c": (r["constructed"] or "")[:10] or None,
        "cs": r["constructed_source"],
    } for i, r in enumerate(roads)]

    # Per window, sparse {pattern_idx: [rate, raw]}.
    windows = []
    for end in ends:
        per = {str(pat_idx[p]): [round(v[0], 3), v[1]]
               for p, v in rates[end].items() if p in pat_idx}
        windows.append({"end": end, "gaps": gap_counts[end], "rates": per})
    # patterns with NO retrieval record at all: absent from the exposure
    # table entirely (unknown, not zero).
    unknown = sorted(p for p in pat_names if p not in exposure)

    data = {
        "windows": windows, "roads": road_rows, "patterns": pat_meta,
        "ceilRate": round(ceil_rate, 3), "ceilRaw": ceil_raw,
        "unknownPatterns": [pat_idx[p] for p in unknown],
        "opMin": OPACITY_MIN, "opMax": OPACITY_MAX,
        "roadColor": ROAD_COLOR, "focusColor": FOCUS_COLOR,
    }
    meta = {"build_seconds": round(time.time() - t0, 1),
            "windows": ends, "ceil_rate": ceil_rate, "ceil_raw": ceil_raw,
            "n_roads": len(roads), "n_patterns": len(patterns),
            "n_unknown": len(unknown)}
    return meta, data


# -------------------------------------------------------------------- page

CONTROLS_HTML = """
<div id="illum-controls" style="position:fixed;right:12px;top:12px;z-index:50;
background:#0c1220ee;border:1px solid #2a3a55;border-radius:8px;
padding:10px 14px;max-width:400px;color:#cfd8ea;font:15px/1.4 sans-serif">
<b>Illumination (prototype B)</b><br>
<input id="illum-slider" type="range" min="0" max="0" value="0" step="1"
 style="width:360px">
<div id="illum-readout">…</div>
<label style="font-size:14px"><input type="checkbox" id="illum-raw">
raw count</label>
<label style="font-size:14px">focus
<select id="illum-focus" style="max-width:220px"></select></label>
<button id="illum-clear" style="font-size:14px">clear</button>
<div id="illum-focus-info" style="font-size:14px;color:#ffd166"></div>
<div style="margin-top:6px;font-size:14px;color:#9fb0c8">__LEGEND__
The rate is per 1,000 turns because monthly turns rose ~60x Apr→Oct 2026:
raw counts brighten recent windows just because more turns happened.
Dashed road = no retrieval record (unknown, not zero). Roads from
data/pattern-roads.json (per-pattern); the baseline's text co-mention
roads are hidden in this variant, not deleted.</div>
</div>
"""

SCRIPT = """
<script>
(function(){
const D = __ILLUM_JSON__;
const NS = 'http://www.w3.org/2000/svg';
const svg = document.getElementById('efe-field');
const g = document.getElementById('illum-roads');
const slider = document.getElementById('illum-slider');
const readout = document.getElementById('illum-readout');
const rawBox = document.getElementById('illum-raw');
const focusSel = document.getElementById('illum-focus');
const focusInfo = document.getElementById('illum-focus-info');
const unknownSet = new Set(D.unknownPatterns);
D.patterns.forEach((p, i) => {
  const o = document.createElement('option');
  o.value = i; o.textContent = p.key + ' (' + p.n + ')';
  focusSel.appendChild(o);
});
focusSel.insertBefore(Object.assign(document.createElement('option'),
  {value: '', textContent: '(none)'}), focusSel.firstChild);
focusSel.value = '';
const els = D.roads.map(r => {
  const l = document.createElementNS(NS, 'line');
  l.setAttribute('x1', r.g[0]); l.setAttribute('y1', r.g[1]);
  l.setAttribute('x2', r.g[2]); l.setAttribute('y2', r.g[3]);
  l.setAttribute('stroke', D.roadColor); l.setAttribute('stroke-width', 1);
  const t = document.createElementNS(NS, 'title');
  l.appendChild(t); g.appendChild(l);
  l.addEventListener('click', () => setFocus(String(r.p)));
  return {el: l, title: t, r};
});
let focus = null;
function op(v, ceil){
  if (v <= 0) return D.opMin;
  return D.opMin + (D.opMax - D.opMin) *
    Math.min(1, Math.log1p(v) / Math.log1p(Math.max(ceil, 1e-9)));
}
function render(){
  const w = D.windows[+slider.value];
  const ceil = rawBox.checked ? D.ceilRaw : D.ceilRate;
  let lit = 0, drawn = 0;
  for (const {el, title, r} of els) {
    const visible = !r.c || r.c <= w.end;
    el.style.display = visible ? '' : 'none';
    if (!visible) continue;
    drawn++;
    const rec = w.rates[String(r.p)];
    const val = rec ? (rawBox.checked ? rec[1] : rec[0]) : 0;
    let o = rec ? op(val, ceil) : D.opMin;
    const unknown = unknownSet.has(r.p);
    const focused = focus !== null && r.p === focus;
    if (focus !== null) o = focused ? 1 : 0.05;
    el.setAttribute('opacity', o);
    el.setAttribute('stroke', focused ? D.focusColor : D.roadColor);
    el.setAttribute('stroke-width', focused ? 2 : 1);
    // Dashed ONLY for no retrieval record at all (unknown). A known pattern with
    // no exposure in this window is solid at the minimum: road_style() in Python.
    el.setAttribute('stroke-dasharray', (!focused && unknown) ? '4,3' : '');
    if (rec && o > D.opMin + 1e-9) lit++;
    const p = D.patterns[r.p];
    title.textContent = p.key + ' · ' + r.a + ' ↔ ' + r.b +
      ' · constructed ' + (r.c || 'unknown') + ' (' + r.cs + ')' +
      (unknown ? ' · no retrieval record'
       : rec ? ' · ' + rec[0].toFixed(1) + ' per 1k turns · ' + rec[1] + ' raw'
             : ' · no exposure this window (known)');
  }
  readout.textContent = 'window ending ' + w.end + ' · ' + drawn + ' roads · ' +
    lit + ' lit above minimum' + (w.gaps ? ' · ' + w.gaps + ' gap day(s)' : '');
  if (focus !== null) {
    const p = D.patterns[focus];
    const rec = w.rates[String(focus)];
    focusInfo.textContent = p.key + ' · ' + p.n + ' missions on-map · ' +
      p.off + ' off-map · ' +
      (rec ? rec[0].toFixed(1) + ' per 1k turns (' + rec[1] + ' raw)'
           : 'no exposure in this window');
  } else focusInfo.textContent = '';
}
function setFocus(v){
  focus = v === '' ? null : +v;
  focusSel.value = v;
  render();
}
focusSel.addEventListener('change', () => setFocus(focusSel.value));
document.getElementById('illum-clear').addEventListener('click', () => setFocus(''));
document.addEventListener('keydown', e => { if (e.key === 'Escape') setFocus(''); });
slider.addEventListener('input', render);
rawBox.addEventListener('change', render);
slider.max = D.windows.length - 1;
slider.value = D.windows.length - 1;
const h = new URLSearchParams((location.hash || '').slice(1));
if (h.get('window')) {
  let best = 0;
  D.windows.forEach((w, i) => { if (w.end <= h.get('window')) best = i; });
  slider.value = best;
}
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
    # Hide the baseline's text co-mention roads in this variant; keep the
    # group in the DOM.
    pat = re.compile(
        r'(<g>)((?:<line[^>]*stroke="' + BASELINE_ROAD_STROKE + r'"[^>]*/>)*</g>)')
    m = pat.search(doc)
    if not m:
        raise SystemExit("could not locate the baseline road group")
    doc = (doc[:m.start()] + '<g style="display:none">' + m.group(2)
           + '<g id="illum-roads"></g>' + doc[m.end():])
    script = SCRIPT.replace(
        "__ILLUM_JSON__",
        json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    return doc + CONTROLS_HTML.replace("__LEGEND__", LEGEND) + script


def main():
    meta, data = build()
    doc = compose_page(data)
    OUT.write_text(doc)
    added = len(doc.encode()) - BASELINE_HTML.stat().st_size
    print(f"windows {meta['windows'][0]}..{meta['windows'][-1]} "
          f"({len(meta['windows'])}), ceilings rate={meta['ceil_rate']:.1f} "
          f"raw={meta['ceil_raw']}, roads={meta['n_roads']}, "
          f"patterns={meta['n_patterns']} ({meta['n_unknown']} with no "
          f"retrieval record), build {meta['build_seconds']}s")
    print(f"wrote {OUT} (+{added/1e6:.2f} MB)")
    latest = data["windows"][-1]
    top = sorted(latest["rates"].items(), key=lambda kv: -kv[1][0])[:5]
    print(f"latest window {latest['end']}: top by rate "
          + ", ".join(f"{data['patterns'][int(i)]['key']}={v[0]:.1f}"
                      for i, v in top))


if __name__ == "__main__":
    main()
