#!/usr/bin/env python3
"""futon_city_essay.py — the Futon City visual essay page.

Reads <shots-dir>/shots.json and the <id>.jpg pictures written by
futon_city_shots.js, copies the pictures to <wip>/futon-city/, and writes
<wip>/futon-city.html: a walk through the baseline map and the archipelago
prototypes A-D, every figure linked to the published page opened in the same
configuration (the deep link is the hash the picture was taken with).

Numbers in the prose come from the pages' own readouts at shooting time, so
the text stays true to the pictures when the publisher re-runs.

Usage: futon_city_essay.py SHOTS_DIR WIP_DIR
"""
import datetime
import html
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path


def num(text, pattern, default="?"):
    m = re.search(pattern, text or "")
    return m.group(1) if m else default


CSS = """
:root{--bg:#0a0f19;--ink:#d7deea;--muted:#8d9ab0;--accent:#ffd166;--rule:#22304a}
body{margin:0;background:var(--bg);color:var(--ink);
  font:17px/1.6 Georgia,'Iowan Old Style',serif}
main{max-width:1060px;margin:0 auto;padding:0 22px 80px}
h1{font-size:2.3rem;line-height:1.15;margin:1.2rem 0 .3rem}
h2{font-size:1.45rem;margin:3rem 0 .6rem;border-top:1px solid var(--rule);padding-top:1.4rem}
p.lede{font-size:1.15rem;color:var(--ink)}
p.byline,p.note{color:var(--muted);font:14px/1.5 system-ui,sans-serif}
a{color:#9cc3ff}
.row{display:grid;gap:14px;margin:1.2rem 0}
.row.two{grid-template-columns:1fr 1fr}
.row.four{grid-template-columns:1fr 1fr 1fr 1fr}
figure{margin:0}
figure img{width:100%;display:block;border:1px solid var(--rule);border-radius:4px}
figure a:hover img{border-color:var(--accent)}
figcaption{font:13px/1.45 system-ui,sans-serif;color:var(--muted);margin-top:.35rem}
figcaption b{color:var(--ink)}
.go{font:13px system-ui,sans-serif;white-space:nowrap}
.readout{font:12px ui-monospace,monospace;color:#7f8ea6;display:block;margin-top:.2rem}
ul.keys{font:15px/1.5 system-ui,sans-serif;color:var(--ink)}
@media (max-width:760px){.row.two,.row.four{grid-template-columns:1fr}}
"""


def build(shots_dir: Path, wip: Path) -> str:
    data = json.loads((shots_dir / "shots.json").read_text())
    S = {s["id"]: s for s in data["shots"]}
    img_dir = wip / "futon-city"
    img_dir.mkdir(parents=True, exist_ok=True)
    for s in data["shots"]:
        src = shots_dir / f"{s['id']}.jpg"
        if src.exists():
            shutil.copyfile(src, img_dir / src.name)
            os.chmod(img_dir / src.name, 0o644)

    def link(i):
        s = S[i]
        return f"{s['page']}{s['hash']}"

    def fig(i, caption, wide=False):
        s = S.get(i)
        if not s or not (img_dir / f"{i}.jpg").exists():
            return f'<figure><figcaption>(picture {html.escape(i)} unavailable this run)</figcaption></figure>'
        ro = (f'<span class="readout">{html.escape(s["readout"])}</span>'
              if s.get("readout") else "")
        return (f'<figure><a href="{html.escape(link(i))}">'
                f'<img src="futon-city/{i}.jpg?v={html.escape(data["generated"][:19])}" '
                f'alt="{html.escape(caption)}" loading="lazy"></a>'
                f'<figcaption>{caption} <a class="go" href="{html.escape(link(i))}">'
                f'open this view →</a>{ro}</figcaption></figure>')

    r = {k: (S.get(k) or {}).get("readout") or "" for k in S}
    isl = lambda k: num(r[k], r"(\d+) islands")
    def exposed(k):
        m = re.search(r"exposed missions (\d+)/(\d+)|(\d+)/(\d+) missions exposed", r[k])
        if not m:
            return "?", "?"
        g = [x for x in m.groups() if x]
        return g[0], g[1]
    e_jun, total = exposed("tide-jun")
    e_sep, _ = exposed("tide-sep")
    e_nov, _ = exposed("tide-nov")
    roads_latest = num(r["roads-latest"], r"(\d+) roads")
    roads_jun = num(r["roads-jun"], r"(\d+) roads")
    stop_n = num(r["roads-stop"], r"(\d+) missions on-map")
    ferries = num(r["c-sep"], r"ferries (\d+)")
    ferry_rows = num(r["c-sep"], r"of (\d+) rows")
    tethers = num(r["c-sep"], r"tethers (\d+)")
    sunken = num(r["c-sep"], r"sunken (\d+)")
    interior = num(r["c-sep"], r"interior (\d+)")
    stamp = datetime.datetime.fromisoformat(data["generated"].replace("Z", "+00:00"))

    body = f"""
<main>
<h1>Futon City: a walk through the archipelago</h1>
<p class="byline">Hyperreal Enterprises · pictures and numbers regenerated with the map,
last at {stamp:%Y-%m-%d %H:%M} UTC · every picture opens the live page in the same view</p>

<p class="lede">Futon City is a map of the missions the FUTON agents work on. Each
district is a mission, placed by what its documents say; the terrain is how much has been
scoped there. The map below is the published page as it stands. The four prototypes after it
each try one change from the design brief: a tide that shows when work last happened, roads
that remember which pattern they share, an archipelago that keeps only the live connections,
and a hierarchy that decides what you see first.</p>

<h2>The city as it is</h2>
<div class="row two">
{fig("baseline", "<b>The baseline map.</b> Terrain height is the density of scopes around each mission; rings show mission-doc and code activity; the saucers are agents, placed at the mission they are working on.")}
<div>
<p>Three things on this map are easy to over-read. Terrain height is <em>scope count</em>,
not importance or age. The purple roads are text co-mentions: they say two missions mention
each other, but not which pattern they share. And the positions come from an embedding of the
mission documents, so distance on the map is distance in what the documents say.</p>
<p>Agents that share a spot now sit on a small ring around it; until 2026-11-25 they were
drawn as a diagonal chain that could run off the bottom of the map.</p>
</div>
</div>

<h2>A. The tide: when was there work here?</h2>
<p>Pick a date, and every place with no dated work since then goes under water. The water is
a separate field from the terrain: terrain stays scope density, and the tide reads the git
date of each scope's text. Water is not failure, and not irrelevance; it only says nothing
was written here since the cutoff. A mission whose own scopes are all older dims to 30%,
but stays on the map.</p>
<div class="row four">
{fig("tide-low", f"<b>Lowest tide.</b> All work counted: the land matches the baseline terrain.")}
{fig("tide-jun", f"<b>Since 1 June.</b> {e_jun} of {total} missions have work since; the periphery floods first.")}
{fig("tide-sep", f"<b>Since 1 September.</b> {e_sep} missions, {isl('tide-sep')} islands.")}
{fig("tide-nov", f"<b>Since 1 November.</b> {e_nov} missions, {isl('tide-nov')} islands.")}
</div>
<p class="note">Open question: the shoreline keeps a place dry when about one recent scope
lies within 80 units, so dry land is larger than the share of live missions, and some dimmed
hubs stand on dry land. A stricter level is waiting on a decision.</p>

<h2>B. Roads that remember their pattern</h2>
<p>Here a road is drawn for each pattern two missions both scope, so two missions sharing
three patterns are joined by three strands. A road appears when its later end was written.
Brightness is how often the retriever <em>offered</em> that pattern to an agent, per 1,000
turns, on one fixed scale. Offered is not used, and not validated.</p>
<div class="row two">
{fig("roads-jun", f"<b>Window ending 1 June.</b> {roads_jun} roads had been built by then.")}
{fig("roads-latest", f"<b>Latest window.</b> {roads_latest} roads: an even haze at this scale.")}
</div>
<div class="row two">
{fig("roads-stop", f"<b>One pattern, all its roads.</b> futon-theory/stop-the-line joins {stop_n} missions on the map. Pick any pattern from the list on the page.")}
<div><p>Per-pattern brightness does not read across the whole map: with nearly two thousand
strands, the eye sees texture, not structure. The query does work: choose one pattern and
every road it makes lights up, which shows how far a method has travelled.</p></div>
</div>

<h2>C. The archipelago: which live islands are connected?</h2>
<p>C puts the tide and the roads on one slider, where everything means "since this date".
Roads inside one island are hidden. Roads between two exposed islands collapse into one
<em>ferry</em>, wider for more distinct shared patterns. Roads from live work to submerged
work become faint dotted <em>tethers</em>. Roads between two submerged missions are only
counted.</p>
<div class="row two">
{fig("c-sep", f"<b>Since 1 September.</b> {ferries} ferries carry {ferry_rows} pattern roads between islands; {tethers} tethers reach under water; {sunken} roads are sunken.")}
{fig("c-sep-interior", f"<b>The same, with the {interior} interior roads shown.</b>")}
</div>
<div class="row two">
{fig("c-sep-stop", "<b>Focus still works:</b> stop-the-line's whole web over the September archipelago, including the strands now under water.")}
<div><p>This is the first view where the connections read without a query. The ferries ask a
question the baseline could not: which areas of current work share method, and how much.
The tethers ask a second one: which live work leans on patterns last used in work that has
gone quiet.</p></div>
</div>

<h2>D. Hierarchy: what should you see first?</h2>
<p>D keeps C's content and changes emphasis: four presets, three zoom levels, and a light
palette. Far out you see only islands, their names and the ferries; mid zoom adds hubs;
close in shows everything.</p>
<div class="row two">
{fig("d-work-far", "<b>Work-first, zoomed out.</b> Flat land, island names, ferries: the clearest overview so far.")}
{fig("d-light", "<b>The same on paper.</b> The hierarchy survives, so the dark scheme is not what does the work.")}
</div>
<div class="row two">
{fig("d-terrain", "<b>Terrain-first.</b> The landmass and coast are the subject; telling hubs apart is harder.")}
{fig("d-agents", "<b>Agents-first.</b> The live layer over a dimmed world: what is happening now.")}
</div>
<div class="row two">
{fig("d-near", "<b>Close in on the largest island.</b> Every label, ring and scope dot.")}
<div><p>Two status colours fall just short of 3:1 contrast on paper (alive green,
unknown grey). On the dark map they are fine.</p></div>
</div>

<h2>What the pictures do not show</h2>
<ul class="keys">
<li>The map positions are re-solved every time the publisher runs (every six hours), so the
layout can shift a little between visits.</li>
<li>The tide reads writing in mission documents. Work done only in code, or only in
conversation, floods.</li>
<li>Exposure counts what the retriever offered, not what an agent used.</li>
</ul>
</main>
"""
    return (f"<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            f"<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>Futon City: a walk through the archipelago</title>"
            f"<meta name=\"description\" content=\"The Futon City mission map and four "
            f"prototypes (tide, pattern roads, archipelago, hierarchy), each picture linked "
            f"to the live view.\"><style>{CSS}</style></head><body>{body}</body></html>\n")


def main():
    shots_dir, wip = Path(sys.argv[1]), Path(sys.argv[2])
    page = build(shots_dir, wip)
    out = wip / "futon-city.html"
    fd, tmp = tempfile.mkstemp(prefix=out.name + ".", suffix=".tmp", dir=wip)
    with os.fdopen(fd, "w") as f:
        f.write(page)
    os.replace(tmp, out)
    os.chmod(out, 0o644)
    print(f"wrote {out} ({len(page)} bytes)")


if __name__ == "__main__":
    main()
