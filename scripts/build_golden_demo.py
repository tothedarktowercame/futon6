#!/usr/bin/env python3
"""Curated highlight page for the GOLDEN-30 math.CT anatomy edition.

A small, self-contained landing page that picks a handful of exemplar papers —
each illustrating a different facet of the current markup (self-definition,
external-concept holes, appositive binds, source repair, compactness) — and
links into its full anatomy page. This is the "show off the current edition"
demo, distinct from index.html's 28-row proofread table.

Counts/titles/strata are read from the rendered index.html so they can't drift
from the actual pages.

    build_golden_demo.py            # -> data/showcases/ct-anatomy/golden/demo.html
"""
from __future__ import annotations

import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "showcases" / "ct-anatomy" / "golden"
INDEX = GOLD / "index.html"
OUT = GOLD / "demo.html"

# curated exemplars: (paper-id, facet tag, why-featured blurb). Ordered as a
# little tour — start compact, then one paper per dominant mark family.
FEATURED = [
    ("0710.2254", "compact",
     "A short erratum — small enough to read end to end. Good first click to see "
     "how holes (amber) and binds (blue) sit in running prose."),
    ("2303.08125", "self-defining",
     "Definition-dense: the detector finds 239 terms the paper defines for itself "
     "(green) against only 73 external holes — a paper that builds its own vocabulary."),
    ("2210.13443", "reference-heavy",
     "The opposite profile: 356 external-concept holes (amber) awaiting canon links "
     "and 125 appositive binds — almost everything is borrowed, little defined in place."),
    ("2310.17349", "bind-rich",
     "110 appositive binds (blue): places where the paper introduces a symbol or type "
     "inline (“let F be the functor…”). A stress test for the bind detector."),
    ("1312.2127", "source-repaired",
     "48 source repairs logged before marking — the proofread/repair pipeline fixing "
     "a degraded source so the anatomy can be detected at all."),
]


def parse_index(text: str) -> dict:
    """pid -> {stratum, defined, holes, binds, repairs, fresh, title}."""
    rows = {}
    row_re = re.compile(
        r'<tr><td><a href="([^"]+)\.html">[^<]+</a></td>'
        r'<td>([^<]*)</td><td>(\d+)</td><td>(\d+)</td><td>(\d+)</td>'
        r'<td>(\d+)</td><td>(\d+)</td><td>(.*?)</td></tr>')
    for m in row_re.finditer(text):
        rows[m.group(1)] = dict(
            stratum=m.group(2), defined=int(m.group(3)), holes=int(m.group(4)),
            binds=int(m.group(5)), repairs=int(m.group(6)), fresh=int(m.group(7)),
            title=m.group(8).strip())
    return rows


STYLE = """
body{font:16px/1.65 Georgia,serif;margin:0;color:#1d1a16;background:#fffdf8}
main{max-width:1080px;margin:0 auto;padding:30px 28px 70px}
a{color:#174ea6}
.top{border-bottom:1px solid #e8dfcf;background:#fff8e8;padding:28px;margin:0 -28px 28px}
.meta{color:#5f5548;font-family:ui-sans-serif,system-ui,sans-serif;font-size:13px}
.lede{max-width:62ch}
.legend span,.mark-defined,.mark-hole,.mark-bind{border-radius:2px;padding:0 3px}
.mark-defined{background:#d3f3df;border-bottom:2px solid #0f766e}
.mark-hole{background:#fdf3d7;border-bottom:2px dashed #9a7b1a}
.mark-bind{background:#dde7fb;border-bottom:2px solid #2a4d9a}
.card{border:1px solid #e5dccd;background:#fff;border-radius:8px;padding:18px 20px;margin:18px 0;
      box-shadow:0 1px 2px rgba(60,40,10,.04)}
.card h2{margin:0 0 2px;font-size:20px}
.card h2 a{text-decoration:none}
.facet{display:inline-block;font:600 11px/1 ui-sans-serif,system-ui,sans-serif;letter-spacing:.04em;
       text-transform:uppercase;color:#7a5; background:#eef6e9;border:1px solid #d6e6c8;
       border-radius:999px;padding:4px 9px;vertical-align:2px;margin-left:8px;color:#4a6b2f}
.why{max-width:66ch;color:#2c271f}
.counts{display:flex;flex-wrap:wrap;gap:10px;margin:12px 0 6px;font-family:ui-sans-serif,system-ui,sans-serif}
.counts b{display:block;font-size:21px;line-height:1}
.counts div{border:1px solid #e9e0d0;border-radius:6px;padding:7px 11px;min-width:78px;font-size:12px;color:#6a5f4f}
.c-def b{color:#0f766e}.c-hole b{color:#9a7b1a}.c-bind b{color:#2a4d9a}.c-rep b{color:#9a3a3a}
.open{font-family:ui-sans-serif,system-ui,sans-serif;font-size:14px;font-weight:600}
footer{margin-top:34px;font-family:ui-sans-serif,system-ui,sans-serif;font-size:14px;color:#5f5548}
"""


def main() -> int:
    idx = parse_index(INDEX.read_text(encoding="utf8"))
    cards = []
    for pid, facet, why in FEATURED:
        d = idx.get(pid)
        if not d:
            print(f"WARN: {pid} not in index, skipping")
            continue
        title = html.unescape(d["title"]) or "(untitled)"
        title = re.sub(r"\\+", " ", title).strip()
        counts = "".join([
            f'<div class="c-def"><b>{d["defined"]}</b>defined</div>',
            f'<div class="c-hole"><b>{d["holes"]}</b>holes</div>',
            f'<div class="c-bind"><b>{d["binds"]}</b>binds</div>',
            f'<div class="c-rep"><b>{d["repairs"]}</b>repairs</div>',
            f'<div><b>{d["fresh"]:,}</b>fresh scopes</div>',
        ])
        cards.append(f"""  <section class="card">
    <h2><a href="{pid}.html">{html.escape(title)}</a><span class="facet">{facet}</span></h2>
    <p class="meta">arXiv {pid} · stratum: {d['stratum']}</p>
    <div class="counts">{counts}</div>
    <p class="why">{why}</p>
    <p class="open"><a href="{pid}.html">Open full anatomy &rarr;</a></p>
  </section>""")

    doc = f"""<!doctype html>
<meta charset="utf-8">
<title>math.CT anatomy — a guided tour</title>
<style>{STYLE}</style>
<main>
  <section class="top">
    <h1>math.CT anatomy &mdash; a guided tour</h1>
    <p class="meta">Curated from the GOLDEN-30 edition &middot; {len(cards)} featured papers</p>
    <p class="lede">Each paper below is shown with its <b>full source marked up</b> by the
      conservative anatomy detector. Three mark families:</p>
    <p class="legend">
      <span class="mark-defined">defined in-paper</span> &mdash; a term the paper introduces itself &middot;
      <span class="mark-hole">external concept (hole)</span> &mdash; borrowed, needs a canon link &middot;
      <span class="mark-bind">appositive bind</span> &mdash; an inline symbol/type introduction.
    </p>
    <p class="lede">The five papers span the range of profiles the detector sees &mdash; from a
      paper that defines its own vocabulary to one that borrows almost everything.</p>
  </section>
{chr(10).join(cards)}
  <footer>All 28 proofread pages &middot; <a href="index.html">GOLDEN-30 index &rarr;</a></footer>
</main>
"""
    OUT.write_text(doc, encoding="utf8")
    print(f"wrote {OUT}  ({len(cards)} cards, {len(doc):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
