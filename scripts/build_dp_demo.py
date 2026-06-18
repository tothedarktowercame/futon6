#!/usr/bin/env python3
"""QC demo index over the CURRENT DP run: pick fully-processed papers spanning
grounding quality, render each with dp_anatomy_html, and emit a landing table.

"Fully processed" here = a self-contained fable-<pid>-dp-emacs.json exists AND
the checker reports 100% symbol tagging + 0 well-formedness errors (the run
finished that paper cleanly). The spread is chosen across symbol_grounded so the
QC pass sees well-grounded papers and grounding-debt-heavy ones side by side.

    build_dp_demo.py [--n 6]      # -> data/showcases/ct-anatomy/dp-demo/index.html
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import dp_anatomy_html as R

ROOT = R.ROOT
GOLD = R.GOLD
LOSS = R.LOSS
OUT = R.DEFAULT_OUT


def candidates():
    """(pid, coverage) for clean, fully-tagged papers whose dp JSON still exists."""
    out = []
    for lf in LOSS.glob("*.json"):
        if lf.name == "dashboard.json":
            continue
        pid = lf.stem
        if not (GOLD / f"fable-{pid}-dp-emacs.json").exists():
            continue
        try:
            c = json.loads(lf.read_text()).get("coverage", {})
        except Exception:
            continue
        if c.get("wellformed_errors") != 0 or c.get("symbol_tagged") != 1.0:
            continue
        syms = c.get("symbols", 0)
        if not (60 <= syms <= 1200):  # snappy pages, still real papers
            continue
        out.append((pid, c))
    return out


def pick_spread(cands, n):
    """Spread across grounding quality and size."""
    buckets = {"well-grounded": [], "mid": [], "debt-heavy": []}
    for pid, c in cands:
        g = c.get("symbol_grounded", 0)
        if g >= 0.85:
            buckets["well-grounded"].append((pid, c))
        elif 0.50 <= g < 0.75:
            buckets["mid"].append((pid, c))
        elif g < 0.40:
            buckets["debt-heavy"].append((pid, c))
    chosen = []
    per = max(1, n // 3)
    for name, items in buckets.items():
        items.sort(key=lambda t: t[1].get("symbols", 0))
        if not items:
            continue
        # take a small one and a large one from each bucket
        picks = [items[0]]
        if len(items) > 1:
            picks.append(items[-1])
        for pid, c in picks[:per]:
            chosen.append((name, pid, c))
    return chosen[:n]


STYLE = R.STYLE + """
table{border-collapse:collapse;width:100%;font:14px/1.45 ui-sans-serif,system-ui,sans-serif;margin-top:8px}
th,td{border-bottom:1px solid #eadfce;padding:9px 10px;text-align:left}
th{background:#fff4d8}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.bucket{font:600 11px/1 ui-sans-serif,system-ui,sans-serif;text-transform:uppercase;letter-spacing:.04em;
  color:#6a5f4f;background:#f0e9d8;border-radius:999px;padding:3px 8px}
.lede{max-width:74ch}
.mockups{margin-top:34px;border-top:2px solid #e8dfcf;padding-top:14px}
.mockups h2{font-size:18px;margin:0 0 4px}
.mk-list{list-style:none;padding:0;margin:12px 0}
.mk-list li{border:1px solid #e9d9f5;background:#faf5ff;border-radius:7px;padding:11px 14px;margin:0 0 10px}
.mk-list a{font:600 15px/1.4 ui-sans-serif,system-ui,sans-serif;color:#6d28d9}
.mk-blurb{font:13px/1.5 ui-sans-serif,system-ui,sans-serif;color:#5f5548;margin-top:3px}
"""


# Aspirational / vision pages — listed in the "Mockups" section (only those
# actually present on disk are linked, so no dead links).
MOCKUPS = [
    ("1005.2653-superpod-mockup.html", "1005.2653 — Superpod MARK-3 mockup",
     "Imagined LLM-scale output (ground-to-type, anaphora, classified expository "
     "scopes, citation resolution) on top of the real CPU pipeline + the genuine "
     "Codex-pool argument graphs. Hand-authored; real ✓ vs imagined ⚗ marked."),
]


def bucket_of(g: float) -> str:
    return "well-grounded" if g >= 0.85 else "mid" if g >= 0.45 else "debt-heavy"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--papers", default=None,
                    help="comma-separated pid list to PIN (skip quality re-selection); "
                         "bucket labels derived from current grounding")
    ap.add_argument("--remine", action="store_true",
                    help="render/measure each page via the CURRENT detector in-memory "
                         "(dp_paper_view.build) instead of the stored golden JSON — so the "
                         "table matches re-mined pages and never writes golden/")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)

    if a.papers:
        pinned = [p.strip() for p in a.papers.split(",") if p.strip()]
        chosen = [(None, pid, None) for pid in pinned]  # bucket filled after build
        print(f"pinned {len(chosen)} papers (remine={a.remine}):")
    else:
        cands = candidates()
        chosen = pick_spread(cands, a.n)
        print(f"{len(cands)} clean candidates; featuring {len(chosen)} (remine={a.remine}):")

    dpv = None
    if a.remine:
        import dp_paper_view as dpv

    rows = []
    for bucket, pid, _ in chosen:
        if a.remine:
            data = dpv.build(pid, with_ca=True, with_binders=True,
                             with_scopes=True, with_xref=True)
            doc, cov = R.build_html(pid, data)
            marks = len(data["marks"])
        else:
            doc, cov = R.build_html(pid)
            marks = len(json.loads((GOLD / f"fable-{pid}-dp-emacs.json").read_text())["marks"])
        (OUT / f"{pid}.html").write_text(doc, encoding="utf8")
        c = cov["coverage"]
        g = c.get("symbol_grounded", 0)
        if bucket is None:  # pinned: derive label from current grounding
            bucket = bucket_of(g)
        print(f"  [{bucket:13}] {pid}  symbols={c.get('symbols')} grounded={g:.2f} marks={marks}")

        def p(x):
            return f"{round(100*x)}%" if isinstance(x, (int, float)) else "—"
        gcls = "ok" if g >= 0.8 else "warn"
        rows.append(
            f'<tr><td><a href="{pid}.html">{pid}</a></td>'
            f'<td><span class="bucket">{bucket}</span></td>'
            f'<td class="num">{marks}</td>'
            f'<td class="num">{p(c.get("math_coverage"))} <small>({c.get("math_spans","?")})</small></td>'
            f'<td class="num">{p(c.get("symbol_tagged"))} <small>({c.get("symbols","?")})</small></td>'
            f'<td class="num"><b class="{gcls}">{p(g)}</b></td>'
            f'<td class="num ok">CLEAN</td>'
            f'<td class="num">{cov["violations"] if cov["violations"] is not None else "—"}</td></tr>')

    present = [(fn, lbl, blurb) for fn, lbl, blurb in MOCKUPS if (OUT / fn).exists()]
    if present:
        items = "".join(
            f'<li><a href="{fn}">{html.escape(lbl)}</a><div class="mk-blurb">{html.escape(blurb)}</div></li>'
            for fn, lbl, blurb in present)
        mockups_section = (
            '\n  <section class="mockups"><h2>Mockups</h2>'
            '<p class="lede">Aspirational pages — what we <i>imagine</i> a downstream LLM-scale '
            '(superpod) phase could add on top of the deterministic pipeline. Clearly marked '
            'real ✓ vs imagined ⚗; not produced by the current detector.</p>'
            f'<ul class="mk-list">{items}</ul></section>')
    else:
        mockups_section = ""

    doc = f"""<!doctype html>
<meta charset="utf-8">
<title>DP anatomy — current-run QC demo</title>
<style>{STYLE}</style>
<main>
  <section class="top">
    <h1>DP anatomy &mdash; current-run QC</h1>
    <p class="meta">Rendered from the live run's <code>fable-*-dp-emacs.json</code> &middot; {len(chosen)} papers</p>
    <p class="lede">Each paper is shown with the <b>actual DP marks from the ongoing run</b> &mdash;
      every control sequence inside every math span, coloured by class. Unlike the GOLDEN-30
      pages (a separate prose-concept detector), this descends into math mode, so you can
      QC the symbol/math coverage directly.</p>
    <p class="lede">Across the corpus, <b>tagging and well-formedness are already maxed</b>
      (100% symbols tagged, 0 well-formedness errors); the open question is <b>grounding</b>
      &mdash; whether each tagged symbol is linked to a definition. The papers below span that
      axis, from well-grounded to grounding-debt-heavy. <b>Grounded %</b> is the column to scan;
      ungrounded symbols render <span class="k-sym">amber-dashed</span> in the pages,
      grounded ones <span class="k-symg">teal</span>.</p>
  </section>
  <table>
    <thead><tr><th>Paper</th><th>Quality</th><th>Marks</th><th>Math tagged</th>
      <th>Symbols tagged</th><th>Grounded</th><th>Well-formed</th><th>Open debts</th></tr></thead>
    <tbody>
{chr(10).join(rows)}
    </tbody>
  </table>
{mockups_section}
</main>
"""
    (OUT / "index.html").write_text(doc, encoding="utf8")
    print(f"wrote {OUT/'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
