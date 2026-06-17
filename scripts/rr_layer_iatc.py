#!/usr/bin/env python3
"""render_run · ④ IATC — formal argument overlay (rail + margin verdict).

Anchored as a line rail (not inline) so it annotates the proof passage without
muddying the composited CPU detail. The margin card says what it ADDED and judges
it honestly."""
from __future__ import annotations
import html, re
from pathlib import Path
from rr_compositor import Annotation, Layer, Span

ROOT = Path("/home/joe/code/futon6")
DIR = ROOT / "data/iatc-argument-graphs/loop-run-70b"


def _src(block: str):
    m = re.search(r":source \{:lines \[(\d+) (\d+)\]", block)
    return (int(m.group(1)), int(m.group(2))) if m else None


def layer(pid: str, line_start=None) -> Layer:
    f = DIR / f"{pid}.edn"
    if not f.exists():
        return Layer("④", "IATC / formal argument", "#a11", "none", False, [],
                     [Annotation(1, "④", "IATC — none",
                                 '<div class="verdict">no IATC graph for this paper.</div>', "#a11")])
    t = f.read_text()
    nodes = {nid: tx for nid, _, tx in
             re.findall(r'\{:id (:[^ ,]+), :kind (:[^ ,]+), :text "([^"]*)"', t)}
    edges = re.findall(
        r"\{:id (:e-[^ ,]+), :kind :infer, :relation (:[^ ,]+), :premise (:[^ ,]+), "
        r':warrant \{:kind (:[^ ,}]+)(?:, :text "([^"]*)")?\}, :conclusion (:[^ ,}]+)'
        r'(?:, :source \{:lines \[(\d+) (\d+)\]\})?', t)
    src = _src(t[: t.find(":nodes")]) or (1, 1)
    warr = sum(1 for e in edges if e[3] != ":missing-warrant")
    n, e = len(nodes), len(edges)
    l0, l1 = src
    s_char = line_start[l0 - 1] if line_start and l0 - 1 < len(line_start) else 0
    e_char = line_start[l1] if line_start and l1 < len(line_start) else s_char + 1
    spans = [Span(s_char, max(e_char, s_char + 1), "ov-iatc", f"IATC L{l0}-{l1}")]

    # per-line reasoning rows: show each inference edge ON the line it is anchored to
    def short(nid):
        return html.escape((nodes.get(nid, "?") or "?")[:38])

    def edge_html(rel, prem, wk, wt, conc):
        warr_html = ('<span class="ir-warr ir-hole">⟨hole⟩</span>' if wk == ":missing-warrant"
                     else f'<span class="ir-warr">⟨{html.escape((wt or wk[1:])[:26])}⟩</span>')
        return (f'<span class="ir-edge"><span class="ir-node ir-prem">{short(prem)}</span>'
                f'<span class="ir-rel">{html.escape(rel[1:])}</span><span class="ir-arrow">▶</span>'
                f'<span class="ir-node ir-concl">{short(conc)}</span>{warr_html}</span>')

    rows: dict = {}
    for _eid, rel, prem, wk, wt, conc, el0, el1 in edges:
        ln = int(el0) if el0 else l0
        rows.setdefault(ln, []).append(edge_html(rel, prem, wk, wt, conc))
    rows = {ln: ['<div class="rrow iatc-row"><span class="sg" style="color:#a11">④</span>'
                 + "".join(es) + "</div>"] for ln, es in rows.items()}

    if e == 0:
        v = "<b>Adds nothing here.</b> 0 edges — a noun list, not an argument."
    elif warr == 0:
        v = (f"<b>Adds a dependency spine</b> ({e} edges) the flat marks lack — <b>but every warrant is "
             f"a hole</b>: it guesses what implies what and can't justify a single step. Skeleton only. "
             f"<i>Expected at this scale; the fix is warrant-retrieval, not a bigger model.</i>")
    elif warr >= e - 1:
        v = (f"<b>The strongest IATC graph in the run</b> — {warr}/{e} edges warranted. Roughly what we'd "
             f"want, though warrants are terse. <i>Works when justification is local; fails when it isn't.</i>")
    else:
        v = f"<b>Partly warranted</b> ({warr}/{e}) — a real but incomplete spine. <i>Expected at this scale.</i>"

    def esc(s):
        return html.escape((s or "")[:42])

    eh = "".join(
        f'<div class="edge">{esc(nodes.get(p, "?"))} <b>{rel[1:]}</b> '
        f'{"⟨hole⟩" if wk == ":missing-warrant" else "⟨" + html.escape((wt or "")[:30]) + "⟩"} '
        f'→ {esc(nodes.get(c, "?"))}</div>'
        for _eid, rel, p, wk, wt, c, _el0, _el1 in edges)
    body = (f'<div class="fact">{n} nodes · {e} edges · {warr}/{e} warranted · L{src[0]}–{src[1]}</div>'
            f'{eh}<div class="verdict">{v}</div>')
    return Layer("④", "IATC / formal argument", "#a11", "rail", True, spans,
                 [Annotation(l0, "④", "IATC — formal argument", body, "#a11")], rows)
