#!/usr/bin/env python3
"""render_run · ④ IATC — formal argument overlay (rail + margin verdict).

Anchored as a line rail (not inline) so it annotates the proof passage without
muddying the composited CPU detail. The margin card says what it ADDED and judges
it honestly."""
from __future__ import annotations
import os
import html, re
from pathlib import Path
from rr_compositor import Annotation, Layer, Span

# Derived, not hardcoded — this only ran from Joe's checkout before.
ROOT = Path(__file__).resolve().parent.parent
# Pinned to loop-run-70b until 2026-08-07, so `render_run --all` over the current
# run produced pages whose pipeline layers were empty while reporting 16/16
# rendered. An overlay that finds nothing renders the same as one that is not
# wired; only the span count tells them apart.
DIR = Path(os.environ.get("FUTON6_IATC_DIR", ROOT / "data/iatc-argument-graphs/run"))


def _src(block: str):
    m = re.search(r":source \{:lines \[(\d+) (\d+)\]", block)
    return (int(m.group(1)), int(m.group(2))) if m else None


def layer(pid: str, line_start=None) -> Layer:
    # A paper has one graph per PROOF (`<pid>__p0.edn`), not one per paper. This
    # looked up `<pid>.edn` only, which existed in loop-run-70b but not in the
    # run directory — so the layer silently found nothing and rendered "IATC —
    # none" over a corpus of 98 graphs. Same per-paper/per-proof confusion that
    # collapsed 98 graphs into 12 files in S5.
    files = [q for q in sorted(DIR.glob(f"{pid}.edn")) + sorted(DIR.glob(f"{pid}__p*.edn"))
             if not q.name.endswith(".rung2.edn")]
    if not files:
        return Layer("④", "IATC / formal argument", "#a11", "none", False, [],
                     [Annotation(1, "④", "IATC — none",
                                 '<div class="verdict">no IATC graph for this paper.</div>', "#a11")])
    nodes: dict = {}
    edges: list = []
    spans: list = []
    src = None
    for q in files:
        t = q.read_text()
        nodes.update({nid: tx for nid, _, tx in
                      re.findall(r'\{:id (:[^ ,]+), :kind (:[^ ,]+), :text "([^"]*)"', t)})
        edges.extend(re.findall(
            r"\{:id (:e-[^ ,]+), :kind :infer, :relation (:[^ ,]+), :premise (:[^ ,]+), "
            r':warrant \{:kind (:[^ ,}]+)(?:, :text "([^"]*)")?\}, :conclusion (:[^ ,}]+)'
            r'(?:, :source \{:lines \[(\d+) (\d+)\]\})?', t))
        q0, q1 = _src(t[: t.find(":nodes")]) or (1, 1)
        if src is None:
            src = (q0, q1)
        qs = line_start[q0 - 1] if line_start and q0 - 1 < len(line_start) else 0
        qe = line_start[q1] if line_start and q1 < len(line_start) else qs + 1
        spans.append(Span(qs, max(qe, qs + 1), "ov-iatc", f"IATC L{q0}-{q1}"))
    warr = sum(1 for e in edges if e[3] != ":missing-warrant")
    n, e = len(nodes), len(edges)
    l0, l1 = src or (1, 1)

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
