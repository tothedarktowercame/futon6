#!/usr/bin/env python3
"""render_run · ⑤ Expository — informal-reasoning moves (rail + margin verdict)."""
from __future__ import annotations
import os
import glob, html, re
from pathlib import Path
from rr_compositor import Annotation, Layer, Span

# Derived, not hardcoded — this only ran from Joe's checkout before.
ROOT = Path(__file__).resolve().parent.parent
# Pinned to loop-run-70b until 2026-08-07, so `render_run --all` over the current
# run produced pages whose pipeline layers were empty while reporting 16/16
# rendered. An overlay that finds nothing renders the same as one that is not
# wired; only the span count tells them apart.
DIR = Path(os.environ.get("FUTON6_EXPO_DIR", ROOT / "data/expository-scope-graphs/run"))


def _src(b: str):
    m = re.search(r":source \{:lines \[(\d+) (\d+)\]", b)
    return (int(m.group(1)), int(m.group(2))) if m else None


def layer(pid: str, line_start=None) -> Layer:
    fs = sorted(glob.glob(f"{DIR}/{pid}_*.edn"))
    if not fs:
        return Layer("⑤", "Expository reasoning", "#6d3aa8", "none", False, [],
                     [Annotation(1, "⑤", "Expository — none",
                                 '<div class="verdict">no expository graph for this paper.</div>', "#6d3aa8")])
    spans, anns, rows = [], [], {}
    for ef in fs:
        t = open(ef).read()
        km = re.search(r":kind (:[^\s]+)", t)
        kind = km.group(1) if km else "?"
        src = _src(t) or (1, 1)
        fill = re.search(r":slot-fill \{([^}]*)\}", t)
        held = re.search(r":held \{([^}]*)\}", t)
        l0, l1 = src
        s_char = line_start[l0 - 1] if line_start and l0 - 1 < len(line_start) else 0
        e_char = line_start[l1] if line_start and l1 < len(line_start) else s_char + 1
        spans.append(Span(s_char, max(e_char, s_char + 1), "ov-expo", f"expository L{l0}"))
        if held:
            bf = f'<div class="fact">held: <code>{html.escape(held.group(1).strip())}</code></div>'
            v = ("<b>Honestly held</b> — typed the move-kind but refused to fill (no support in text) "
                 "rather than hallucinate. The right behaviour.")
        elif fill:
            bf = f'<div class="fact">fill: <code>{html.escape(fill.group(1).strip())}</code></div>'
            v = ("<b>Faithful, and adds a reading cue the other layers miss</b> — a typed annotation of "
                 "<i>what the prose is doing</i>, not just its symbols. Low-stakes here but correct; this "
                 "is ⑤'s value. <i>Only 1 region/paper ran — needs scale.</i>")
        else:
            bf, v = "", "extracted a move."
        anns.append(Annotation(src[0], "⑤", f'Expository · <code>{html.escape(kind)}</code>',
                               bf + f'<div class="verdict">{v}</div>', "#6d3aa8"))
        if held:
            tail = f'⟨held: {html.escape(held.group(1).strip()[:60])}⟩'
        elif fill:
            tail = f'▶ {html.escape(fill.group(1).strip()[:70])}'
        else:
            tail = ""
        rows.setdefault(l0, []).append(
            f'<div class="rrow expo-row"><span class="sg" style="color:#6d3aa8">⑤</span>'
            f'<code>{html.escape(kind)}</code> {tail}</div>')
    return Layer("⑤", "Expository reasoning", "#6d3aa8", "rail", True, spans, anns, rows)
