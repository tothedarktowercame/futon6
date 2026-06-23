#!/usr/bin/env python3
"""Side-by-side: CPU golden anatomy vs the REAL 70B IATC reconstruction, for the
dp-demo goldens. Reuses the existing dp_anatomy_html engine end to end —
render_marked_source (CPU) + render_argument_graphs (IATC) + R.STYLE + the dp-demo
two-up chrome (as build_superpod_mockup does). NOT a reimplementation.

    build_iatc_goldens.py  # -> data/showcases/ct-anatomy/dp-demo/mark4-iatc-goldens.html
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dp_anatomy_html as R
import dp_paper_view as dpv

import os
IDS = (os.environ["IATC_IDS"].split() if os.environ.get("IATC_IDS") else
       ["0705.0452", "0706.1286", "0708.1921", "0708.2067", "0708.2185",
        "0709.0248", "0711.0473", "0712.0724", "0801.0199", "0801.3843"])
RUN = Path(os.environ.get("IATC_RUN", str(R.ROOT / "data" / "iatc-argument-graphs" / "loop-run-70b")))
OUT = Path(os.environ.get("IATC_OUT", str(R.DEFAULT_OUT / "mark4-iatc-goldens.html")))
FIXED: dict[str, list] = {}   # pid -> [normalized 70B graph]
KMAP = {"claim": "claim", "object": "concept", "ref": "label"}
# An IATC edge's :source spans premise->conclusion lines, but the ARROW itself is
# just the illative connective. Mark only that token as `inference` (pink); if the
# prose carries no explicit connective in the edge's range, draw no arrow mark
# (don't flood whole line-ranges pink). Connective set mirrors dp_paper_view.
_ILLATIVE_CONN = re.compile(
    r"\b(if and only if|iff|it follows that|it follows|it is clear that|in fact|"
    r"without loss of generality|wlog|by contradiction|contradicts?|contradiction|"
    r"therefore|hence|thus|consequently|because|since|whence|so that|"
    r"implies that|implies|clearly)\b",
    re.IGNORECASE)


def line_offsets(text: str) -> list[int]:
    offsets = [0]
    for line in text.split("\n"):
        offsets.append(offsets[-1] + len(line) + 1)
    return offsets


def source_line_ranges(graph: dict[str, Any]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []

    def add(value: Any) -> None:
        if isinstance(value, list) and len(value) >= 2 and all(isinstance(x, int) for x in value[:2]):
            a, b = int(value[0]), int(value[1])
            if a > 0 and b >= a:
                ranges.append((a, b))

    source = graph.get("source") if isinstance(graph.get("source"), dict) else {}
    add(source.get("lines"))
    for item in list(graph.get("nodes", [])) + list(graph.get("edges", [])):
        if isinstance(item, dict) and isinstance(item.get("source"), dict):
            add(item["source"].get("lines"))
    return ranges


def passage_window(graph: dict[str, Any], text: str, margin: int = 2) -> tuple[int, int, int, int, str]:
    """Return 1-based line bounds, global char bounds, and the cropped passage text."""
    lines = text.split("\n")
    ranges = source_line_ranges(graph)
    if ranges:
        line_start = max(1, min(a for a, _ in ranges) - margin)
        line_end = min(len(lines), max(b for _, b in ranges) + margin)
    else:
        line_start, line_end = 1, len(lines)
    offsets = line_offsets(text)
    char_start = offsets[line_start - 1]
    char_end = min(offsets[line_end] - 1, len(text))
    return line_start, line_end, char_start, char_end, text[char_start:char_end]


def rebase_marks(marks: list[dict[str, Any]], char_start: int, char_end: int) -> list[dict[str, Any]]:
    rebased: list[dict[str, Any]] = []
    for mark in marks:
        start = mark.get("start")
        end = mark.get("end")
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        if start >= char_end or end <= char_start:
            continue
        updated = dict(mark)
        updated["start"] = max(start, char_start) - char_start
        updated["end"] = min(end, char_end) - char_start
        if updated["end"] > updated["start"]:
            rebased.append(updated)
    return rebased


def iatc_to_marks(graph: dict, text: str) -> list:
    """Project the standoff IATC graph onto the source text as marks, so the GPU
    side renders as MARKED-UP TEXT (same render_marked_source as the CPU side) —
    not a summary. Each node/edge :source {:lines [a b]} -> a char-offset span."""
    lines = text.split("\n")
    off = [0]
    for ln in lines:
        off.append(off[-1] + len(ln) + 1)

    def span(lr):
        if not isinstance(lr, list) or len(lr) < 2:
            return None
        a, b = lr[0], lr[1]
        if not (isinstance(a, int) and isinstance(b, int) and 1 <= a <= b <= len(lines)):
            return None
        return off[a - 1], min(off[b] - 1, len(text))

    marks = []
    for n in graph.get("nodes", []):
        sp = span((n.get("source") or {}).get("lines"))
        if sp and sp[1] > sp[0]:
            marks.append({"start": sp[0], "end": sp[1], "kind": KMAP.get(n.get("kind"), "claim"),
                          "tip": f'{n.get("kind","")}: {n.get("text","")}'.strip(": ")})
    for e in graph.get("edges", []):
        sp = span((e.get("source") or {}).get("lines"))
        if not (sp and sp[1] > sp[0]):
            continue
        w = e.get("warrant") if isinstance(e.get("warrant"), dict) else {}
        rel = str(e.get("relation", "infer"))
        if w.get("kind") == "missing-warrant":
            tip = f'⚠ {rel} — missing warrant: wants {w.get("wanted","")}'
        else:
            tip = f'{rel} — warrant: {w.get("kind","(stated)")}'
        # the arrow == a narrow anchor, NOT the whole premise->conclusion span.
        # Prefer the surface illative connective; if the prose carries none in
        # range, anchor at the start of the conclusion (final cited) line so the
        # edge is ALWAYS shown (never dropped) but never floods the window.
        seg = text[sp[0]:sp[1]]
        cm = _ILLATIVE_CONN.search(seg)
        if cm:
            cs, ce = sp[0] + cm.start(), sp[0] + cm.end()
        else:
            concl = (e.get("source") or {}).get("lines")[1]
            cs = off[concl - 1]
            ce = min(cs + 24, off[concl] - 1)
        marks.append({"start": cs, "end": max(ce, cs + 1), "kind": "inference", "tip": tip})
    return marks


def stage_graphs() -> None:
    """Parse our best 70B graph per golden, normalize to the renderer's schema —
    every node gets an :id (the model sometimes omits it), and :premise/:conclusion
    {:id N} maps become bare node-ids N — then feed them straight into the renderer
    by overriding load_iatc_graphs. No file round-trip, no schema guessing."""
    import edn_format
    import re as _re

    def _lenient_edn(text: str) -> str:
        # edn_format's lexer rejects apostrophes in symbols/keywords (e.g. :g', :phi'),
        # which are valid EDN and pass bb's reader. Strip them OUTSIDE string literals
        # so the demo can parse the graph; symbol names are cosmetic for the marks view.
        parts = _re.split(r'("(?:[^"\\]|\\.)*")', text)
        for i in range(0, len(parts), 2):
            parts[i] = parts[i].replace("'", "")
        return "".join(parts)

    for pid in IDS:
        g = RUN / f"{pid}.edn"
        if not g.exists():
            atts = sorted((RUN / ".attempts").glob(f"{pid}.attempt*.edn"))
            if not atts:
                continue
            g = atts[-1]
        try:
            parsed = R._edn_to_py(edn_format.loads(_lenient_edn(g.read_text(encoding="utf8"))))
        except Exception:
            continue
        def nid(v):  # resolve a node reference to a scalar id (id or node-id)
            if isinstance(v, dict):
                return v.get("id", v.get("node-id", str(v)))
            return v
        for i, n in enumerate(parsed.get("nodes", [])):
            if isinstance(n, dict) and "id" not in n:
                n["id"] = n.get("node-id", i)
        for e in parsed.get("edges", []):
            if not isinstance(e, dict):
                continue
            for k in ("premise", "conclusion", "given", "depends-on"):
                v = e.get(k)
                if isinstance(v, list):
                    e[k] = [nid(x) for x in v]
                elif v is not None:
                    e[k] = nid(v)
        FIXED[pid] = [parsed]


def section(pid: str) -> str:
    d = dpv.build(pid, with_ca=True, with_binders=True, with_scopes=True, with_xref=True)
    text = d["text"]
    graph = FIXED[pid][0] if FIXED.get(pid) else {}
    line_start, line_end, char_start, char_end, window_text = passage_window(graph, text)
    cmarks = rebase_marks(d["marks"], char_start, char_end)
    cpu = R.render_marked_source(window_text, cmarks)
    gmarks = rebase_marks(iatc_to_marks(graph, text), char_start, char_end) if graph else []
    gpu = (R.render_marked_source(window_text, gmarks) if gmarks
           else "<p style='padding:10px'><i>(no GPU marks)</i></p>")
    return (f'<h2>{pid} <span style="font:400 12px ui-sans-serif">'
            f'lines {line_start}&ndash;{line_end} &middot; '
            f'CPU: {len(cmarks)} marks &middot; GPU/IATC: {len(gmarks)} marks</span></h2>\n'
            f'<div class="twoup-scroll"><div class="twoup">'
            f'<div class="col"><div class="col-h now">CPU run &middot; deterministic anatomy marks</div>'
            f'<div class="paper">{cpu}</div></div>'
            f'<div class="col"><div class="col-h mk3">GPU run &middot; 70B IATC marks (on the same text)</div>'
            f'<div class="paper">{gpu}</div></div>'
            f'</div></div>')


def s5_marks(pid: str, text: str) -> list:
    """S5 grounding stage's contribution: the comprehension gaps — nouns the substrate
    could not ground — projected onto the text as :undefined marks (the red wavy ones)."""
    import json
    cf = R.ROOT / "data" / "showcases" / "clean-demo" / "comprehension.json"
    if not cf.exists():
        return []
    recs = json.load(open(cf)).get("proofs", [])
    rec = next((r for r in recs if r["pid"].split("__")[0] == pid and "rung2" not in r["pid"]), None)
    if not rec:
        return []
    marks = []
    for noun in rec.get("undefined_nouns", []):
        i = text.find(noun)
        if i >= 0:
            marks.append({"start": i, "end": i + len(noun), "kind": "undefined",
                          "tip": f"undefined in substrate — comprehension gap (comp={rec.get('comprehension')})"})
    return marks


def section_nup(pid: str) -> str:
    """N-up COMPOSITION: the same passage rendered stage by stage, each panel ADDING the
    next stage's annotations to the previous, up to a final panel where all compose."""
    d = dpv.build(pid, with_ca=True, with_binders=True, with_scopes=True, with_xref=True)
    text = d["text"]
    graph = FIXED[pid][0] if FIXED.get(pid) else {}
    ls, le, cs, ce, window = passage_window(graph, text)
    s1 = rebase_marks(d["marks"], cs, ce)
    s3 = rebase_marks(iatc_to_marks(graph, text), cs, ce) if graph else []
    s5 = rebase_marks(s5_marks(pid, text), cs, ce)
    panels = [("S1 · anatomy", s1, "deterministic concepts · binders · scopes"),
              ("+ S3 · IATC structure", s1 + s3, "the 70B inference DAG, on the same text"),
              ("+ S5 · grounding", s1 + s3 + s5, "comprehension gaps (red) — everything composed")]
    cols = "".join(
        f'<div class="col"><div class="col-h" style="background:{c}">{title}'
        f'<br><span style="font-weight:400;opacity:.9">{sub} &middot; {len(marks)} marks</span></div>'
        f'<div class="paper">{R.render_marked_source(window, marks)}</div></div>'
        for (title, marks, sub), c in zip(panels, ["#9a3412", "#0f766e", "#991b1b"]))
    return (f'<h2>{pid} <span style="font:400 12px ui-sans-serif">lines {ls}&ndash;{le} '
            f'&middot; stages COMPOSE — each panel adds the next</span></h2>\n'
            f'<div class="twoup-scroll"><div class="twoup" '
            f'style="grid-template-columns:repeat({len(panels)},1fr)">{cols}</div></div>')


def main() -> int:
    stage_graphs()
    sec = section_nup if os.environ.get("IATC_NUP") else section
    body = "\n".join(sec(pid) for pid in IDS)
    OUT.write_text(f"""<!doctype html><meta charset="utf-8">
<title>mark4 — CPU goldens vs 70B IATC</title>
<style>
body{{font:16px/1.6 Georgia,serif;margin:0;color:#1d1a16;background:#fffdf8}}
main{{max-width:1180px;margin:0 auto;padding:0 28px 70px}}
.banner{{background:#3a1d5e;color:#fbeffd;padding:16px 28px;margin:0 0 18px;font-family:ui-sans-serif,system-ui,sans-serif}}
.banner b{{color:#ffd9a8}} .banner .tag{{font-size:12.5px;opacity:.92}}
h2{{font-size:18px;border-bottom:2px solid #e8dfcf;margin-top:30px}}
.twoup-scroll{{max-height:82vh;overflow:auto;border:1px solid #d9cdbd;border-radius:7px}}
.twoup{{display:grid;grid-template-columns:1fr 1fr;gap:0;align-items:start}}
.twoup .col{{min-width:0}} .twoup .col:first-child{{border-right:1px solid #e5dccd}}
.twoup .col-h{{position:sticky;top:0;z-index:3;font:700 12px/1 ui-sans-serif,system-ui,sans-serif;padding:8px 11px;color:#fff}}
.twoup .col-h.now{{background:#9a3412}} .twoup .col-h.mk3{{background:#0f766e}}
.twoup .paper{{padding:12px;font-size:12px}}
{R.STYLE}
</style>
<main>
<div class="banner"><b>mark4 &mdash; CPU goldens vs 70B IATC reconstruction</b>
<div class="tag">Left: the deterministic CPU anatomy (human-reviewed golden). Right: the real
Llama-3.1-70B-AWQ IATC argument graph (+ mechanical repair), rendered by the same dp-anatomy engine.</div></div>
{body}
</main>""", encoding="utf8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
