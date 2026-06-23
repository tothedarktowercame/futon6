#!/usr/bin/env python3
"""Render a paper's CURRENT DP anatomy markup to standalone HTML for QC.

Unlike build_golden_paper.py (a separate, conservative prose-concept detector
that never enters math mode), this renders the *actual* DP marks emitted by the
ongoing run — every control sequence inside a math span, coloured by class — so
the symbol/math coverage invariant is visible. Source text + mark offsets both
come from the self-contained fable-<pid>-dp-emacs.json, so no eprint lookup or
offset-alignment is needed.

Marks overlap (a `math` scope contains `symbol` marks; a `let-binder` contains a
`definiendum`), so rendering is a sweep: emit a new <span> whenever the set of
active marks changes, carrying every active class + a combined tooltip.

    dp_anatomy_html.py <paper-id> [--out DIR]
"""
from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "showcases" / "ct-anatomy" / "golden"
LOSS = ROOT / "data" / "loss"
DEFAULT_OUT = ROOT / "data" / "showcases" / "ct-anatomy" / "dp-demo"

# kind (or kind-prefix) -> css class. Order matters: first match wins.
KIND_CLASS = [
    ("undefined", "k-undef"),
    ("symbol-grounded", "k-symg"),
    ("symbol", "k-sym"),
    ("math", "k-math"),
    ("let-binder", "k-bind"),
    ("bind/", "k-bind"),
    ("definiendum", "k-defd"),
    ("definiens", "k-defs"),
    ("constrain/", "k-con"),
    ("quant/", "k-quant"),
    ("assume/", "k-assume"),
    ("cite", "k-cite"),
    ("kw-hyp", "k-kw-hyp"),
    ("kw-con", "k-kw-con"),
    ("anaphor", "k-anaphor"),
    ("implies", "k-impl"),
    ("env/", "k-env"),
    ("inference", "k-inf"),
    ("claim", "k-claim"),
    ("concept", "k-concept"),
    ("classified", "k-cls"),
    ("unknown", "k-unk"),
    ("label", "k-label"),
]
# kinds rendered with no visual treatment (structural noise for this view)
SKIP_KINDS = {"layout", "text-mode"}

LEGEND = [
    ("k-math", "math span"),
    ("k-symg", "symbol — grounded"),
    ("k-sym", "symbol — tagged, ungrounded (grounding debt)"),
    ("k-bind", "binder / appositive bind"),
    ("k-defd", "definiendum"),
    ("k-defs", "definiens"),
    ("k-concept", "named concept / term"),
    ("k-con", "constraint"),
    ("k-quant", "quantifier"),
    ("k-cite", "citation"),
    ("k-impl", "Let–Then implication (one scope)"),
    ("k-kw-hyp", "hypothesis keyword"),
    ("k-kw-con", "conclusion keyword"),
    ("k-anaphor", "anaphor → bound item"),
    ("k-inf", "inference (illative)"),
    ("k-claim", "claim / proposition"),
    ("k-env", "theorem/lemma/def environment"),
    ("k-cls", "recognised control seq"),
    ("k-unk", "genuine unknown"),
]

STYLE = """
body{font:16px/1.6 Georgia,serif;margin:0;color:#1d1a16;background:#fffdf8}
main{max-width:1180px;margin:0 auto;padding:26px 28px 70px}
a{color:#174ea6}
.top{border-bottom:1px solid #e8dfcf;background:#fff8e8;padding:22px 26px;margin:0 -28px 22px}
.meta{color:#5f5548;font-family:ui-sans-serif,system-ui,sans-serif;font-size:13px}
.stat{display:flex;flex-wrap:wrap;gap:10px;margin:14px 0 4px;font-family:ui-sans-serif,system-ui,sans-serif}
.stat div{border:1px solid #e9e0d0;background:#fff;border-radius:6px;padding:7px 11px;min-width:96px;font-size:12px;color:#6a5f4f}
.stat b{display:block;font-size:20px;line-height:1.1;color:#1d1a16}
.ok{color:#0f766e}.warn{color:#c2410c}
.legend{font-family:ui-sans-serif,system-ui,sans-serif;font-size:12.5px;line-height:2.1}
.legend span{padding:1px 5px;border-radius:3px;margin-right:3px}
.paper{font:13px/1.7 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  background:#fff;border:1px solid #e5dccd;border-radius:6px;padding:18px 18px 18px 0;overflow-wrap:anywhere}
.codeline{display:grid;grid-template-columns:4em auto 1fr;align-items:stretch}
.codeline:target .lc,.codeline:hover .lc{background:#fff4d8}
.ln{color:#c2b9a7;text-align:right;user-select:none;-webkit-user-select:none;
  font-size:11.5px;padding-right:12px}
.ln a{color:inherit;text-decoration:none}
.rails{display:flex;align-items:stretch}
.lc{white-space:pre-wrap;overflow-wrap:anywhere;min-height:1.7em;padding-left:2px}
/* nested-blockquote rails: one CONTINUOUS full-height vertical bar per active
   multi-line ENVIRONMENT/implication scope, outer to inner — like nested
   follow-up threads. Each bar fills its row and rows abut, so a scope's rail is
   one unbroken line across all its lines. Only containing scopes (env/*, the
   Let–Then implication) draw rails; short inline scopes (binders) do not, so a
   binder that runs past an environment boundary can't cross a rail. */
.rail{display:block;width:0;border-left:3px solid #d8cdb6;margin-right:7px}
.rail.r-impl{border-color:#7c6bd6}
.rail.r-env{border-color:#c9a227}
.rail.r-claim{border-color:#c026d3}
.rail.r-expo{border-color:#0f766e}  /* expository scope (teal) — the prose layer */
/* a rail BREAKS (gap above) only when ITS scope changes — env rail breaks
   between environments, pink rail breaks between reasoning regions —
   independently, so the proof rail stays continuous across region breaks. */
.rail.rail-break{margin-top:.7em}
/* re-laid-out IATC reasoning rows (claims + arrow divider), nested under env */
.codeline.iq .lc{padding:1px 0 1px 8px;color:#3a2f46}
/* IATC inference broken out from the source (a DERIVED view of the standoff
   claim/inference marks — the reasoning is styled here, not echoed inline):
   LHS claim / magenta arrow divider / RHS claim, one continuous magenta
   blockquote bar through all three, nested under whatever environment rails. */
.codeline.iq .lc{padding:1px 0 1px 8px;color:#3a2f46}
.iq .qarrow{color:#c026d3;font-weight:700}
.iq .qarrow::before{content:"="}
.iq .qarrow::after{content:"=>"}
/* mark families — combine when nested (e.g. a grounded symbol inside a math span) */
.k-math{background:#eef3fb;border-radius:2px}
.k-undef{background:#fde8e8;border-bottom:2px wavy #dc2626;color:#991b1b;font-weight:600}
.k-symg{color:#0b5e57;border-bottom:2px solid #0f766e;font-weight:600}
.k-sym{background:#fdf3d7;border-bottom:2px dashed #c2410c;color:#8a3d05;font-weight:600}
.k-bind{border-bottom:2px solid #2a4d9a}
.k-defd{background:#efe3fb;color:#6b21a8;font-weight:700;border-radius:2px}
.k-defs{color:#6b21a8;font-style:italic}
.k-concept{background:#e9f7ef;border-bottom:2px dotted #15803d;color:#166534}
/* Let–Then implication = ONE scope: a single continuous underline runs the whole
   span (text-decoration flows through nested marks and abutting segments, so it
   reads as one unit even across the sentence boundary). */
.k-impl{background:#faf7ff;text-decoration:underline;text-decoration-color:#7c6bd6;
  text-decoration-thickness:2px;text-underline-offset:4px;text-decoration-skip-ink:none}
/* hypothesis / conclusion keywords, coloured by syntax class (Let = binder,
   Then = inference), bold so they read as keywords. */
.k-kw-hyp{font-weight:700;color:#1d4ed8}
.k-kw-con{font-weight:700;color:#6d28d9}
/* IATC reasoning layer: claims (clause-level propositions, dotted) + the inference
   illative rendered INLINE as a stylized magenta arrow =relation=>, so the triple
   reads in place (subject =relation=> object) without a separate table. */
.k-claim{border-bottom:1px dotted #9a8fb0}
/* anaphor: a (1)/(2) reference resolved to its bound enumerate item (hover) */
.k-anaphor{border-bottom:1px dotted #1d4ed8;color:#1d4ed8;cursor:help}
/* inference illative: ALWAYS magenta — highest precedence so a surrounding
   quant/constraint colour can never override it. */
.k-inf,.lc .k-inf{color:#c026d3 !important;font-weight:700}
.k-inf::before{content:"="}
.k-inf::after{content:"=>"}
/* a single logical arrow can be SPLIT into adjacent k-inf spans when another
   mark overlaps it (e.g. the anaphor on the ref in "following (1)"). Decorate
   the RUN once: "=" only before the first segment, "=>" only after the last, so
   a split arrow reads "=following (1)=>" not "=following =>=(1)=>". */
.k-inf + .k-inf::before{content:""}
.k-inf:has(+ .k-inf)::after{content:""}
.k-con{border-bottom:1px dotted #9a3a3a}
.k-quant{border-bottom:1px solid #4a6b2f;color:#3f5d28}
.k-assume{border-bottom:1px solid #4a6b2f}
.k-cite{color:#9a8f80}
.k-env{background:#fff4d8;border-radius:2px}
.k-cls{border-bottom:1px solid #c8bca6}
.k-unk{background:#ffe2e2;border-bottom:2px dotted #b91c1c}
.k-label{color:#b9b0a0}
/* IATC argument-graph panel (layer b: interpretive reconstruction, standoff) */
.iatc{margin:26px 0 0;border-top:2px solid #e8dfcf;padding-top:14px}
.iatc h2{font-size:17px;margin:0 0 2px}
.iatc .sub{color:#6a5f4f;font:13px/1.5 ui-sans-serif,system-ui,sans-serif;margin:0 0 12px}
.ax{border:1px solid #e5dccd;border-radius:7px;padding:11px 14px;margin:0 0 12px;background:#fffdf8}
.ax > h3{font:600 13px/1.3 ui-sans-serif,system-ui,sans-serif;margin:0 0 8px;color:#3a342b}
.ax .lines{font-weight:400;color:#9a8f80}
.ax .lines a{color:#9a8f80;text-decoration:none;border-bottom:1px dotted #c9bfae}
.ax-edge{display:grid;grid-template-columns:1fr auto 1fr;gap:8px;align-items:start;
  padding:7px 0;border-top:1px dashed #efe7d7;font:13px/1.45 ui-sans-serif,system-ui,sans-serif}
.ax-edge:first-of-type{border-top:0}
.ax-prem,.ax-concl{color:#2b2722}
.ax-rel{color:#c026d3;font-weight:700;white-space:nowrap;align-self:center}
.ax-rel::before{content:"="}.ax-rel::after{content:"=>"}
.ax-warr{grid-column:1/-1;font-size:12px;color:#3f6b4a;padding:2px 0 0 2px}
.ax-warr.cite{color:#6a5f4f}
.ax-hole{grid-column:1/-1;font-size:12px;color:#9a3412;background:#fff3e8;
  border-left:3px solid #ea7317;padding:3px 8px;margin:2px 0 0;border-radius:3px}
.ax-meta{color:#8a8276;font-style:italic}
"""


def klass(kind: str) -> str | None:
    if kind in SKIP_KINDS:
        return None
    for key, cls in KIND_CLASS:
        if kind == key or (key.endswith("/") and kind.startswith(key)):
            return cls
    return "k-cls"  # any other recognised mark


def _render_range(text: str, usable: list, lo: int, hi: int) -> str:
    """Sweep-line over [lo,hi): emit a span each time the active mark-set changes.
    Marks are clipped to the range, so a mark crossing a line boundary still
    decorates its portion on each line."""
    pts = {lo, hi}
    for m in usable:
        if m["start"] < hi and m["end"] > lo:
            pts.add(max(lo, m["start"])); pts.add(min(hi, m["end"]))
    bounds = sorted(pts)
    out = []
    for a, b in zip(bounds, bounds[1:]):
        seg = html.escape(text[a:b]).replace("\n", " ")  # flow, no hard wrap
        active = [m for m in usable if m["start"] <= a and m["end"] >= b]
        if not active:
            out.append(seg)
            continue
        classes, tips = [], []
        for m in active:
            c = klass(m["kind"])
            if c not in classes:
                classes.append(c)
            tip = m.get("tip") or m["kind"]
            if tip not in tips:
                tips.append(tip)
        title = html.escape(" · ".join(tips))
        out.append(f'<span class="{" ".join(classes)}" title="{title}">{seg}</span>')
    return "".join(out)


def _rail_family(kind: str) -> str | None:
    """Which nested-blockquote rail a scope draws (None = no rail). Containing
    scopes that nest cleanly: environments, the Let–Then implication, and claim
    propositions (the IATC operands — a claim that spans lines gets its own
    magenta blockquote rail; inline claims get the dotted style instead).
    Binders run past environment boundaries, so they get no rail."""
    if kind.startswith("env/"):
        return "env"
    return {"implies": "impl", "claim": "claim", "exposition": "expo"}.get(kind)


def render_marked_source(text: str, marks: list) -> str:
    """Render the source ONCE as a sequence of SEQUENTIALLY numbered rows (so any
    row is referenceable). Rows break at source newlines and — inside a reasoning
    region — at claim and sentence boundaries, so a PINK claim rail brackets
    EXACTLY the claim/inference content and never sweeps in adjacent prose.
    Environments / implications draw their own (gold / purple) rails; a nested
    inference adds an extra indented pink rail. Pure standoff: nothing restated,
    every character rendered exactly once."""
    usable = [m for m in marks
              if 0 <= m.get("start", 0) < m.get("end", 0) <= len(text)
              and klass(m.get("kind", ""))]
    # environment / implication scopes -> rails; claims drive the PINK depth.
    scope_rails = [(m["start"], m["end"], _rail_family(m["kind"]))
                   for m in marks if _rail_family(m.get("kind", "")) in ("env", "impl", "expo")]
    # reasoning regions: cluster claim+inference spans across SMALL connective
    # gaps (commas, ", (2) ", " and, ") but NEVER across a ". " sentence
    # boundary. So one inference sentence is a single continuous region, the
    # next sentence (and any descriptive prose) is outside it.
    rspans = sorted((m["start"], m["end"]) for m in marks
                    if m.get("kind") in ("claim", "inference"))
    regions = []
    for s, e in rspans:
        if regions:
            gap = text[regions[-1][1]:s]
            if len(gap) <= 24 and ". " not in gap and ".\n" not in gap:
                regions[-1][1] = max(regions[-1][1], e)
                continue
        regions.append([s, e])
    # a reasoning region must carry illative structure: at least one inference
    # ARROW. A claim with no arrow (e.g. an imperative "Now consider the
    # category …" mis-tagged as a claim) is not a reasoning region and must NOT
    # draw a pink blockquote bar — the bar without an arrow reads as a dangling
    # annotation. Drop claim-only regions before they shape the rails.
    _inf_spans = [(m["start"], m["end"]) for m in marks if m.get("kind") == "inference"]
    regions = [r for r in regions
               if any(a < r[1] and b > r[0] for a, b in _inf_spans)]
    # absorb each region's sentence-ending period so it doesn't fall onto the
    # descriptive row below.
    for r in regions:
        j = r[1]
        while j < len(text) and text[j] == " ":
            j += 1
        if j < len(text) and text[j] == ".":
            r[1] = j + 1

    def in_region(p):
        return any(a <= p < b for a, b in regions)

    inf_regions = []
    for m in marks:
        if m.get("kind") == "inference":
            n = m.get("nest", 0)
            for sp in (m.get("subj_span"), m.get("obj_span"), [m["start"], m["end"]]):
                if sp:
                    inf_regions.append((sp[0], sp[1], n))

    def pink_depth(x):
        ns = [n for a, b, n in inf_regions if a <= x < b]
        return 1 + (max(ns) if ns else 0)

    # cut points: every newline (hard); region edges; and where the nesting
    # depth changes (inference sub-span edges) — NOT at every claim, so the bar
    # stays continuous within a region.
    cuts = {0, len(text)}
    for i, ch in enumerate(text):
        if ch == "\n":
            cuts.update((i, i + 1))
    for a, b in regions:
        cuts.update((a, b))
    for a, b, _ in inf_regions:
        cuts.update((a, b))
    cuts = sorted(c for c in cuts if 0 <= c <= len(text))

    def region_id(p):
        for i, (a, b) in enumerate(regions):
            if a <= p < b:
                return i
        return -1

    def _rails(x, y):
        env = [fam for _, _, fam in
               sorted(((e - s, s, fam) for s, e, fam in scope_rails
                       if s <= x and y <= e), reverse=True)]
        pink = pink_depth((x + y) // 2) if in_region(x) else 0
        return tuple(env), pink

    arrow_spans = [(m["start"], m["end"]) for m in marks if m.get("kind") == "inference"]

    def is_arrow(x, y):
        mid = (x + y) // 2
        return any(a <= mid < b for a, b in arrow_spans)

    # build rows: the inference ARROW is always a STANDALONE row, so the region
    # splits cleanly into LHS (before) / arrow / RHS (after). Claim text and
    # connective glue coalesce into the LHS/RHS around it (same region, rails,
    # no newline); nothing merges into or across an arrow row.
    merged = []  # [x, y, (env), pink, region_id, is_arrow]
    for x, y in zip(cuts, cuts[1:]):
        if not text[x:y].strip():
            continue
        env, pink = _rails(x, y)
        rid = region_id(x)
        arrow = is_arrow(x, y)
        prev = merged[-1] if merged else None
        # within ONE region (a single sentence) source newlines are just
        # wrapping — collapse them so each operand is one row; elsewhere a
        # newline ends the row.
        same_region = prev is not None and prev[4] == rid and rid >= 0
        nonl = prev is not None and ("\n" not in text[prev[1]:x] or same_region)
        punct = not any(c.isalnum() for c in text[x:y])
        if arrow:
            merged.append([x, y, env, pink, rid, True])
        elif nonl and punct:           # trailing punctuation rides with prev row
            prev[1] = y
        elif nonl and not prev[5] and prev[2] == env and prev[3] == pink \
                and prev[4] == rid:
            prev[1] = y
        else:
            merged.append([x, y, env, pink, rid, False])

    rows, prev_rid, prev_envset = [], None, set()
    for n, (x, y, _env, pink, rid, _arrow) in enumerate(merged, start=1):
        while x < y and text[x] in " \n\t":
            x += 1
        while y > x and text[y - 1] in " \n\t":
            y -= 1
        seg = _render_range(text, usable, x, y)
        # PER-SCOPE breaking: an env rail breaks (gap) only when THAT scope
        # starts (the row above wasn't in it), so an inner scope ending (the
        # enumerate) doesn't break the outer scope's rail (the proof). Outer
        # scopes first. Pink rail breaks only when the reasoning region changes.
        # the expository scope is the CONTINUOUS outer rail of the passage; the
        # reasoning (pink) region nests INSIDE it as an inner rail. Both are kept
        # (nested blockquotes) — the expo span contains the reasoning span, so
        # they nest cleanly rather than cross.
        env_scopes = sorted(((s, e, f) for s, e, f in scope_rails if s <= x and y <= e),
                            key=lambda t: -(t[1] - t[0]))
        env_rails = ""
        for s, e, f in env_scopes:
            brk = "" if (s, e) in prev_envset else " rail-break"
            env_rails += f'<span class="rail r-{f}{brk}"></span>'
        prev_envset = {(s, e) for s, e, _f in env_scopes}
        pbrk = " rail-break" if rid != prev_rid else ""
        prev_rid = rid
        rails = env_rails + f'<span class="rail r-claim{pbrk}"></span>' * pink
        rows.append(
            f'<div class="codeline" id="L{n}">'
            f'<span class="ln"><a href="#L{n}">{n}</a></span>'
            f'<span class="rails">{rails}</span>'
            f'<span class="lc">{seg}</span></div>')
    return "".join(rows)


def fresh_coverage(pid: str) -> dict:
    """Run check_invariants for current stats; fall back to existing loss file."""
    try:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "check_invariants.py"), pid],
                       cwd=ROOT, capture_output=True, timeout=180, check=False)
    except Exception:
        pass
    lf = LOSS / f"{pid}.json"
    if lf.exists():
        d = json.loads(lf.read_text())
        return {"coverage": d.get("coverage", {}), "violations": len(d.get("violations", []))}
    return {"coverage": {}, "violations": None}


# --- IATC argument graphs (layer b: interpretive reconstruction) -----------
# Codex-pool-built, checker-PASS .edn argument graphs (warrants + typed holes),
# line-anchored standoff over the source. We render them as a reasoning panel
# beneath the source — nothing is restated inline; the panel IS the layer-b view.
IATC_GRAPH_DIR = Path(
    "/home/joe/code/futon3c/holes/excursions/close-reading/iatc-clojure")


def _edn_to_py(o):
    import edn_format
    if isinstance(o, edn_format.Keyword):
        return str(o).lstrip(":")
    if isinstance(o, (edn_format.ImmutableDict, dict)):
        return {_edn_to_py(k): _edn_to_py(v) for k, v in dict(o).items()}
    if isinstance(o, (edn_format.ImmutableList, list, tuple)):
        return [_edn_to_py(x) for x in o]
    return o


def load_iatc_graphs(pid: str) -> list:
    """Parsed .edn argument graphs for `pid`, sorted by source line. [] if none."""
    d = IATC_GRAPH_DIR / pid
    if not d.is_dir():
        return []
    import edn_format
    graphs = []
    for f in sorted(d.glob("*.edn")):
        try:
            graphs.append(_edn_to_py(edn_format.loads(f.read_text(encoding="utf8"))))
        except Exception as exc:
            print(f"  iatc graph skipped ({f.name}: {exc})", file=sys.stderr)
    graphs.sort(key=lambda g: (g.get("source", {}).get("lines") or [0])[0])
    return graphs


def _ids(v):
    return v if isinstance(v, list) else [v] if v else []


def render_argument_graphs(pid: str) -> str:
    graphs = load_iatc_graphs(pid)
    if not graphs:
        return ""
    out = ['<section class="iatc">',
           '<h2>IATC argument reconstruction</h2>',
           '<p class="sub">Interpretive layer (b): inference graphs anchored to '
           'source lines — warrants made explicit, and <b>typed holes</b> naming '
           'exactly what the prose elides. Standoff: nothing is restated inline.</p>']
    for g in graphs:
        nodes = {n["id"]: n for n in g.get("nodes", [])}

        def txt(nid):
            n = nodes.get(nid)
            return html.escape(n["text"]) if n else html.escape(str(nid))
        ln = g.get("source", {}).get("lines") or [0, 0]
        label = html.escape(g.get("passage/id", "").split(":", 1)[-1] or g.get("passage/id", ""))
        out.append(f'<div class="ax"><h3>{label} '
                   f'<span class="lines"><a href="#L{ln[0]}">L{ln[0]}–{ln[1]}</a></span></h3>')
        for e in g.get("edges", []):
            prem = _ids(e.get("premise")) + _ids(e.get("given")) + _ids(e.get("depends-on"))
            concl = _ids(e.get("conclusion"))
            rel = html.escape(str(e.get("relation", "infer")).replace("-", " "))
            prem_h = " · ".join(txt(p) for p in prem) or "<span class=ax-meta>(prior context)</span>"
            concl_h = " · ".join(txt(c) for c in concl) or "<span class=ax-meta>(aside)</span>"
            meta = _ids(e.get("meta"))
            meta_h = (" <span class='ax-meta'>[" + " · ".join(txt(m) for m in meta) + "]</span>") if meta else ""
            out.append(f'<div class="ax-edge"><span class="ax-prem">{prem_h}{meta_h}</span>'
                       f'<span class="ax-rel">{rel}</span>'
                       f'<span class="ax-concl">{concl_h}</span>')
            w = e.get("warrant")
            if isinstance(w, dict):
                if w.get("kind") == "missing-warrant":
                    out.append(f'<div class="ax-hole">⚠ missing warrant: {html.escape(w.get("text",""))}</div>')
                elif w.get("kind") == "citation":
                    out.append(f'<div class="ax-warr cite">warrant: citation {html.escape(str(w.get("target","")))}</div>')
                elif w.get("text"):
                    out.append(f'<div class="ax-warr">warrant: {html.escape(w["text"])}</div>')
            out.append('</div>')
        for h in g.get("holes", []):
            wanted = html.escape(str(h.get("wanted", "")).replace("-", " "))
            out.append(f'<div class="ax-hole">⚠ hole ({html.escape(str(h.get("kind","")))}): '
                       f'wants {wanted}</div>')
        out.append('</div>')
    out.append('</section>')
    return "".join(out)


def build_html(pid: str, data: dict | None = None) -> tuple[str, dict]:
    if data is None:
        src = GOLD / f"fable-{pid}-dp-emacs.json"
        data = json.loads(src.read_text(encoding="utf8"))
    text, marks = data["text"], data["marks"]
    import dp_enrich
    marks = dp_enrich.enrich(text, marks)  # DC-1/DC-2: merge the prose-concept layer
    # Coverage from the SAME (enriched) marks we render, computed in-memory — so
    # the stat panel matches the page, and we never write loss/ (no race with the
    # live mine). The enrichment only adds `concept` marks, so symbol/math/wf are
    # identical to the raw checker; term_coverage reflects the concept layer.
    import check_invariants as chk
    rep = chk.check_paper(pid, {"text": text, "marks": marks})
    cov = {"coverage": rep["coverage"],
           "violations": sum(1 for v in rep["violations"] if v["severity"] == "debt")}
    c = cov["coverage"]

    from collections import Counter
    kinds = Counter(m.get("kind") for m in marks)

    def pct(x):
        return f"{round(100 * x)}%" if isinstance(x, (int, float)) else "—"

    we = c.get("wellformed_errors")
    wf = (f'<span class="ok">CLEAN</span>' if we == 0
          else f'<span class="warn">{we} error(s)</span>' if we is not None else "—")
    stat = "".join([
        f'<div><b>{pct(c.get("math_coverage"))}</b>math spans tagged<br>({c.get("math_spans","?")} spans)</div>',
        f'<div><b>{pct(c.get("symbol_tagged"))}</b>symbols tagged<br>({c.get("symbols","?")} symbols)</div>',
        f'<div><b class="{"ok" if (c.get("symbol_grounded") or 0)>=0.8 else "warn"}">{pct(c.get("symbol_grounded"))}</b>symbols grounded</div>',
        f'<div><b class="{"ok" if (c.get("term_coverage") or 0)>=0.8 else "warn"}">{pct(c.get("term_coverage"))}</b>emph. terms noticed<br>({c.get("terms_emphasised","?")} emph.)</div>',
        f'<div><b>{wf}</b>well-formedness</div>',
        f'<div><b>{cov["violations"] if cov["violations"] is not None else "—"}</b>open debts</div>',
    ])
    legend = " ".join(f'<span class="{cls}">{html.escape(lbl)}</span>' for cls, lbl in LEGEND)
    body = render_marked_source(text, marks)
    kind_rows = " · ".join(f"{k} {n}" for k, n in kinds.most_common(10))

    doc = f"""<!doctype html>
<meta charset="utf-8">
<title>{pid} — DP anatomy (current run)</title>
<style>{STYLE}</style>
<main>
  <section class="top">
    <p class="meta"><a href="index.html">DP anatomy demo</a> / {pid}</p>
    <h1>{pid} — DP anatomy <span style="font-weight:400;font-size:15px;color:#5f5548">(current run, {len(marks)} marks)</span></h1>
    <div class="stat">{stat}</div>
    <p class="meta">top kinds: {html.escape(kind_rows)}</p>
    <p class="legend">{legend}</p>
  </section>
  <div class="paper">{body}</div>
  {render_argument_graphs(pid)}
</main>
"""
    return doc, cov


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paper")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--remine", action="store_true",
                    help="re-mine the paper in-memory with the CURRENT detector "
                         "(DC-6 split, etc.) and render that, instead of reading "
                         "the (possibly older) golden/ JSON. Never writes golden/.")
    a = ap.parse_args(argv)
    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    data = None
    if a.remine:
        import dp_paper_view as dpv
        data = dpv.build(a.paper, with_ca=True, with_binders=True,
                         with_scopes=True, with_xref=True)
    doc, cov = build_html(a.paper, data)
    p = outdir / f"{a.paper}.html"
    p.write_text(doc, encoding="utf8")
    c = cov["coverage"]
    print(f"wrote {p}  ({len(doc):,} bytes)  "
          f"grounded={c.get('symbol_grounded')} tagged={c.get('symbol_tagged')} "
          f"math={c.get('math_coverage')} wf_err={c.get('wellformed_errors')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
