#!/usr/bin/env python3
"""render_run · compositor — the engine shared by every stage layer.

Mirrors the runner: each phase contributes a Layer; the compositor UNIONS them
(never removes) onto the paper, so the presentation is additive BY CONSTRUCTION.

Two-up:
  * LEFT  = the pre-GPU GOLDEN — every CPU anatomy layer, composited via a
            sweep-line (a <span> per change of the active mark-set, carrying EVERY
            active class) and coloured in the goldens' own vocabulary (k-math /
            k-sym / k-bind / k-quant / …). Nested marks keep their detail.
  * RIGHT = that same golden PLUS the post-GPU layers (④ IATC, ⑤ expository) as
            coloured line-rails + badges. The visual diff IS what the GPU added.

`assert_additive` is a checked invariant: every CPU layer's marks must remain
active over their range in the composite, or the build fails loudly.

Golden colour vocabulary (KIND_CLASS / CSS) is reproduced from dp_anatomy_html.py
and kept here decoupled, so this renderer never edits that (live) file.
"""
from __future__ import annotations
import bisect
import html as _html
from dataclasses import dataclass, field

SKIP_KINDS = {"layout", "text-mode"}
KIND_CLASS = [
    ("symbol-grounded", "k-symg"), ("symbol", "k-sym"), ("math", "k-math"),
    ("let-binder", "k-bind"), ("bind/", "k-bind"), ("definiendum", "k-defd"),
    ("definiens", "k-defs"), ("constrain/", "k-con"), ("quant/", "k-quant"),
    ("assume/", "k-assume"), ("cite", "k-cite"), ("kw-hyp", "k-kw-hyp"),
    ("kw-con", "k-kw-con"), ("anaphor", "k-anaphor"), ("implies", "k-impl"),
    ("env/", "k-env"), ("inference", "k-inf"), ("claim", "k-claim"),
    ("concept", "k-concept"), ("classified", "k-cls"), ("unknown", "k-unk"),
    ("label", "k-label"),
]


def golden_class(kind: str):
    if kind in SKIP_KINDS:
        return None
    for key, cls in KIND_CLASS:
        if kind == key or (key.endswith("/") and kind.startswith(key)):
            return cls
    return "k-cls"


@dataclass
class Span:
    start: int
    end: int
    cls: str
    tip: str = ""


@dataclass
class Annotation:
    line: int
    sigil: str
    title: str
    body_html: str
    color: str = "#555"


@dataclass
class Layer:
    sigil: str
    name: str
    color: str
    mode: str = "inline"               # "inline" (CPU marks) | "rail" (GPU passage) | "none"
    gpu: bool = False
    spans: list[Span] = field(default_factory=list)
    annotations: list[Annotation] = field(default_factory=list)
    rows: dict = field(default_factory=dict)  # {source-line: [html]} reasoning rows injected
                                              # AFTER that line in the RIGHT pane only


def _esc(s: str) -> str:
    return _html.escape(s, quote=True)


def line_index(text: str):
    starts, off = [], 0
    for ln in text.split("\n"):
        starts.append(off)
        off += len(ln) + 1
    return starts, (lambda c: bisect.bisect_right(starts, c))


def _sweep_range(text: str, spans: list[Span], lo: int, hi: int):
    """Sweep [lo,hi) (one line). Each span is CLIPPED to the range, so a mark that
    crosses a line boundary becomes a self-contained <span> on each line it touches
    — no tag ever straddles a newline. Returns (html, coverage-fragment)."""
    pts = {lo, hi}
    for s in spans:
        if lo < s.start < hi:
            pts.add(s.start)
        if lo < s.end < hi:
            pts.add(s.end)
    bounds = sorted(pts)
    out, cov = [], {}
    for a, b in zip(bounds, bounds[1:]):
        seg = text[a:b]
        active = [s for s in spans if s.start <= a and s.end >= b]
        cls_set = frozenset(s.cls for s in active)
        cov[a] = (b, cls_set)
        if not active:
            out.append(_esc(seg))
            continue
        tips = " ".join((" | ".join(dict.fromkeys(s.tip for s in active if s.tip))).split())
        out.append(f'<span class="{" ".join(sorted(cls_set))}" title="{_esc(tips)}">{_esc(seg)}</span>')
    return "".join(out), cov


def composite_lines(text: str, spans: list[Span]):
    """Render the paper as one self-contained HTML string PER LINE (marks clipped
    to line bounds). Returns (list-of-line-html, coverage) — len == source lines."""
    starts, c2l = line_index(text)
    n = len(starts)
    line_hi = [starts[i + 1] - 1 if i + 1 < n else len(text) for i in range(n)]
    buckets: list[list[Span]] = [[] for _ in range(n)]
    for s in spans:
        if not (0 <= s.start < s.end <= len(text)):
            continue
        a, b = c2l(s.start), c2l(s.end - 1)        # 1-based first/last line touched
        for li in range(a - 1, b):
            buckets[li].append(s)
    out, coverage = [], {}
    for li in range(n):
        html_line, cov = _sweep_range(text, buckets[li], starts[li], line_hi[li])
        out.append(html_line)
        coverage.update(cov)
    return out, coverage


def assert_additive(text: str, layers: list[Layer], coverage) -> None:
    starts = sorted(coverage)

    def classes_at(c: int) -> frozenset:
        i = bisect.bisect_right(starts, c) - 1
        if i < 0:
            return frozenset()
        b, cls = coverage[starts[i]]
        return cls if c < b else frozenset()

    for L in layers:
        if L.mode != "inline":
            continue
        for s in L.spans:
            if not (0 <= s.start < s.end <= len(text)):
                continue
            probe = s.start                       # first non-newline char of the span
            while probe < s.end and text[probe] == "\n":
                probe += 1
            if probe >= s.end:
                continue
            if s.cls not in classes_at(probe):
                raise AssertionError(
                    f"ADDITIVITY VIOLATED: layer {L.sigil} {L.name} span "
                    f"{s.cls}[{s.start}:{s.end}] missing at char {probe}.")


_CSS = """
*{box-sizing:border-box}
body{margin:0;background:#f7f5ef;color:#1a1a1a;font:13px/1.5 ui-sans-serif,system-ui}
header{padding:10px 16px;border-bottom:1px solid #ddd6c8;background:#fff}
h1{font-size:16px;margin:0 0 2px} .sub{color:#666;font-size:11.5px}
.legend{margin-top:5px;font-size:11px;display:flex;flex-wrap:wrap;gap:5px 12px}
.lg{white-space:nowrap}
.cb{display:inline-block;color:#fff;font:700 9.5px/1 ui-sans-serif;padding:2px 5px;border-radius:3px;margin-right:3px}
.idx{margin-top:3px;font-size:11.5px}
.wrap{display:grid;grid-template-columns:1fr 1fr 380px;height:calc(100vh - 84px)}
.colhead{position:sticky;top:0;z-index:2;background:#f1ede2;border-bottom:1px solid #ddd6c8;
  font:700 11px/1.6 ui-sans-serif;padding:3px 10px;color:#555}
.pane{overflow:auto;padding:6px 10px;position:relative;
  font:12px/1.5 "SF Mono",ui-monospace,Menlo,monospace;border-right:1px solid #ddd6c8;background:#fffdf8}
.pane.orig{background:#fbfaf5}
.ln{display:flex;align-items:baseline;gap:8px;white-space:pre-wrap}
.ln:target{outline:2px solid #f0c000}
.lno{color:#c2b9a7;min-width:34px;text-align:right;user-select:none;flex:0 0 auto}
.lt{flex:1 1 auto;white-space:pre-wrap;word-break:break-word}
.gb{display:inline-block;color:#fff;font:700 10px/1 ui-sans-serif;padding:2px 5px;border-radius:3px;margin-left:5px}
/* GPU passage rails (right pane) */
.rail-iatc{background:rgba(170,17,17,.045)} .rail-expo{background:rgba(109,58,168,.05)}
/* GPU reasoning rows injected beneath a source line (right pane) */
.rrow{font:11px/1.45 ui-sans-serif,system-ui;padding:2px 6px 2px 44px;margin:1px 0}
.rrow .sg{font-weight:700;margin-right:6px}
.iatc-row{border-left:3px solid #a11;background:#fdf6f6}
.expo-row{border-left:3px solid #6d3aa8;background:#f8f4fc}
.ir-edge{display:inline-flex;gap:4px;align-items:baseline;flex-wrap:wrap;margin-right:14px}
.ir-node{padding:0 4px;border-radius:3px;background:#eef0f3}
.ir-prem{background:#eef3fb} .ir-concl{background:#fdeef0;font-weight:600}
.ir-rel{color:#c026d3;font-weight:700;font-variant:small-caps}
.ir-arrow{color:#c026d3;font-weight:700}
.ir-warr{color:#15803d;font-size:10px} .ir-hole{color:#a11;font-style:italic;font-size:10px}
.expo-row code{background:#efe7fa}
/* margin annotation cards, pinned to their line */
.margin{overflow:auto;padding:6px 10px;position:relative;background:#faf8f2}
#marginInner{position:relative}
.card{position:absolute;left:8px;right:8px;border:1px solid #e3ddcf;border-radius:6px;
  background:#fff;padding:0 10px 9px;box-shadow:0 1px 3px rgba(0,0,0,.06)}
.cline{position:absolute;right:8px;top:7px;font:700 9.5px/1 ui-sans-serif;color:#bbb}
.chead{display:block;text-decoration:none;color:#1a1a1a;font-weight:600;font-size:12px;
  border-left:4px solid;padding:7px 8px;margin:0 -10px 6px;background:#fcfbf7}
.cbody .fact{font-size:11px;color:#444;margin:3px 0;font-family:"SF Mono",ui-monospace,monospace}
.cbody .edge{font-size:11px;margin:2px 0;padding-left:7px;border-left:2px solid #eee}
.cbody .verdict{font-size:12px;margin-top:6px}
code{background:#ece8dd;padding:1px 4px;border-radius:3px;font-size:11px}
/* ---- golden anatomy vocabulary (from dp_anatomy_html.py) ---- */
.k-math{background:#eef3fb;border-radius:2px}
.k-symg{color:#0b5e57;border-bottom:2px solid #0f766e;font-weight:600}
.k-sym{background:#fdf3d7;border-bottom:2px dashed #c2410c;color:#8a3d05;font-weight:600}
.k-bind{border-bottom:2px solid #2a4d9a}
.k-defd{background:#efe3fb;color:#6b21a8;font-weight:700;border-radius:2px}
.k-defs{color:#6b21a8;font-style:italic}
.k-concept{background:#e9f7ef;border-bottom:2px dotted #15803d;color:#166534}
.k-impl{background:#faf7ff;text-decoration:underline;text-decoration-color:#7c6bd6}
.k-kw-hyp{font-weight:700;color:#1d4ed8} .k-kw-con{font-weight:700;color:#6d28d9}
.k-claim{border-bottom:1px dotted #9a8fb0}
.k-anaphor{border-bottom:1px dotted #1d4ed8;color:#1d4ed8}
.k-inf{color:#c026d3;font-weight:700}
.k-con{border-bottom:1px dotted #9a3a3a}
.k-quant{border-bottom:1px solid #4a6b2f;color:#3f5d28}
.k-assume{border-bottom:1px solid #4a6b2f}
.k-cite{color:#9a8f80} .k-env{background:#fff4d8;border-radius:2px}
.k-cls{border-bottom:1px solid #c8bca6} .k-unk{background:#ffe2e2;border-bottom:2px dotted #b91c1c}
.k-label{color:#b9b0a0}
"""

_JS = """
const L=document.getElementById('paneL'),R=document.getElementById('paneR'),
      M=document.getElementById('paneM'),MI=document.getElementById('marginInner');
function layoutCards(){
  const cards=[...MI.querySelectorAll('.card')].sort((a,b)=>(+a.dataset.line)-(+b.dataset.line));
  let y=0;
  for(const c of cards){
    const ln=document.getElementById('R'+c.dataset.line);
    let top=Math.max(ln?ln.offsetTop:0, y);
    c.style.top=top+'px'; y=top+c.offsetHeight+8;
  }
  MI.style.height=Math.max(y, R.scrollHeight)+'px';
}
function lineRef(p){
  const ls=p.getElementsByClassName('ln');let lo=0,hi=ls.length-1,a=0,t=p.scrollTop;
  while(lo<=hi){const m=(lo+hi)>>1; if(ls[m].offsetTop<=t){a=m;lo=m+1;}else hi=m-1;}
  const el=ls[a]; return {i:a, f:(t-el.offsetTop)/(el.offsetHeight||1)};
}
function targetTop(dst,r){const ls=dst.getElementsByClassName('ln');
  const el=ls[Math.min(r.i,ls.length-1)]; return el.offsetTop+r.f*(el.offsetHeight||1);}
// Echo-swallow sync: the user's pane is the sole driver; each programmatic
// scroll is tagged and the single scroll event it emits is ignored, so there is
// no feedback ping-pong (the jitter).
const prog=new Set();
function setTop(p,top){ top=Math.round(top); if(p.scrollTop===top) return; prog.add(p); p.scrollTop=top; }
function onScroll(src){
  if(prog.has(src)){ prog.delete(src); return; }   // swallow our own programmatic echo
  if(src===M){ setTop(R, M.scrollTop); setTop(L, targetTop(L, lineRef(R))); return; }
  const r=lineRef(src);
  if(src!==L) setTop(L, targetTop(L,r));
  if(src!==R) setTop(R, targetTop(R,r));
  setTop(M, R.scrollTop);
}
L.addEventListener('scroll',()=>onScroll(L),{passive:true});
R.addEventListener('scroll',()=>onScroll(R),{passive:true});
M.addEventListener('scroll',()=>onScroll(M),{passive:true});
document.querySelectorAll('[data-line]').forEach(el=>el.addEventListener('click',e=>{
  e.preventDefault(); const n=el.getAttribute('data-line');
  const t=document.getElementById('R'+n); if(t){ setTop(R, Math.max(0,t.offsetTop-60));
    setTop(L, targetTop(L, lineRef(R))); setTop(M, R.scrollTop); }
}));
window.addEventListener('load',layoutCards);
"""


def render_two_up(paper_id: str, text: str, layers: list[Layer]) -> str:
    starts, c2l = line_index(text)
    cpu = [L for L in layers if L.mode == "inline" and not L.gpu]
    gpu = [L for L in layers if L.gpu]

    cpu_spans = [s for L in cpu for s in L.spans]
    comp_lines, coverage = composite_lines(text, cpu_spans)   # marks clipped per line
    assert_additive(text, cpu, coverage)         # GUARANTEE (CPU layers)
    # invariant: one rendered line per source line, or every rail/anchor shifts.
    n_src = text.count("\n") + 1
    if len(comp_lines) != n_src:
        raise AssertionError(f"LINE COUNT DRIFT: composite has {len(comp_lines)} lines, "
                             f"source has {n_src} — anchors would shift.")

    # GPU passage rails + badge (right pane only)
    rails: dict[int, list] = {}
    for Lg in gpu:
        rcls = "rail-iatc" if "IATC" in Lg.name else "rail-expo"
        for s in Lg.spans:
            l0, l1 = c2l(s.start), c2l(max(s.start, s.end - 1))
            for ln in range(l0, l1 + 1):
                rails.setdefault(ln, []).append((Lg.color, Lg.sigil if ln == l0 else "", rcls, s.tip))
    # additivity for GPU: each gpu layer's first line must carry a rail
    for Lg in gpu:
        for s in Lg.spans:
            if c2l(s.start) not in rails:
                raise AssertionError(f"GPU layer {Lg.sigil} dropped: no rail at L{c2l(s.start)}")

    line_rows: dict[int, list] = {}            # GPU reasoning rows injected after a line (right pane)
    for Lg in gpu:
        for ln, htmls in Lg.rows.items():
            line_rows.setdefault(ln, []).extend(htmls)

    def pane(prefix: str, with_gpu: bool) -> str:
        out = []
        for i, lh in enumerate(comp_lines, start=1):
            cls, style, badge = "ln", "", ""
            if with_gpu and i in rails:
                rcls = " ".join(dict.fromkeys(r[2] for r in rails[i]))
                cls = f"ln {rcls}"
                cols = list(dict.fromkeys(r[0] for r in rails[i]))
                style = ' style="box-shadow:' + ",".join(
                    f"inset {3 + 4 * j}px 0 0 {c}" for j, c in enumerate(cols)) + '"'
                badge = "".join(f'<span class="gb" style="background:{c}">{sg}</span>'
                                for c, sg, _r, _t in rails[i] if sg)
            out.append(f'<div class="{cls}" id="{prefix}{i}"{style}>'
                       f'<span class="lno">{i}</span><span class="lt">{lh or "&nbsp;"}</span>{badge}</div>')
            if with_gpu and i in line_rows:       # reasoning rows beneath the line
                out.extend(line_rows[i])
        return "".join(out)

    left, right = pane("L", False), pane("R", True)

    anns = sorted((a for Lr in layers for a in Lr.annotations), key=lambda a: a.line)
    cards = "".join(
        f'<div class="card" data-line="{a.line}"><span class="cline">L{a.line}</span>'
        f'<a class="chead" href="#R{a.line}" data-line="{a.line}" style="border-left-color:{a.color}">'
        f'<span class="cb" style="background:{a.color}">{a.sigil}</span> {a.title}</a>'
        f'<div class="cbody">{a.body_html}</div></div>'
        for a in anns)
    idx = " · ".join(f'<a href="#R{a.line}" data-line="{a.line}" style="color:{a.color}">{a.sigil} L{a.line}</a>'
                     for a in anns if a.line > 1)
    legend = "".join(
        f'<span class="lg"><span class="cb" style="background:{Lr.color}">{Lr.sigil}</span>{Lr.name}'
        f'{" (corpus-scale)" if Lr.mode == "none" else ""}</span>' for Lr in layers)

    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<title>render_run · {paper_id}</title><style>{_CSS}</style></head><body>'
            f'<header><h1>render_run — paper {paper_id} '
            f'<span style="font-weight:400;color:#999;font-size:11px">· additive all-stage composite '
            f'(✓ assert_additive)</span></h1>'
            f'<div class="sub"><b>left</b> = pre-GPU golden (all CPU anatomy layers) · '
            f'<b>right</b> = same golden + GPU stage rails. The visual diff is what the GPU added.</div>'
            f'<div class="legend">{legend}</div><div class="idx">GPU passages: {idx}</div></header>'
            f'<div class="wrap"><div class="pane orig" id="paneL">{left}</div>'
            f'<div class="pane" id="paneR">{right}</div>'
            f'<div class="margin" id="paneM"><div id="marginInner">{cards}</div></div></div>'
            f'<script>{_JS}</script></body></html>')
