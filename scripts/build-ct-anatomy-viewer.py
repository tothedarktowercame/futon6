#!/usr/bin/env python3
"""Build the math.CT paper anatomy showcase — sibling of proof-anatomy.

Outputs:
  - data/showcases/ct-anatomy/index.html
  - data/showcases/ct-anatomy/<paper>.html   (the 30 audited sample papers)

Data source: the superpod CT handoff (storage/mark2/ct-handoff), via the
per-paper slices made by ct_anatomy_slice.py from the 19G scopes.json.
Visual language follows build-proof-anatomy-viewer.py (depth gradients,
binder chips) so a prelim proof and a CT paper read as two specimens in
one anatomy atlas. Scope nesting here is POSITIONAL (hx/parent is null in
the corpus): a scope is a child of the tightest interval containing it.

The page foregrounds the BINDER SKELETON (bind/*, constrain/*, comment/*)
and reports the fine-grain math/* layer as a mix table — a 6k-scope paper
must read as anatomy, not as a haystack.
"""
from __future__ import annotations

import html
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HANDOFF = Path("/home/joe/code/storage/mark2/ct-handoff")
SLICES = HANDOFF / "ct-anatomy-slices"
OUTPUT = HANDOFF / "output"
OUT_DIR = ROOT / "data" / "showcases" / "ct-anatomy"

SKELETON_PREFIXES = ("bind/", "constrain/", "comment/")
DEPTH_MAX = 5


def esc(v) -> str:
    return html.escape(str(v), quote=True)


def binder_css(hx_type: str) -> str:
    return hx_type.replace("/", "-").replace(".", "-")


STYLE = """
body { margin:0; font:14px/1.5 system-ui,sans-serif; background:#0b0e14; color:#d8dee9; }
header { padding:18px 26px; background:#11151f; border-bottom:1px solid #232a38; }
h1 { font-size:19px; margin:0 0 6px; } h2 { font-size:15px; margin:18px 0 8px; }
p.lede { color:#8b95a7; max-width:1100px; margin:0; }
main { padding:18px 26px; max-width:1280px; }
table { border-collapse:collapse; margin:8px 0; } td,th { padding:4px 10px; border-bottom:1px solid #1d2433; text-align:left; }
a { color:#7fb4ff; text-decoration:none; } a:hover { text-decoration:underline; }
.chip { display:inline-block; padding:1px 7px; border-radius:9px; font-size:11px; margin:1px 2px; background:#1d2433; }
.chip.bind-typed,.chip.bind-let { background:#234d32; color:#9fe2b5; }
.chip.constrain-such-that,.chip.constrain-relation { background:#3a2d4d; color:#cdb6ef; }
.chip.comment-unreachable { background:#4d2323; color:#efb6b6; }
.scope { display:block; margin:3px 0 3px 14px; padding:4px 8px; border-radius:6px; }
.scope .match { font-family:ui-monospace,monospace; font-size:12px; color:#e9eef7; }
.scope .meta { font-size:11px; color:#7d8799; }
.scope.depth-1 { background:linear-gradient(90deg,#3d2f16,#2a2113); }
.scope.depth-2 { background:linear-gradient(90deg,#3d1630,#2a1322); }
.scope.depth-3 { background:linear-gradient(90deg,#251d45,#1b1733); }
.scope.depth-4 { background:linear-gradient(90deg,#16304d,#13233a); }
.scope.depth-5 { background:linear-gradient(90deg,#163d3a,#132a29); outline:1px dashed #3c5350; }
.sparkbar { display:inline-block; width:9px; background:#5a8bd0; margin-right:1px; vertical-align:bottom; }
.panel { background:#11151f; border:1px solid #232a38; border-radius:8px; padding:10px 14px; margin:10px 0; }
.kv b { color:#e9eef7; }
"""


def build_tree(skel: list[dict]) -> list[dict]:
    """Positional nesting: child of the tightest containing interval."""
    spans = []
    for s in skel:
        c = s.get("hx/content") or {}
        pos, end = c.get("position"), c.get("end")
        if pos is None or end is None or end <= pos:
            continue
        spans.append({"s": s, "pos": pos, "end": end, "children": [], "depth": 1})
    spans.sort(key=lambda n: (n["pos"], -(n["end"] - n["pos"])))
    stack: list[dict] = []
    roots: list[dict] = []
    for node in spans:
        while stack and not (stack[-1]["pos"] <= node["pos"] and node["end"] <= stack[-1]["end"]):
            stack.pop()
        if stack:
            node["depth"] = min(DEPTH_MAX, stack[-1]["depth"] + 1)
            stack[-1]["children"].append(node)
        else:
            roots.append(node)
        stack.append(node)
    return roots


def render_node(node: dict) -> str:
    s = node["s"]
    hxt = s.get("hx/type", "?")
    match = (s.get("hx/content") or {}).get("match", "")[:160]
    ends = s.get("hx/ends") or []
    sym = next((e.get("latex") for e in ends if e.get("role") == "symbol"), None)
    typ = next((e.get("text") for e in ends if e.get("role") == "type"), None)
    bits = f'<span class="chip {binder_css(hxt)}">{esc(hxt)}</span>'
    if sym:
        bits += f' <span class="meta">${esc(sym)}$</span>'
    if typ:
        bits += f' <span class="meta">: {esc(typ[:60])}</span>'
    kids = "".join(render_node(c) for c in node["children"])
    return (f'<div class="scope depth-{node["depth"]}">{bits}'
            f'<div class="match">{esc(match)}</div>{kids}</div>')


def spark(depths: dict) -> str:
    if not depths:
        return '<span class="meta">—</span>'
    mx = max(depths.values())
    return "".join(
        f'<span class="sparkbar" style="height:{3 + 14 * depths.get(str(d), 0) / mx:.0f}px" title="depth {d}: {depths.get(str(d), 0)}"></span>'
        for d in range(1, 8))


def main() -> None:
    import sys
    all_mode = "--all" in sys.argv
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit = json.load(open(OUTPUT / "audit-summary.json"))
    audited = {p["entity_id"]: p for p in audit["papers"]}
    if all_mode:
        papers = []
        for sl in sorted(SLICES.glob("*.json")):
            if sl.name == "_meta.json":
                continue
            eid = sl.stem.replace("_", "/", 1) if sl.stem.startswith("arxiv-math_") else sl.stem
            papers.append(eid)
        audit = {"sample_size": len(papers),
                 "papers": [audited.get(e, {"entity_id": e, "total": None,
                                            "inhabited": None, "outer": None,
                                            "straddled": None,
                                            "depth_distribution": {}})
                            for e in papers]}
    stats = json.load(open(OUTPUT / "stats.json"))
    repair = json.load(open(OUTPUT / "mparts-repair-receipt.json"))
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%MZ")
    rows = []
    for p in sorted(audit["papers"], key=lambda p: -(p["total"] or 0)):
        eid = p["entity_id"]
        slice_path = SLICES / (eid.replace("/", "_") + ".json")
        if not slice_path.exists():
            continue
        rec = json.loads(slice_path.read_text())
        scopes = rec.get("scopes", [])
        if p["total"] is None:
            p = dict(p, total=len(scopes), inhabited="·", outer="·", straddled="·")
        mix = Counter(s.get("hx/type", "?") for s in scopes)
        skel = [s for s in scopes if str(s.get("hx/type", "")).startswith(SKELETON_PREFIXES)]
        roots = build_tree(skel)
        page = OUT_DIR / (eid.replace("/", "_") + ".html")
        arxiv_id = eid.removeprefix("arxiv-")
        skel_html = "".join(render_node(r) for r in roots) or '<p class="meta">no binder-skeleton scopes</p>'
        mix_rows = "".join(f"<tr><td><span class='chip {binder_css(k)}'>{esc(k)}</span></td><td>{v}</td></tr>"
                           for k, v in mix.most_common())
        page.write_text(f"""<!doctype html><meta charset=utf-8><title>{esc(eid)} — CT anatomy</title>
<style>{STYLE}</style>
<header><a href="index.html">← math.CT anatomy index</a>
<h1>{esc(eid)}</h1>
<p class="lede kv"><b>{p['total']}</b> scopes · <b>{p['inhabited']}</b> inhabited · <b>{p['outer']}</b> outer ·
<b>{p['straddled']}</b> straddled · depth {spark(p.get('depth_distribution', {}))} ·
<a href="https://arxiv.org/abs/{esc(arxiv_id)}">arXiv:{esc(arxiv_id)}</a></p></header>
<main>
<div class="panel"><h2>Binder skeleton ({len(skel)} scopes: bind/* · constrain/* · comment/*)</h2>{skel_html}</div>
<div class="panel"><h2>Scope mix (all {p['total']})</h2><table>{mix_rows}</table></div>
</main>""")
        top_binds = " ".join(f'<span class="chip {binder_css(k)}">{esc(k)} {v}</span>'
                             for k, v in mix.most_common(20) if k.startswith(SKELETON_PREFIXES))
        rows.append(f"<tr><td><a href='{page.name}'>{esc(eid)}</a></td><td>{p['total']}</td>"
                    f"<td>{p['inhabited']}</td><td>{p['outer']}</td><td>{p['straddled']}</td>"
                    f"<td>{spark(p.get('depth_distribution', {}))}</td><td>{top_binds or '—'}</td></tr>")

    (OUT_DIR / "index.html").write_text(f"""<!doctype html><meta charset=utf-8><title>math.CT — paper anatomy</title>
<style>{STYLE}</style>
<header><h1>math.CT Paper Anatomy — real scopes on real papers</h1>
<p class="lede">The superpod CT run over <b>all of math.CT ({stats.get('qa_pairs', '?')} papers)</b>;
Stage-3 repair parse rate <b>{repair['stage3']['parse_rate']:.2%}</b> ({repair['stage3']['parsed_ok']}/{repair['stage3']['reparsed']},
clipped-JSON salvage per futon6 PR #49). This sample: the run's own <b>{audit['sample_size']}-paper scope audit</b>,
each rendered as a binder skeleton (bind/*, constrain/*, comment/*) over its positional scope nesting.
Sibling atlas: <a href="../proof-anatomy/index.html">First Proof anatomy</a> — same visual grammar,
prelim proofs beside CT papers. Generated {generated}.</p></header>
<main><table><thead><tr><th>paper</th><th>scopes</th><th>inhabited</th><th>outer</th><th>straddled</th>
<th>depth</th><th>binder skeleton (top)</th></tr></thead><tbody>{''.join(rows)}</tbody></table></main>""")
    print(f"wrote {OUT_DIR}/index.html + {len(rows)} paper pages")


if __name__ == "__main__":
    main()
