#!/usr/bin/env python3
# mission_hypergraph_demo.py — the CHEAP CONFIRM (Joe, 2026-06-08), v2: NESTED SCOPES.
#
# v1 flattened each phase to a word-list, losing the tree. The scope-tree actually nests:
# phase (eightfold) -> sub-scopes (capability-scope / map-item / relates-to / ...) -> concepts,
# 2-3 deep. This renders that nesting as Scratch-style nested blocks (scopes ARE the tangible
# thing), and DEDUPS the binder-duplication (the detector emits one heading as several binder-
# types — e.g. "Q5" as capability-scope AND map-item; merged by title here).
#
# Still a cheap static HTML confirm — "are we capturing structure worth thinking about?" —
# before any representation commitment (node-link / Scratch / ChipWits).
import glob, html
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from mission_fold import load_sip, load_tree, build, top_sip  # noqa: E402

ROOT = Path(__file__).parent.parent
OUT = ROOT / "data" / "mission-hypergraph-demo.html"
OUTWARD = ("relates-to", "source-material", "mission-scope-out", "mission-scope-in")
MAX_CHILDREN = 9

DEMO = ["M-agency-forum", "M-agency-rebuild", "M-war-machine", "M-war-machine-tuning",
        "M-capability-star-map", "M-web-arxana-missions", "M-self-documenting-stack",
        "M-canon-fingerprint-store", "M-symbol-grounding", "M-bayesian-structure-learning",
        "M-differentiable-code", "M-aif2", "M-weird-modernism"]


def render_level(nodes, sip, node_ids, depth):
    """node_ids = sibling scope-instances that share a title (merged: dedups the
    binder-duplication). Renders this scope's concepts + recurses on its children,
    themselves grouped-by-title."""
    own = [f for nid in node_ids for f in nodes[nid]["fillers"]]
    title = nodes[node_ids[0]]["title"]
    binders = sorted({nodes[nid]["binder"] for nid in node_ids})
    outward = any(b in OUTWARD for b in binders)
    chips = "".join(f'<span class="chip">{html.escape(c)}</span>'
                    for c in top_sip(own, sip, 7))
    badges = "".join(f'<span class="b b-{b}">{b}</span>' for b in binders)

    child_ids = [c for nid in node_ids for c in nodes[nid]["children"]]
    groups = {}
    for c in child_ids:
        groups.setdefault(nodes[c]["title"].strip().lower(), []).append(c)
    ordered = sorted(groups.values(), key=lambda ids: -sum(nodes[i]["sub_mass"] for i in ids))
    inner = "".join(render_level(nodes, sip, ids, depth + 1) for ids in ordered[:MAX_CHILDREN])
    if len(ordered) > MAX_CHILDREN:
        inner += f'<div class="more">+{len(ordered)-MAX_CHILDREN} more sub-scopes</div>'

    arrow = "→ " if outward else ""
    cls = "scope outward" if outward else "scope"
    return (f'<div class="{cls} d{min(depth,4)}">'
            f'<div class="stitle">{arrow}{html.escape(title[:54])} {badges}</div>'
            f'{("<div class=chips>"+chips+"</div>") if chips else ""}{inner}</div>')


def mission_block(stem, sip):
    try:
        tree, _ = load_tree(stem)
    except Exception:
        return None
    nodes, roots = build(tree, sip)
    raw = len(nodes) + sum(len(n["fillers"]) for n in nodes.values())
    phase_ids = [c for r in roots for c in nodes[r]["children"]]
    groups = {}
    for c in phase_ids:
        groups.setdefault(nodes[c]["title"].strip().lower(), []).append(c)
    ordered = sorted(groups.values(), key=lambda ids: -sum(nodes[i]["sub_mass"] for i in ids))
    phases = "".join(render_level(nodes, sip, ids, 0) for ids in ordered)
    rootcon = "".join(f'<span class="chip">{html.escape(c)}</span>'
                      for c in top_sip([f for r in roots for f in nodes[r]["fillers"]], sip, 7))
    return (f'<section class="mission"><h2>{html.escape(stem)}</h2>'
            f'<div class="meta">raw {raw} nodes · {len(ordered)} phases</div>'
            f'{("<div class=chips>"+rootcon+"</div>") if rootcon else ""}{phases}</section>')


def main():
    sip = load_sip()
    have = {Path(f).stem for f in glob.glob(str(ROOT / "data/mission-scope-trees/*.json"))}
    blocks, shown = [], []
    for stem in DEMO:
        if stem in have and (b := mission_block(stem, sip)):
            blocks.append(b); shown.append(stem)
    css = """
    body{font:14px/1.5 -apple-system,Segoe UI,sans-serif;margin:0;background:#0f1115;color:#d8dee9}
    header{padding:18px 26px;background:#161922;border-bottom:1px solid #2a2f3a}
    header h1{margin:0 0 5px;font-size:17px}header p{margin:0;color:#8b95a7;font-size:12.5px}
    .wrap{padding:18px 24px;display:grid;gap:16px;grid-template-columns:repeat(auto-fill,minmax(440px,1fr));align-items:start}
    .mission{background:#161922;border:1px solid #2a2f3a;border-radius:10px;padding:13px 15px}
    .mission h2{margin:0;font-size:15px;color:#88c0d0}
    .meta{color:#6b7280;font-size:11.5px;margin:2px 0 8px}
    .scope{margin:5px 0 5px 0;padding:5px 8px;border-left:3px solid #5e81ac;border-radius:5px;background:#1b1f2a}
    .scope.d1{margin-left:13px;border-left-color:#a3be8c;background:#1a1e27}
    .scope.d2{margin-left:13px;border-left-color:#ebcb8b;background:#191d25}
    .scope.d3,.scope.d4{margin-left:13px;border-left-color:#b48ead;background:#181b23}
    .scope.outward{border-left-style:dashed;border-left-color:#bf616a}
    .stitle{font-weight:600;color:#e5e9f0;font-size:12.5px}
    .chips{margin:3px 0 1px}
    .chip{display:inline-block;background:#2a2f3a;color:#a3be8c;border-radius:4px;padding:0 6px;margin:2px 3px 0 0;font-size:11px}
    .b{display:inline-block;border-radius:3px;padding:0 4px;margin-left:4px;font-size:9.5px;font-weight:400;vertical-align:middle}
    .b-eightfold-phase{background:#3b4a63;color:#9cc}.b-capability-scope{background:#3a4a38;color:#a3be8c}
    .b-map-item{background:#4a432f;color:#ebcb8b}.b-relates-to{background:#4a3540;color:#d8a3b4}
    .b-source-material{background:#2f4a47;color:#8fbcbb}.b-loose-section{background:#33384a;color:#aab}
    .b-mission-scope-in,.b-mission-scope-out{background:#4a4630;color:#ebcb8b}
    .more{color:#6b7280;font-size:11px;margin:3px 0 0 13px;font-style:italic}
    """
    doc = f"""<!doctype html><meta charset=utf-8><title>Mission hypergraphs — nested scopes</title>
<style>{css}</style>
<header><h1>Mission hypergraphs — nested scopes (cheap confirm v2)</h1>
<p>{len(shown)} missions. Phase → sub-scopes → concepts, as nested blocks; binder-types badged; outward
(relates-to / source) edges dashed-red with →. Binder-duplication deduped by title. Confirm the STRUCTURE
before any representation choice.</p></header><div class="wrap">{''.join(blocks)}</div>"""
    OUT.write_text(doc)
    print(f"wrote {OUT}  ({len(shown)} missions)")


if __name__ == "__main__":
    main()
