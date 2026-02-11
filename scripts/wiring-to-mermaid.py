#!/usr/bin/env python3
"""Convert a proof wiring diagram JSON to Mermaid flowchart.

Usage:
    python3 scripts/wiring-to-mermaid.py data/first-proof/problem10-wiring.json
    python3 scripts/wiring-to-mermaid.py data/first-proof/problem10-wiring.json -o data/first-proof/problem10-v1.mmd
"""

import argparse
import json
import sys
from pathlib import Path


# Edge type → Mermaid arrow style + label color hint
EDGE_STYLES = {
    "assert":    ("==>",  "🟢"),
    "challenge": ("-.->", "🔴"),
    "reform":    ("-->",  "🟡"),
    "clarify":   ("-->",  "🔵"),
    "exemplify": ("-.->", "🟣"),
    "reference": ("-->",  "⚪"),
    "agree":     ("==>",  "🟢"),
    "query":     ("-.->", "🟠"),
    "retract":   ("-.->", "⚫"),
}

# Node type → shape
NODE_SHAPES = {
    "question": ('{{', '}}'),   # hexagon
    "answer":   ('[',  ']'),    # rectangle
    "comment":  ('([', '])'),   # stadium
}


def truncate(text: str, maxlen: int = 50) -> str:
    text = text.replace('"', "'").replace('\n', ' ')
    if len(text) > maxlen:
        return text[:maxlen-3] + "..."
    return text


def wiring_to_mermaid(wiring: dict, title: str | None = None) -> str:
    lines = []
    lines.append("flowchart TD")

    if title:
        lines.append(f"    %% {title}")
    lines.append("")

    # Class definitions for edge types
    lines.append("    %% Edge type legend:")
    lines.append("    %% assert=solid thick  challenge=dashed red  reform=solid yellow")
    lines.append("    %% clarify=solid blue  exemplify=dashed purple  reference=solid gray")
    lines.append("")

    # Nodes
    lines.append("    %% Nodes")
    for node in wiring["nodes"]:
        nid = node["id"]
        label = truncate(node["body_text"], 55)
        ntype = node["node_type"]
        l, r = NODE_SHAPES.get(ntype, ('[', ']'))
        lines.append(f'    {nid}{l}"{nid}<br/>{label}"{r}')

    lines.append("")

    # Edges
    lines.append("    %% Edges")
    for edge in wiring["edges"]:
        src = edge["source"]
        tgt = edge["target"]
        etype = edge["edge_type"]
        evidence = truncate(edge["evidence"], 40)
        arrow, icon = EDGE_STYLES.get(etype, ("-->", ""))

        lines.append(f'    {src} {arrow}|"{icon} {etype}: {evidence}"| {tgt}')

    lines.append("")

    # Style classes based on node type
    lines.append("    %% Styling")
    questions = [n["id"] for n in wiring["nodes"] if n["node_type"] == "question"]
    answers = [n["id"] for n in wiring["nodes"] if n["node_type"] == "answer"]
    comments = [n["id"] for n in wiring["nodes"] if n["node_type"] == "comment"]

    if questions:
        lines.append(f"    classDef question fill:#f9e,stroke:#c06,stroke-width:3px")
        lines.append(f"    class {','.join(questions)} question")
    if answers:
        lines.append(f"    classDef answer fill:#cef,stroke:#38b,stroke-width:2px")
        lines.append(f"    class {','.join(answers)} answer")
    if comments:
        lines.append(f"    classDef comment fill:#eee,stroke:#999,stroke-width:1px")
        lines.append(f"    class {','.join(comments)} comment")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path, help="Wiring diagram JSON")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="Output .mmd file (default: stdout)")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    wiring = json.loads(args.input.read_text())

    title = args.title or wiring.get("thread_id", "")
    mmd = wiring_to_mermaid(wiring, title)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(mmd)
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(mmd)


if __name__ == "__main__":
    main()
