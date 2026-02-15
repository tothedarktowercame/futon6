#!/usr/bin/env python3
"""Generate a Graphviz DOT diagram for sprint-review wiring JSON.

Usage:
    python3 scripts/sprint-review-wiring-to-graphviz.py \
        data/first-proof/sprint-review-wiring.json \
        -o data/first-proof/latex/plates/sprint-review-wiring-graphviz.dot \
        --pdf data/first-proof/latex/plates/sprint-review-wiring-graphviz.pdf
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


NODE_COLORS = {
    "data": "#d7dcff",
    "method": "#efe4ff",
    "evidence": "#d5f0d8",
    "synthesis": "#f4f0bf",
    "challenge": "#f8dede",
    "prescription": "#f7dfc6",
    "outcome": "#c6f5ca",
}


EDGE_STYLES = {
    "assert": {"style": "solid", "color": '"#333333"', "penwidth": "2.0"},
    "classify": {"style": "dashed", "color": '"#333333"', "penwidth": "1.7"},
    "exemplify": {"style": "dotted", "color": '"#333333"', "penwidth": "2.0"},
    "reference": {"style": "dotted", "color": '"#8a8a8a"', "penwidth": "1.5", "constraint": "false"},
    "challenge": {"style": "dashed", "color": '"#e88a8a"', "penwidth": "2.2", "constraint": "false"},
}


# Compact labels used in Figure 1 for an apples-to-apples layout comparison.
DISPLAY_LABELS = {
    "ret-D0": "Sprint data\\n341 commits, 55 h",
    "ret-D1": "Official solutions\\nGround truth",
    "ret-D2": "Scorecard\\n4 / 4 / 2",
    "ret-M0": "Infrastructure\\nWiring + review",
    "ret-M1": "Methodology\\nTypes + layer switch",
    "ret-TA": "Tier A\\nP2, 8, 9, 10",
    "ret-TB": "Tier B\\nP3, 4, 5, 6",
    "ret-TC": "Tier C\\nP1, 7",
    "ret-S1": "Structural\\ndecomposition",
    "ret-S2": "Computation\\nguides proof",
    "ret-S3": "Cycle\\nmethodology",
    "ret-S4": "Protocol\\n> baselines",
    "ret-F1": "YES bias",
    "ret-F2": "False-premise\\ntrap",
    "ret-F3": "Conditional\\ngap",
    "ret-F4": "Domain-depth\\ngap",
    "ret-P1": "Coaching >\\ndispatching",
    "ret-P2": "Negative\\npheromone",
    "ret-N1": "P4 near miss\\nendpoint fixation",
    "ret-N2": "P6 near miss\\nobject inertia",
    "ret-V1": "Non-blinded\\nevaluation",
    "ret-V2": "Baseline\\ncomparability",
    "ret-V3": "Attribution\\nuncertainty",
    "ret-R1": "Mandatory\\nfalsification",
    "ret-R2": "Domain-depth\\naudit",
    "ret-R3": "State-variable\\nhop",
    "ret-C0": "Epistemic weakness\\nFalse premises invisible",
    "ret-C1": "thesis\\nFrontier = falsification depth",
}


RANK_ROWS = [
    ["ret-D0", "ret-D1"],
    ["ret-M0", "ret-D2"],
    ["ret-M1", "ret-TA", "ret-TB"],
    ["ret-TC", "ret-S1", "ret-S2"],
    ["ret-S3", "ret-F1", "ret-F2"],
    ["ret-F3", "ret-V1", "ret-S4"],
    ["ret-P1", "ret-P2", "ret-F4"],
    ["ret-N1", "ret-N2", "ret-V2"],
    ["ret-R1", "ret-R2", "ret-R3"],
    ["ret-V3", "ret-C0"],
    ["ret-C1"],
]


def dot_escape(text: str) -> str:
    return text.replace('"', '\\"')


def heuristic_label(node: dict) -> str:
    text = node.get("body_text", "")
    head = text.split(":", 1)[0].strip()
    if not head:
        head = node.get("id", "")
    words = head.split()
    if len(words) > 5:
        head = " ".join(words[:5]) + "..."
    return head


def make_dot(wiring: dict) -> str:
    nodes = wiring.get("nodes", [])
    edges = wiring.get("edges", [])
    node_ids = {n["id"] for n in nodes}
    lines: list[str] = []

    lines.append("digraph SprintReviewWiring {")
    lines.append('  graph [rankdir=TB, overlap=false, splines=true, outputorder=edgesfirst, nodesep=0.40, ranksep=1.28, pad=0.10];')
    lines.append('  node [shape=box, style="rounded,filled", color="#666666", penwidth=1.0, fontname="Helvetica", fontsize=18, margin="0.08,0.06"];')
    lines.append('  edge [arrowsize=0.75, color="#444444"];')
    lines.append("")

    for n in nodes:
        nid = n["id"]
        ntype = n.get("node_type", "")
        fill = NODE_COLORS.get(ntype, "#f2f2f2")
        label = DISPLAY_LABELS.get(nid, heuristic_label(n))
        attrs = {
            "fillcolor": f'"{fill}"',
            "label": f'"{dot_escape(label)}"',
        }
        if ntype == "outcome":
            attrs["fontname"] = '"Helvetica-Bold"'
            attrs["penwidth"] = "1.0"
        attr_text = ", ".join(f"{k}={v}" for k, v in attrs.items())
        lines.append(f'  "{nid}" [{attr_text}];')

    lines.append("")
    lines.append("  // Rank constraints")
    for i, row in enumerate(RANK_ROWS):
        present = [n for n in row if n in node_ids]
        if not present:
            continue
        members = "; ".join(f'"{nid}"' for nid in present)
        lines.append(f"  subgraph rank_{i} {{ rank=same; {members}; }}")
        if len(present) > 1:
            for a, b in zip(present, present[1:]):
                lines.append(f'  "{a}" -> "{b}" [style=invis, weight=40];')

    lines.append("")
    lines.append("  // Central top-down spine")
    for a, b in zip(["ret-D2", "ret-TB", "ret-S4", "ret-R2", "ret-C0", "ret-C1"],
                    ["ret-TB", "ret-S4", "ret-R2", "ret-C0", "ret-C1", "ret-C1"]):
        if a in node_ids and b in node_ids and a != b:
            lines.append(f'  "{a}" -> "{b}" [style=invis, weight=80];')

    lines.append("")
    lines.append("  // Typed edges from wiring JSON")
    for e in edges:
        src = e["source"]
        tgt = e["target"]
        et = e.get("edge_type", "")
        attrs = EDGE_STYLES.get(et, {"style": "solid", "color": "#444444", "penwidth": "1.0"})
        attr_text = ", ".join(f"{k}={v}" for k, v in attrs.items())
        lines.append(f'  "{src}" -> "{tgt}" [{attr_text}];')

    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path, help="Input wiring JSON path")
    ap.add_argument("-o", "--output", type=Path, required=True, help="Output .dot path")
    ap.add_argument("--pdf", type=Path, default=None, help="Optional output .pdf render path")
    args = ap.parse_args()

    wiring = json.loads(args.input.read_text())
    dot_text = make_dot(wiring)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(dot_text)
    print(f"Wrote {args.output}")

    if args.pdf is not None:
        dot_bin = shutil.which("dot")
        if dot_bin is None:
            print("ERROR: 'dot' binary not found in PATH", file=sys.stderr)
            return 2
        args.pdf.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [dot_bin, "-Tpdf", str(args.output), "-o", str(args.pdf)],
            check=True,
        )
        print(f"Wrote {args.pdf}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
