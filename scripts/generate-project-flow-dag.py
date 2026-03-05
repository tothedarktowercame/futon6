#!/usr/bin/env python3
"""Generate an evidence-backed First Proof project DAG from milestone commits."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


@dataclass
class Milestone:
    id: str
    hash: str
    lane: str
    short_label: str
    significance: str


MILESTONES: List[Milestone] = [
    # Problem 4 lane
    Milestone("p4-id-search", "9db6b4f", "P4", "P4 identity hunt", "Haar-orbit exploration isolates key invariant identity target."),
    Milestone("p4-n3-proof", "0003300", "P4", "P4 n=3 proof", "Phi_3 * disc identity + Cauchy-Schwarz closes n=3 route."),
    Milestone("p4-n4-cert", "84c0041", "P4", "P4 n>=4 cert route", "PHC parsing hardened; computational certification tracked for n>=4 branch."),

    # Problem 7 lane
    Milestone("p7-obstruction", "0fa4e82", "P7", "P7 obstruction isolated", "Reflection surgery branch reframed as open, with explicit obligations."),
    Milestone("p7-rotation-switch", "287a41c", "P7", "P7 layer switch", "Rotation architecture introduced as codim-2 route."),
    Milestone("p7-e2", "620ed57", "P7", "P7 E2 discharged", "Rotation lattice existence + E2 obligation resolved."),
    Milestone("p7-s-rational", "8ce9771", "P7", "P7 rational S closed", "Flat normal bundle argument kills rational obstruction."),
    Milestone("p7-s-integral", "158bc4f", "P7", "P7 integral S closed", "Integral obstruction closed; surgery obligation discharged."),

    # Problem 6 lane: A->E+F trajectory
    Milestone("p6-dispatch", "5289ca8", "P6", "P6 A/B/C dispatch", "Formal attack dispatch with three directions launched."),
    Milestone("p6-dir-c", "b6a7625", "P6", "Direction C probe", "Fixed-block interlacing route tested; did not close GPL-H."),
    Milestone("p6-dir-a", "63a23ba", "P6", "Direction A probe", "SR edge-to-star transfer quantified; narrowed to transfer lemma."),
    Milestone("p6-dir-b", "22c091f", "P6", "Direction B probe", "Hyperbolic/self-concordant route tested; structural mismatch logged."),
    Milestone("p6-a-stress", "3897a41", "P6", "A theorem-shape stress", "Uniform kappa transfer vs barrier target tension quantified."),
    Milestone("p6-transfer-handoff", "c623eff", "P6", "Transfer handoff", "Residual gap reframed; D/E/F routes introduced explicitly."),
    Milestone("p6-dir-d", "d64fd13", "P6", "Direction D probe", "Near-rank-1 universality fails in dense late-step rows."),
    Milestone("p6-arnt", "a32b83e", "P6", "AR-NT bridge", "Open bridge reduced to nontrivial dbar<gbar inequality."),
    Milestone("p6-gl-balance", "975748c", "P6", "GL-Balance route", "Gain-loss reformulation provides F-style bridge target."),
    Milestone("p6-layer-switch", "73aa112", "P6", "Elementary layer switch", "Turan+Foster+pigeonhole pivot breaks TryHarder loop."),
    Milestone("p6-pigeonhole-artifacts", "ec8838a", "P6", "Pigeonhole artifacts", "Wiring and verification artifacts added for new proof core."),
    Milestone("p6-proof-draft", "059c1fd", "P6", "Near-final draft", "Main draft and Codex review handoff captured."),
    Milestone("p6-partial-avg", "6f24907", "P6", "Partial averages update", "Partial-averages breakthrough recorded in handoff."),
    Milestone("p6-gap-honesty", "b55eb7d", "P6", "Gap-status correction", "Explicitly reclassifies unresolved caveats and eps^2 bottleneck."),
    Milestone("p6-amplification", "e339efa", "P6", "Amplification diagnostics", "Candidate amplification inequalities empirically probed."),
    Milestone("p6-coupling", "48dee8d", "P6", "Trajectory coupling diag", "Coupling/spectral spread issues isolated."),
    Milestone("p6-architectural", "26998c3", "P6", "Architectural diagnosis", "Remaining formal gap identified as strategy-level issue."),
    Milestone("p6-e-f-hybrid", "7e03174", "P6", "E+F hybrid reduction", "MO evidence + E/F reduction theorem package committed."),

    # Monograph/meta lane
    Milestone("meta-patterns", "3454c9d", "Meta", "Part IV pivotal hashes", "Cross-problem proof-pattern chapter introduced."),
    Milestone("meta-aif", "fae0847", "Meta", "AIF design note", "S-expression + peripheral reflection framing documented."),
]


EDGES = [
    # P4
    ("p4-id-search", "p4-n3-proof", "identity->proof"),
    ("p4-n3-proof", "p4-n4-cert", "extends"),

    # P7
    ("p7-obstruction", "p7-rotation-switch", "layer-switch"),
    ("p7-rotation-switch", "p7-e2", "resolve-E2"),
    ("p7-e2", "p7-s-rational", "surgery-step"),
    ("p7-s-rational", "p7-s-integral", "integral-close"),

    # P6 main chain
    ("p6-dispatch", "p6-dir-c", "dispatch"),
    ("p6-dispatch", "p6-dir-a", "dispatch"),
    ("p6-dispatch", "p6-dir-b", "dispatch"),
    ("p6-dir-a", "p6-a-stress", "stress-test"),
    ("p6-dir-a", "p6-transfer-handoff", "residual->handoff"),
    ("p6-dir-b", "p6-transfer-handoff", "residual->handoff"),
    ("p6-dir-c", "p6-transfer-handoff", "residual->handoff"),
    ("p6-transfer-handoff", "p6-dir-d", "next-route"),
    ("p6-dir-d", "p6-arnt", "reduction"),
    ("p6-arnt", "p6-gl-balance", "direction-F"),
    ("p6-gl-balance", "p6-layer-switch", "blocked->switch"),
    ("p6-layer-switch", "p6-pigeonhole-artifacts", "artifactize"),
    ("p6-pigeonhole-artifacts", "p6-proof-draft", "draft"),
    ("p6-proof-draft", "p6-partial-avg", "refine"),
    ("p6-partial-avg", "p6-gap-honesty", "audit"),
    ("p6-gap-honesty", "p6-amplification", "subgap-2"),
    ("p6-gap-honesty", "p6-coupling", "subgap-1"),
    ("p6-amplification", "p6-architectural", "synthesis"),
    ("p6-coupling", "p6-architectural", "synthesis"),
    ("p6-architectural", "p6-e-f-hybrid", "A->E+F"),

    # cross-lane/meta
    ("p7-rotation-switch", "meta-patterns", "feeds-patterns"),
    ("p6-layer-switch", "meta-patterns", "feeds-patterns"),
    ("p4-id-search", "meta-patterns", "feeds-patterns"),
    ("meta-aif", "meta-patterns", "theory->chapter"),
]


def run_git(*args: str) -> str:
    p = subprocess.run(["git", *args], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip())
    return p.stdout.strip()


def commit_info(h: str) -> Dict[str, str]:
    fmt = "%H|%h|%ad|%s"
    out = run_git("show", "-s", f"--format={fmt}", "--date=iso-strict", h)
    full, short, date_s, subj = out.split("|", 3)
    return {"hash": full, "short": short, "date": date_s, "subject": subj}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_json = root / "data/first-proof/project-flow-dag.json"
    out_md = root / "data/first-proof/project-flow-dag.md"

    nodes = []
    for m in MILESTONES:
        info = commit_info(m.hash)
        nodes.append(
            {
                "id": m.id,
                "lane": m.lane,
                "short_label": m.short_label,
                "significance": m.significance,
                "commit": info,
            }
        )

    node_ids = {n["id"] for n in nodes}
    edges = []
    for s, t, et in EDGES:
        if s not in node_ids or t not in node_ids:
            continue
        edges.append({"source": s, "target": t, "edge_type": et})

    payload = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "repo": str(root),
        "nodes": nodes,
        "edges": edges,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    by_id = {n["id"]: n for n in nodes}

    lines: List[str] = []
    lines.append("# First Proof Project Flow DAG")
    lines.append("")
    lines.append(f"Generated: `{payload['generated']}`")
    lines.append("Source of truth: git commit metadata + curated milestone map.")
    lines.append("")

    lines.append("## DAG (Mermaid)")
    lines.append("")
    lines.append("```mermaid")
    lines.append("graph TD")

    lane_order = ["P4", "P7", "P6", "Meta"]
    for lane in lane_order:
        lines.append(f"  subgraph {lane}")
        lane_nodes = [n for n in nodes if n["lane"] == lane]
        lane_nodes.sort(key=lambda n: n["commit"]["date"])
        for n in lane_nodes:
            label = f"{n['short_label']}\\n{n['commit']['short']}"
            lines.append(f"    {n['id']}[{json.dumps(label)}]")
        lines.append("  end")

    for e in edges:
        lines.append(f"  {e['source']} -->|{e['edge_type']}| {e['target']}")

    lines.append("```")
    lines.append("")

    lines.append("## P6 A->E+F Route")
    lines.append("")
    p6_route = [
        "p6-dispatch",
        "p6-dir-a",
        "p6-a-stress",
        "p6-transfer-handoff",
        "p6-dir-d",
        "p6-arnt",
        "p6-gl-balance",
        "p6-layer-switch",
        "p6-proof-draft",
        "p6-gap-honesty",
        "p6-architectural",
        "p6-e-f-hybrid",
    ]
    lines.append("| Step | Commit | Date (UTC) | What changed |")
    lines.append("|---|---|---|---|")
    for pid in p6_route:
        n = by_id[pid]
        lines.append(
            f"| `{n['short_label']}` | `{n['commit']['short']}` | `{n['commit']['date']}` | {n['commit']['subject']} |"
        )
    lines.append("")

    lines.append("## Pivotal Moments (Evidence-Backed)")
    lines.append("")
    lines.append("| Lane | Commit | Pivotal moment | Why pivotal |")
    lines.append("|---|---|---|---|")
    pivotal_ids = [
        "p4-n3-proof",
        "p7-rotation-switch",
        "p7-s-integral",
        "p6-layer-switch",
        "p6-partial-avg",
        "p6-e-f-hybrid",
        "meta-patterns",
    ]
    for pid in pivotal_ids:
        n = by_id[pid]
        lines.append(
            f"| `{n['lane']}` | `{n['commit']['short']}` | {n['short_label']} | {n['significance']} |"
        )
    lines.append("")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
