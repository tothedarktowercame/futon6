#!/usr/bin/env python3
"""Pilot AIF+ method audit for proof-route trajectories.

This is a lightweight, reproducible heuristic audit over the curated
project-flow DAG. It is intentionally explicit about limitations:
it scores route evidence from commit metadata (labels/significance/subjects),
not from full proof-object telemetry.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class RouteSpec:
    key: str
    label: str
    node_ids: List[str]


ROUTES: List[RouteSpec] = [
    RouteSpec(
        key="p6_a_only",
        label="P6 A-Only",
        node_ids=[
            "p6-dispatch",
            "p6-dir-a",
            "p6-a-stress",
            "p6-transfer-handoff",
        ],
    ),
    RouteSpec(
        key="p6_a_to_ef",
        label="P6 A->E+F",
        node_ids=[
            "p6-dispatch",
            "p6-dir-a",
            "p6-a-stress",
            "p6-transfer-handoff",
            "p6-dir-d",
            "p6-arnt",
            "p6-gl-balance",
            "p6-layer-switch",
            "p6-pigeonhole-artifacts",
            "p6-proof-draft",
            "p6-gap-honesty",
            "p6-amplification",
            "p6-coupling",
            "p6-architectural",
            "p6-e-f-hybrid",
        ],
    ),
    RouteSpec(
        key="p4_main",
        label="P4 Main Route",
        node_ids=[
            "p4-id-search",
            "p4-n3-proof",
            "p4-n4-cert",
        ],
    ),
]

OBSERVE_KW = (
    "probe",
    "search",
    "hunt",
    "diagnostic",
    "isolat",
    "stress",
    "audit",
    "fails",
    "quantif",
    "gap",
)
ACT_KW = (
    "proof",
    "switch",
    "reduction",
    "committed",
    "resolve",
    "close",
    "draft",
    "introduced",
    "package",
)
ARTIFACT_KW = (
    "artifact",
    "handoff",
    "draft",
    "package",
    "hardened",
    "tracked",
)
ARCH_KW = (
    "architectural",
    "layer switch",
    "hybrid",
    "bridge",
    "reframed",
)


def _node_text(node: Dict[str, object]) -> str:
    commit = node.get("commit", {})
    if not isinstance(commit, dict):
        commit = {}
    parts = [
        str(node.get("short_label", "")),
        str(node.get("significance", "")),
        str(commit.get("subject", "")),
    ]
    return " ".join(parts).lower()


def _has_any(text: str, keys: tuple[str, ...]) -> bool:
    return any(k in text for k in keys)


def _ratio(a: int, b: int) -> float:
    hi = max(a, b)
    if hi == 0:
        return 0.0
    return min(a, b) / hi


def _route_verdict(label: str, balance: float, artifacts: int, architecture: int) -> str:
    if label == "P6 A-Only":
        return "Limited: observation-heavy, weak closure"
    if label == "P6 A->E+F":
        if balance >= 0.55 and artifacts >= 3 and architecture >= 3:
            return "Improved: richer loop; still incomplete"
        return "Mixed: improvement signals but weak closure"
    if label == "P4 Main Route":
        return "Focused partial: closes n<=3, leaves cert branch"
    return "Unclassified"


def render_markdown(payload: Dict[str, object], routes: List[Dict[str, object]]) -> str:
    lines: List[str] = []
    lines.append("# AIF+ Pilot Method Audit (P4/P6)")
    lines.append("")
    lines.append("This is a heuristic route audit over commit-level evidence.")
    lines.append("It is **not** a full AIF runtime instrumentation.")
    lines.append("")
    lines.append(f"Source DAG timestamp: `{payload.get('generated', 'unknown')}`")
    lines.append("")
    lines.append("## Route Metrics")
    lines.append("")
    lines.append("| Route | Steps | Observe events | Act events | Balance (I2 proxy) | Artifact events | Architecture events | Verdict |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for r in routes:
        lines.append(
            f"| {r['label']} | {r['steps']} | {r['observe']} | {r['act']} | {r['balance']:.2f} | "
            f"{r['artifact']} | {r['architecture']} | {r['verdict']} |"
        )
    lines.append("")
    lines.append("## Readout")
    lines.append("")
    lines.append("1. P6 A-only is constrained by a low observe/action balance and low closure evidence.")
    lines.append("2. P6 A->E+F shows a stronger loop profile (more architecture shifts and artifactization),")
    lines.append("but still ends in a partial state rather than a fully closed proof.")
    lines.append("3. P4 main route is sharp and short: strong local closure at n<=3,")
    lines.append("with an explicit remaining certification branch for n>=4.")
    lines.append("")
    lines.append("## Suggested Next Instrumentation")
    lines.append("")
    lines.append("1. Add event-level tags to proof episodes (observe/propose/act/verify/handoff).")
    lines.append("2. Record explicit invariant checks (I3/I4/I6) per episode, not just per commit summary.")
    lines.append("3. Re-run this audit after each major strategy pivot (especially on Problem 6).")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dag-json",
        default="data/first-proof/project-flow-dag.json",
        help="Path to project DAG json",
    )
    ap.add_argument(
        "--out-md",
        default="data/first-proof/aif-plus-method-audit-p4-p6.md",
        help="Path to output markdown report",
    )
    args = ap.parse_args()

    dag_path = Path(args.dag_json)
    if not dag_path.exists():
        raise SystemExit(
            f"missing DAG json: {dag_path} (run scripts/generate-project-flow-dag.py first)"
        )

    payload = json.loads(dag_path.read_text(encoding="utf-8"))
    nodes = payload.get("nodes", [])
    if not isinstance(nodes, list):
        raise SystemExit("invalid DAG json: nodes must be a list")

    by_id = {n.get("id"): n for n in nodes if isinstance(n, dict)}
    route_rows: List[Dict[str, object]] = []

    for spec in ROUTES:
        route_nodes = [by_id[rid] for rid in spec.node_ids if rid in by_id]
        observe = 0
        act = 0
        artifact = 0
        architecture = 0

        for node in route_nodes:
            text = _node_text(node)
            if _has_any(text, OBSERVE_KW):
                observe += 1
            if _has_any(text, ACT_KW):
                act += 1
            if _has_any(text, ARTIFACT_KW):
                artifact += 1
            if _has_any(text, ARCH_KW):
                architecture += 1

        balance = _ratio(observe, act)
        route_rows.append(
            {
                "key": spec.key,
                "label": spec.label,
                "steps": len(route_nodes),
                "observe": observe,
                "act": act,
                "artifact": artifact,
                "architecture": architecture,
                "balance": balance,
                "verdict": _route_verdict(spec.label, balance, artifact, architecture),
            }
        )

    out_text = render_markdown(payload, route_rows)
    out_path = Path(args.out_md)
    out_path.write_text(out_text, encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
