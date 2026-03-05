#!/usr/bin/env python3
"""Self-play over internal proof history.

Inputs:
- data/first-proof/project-flow-dag.json

Outputs:
- data/first-proof/selfplay-proof-episodes.jsonl
- data/first-proof/selfplay-policy-summary.md

The goal is to learn strategy transitions from our own history rather than
external corpora.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


CLOSE_PAT = re.compile(r"(closed|discharged|breakthrough|proof|integral s closed|n=3 proof)", re.I)
REDUCE_PAT = re.compile(r"(reduction|bridge|switch|draft|update|artifact|handoff|partial averages|hybrid)", re.I)
MAP_PAT = re.compile(r"(probe|stress|diagnosis|dispatch|counterexample|audit|hunt)", re.I)


@dataclass
class Node:
    id: str
    lane: str
    short_label: str
    significance: str
    subject: str
    date: str


@dataclass
class Episode:
    lane: str
    source: str
    target: str
    edge_type: str
    source_label: str
    target_label: str
    source_subject: str
    target_subject: str
    source_state: str
    target_outcome: str
    reward: int
    rationale: str


def classify_outcome(text: str) -> Tuple[str, int, str]:
    if CLOSE_PAT.search(text):
        return "close", 3, "closure signal"
    if REDUCE_PAT.search(text):
        return "reduce", 2, "reduction/localization signal"
    if MAP_PAT.search(text):
        return "map", 1, "mapping/diagnostic signal"
    return "map", 1, "default mapping signal"


def tokenize(text: str) -> set[str]:
    toks = re.findall(r"[a-z0-9\+\-]+", text.lower())
    stop = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "add",
        "problem",
        "route",
        "proof",
        "p6",
        "p7",
        "p4",
        "meta",
    }
    return {t for t in toks if len(t) > 2 and t not in stop}


def jacc(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return inter / union


def load_dag(path: Path) -> Tuple[Dict[str, Node], List[dict], List[dict]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    nodes: Dict[str, Node] = {}
    for n in obj["nodes"]:
        nodes[n["id"]] = Node(
            id=n["id"],
            lane=n["lane"],
            short_label=n["short_label"],
            significance=n["significance"],
            subject=n["commit"]["subject"],
            date=n["commit"]["date"],
        )
    return nodes, obj["edges"], obj["nodes"]


def build_episodes(nodes: Dict[str, Node], edges: List[dict]) -> List[Episode]:
    out: List[Episode] = []
    for e in edges:
        s = nodes[e["source"]]
        t = nodes[e["target"]]
        outcome, reward, rationale = classify_outcome(f"{t.short_label} {t.subject} {t.significance}")
        state_text = f"{s.short_label}. {s.subject}. {s.significance}"
        out.append(
            Episode(
                lane=s.lane,
                source=s.id,
                target=t.id,
                edge_type=e["edge_type"],
                source_label=s.short_label,
                target_label=t.short_label,
                source_subject=s.subject,
                target_subject=t.subject,
                source_state=state_text,
                target_outcome=outcome,
                reward=reward,
                rationale=rationale,
            )
        )
    return out


def write_jsonl(path: Path, episodes: List[Episode]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for ep in episodes:
            f.write(json.dumps(ep.__dict__, ensure_ascii=True) + "\n")


def frontier_nodes(nodes: Dict[str, Node], edges: List[dict]) -> List[Node]:
    has_out = {e["source"] for e in edges}
    fr = [n for nid, n in nodes.items() if nid not in has_out]
    fr.sort(key=lambda n: (n.lane, n.date, n.id))
    return fr


def latest_frontier_per_lane(frontiers: List[Node]) -> List[Node]:
    by_lane: Dict[str, Node] = {}
    for n in frontiers:
        cur = by_lane.get(n.lane)
        if cur is None or n.date > cur.date:
            by_lane[n.lane] = n
    out = list(by_lane.values())
    out.sort(key=lambda n: (n.lane, n.date, n.id))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Run self-play policy extraction from project DAG")
    ap.add_argument(
        "--dag",
        type=Path,
        default=Path("data/first-proof/project-flow-dag.json"),
    )
    ap.add_argument(
        "--out-jsonl",
        type=Path,
        default=Path("data/first-proof/selfplay-proof-episodes.jsonl"),
    )
    ap.add_argument(
        "--out-md",
        type=Path,
        default=Path("data/first-proof/selfplay-policy-summary.md"),
    )
    args = ap.parse_args()

    nodes, edges, raw_nodes = load_dag(args.dag)
    episodes = build_episodes(nodes, edges)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_jsonl, episodes)

    by_lane = defaultdict(list)
    by_lane_edge = defaultdict(list)
    for ep in episodes:
        by_lane[ep.lane].append(ep)
        by_lane_edge[(ep.lane, ep.edge_type)].append(ep.reward)

    edge_rank = []
    for (lane, edge_type), vals in by_lane_edge.items():
        edge_rank.append((lane, edge_type, sum(vals) / len(vals), len(vals)))
    edge_rank.sort(key=lambda x: (x[0], -x[2], -x[3], x[1]))

    # recommend for frontier nodes by lane and state similarity
    frontiers_all = frontier_nodes(nodes, edges)
    frontiers = latest_frontier_per_lane(frontiers_all)
    lane_best_edge = {}
    for lane in sorted({n.lane for n in nodes.values()}):
        cand = [r for r in edge_rank if r[0] == lane]
        if cand:
            lane_best_edge[lane] = cand[0][1]

    lines: List[str] = []
    lines.append("# Self-Play Policy Summary (Internal Proof History)")
    lines.append("")
    lines.append(f"Generated: `{datetime.now(timezone.utc).isoformat()}`")
    lines.append(f"Episodes: `{len(episodes)}` from DAG edges `{len(edges)}`")
    lines.append(f"Input DAG: `{args.dag}`")
    lines.append("")

    lines.append("## Outcome Mix")
    lines.append("")
    counts = defaultdict(int)
    for ep in episodes:
        counts[ep.target_outcome] += 1
    for k in ["close", "reduce", "map"]:
        lines.append(f"- `{k}`: `{counts[k]}`")
    lines.append("")

    lines.append("## Best Action Types By Lane")
    lines.append("")
    lines.append("| Lane | Edge Type | Mean Reward | Support |")
    lines.append("|---|---|---:|---:|")
    for lane, edge_type, mr, n in edge_rank:
        lines.append(f"| `{lane}` | `{edge_type}` | {mr:.2f} | {n} |")
    lines.append("")

    lines.append("## Frontier Recommendations")
    lines.append("")
    lines.append("| Frontier Node | Lane | Suggested Next Move | Basis |")
    lines.append("|---|---|---|---|")

    for fn in frontiers:
        best_edge = lane_best_edge.get(fn.lane, "reform")
        state_tokens = tokenize(f"{fn.short_label} {fn.subject} {fn.significance}")

        exemplars = [ep for ep in episodes if ep.lane == fn.lane and ep.edge_type == best_edge]
        scored = []
        for ep in exemplars:
            score = jacc(state_tokens, tokenize(ep.source_state))
            scored.append((score, ep))
        scored.sort(key=lambda x: (-x[0], -x[1].reward, x[1].source))
        top = scored[0][1] if scored else None

        if top is None:
            basis = "lane-level reward only"
        else:
            basis = f"closest prior: `{top.source_label} -> {top.target_label}` ({top.edge_type})"

        lines.append(
            f"| `{fn.short_label}` | `{fn.lane}` | `{best_edge}` | {basis} |"
        )

    lines.append("")
    lines.append("Tip frontier policy: latest frontier node per lane.")
    lines.append("")

    lines.append("## P6 Tactical Recommendation")
    lines.append("")
    p6_front = [f for f in frontiers if f.lane == "P6"]
    if p6_front:
        f = p6_front[0]
        lines.append(
            "From internal self-play, the next highest-value move after the current P6 frontier "
            f"(`{f.short_label}`) is to run a `reduction`/`refine` style step that converts the open package into one concrete lemma family with falsifiable thresholds."
        )
        lines.append(
            "In practice: lock one E-regime lemma target and one F-regime gain-loss inequality, then emit a dedicated verifier script/report pair for each."
        )
    else:
        lines.append("No P6 frontier node found in DAG input.")

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- This is internal self-play over our own proof traces (no external MO/arXiv dependency).")
    lines.append("- Rewards are intentionally simple (`close`=3, `reduce`=2, `map`=1) to keep behavior auditable.")
    lines.append("- Upgrade path: replace heuristic outcome labels with explicit status fields from wiring nodes.")

    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
