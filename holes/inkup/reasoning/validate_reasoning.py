#!/usr/bin/env python3
"""Validate secondary reasoning graphs without external dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIGNATURES = {
    "answered-by": {("question", "answer")},
    "cites-reason": {("answer", "obstacle"), ("answer", "move")},
    "blocks": {("obstacle", "method")},
    "resolved-by": {("obstacle", "move")},
    "preserves": {("move", "invariant")},
    "produces": {("move", "conversion"), ("method", "result")},
    "enables": {("conversion", "method")},
    "transported-by": {("result", "return")},
    "summarizes": {("answer", "return")},
}


def fail(message: str) -> None:
    raise SystemExit(message)


def validate(path: Path) -> dict:
    doc = json.loads(path.read_text())
    if doc.get("schema") != "inkup.reasoning-graph/v0":
        fail("wrong reasoning graph schema")
    for name, source in doc.get("sources", {}).items():
        source_path = (path.parent / source["path"]).resolve()
        if not source_path.is_file():
            fail(f"{name}: missing source {source_path}")
        digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if digest != source["sha256"]:
            fail(f"{name}: source hash drifted")

    nodes = {node["id"]: node for node in doc.get("nodes", [])}
    if len(nodes) != len(doc.get("nodes", [])):
        fail("duplicate node id")
    if not nodes:
        fail("no reasoning nodes")
    outgoing = defaultdict(list)
    for edge in doc.get("edges", []):
        if edge["relation"] not in SIGNATURES:
            fail(f"unknown relation {edge['relation']}")
        if edge["from"] not in nodes or edge["to"] not in nodes:
            fail(f"unresolved edge {edge}")
        signature = (nodes[edge["from"]]["kind"], nodes[edge["to"]]["kind"])
        if signature not in SIGNATURES[edge["relation"]]:
            fail(f"bad signature for {edge['relation']}: {signature}")
        outgoing[edge["from"]].append(edge["to"])

    questions = [node["id"] for node in nodes.values() if node["kind"] == "question"]
    answers = {node["id"] for node in nodes.values() if node["kind"] == "answer"}
    if len(questions) != 1 or len(answers) != 1:
        fail("expected exactly one question and one answer")
    seen = {questions[0]}
    queue = deque(seen)
    while queue:
        for target in outgoing[queue.popleft()]:
            if target not in seen:
                seen.add(target)
                queue.append(target)
    missing = sorted(nodes.keys() - seen)
    if missing:
        fail(f"nodes unreachable from question: {missing}")
    required_kinds = {"obstacle", "move", "invariant", "conversion", "method", "result", "return"}
    present_kinds = {nodes[node_id]["kind"] for node_id in seen}
    if not required_kinds <= present_kinds:
        fail(f"answer lacks reasoning stages: {sorted(required_kinds - present_kinds)}")
    for node in nodes.values():
        if not node.get("claim") or not node.get("evidence"):
            fail(f"{node['id']}: claim or evidence missing")
    return doc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("graph", type=Path, nargs="?", default=HERE / "a96J06-why-rescale.json")
    args = parser.parse_args()
    doc = validate(args.graph.resolve())
    print(f"valid secondary reasoning graph: {len(doc['nodes'])} nodes, {len(doc['edges'])} edges")


if __name__ == "__main__":
    main()
