#!/usr/bin/env python3
"""Validate secondary reasoning graphs without external dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict, deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE_ROOTS = {
    "apm-lean": Path(os.environ.get("INKUP_APM_LEAN_ROOT", ROOT.parent / "apm-lean")),
}
STRATEGY_SIGNATURES = {
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
WARRANT_SIGNATURES = {
    "answered-by": {("question", "answer")},
    "cites-reason": {("answer", "warrant")},
    "requires": {("warrant", "premise"), ("warrant", "claim")},
    "warrants": {("warrant", "claim")},
    "violated-in": {("premise", "boundary-case")},
}


def fail(message: str) -> None:
    raise SystemExit(message)


def resolve_source(graph_path: Path, spec: str) -> Path:
    root, sep, relative = spec.partition(":")
    if sep and root in SOURCE_ROOTS:
        if relative.startswith("/"):
            fail(f"source path under {root} must be relative: {spec}")
        return SOURCE_ROOTS[root] / relative
    return (graph_path.parent / spec).resolve()


def validate(path: Path) -> dict:
    doc = json.loads(path.read_text())
    if doc.get("schema") != "inkup.reasoning-graph/v0":
        fail("wrong reasoning graph schema")
    form = doc.get("form", "strategy-answer")
    if form not in {"strategy-answer", "warrant-chain"}:
        fail(f"unknown reasoning graph form {form}")
    signatures = STRATEGY_SIGNATURES if form == "strategy-answer" else WARRANT_SIGNATURES
    for name, source in doc.get("sources", {}).items():
        source_path = resolve_source(path, source["path"])
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
    outgoing_relations = defaultdict(list)
    for edge in doc.get("edges", []):
        if edge["relation"] not in signatures:
            fail(f"relation {edge['relation']} is not allowed in {form}")
        if edge["from"] not in nodes or edge["to"] not in nodes:
            fail(f"unresolved edge {edge}")
        signature = (nodes[edge["from"]]["kind"], nodes[edge["to"]]["kind"])
        if signature not in signatures[edge["relation"]]:
            fail(f"bad signature for {edge['relation']}: {signature}")
        outgoing[edge["from"]].append(edge["to"])
        outgoing_relations[edge["from"]].append(edge["relation"])

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
    if form == "strategy-answer":
        required_kinds = {"obstacle", "move", "invariant", "conversion", "method", "result", "return"}
        present_kinds = {nodes[node_id]["kind"] for node_id in seen}
        if not required_kinds <= present_kinds:
            fail(f"answer lacks reasoning stages: {sorted(required_kinds - present_kinds)}")
    else:
        allowed_kinds = {"question", "answer", "premise", "warrant", "claim", "boundary-case"}
        unknown_kinds = {node["kind"] for node in nodes.values()} - allowed_kinds
        if unknown_kinds:
            fail(f"unknown warrant-chain node kinds: {sorted(unknown_kinds)}")
        warrants = [node["id"] for node in nodes.values() if node["kind"] == "warrant"]
        if not warrants:
            fail("warrant-chain has no warrant")
        for warrant in warrants:
            relations = outgoing_relations[warrant]
            if relations.count("requires") < 1:
                fail(f"warrant {warrant} has no requires edge")
            if relations.count("warrants") != 1:
                fail(f"warrant {warrant} must have exactly one warrants edge")
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
