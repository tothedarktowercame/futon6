#!/usr/bin/env python3
"""Segment an IATC argument graph into CAS-SEL proof steps.

This is the seam-6 deterministic producer: it turns graph nodes/edges into the
`{paper_id, steps:[{id,text}]}` schema consumed by `cas_select.load_steps`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from r2d_concept_coverage import load_edn  # noqa: E402

DEFAULT_OUT_DIR = ROOT / "data" / "cas-select-steps" / "loop-run-70b"


def source_start(row: dict[str, Any]) -> int:
    lines = ((row.get("source") or {}).get("lines") or [10**9])
    return int(lines[0]) if lines else 10**9


def source_end(row: dict[str, Any]) -> int:
    lines = ((row.get("source") or {}).get("lines") or [10**9, 10**9])
    return int(lines[-1]) if lines else 10**9


def paper_id(graph_path: Path, graph: dict[str, Any]) -> str:
    return str(graph.get("paper/id") or graph.get("paper-id") or graph_path.stem)


def clean_sentence(text: str) -> str:
    text = " ".join(str(text or "").split())
    if not text:
        return ""
    return text if text.endswith((".", "!", "?")) else text + "."


def id_key(value: Any) -> str:
    return str(value)


def node_text(node_id: Any, nodes: dict[str, dict[str, Any]]) -> str:
    node = nodes.get(id_key(node_id))
    if not node:
        return str(node_id)
    return str(node.get("text") or node_id)


def premise_text(premise: Any, nodes: dict[str, dict[str, Any]]) -> str:
    if isinstance(premise, list):
        return "; ".join(node_text(row, nodes) for row in premise)
    if premise in (None, "", "?"):
        return ""
    return node_text(premise, nodes)


def warrant_text(edge: dict[str, Any]) -> str:
    warrant = edge.get("warrant") or {}
    if warrant.get("kind") == ":missing-warrant":
        return ""
    return str(warrant.get("text") or "").strip()


def edge_step_text(edge: dict[str, Any], nodes: dict[str, dict[str, Any]]) -> str:
    premise = premise_text(edge.get("premise"), nodes)
    conclusion = node_text(edge.get("conclusion"), nodes)
    warrant = warrant_text(edge)
    relation = str(edge.get("relation") or "").lstrip(":")

    if premise and relation == "because":
        text = f"Because {premise}, {conclusion}"
    elif premise:
        text = f"{premise}; therefore {conclusion}"
    else:
        text = conclusion
    if warrant:
        text = f"{text} because {warrant}"
    return clean_sentence(text)


def conclusion_ids(edges: list[dict[str, Any]]) -> set[str]:
    return {id_key(edge.get("conclusion")) for edge in edges if edge.get("conclusion") is not None}


def build_step_entries(graph: dict[str, Any]) -> list[dict[str, Any]]:
    nodes_list = list(graph.get("nodes") or [])
    edges = [edge for edge in graph.get("edges") or [] if edge.get("kind") in (None, ":infer")]
    nodes = {id_key(node.get("id")): node for node in nodes_list}
    derived = conclusion_ids(edges)
    entries: list[dict[str, Any]] = []

    for node in nodes_list:
        node_id = id_key(node.get("id"))
        if node_id in derived:
            continue
        text = clean_sentence(node.get("text") or "")
        if not text:
            continue
        entries.append(
            {
                "kind": "setup",
                "source_id": node_id,
                "line": source_start(node),
                "end_line": source_end(node),
                "text": text,
            }
        )

    for edge in edges:
        entries.append(
            {
                "kind": "edge",
                "source_id": id_key(edge.get("id")),
                "line": source_start(edge),
                "end_line": source_end(edge),
                "text": edge_step_text(edge, nodes),
            }
        )

    entries.sort(key=lambda row: (row["line"], row["end_line"], 0 if row["kind"] == "setup" else 1, row["source_id"]))
    for i, row in enumerate(entries, start=1):
        row["id"] = f"s{i}"
    return entries


def segment_graph(graph_path: Path) -> dict[str, Any]:
    graph = load_edn(graph_path)
    entries = build_step_entries(graph)
    return {
        "paper_id": paper_id(graph_path, graph),
        "steps": [{"id": row["id"], "text": row["text"]} for row in entries],
    }


def write_steps(graph_path: Path, out_dir: Path = DEFAULT_OUT_DIR, out_path: Path | None = None) -> Path:
    doc = segment_graph(graph_path)
    # Name by the GRAPH (proof) id, not the paper id. Naming by paper made every
    # proof of a paper write the same file, so 98 graphs collapsed to 12 with the
    # later silently overwriting the earlier — and S5 looks up per-proof
    # (`<pid>__pN.steps.json`), so it found nothing and the whole rung-3 half of
    # comprehension went dark (E-superpod-hardening H13).
    stem = graph_path.stem
    target = out_path or out_dir / f"{stem}.steps.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(doc, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    return target


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("graphs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, help="Output path for a single graph")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--stdout", action="store_true", help="Print a single graph's steps JSON")
    args = ap.parse_args(argv)

    if (args.out or args.stdout) and len(args.graphs) != 1:
        ap.error("--out/--stdout require exactly one graph")

    if args.stdout:
        print(json.dumps(segment_graph(args.graphs[0]), indent=2, ensure_ascii=False, sort_keys=True))
        return 0

    paths = [write_steps(path, args.out_dir, args.out) for path in args.graphs]
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
