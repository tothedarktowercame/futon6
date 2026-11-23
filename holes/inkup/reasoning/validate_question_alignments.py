#!/usr/bin/env python3
"""Validate the support gate between reader questions and content plans."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FUNCTIONS = ROOT / "holes/inkup/grammar/question-functions-v0.tsv"
SUPPORT = ROOT / "holes/inkup/grammar/relation-support-v0.tsv"

FIELDS = {
    "candidate_id", "source_wording", "question_function", "requested_relation",
    "mapping_state", "proposers", "reviewer", "support", "support_state",
    "disposition", "note",
}
MAPPING_STATES = {"accepted", "contested", "rejected"}
SUPPORT_STATES = {"supported", "contested", "unsupported"}
DISPOSITIONS = {"anticipate", "defer", "omit", "diagnostic"}


def read_tsv(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def read_jsonl(path: Path) -> tuple[list[dict], list[tuple[int, str]]]:
    records, errors = [], []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append((line_no, f"QA007 malformed JSON: {exc.msg}"))
            continue
        if not isinstance(value, dict):
            errors.append((line_no, "QA007 record is not an object"))
            continue
        records.append(value)
    return records, errors


def graph_inventory(raw_path: str) -> tuple[set[str], set[tuple[str, str, str]]]:
    path = (ROOT / raw_path).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError("graph path leaves repository") from exc
    if not path.is_file():
        raise ValueError("graph file does not exist")
    doc = json.loads(path.read_text())
    nodes = {node.get("id") for node in doc.get("nodes", []) if node.get("id")}
    if doc.get("schema") == "inkup.reasoning-graph/v0":
        edges = {(edge.get("from"), edge.get("relation"), edge.get("to"))
                 for edge in doc.get("edges", [])}
    elif doc.get("schema") == "inkup.blueprint-lineage/v0":
        edges = {(node.get("id"), "depends-on", target)
                 for node in doc.get("nodes", []) for target in node.get("depends_on", [])}
    else:
        raise ValueError("unsupported graph schema")
    return nodes, edges


def load_candidates(path: Path | None) -> dict[str, dict] | None:
    if path is None:
        return None
    records, errors = read_jsonl(path)
    if errors:
        raise ValueError(f"invalid candidates file: {errors[0][1]}")
    return {str(item.get("entity_id")): item for item in records}


def validate(path: Path, candidates_path: Path | None = None) -> tuple[list[dict], list[tuple[int, str]]]:
    functions = {row["id"]: row for row in read_tsv(FUNCTIONS)}
    admissible: dict[str, set[str]] = {}
    for row in read_tsv(SUPPORT):
        admissible.setdefault(row["requested_relation"], set()).add(row["admissible_edge_relation"])
    candidates = load_candidates(candidates_path)
    records, errors = read_jsonl(path)
    graph_cache = {}

    for line_no, record in enumerate(records, 1):
        missing = FIELDS - record.keys()
        unknown = record.keys() - FIELDS
        enums_valid = (record.get("mapping_state") in MAPPING_STATES and
                       record.get("support_state") in SUPPORT_STATES and
                       record.get("disposition") in DISPOSITIONS)
        shape_valid = (isinstance(record.get("candidate_id"), str) and
                       isinstance(record.get("source_wording"), str) and
                       bool(record.get("source_wording")) and
                       isinstance(record.get("proposers"), list) and
                       all(isinstance(x, str) and x for x in record.get("proposers", [])) and
                       (record.get("reviewer") is None or isinstance(record.get("reviewer"), str)) and
                       isinstance(record.get("support"), list) and
                       isinstance(record.get("note"), str))
        if missing or unknown or not enums_valid or not shape_valid:
            detail = f"missing={sorted(missing)} unknown={sorted(unknown)}"
            errors.append((line_no, f"QA007 invalid record shape or enum: {detail}"))
            continue

        function_id = record["question_function"]
        function = functions.get(function_id) if function_id is not None else None
        if ((function_id is not None and function is None) or
                (function is not None and record["requested_relation"] != function["missing_relation"]) or
                (function_id is None and record["requested_relation"] is not None)):
            errors.append((line_no, "QA001 unknown question function or relation mismatch"))

        resolved_edges = []
        anchor_error = False
        for anchor in record["support"]:
            anchor_ok = isinstance(anchor, dict) and isinstance(anchor.get("graph"), str)
            has_node = anchor_ok and set(anchor) == {"graph", "node"} and isinstance(anchor.get("node"), str)
            edge = anchor.get("edge") if anchor_ok else None
            has_edge = (anchor_ok and set(anchor) == {"graph", "edge"} and isinstance(edge, dict) and
                        set(edge) == {"from", "relation", "to"} and
                        all(isinstance(edge.get(k), str) for k in ("from", "relation", "to")))
            if not has_node and not has_edge:
                errors.append((line_no, "QA002 malformed support anchor"))
                anchor_error = True
                continue
            try:
                if anchor["graph"] not in graph_cache:
                    graph_cache[anchor["graph"]] = graph_inventory(anchor["graph"])
                nodes, edges = graph_cache[anchor["graph"]]
                if has_node and anchor["node"] not in nodes:
                    raise ValueError("node does not resolve")
                if has_edge:
                    triple = (edge["from"], edge["relation"], edge["to"])
                    if triple not in edges:
                        raise ValueError("edge does not resolve")
                    resolved_edges.append(triple)
            except (ValueError, json.JSONDecodeError) as exc:
                errors.append((line_no, f"QA002 unresolved support anchor: {exc}"))
                anchor_error = True

        if record["support_state"] == "supported" and not anchor_error:
            allowed = admissible.get(record["requested_relation"], set())
            if not allowed or not any(relation in allowed for _source, relation, _target in resolved_edges):
                errors.append((line_no, "QA003 supported record lacks an admissible resolved edge"))

        if ((record["disposition"] == "anticipate" and
             (record["mapping_state"] != "accepted" or record["support_state"] != "supported")) or
                (record["support_state"] == "unsupported" and record["disposition"] != "diagnostic")):
            errors.append((line_no, "QA004 disposition is inconsistent with mapping/support state"))

        if (record["mapping_state"] == "accepted" and
                (not record["reviewer"] or record["reviewer"] in record["proposers"])):
            errors.append((line_no, "QA005 accepted mapping lacks an independent reviewer"))

        if candidates is not None and not record["candidate_id"].startswith("seed:"):
            source = candidates.get(record["candidate_id"])
            haystack = "" if source is None else ((source.get("title") or "") + "\n" +
                                                   (source.get("question_passage") or ""))
            if source is None or record["source_wording"] not in haystack:
                errors.append((line_no, "QA006 candidate missing or source wording is not verbatim"))
    return records, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("alignments", type=Path)
    parser.add_argument("--candidates", type=Path)
    args = parser.parse_args()
    records, errors = validate(args.alignments, args.candidates)
    if errors:
        for line_no, message in errors:
            print(f"{args.alignments}:{line_no}: {message}")
        raise SystemExit(1)
    for field in ("mapping_state", "support_state", "disposition"):
        counts = Counter(record[field] for record in records)
        print(field + ": " + ", ".join(f"{key}={counts[key]}" for key in sorted(counts)))
    print(f"valid question alignments: {len(records)}")


if __name__ == "__main__":
    main()
