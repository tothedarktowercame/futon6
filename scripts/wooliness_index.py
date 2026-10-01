#!/usr/bin/env python3
"""Report deterministic pre-call wooliness from frozen Mark8 inputs."""
from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import re
from typing import Any, Iterable


SCHEMA = "futon6/mark8-wooliness/v1"
WEIGHTS = {"U": 0.45, "C": 0.25, "D": 0.30}
MAX_HOPS = 3
# This is also the executable security boundary: load_role refuses every other role.
PRECALL_INPUT_ROLES = frozenset({"candidate", "marks", "strategies", "citation-index",
                                 "concept-encyclopedia"})
CANDIDATE_SCHEMAS = frozenset({"iatc-candidate/v5-proof", "expo-candidate/v2"})
MODEL_DERIVED_FIELDS = frozenset({
    "model-output", "model_output", "graph", "outcome", "comprehension",
    "completion", "model-response", "model_response",
})


def model_derived_fields(value: Any) -> set[str]:
    """Forbidden model-result containers found anywhere in a candidate payload."""
    if isinstance(value, dict):
        found = {str(key) for key in value if str(key) in MODEL_DERIVED_FIELDS}
        return found | set().union(*(model_derived_fields(child) for child in value.values()), set())
    if isinstance(value, list):
        return set().union(*(model_derived_fields(child) for child in value), set())
    return set()


def load_role(role: str, path: Path) -> Any:
    if role not in PRECALL_INPUT_ROLES:
        raise ValueError(f"input role is not a frozen pre-call source: {role}")
    doc = json.loads(Path(path).read_text())
    valid = {
        "candidate": lambda d: d.get("schema") in CANDIDATE_SCHEMAS and
                               all(key in d for key in ("paper-id", "passage-id", "window-lines")),
        # Mark JSON predates schema tags; its structural contract is text plus standoff marks.
        "marks": lambda d: isinstance(d.get("text"), str) and isinstance(d.get("marks"), list),
        "strategies": lambda d: d.get("schema") == "markup-strategies/v1" and
                                  isinstance(d.get("terms"), list),
        "citation-index": lambda d: d.get("schema") == "futon6/h7-cite-resolution/v1" and
                                      isinstance(d.get("records"), list),
        "concept-encyclopedia": lambda d: d.get("schema") == "concept-encyclopedia-v0" and
                                          isinstance(d.get("entries"), list),
    }[role]
    if not isinstance(doc, dict) or not valid(doc):
        raise ValueError(f"{path}: does not satisfy frozen {role} schema")
    if role == "candidate":
        forbidden = sorted(model_derived_fields(doc))
        if forbidden:
            raise ValueError(f"{path}: candidate contains model-derived field(s): {forbidden}")
    return doc


def norm(term: str) -> str:
    return re.sub(r"\s+", " ", term.strip().lower())


def line_starts(text: str) -> list[int]:
    return [0] + [match.end() for match in re.finditer("\n", text)]


def char_span(text: str, lines: list[int]) -> tuple[int, int]:
    if (len(lines) != 2 or lines[0] < 1 or lines[1] < lines[0]):
        raise ValueError(f"invalid candidate window-lines: {lines!r}")
    starts = line_starts(text)
    if lines[0] > len(starts):
        raise ValueError(f"candidate starts beyond marks text: {lines!r}")
    start = starts[lines[0] - 1]
    end = starts[lines[1]] if lines[1] < len(starts) else len(text)
    return start, end


def fields(mark: dict) -> dict[str, str]:
    return {str(k): str(v) for k, v in mark.get("fields", [])}


def concepts_in(marks_doc: dict, span: tuple[int, int]) -> dict[str, int]:
    text = str(marks_doc["text"])
    lo, hi = span
    out: dict[str, int] = {}
    for mark in marks_doc.get("marks", []):
        if mark.get("kind") != "concept" or not (int(mark["start"]) < hi and lo < int(mark["end"])):
            continue
        term = fields(mark).get("term") or text[int(mark["start"]):int(mark["end"])]
        if norm(term):
            key = norm(term)
            out[key] = min(out.get(key, int(mark["start"])), int(mark["start"]))
    return out


def encyclopedia_index(doc: dict) -> tuple[set[str], dict[str, set[str]]]:
    known, definitions = set(), {}
    for entry in doc.get("entries", []):
        term = norm(str(entry.get("concept", "")))
        if not term:
            continue
        known.add(term)
        papers = {str(p) for p in (entry.get("defined_in") or {}).get("sample", [])}
        gloss_paper = (entry.get("gloss") or {}).get("paper")
        if gloss_paper:
            papers.add(str(gloss_paper))
        definitions[term] = papers
    return known, definitions


def local_definitions(strategies: dict, first_uses: dict[str, int]) -> set[str]:
    """Terms whose witnessed definition precedes their first use in this passage."""
    return {term for row in strategies.get("terms", [])
            if (term := norm(str(row.get("term", "")))) in first_uses
            and isinstance(row.get("at"), int) and row["at"] <= first_uses[term]}


def citation_records_in(doc: dict, span: tuple[int, int]) -> list[dict]:
    lo, hi = span
    return [row for row in doc.get("records", [])
            if isinstance(row.get("char-anchor"), list) and len(row["char-anchor"]) == 2
            and row["char-anchor"][0] < hi and lo < row["char-anchor"][1]]


def citation_graph(docs: Iterable[dict]) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = {}
    for doc in docs:
        source = str(doc["paper-id"])
        graph.setdefault(source, set()).update(
            str(row["resolved-corpus-id"]) for row in doc.get("records", [])
            if row.get("resolved-corpus-id"))
    return graph


def distance(source: str, targets: set[str], graph: dict[str, set[str]]) -> int:
    if source in targets:
        return 0
    queue = deque([(source, 0)])
    seen = {source}
    while queue:
        node, hops = queue.popleft()
        if hops == MAX_HOPS:
            continue
        for nxt in sorted(graph.get(node, ())):
            if nxt in targets:
                return hops + 1
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, hops + 1))
    return MAX_HOPS


def score(candidate: dict, marks: dict, strategies: dict, citation_doc: dict,
          encyclopedia: dict, graph: dict[str, set[str]]) -> dict:
    span = char_span(str(marks["text"]), candidate["window-lines"])
    occurrences = concepts_in(marks, span)
    terms = set(occurrences)
    known, definition_papers = encyclopedia_index(encyclopedia)
    prior = local_definitions(strategies, occurrences)
    grounded = prior | known
    u = sum(term not in grounded for term in terms) / len(terms) if terms else 0.0
    citations = citation_records_in(citation_doc, span)
    c = (sum(not row.get("resolved-corpus-id") for row in citations) / len(citations)
         if citations else 0.0)
    distances = [0 if term in prior else distance(str(candidate["paper-id"]),
                                                  definition_papers.get(term, set()), graph)
                 for term in sorted(terms)]
    d = sum(distances) / (MAX_HOPS * len(distances)) if distances else 0.0
    w = min(1.0, max(0.0, WEIGHTS["U"] * u + WEIGHTS["C"] * c + WEIGHTS["D"] * d))
    return {"paper-id": str(candidate["paper-id"]), "passage-id": str(candidate["passage-id"]),
            "window-lines": candidate["window-lines"], "U": u, "C": c, "D": d, "W": w,
            "counts": {"used-terms": len(terms), "prior-defined-terms": len(terms & prior),
                       "citations": len(citations),
                       "resolved-citations": sum(bool(row.get("resolved-corpus-id")) for row in citations)}}


def build(candidates: list[dict], marks_by_paper: dict[str, dict],
          strategies_by_paper: dict[str, dict], citations_by_paper: dict[str, dict],
          encyclopedia: dict) -> dict:
    graph = citation_graph(citations_by_paper.values())
    records = [score(candidate, marks_by_paper[str(candidate["paper-id"])],
                     strategies_by_paper[str(candidate["paper-id"])],
                     citations_by_paper.get(str(candidate["paper-id"]),
                                            {"paper-id": candidate["paper-id"], "records": []}),
                     encyclopedia, graph) for candidate in candidates]
    records.sort(key=lambda row: (row["paper-id"], row["passage-id"]))
    return {"schema": SCHEMA, "weights": WEIGHTS, "max-citation-hops": MAX_HOPS,
            "input-roles": sorted(PRECALL_INPUT_ROLES), "records": records}


def load_named(directory: Path, suffix: str, role: str) -> dict[str, dict]:
    out = {}
    for path in sorted(Path(directory).glob(f"*{suffix}")):
        doc = load_role(role, path)
        paper = str(doc.get("paper-id") or path.name.removesuffix(suffix).removeprefix("fable-"))
        out[paper] = doc
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--marks-dir", type=Path, required=True)
    parser.add_argument("--strategies-dir", type=Path, required=True)
    parser.add_argument("--citation-dir", type=Path, required=True)
    parser.add_argument("--concept-encyclopedia", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    candidates = [load_role("candidate", path) for path in sorted(args.candidate)]
    marks = load_named(args.marks_dir, "-dp-emacs.json", "marks")
    strategies = load_named(args.strategies_dir, ".strategies.json", "strategies")
    citations = load_named(args.citation_dir, ".cite-resolution.json", "citation-index")
    encyclopedia = load_role("concept-encyclopedia", args.concept_encyclopedia)
    args.out.write_text(json.dumps(build(candidates, marks, strategies, citations, encyclopedia),
                                   indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
