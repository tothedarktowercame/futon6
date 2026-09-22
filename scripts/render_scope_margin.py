#!/usr/bin/env python3
"""Put a run's reading of a paper in the paper's margin, with what to distrust.

A companion to render_scope_view.py (same inputs, same source-identity rules),
laid out for reading instead of searching: each proof graph and each expository
region gets a margin note beside the passage it describes, saying what the model
made of it and which parts fail a mechanical check.

What the notes check is what a run can get wrong while every gate passes:
  proof nodes   whether a node's quoted mathematics is the passage its own gloss
                describes. S3 asks the model to cite S1 clause units by id; in
                mark7master-20260921, 87% of nodes cite span s_i as node i, so
                the quote is often a different clause from the gloss. Each node
                shows its gloss, its quote, and - when another offered unit
                matches the gloss better - that unit, as a proposed re-anchor.
  warrants      stated / cited / missing; the missing ones are the holes S9 mines.
  scopes        the expository_scope_audit flags (echo, unanchored, bare-noun)
                and whether the kind is a bare parent with specific children.

Usage: scripts/render_scope_margin.py RUN PAPER TYPESET_DIR OUTPUT
TYPESET_DIR must contain PAPER.tex, PAPER-tufte.html and conversion.log.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import expository_scope_audit as scope_audit

ALIGNED = 0.5            # share of a gloss's content words found in a quote
BETTER = 0.25            # how much better another unit must match to be proposed

# MOCK of the proposed S4 contract: a bare parent kind is not selectable, so the
# scope takes a specific child or is held. These cues only suggest which child the
# passage's own wording points to; they are a mock-up, not a classifier.
CHILD_CUES = {
    "connection/example-source": r"\bexamples?\b|for instance|\be\.g\.",
    "connection/literature-gap/terminology-origin": r"terminolog|\bcalled\b|we (?:refer to|call)|in the sense of|\bnotion of\b",
    "connection/literature-gap": r"\\cite|\bshow(?:s|ed|n)? that\b|\bprove[sd]?\b|\bresults?\b",
    "connection/application-domain": r"\bappl(?:y|ied|ies|ication)|\buseful\b",
    "connection/transfer": r"analog|thought of as|behav(?:e|ing) like|\bdual\b",
    "rationale/telos/organization-roadmap": r"\bsections?\b|\bwe (?:shall|will)\b|\bin section\b",
}


def read_edn(paths):
    return scope_audit.read_edn(list(paths))


def words(text: str) -> set[str]:
    text = re.sub(r"\\[A-Za-z]+", " ", text or "")
    return set(scope_audit.content_words(text))


def hits(gloss: set[str], text: str) -> int:
    return len(gloss & words(text))


def match(gloss: set[str], text: str) -> float:
    return hits(gloss, text) / len(gloss) if gloss else 0.0


def source_file_index(typeset: Path, paper: str) -> int:
    found = dict(re.findall(r"^Info:source-map:source\s+\[(\d+)\]\s+(.+)$",
                            (typeset / "conversion.log").read_text(), re.M))
    indices = [int(i) for i, name in found.items()
               if Path(name).resolve() == (typeset / f"{paper}.tex").resolve()]
    if len(indices) != 1:
        raise ValueError("converter log must identify exactly one matching source file")
    return indices[0]


def line_of(starts: list[int], offset: int) -> int:
    lo, hi = 0, len(starts)
    while lo < hi:
        mid = (lo + hi) // 2
        if starts[mid] <= offset:
            lo = mid + 1
        else:
            hi = mid
    return lo


def proof_note(graph: dict, candidate: dict | None, starts: list[int]) -> dict:
    offered = [dict(sp, id=f"s{i}") for i, sp in enumerate((candidate or {}).get("spans") or (), 1)]
    by_id = {sp["id"]: sp for sp in offered}
    nodes = []
    for i, n in enumerate(graph["nodes"], 1):
        gloss = words(n.get("gloss", ""))
        cited = n.get("quote-spans") or []
        own = match(gloss, n.get("text", ""))
        own_hits = hits(gloss, n.get("text", ""))
        best = max(offered, key=lambda sp: (hits(gloss, sp["text"]), -len(sp["text"])), default=None)
        best_score = match(gloss, best["text"]) if best else 0.0
        proposal = None
        # Two shared words at least: one ("structure") is coincidence, not evidence.
        if (best and best["id"] not in cited and hits(gloss, best["text"]) >= 2
                and best_score >= ALIGNED and best_score >= own + BETTER):
            proposal = {"span": best["id"], "text": best["text"], "kind": best.get("kind"),
                        "line": line_of(starts, best["start"]), "match": round(best_score, 2)}
        # Fewer than two content words (mostly formulae: "Hom(S,ΣM_0)=0") and a word
        # check cannot judge either way - say so rather than count it as a failure.
        verdict = ("uncheckable" if len(gloss) < 2
                   else "agrees" if own >= ALIGNED and own_hits >= 2
                   else "re-anchor" if proposal else "unclear")
        mock = {"agrees": ("kept", "quote matches the gloss"),
                "re-anchor": ("re-anchored", f"quote replaced by {proposal['span']}, the clause the gloss describes"
                              if proposal else ""),
                "unclear": ("unanchored", "no offered clause matches the gloss; the gloss is kept, the quote dropped"),
                "uncheckable": ("kept", "a mostly-formula gloss: this check cannot judge it")}[verdict]
        nodes.append({"id": n["id"], "kind": n.get("kind"), "gloss": n.get("gloss", ""),
                      "quote": n.get("text", ""), "cites": cited, "lines": n["source"]["lines"],
                      "sequential": cited == [f"s{i}"], "match": round(own, 2),
                      "verdict": verdict, "proposal": proposal,
                      "mock": {"verdict": mock[0], "reason": mock[1]}})
    edges = [{"id": e["id"], "relation": e.get("relation") or e.get("kind"),
              "premises": e.get("premise") or [], "conclusion": e.get("conclusion"),
              "warrant": (e.get("warrant") or {}).get("kind"),
              "why": (e.get("warrant") or {}).get("text", "")} for e in graph["edges"]]
    lo, hi = graph["source"]["lines"]
    proved = (graph.get("provenance") or {}).get("proved") or {}
    return {"type": "proof", "id": graph["passage/id"].split(":")[1], "lo": lo, "hi": hi,
            "proved": proved.get("kind"), "nodes": nodes, "edges": edges,
            "offered": len(offered)}


def children(kind: str, definitions) -> list[str]:
    return [k for k in definitions if k.startswith(kind + "/") and k.count("/") == kind.count("/") + 1]


def mock_scope(scope: dict, definitions) -> dict:
    """What the proposed S4 contract would do with this scope (a mock-up)."""
    if scope["fill"] is None:
        return {"verdict": "held", "reason": "held with a reason, as now"}
    if "unanchored" in scope["flags"]:
        return {"verdict": "rejected", "reason": "the fill is not in the lines it cites"}
    if "echo" in scope["flags"]:
        return {"verdict": "rejected", "reason": "the fill repeats its slot's definition"}
    if scope["bare-parent"]:
        # Only the fill's own words: the surrounding lines mention too much.
        options = sorted((c for c in CHILD_CUES if c in definitions and c.startswith(scope["kind"] + "/")),
                         key=lambda c: -c.count("/"))           # most specific first
        for child in options:
            if re.search(CHILD_CUES[child], scope["fill"], re.I):
                return {"verdict": "retyped", "reason": f"a specific kind is required; the passage points to {child}",
                        "suggest": child}
        return {"verdict": "held", "reason": f"a specific kind is required and none of {scope['kind']}'s "
                                             "children fits, so the scope is held"}
    if "bare-noun" in scope["flags"]:
        return {"verdict": "kept", "reason": "kept, but a bare noun says little about what the passage does"}
    return {"verdict": "kept", "reason": "passes the proposed checks"}


def build(run: Path, paper: str, typeset: Path) -> tuple[str, dict]:
    marks = json.loads((run / "artifacts/marks" / f"fable-{paper}-dp-emacs.json").read_text())
    source = marks["text"]
    if (typeset / f"{paper}.tex").read_text() != source:
        raise ValueError("typeset source differs from the run-owned marks text")
    starts = [0] + [m.end() for m in re.finditer("\n", source)]
    notes = []
    graph_paths = sorted(p for p in (run / "artifacts/graphs").glob(f"{paper}__p*.edn")
                         if not p.name.endswith(".rung2.edn"))
    for path, graph in zip(graph_paths, read_edn(graph_paths)):
        cand = run / "artifacts/candidates" / (path.stem + ".candidate.json")
        notes.append(proof_note(graph, json.loads(cand.read_text()) if cand.is_file() else None, starts))
    audit = scope_audit.audit(run / "artifacts/expo", run / "artifacts/expo-candidates")
    definitions, _ = scope_audit.vocabulary_shape()
    for s in audit["scopes"]:
        if s["paper"] == paper:
            s["mock"] = mock_scope(s, definitions)
    regions = {}
    for s in audit["scopes"]:
        if s["paper"] != paper:
            continue
        region = regions.setdefault(s["passage"], {"type": "region", "id": s["passage"].split(":")[1],
                                                   "lo": None, "hi": None, "scopes": []})
        region["scopes"].append(s)
    for region in regions.values():
        lines = [x for s in region["scopes"] for x in s["lines"]]
        region["lo"], region["hi"] = min(lines), max(lines)
        m = re.search(r"L(\d+)-(\d+)$", region["scopes"][0]["passage"])
        if m:
            region["lo"], region["hi"] = int(m.group(1)), int(m.group(2))
    notes += list(regions.values())
    notes.sort(key=lambda n: (n["lo"], n["hi"]))
    nodes = [x for n in notes if n["type"] == "proof" for x in n["nodes"]]
    edges = [x for n in notes if n["type"] == "proof" for x in n["edges"]]
    scopes = [x for n in notes if n["type"] == "region" for x in n["scopes"]]
    summary = {"paper": paper, "run": json.loads((run / "run-manifest.json").read_text())["run-id"],
               "proofs": sum(n["type"] == "proof" for n in notes),
               "nodes": len(nodes), "sequential": sum(x["sequential"] for x in nodes),
               "agrees": sum(x["verdict"] == "agrees" for x in nodes),
               "re-anchor": sum(x["verdict"] == "re-anchor" for x in nodes),
               "uncheckable": sum(x["verdict"] == "uncheckable" for x in nodes),
               "edges": len(edges),
               "warrants": {k: sum(e["warrant"] == k for e in edges)
                            for k in ("claim", "citation", "missing-warrant")},
               "regions": sum(n["type"] == "region" for n in notes), "scopes": len(scopes),
               "scopes-flagged": sum(bool(s["flags"]) for s in scopes),
               "scopes-bare-parent": sum(s["bare-parent"] for s in scopes),
               "mock": {"nodes": {v: sum(x["mock"]["verdict"] == v for x in nodes)
                                  for v in ("kept", "re-anchored", "unanchored")},
                        "scopes": {v: sum(x["mock"]["verdict"] == v for x in scopes)
                                   for v in ("kept", "retyped", "held", "rejected")}}}
    payload = {"file": source_file_index(typeset, paper), "summary": summary, "notes": notes}
    page = (typeset / f"{paper}-tufte.html").read_text()
    if "data-sourcepos=" not in page:
        raise ValueError("typeset page has no source positions")
    assets = Path(__file__).with_name("scope_margin")
    data = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    page = page.replace("</head>", "<style>" + (assets / "margin.css").read_text() + "</style></head>", 1)
    page = page.replace("</body>", '<script id="m7-data" type="application/json">' + data
                        + "</script><script>" + (assets / "margin.js").read_text() + "</script></body>", 1)
    return page, summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("paper")
    ap.add_argument("typeset", type=Path)
    ap.add_argument("output", type=Path)
    a = ap.parse_args()
    page, summary = build(a.run, a.paper, a.typeset)
    a.output.write_text(page)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
