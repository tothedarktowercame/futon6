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

All three layers also show at once, as ink plates (scope_margin/plates.js): S1 marks
on cyan, S3 proof-graph quotes on magenta, S4 scope lines on yellow, placed on each
word and formula by its offset in the run's source, so overlaps mix like ink. Every
label the page shows is defined in one glossary (MARK_KINDS, NODE_MEANING,
WARRANT_MEANING, the S4 vocabulary, PAGE_TERMS); an S1 kind without a definition
stops the build. The terms the paper itself defines are underlined by how S1 tagged
each use: as the paper's defined term, only as a generic lexicon word, or not at all.

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

import expository_region_extract as region_extract
import markup_strategies
import expository_scope_audit as scope_audit
import iatc_quote_check as quote_check
import iatc_json

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


# What each S1 mark kind means, read from the code that emits it (dp_paper_view.build
# and the detectors it calls). The group says which cyan plate draws it: "term" marks
# name one symbol or word, "clause" marks a phrase or sentence, "region" a whole
# environment or display, and "structure" is bookkeeping no plate draws. A kind the
# run emits and this table does not define stops the build: an undefined label on
# the page is exactly what the page exists to rule out.
MARK_KINDS = {
    "symbol": ("term", "a letter or identifier inside math for which S1 found no binding or other grounding"),
    "symbol-grounded": ("term", "a letter or identifier inside math that S1 tied to a meaning: the nearest earlier "
                        "binding (Let $X$ be ..., $X$ is a ...), its base symbol, an X := display, or a known operator name"),
    "classified": ("term", "a control sequence inside math with a known role: a macro the paper itself defines "
                   "(author-defined), a LaTeXML standard-math command, or plain TeX"),
    "unknown": ("term", "a control sequence inside math S1 could not identify (here mostly xymatrix diagram commands)"),
    "role-gap": ("term", "a macro the paper defines whose role S1 could not work out"),
    "concept-typed": ("term", "a macro the paper defines, typed as a concept"),
    "concept": ("term", "a prose term, never overlapping math: a match in the background lexicon (the only source "
                "with a grounding target), an emphasised phrase, a term the paper defines, or an undefined noun phrase"),
    "definiendum": ("term", "the $symbol$ introduced by a Let $X$ be ... or $X$ is a ... sentence; prose terms "
                    "are never definienda"),
    "definiens": ("clause", "the phrase that says what a definiendum is (the 'a triangulated category' of Let $T$ be "
                  "a triangulated category)"),
    "cite": ("term", "a \\cite command; its target is the bibliography"),
    "ref": ("term", "a \\ref-style command; in-paper if its label exists in the paper, otherwise dangling"),
    "label": ("structure", "a \\label command, or an enumerate item label, with the environment it names"),
    "anaphor": ("term", "a phrase that refers back to something bound earlier"),
    "let-binder": ("clause", "a whole Let $X$ be/denote ... or $X$ is a ... sentence"),
    "bind/let": ("clause", "a scope opened by Let $X$ be ..., Fix $X$, Take x = ..."),
    "bind/define": ("clause", "a scope opened by Define ..., denote by $X$, write $X$ for, $X$ is called ..."),
    "bind/typed": ("clause", "a formula containing an arrow (f : A -> B); the symbol before the colon is bound"),
    "assume/explicit": ("clause", "Assume/Suppose (that) ..., or If $...$, with the condition recorded"),
    "assume/consider": ("clause", "Consider ..., Choose $X$"),
    "quant/universal": ("clause", "For all/each/every/any ... $x$ (in $S$), or \\forall inside math"),
    "quant/existential": ("clause", "There is/exists $x$ ..., or \\exists inside math"),
    # A large operator binds its index: the i of \\sum_{i \\in I}. One kind per operator.
    "bind/summation": ("clause", "a summation whose subscript binds an index variable"),
    "bind/product": ("clause", "a product whose subscript binds an index variable"),
    "bind/coprod": ("clause", "a coproduct whose subscript binds an index variable"),
    "bind/big-union": ("clause", "a big union whose subscript binds an index variable"),
    "bind/big-intersection": ("clause", "a big intersection whose subscript binds an index variable"),
    "bind/integral": ("clause", "an integral whose subscript binds a variable of integration"),
    "constrain/relation": ("clause", "a formula containing a relation (=, <, \\in, \\subseteq, \\cong, ...)"),
    "constrain/such-that": ("clause", "a formula containing \\in (set membership); overlaps constrain/relation"),
    "constrain/where": ("clause", "where $x$ is/denotes ... or where x = ..."),
    "implies": ("clause", "two consecutive sentences Let/Given/Suppose ... . Then/Hence/Thus ... ."),
    "kw-hyp": ("term", "the Let/Given/Suppose/Assume keyword of an implies span"),
    "kw-con": ("term", "the Then/Hence/Thus/Therefore keyword of an implies span"),
    "inference": ("term", "a connective word (implies, follows from, iff, Thus, Hence, the then of If ..., then); "
                  "deductive inside a statement or proof, body elsewhere"),
    "claim": ("clause", "the subject or object clause of an inference connective"),
    "proof-move": ("clause", "a hedging phrase: clearly, it is easy to see, left to the reader, it suffices to show, "
                   "without loss of generality"),
    "math": ("region", "a whole $...$ span or display environment"),
    "exposition": ("region", "a prose region picked out for S4, with its section title"),
    "text-mode": ("structure", "letters inside \\text/\\mbox/\\textit within math: prose, not symbols"),
    "layout": ("structure", "letters inside math that are environment names, units, label keys or column specs"),
}
ENV_MEANING = ("region", "a whole \\begin{NAME}...\\end{NAME} environment (or the author's macro for one), "
               "named by its canonical kind")

# S3's node kinds and warrant kinds, in the words of the S3 prompt (mark3_iatc_loop).
NODE_MEANING = {"claim": "an assertion", "object": "a mathematical object the proof introduces or constructs",
                "definition": "a definition the proof uses or makes", "ref": "a result the proof points to"}
WARRANT_MEANING = {"claim": "stated: the proof itself gives the reason, in the text",
                   "citation": "cited: the step points to a result",
                   "missing-warrant": "missing: the text asserts the step without saying why (including "
                                      "'clearly' or 'a routine computation'); the elided fact is a hole S9 mines"}

# Labels this page adds of its own, keyed as they appear (margin.js strips a leading
# count and "mock: "); a label the page shows without an entry here is flagged on it.
PAGE_TERMS = {
    "stated": WARRANT_MEANING["claim"], "cited": WARRANT_MEANING["citation"],
    "missing": WARRANT_MEANING["missing-warrant"],
    "quote matches gloss": "at least half the gloss's content words, and at least two, appear in the text the node quotes",
    "quotes match": "nodes whose quote matches their gloss (see: quote matches gloss)",
    "quote is another clause": "the node's quote does not match its gloss, and another clause offered to S3 does",
    "quote another clause": "nodes whose quote is another clause (see: quote is another clause)",
    "no clause matches": "neither the node's quote nor any offered clause shares enough words with the gloss",
    "formula: can't check": "the gloss has fewer than two content words, so a word check cannot judge it",
    "formula-only": "nodes with a mostly-formula gloss (see: formula: can't check)",
    "cite sᵢ in order": "node i cites clause unit s_i: the citations follow the order of the list, not the text",
    "pass the checks": "S4 scopes with no mechanical flag",
    "generic kind": "an S4 scope kind that has more specific children in the vocabulary",
    "flagged": "an S4 scope that fails a mechanical check (the three below)",
    "repeats its slot definition": "echo: the fill restates the definition of its slot instead of the passage",
    "not in the lines it cites": "unanchored: the fill's words are not in the source lines the scope cites",
    "a bare noun phrase": "bare-noun: the fill is a noun phrase that says little about what the passage does",
    "kept": "mock: the output stands",
    "re-anchored": "mock: the node's quote is replaced by the clause its gloss describes",
    "unanchored": "mock: the node keeps its gloss and loses its quote",
    "retyped": "mock: the scope takes the specific child kind its own wording points to",
    "held": "mock: the scope is held with a reason",
    "rejected": "mock: the scope is dropped",
    "defined term": "a term the paper typesets (\\textit, \\emph, \\textbf) as the thing being defined, inside one "
                    "of S1's definition environments or a phrase S1's definition miner found",
    "tagged as defined": "an occurrence S1 marked as a concept the paper defines",
    "tagged generically": "an occurrence S1 marked only as a background-lexicon word or phrase "
                          "(e.g. co-t-structure as lexicon:structure)",
    "untagged": "an occurrence of a defined term that no S1 concept mark touches",
    "quoted": "the scope's fill is in the passage word for word, so the words it is about can be shown",
    "in the model\u2019s words": "the fill is not in the passage: it is the model's paraphrase, and only "
                                  "the cited lines can be shown",
    "leaf-section": "S4 region: a whole section with no formal block in it",
    "inflight": "S4 region: prose between two formal blocks of the same section and depth",
    "section-lead": "S4 region: a section's prose before its first formal block",
    "section-tail": "S4 region: a section's prose after its last formal block",
    "in-proof": "S4 region: prose between displays inside a proof; S3 also reads it",
    "in-environment": "binding rule: a binding of this symbol earlier in the same environment",
    "proved-statement": "binding rule: inside a proof, a binding in the statement it proves",
    "in-section": "binding rule (assumed): the nearest earlier binding in this section",
    "in-paper": "binding rule (assumed): the nearest earlier binding anywhere before it",
    "unbound": "no binding of this symbol's name occurs before this point",
    "definiendum": "binding site: the $symbol$ of a Let $X$ be ... sentence (S1)",
    "bind/let": "binding site: a Let/Fix/Take scope (S1)",
    "bind/define": "binding site: a Define/denote/is called scope (S1)",
    "bind/typed": "binding site: a formula with an arrow, f : A -> B (S1)",
    "apposition": "binding site: an article-noun apposition, \"An object $S$ of $T$\" (this page's strategy, not S1)",
    "quantifier": "binding site: for all/each/every $x$ (this page's strategy, not S1)",
}


def mark_kind(kind: str) -> tuple[str, str]:
    if kind in MARK_KINDS:
        return MARK_KINDS[kind]
    if kind.startswith("env/"):
        return ENV_MEANING
    raise ValueError(f"S1 mark kind {kind!r} has no definition in MARK_KINDS")


def grounded(mark: dict) -> bool:
    """Whether S1 tied this mark to a meaning, rather than only tagging it."""
    fields = dict(mark.get("fields") or [])
    if mark["kind"] == "concept":
        return bool(fields.get("grounded")) or fields.get("source") == "defined-in-paper"
    if mark["kind"] == "ref":
        return "dangling" not in (mark.get("tip") or "")
    return mark["kind"] not in ("symbol", "unknown", "role-gap")


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
    import iatc_json
    offered = iatc_json.source_spans(candidate or {})
    by_id = {sp["id"]: sp for sp in offered}
    nodes = []
    for i, n in enumerate(graph["nodes"], 1):
        # One definition of this check, shared with the run's measurement (S3's tail).
        row = quote_check.node_check(n, offered, line_of=lambda off: line_of(starts, off))
        cited, verdict, proposal = row["cites"], row["verdict"], row["proposal"]
        mock = {"agrees": ("kept", "quote matches the gloss"),
                "re-anchor": ("re-anchored", f"quote replaced by {proposal['span']}, the clause the gloss describes"
                              if proposal else ""),
                "unclear": ("unanchored", "no offered clause matches the gloss; the gloss is kept, the quote dropped"),
                "uncheckable": ("kept", "a mostly-formula gloss: this check cannot judge it")}[verdict]
        nodes.append({"id": n["id"], "kind": n.get("kind"), "gloss": n.get("gloss", ""),
                      "quote": n.get("text", ""), "cites": cited, "lines": n["source"]["lines"],
                      "at": [[by_id[c]["start"], by_id[c]["end"]] for c in cited if c in by_id],
                      "sequential": quote_check.sequential(i, cited, offered), "match": row["match"],
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
    # Compare the bytes: 0708.2185's marks text holds 28 CRLF line endings, and reading
    # the file in text mode turns them into LF, so an exact copy would look different.
    if (typeset / f"{paper}.tex").read_bytes() != source.encode():
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
            # Where the scope's own words sit, so a reader can be shown them instead of
            # the whole line. None when the fill is not quoted from the source at all.
            lo, hi = s["lines"]
            a = starts[lo - 1] if 0 < lo <= len(starts) else 0
            b = starts[hi] if 0 < hi < len(starts) else len(source)
            if not s.get("span") and s["fill"]:
                s["span"] = scope_audit.locate(s["fill"], source[a:b], a)
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
               "scopes-filled": sum(s["fill"] is not None for s in scopes),
               "scopes-quoted": sum(bool(s.get("span")) for s in scopes),
               "mock": {"nodes": {v: sum(x["mock"]["verdict"] == v for x in nodes)
                                  for v in ("kept", "re-anchored", "unanchored")},
                        "scopes": {v: sum(x["mock"]["verdict"] == v for x in scopes)
                                   for v in ("kept", "retyped", "held", "rejected")}}}
    body = source.find("\\begin{document}")
    kinds = sorted({m["kind"] for m in marks["marks"]})
    kind_ix = {k: i for i, k in enumerate(kinds)}
    strat = markup_strategies.strategies(source, marks["marks"])
    terms = strat["terms"]
    s4_used = sorted({s["kind"] for s in scopes} | {s["mock"].get("suggest") for s in scopes if s["mock"].get("suggest")})
    glossary = {
        "S1 mark kinds": {k: mark_kind(k)[1] for k in kinds},
        "S3 node kinds": NODE_MEANING,
        "S3 warrants": WARRANT_MEANING,
        "S4 scope kinds": {k: definitions[k] for k in s4_used},
        "This page": PAGE_TERMS,
    }
    if set(NODE_MEANING) != set(iatc_json.NODE_KINDS) or set(WARRANT_MEANING) != set(iatc_json.WARRANT_KINDS.values()):
        raise ValueError("S3's node or warrant kinds changed; update NODE_MEANING / WARRANT_MEANING")
    summary["terms"] = {"defined": len(terms), "occurrences": sum(len(t["uses"]) for t in terms),
                        **{st: sum(u["s1"] == st for t in terms for u in t["uses"])
                           for st in ("tagged as defined", "tagged generically", "untagged")}}
    summary["strategies"] = strat["summary"]
    # The regions S4 would read now, beside the ones this run's extractor carved.
    was = region_extract.extract_regions(paper, source)
    now = region_extract.extract_regions(paper, source, marks["marks"])
    carved = [{"id": r["region_id"], "type": r["type"], "section": r["section_title"],
               "lines": [r["line_start"], r["line_end"]]} for r in now["regions"]]
    summary["carving"] = {"run": {"regions": len(was["regions"]), **was["coverage"]},
                          "now": {"regions": len(now["regions"]), **now["coverage"],
                                  "types": {t: sum(r["type"] == t for r in carved) for t in sorted({r["type"] for r in carved})}}}
    payload = {"file": source_file_index(typeset, paper), "summary": summary, "notes": notes, "carved": carved,
               "starts": starts, "body": body,
               "kinds": [[k, *mark_kind(k)] for k in kinds],
               "marks": [[m["start"], m["end"], kind_ix[m["kind"]], int(grounded(m)), m.get("tip") or ""]
                         for m in marks["marks"]],
               "strategies": strat, "glossary": glossary}
    page = (typeset / f"{paper}-tufte.html").read_text()
    if "data-sourcepos=" not in page:
        raise ValueError("typeset page has no source positions")
    assets = Path(__file__).with_name("scope_margin")
    data = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    page = page.replace("</head>", "<style>" + (assets / "margin.css").read_text() + "</style></head>", 1)
    page = page.replace("</body>", '<script id="m7-data" type="application/json">' + data
                        + "</script><script>" + (assets / "margin.js").read_text()
                        + '</script><script id="m7-source" type="application/json">'
                        + json.dumps(source, ensure_ascii=False).replace("<", "\\u003c")
                        + "</script><script>" + (assets / "plates.js").read_text() + "</script></body>", 1)
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
