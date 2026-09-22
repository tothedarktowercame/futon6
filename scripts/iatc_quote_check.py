#!/usr/bin/env python3
"""Does a node quote the clause its own gloss describes? A measurement, not a gate.

S3 asks the model for a prose gloss AND the ids of the marked units that state the
node. The gates check that the cited units exist and lie in the passage, which they
always do, so a node can quote a clause that has nothing to do with its gloss and
every check still passes. Under the v3 contract the units were never listed in the
prompt, and in mark7master-20260921 87% of nodes cited unit s_i as node i.

This compares each node's gloss with the text it quotes, by shared content words:

  agrees        at least half the gloss's words, and at least two, are in the quote
  re-anchor     another offered unit matches the gloss better by that measure
  unclear       neither the quote nor any offered unit shares enough words
  uncheckable   the gloss has fewer than two content words (mostly formulae)

It is lexical. It cannot judge a paper whose prose is not in the glosses' language
(math/0409598 is French), and a gloss that paraphrases heavily reads as `unclear`.
Read it as a rate to compare between runs, not as a verdict on a single node.

Usage: scripts/iatc_quote_check.py --graphs DIR --candidates DIR [--out JSON]
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))

import argparse
import json
import re
import sys
from pathlib import Path

import expository_scope_audit as scope_audit

ALIGNED = 0.5            # share of a gloss's content words found in a quote
BETTER = 0.25            # how much better another unit must match to be proposed


def words(text: str) -> set[str]:
    return set(scope_audit.content_words(re.sub(r"\\[A-Za-z]+", " ", text or "")))


def hits(gloss: set[str], text: str) -> int:
    return len(gloss & words(text))


def match(gloss: set[str], text: str) -> float:
    return hits(gloss, text) / len(gloss) if gloss else 0.0


def node_check(node: dict, offered: list[dict], line_of=None) -> dict:
    """One node's verdict, and the unit that would fit its gloss better."""
    gloss = words(node.get("gloss", ""))
    cited = node.get("quote-spans") or node.get("quote_spans") or []
    quote = node.get("text", "")
    own, own_hits = match(gloss, quote), hits(gloss, quote)
    best = max(offered, key=lambda sp: (hits(gloss, sp["text"]), -len(sp["text"])), default=None)
    best_score = match(gloss, best["text"]) if best else 0.0
    proposal = None
    # Two shared words at least: one ("structure") is coincidence, not evidence.
    if (best and best["id"] not in cited and hits(gloss, best["text"]) >= 2
            and best_score >= ALIGNED and best_score >= own + BETTER):
        proposal = {"span": best["id"], "text": best["text"], "kind": best.get("kind"),
                    "at": [best["start"], best["end"]], "match": round(best_score, 2)}
        if line_of:
            proposal["line"] = line_of(best["start"])
    verdict = ("uncheckable" if len(gloss) < 2
               else "agrees" if own >= ALIGNED and own_hits >= 2
               else "re-anchor" if proposal else "unclear")
    return {"verdict": verdict, "match": round(own, 2), "cites": cited, "proposal": proposal}


def sequential(node_index: int, cited: list[str], offered: list[dict]) -> bool:
    """Whether this node cites exactly the node-th unit: the counting answer."""
    return len(cited) == 1 and node_index <= len(offered) and cited[0] == offered[node_index - 1]["id"]


def check_graph(graph: dict, candidate: dict | None) -> dict:
    import iatc_json
    offered = iatc_json.source_spans(candidate or {})
    rows = []
    for i, node in enumerate(graph.get("nodes") or [], 1):
        row = node_check(node, offered)
        row.update(id=node.get("id"), sequential=sequential(i, row["cites"], offered))
        rows.append(row)
    return {"passage": graph.get("passage/id"), "offered": len(offered), "nodes": rows}


def tally(graphs: list[dict]) -> dict:
    rows = [r for g in graphs for r in g["nodes"]]
    counts = {v: sum(r["verdict"] == v for r in rows)
              for v in ("agrees", "re-anchor", "unclear", "uncheckable")}
    checkable = len(rows) - counts["uncheckable"]
    return {"graphs": len(graphs), "nodes": len(rows), "checkable": checkable, **counts,
            "sequential": sum(r["sequential"] for r in rows),
            "agrees-share": round(counts["agrees"] / checkable, 3) if checkable else None,
            "sequential-share": round(sum(r["sequential"] for r in rows) / len(rows), 3) if rows else None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graphs", type=Path, required=True)
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    paths = sorted(p for p in a.graphs.glob("*.edn") if not p.name.endswith(".rung2.edn"))
    out = []
    for path, graph in zip(paths, scope_audit.read_edn(paths)):
        cand = a.candidates / f"{path.stem}.candidate.json"
        out.append(check_graph(graph, json.loads(cand.read_text()) if cand.is_file() else None))
    summary = tally(out)
    if a.out:
        a.out.write_text(json.dumps({"summary": summary, "graphs": out}, ensure_ascii=False))
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
