#!/usr/bin/env python3
"""Typed hole inventory — holes keyed by (hole-type, grounded concept).

E-comprehension-foundation build step 1: extend R2d from "concepts a proof
uses" to "the concept a HOLE wants".  The raw :wanted slug is paper-local
(zero cross-paper recurrence, warrant_normalize.py, 2026-06-22), so the
recurring/foundational key is (type, concept) with the concept drawn from the
R2d substrate — exactly like the noun vocabulary, but for holes.

For each hole in each IATC argument graph we ground the CONCLUSION node of the
holed edge (the claim the missing warrant was supposed to support — i.e. what
the hole wants) against the concept index.  A hole whose target concept is in
the substrate is keyed by (hole-kind, concept); recurrence across papers is
counted, which the raw slug can never show.  Ungrounded holes are reported
separately (still keyed by type, concept=None) so nothing is silently dropped.

Usage:
  futon6/.venv/bin/python scripts/clean_typed_holes.py \
      [--graphs data/iatc-argument-graphs/loop-run-70b] \
      [--out holes/excursions/E-comprehension-foundation.typed-holes.json]

  Corpus-wide (2026-11-06): pass several set directories to aggregate across
  graph sets.  A paper reprocessed in a later set (run/run-retry) is the SAME
  paper: it contributes its pid once, and the key's "sets" field records where
  it was seen.
  futon6/.venv/bin/python scripts/clean_typed_holes.py \
      --graphs data/iatc-argument-graphs/anchortest \
               data/iatc-argument-graphs/gh200 ... \
      --out holes/excursions/E-comprehension-foundation.typed-holes-corpus.json
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import r2d_concept_coverage as r2d  # noqa: E402


def hole_targets(graph):
    """Yield (hole_kind, edge_id, conclusion_node_text) for each declared hole."""
    for h in graph.get("holes", []):
        edge_id = str(h.get("edge", "")).lstrip(":")
        yield (str(h.get("kind", "unknown")).lstrip(":"), edge_id, h.get("wanted"))


def conclusion_text(graph, edge_id):
    """The text of the conclusion node of one edge (what the warrant supports)."""
    concl = None
    for e in graph.get("edges", []):
        if str(e.get("id", "")).lstrip(":") == edge_id:
            concl = e.get("conclusion")
            break
    for n in graph.get("nodes", []):
        if str(n.get("id", "")).lstrip(":") == str(concl).lstrip(":"):
            return n.get("text", "")
    return ""


def concepts_for_text(text, concept_index):
    """Concepts the R2d substrate recognizes in one node's text — the same
    n-gram/pattern candidate extraction the noun side uses, pointed at the
    holed edge's conclusion (what the missing warrant was supposed to
    establish, i.e. what the hole wants)."""
    if not text:
        return set()
    known = set(concept_index)
    out = set(r2d.ngram_candidates(text, known)) | set(r2d.pattern_candidates(text))
    exact = r2d.normalize_phrase(text)
    if exact in known:
        out.add(exact)
    return {c for c in out if not r2d.sfc.boilerplate_phrase(c)}


def typed_holes_for_graph(graph, substrate):
    """[(hole_kind, concept_or_None, wanted_slug)] for one graph, concepts
    grounded against the R2d substrate."""
    out = []
    for kind, edge_id, wanted in hole_targets(graph):
        text = conclusion_text(graph, edge_id)
        concepts = concepts_for_text(text, substrate.concept_index)
        if concepts:
            for c in sorted(concepts):
                out.append((kind, c, wanted))
        else:
            out.append((kind, None, wanted))
    return out


def _add_paper(path, substrate, keyed, ungrounded, set_name):
    pid = Path(path).stem
    graph = r2d.load_edn(Path(path))
    for kind, concept, wanted in typed_holes_for_graph(graph, substrate):
        if concept is None:
            ungrounded.append({"pid": pid, "type": kind,
                               "wanted": str(wanted).lstrip(":")})
            continue
        key = (kind, concept)
        keyed[key]["papers"].add(pid)
        keyed[key]["sets"].add(set_name)
        keyed[key]["n"] += 1
        if wanted is not None:
            keyed[key]["wanted"].append(str(wanted).lstrip(":"))


def _rows(keyed, ungrounded):
    rows = [{"type": k[0], "concept": k[1], "occurrences": v["n"],
             "papers": sorted(v["papers"]), "sets": sorted(v["sets"]),
             "wanted": sorted(set(v["wanted"]))}
            for k, v in sorted(keyed.items())]
    rows.sort(key=lambda r: (-r["occurrences"], r["type"], r["concept"]))
    return rows


def inventory(graph_paths, substrate, set_name="default"):
    """Aggregate the typed-hole inventory for one graph set: keyed entries
    with recurrence.  (Paper ids are unique within a set, so no dedupe here.)"""
    keyed = defaultdict(lambda: {"papers": set(), "sets": set(), "n": 0,
                                 "wanted": []})
    ungrounded = []
    for path in graph_paths:
        _add_paper(path, substrate, keyed, ungrounded, set_name)
    rows = _rows(keyed, ungrounded)
    return {"n_graph_sets": 1, "n_papers": len(graph_paths),
            "n_holes_keyed": sum(r["occurrences"] for r in rows),
            "n_distinct_keys": len(rows),
            "recurring": [r for r in rows if len(r["papers"]) > 1],
            "ungrounded": ungrounded,
            "typed_holes": rows}


def corpus_inventory(graph_set_pairs, substrate):
    """Aggregate across graph SETS.  A paper reprocessed in a later set
    (run/run-retry) is the SAME paper: each pid is counted once, from its
    first set, and each key's "sets" field records where it was seen."""
    keyed = defaultdict(lambda: {"papers": set(), "sets": set(), "n": 0,
                                 "wanted": []})
    ungrounded = []
    seen = {}
    for path, set_name in graph_set_pairs:
        pid = Path(path).stem
        if pid in seen:
            continue
        seen[pid] = set_name
        _add_paper(path, substrate, keyed, ungrounded, set_name)
    rows = _rows(keyed, ungrounded)
    return {"n_graph_sets": len({s for _, s in graph_set_pairs}),
            "n_papers": len(seen),
            "n_holes_keyed": sum(r["occurrences"] for r in rows),
            "n_distinct_keys": len(rows),
            "recurring": [r for r in rows if len(r["papers"]) > 1],
            "ungrounded": ungrounded,
            "typed_holes": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", nargs="+",
                    default=["data/iatc-argument-graphs/loop-run-70b"])
    ap.add_argument("--out", default="holes/excursions/E-comprehension-foundation.typed-holes.json")
    args = ap.parse_args()

    substrate = r2d.load_substrate(r2d.parse_args([]))
    per_set = []
    for graphs in args.graphs:
        graph_dir = ROOT / graphs
        paths = [p for p in sorted(graph_dir.glob("*.edn"))
                 if not p.name.endswith(".rung2.edn")]
        per_set.append((paths, Path(graphs).name))
    if len(per_set) == 1:
        paths, name = per_set[0]
        result = inventory(paths, substrate, set_name=name)
    else:
        pairs = [(p, name) for paths, name in per_set for p in paths]
        result = corpus_inventory(pairs, substrate)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(out, "w"), indent=2, sort_keys=True)
    print(f"wrote {out}")
    print(f"papers: {result['n_papers']} across {result['n_graph_sets']} sets")
    print(f"holes keyed by (type, concept): {result['n_holes_keyed']} "
          f"into {result['n_distinct_keys']} distinct keys; "
          f"{len(result['recurring'])} recurring across papers; "
          f"{len(result['ungrounded'])} ungrounded (reported separately)")


if __name__ == "__main__":
    main()
