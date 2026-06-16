#!/usr/bin/env python3
"""mark3 H11 — cross-document dependency/claim graph builder.

The loom thesis (E-superpod-mark3.md): a `[N]` citation edge and a
`:warrant`/`:missing-warrant` edge are the SAME dependency at different scales —
a hole in paper A's argument graph can be discharged by a claim EXPORTED by a
paper A cites. This builder assembles, from the per-paper artifacts:

  nodes  = papers + their exported claims (the :claim nodes of each IATC graph)
  edges  = (a) citation edges  (paper A --cites--> corpus-id B), from H7
           (b) discharge candidates: a :missing-warrant hole in A whose :wanted
               lexically matches an exported claim of a paper A cites.

Inputs:
  - H7 cite-resolution: data/warp/cite-resolution/*.cite-resolution.json
      {paper-id, records:[{cite/marker, resolved-corpus-id, title, confidence}]}
  - H2 IATC graphs: data/iatc-argument-graphs/gh200/*.edn
      :paper/id, :nodes [{:id :kind :text}], :holes [{:wanted ... :edge}]

The discharge matcher is a deliberately conservative LEXICAL heuristic (kebab
:wanted ∩ claim-text tokens); the embedding refinement (H8) is future work — this
is the buildable-now, sample-validatable skeleton Rob runs at scale.

Usage:
  python scripts/mark3_xdoc_graph.py [--graphs DIR] [--cites DIR] [--out FILE]
  python scripts/mark3_xdoc_graph.py --self-test
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GRAPHS = REPO / "data" / "iatc-argument-graphs" / "gh200"
CITES = REPO / "data" / "warp" / "cite-resolution"
STOP = {"the", "a", "of", "is", "to", "and", "in", "for", "by", "with", "an",
        "from", "as", "on", "that", "this", "are", "be", "or"}
MIN_OVERLAP = 2  # >= this many shared content tokens => candidate discharge


def toks(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP and len(w) > 2}


def load_iatc(graphs_dir: Path) -> dict:
    """paper-id -> {claims: [{id,text}], holes: [{wanted, edge, tokens}]}."""
    out = {}
    for f in sorted(graphs_dir.glob("*.edn")):
        t = f.read_text()
        m = re.search(r':paper/id\s+"([^"]+)"', t)
        if not m:
            continue
        pid = m.group(1)
        nodes_seg = t[t.find(":nodes"): t.find(":edges")] if ":edges" in t else t
        # node = {:id :x :kind :claim/:object :text "..."}
        claims = []
        for blk in re.split(r"\{:id ", nodes_seg)[1:]:
            nid = re.match(r"(:[\w./-]+)", blk)
            kind = re.search(r":kind\s+(:[\w./-]+)", blk)
            text = re.search(r':text\s+"([^"]*)"', blk)
            if nid and kind and kind.group(1) in (":claim", ":object") and text:
                claims.append({"id": nid.group(1), "text": text.group(1),
                               "tokens": toks(text.group(1))})
        holes_seg = t[t.rfind(":holes"):]
        holes = []
        for hb in re.split(r"\{:kind", holes_seg)[1:]:
            w = re.search(r":wanted\s+(:[\w./-]+)", hb)
            e = re.search(r":edge\s+(:[\w./-]+)", hb)
            if w:
                holes.append({"wanted": w.group(1), "edge": e.group(1) if e else None,
                              "tokens": toks(w.group(1).replace("-", " ").replace(":", " "))})
        out[pid] = {"claims": claims, "holes": holes}
    return out


def load_cites(cites_dir: Path) -> dict:
    """paper-id -> [{marker, corpus_id, title, confidence}] (resolved only)."""
    out = {}
    for f in sorted(cites_dir.glob("*.cite-resolution.json")):
        d = json.loads(f.read_text())
        pid = d.get("paper-id")
        if not pid:
            continue
        out[pid] = [
            {"marker": r.get("cite/marker"), "corpus_id": r["resolved-corpus-id"],
             "title": r.get("title"), "confidence": r.get("confidence", 0.0)}
            for r in d.get("records", []) if r.get("resolved-corpus-id")
        ]
    return out


def build(graphs_dir: Path, cites_dir: Path) -> dict:
    iatc = load_iatc(graphs_dir)
    cites = load_cites(cites_dir)

    papers = sorted(set(iatc) | set(cites))
    citation_edges = []
    for a, recs in cites.items():
        for r in recs:
            citation_edges.append({"from": a, "to": r["corpus_id"],
                                   "marker": r["marker"], "via": "citation",
                                   "confidence": r["confidence"]})

    # (b) cross-doc warrant-discharge candidates: a hole in A matches an exported
    #     claim of a paper A cites (and that cited paper has an IATC graph).
    discharges = []
    for a, g in iatc.items():
        cited_ids = {r["corpus_id"] for r in cites.get(a, [])}
        targets = cited_ids & set(iatc)            # cited papers we also have graphs for
        for h in g["holes"]:
            for b in targets:
                for c in iatc[b]["claims"]:
                    overlap = h["tokens"] & c["tokens"]
                    if len(overlap) >= MIN_OVERLAP:
                        discharges.append({
                            "hole-paper": a, "wanted": h["wanted"], "hole-edge": h["edge"],
                            "discharged-by-paper": b, "claim-id": c["id"],
                            "claim-text": c["text"], "shared-tokens": sorted(overlap),
                            "via": "cross-doc-warrant-discharge"})

    exported = [{"paper": p, "claim-id": c["id"], "text": c["text"]}
                for p, g in iatc.items() for c in g["claims"]]
    return {
        "schema": "futon6/h11-xdoc-graph/v1",
        "papers": papers,
        "exported-claims": exported,
        "citation-edges": citation_edges,
        "discharge-candidates": discharges,
        "stats": {
            "papers": len(papers),
            "papers-with-iatc": len(iatc),
            "papers-with-cites": len(cites),
            "citation-edges": len(citation_edges),
            "exported-claims": len(exported),
            "holes-total": sum(len(g["holes"]) for g in iatc.values()),
            "discharge-candidates": len(discharges),
        },
    }


def self_test() -> int:
    """Synthetic: a hole in A whose :wanted matches a claim B that A cites."""
    import tempfile
    d = Path(tempfile.mkdtemp())
    g, c = d / "g", d / "c"
    g.mkdir(); c.mkdir()
    (g / "A.edn").write_text(
        '{:paper/id "A" :nodes [{:id :n1 :kind :claim :text "trivial"}] '
        ':holes [{:kind :missing-warrant :edge :e1 :wanted :exactness-of-the-snake-sequence}]}')
    (g / "B.edn").write_text(
        '{:paper/id "B" :nodes [{:id :m1 :kind :claim :text "the snake sequence is exact"}] :holes []}')
    (c / "A.cite-resolution.json").write_text(json.dumps(
        {"paper-id": "A", "records": [{"cite/marker": "[1]", "resolved-corpus-id": "B",
                                       "title": "Snake", "confidence": 0.9}]}))
    out = build(g, c)
    ok = (out["stats"]["citation-edges"] == 1 and out["stats"]["discharge-candidates"] == 1
          and out["discharge-candidates"][0]["discharged-by-paper"] == "B")
    print("SELF-TEST", "PASS" if ok else "FAIL", "->", out["stats"])
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", default=str(GRAPHS))
    ap.add_argument("--cites", default=str(CITES))
    ap.add_argument("--out", default=str(REPO / "data" / "iatc-xdoc-graph.json"))
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    out = build(Path(a.graphs), Path(a.cites))
    Path(a.out).write_text(json.dumps(out, indent=2))
    s = out["stats"]
    print(json.dumps(s, indent=2))
    print(f"\nwrote {a.out}")
    if out["discharge-candidates"]:
        ex = out["discharge-candidates"][0]
        print(f"example discharge: {ex['hole-paper']} hole {ex['wanted']} "
              f"~ {ex['discharged-by-paper']} claim \"{ex['claim-text'][:60]}\" "
              f"(shared {ex['shared-tokens']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
