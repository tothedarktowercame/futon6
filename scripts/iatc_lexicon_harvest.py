#!/usr/bin/env python3
"""Harvest a grounded LEXICON OF INFERENCE TERMS from IATC argument graphs.

The IATC's strength is that it NAMES inference moves in prose and anchors each to a
:source span — "reduction-to-subgoals", "functoriality of _*", "Yoneda embedding
preserves and reflects isomorphisms", the relation grammar (because / therefore /
suffices-to-show / arises-from …). Those names ARE the inference lexicon; "linearizing"
the structure is the harvest, not a failure (per Joe).

Each harvested entry carries the IATC's OWN CONFIDENCE in that anchoring:
  confidence = anchor-faithfulness × formal-corroboration
    anchor-faithfulness = |claim head-terms ∩ anchored span| / |claim head-terms|
    formal-corroboration = 0.3 when the span carries formal structure (\\judge/\\justifies/
        prooftree/\\infer) that the :text DID NOT recover (extremal-low — the prooftree
        flatten case; flips to ~1.0 once a deterministic recognizer exists).

  futon6/.venv/bin/python scripts/iatc_lexicon_harvest.py --graphs data/iatc-argument-graphs/loop-run-70b
"""
import argparse
import glob
import os
import re
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

_FORMAL = re.compile(r"\\judge|\\justifies|prooftree|\\infer|\\inferrule|mathpartir")
_NAMES_FORMAL = re.compile(r"judge|justif|infer|proof.?tree", re.I)


def _text_lines(pid, cache):
    if pid not in cache:
        try:
            import dp_paper_view as dpv
            cache[pid] = dpv.build(pid)["text"].split("\n")
        except Exception:
            cache[pid] = []
    return cache[pid]


def _span(lines, a, b):
    return " ".join(" ".join(lines[a - 1:b]).split()) if lines else ""


def _confidence(claim, span):
    """anchor-faithfulness × formal-corroboration; returns (conf, faithfulness, flattened)."""
    head = re.findall(r"[A-Za-z]{4,}", claim)[:4]
    f = sum(w.lower() in span.lower() for w in head) / max(1, len(head))
    flattened = bool(_FORMAL.search(span)) and not _NAMES_FORMAL.search(claim)
    return round(f * (0.3 if flattened else 1.0), 3), round(f, 3), flattened


def _norm(phrase):
    return re.sub(r"\s+", " ", phrase.strip().lower())[:80]


def harvest(graph_dir):
    lex = defaultdict(lambda: {"count": 0, "conf": [], "kinds": Counter(), "exemplars": []})
    relations = Counter()
    flattened = []
    cache = {}
    for f in sorted(glob.glob(os.path.join(graph_dir, "*.edn"))):
        if "rung2" in f:
            continue
        pid = os.path.basename(f).split("__")[0][:-4] if os.path.basename(f).endswith(".edn") and "__" not in os.path.basename(f) else os.path.basename(f).split("__")[0]
        lines = _text_lines(pid, cache)
        t = open(f).read()
        # node move-names (the linearized inference descriptions)
        for m in re.finditer(r':kind :([a-z-]+),?\s*:text "([^"]{4,80})"[^}]*?:source \{:lines \[(\d+) (\d+)\]', t):
            kind, text, a, b = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
            conf, faith, flat = _confidence(text, _span(lines, a, b))
            key = _norm(text)
            e = lex[key]; e["count"] += 1; e["conf"].append(conf); e["kinds"][kind] += 1
            if len(e["exemplars"]) < 3:
                e["exemplars"].append({"pid": pid, "lines": [a, b], "conf": conf})
            if flat:
                flattened.append({"pid": pid, "lines": [a, b], "text": text})
        # warrant phrases (the justification vocabulary)
        for w in re.findall(r':warrant \{[^}]*?:text "([^"]{4,80})"', t):
            key = _norm(w); e = lex[key]; e["count"] += 1; e["kinds"]["warrant"] += 1
            e["conf"].append(0.5)  # warrant text w/o its own span check — neutral prior
        # relation grammar
        for r in re.findall(r":relation :([a-z-]+)", t):
            relations[r] += 1
    return lex, relations, flattened


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", default="data/iatc-argument-graphs/loop-run-70b")
    ap.add_argument("--run-dir", help="emit lexicon-accretion MetricRecord here")
    ap.add_argument("--run-id", default="adhoc")
    ap.add_argument("--corpus-id", default="adhoc")
    a = ap.parse_args()
    lex, relations, flattened = harvest(os.path.join(ROOT, a.graphs) if not os.path.isabs(a.graphs) else a.graphs)

    def mc(e):
        return sum(e["conf"]) / len(e["conf"]) if e["conf"] else 0.0
    entries = sorted(lex.items(), key=lambda kv: (-kv[1]["count"], -mc(kv[1])))
    print(f"=== inference lexicon: {len(lex)} distinct entries ===\n")
    print("relation grammar:", dict(relations.most_common()))
    print(f"\ntop entries (phrase · count · mean-confidence):")
    for k, e in entries[:18]:
        print(f"  conf={mc(e):.2f} ×{e['count']}  {k}")
    hi = [k for k, e in lex.items() if mc(e) >= 0.7]
    lo = [k for k, e in lex.items() if mc(e) < 0.3]
    print(f"\nconfidence split: {len(hi)} high (≥0.7) · {len(lo)} low (<0.3)")
    print(f"FORMAL-STRUCTURE-FLATTENED anchorings (extremal-low now → high w/ a recognizer): {len(flattened)}")
    for fl in flattened[:4]:
        print(f"  {fl['pid']} L{fl['lines'][0]}-{fl['lines'][1]}: {fl['text'][:55]!r}")
    if a.run_dir:
        try:
            import metric_harness as mh
            mh.emit_record(a.run_dir, run_id=a.run_id, corpus_id=a.corpus_id, paper_id="(corpus)",
                           stage="S3", metric="inference-lexicon-size", axis="accretion",
                           value=len(lex), computable=True)
            mh.emit_record(a.run_dir, run_id=a.run_id, corpus_id=a.corpus_id, paper_id="(corpus)",
                           stage="S3", metric="inference-anchor-confidence", axis="quality",
                           value=round(sum(mc(e) for _, e in entries) / max(1, len(entries)), 4), computable=True)
        except Exception as ee:
            print(f"  (metric emit skipped: {ee})")


if __name__ == "__main__":
    main()
