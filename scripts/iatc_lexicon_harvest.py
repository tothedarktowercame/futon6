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
import json
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


def _pid_of(fname):
    """Paper id from a graph filename `<pid>__p<N>.edn`.

    The pid ITSELF contains `__` for old-style arXiv ids (`math__0608040` is the
    safe-form of `math/0608040`), so splitting on `__` and taking the first field
    collapses every pre-2007 paper to the bare archive name `math` — the harvest
    then looks up an eprint called "math", finds none, and aborts the whole run
    (E-superpod-hardening H14, 2026-08-06). Strip the proof suffix instead; that
    is the convention the rest of the pipeline already uses.
    """
    base = os.path.basename(fname)
    if base.endswith(".edn"):
        base = base[:-4]
    return base.rsplit("__p", 1)[0]


def harvest(graph_dir):
    """Layer-agnostic: an IATC inference-move, an expository scope, and (in principle) an
    SFC typed-slot are the same shape — a CLASSIFIED scope (kind) + a :source anchor +
    a grounding confidence. Harvests whichever the file carries."""
    lex = defaultdict(lambda: {"count": 0, "conf": [], "kinds": Counter(), "layer": Counter(), "exemplars": []})
    grammar = Counter()       # relation grammar (IATC) + move grammar (expository kinds)
    flattened = []
    cache = {}
    for f in sorted(glob.glob(os.path.join(graph_dir, "*.edn"))):
        if "rung2" in f:
            continue
        pid = _pid_of(f)
        lines = _text_lines(pid, cache)
        t = open(f).read()
        expository = ":scopes" in t or ":slot-fill" in t

        def add(key, conf, kind, layer, a=None, b=None):
            e = lex[_norm(key)]; e["count"] += 1; e["conf"].append(conf)
            e["kinds"][kind] += 1; e["layer"][layer] += 1
            if a and len(e["exemplars"]) < 3:
                e["exemplars"].append({"pid": pid, "lines": [a, b], "conf": conf})

        if expository:  # EXPOSITORY: scope kind (move grammar) + slot-fill text, span-anchored
            for block in re.split(r'(?=:kind :)', t):     # one scope per :kind block
                km = re.search(r':kind (:[a-z/-]+)', block)
                if not km:
                    continue
                kind = km.group(1)
                grammar[kind] += 1
                lm = re.search(r':lines \[(\d+) (\d+)\]', block)
                fills = re.findall(r'"([^"]{4,})"', block.split(":slot-fill", 1)[1]) if ":slot-fill" in block else []
                fill = " ".join(fills)[:90]
                if fill and lm:
                    a, b = int(lm.group(1)), int(lm.group(2))
                    conf, _, _ = _confidence(fill, _span(lines, a, b))
                    add(fill, conf, kind, "expository", a, b)
        else:           # IATC: node move-names + warrant phrases + relation grammar
            for m in re.finditer(r':kind :([a-z-]+),?\s*:text "([^"]{4,80})"[^}]*?:source \{:lines \[(\d+) (\d+)\]', t):
                kind, text, a, b = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
                conf, faith, flat = _confidence(text, _span(lines, a, b))
                add(text, conf, kind, "iatc", a, b)
                if flat:
                    flattened.append({"pid": pid, "lines": [a, b], "text": text})
            for w in re.findall(r':warrant \{[^}]*?:text "([^"]{4,80})"', t):
                add(w, 0.5, "warrant", "iatc")
            for r in re.findall(r":relation :([a-z-]+)", t):
                grammar[r] += 1
    return lex, grammar, flattened


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", default="data/iatc-argument-graphs/loop-run-70b")
    ap.add_argument("--run-dir", help="emit lexicon-accretion MetricRecord here")
    ap.add_argument("--run-id", default="adhoc")
    ap.add_argument("--corpus-id", default="adhoc")
    a = ap.parse_args()
    lex, grammar, flattened = harvest(os.path.join(ROOT, a.graphs) if not os.path.isabs(a.graphs) else a.graphs)

    def mc(e):
        return sum(e["conf"]) / len(e["conf"]) if e["conf"] else 0.0
    entries = sorted(lex.items(), key=lambda kv: (-kv[1]["count"], -mc(kv[1])))
    layers = Counter()
    for _, e in entries:
        layers.update(e["layer"])
    print(f"=== move lexicon: {len(lex)} distinct entries  (layers: {dict(layers)}) ===\n")
    print("grammar (IATC relations / expository move-kinds):", dict(grammar.most_common()))
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
        # PERSIST the lexicon (E-superpod-hardening H15, 2026-08-06). The playbook's
        # RETRIEVE manifest promises "the harvested lexicons" under data/runs/<id>
        # and names move-lexicon convergence as learning goal #2 — but every S10
        # script was stdout-only, so on a cluster the corpus's move vocabulary died
        # with the terminal at teardown. Exactly the mark6 lost-CLeans shape.
        try:
            out_dir = a.run_dir if os.path.isabs(a.run_dir) else os.path.join(ROOT, a.run_dir)
            os.makedirs(out_dir, exist_ok=True)
            payload = {
                "run_id": a.run_id, "corpus_id": a.corpus_id,
                "distinct_entries": len(lex),
                "layers": dict(layers),
                "grammar": dict(grammar.most_common()),
                "confidence": {"high_ge_0.7": len(hi), "low_lt_0.3": len(lo),
                               "mean": round(sum(mc(e) for _, e in entries) / max(1, len(entries)), 4)},
                "entries": [{"phrase": k, "count": e["count"],
                             "mean_conf": round(mc(e), 4),
                             "layers": sorted(set(e["layer"])),
                             "exemplars": e.get("exemplars", [])[:3]}
                            for k, e in entries],
                "flattened": flattened,
            }
            lex_path = os.path.join(out_dir, "inference-lexicon.json")
            with open(lex_path, "w") as fh:
                json.dump(payload, fh, indent=1)
            print(f"\nwrote {lex_path}  ({len(lex)} entries)")
        except Exception as ee:
            print(f"  (lexicon persist skipped: {ee})")
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
