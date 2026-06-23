#!/usr/bin/env python3
"""Tier-2 of the 'improve as we run' spine: cluster the harvested IATC move-lexicon into
data-driven move-cues, feed them back to the strategy recognizer, and re-ground proof-moves
against the corpus's OWN vocabulary. Measures proof-move grounding base vs augmented.

The recognizer's gestures are sparse Lean-style cues; CT prose moves ("by functoriality",
"it suffices", "by Yoneda", "reduces to") aren't in it — which is why proof-move grounding
sits at ~0.14. The harvested move-phrases supply exactly those cues. Added as HEURISTIC
gestures (recognized-but-unverified = honest 'thin', 0.5 credit), not 'grounded'.

  futon6/.venv/bin/python scripts/iatc_move_reground.py
"""
import glob
import json
import os
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import strategy_recognizer as sr  # noqa: E402
from iatc_lexicon_harvest import harvest  # noqa: E402

# words that are entities/structure, not move-signals — kept out of the cue set
STOP = {"category", "categories", "object", "objects", "morphism", "morphisms", "functor",
        "diagram", "theorem", "lemma", "result", "rules", "structure", "groupoid",
        "model", "since", "where", "which", "their", "these", "above", "given",
        # entity/noun residue (not move-signals) — keep the cue set to inference moves
        "sigma", "group", "square", "squares", "alpha", "small", "domains", "trans",
        "calmod", "bicategories", "identity", "arrangements", "consisting", "compatible"}


def cluster_cues(lex, min_count=2, top=40):
    """Data-driven move-cues = recurring content words across harvested move-phrases
    (frequency ≥ min_count), excluding entity/structure words. Each cue is the corpus
    naming a recurring move (functoriality, suffices, construction, naturality, …)."""
    tok = Counter()
    for phrase, e in lex.items():
        if (sum(e["conf"]) / len(e["conf"]) if e["conf"] else 0) < 0.3:
            continue  # only confident anchorings contribute cues
        for w in re.findall(r"[a-z]{5,}", phrase):
            if w not in STOP:
                tok[w] += e["count"]
    return [w for w, c in tok.most_common(top) if c >= min_count]


def score(vocab, windows):
    bs = Counter()
    for w in windows:
        b, _ = sr.recognize_text(w, vocab)
        bs.update(b)
    g, t, u = bs["grounded"], bs["thin"], bs["ungrounded"]
    tot = g + t + u
    return {"grounded": g, "thin": t, "ungrounded": u,
            "proof-move-grounding": round((g + 0.5 * t) / tot, 3) if tot else 0.0}


def main():
    lex, _, _ = harvest(os.path.join(ROOT, "data/iatc-argument-graphs/loop-run-70b"))
    cues = cluster_cues(lex)
    print(f"data-driven move-cues harvested from the corpus ({len(cues)}):")
    print("  " + ", ".join(cues) + "\n")
    vocab = sr.load_vocab(os.path.join(ROOT, "holes/clean/tactic-gesture-vocab.edn"))
    aug = {**vocab, "heuristic": {**vocab["heuristic"], "corpus-move": cues}}
    windows = [json.load(open(f)).get("source-window", "")
               for f in glob.glob(os.path.join(ROOT, "data/cand-neighborhood/*.candidate.json"))]
    print(f"measuring proof-move grounding over {len(windows)} proof windows:\n")
    base = score(vocab, windows)
    augd = score(aug, windows)
    print(f"  BASE      : {base}")
    print(f"  AUGMENTED : {augd}")
    d = augd["proof-move-grounding"] - base["proof-move-grounding"]
    print(f"\n  proof-move grounding: {base['proof-move-grounding']} -> {augd['proof-move-grounding']}  "
          f"({'+' if d >= 0 else ''}{round(d, 3)})  — grounding the corpus against its OWN harvested moves")


if __name__ == "__main__":
    main()
