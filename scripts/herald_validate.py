#!/usr/bin/env python3
"""Validate the strategy recognizer against Herald (the self-application study).

Herald_proofs pairs an informal NL proof with its formal Lean proof. Ground truth
for the recognizer: run our gesture recognizer on `informal_proof` -> predicted
Lean-tactic-classes; extract the ACTUAL tactics from `formal_proof`; measure how
well we recover them. This is the Herald-style round-trip check
(E-strategy-recognizer §validation) — and the gaps tell us what to RECONCILE in
the seed vocab (low-recall tactics, missing classes, noisy gestures).

Usage:
  futon6/.venv/bin/python scripts/herald_validate.py [--n 3000] \
      [--parquet data/lean-nl/herald_proofs.parquet]
"""
import argparse
import re
import sys
from collections import Counter
import pyarrow.parquet as pq

sys.path.insert(0, "scripts")
from strategy_recognizer import load_vocab, recognize_text  # noqa: E402

# Lean/Mathlib tactic token -> our vocab tactic-class (synonyms folded)
LEAN_TO_CLASS = {
    "intro": "intro", "intros": "intro", "rintro": "intro",
    "apply": "apply", "exact": "exact", "exact?": "exact", "refine": "refine",
    "rw": "rw", "rewrite": "rw", "erw": "rw", "subst": "rw",
    "simp": "simp", "simp_all": "simp", "dsimp": "simp", "simpa": "simp",
    "induction": "induction", "cases": "cases", "rcases": "cases", "obtain": "obtain",
    "by_contra": "by_contra", "by_contradiction": "by_contra",
    "suffices": "suffices", "wlog": "wlog",
    "use": "use", "exists": "use", "existsi": "use",
    "constructor": "constructor", "refine'": "constructor",
    "unfold": "unfold", "delta": "unfold", "show": "unfold", "change": "unfold",
    "calc": "calc", "ring": "ring", "ring_nf": "ring",
    "linarith": "linarith", "nlinarith": "linarith", "omega": "linarith",
    "ext": "ext", "funext": "ext", "ext1": "ext",
    "contrapose": "contrapose", "contrapose!": "contrapose",
    "norm_num": "norm_num", "gcongr": "gcongr", "mono": "gcongr",
    "aesop_cat": "aesop_cat",
}
TOKEN_RE = re.compile(r"^[·\-\s]*([a-zA-Z_][a-zA-Z0-9_'?!]*)")


def actual_tactic_classes(formal_proof):
    """Set of our tactic-classes for the tactics that appear in the Lean proof."""
    classes = set()
    body = formal_proof.split(":= by", 1)[-1] if ":= by" in formal_proof else formal_proof
    # split on newlines AND tactic separators
    for chunk in re.split(r"[\n;]|<;>", body):
        m = TOKEN_RE.match(chunk.strip())
        if not m:
            continue
        cls = LEAN_TO_CLASS.get(m.group(1))
        if cls:
            classes.add(cls)
    return classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/lean-nl/herald_proofs.parquet")
    ap.add_argument("--vocab", default="holes/clean/tactic-gesture-vocab.edn")
    ap.add_argument("--n", type=int, default=3000)
    args = ap.parse_args()

    vocab = load_vocab(args.vocab)
    t = pq.read_table(args.parquet, columns=["informal_proof", "formal_proof"])
    n = min(args.n, t.num_rows)
    informal = t.column("informal_proof").to_pylist()[:n]
    formal = t.column("formal_proof").to_pylist()[:n]

    vocab_keys = set(vocab["tactics"].keys())   # discursive recognition targets
    TP = FP = FN = 0
    per_class = Counter()           # actual occurrences per class
    per_class_hit = Counter()       # recalled (predicted & actual)
    pred_total = Counter()          # our predictions per class
    pred_wrong = Counter()          # predicted but not actual
    hidden_moves = 0                # bookkeeping tactic occurrences (not targets)
    disc_moves = 0                  # discursive tactic occurrences
    any_actual = 0

    for inf, fm in zip(informal, formal):
        actual_all = actual_tactic_classes(fm or "")
        hidden_moves += len(actual_all - vocab_keys)
        disc_moves += len(actual_all & vocab_keys)
        actual = actual_all & vocab_keys      # score on the DISCURSIVE layer only
        _, tac = recognize_text(inf or "", vocab)
        pred = set(tac.keys())
        if actual:
            any_actual += 1
        TP += len(pred & actual)
        FP += len(pred - actual)
        FN += len(actual - pred)
        for c in actual:
            per_class[c] += 1
            if c in pred:
                per_class_hit[c] += 1
        for c in pred:
            pred_total[c] += 1
            if c not in actual:
                pred_wrong[c] += 1

    prec = TP / (TP + FP) if TP + FP else 0
    rec = TP / (TP + FN) if TP + FN else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    print(f"Herald self-application: {n} proofs ({any_actual} with extractable tactics)")
    tot_moves = disc_moves + hidden_moves
    print(f"tactic occurrences: discursive (targets) {disc_moves} | "
          f"hidden-layer (bookkeeping) {hidden_moves} = {hidden_moves/tot_moves:.0%} silent\n")
    print(f"DISCURSIVE-LAYER  precision={prec:.3f}  recall={rec:.3f}  F1={f1:.3f}")
    print(f"(TP={TP} FP={FP} FN={FN})\n")
    print(f"{'tactic-class':18s} {'actual':>7s} {'recall':>7s} {'pred':>6s} {'FP-rate':>8s}")
    print("-" * 52)
    for c, a in per_class.most_common():
        r = per_class_hit[c] / a if a else 0
        pt = pred_total.get(c, 0)
        fpr = pred_wrong.get(c, 0) / pt if pt else 0
        print(f"{c:18s} {a:>7d} {r:>7.2f} {pt:>6d} {fpr:>8.2f}")
    missing = [c for c in per_class if per_class_hit[c] == 0]
    print(f"\nRECONCILE — high-frequency, low-recall (gestures need work): "
          f"{[c for c,a in per_class.most_common() if per_class_hit[c]/a < 0.15 and a >= 20]}")
    print(f"never-recalled actual classes: {missing}")


if __name__ == "__main__":
    main()
