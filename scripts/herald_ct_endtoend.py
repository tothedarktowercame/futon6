#!/usr/bin/env python3
"""mathlib-CT replication / end-to-end strategy profile on the CT domain.

Herald is generated FROM Mathlib4, so its CategoryTheory subset IS mathlib-CT with
the NL side attached — the replication slice, no LLM informalization needed. For
each CT proof we build the TWO-LAYER strategy profile:
  - discursive layer : recognizer on `informal_proof`  (strategies stated in prose)
  - hidden layer     : bookkeeping tactics from `formal_proof` (recovered from the
                       formal trace, not guessed — Joe's "rw is a hidden layer")
This is the end-to-end object Phase 2 reasons over, on the actual CT domain.

Usage:
  futon6/.venv/bin/python scripts/herald_ct_endtoend.py [--n 0 (all)] [--examples 3]
"""
import argparse
import sys
from collections import Counter
import pyarrow.parquet as pq

sys.path.insert(0, "scripts")
from strategy_recognizer import load_vocab, recognize_text  # noqa: E402
from herald_validate import actual_tactic_classes  # noqa: E402


def is_ct(header, name):
    h = (header or "") + " " + (name or "")
    return "CategoryTheory" in h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/lean-nl/herald_proofs.parquet")
    ap.add_argument("--vocab", default="holes/clean/tactic-gesture-vocab.edn")
    ap.add_argument("--n", type=int, default=0, help="0 = all CT proofs")
    ap.add_argument("--examples", type=int, default=3)
    args = ap.parse_args()

    vocab = load_vocab(args.vocab)
    vocab_keys = set(vocab["tactics"].keys())
    t = pq.read_table(args.parquet, columns=["name", "informal_proof", "formal_proof", "header"])
    cols = {c: t.column(c).to_pylist() for c in t.column_names}

    ct = [i for i in range(t.num_rows) if is_ct(cols["header"][i], cols["name"][i])]
    if args.n:
        ct = ct[:args.n]

    disc_dist = Counter()      # discursive tactics present (from formal) in CT
    hidden_dist = Counter()    # hidden-layer tactics present in CT
    rec_hit = Counter()        # recognizer recalled (discursive, NL∩formal)
    TP = FN = 0
    examples = []
    for i in ct:
        formal_all = actual_tactic_classes(cols["formal_proof"][i] or "")
        formal_disc = formal_all & vocab_keys
        formal_hidden = formal_all - vocab_keys
        _, tac = recognize_text(cols["informal_proof"][i] or "", vocab)
        nl_disc = set(tac.keys())
        for c in formal_disc:
            disc_dist[c] += 1
            if c in nl_disc:
                rec_hit[c] += 1
        for c in formal_hidden:
            hidden_dist[c] += 1
        TP += len(nl_disc & formal_disc)
        FN += len(formal_disc - nl_disc)
        if len(examples) < args.examples and (nl_disc & formal_disc) and formal_hidden:
            examples.append((cols["name"][i], sorted(nl_disc & formal_disc),
                             sorted(formal_hidden)))

    recall = TP / (TP + FN) if TP + FN else 0
    print(f"mathlib-CT replication: {len(ct)} CategoryTheory proofs (of {t.num_rows} Herald)\n")
    print(f"recognizer recall on CT discursive strategies: {recall:.2f} "
          f"(TP={TP} FN={FN})\n")
    print("CT DISCURSIVE strategy distribution (from formal; recall in parens):")
    for c, n in disc_dist.most_common():
        print(f"  {c:14s} {n:>5d}   recall {rec_hit[c]/n:.2f}")
    print("\nCT HIDDEN-LAYER (bookkeeping) distribution — recovered from formal trace:")
    for c, n in hidden_dist.most_common():
        print(f"  {c:14s} {n:>5d}")
    print("\nend-to-end two-layer profiles (NL-recognized discursive | formal-recovered hidden):")
    for name, disc, hidden in examples:
        print(f"  • {name}")
        print(f"      discursive (from prose): {disc or '—'}")
        print(f"      hidden (from Lean trace): {hidden}")


if __name__ == "__main__":
    main()
