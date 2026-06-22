#!/usr/bin/env python3
"""G-coverage gate (linode-stepper-contract S2): does concept coverage RISE with
corpus fraction, then saturate? Tests Joe's "improves as we run" — and exposes
the ceiling — before any Linode spend.

Method (no substrate rebuild needed): the concept-index records, per concept, the
set of papers it occurs in. Hold out a deterministic sample of papers; for each
corpus fraction X, a held-out paper's concept counts as COVERED if it recurs in
>= 2 of the first X% of the (sorted) training papers — the df>=2 "two usages" bar
(Joe). coverage(X) = mean over held-out papers of covered/used concepts.

FINDING (2026-06-22): run post-hoc on the existing artifacts (concept-index /
concept-usage) this reads ~1.0 flat — those are already HAPAX-filtered (the
term-prior df=1 drop), so the rise happened upstream and is invisible here. The
honest G-coverage gate therefore CANNOT be post-hoc; it must run INLINE at S2 on
RAW per-paper concepts (pre-drop) as the substrate grows. This script is the
diagnostic that established that (and stays useful once raw extraction is wired).

Usage:
  futon6/.venv/bin/python scripts/coverage_curve.py \
      [--index data/warp/concept-index.json] [--held-out 40] [--min-usages 2]
"""
import argparse
import json
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--usage", default="data/warp/concept-usage.json",
                    help="raw per-paper concepts (incl. hapax tail) — the honest input")
    ap.add_argument("--held-out", type=int, default=40)
    ap.add_argument("--min-usages", type=int, default=2)
    ap.add_argument("--fractions", default="0.1,0.25,0.5,0.75,1.0")
    args = ap.parse_args()

    usage = json.load(open(args.usage))["paper_concepts"]
    # RAW per-paper concepts (includes the hapax tail) — non-circular: held-out
    # concepts are a paper's own raw extraction, not the curated index.
    paper_concepts = {p: set(cs) for p, cs in usage.items()}
    concept_papers = defaultdict(set)
    all_papers = set(paper_concepts)
    for p, cs in paper_concepts.items():
        for c in cs:
            concept_papers[c].add(p)

    papers = sorted(all_papers)
    N = len(papers)
    # deterministic spread of held-out papers across the corpus
    step = max(1, N // args.held_out)
    held = papers[::step][:args.held_out]
    held_set = set(held)
    train = [p for p in papers if p not in held_set]
    fracs = [float(x) for x in args.fractions.split(",")]

    print(f"concept-usage (raw): {len(concept_papers)} concepts, {N} papers")
    print(f"held-out: {len(held)} papers (deterministic spread); train: {len(train)}")
    print(f"covered = concept recurs in >= {args.min_usages} of the sub-corpus "
          f"(the df>={args.min_usages} 'two usages' bar)\n")
    print(f"{'corpus frac':>11s} {'#train':>7s} {'coverage':>9s} {'delta':>7s}")
    print("-" * 40)
    prev = None
    rows = []
    for X in fracs:
        sub = set(train[:int(X * len(train))])
        per_paper = []
        for p in held:
            cps = paper_concepts.get(p, set())
            if not cps:
                continue
            cov = sum(1 for c in cps if len(concept_papers[c] & sub) >= args.min_usages)
            per_paper.append(cov / len(cps))
        coverage = sum(per_paper) / len(per_paper) if per_paper else 0.0
        delta = "" if prev is None else f"{coverage - prev:+.3f}"
        print(f"{X:>11.2f} {len(sub):>7d} {coverage:>9.3f} {delta:>7s}")
        rows.append((X, coverage))
        prev = coverage

    rise = rows[-1][1] - rows[0][1]
    last_delta = rows[-1][1] - rows[-2][1]
    if rows[0][1] > 0.9 and rise < 0.02:
        # honest finding (2026-06-22): the available artifacts are already
        # post-filtered to RECURRING concepts (the term-prior HAPAX df=1 drop),
        # so a post-hoc curve reads ~1.0 flat — the rise happened UPSTREAM and is
        # not reconstructable from these outputs. This is NOT "substrate useless".
        print(f"\ncoverage flat-high ({rows[0][1]:.3f} at 10%): the input is "
              f"already HAPAX-filtered, so the rise is invisible post-hoc.")
        print("=> G-coverage cannot be a post-hoc gate. It must run INLINE at S2 on "
              "RAW per-paper concepts (pre-HAPAX-drop), measuring coverage as the "
              "substrate grows. Wire it into the detector/substrate stage, not here.")
    else:
        verdict = ("RISES + saturating" if last_delta < rise * 0.4
                   else "RISES, not yet saturating")
        print(f"\nrise {rows[0][1]:.3f} -> {rows[-1][1]:.3f} (+{rise:.3f}); "
              f"last-step delta {last_delta:+.3f}  =>  {verdict}")


if __name__ == "__main__":
    main()
