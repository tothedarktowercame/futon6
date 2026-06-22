#!/usr/bin/env python3
"""Inline G-coverage instrument (linode-stepper-contract S2).

The post-hoc coverage curve reads flat because every committed concept artifact is
df-filtered (concept-usage is df>=10 — zero rare/hapax tail). The rise lives in the
tail, which only exists in the RAW detector output during a live S1 run. This is
that instrument: it consumes RAW per-paper concepts (pre-HAPAX-drop) as the
substrate accumulates and reports coverage vs corpus-fraction, segmented by concept
rarity so the rise is visible. On Linode it is fed S1's raw concept stream at S2.

A concept is COVERED at corpus-fraction X if it recurs in >= min-usages of the
first X% of (sorted) training papers — the df>=2 "two usages" bar (Joe).

Modes:
  --concepts <json>   {paper: [raw concepts]} — real input (S1 stream / a dump)
  --self-test         synthetic Zipf corpus (seed 0) proving the mechanism

Usage:
  futon6/.venv/bin/python scripts/coverage_inline.py --self-test
  futon6/.venv/bin/python scripts/coverage_inline.py --concepts data/warp/concept-usage.json --field paper_concepts
"""
import argparse
import json
from collections import defaultdict
import numpy as np


def band(df):
    if df <= 1:
        return "hapax(1)"
    if df <= 3:
        return "rare(2-3)"
    if df <= 9:
        return "uncommon(4-9)"
    return "common(>=10)"


BANDS = ["hapax(1)", "rare(2-3)", "uncommon(4-9)", "common(>=10)"]


def coverage_by_band(paper_concepts, held, fractions, min_usages):
    concept_papers = defaultdict(set)
    for p, cs in paper_concepts.items():
        for c in cs:
            concept_papers[c].add(p)
    held_set = set(held)
    train = [p for p in sorted(paper_concepts) if p not in held_set]
    rows = []
    for X in fractions:
        sub = set(train[:int(X * len(train))])
        num = defaultdict(int)   # covered per band
        den = defaultdict(int)   # total per band
        for p in held:
            for c in paper_concepts[p]:
                b = band(len(concept_papers[c]))
                den[b] += 1
                if len(concept_papers[c] & sub) >= min_usages:
                    num[b] += 1
        cov = {b: (num[b] / den[b] if den[b] else None) for b in BANDS}
        allnum, allden = sum(num.values()), sum(den.values())
        cov["ALL"] = allnum / allden if allden else 0.0
        rows.append((X, cov))
    return rows


def synth_corpus(n_papers=600, vocab=2000, per_paper=30, seed=0):
    rng = np.random.default_rng(seed)
    w = np.array([1.0 / (i + 1) for i in range(vocab)])  # Zipf-ish popularity
    w /= w.sum()
    papers = {}
    for j in range(n_papers):
        idx = rng.choice(vocab, size=per_paper, replace=False, p=w)
        papers[f"p{j:04d}"] = {f"c{i}" for i in idx}
    return papers


def report(paper_concepts, held, fractions, min_usages, title):
    rows = coverage_by_band(paper_concepts, held, fractions, min_usages)
    cols = BANDS + ["ALL"]
    print(title)
    print(f"{'frac':>5s} " + " ".join(f"{c:>13s}" for c in cols))
    print("-" * (6 + 14 * len(cols)))
    for X, cov in rows:
        cells = []
        for c in cols:
            v = cov.get(c)
            cells.append("   —   " if v is None else f"{v:.2f}")
        print(f"{X:>5.2f} " + " ".join(f"{c:>13s}" for c in cells))
    a0, a1 = rows[0][1]["ALL"], rows[-1][1]["ALL"]
    # rise of the rarest non-empty band
    def band_rise(b):
        s, e = rows[0][1].get(b), rows[-1][1].get(b)
        return None if s is None or e is None else e - s
    rare = band_rise("hapax(1)") or band_rise("rare(2-3)")
    print(f"\nALL coverage {a0:.2f} -> {a1:.2f} (+{a1-a0:.2f}); "
          f"rarest-band rise {rare:+.2f}" if rare is not None else
          f"\nALL coverage {a0:.2f} -> {a1:.2f} (+{a1-a0:.2f}); no rare tail present")
    if rare is None or rare < 0.05:
        print("=> no rare tail in this input (df-filtered) — feed RAW S1 concepts "
              "inline at S2 to see the real rise")
    else:
        print("=> rise concentrated in the rare/hapax tail, saturating — the substrate "
              "helps exactly there (instrument validated)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", default=None)
    ap.add_argument("--field", default="paper_concepts",
                    help="if the json wraps the map under a key")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--held-out", type=int, default=40)
    ap.add_argument("--min-usages", type=int, default=2)
    ap.add_argument("--fractions", default="0.1,0.25,0.5,0.75,1.0")
    args = ap.parse_args()
    fracs = [float(x) for x in args.fractions.split(",")]

    if args.self_test:
        pc = synth_corpus()
        papers = sorted(pc)
        step = max(1, len(papers) // args.held_out)
        held = papers[::step][:args.held_out]
        report(pc, held, fracs, args.min_usages,
               f"SELF-TEST synthetic Zipf corpus (seed 0): {len(pc)} papers, "
               f"{len({c for cs in pc.values() for c in cs})} concepts\n")
        return

    raw = json.load(open(args.concepts))
    pc = raw.get(args.field, raw) if isinstance(raw, dict) else raw
    pc = {p: set(cs) for p, cs in pc.items()}
    papers = sorted(pc)
    step = max(1, len(papers) // args.held_out)
    held = papers[::step][:args.held_out]
    report(pc, held, fracs, args.min_usages,
           f"INLINE coverage on {args.concepts}: {len(pc)} papers\n")


if __name__ == "__main__":
    main()
