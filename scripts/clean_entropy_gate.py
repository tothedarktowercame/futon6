#!/usr/bin/env python3
"""G-entropy gate (linode-stepper-contract S7 postcondition).

Structure embeddings are only worth shipping if they carry DISCRIMINATIVE signal —
not collapsed onto one macro/cluster (the failure mode where every card stays green
while retrieval is useless). Reads the embedder's clean-embed.json and checks:
  - macro-distribution entropy (normalized) above a floor — the corpus isn't all
    one shape;
  - mean off-diagonal structure cosine below a ceiling — proofs are distinguishable.
PASS iff both. Exits nonzero on a collapse (red gate).

Usage:
  futon6/.venv/bin/python scripts/clean_entropy_gate.py \
      [--embed data/showcases/clean-demo/clean-embed.json] [--min-entropy 0.5] [--max-sim 0.85]
"""
import argparse
import json
import math
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", default="data/showcases/clean-demo/clean-embed.json")
    ap.add_argument("--min-entropy", type=float, default=0.5)
    ap.add_argument("--max-sim", type=float, default=0.85)
    args = ap.parse_args()

    d = json.load(open(args.embed))
    macros = d.get("macros", [])
    sim = d.get("structure_sim", [])
    n = len(macros)

    counts = Counter(macros)
    H = -sum((c / n) * math.log2(c / n) for c in counts.values()) if n else 0.0
    Hmax = math.log2(len(counts)) if len(counts) > 1 else 1.0
    Hnorm = H / Hmax if Hmax else 0.0

    off = [sim[i][j] for i in range(len(sim)) for j in range(len(sim)) if i != j]
    mean_sim = sum(off) / len(off) if off else 1.0

    ent_ok = Hnorm >= args.min_entropy
    sim_ok = mean_sim <= args.max_sim
    print(f"proofs={n}  distinct macros={len(counts)}  macro-entropy(norm)={Hnorm:.2f} "
          f"(floor {args.min_entropy})")
    print(f"mean off-diagonal structure cosine={mean_sim:.2f} (ceiling {args.max_sim})")
    print(f"macro dist: {dict(counts)}")
    if ent_ok and sim_ok:
        print("PASS — structure embeddings are discriminative")
        return 0
    reasons = []
    if not ent_ok:
        reasons.append("macro-entropy collapsed (corpus too monotone)")
    if not sim_ok:
        reasons.append("mean similarity too high (proofs not distinguishable)")
    print("FAIL — " + "; ".join(reasons))
    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
