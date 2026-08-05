#!/usr/bin/env python3
"""S2 substrate-corpus match check (E-superpod-hardening H1, tier 1).

The mark5 lesson was SILENT staleness: S6 grounding against a prior-corpus
concept-index with nothing flagging it. This makes the match explicit: verify
the WARP substrate files exist and report what fraction of the run's ids the
substrate's paper_concepts actually covers, failing loudly below threshold.

This does NOT rebuild the spine (that is tier 2 — warp_run.py portability);
it turns "corpus-fresh" from an unenforced intention into a measured gate.

Usage (stepper S2):
  python scripts/warp_substrate_check.py --ids holes/math-ct-full.ids.txt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SUBSTRATE = [
    "data/warp/concept-index.json",
    "data/warp/def-snippets.json",
    "data/warp/defined-index.json",
    "data/warp/concept-usage.json",
    "data/concept-encyclopedia-ct.json",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help="run id manifest (one paper id per line)")
    ap.add_argument("--concepts", default="data/warp/concept-usage.json")
    ap.add_argument("--field", default="paper_concepts")
    ap.add_argument("--require", type=float, default=0.95,
                    help="minimum fraction of run ids the substrate must cover")
    args = ap.parse_args()

    missing = [rel for rel in SUBSTRATE if not (ROOT / rel).exists()]
    for rel in SUBSTRATE:
        f = ROOT / rel
        if f.exists():
            st = f.stat()
            print(f"  substrate: {rel}  {st.st_size/1e6:8.1f} MB  "
                  f"mtime {time.strftime('%Y-%m-%d %H:%M', time.localtime(st.st_mtime))}")
    if missing:
        print(f"✗ substrate INCOMPLETE — missing: {missing}")
        print("  (STAGE step ships these; see _STAGE_MANIFEST in linode_stepper.py)")
        return 1

    ids = [l.strip() for l in open(args.ids) if l.strip()]
    raw = json.load(open(ROOT / args.concepts))
    pc = raw.get(args.field, raw) if isinstance(raw, dict) else raw
    covered = [p for p in ids if p in pc]
    frac = len(covered) / len(ids) if ids else 0.0
    print(f"  corpus match: {len(covered)}/{len(ids)} run ids in "
          f"{args.concepts}:{args.field} ({frac:.1%}; substrate holds {len(pc)} papers)")
    if frac < args.require:
        print(f"✗ substrate-corpus match {frac:.1%} < required {args.require:.0%} — "
              f"the substrate was mined from a different corpus. Rebuild the WARP "
              f"spine for THIS corpus before trusting S5/S6 grounding "
              f"(warp_run.py — note its dev-box path assumptions, H1 tier 2).")
        return 1
    print(f"✓ substrate matches run corpus at {frac:.1%} (threshold {args.require:.0%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
