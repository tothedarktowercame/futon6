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
sys.path.insert(0, str(ROOT / "scripts"))
import futon6_config as config  # noqa: E402

# The subject's concept substrate, wherever it is configured: the gate must check
# the vocabulary the run will actually read, not this checkout's default one.
SUBSTRATE = [
    config.warp() / "concept-index.json",
    config.warp() / "def-snippets.json",
    config.warp() / "defined-index.json",
    config.warp() / "concept-usage.json",
    config.concept_encyclopedia(),
]


def shown(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def substrate_paper_ids(raw: object, field: str) -> tuple[set[str], str]:
    """Return witnessed scanned identities, retaining legacy index support."""
    if not isinstance(raw, dict):
        raise ValueError("concept usage must be a JSON object")
    scanned = raw.get("papers_scanned_ids")
    if scanned is not None:
        if (not isinstance(scanned, list)
                or any(not isinstance(pid, str) or not pid for pid in scanned)
                or len(set(scanned)) != len(scanned)):
            raise ValueError("papers_scanned_ids must contain unique nonempty strings")
        declared = raw.get("papers_scanned")
        if isinstance(declared, bool) or not isinstance(declared, int):
            raise ValueError("papers_scanned must be an integer")
        if declared != len(scanned):
            raise ValueError("papers_scanned does not match papers_scanned_ids")
        return set(scanned), "papers_scanned_ids"
    pc = raw.get(field, raw)
    if not isinstance(pc, dict):
        raise ValueError(f"{field} must be a JSON object")
    return set(pc), field


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help="run id manifest (one paper id per line)")
    ap.add_argument("--concepts", type=Path, default=config.warp() / "concept-usage.json")
    ap.add_argument("--field", default="paper_concepts")
    ap.add_argument("--require", type=float, default=0.95,
                    help="minimum fraction of run ids the substrate must cover")
    args = ap.parse_args()
    concepts = args.concepts if args.concepts.is_absolute() else ROOT / args.concepts

    missing = [shown(f) for f in SUBSTRATE if not f.exists()]
    for f in SUBSTRATE:
        if f.exists():
            st = f.stat()
            print(f"  substrate: {shown(f)}  {st.st_size/1e6:8.1f} MB  "
                  f"mtime {time.strftime('%Y-%m-%d %H:%M', time.localtime(st.st_mtime))}")
    if missing:
        print(f"✗ substrate INCOMPLETE — missing: {missing}")
        print("  (STAGE step ships these; see _STAGE_MANIFEST in linode_stepper.py)")
        return 1

    ids = [l.strip() for l in open(args.ids) if l.strip()]
    raw = json.load(open(concepts))
    try:
        substrate_ids, identity_field = substrate_paper_ids(raw, args.field)
    except ValueError as exc:
        print(f"✗ invalid concept usage identity evidence: {exc}")
        return 1
    covered = [p for p in ids if p in substrate_ids]
    frac = len(covered) / len(ids) if ids else 0.0
    print(f"  corpus match: {len(covered)}/{len(ids)} run ids in "
          f"{shown(concepts)}:{identity_field} ({frac:.1%}; "
          f"substrate holds {len(substrate_ids)} papers)")
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
