#!/usr/bin/env python3
"""Build the recognizer registry + hunger field from the anatomy-v0 corpus.

The killer-idea artifact (M-distributed-proofreaders): "parse once, recognize
forever". A notation/macro defined or used across many papers is a corpus-wide
recognizer; aggregating the per-paper sweep output gives:

  - registry: cseq -> {role, defining-papers, occurrences, class}, the
    shared-notation core that classifies occurrences corpus-wide;
  - hunger field: the genuine-unknown and role-gap tallies, ranked — the
    Distributed-Proofreaders priority queue (what to fix/resolve next).

This is the per-MSC-replicable unit: run the sweep + this over any arXiv
subject class and you get that class's registry. Usage:
    build_recognizer_registry.py [--anatomy-dir DIR] [--min-papers K] [--out FILE]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
from pathlib import Path

DEFAULT_DIR = Path("/home/joe/code/storage/futon6/data/ct-anatomy-v0")
DEFAULT_OUT = Path("/home/joe/code/futon6/data/ct-recognizer-registry.json")


def build(anatomy_dir: Path, min_papers: int) -> dict:
    define_papers: dict[str, int] = collections.Counter()   # cseq -> #papers defining it
    role_votes: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    rhs_votes: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    unknown_papers: dict[str, int] = collections.Counter()  # genuine unknown -> #papers
    unknown_occ: dict[str, int] = collections.Counter()
    rolegap_papers: dict[str, int] = collections.Counter()
    rolegap_occ: dict[str, int] = collections.Counter()
    papers = 0

    for f in glob.glob(str(anatomy_dir / "*.json")):
        if os.path.basename(f).startswith("_"):
            continue
        try:
            d = json.loads(Path(f).read_text())
        except Exception:
            continue
        papers += 1
        for r in d.get("symbol-table", []):
            cs = r.get("cs")
            if not cs:
                continue
            define_papers[cs] += 1
            if r.get("role") and r.get("role") != "UNKNOWN":
                role_votes[cs][r["role"]] += 1
            if r.get("rhs"):
                rhs_votes[cs][r["rhs"].strip()[:40]] += 1
        tc = d.get("token-census", {})
        # genuine unknowns: distinct list per paper + occurrence count
        for cs in tc.get("unknown-list", []):
            unknown_papers[cs] += 1
        # role-gaps: recognised-but-untyped (the real frontier)
        for cs in tc.get("role-gap-list", []):
            rolegap_papers[cs] += 1

    def majority_role(cs: str) -> str:
        v = role_votes.get(cs)
        return v.most_common(1)[0][0] if v else "UNKNOWN"

    def top_rhs(cs: str) -> str:
        v = rhs_votes.get(cs)
        return v.most_common(1)[0][0] if v else ""

    registry = []
    for cs, n in define_papers.most_common():
        if n < min_papers:
            break
        registry.append({
            "cs": cs, "defining-papers": n,
            "role": majority_role(cs), "top-rhs": top_rhs(cs),
            "role-resolved": majority_role(cs) != "UNKNOWN",
        })

    return {
        "meta": {
            "papers": papers, "min-papers": min_papers,
            "distinct-author-macros": len(define_papers),
            "shared-notation-core": len(registry),
        },
        "registry": registry,
        "hunger": {
            "genuine-unknown": [
                {"cs": cs, "papers": n}
                for cs, n in unknown_papers.most_common(60)
            ],
            "role-gap": [
                {"cs": cs, "papers": n, "top-rhs": top_rhs(cs)}
                for cs, n in (
                    collections.Counter(
                        {cs: define_papers.get(cs, 0)
                         for cs in rolegap_papers}
                    ).most_common(60)
                )
            ],
            "genuine-unknown-distinct": len(unknown_papers),
            "role-gap-distinct": len(rolegap_papers),
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anatomy-dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--min-papers", type=int, default=10)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    reg = build(args.anatomy_dir, args.min_papers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(reg, indent=1))
    m = reg["meta"]
    print(f"registry: {m['shared-notation-core']} recognizers "
          f"(>= {m['min-papers']} papers) of {m['distinct-author-macros']} "
          f"distinct macros over {m['papers']} papers")
    print(f"hunger: {reg['hunger']['genuine-unknown-distinct']} genuine-unknown, "
          f"{reg['hunger']['role-gap-distinct']} role-gap distinct cseqs")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
