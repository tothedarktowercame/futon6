#!/usr/bin/env python3
r"""Build the MSC topic prior from PlanetMath's MSC-tagged corpus.

For each entry in `~/code/planetmath/*.edn` we extract `:id` (the
canon name, e.g. "RingedSpace") and `:msc-codes` (list of MSC codes,
e.g. ["18F20"]). We bucket each (canon, msc_primary) where msc_primary
is the first two chars of the code (e.g. "18" from "18F20").

The output `topic-prior-msc.json` is consumed by
`scripts/eval-arxiv-domain-coherence.py` (and later, the superpod
Stage 5 inference pipeline) via `futon6.topic_prior.MSCTopicPrior`.

Usage:
    python scripts/build-msc-topic-prior.py \\
        --pm-root /home/joe/code/planetmath \\
        --out data/topic-prior-msc.json
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from futon6.topic_prior import MSCTopicPrior


_ENTRY_RE = re.compile(r"#:entry\{")
_ID_RE = re.compile(r":id\s+\"([^\"]+)\"")
_CODE_RE = re.compile(r":code\s+\"([^\"]+)\"")


def parse_edn_entries(text: str):
    """Yield (canon, [msc_codes]) tuples from an EDN file's text.

    Uses regex rather than a real EDN parser because the structure
    is shallow (one level of nested maps) and the relevant fields
    are quoted strings. If the EDN format ever gets fancier this
    needs revisiting."""
    chunks = _ENTRY_RE.split(text)
    for chunk in chunks[1:]:
        m_id = _ID_RE.search(chunk)
        if not m_id:
            continue
        canon = m_id.group(1)
        codes = _CODE_RE.findall(chunk)
        # Dedup codes within an entry (the EDN often lists them twice
        # under :msc-codes + per-code nested maps)
        codes = list(dict.fromkeys(codes))
        yield canon, codes


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pm-root", type=Path,
                        default=Path("/home/joe/code/planetmath"))
    parser.add_argument("--out", type=Path,
                        default=Path("data/topic-prior-msc.json"))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    edn_files = sorted(p for p in args.pm_root.glob("*.edn"))
    print(f"[msc-prior] found {len(edn_files)} EDN files in {args.pm_root}")

    prior = MSCTopicPrior()
    primary_dist: Counter[str] = Counter()
    n_entries = 0
    n_with_codes = 0
    for ef in edn_files:
        text = ef.read_text(encoding="utf-8", errors="replace")
        for canon, codes in parse_edn_entries(text):
            n_entries += 1
            if not codes:
                continue
            n_with_codes += 1
            # Dedup primaries within this entry — multiple secondary
            # codes under the same primary shouldn't double-count.
            primaries = list(dict.fromkeys(c[:2] for c in codes if len(c) >= 2))
            for p in primaries:
                prior.add(canon, p, n=1)
                primary_dist[p] += 1

    print(f"[msc-prior] {n_entries} entries scanned, "
          f"{n_with_codes} with MSC codes")
    print(f"[msc-prior] {len(prior.counts)} unique canons; "
          f"grand_total={prior.grand_total}")
    print("[msc-prior] top 10 MSC primaries by mass:")
    for p, n in primary_dist.most_common(10):
        print(f"  MSC {p}: {n} entry-bindings")

    # Sanity check: smoke-test the prior on a few known canons
    print()
    print("[msc-prior] sanity check — prior for math.CT (MSC 18):")
    for canon in ["Functor", "Category", "AbelianGroup",
                  "StableMarriageProblem", "RiemannHypothesis",
                  "RingedSpace", "Limit"]:
        v_ct = prior.prior(canon, ["18"])
        v_91 = prior.prior(canon, ["91"])  # operations research
        print(f"  {canon:24s} P(18)={v_ct:.3f}  P(91)={v_91:.3f}")

    prior.save(args.out)
    print(f"[msc-prior] wrote {args.out}")


if __name__ == "__main__":
    main()
