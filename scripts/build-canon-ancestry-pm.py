#!/usr/bin/env python3
r"""Build a canon-ancestry index from PlanetMath `\pmrelated` links.

Each PM .tex file has:
  \pmcanonicalname{ThisCanon}
  \pmrelated{RelatedCanonA}
  \pmrelated{RelatedCanonB}
  ...

The relation is symmetric in practice — if A links B, B is also
related to A — so we emit both directions when building the index.

Output JSON:
    {
      "source": "planetmath",
      "extractor_version": "v1",
      "entries": <count of canons indexed>,
      "by_canon": {
        "TopologicalGroup": ["ContinuousFunction", "Group", ...],
        "Group": ["TopologicalGroup", "Monoid", ...],
        ...
      }
    }

Used by eval-grounding-gold.py's --ancestry-index flag to widen
canon matching so "Group" can count as matching "TopologicalGroup"
when both refer to the same family of structures.

Usage:
    python scripts/build-canon-ancestry-pm.py \
        --pm-root /home/joe/code/planetmath \
        --out data/canon-ancestry-pm.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


_CANONICAL_RE = re.compile(r"\\pmcanonicalname\{([^}]+)\}")
_RELATED_RE = re.compile(r"\\pmrelated\{([^}]+)\}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pm-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    by_canon: dict[str, set[str]] = defaultdict(set)
    n_scanned = 0
    n_with_links = 0
    for tex_path in args.pm_root.rglob("*.tex"):
        n_scanned += 1
        try:
            text = tex_path.read_text(errors="replace")
        except Exception:
            continue
        canon_m = _CANONICAL_RE.search(text)
        if not canon_m:
            continue
        this_canon = canon_m.group(1).strip()
        related = [m.group(1).strip() for m in _RELATED_RE.finditer(text)]
        if not related:
            continue
        n_with_links += 1
        for r in related:
            # Symmetric edges so the eval doesn't need to check both
            # directions.
            by_canon[this_canon].add(r)
            by_canon[r].add(this_canon)

    out = {
        "source": "planetmath",
        "extractor_version": "v1",
        "scanned": n_scanned,
        "with_links": n_with_links,
        "entries": len(by_canon),
        "by_canon": {k: sorted(v) for k, v in by_canon.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    edges = sum(len(v) for v in by_canon.values())
    print(f"[ancestry] Scanned {n_scanned} PM .tex; {n_with_links} with "
          f"\\pmrelated links; {len(by_canon)} distinct canons; "
          f"{edges} directed edges. Wrote {args.out}")
    return out


if __name__ == "__main__":
    main()
