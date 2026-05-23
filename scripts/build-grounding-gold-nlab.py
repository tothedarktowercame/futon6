#!/usr/bin/env python3
r"""Build labeled (symbol, canon) gold from nLab content pages.

nLab is the smallest-N of our gold corpora but the most domain-
focused: every page is a math concept written by mathematicians for
mathematicians. The markup combines:
  - PM-style math:  $X$, $\mathcal{C}$, etc.
  - Wikipedia-style links: [[concept]] or [[concept|display]]

So this extractor mostly cribs from build-grounding-gold.py (the PM
extractor) but uses the Wikipedia-shaped link regex from
build-grounding-gold-wikipedia.py.

Source: /home/joe/code/nlab-content/pages/**/content.md — sourced
from the nLab content git mirror, ~41K pages.

In the M-canon-fingerprint-store.md framing, nLab is the HELD-OUT
corpus: it never goes into the canon store or strategy reliability
init. We use it only to measure whether the system generalises
cleanly to a corpus it has never been calibrated against — the
"Rob's batch is temporally selected" test, extended to a never-seen
domain.

Output JSON shape matches the other gold extractors so
eval-grounding-arbitration.py can consume it via --gold.

Usage:
    python scripts/build-grounding-gold-nlab.py \
        --pages-dir /home/joe/code/nlab-content/pages \
        --out data/grounding-gold-nlab.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


_STRICT_PAT = re.compile(
    r"\$([^$\n]{1,30})\$\s+"
    r"(?:be|denotes?|stands?\s+for|is|represents?|denote\s+the)\s+"
    r"(?:(?:an|a|the)\s+)?"
    r"(?:[a-z][\w\s\-]{0,40}\s+)?"
    r"\[\[([^\[\]|#]{2,60})(?:#[^\[\]|]*)?(?:\|[^\[\]]*)?\]\]",
    re.IGNORECASE,
)

_LET_FIX_PAT = re.compile(
    r"(?:Let|Fix|Suppose|Assume|Take|Consider)\s+"
    r"\$([^$\n]{1,30})\$\s+"
    r"(?:be|denote|denotes?|=|to\s+be)\s+"
    r"(?:(?:an|a|the)\s+)?"
    r"(?:[a-z][\w\s\-]{0,40}\s+)?"
    r"\[\[([^\[\]|#]{2,60})(?:#[^\[\]|]*)?(?:\|[^\[\]]*)?\]\]",
    re.IGNORECASE,
)

_SINGLE_SHAPE = re.compile(
    r"^\s*(?:"
    r"[A-Za-z]"
    r"|\\[A-Za-z]+"
    r"|\\[A-Za-z]+\{[A-Za-z0-9]+\}"
    r")\s*$"
)

_PLAIN_LINK_RE = re.compile(r"\[\[([^\[\]|#]+)(?:#[^\[\]|]*)?(?:\|([^\[\]]+))?\]\]")


def _normalize_canon(target: str) -> str:
    """nLab link target → canon string.

    nLab pages mostly use lowercase or two-word titles like
    "abelian group" or "topological group". CamelCase them so they
    align with the other gold extractors' canon shape.
    """
    s = target.strip()
    s = s.split("#", 1)[0]
    s = re.sub(r"\s+", " ", s)
    parts = s.split(" ")
    return "".join(p[:1].upper() + p[1:] for p in parts if p)


def _strip_wiki_markup(text: str) -> str:
    """Strip `[[link|display]]` markup, leaving display text. Math
    in `$X$` is left untouched (engine consumes those directly)."""
    def _repl(m):
        return m.group(2) if m.group(2) else m.group(1)
    return _PLAIN_LINK_RE.sub(_repl, text)


def extract_gold_from_page(text: str) -> tuple[str, list[dict]] | None:
    gold = []
    seen: set[tuple[str, str]] = set()
    for pat_name, pat in (("strict", _STRICT_PAT), ("let-fix", _LET_FIX_PAT)):
        for m in pat.finditer(text):
            symbol = m.group(1).strip()
            target = m.group(2).strip()
            if not _SINGLE_SHAPE.match(symbol):
                continue
            canon = _normalize_canon(target)
            if not canon:
                continue
            key = (symbol, canon)
            if key in seen:
                continue
            seen.add(key)
            gold.append({
                "symbol": symbol,
                "canon": canon,
                "evidence_span": [m.start(), m.end()],
                "pattern": pat_name,
            })
    if not gold:
        return None
    raw_text = _strip_wiki_markup(text)
    return raw_text, gold


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages-dir", type=Path, required=True,
                        help="Root of nlab-content/pages")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-entries", type=int, default=0,
                        help="Cap entries with ≥1 gold pair (0 = no cap)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    entries = []
    n_scanned = 0
    n_with_gold = 0
    total_gold = 0
    for content_path in sorted(args.pages_dir.rglob("content.md")):
        n_scanned += 1
        name_path = content_path.parent / "name"
        page_name = (
            name_path.read_text(errors="replace").strip()
            if name_path.exists()
            else content_path.parent.name
        )
        try:
            text = content_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        result = extract_gold_from_page(text)
        if result is None:
            continue
        raw_text, gold = result
        n_with_gold += 1
        total_gold += len(gold)
        entries.append({
            "id": f"nlab:{page_name}",
            "path": str(content_path),
            "raw_text": raw_text,
            "gold": gold,
        })
        if args.max_entries and n_with_gold >= args.max_entries:
            break
        if n_scanned % 5000 == 0:
            print(f"[nlab-gold] ...{n_scanned} pages scanned, "
                  f"{n_with_gold} have gold, {total_gold} pairs")

    out = {
        "source": "nlab",
        "extractor_version": "v1",
        "scanned": n_scanned,
        "with_gold": n_with_gold,
        "total_gold_pairs": total_gold,
        "entries": entries,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"[nlab-gold] Scanned {n_scanned} nLab pages; "
          f"{n_with_gold} carried ≥1 gold pair; "
          f"{total_gold} pairs total. Wrote {args.out}")
    return out


if __name__ == "__main__":
    main()
