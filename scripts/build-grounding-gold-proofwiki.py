#!/usr/bin/env python3
r"""Build labeled (symbol, canon) gold from the NaturalProofs ProofWiki dump.

NaturalProofs (Welleck et al., NeurIPS 2021) is a Zenodo-hosted
preprocessed dump of ProofWiki (CC-BY): ~19K theorems + ~12K
definitions + ~1K others, JSON-shaped with `dataset.definitions`,
`dataset.theorems`, `dataset.others` arrays. Each entry has
`contents` (list of plain-text paragraphs with wiki + math markup
preserved).

ProofWiki uses `$X$` for math (like PlanetMath) and typed wiki-links
`[[Definition:Set|set]]` / `[[Theorem:Cauchy-Schwarz|...]]` —
*declarative* shape that maps perfectly onto our gold extraction
pattern. Strip the link namespace prefix ("Definition:", "Theorem:",
"Axiom:") to produce a clean canon.

Output JSON matches the PM extractor shape:
    {"source": "proofwiki", "entries": [{"id", "raw_text", "gold": ...}, ...]}

Source:
    https://zenodo.org/records/4902289/files/naturalproofs_proofwiki.json
    DOI: 10.5281/zenodo.4902289

Usage:
    python scripts/build-grounding-gold-proofwiki.py \
        --input /tmp/naturalproofs_proofwiki.json \
        --out data/grounding-gold-proofwiki.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


# ProofWiki typed wiki-link: [[Namespace:Concept]] or [[Namespace:Concept|display]]
# Namespaces we recognise (in priority order — Definition is the
# concept-bearing one; Theorem/Axiom are pointers to specific results).
_NAMESPACED_LINK_RE = re.compile(
    r"\[\["
    r"(Definition|Theorem|Axiom|Symbol|Notation|Lemma|Corollary)"
    r":([^\[\]|#]+)"
    r"(?:#[^\[\]|]*)?"
    r"(?:\|([^\[\]]+))?"
    r"\]\]"
)
# Plain `[[Target]]` wiki-link without namespace (treated as a regular
# concept reference in the link target).
_PLAIN_LINK_RE = re.compile(r"\[\[([^\[\]|:#]+)(?:#[^\[\]|]*)?(?:\|([^\[\]]+))?\]\]")

# Declarative shape: $X$ <verb> [article] [adj-phrase]
# [[Definition:Target|display]] or [[Target|display]]
_STRICT_PAT = re.compile(
    r"\$([^$\n]{1,30})\$\s+"
    r"(?:be|denotes?|stands?\s+for|is|represents?|denote\s+the)\s+"
    r"(?:(?:an|a|the)\s+)?"
    r"(?:[a-z][\w\s\-]{0,40}\s+)?"
    r"\[\["
    r"(?:Definition|Theorem|Axiom|Symbol|Notation|Lemma|Corollary):"
    r"([^\[\]|#]+)"
    r"(?:#[^\[\]|]*)?"
    r"(?:\|[^\[\]]*)?"
    r"\]\]",
    re.IGNORECASE,
)

_LET_FIX_PAT = re.compile(
    r"(?:Let|Fix|Suppose|Assume|Take|Consider)\s+"
    r"\$([^$\n]{1,30})\$\s+"
    r"(?:be|denote|denotes?|=|to\s+be)\s+"
    r"(?:(?:an|a|the)\s+)?"
    r"(?:[a-z][\w\s\-]{0,40}\s+)?"
    r"\[\["
    r"(?:Definition|Theorem|Axiom|Symbol|Notation|Lemma|Corollary):"
    r"([^\[\]|#]+)"
    r"(?:#[^\[\]|]*)?"
    r"(?:\|[^\[\]]*)?"
    r"\]\]",
    re.IGNORECASE,
)

# Single-symbol shape — same filter as PM/Wiki extractors so
# multi-symbol LHSes get rejected (would be constructor declarations
# in the engine, not single bindings).
_SINGLE_SHAPE = re.compile(
    r"^\s*(?:"
    r"[A-Za-z]"
    r"|\\[A-Za-z]+"
    r"|\\[A-Za-z]+\{[A-Za-z0-9]+\}"
    r")\s*$"
)


def _normalize_canon(target: str) -> str:
    """ProofWiki link target → canon string.

    Strip whitespace, normalize spaces, drop section anchors,
    CamelCase multi-word so "Cauchy-Schwarz Inequality" becomes
    "Cauchy-SchwarzInequality" aligning with PM-style canons.
    """
    s = target.strip()
    s = s.split("#", 1)[0]
    s = re.sub(r"\s+", " ", s)
    parts = s.split(" ")
    return "".join(p[:1].upper() + p[1:] for p in parts if p)


def _strip_wiki_markup(text: str) -> str:
    """Convert wiki markup to engine-friendly raw text:
    - `[[Namespace:Target|Display]]` → `Display`
    - `[[Namespace:Target]]` → `Target` (no display means target is the display)
    - `[[Target|Display]]` → `Display`
    - `[[Target]]` → `Target`
    Math is already in `$X$` form so the engine sees it unchanged.
    """
    def _ns_repl(m):
        return m.group(3) if m.group(3) else m.group(2)
    s = _NAMESPACED_LINK_RE.sub(_ns_repl, text)

    def _plain_repl(m):
        return m.group(2) if m.group(2) else m.group(1)
    s = _PLAIN_LINK_RE.sub(_plain_repl, s)
    return s


def extract_gold_from_entry(entry: dict) -> tuple[str, list[dict]] | None:
    """Pull (raw_text, gold_pairs) from one NaturalProofs JSON entry."""
    contents = entry.get("contents") or []
    if not contents:
        return None
    body = "\n".join(contents)

    gold = []
    seen: set[tuple[str, str]] = set()
    for pat_name, pat in (("strict", _STRICT_PAT), ("let-fix", _LET_FIX_PAT)):
        for m in pat.finditer(body):
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
    raw_text = _strip_wiki_markup(body)
    return raw_text, gold


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="naturalproofs_proofwiki.json from Zenodo")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-entries", type=int, default=0,
                        help="Cap entries with ≥1 gold pair (0 = no cap)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    data = json.loads(args.input.read_text(encoding="utf-8"))
    ds = data.get("dataset", {})
    # Walk all three buckets: definitions, theorems, others
    bucket_keys = ("definitions", "theorems", "others")
    n_scanned = 0
    n_with_gold = 0
    total_gold = 0
    entries = []
    for bucket in bucket_keys:
        for entry in ds.get(bucket, []):
            n_scanned += 1
            result = extract_gold_from_entry(entry)
            if result is None:
                continue
            raw_text, gold = result
            n_with_gold += 1
            total_gold += len(gold)
            entries.append({
                "id": f"{bucket}:{entry.get('label') or entry.get('title') or entry.get('id')}",
                "path": str(args.input),
                "raw_text": raw_text,
                "gold": gold,
            })
            if args.max_entries and n_with_gold >= args.max_entries:
                break
        if args.max_entries and n_with_gold >= args.max_entries:
            break

    out = {
        "source": "proofwiki",
        "extractor_version": "v1",
        "scanned": n_scanned,
        "with_gold": n_with_gold,
        "total_gold_pairs": total_gold,
        "entries": entries,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"[pw-gold] Scanned {n_scanned} ProofWiki entries; "
          f"{n_with_gold} carried ≥1 gold pair; "
          f"{total_gold} pairs total. Wrote {args.out}")
    return out


if __name__ == "__main__":
    main()
