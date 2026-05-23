#!/usr/bin/env python3
r"""Build labeled (symbol, canon) gold from Wikipedia math articles.

Wikipedia uses `<math>X</math>` for math content and `[[Target]]` or
`[[Target|Display]]` for wiki-links. When a math expression appears
in a declarative sentence whose type-phrase carries one of those
links, the target is gold ground for the symbol → concept mapping.

This mirrors `build-grounding-gold.py` (PlanetMath) but adapted to
MediaWiki markup conventions. Source corpus: the Zenodo math dump
at `~/Downloads/math.tar` (record 15107679, multi-lingual; this
extractor reads one decompressed XML at a time).

Output JSON shape matches the PM extractor:
    {
      "source": "wikipedia.<lang>",
      "extractor_version": "v1",
      "entries": [
        {"id": "...", "path": "...", "raw_text": "...",
         "gold": [{"symbol": "X", "canon": "...", ...}, ...]},
        ...
      ]
    }

Usage:
    bunzip2 -k ~/Downloads/math/enwiki.xml.bz   # if needed
    python scripts/build-grounding-gold-wikipedia.py \
        --xml /tmp/math-sample/math/enwiki.xml \
        --lang en \
        --out data/grounding-gold-wiki-en.json \
        --max-entries 0          # 0 = no cap
"""

from __future__ import annotations

import argparse
import html
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path


# `[[Target|Display]]` -> ('Target', 'Display'); `[[Target]]` -> ('Target', 'Target').
# We strip section anchors (#) and any leading colon or namespace.
_WIKILINK_RE = re.compile(r"\[\[([^\[\]|#]+)(?:#[^\[\]|]*)?(?:\|([^\[\]]+))?\]\]")

# `<math>X</math>` — case-insensitive, non-greedy. The text element XML-
# unescaped already, so this matches plain `<math>` tags.
_MATH_RE = re.compile(r"<math(?:\s[^>]*)?>([^<]{1,200})</math>", re.IGNORECASE)

# Strict declaration: <math>X</math> <verb> [article] [adjectives] [[Target|Display]]
# The math content must be a simple atom (letter, \macro, or \macro{X}).
_STRICT_PAT = re.compile(
    r"<math(?:\s[^>]*)?>([^<]{1,30})</math>\s+"
    r"(?:be|denotes?|stands?\s+for|is|represents?|denote\s+the)\s+"
    r"(?:(?:an|a|the)\s+)?"
    r"(?:[a-z][\w\s\-]{0,40}\s+)?"
    r"\[\[([^\[\]|#]{2,60})(?:#[^\[\]|]*)?(?:\|[^\[\]]*)?\]\]",
    re.IGNORECASE,
)

# Let/Fix/Suppose form
_LET_FIX_PAT = re.compile(
    r"(?:Let|Fix|Suppose|Assume|Take|Consider)\s+"
    r"<math(?:\s[^>]*)?>([^<]{1,30})</math>\s+"
    r"(?:be|denote|denotes?|=|to\s+be)\s+"
    r"(?:(?:an|a|the)\s+)?"
    r"(?:[a-z][\w\s\-]{0,40}\s+)?"
    r"\[\[([^\[\]|#]{2,60})(?:#[^\[\]|]*)?(?:\|[^\[\]]*)?\]\]",
    re.IGNORECASE,
)

# Single-symbol shape (mirror of classify_lhs in symbol_grounding).
# We only emit gold when the symbol is a single atom; multi-symbol or
# relation-chain LHSes would be quoted-constructor bindings on the engine
# side, so comparing them to a canon is meaningless.
_SINGLE_SHAPE = re.compile(
    r"^\s*(?:"
    r"[A-Za-z]"
    r"|\\[A-Za-z]+"
    r"|\\[A-Za-z]+\{[A-Za-z0-9]+\}"
    r")\s*$"
)


def _strip_wiki_markup(text: str) -> str:
    """Convert wiki markup to engine-friendly raw text.

    - `<math>X</math>` -> `$X$` (engine expects dollar-delimited math)
    - `[[Target|Display]]` -> `Display` (drop the link, keep display text)
    - `[[Target]]` -> `Target`
    - Strip section anchors inside links
    """
    s = _MATH_RE.sub(lambda m: f"${m.group(1)}$", text)

    def _link_repl(m):
        display = m.group(2) if m.group(2) else m.group(1)
        return display
    s = _WIKILINK_RE.sub(_link_repl, s)
    return s


def _normalize_canon(target: str) -> str:
    """Wikipedia link target → canon string.

    Strip whitespace, normalize spaces to single, drop section anchors,
    apply CamelCase if the target is multi-word (so "abelian group"
    becomes "AbelianGroup", aligning with PM-style canons).
    """
    s = target.strip()
    s = re.sub(r"\s+", " ", s)
    s = s.split("#", 1)[0]
    parts = s.split(" ")
    return "".join(p[:1].upper() + p[1:] for p in parts if p)


def extract_gold_from_page(title: str, text: str) -> tuple[str, list[dict]] | None:
    """Return (raw_text, gold_pairs) for one Wikipedia page, or None on miss."""
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


def iter_pages(xml_path: Path):
    """Stream `(title, text)` from a MediaWiki XML dump.

    Uses iterparse so memory stays bounded regardless of file size.
    """
    # Detect namespace from root once.
    ns = None
    title = None
    text = None
    context = ET.iterparse(str(xml_path), events=("start", "end"))
    for event, elem in context:
        tag = elem.tag.split("}", 1)[-1] if "}" in elem.tag else elem.tag
        if ns is None and "}" in elem.tag:
            ns = elem.tag.split("}", 1)[0].strip("{")
        if event == "end":
            if tag == "title":
                title = elem.text or ""
            elif tag == "text":
                text = elem.text or ""
            elif tag == "page":
                if title and text:
                    yield title, text
                title = None
                text = None
                elem.clear()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, required=True,
                        help="Decompressed MediaWiki XML dump")
    parser.add_argument("--lang", default="en",
                        help="Language tag for the `source` field")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-entries", type=int, default=0,
                        help="Cap pages with ≥1 gold pair (0 = no cap)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    entries = []
    n_scanned = 0
    n_with_gold = 0
    total_gold = 0
    for title, raw_text in iter_pages(args.xml):
        n_scanned += 1
        result = extract_gold_from_page(title, raw_text)
        if result is None:
            continue
        cleaned, gold = result
        n_with_gold += 1
        total_gold += len(gold)
        entries.append({
            "id": title,
            "path": str(args.xml),
            "raw_text": cleaned,
            "gold": gold,
        })
        if args.max_entries and n_with_gold >= args.max_entries:
            break
        if n_scanned % 5000 == 0:
            print(f"[wiki-gold] ...{n_scanned} pages scanned, "
                  f"{n_with_gold} have gold, {total_gold} pairs")

    out = {
        "source": f"wikipedia.{args.lang}",
        "extractor_version": "v1",
        "scanned": n_scanned,
        "with_gold": n_with_gold,
        "total_gold_pairs": total_gold,
        "entries": entries,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"[wiki-gold] Scanned {n_scanned} Wikipedia pages; "
          f"{n_with_gold} carried ≥1 gold pair; "
          f"{total_gold} pairs total. Wrote {args.out}")
    return out


if __name__ == "__main__":
    main()
