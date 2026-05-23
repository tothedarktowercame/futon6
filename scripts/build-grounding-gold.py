#!/usr/bin/env python3
r"""Build labeled (symbol, canon) gold from PlanetMath source.

PlanetMath authors use `\PMlinkname{display}{LinkTarget}` to link prose
phrases to canonical concept pages. When a `$X$` symbol appears in a
declarative sentence whose type-phrase carries one of those links, the
target is gold ground for the symbol → concept mapping. This is the
labeled data set Joe's design proposed: take a source corpus that
ALREADY has markup, strip the markup to produce raw text, and remember
the original links as ground truth.

Output JSON shape:
    {
      "source": "planetmath",
      "extractor_version": "v1",
      "entries": [
        {
          "id": "<filename-without-suffix>",
          "path": "<absolute path>",
          "raw_text": "<markup-stripped body text>",
          "gold": [
            {"symbol": "G", "canon": "TopologicalGroup",
             "evidence_span": [start, end], "pattern": "strict|let-fix"}
          ]
        },
        ...
      ]
    }

Usage:
    python scripts/build-grounding-gold.py \
        --pm-root /home/joe/code/planetmath \
        --out data/grounding-gold-pm.json \
        --max-entries 0       # 0 = no cap
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


# Declarative shapes that anchor a symbol to a linked concept.
#
# STRICT_PAT: `$X$ <verb> [article] [optional adjective string] \PMlinkname{display}{TARGET}`
#   verb in {be, denotes, denote, is, stands for, denote the}
#
# LET_FIX_PAT: `Let|Fix|Suppose|Assume|Take|Consider $X$ <verb> ... \PMlinkname{...}{TARGET}`
#
# Both patterns reject targets containing digits (heuristic: pure-concept link
# targets are alpha; targets like "FooBar3" are usually variant pages that
# would confuse downstream comparisons).

_STRICT_PAT = re.compile(
    r"\$([^$\n]{1,30})\$\s+"
    r"(?:be|denotes?|stands?\s+for|is|denote\s+the)\s+"
    r"(?:(?:an|a|the)\s+)?"
    r"(?:[a-z][\w\s\-]{0,30}\s+)?"
    r"\\PMlinkname\{([^}]+)\}\{([A-Za-z][A-Za-z]+)\}",
    re.IGNORECASE,
)

_LET_FIX_PAT = re.compile(
    r"(?:Let|Fix|Suppose|Assume|Take|Consider)\s+"
    r"\$([^$\n]{1,30})\$\s+"
    r"(?:be|denote|denotes?|=|to\s+be)\s+"
    r"(?:(?:an|a|the)\s+)?"
    r"(?:[a-z][\w\s\-]{0,40}\s+)?"
    r"\\PMlinkname\{([^}]+)\}\{([A-Za-z][A-Za-z]+)\}",
    re.IGNORECASE,
)

# Markup stripping: replace \PMlinkname{display}{target} with `display`
# so the engine sees natural prose without the link-target leaking in.
# Same for \PMlinkescaptext{display}{target} and bare \PMlink{target}.
_STRIP_PMLINKNAME = re.compile(r"\\PMlinkname\{([^}]*)\}\{[^}]*\}")
_STRIP_PMLINKESC = re.compile(r"\\PMlinkescaptext\{([^}]*)\}\{[^}]*\}")
_STRIP_PMLINK = re.compile(r"\\PMlink\{[^}]*\}\{([^}]*)\}")

# PM tex files are wrapped in \documentclass{article}...\endmetadata...
# \begin{document}...\end{document}. We slice between begin/end to drop
# the preamble (author macros aren't part of the eval target).
_BODY_RE = re.compile(
    r"\\begin\{document\}([\s\S]*?)\\end\{document\}"
)


def strip_markup(body: str) -> str:
    s = _STRIP_PMLINKNAME.sub(r"\1", body)
    s = _STRIP_PMLINKESC.sub(r"\1", s)
    s = _STRIP_PMLINK.sub(r"\1", s)
    return s


def extract_gold_from_tex(path: Path) -> tuple[str, list[dict]] | None:
    """Return (raw_text, gold_pairs) for a PM .tex file, or None on miss."""
    try:
        text = path.read_text(errors="replace")
    except Exception:
        return None
    body_match = _BODY_RE.search(text)
    if not body_match:
        return None
    body = body_match.group(1)

    gold = []
    seen: set[tuple[str, str]] = set()
    for pat_name, pat in (("strict", _STRICT_PAT), ("let-fix", _LET_FIX_PAT)):
        for m in pat.finditer(body):
            symbol = m.group(1).strip()
            target = m.group(3 if pat_name == "strict" else 3).strip()
            key = (symbol, target)
            if key in seen:
                continue
            seen.add(key)
            gold.append({
                "symbol": symbol,
                "canon": target,
                "evidence_span": [m.start(), m.end()],
                "pattern": pat_name,
            })
    if not gold:
        return None

    raw_text = strip_markup(body)
    return raw_text, gold


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pm-root", type=Path, required=True,
        help="Root directory containing PM .tex files (recursively scanned).",
    )
    parser.add_argument(
        "--out", type=Path, required=True,
        help="Output JSON path",
    )
    parser.add_argument(
        "--max-entries", type=int, default=0,
        help="Cap entries with at least one gold pair (0 = no cap).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)

    entries = []
    n_scanned = 0
    n_with_gold = 0
    total_gold = 0
    for tex_path in sorted(args.pm_root.rglob("*.tex")):
        n_scanned += 1
        result = extract_gold_from_tex(tex_path)
        if result is None:
            continue
        raw_text, gold = result
        n_with_gold += 1
        total_gold += len(gold)
        entries.append({
            "id": tex_path.stem,
            "path": str(tex_path),
            "raw_text": raw_text,
            "gold": gold,
        })
        if args.max_entries and n_with_gold >= args.max_entries:
            break

    out = {
        "source": "planetmath",
        "extractor_version": "v1",
        "scanned": n_scanned,
        "with_gold": n_with_gold,
        "total_gold_pairs": total_gold,
        "entries": entries,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[gold] Scanned {n_scanned} PM .tex files; "
        f"{n_with_gold} carried ≥1 gold pair; "
        f"{total_gold} pairs total. Wrote {args.out}"
    )
    return out


if __name__ == "__main__":
    main()
