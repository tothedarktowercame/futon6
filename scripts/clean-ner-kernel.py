#!/usr/bin/env python3
r"""Filter the NER kernel to remove garbage annotations.

Drops entries that match the "stable → StableMarriageProblem"
pattern: a short common-word term mapped to a specific
multi-word PM page title where the term is just a constituent
word, not the actual concept name.

Filter rule (drop iff ALL hold):
  - term_lower is a single word, ≤ 8 chars
  - canon length is ≥ 2x term length
  - canon (lowercased) starts with term_lower
  - canon (lowercased) != term_lower (so "Category" survives — canon
    IS the term, just camelcased)

Keeps:
  - Multi-word terms (e.g. "abelian group" → AbelianGroup)
  - Terms whose canon IS the term (e.g. "category" → Category)
  - Longer terms (≥ 9 chars) — these are usually genuine concept names
  - Entries where the canon doesn't start with the term

Optional whitelist file: one term_lower per line, never dropped
even if scored. Optional blocklist file: one term_lower per line,
always dropped regardless of score.

Usage:
    python scripts/clean-ner-kernel.py \\
        --input /home/joe/code/storage/futon6/data/ner-kernel/terms.tsv \\
        --output /home/joe/code/futon6/data/ner-kernel-clean.tsv \\
        --whitelist data/ner-kernel-whitelist.txt
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


_ENGLISH_DICT: set[str] | None = None


def _load_english_dict() -> set[str]:
    """Load /usr/share/dict/words as a set of lowercased entries.

    Used to distinguish common English words ("stable", "right") that
    cause garbage annotations from math-domain abbreviations ("acc",
    "a.e.", "1-form") which are usually genuine. Falls back to a
    small built-in list if the system dictionary is unavailable.
    """
    global _ENGLISH_DICT
    if _ENGLISH_DICT is not None:
        return _ENGLISH_DICT
    p = Path("/usr/share/dict/words")
    if p.exists():
        _ENGLISH_DICT = {
            w.strip().lower()
            for w in p.read_text(encoding="utf-8", errors="ignore").splitlines()
            if w.strip()
        }
    else:
        _ENGLISH_DICT = set()  # caller will rely on shape-only filter
    return _ENGLISH_DICT


def is_garbage(term_lower: str, canon: str) -> bool:
    """Return True if the entry should be dropped.

    Covers THREE garbage shapes seen in the audit:
      (A) "stable → StableMarriageProblem" — canon starts with term
          and is much longer (term is just a constituent word).
      (B) "right → ConvexAngle" — term is a common English word
          (in /usr/share/dict/words) attached to an unrelated specific
          concept page via PM's `pmdefines`.
      (C) (Not handled here, e.g. "morphism → StructureHomomorphism"
          where term is a math-specific suffix of a word in canon —
          requires deeper analysis. Surfaced via the audit script.)

    Conservative: only drops single-word, short, common-English-word
    terms whose canon is much longer than the term itself. Spares
    "category → Category" (canon = term), math abbreviations not in
    English dict, and longer specialised vocabulary.
    """
    if " " in term_lower:
        return False
    if len(term_lower) > 8:
        return False
    if not canon:
        return False
    cl = canon.lower()
    if cl == term_lower:
        return False  # canon IS the term — keep (Category, Functor, ...)
    if len(canon) < 2 * len(term_lower):
        return False  # canon not much longer — likely a genuine variant
    # Pattern (A): canon starts with term as prefix
    if cl.startswith(term_lower):
        return True
    # Pattern (B): term is a common English word
    eng = _load_english_dict()
    if eng and term_lower in eng:
        return True
    return False


def load_listfile(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    return {
        line.strip().lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Input NER kernel TSV")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output cleaned TSV")
    parser.add_argument("--whitelist", type=Path, default=None,
                        help="Newline-separated term_lower values "
                             "never dropped (override the filter)")
    parser.add_argument("--blocklist", type=Path, default=None,
                        help="Newline-separated term_lower values "
                             "always dropped (regardless of score)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    whitelist = load_listfile(args.whitelist)
    blocklist = load_listfile(args.blocklist)

    with open(args.input, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        rows = list(reader)
    print(f"[clean] header: {header}")
    print(f"[clean] input rows: {len(rows)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_kept = 0
    n_dropped_garbage = 0
    n_dropped_blocklist = 0
    n_whitelist_saved = 0
    dropped_samples: list[tuple[str, str]] = []
    with open(args.output, "w", encoding="utf-8") as fout:
        writer = csv.writer(fout, delimiter="\t")
        writer.writerow(header)
        for row in rows:
            if len(row) < 4:
                writer.writerow(row)
                n_kept += 1
                continue
            term_lower, term_orig, _unused, canon = row[0], row[1], row[2], row[3]
            tl = term_lower.lower().strip()
            if tl in blocklist:
                n_dropped_blocklist += 1
                continue
            if tl in whitelist:
                writer.writerow(row)
                n_kept += 1
                n_whitelist_saved += 1
                continue
            if is_garbage(tl, canon):
                n_dropped_garbage += 1
                if len(dropped_samples) < 20:
                    dropped_samples.append((tl, canon))
                continue
            writer.writerow(row)
            n_kept += 1

    print(f"[clean] kept: {n_kept}")
    print(f"[clean] dropped (garbage pattern): {n_dropped_garbage}")
    print(f"[clean] dropped (blocklist): {n_dropped_blocklist}")
    if whitelist:
        print(f"[clean] whitelist saved (would have been dropped): "
              f"{n_whitelist_saved}")
    print()
    print(f"[clean] sample dropped entries (first 20):")
    for term, canon in dropped_samples:
        print(f"  {term!r:18s} → {canon!r}")
    print(f"[clean] wrote cleaned kernel to {args.output}")


if __name__ == "__main__":
    main()
