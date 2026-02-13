#!/usr/bin/env python3
"""Ratchet check for alphabetic words inside LaTeX math spans.

This script scans generated full TeX, extracts words appearing in math
environments, and enforces an allowlist so new prose leakage inside math
does not silently regress.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

MATH_INLINE_RE = re.compile(r"\\\((.*?)\\\)", re.S)
MATH_DISPLAY_RE = re.compile(r"\\\[(.*?)\\\]", re.S)
OP_MACRO_RE = re.compile(r"\\(?:operatorname|mOpName)\{([^{}]+)\}")
TEXTLIKE_MACRO_RE = re.compile(r"\\(?:text|mathit|mathrm|mathup|emph)\{([^{}]+)\}")
WORD_RE = re.compile(r"[A-Za-z]{2,}")
CMD_RE = re.compile(r"\\[A-Za-z@]+")
SUPSUB_GROUP_RE = re.compile(r"[_^]\s*\{[^{}]*\}")
SUPSUB_ATOM_RE = re.compile(r"[_^]\s*[A-Za-z0-9]+")
COMMENT_RE = re.compile(r"\s*#.*$")


def load_allowlist(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = COMMENT_RE.sub("", raw).strip().lower()
        if not line:
            continue
        out.add(line)
    return out


def write_allowlist(path: Path, words: set[str]) -> None:
    lines = [
        "# Words allowed as bare alphabetic tokens inside math spans.",
        "# Ratchet file: new words should be reviewed before being added.",
        "",
    ]
    lines.extend(sorted(words))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def extract_bare_words(expr: str) -> list[str]:
    s = OP_MACRO_RE.sub(" ", expr)
    s = TEXTLIKE_MACRO_RE.sub(" ", s)
    s = CMD_RE.sub(" ", s)
    s = SUPSUB_GROUP_RE.sub(" ", s)
    s = SUPSUB_ATOM_RE.sub(" ", s)
    return [w.lower() for w in WORD_RE.findall(s)]


def scan_math_words(files: list[Path]) -> tuple[Counter[str], dict[str, list[str]]]:
    counts: Counter[str] = Counter()
    samples: dict[str, list[str]] = defaultdict(list)
    patterns = (MATH_INLINE_RE, MATH_DISPLAY_RE)

    for path in files:
        text = path.read_text(encoding="utf-8")
        for pat in patterns:
            for m in pat.finditer(text):
                expr = m.group(1)
                line = text.count("\n", 0, m.start()) + 1
                excerpt = re.sub(r"\s+", " ", expr).strip()
                excerpt = excerpt[:180]
                for w in extract_bare_words(expr):
                    counts[w] += 1
                    if len(samples[w]) < 2:
                        samples[w].append(f"{path}:{line}: {excerpt}")
    return counts, samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-dir",
        default="data/first-proof/latex/full",
        help="Directory containing problem*-solution-full.tex files",
    )
    parser.add_argument(
        "--allowlist",
        default="scripts/math-word-allowlist.txt",
        help="Path to allowlist file",
    )
    parser.add_argument(
        "--write-allowlist",
        action="store_true",
        help="Overwrite allowlist with currently observed words",
    )
    args = parser.parse_args()

    full_dir = Path(args.full_dir)
    allowlist_path = Path(args.allowlist)
    files = sorted(full_dir.glob("problem*-solution-full.tex"))
    if not files:
        print(f"FAIL: no generated TeX files found under {full_dir}")
        return 1

    counts, samples = scan_math_words(files)
    seen = set(counts.keys())

    if args.write_allowlist:
        write_allowlist(allowlist_path, seen)
        print(f"Wrote allowlist with {len(seen)} words to {allowlist_path}")
        return 0

    allow = load_allowlist(allowlist_path)
    if not allow:
        print(f"FAIL: allowlist is empty or missing: {allowlist_path}")
        print("Run with --write-allowlist once to initialize baseline.")
        return 1

    unexpected = sorted(seen - allow)
    if unexpected:
        print(f"FAIL: found {len(unexpected)} non-allowlisted bare math words:")
        for w in unexpected:
            print(f"  - {w}: {counts[w]}")
            for s in samples[w]:
                print(f"      {s}")
        return 1

    print(
        "Math-word allowlist check passed: "
        f"{len(seen)} observed words, no new non-allowlisted entries."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
