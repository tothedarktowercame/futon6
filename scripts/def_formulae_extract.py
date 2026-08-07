#!/usr/bin/env python3
"""Produce def-formulae.txt: the definition formulae S11's structural canon consumes.

The missing producer. `sfc_struct_canon --formulae` expects raw LaTeX formulae,
one per line, and pipes each through `sfc_def_structure.bb` (LaTeXML) to get a
:structure tree. Nothing in the pipeline emitted that file, so S11's first half
had never run — and because the stage chained its scripts with `;`, it reported
PASS throughout (E-superpod-hardening H22).

Source: `data/warp/def-snippets.json`, which holds definition prose per concept
with its paper. We take the display/inline math out of those snippets, scoped to
the run's own manifest (corpus identity — H19), and drop fragments too small to
have structure.

  python scripts/def_formulae_extract.py --ids holes/mark7z-e2e16.ids.txt \\
      --out data/runs/mark7z/def-formulae.txt
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# display math first (more likely to be a whole definition), then inline
PATTERNS = [
    re.compile(r"\\\[(.+?)\\\]", re.S),
    re.compile(r"\\begin\{equation\*?\}(.+?)\\end\{equation\*?\}", re.S),
    re.compile(r"\$\$(.+?)\$\$", re.S),
    re.compile(r"(?<!\$)\$([^$]{6,240})\$(?!\$)"),
]

# A formula worth canonicalising has a relation in it: a definition shape is
# built from = ∈ ⊆ → ↦ ≅ etc. Bare terms ("\C{T}") have no structure to merge.
RELATION = re.compile(r"(=|\\in\b|\\subseteq\b|\\subset\b|\\to\b|\\mapsto\b|\\cong\b|"
                      r"\\simeq\b|\\colon\b|:|\\rightarrow\b|\\Rightarrow\b|\\iff\b)")


def clean(f: str) -> str:
    f = " ".join(f.split())
    f = re.sub(r"\\label\{[^}]*\}", "", f)
    return f.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snippets", default="data/warp/def-snippets.json")
    ap.add_argument("--ids", help="restrict to this manifest's papers (corpus identity)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-concept", type=int, default=2)
    ap.add_argument("--min-len", type=int, default=8)
    a = ap.parse_args()

    def R(p):
        return p if os.path.isabs(p) else os.path.join(ROOT, p)

    want = None
    if a.ids:
        want = {l.strip() for l in open(R(a.ids)) if l.strip()}

    d = json.load(open(R(a.snippets)))
    snippets = d.get("snippets", d)

    seen, out = set(), []
    concepts_used = 0
    for concept, rows in snippets.items():
        got = 0
        for row in rows if isinstance(rows, list) else [rows]:
            if want and row.get("paper") not in want:
                continue
            text = row.get("snippet") or ""
            for rx in PATTERNS:
                for m in rx.finditer(text):
                    f = clean(m.group(1))
                    if len(f) < a.min_len or not RELATION.search(f):
                        continue
                    if f in seen:
                        continue
                    seen.add(f)
                    out.append(f)
                    got += 1
                    if got >= a.max_per_concept:
                        break
                if got >= a.max_per_concept:
                    break
            if got >= a.max_per_concept:
                break
        if got:
            concepts_used += 1

    op = R(a.out)
    os.makedirs(os.path.dirname(op), exist_ok=True)
    with open(op, "w") as fh:
        fh.write("\n".join(out) + "\n")
    scope = f" (scoped to {len(want)} manifest papers)" if want else ""
    print(f"wrote {a.out}: {len(out)} formulae from {concepts_used} concepts{scope}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
