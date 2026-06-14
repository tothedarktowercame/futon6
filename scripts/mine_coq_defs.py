#!/usr/bin/env python3
"""Mine definition-NAMES from Coq math-comp (.v) — a second formal source for
the three-Norn shuttle's Urðr column (the shuttle was always built to pluralize
beyond Lean). Same shape as mine_mathlib_defs.py: definiendum = the declared
name; we harvest the name + kind so the shuttle can answer "does Coq/mathcomp
define term Y?" and flip a verdict from "Lean should have it" to "covered in
Coq" (or confirm a cross-prover DEBT hole).

math-comp uses both vanilla Coq decls and Hierarchy Builder (HB.*). Names are
terse (lmodType, algType, comRingType) so the shuttle's term→name match must be
fuzzy (substring on the camel/stem), as it already is for mathlib abbreviations.

    mine_coq_defs.py [--root DIR] [--json out.json] [--test term ...]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path("/home/joe/code/storage/futon6/data/formal-sources/math-comp")
OUT = Path("/home/joe/code/futon6/data/coq-mathcomp-defs.json")

DECL_RE = re.compile(
    r"^\s*(?:Global\s+|Local\s+|#\[[^\]]*\]\s*)?"
    r"(Definition|Record|Structure|Variant|Inductive|Class|Fixpoint|"
    r"Notation|Theorem|Lemma|Module)\s+([A-Za-z_][\w']*)", re.M)
# Hierarchy Builder: `HB.structure Definition Name := ...`, `HB.mixin Record Name`
HB_RE = re.compile(
    r"^\s*HB\.(?:structure|mixin|factory|builders)\s+"
    r"(?:Definition|Record)\s+([A-Za-z_][\w']*)", re.M)


def mine(root: Path):
    defs = {}
    for vf in root.rglob("*.v"):
        txt = vf.read_text(errors="replace")
        rel = str(vf.relative_to(root))
        for kw, name in DECL_RE.findall(txt):
            defs.setdefault(name, {"name": name, "kind": kw.lower(), "file": rel})
        for name in HB_RE.findall(txt):
            defs[name] = {"name": name, "kind": "hb-structure", "file": rel}
    return list(defs.values())


def _stem(term: str) -> str:
    return re.sub(r"[^a-z]", "", term.lower())


def lookup(term: str, names):
    """Fuzzy: a mathcomp name whose lowercased form contains the term stem (or
    vice-versa for short terms). Mirrors the mathlib abbreviation match."""
    st = _stem(term)
    if len(st) < 3:
        return None
    hits = [n for n in names
            if st in n["name"].lower() or n["name"].lower().rstrip("type") == st]
    # prefer Type/structure decls (the real objects), shortest name
    hits.sort(key=lambda n: (n["kind"] not in ("hb-structure", "structure", "record"),
                             len(n["name"])))
    return hits[0] if hits else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--json", default=str(OUT))
    ap.add_argument("--test", nargs="*")
    a = ap.parse_args(argv)
    names = mine(Path(a.root))
    Path(a.json).write_text(json.dumps(names))
    print(f"mined {len(names)} Coq/math-comp definition-names → {a.json}")
    # what algebra-hierarchy objects did we get? (sanity)
    objs = sorted(n["name"] for n in names if n["name"].lower().endswith("type"))
    print(f"  {len(objs)} *Type structures, e.g.: {', '.join(objs[:14])}")
    if a.test:
        print("\nshuttle lookups (Urðr/Coq column):")
        for t in a.test:
            hit = lookup(t, names)
            print(f"  {t:24} → {hit['name']+' ['+hit['kind']+']' if hit else '– (not in mathcomp)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
