#!/usr/bin/env python3
"""The definition shuttle — weaves a concept across three coverage layers,
the three Norns of a definition's existence:

  Urðr (既, what-is-laid-down)  → FORMAL libraries (mathlib; extensible to
                                  Coq/Isabelle for what Lean "should" have).
  Verðandi (化, the-becoming)   → PROSE commons (PlanetMath / nLab).
  Skuld (應/債, the-debt)        → arXiv USAGE — terms USED, which pull a
                                  definition into being where none is laid down.

The shuttle reports each term's coverage triple and the verdict the weave
implies — most importantly the DEBT: terms used in arXiv but defined in
neither the formal nor the prose layer (a formalisation/definition hole).

    concept_shuttle.py <term...> [--usage-tex DIR_OR_FILE]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path("/home/joe/code/futon6")
# prefer the FULL mathlib index (3454 structure/class defs); fall back to the
# Monoidal-only slice if the broad mine has not been run yet.
_BROAD = ROOT / "data" / "mathlib-defs.json"
MATHLIB_DEFS = _BROAD if _BROAD.exists() else ROOT / "data" / "mathlib-defs-monoidal.json"

import importlib.util as _ilu
_pd = _ilu.spec_from_file_location("mpd", ROOT / "scripts" / "mine_prose_def.py")
mpd = _ilu.module_from_spec(_pd); _pd.loader.exec_module(mpd)


def _camel(t): return "".join(w.capitalize() for w in re.split(r"[\s-]+", t.strip()))


# Generic categorical head-nouns a prose concept may carry that mathlib folds
# into a qualified identifier ("Galois object" -> IsGalois / PointedGaloisObject).
GENERIC_NOUNS = {"object", "objects", "morphism", "structure", "element"}


def _boundary_suffix(name, key):
    """True if mathlib NAME ends with camel-KEY (lowercased) at a CamelCase token
    boundary — so 'galoisobject' matches 'Pointed|GaloisObject', not a mid-word
    substring. The boundary guard keeps the alias match from latching a
    coincidental tail (e.g. it will not call 'AddGroup' a match for 'dGroup')."""
    nl = name.lower()
    if key == nl or not nl.endswith(key):
        return False
    start = len(name) - len(key)
    return start == 0 or name[start].isupper()


def in_mathlib(term, mathlib_names):
    """Connect a prose concept name to a mathlib identifier. Beyond exact
    camel-case equality, bridge the two mathlib naming conventions GENERALLY
    (not a Galois one-off): properties are stated as Is<Concept> ("Galois
    object" -> IsGalois) and structures are qualified with prefixes ("Galois
    object" -> PointedGaloisObject). Priority: an EXACT identifier always wins
    over an aliased variant, and aliases never fire for a bare generic noun."""
    c = _camel(term)
    cl = c.lower()
    words = re.split(r"[\s-]+", term.strip())
    multi = len(words) >= 2
    # head = the concept minus a trailing generic noun ("Galois object"->"galois")
    head = (_camel(" ".join(words[:-1])).lower()
            if multi and words[-1].lower() in GENERIC_NOUNS else None)
    # (1) direct camel-equality (Foo / FooObj / FooBar, with obj/_ normalization)
    #     — an exact identifier must win before any alias rule is tried.
    for n in mathlib_names:
        nl = n.lower()
        if cl == nl or cl == nl.rstrip("_") or nl.rstrip("obj") == cl:
            return n
    # (2) Is<Concept> predicate convention (the dominant false-DEBT cause).
    for n in mathlib_names:
        nl = n.lower()
        if nl == "is" + cl or (head and nl == "is" + head):
            return n
    # (3) CamelCase-boundary suffix = a qualified variant of the concept; require
    #     multi-token (so a bare "group"/"module" cannot match everything) and
    #     pick the SHORTEST hit as the most canonical identifier.
    if multi:
        cands = [n for n in mathlib_names if _boundary_suffix(n, cl)]
        if cands:
            return min(cands, key=len)
    return None


def in_planetmath(term):
    p = mpd.find_planetmath(term)
    return str(p.relative_to(mpd.PLANETMATH)) if p else None


def used_in(term, usage_path):
    """arXiv usage proxy: does the term phrase occur in the usage corpus?"""
    if not usage_path:
        return None
    # phrase grep: words joined by space/hyphen (and optional $...$ between)
    pat = r"[ -]+".join(re.escape(w) for w in term.split())
    try:
        r = subprocess.run(["grep", "-rliE", pat, str(usage_path)],
                           capture_output=True, text=True, timeout=30)
        return r.returncode == 0
    except Exception:
        return None


def verdict(lean, pm, used):
    if used and not lean and not pm:
        return "DEBT — used but undefined (Skuld: a definition hole)"
    if used and not lean and pm:
        return "formalisation candidate (prose only; Lean should have it)"
    if lean and pm:
        return "fully covered (formal + prose)"
    if lean and not used:
        return "formal, dormant in this corpus"
    if pm and not used:
        return "prose, dormant"
    if lean:
        return "formal (Lean)"
    if pm:
        return "prose (PlanetMath)"
    return "uncovered everywhere"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("terms", nargs="+")
    ap.add_argument("--usage-tex", default="/tmp/ep0809b")
    a = ap.parse_args(argv)
    mathlib_names = ([d["name"] for d in json.loads(MATHLIB_DEFS.read_text())]
                     if MATHLIB_DEFS.exists() else [])
    print(f"{'term':28} {'Urðr/Lean':12} {'Verðandi/PM':14} {'Skuld/used':10} verdict")
    print("-" * 100)
    for t in a.terms:
        lean = in_mathlib(t, mathlib_names)
        pm = in_planetmath(t)
        used = used_in(t, a.usage_tex)
        lc = ("✓ " + lean)[:12] if lean else "–"
        pc = ("✓ " + pm.split("/")[-1].split("-")[-1][:11]) if pm else "–"
        uc = "✓" if used else ("–" if used is False else "?")
        print(f"{t:28} {lc:12} {pc:14} {uc:10} {verdict(lean, pm, used)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
