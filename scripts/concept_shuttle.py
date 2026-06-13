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
MATHLIB_DEFS = ROOT / "data" / "mathlib-defs-monoidal.json"

import importlib.util as _ilu
_pd = _ilu.spec_from_file_location("mpd", ROOT / "scripts" / "mine_prose_def.py")
mpd = _ilu.module_from_spec(_pd); _pd.loader.exec_module(mpd)


def _camel(t): return "".join(w.capitalize() for w in re.split(r"[\s-]+", t.strip()))


def in_mathlib(term, mathlib_names):
    c = _camel(term).lower()
    # match Foo / FooObj / FooBar against the mined structure/class names
    return next((n for n in mathlib_names
                 if c == n.lower() or c == (n.lower().rstrip("_") )
                 or n.lower().rstrip("obj") == c or c == n.lower().rstrip("obj")), None)


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
