#!/usr/bin/env python3
"""Backward definition-scope mining: terms NOT formalised in mathlib still
have prose definitions in PlanetMath / nLab. Extract the same lightweight
skeleton (definiendum + ambient let-context + definiens + such-that
conditions) so every concept a paper uses resolves to *some* structured
definition, formal or prose (definition-scope-mining.md).

Pairs with mine_mathlib_defs.py: that is the forward (formal) source; this
is the backward (prose) fallback for the long tail of working terms like
"$H$-comodule algebra", "$H$-Galois object".

    mine_prose_def.py <term>            # e.g. "comodule algebra"
    mine_prose_def.py --not-in-mathlib <mathlib-defs.json> term1 term2 ...
"""
from __future__ import annotations

import argparse
import importlib.util as ilu
import json
import re
from pathlib import Path

PLANETMATH = Path("/home/joe/code/planetmath")
NW_PATH = Path(__file__).resolve().parent / "nlab-wiring.py"


def _nw():
    spec = ilu.spec_from_file_location("nw", NW_PATH)
    m = ilu.module_from_spec(spec); spec.loader.exec_module(m); return m


def _camel(term: str) -> str:
    return "".join(w.capitalize() for w in re.split(r"[\s-]+", term.strip()))


def find_planetmath(term: str) -> Path | None:
    """Match a PlanetMath .tex by its CamelCase concept name in the filename."""
    cam = _camel(term)
    cands = []
    for f in PLANETMATH.rglob("*.tex"):
        stem = f.stem.split("-", 1)[-1]  # drop the MSC code
        if stem.lower() == cam.lower():
            return f
        # substring fallback, but only for non-trivial stems (avoid "C.tex"
        # matching "GaloisObje[c]t" on a single letter)
        if len(stem) >= 5 and (cam.lower() in stem.lower() or stem.lower() in cam.lower()):
            cands.append(f)
    return cands[0] if cands else None


def body(path: Path) -> str:
    txt = path.read_text(errors="replace")
    m = re.search(r"\\begin\{document\}(.*?)\\end\{document\}", txt, re.S)
    src = m.group(1) if m else txt
    src = re.sub(r"^\s*\\(usepackage|newcommand|newtheorem|theoremstyle)\b.*$",
                 "", src, flags=re.M)
    return re.sub(r"%.*$", "", src, flags=re.M).strip()


def skeletonize(term: str, path: Path, nw) -> dict:
    txt = body(path)
    # definiendum: the bolded/emphasised defined term (PlanetMath convention)
    dfn = re.search(r"\\(?:textbf|emph|definitionname|pmdefines)\{([^}]+)\}", txt)
    definiendum = (dfn.group(1) if dfn else term).strip()
    # the defining sentence: "A <definiendum> is a/an <definiens> ..."
    defsent = re.search(r"(A|An|The)\s+\\?\w*\{?[^.]{0,160}?\bis\s+(?:a|an|the)\s+"
                        r"([^.]+?)(?:\.|satisfying|such that)", txt, re.I)
    definiens = (defsent.group(2).strip()[:140] if defsent else "")
    # ambient let-context (Let $X$ be a Y) + such-that conditions, via the scope detector
    scopes = nw.detect_scopes(f"pm-{path.stem}", txt)
    ambient = [e.get("text") or e.get("latex")
               for s in scopes if s["hx/type"] == "bind/let"
               for e in s["hx/ends"] if e.get("role") in ("symbol", "type")]
    conditions = [s.get("hx/content", {}).get("match", "")[:80]
                  for s in scopes if s["hx/type"].startswith("constrain")]
    cond_words = re.findall(r"\b(satisfying|such that)\b", txt, re.I)
    return {
        "definiendum": definiendum, "kind": "prose-def", "source": "planetmath",
        "file": str(path.relative_to(PLANETMATH)),
        "ambient": ambient[:6],
        "definiens": definiens,
        "has-conditions": bool(cond_words) or bool(conditions),
        "conditions": conditions[:4],
        "prose-derived": True,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("terms", nargs="*")
    ap.add_argument("--not-in-mathlib", metavar="JSON")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    terms = list(a.terms)
    if a.not_in_mathlib:
        mathlib = {d["name"].lower() for d in json.loads(Path(a.not_in_mathlib).read_text())}
        terms = [t for t in terms if _camel(t).lower() not in mathlib]
        print(f"backward set (not in mathlib): {terms}")
    nw = _nw()
    out = []
    for t in terms:
        p = find_planetmath(t)
        if not p:
            print(f"\n✗ {t}: no PlanetMath def found"); continue
        sk = skeletonize(t, p, nw)
        out.append(sk)
        print(f"\n● {sk['definiendum']}  [{sk['source']}:{sk['file']}]  (prose skeleton)")
        if sk["ambient"]:
            print(f"   ambient: {' ; '.join(filter(None, sk['ambient']))}")
        if sk["definiens"]:
            print(f"   is a: {sk['definiens']}")
        print(f"   conditions: {'yes — ' + ' | '.join(sk['conditions'][:2]) if sk['has-conditions'] else 'none stated'}")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
