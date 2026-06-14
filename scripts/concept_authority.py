#!/usr/bin/env python3
"""Concept-authority lookup — NNexus + nLab + CT-term-prior, brought online.

Legacy reuse: the authority is already materialized in
data/background-corpus-index.json (80,586 NNexus rows + 20,653 nLab names +
the CT term prior; 130,960 normalized term keys). This is the thin callable
surface over it, used by the Distributed-Proofreaders loop to resolve
role-gap operator-names (\\Hom \\End \\colim ...) against a real concept
authority instead of flattening them to atoms.

    from concept_authority import ConceptAuthority
    ca = ConceptAuthority()
    ca.resolve("colim")  -> {"term": "colimit", "target": "nnexus:colimit", ...}

CLI:  concept_authority.py hom colim "kan extension" ...
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

DEFAULT_INDEX = Path("/home/joe/code/futon6/data/background-corpus-index.json")

# Common math macro/abbreviation -> concept name, where the macro surface
# differs from the indexed concept term. Kept small and explicit (the macro
# RHS usually already carries the full name; these are the residual abbrevs
# the registry's role-gap list surfaces as misses).
ALIASES = {
    "colim": "colimit", "ob": "object", "mor": "morphism", "obj": "object",
    "aut": "automorphism", "hom": "hom", "coker": "cokernel", "ker": "kernel",
    "im": "image", "coim": "coimage", "id": "identity morphism",
    "op": "opposite category", "ev": "evaluation", "coev": "coevaluation",
    "nat": "natural transformation", "lim": "limit", "spec": "spectrum",
}


def normalize_term(term: str) -> str:
    term = re.sub(r"[`*_{}()\[\],.;:]+", " ", str(term))
    term = re.sub(r"\s+", " ", term).strip().lower()
    return term


class ConceptAuthority:
    def __init__(self, index_path: Path = DEFAULT_INDEX):
        data = json.loads(Path(index_path).read_text())
        self.terms: dict = data["terms"]
        self.meta = {
            "nnexus-rows": data.get("nnexus-row-count"),
            "nlab-names": data.get("nlab-name-count"),
            "ct-prior": data.get("ct-prior-count"),
            "term-keys": len(self.terms),
        }

    def resolve(self, term: str) -> dict | None:
        """Resolve a term (or macro surface) to its best concept hit, or None.
        Tries: normalized term, singularised, alias, alias-of-normalized."""
        for cand in self._candidates(term):
            hit = self.terms.get(cand)
            if hit:
                best = hit[0] if isinstance(hit, list) else hit
                return {**best, "matched-on": cand}
        return None

    def _candidates(self, term: str):
        norm = normalize_term(term)
        seen = []
        for c in (norm,
                  norm[:-1] if norm.endswith("s") and len(norm) > 3 else None,
                  ALIASES.get(norm),
                  ALIASES.get(norm.lstrip("\\"))):
            if c and c not in seen:
                seen.append(c)
        return seen


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    ca = ConceptAuthority()
    if not argv:
        print(f"concept authority online: {ca.meta}")
        return 0
    for term in argv:
        hit = ca.resolve(term)
        if hit:
            print(f"  {term:18} -> {hit.get('term')}  [{hit.get('resolution-kind')}:{hit.get('target')}]  (via {hit.get('matched-on')})")
        else:
            print(f"  {term:18} -> (unresolved)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
