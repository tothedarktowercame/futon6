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
import os
import re
import unicodedata
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import futon6_config as config

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = ROOT / "data" / "background-corpus-index.json"


def configured_index() -> Path:
    """Resolve configuration at use time, including in imported callers."""
    return config.authority()

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


# "fubini's theorem" -> "fubini theorem"; also the apostrophe-less "fubinis theorem"
# a normaliser that drops punctuation would leave behind.
POSSESSIVE = re.compile(r"\b(\w+?)(?:'s|\u2019s)\b")


def fold(term: str) -> str:
    """The same name with its dashes and diacritics flattened, for lookup only."""
    term = term.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    return "".join(c for c in unicodedata.normalize("NFKD", term) if not unicodedata.combining(c))


def normalize_term(term: str) -> str:
    term = re.sub(r"[`*_{}()\[\],.;:]+", " ", str(term))
    term = re.sub(r"\s+", " ", term).strip().lower()
    return term


class ConceptAuthority:
    def __init__(self, index_path: Path | None = None):
        self.index_path = Path(index_path) if index_path is not None else configured_index()
        data = json.loads(self.index_path.read_text())
        if not isinstance(data, dict) or data.get("schema-version") != 2:
            raise ValueError(f"{self.index_path}: concept authority requires schema-version 2")
        terms = data.get("terms")
        if not isinstance(terms, dict) or not terms:
            raise ValueError(f"{self.index_path}: concept authority has no terms")
        for term, hits in terms.items():
            entries = hits if isinstance(hits, list) else [hits]
            if not isinstance(term, str) or not term.strip() or not entries or any(
                not isinstance(hit, dict) or any(
                    not isinstance(hit.get(key), str) or not hit[key].strip()
                    for key in ("term", "target", "resolution-kind")
                ) for hit in entries
            ):
                raise ValueError(f"{self.index_path}: malformed authority entry for {term!r}")
        self.terms: dict = data["terms"]
        self.degraded = False
        self.meta = {
            "nnexus-rows": data.get("nnexus-row-count"),
            "nlab-names": data.get("nlab-name-count"),
            "ct-prior": data.get("ct-prior-count"),
            "term-keys": len(self.terms),
            "schema-version": data["schema-version"],
            "index-path": str(self.index_path),
        }
        # These operators are required by the S1 role-gap lookup. Check the
        # actual resolver, so an existing but unsuitable index cannot pass.
        for query in (r"\Hom", r"\End", r"\colim"):
            if self.resolve(query) is None:
                raise ValueError(f"{self.index_path}: required concept {query} does not resolve")

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
        # Mathematics names a result after a person and the authority stores the bare
        # form: "fubini theorem" resolves, "Fubini's theorem" did not, though that is
        # how it is written. 91 of the 845 APM proofs say "X's theorem" at least once
        # (Joe, 2026-09-22).
        plain = POSSESSIVE.sub(r"\1", norm)
        # A name may be typed with an en dash or with its accents ("Hahn-Banach" is
        # stored; "Hahn\u2013Banach" and "Arzel\u00e0-Ascoli" are how papers write it).
        # Folded forms are tried as EXTRA candidates: the stored keys were normalised
        # by the current rule, so folding them away in place would lose entries that
        # carry a dash or an accent of their own.
        folded = fold(plain)
        seen = []
        for c in (norm,
                  plain if plain != norm else None,
                  folded if folded not in (norm, plain) else None,
                  fold(norm) if fold(norm) not in (norm, plain, folded) else None,
                  norm.lstrip("\\") if norm.startswith("\\") else None,
                  norm[:-1] if norm.endswith("s") and len(norm) > 3 else None,
                  ALIASES.get(norm),
                  ALIASES.get(plain),
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
