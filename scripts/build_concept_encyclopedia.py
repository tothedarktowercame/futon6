#!/usr/bin/env python3
"""Build the "cheap PlanetMath" concept encyclopedia (E-superpod-mark3, Joe) —
the structure-first NOUN substrate. Each entry grounds a concept to its ACTUAL
in-corpus definition passage + NNexus/nLab provenance + concept-dependency edges
+ centrality, with the deep semi-formalisation (:structure / defining property)
left as an honest HOLE for the mark3/superpod fill — exactly as the IATC graphs
do for reasoning.

Assembled from already-built artifacts (no fresh mine):
  term-prior-ct.json            -> df ranking
  background-corpus-index.json  -> provenance (nLab/NNexus target, MSC, domains)
  warp/def-snippets.json        -> the real definition passages (972 concepts)
  warp/concept-graph.json       -> PageRank / dependency centrality
  warp/defined-index.json       -> concept -> defining papers

    build_concept_encyclopedia.py [--n 200] [--msc ct] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA, WARP = ROOT / "data", ROOT / "data" / "warp"

_MORPH = re.compile(r"morphism|functor|\bmap\b|transformation|arrow|adjoint")
_PROP = re.compile(r"property|condition|ness$|bility$|\baxiom")
_CONSTR = re.compile(r"ization|construction|completion|localization|quotient|product$|limit$")


def _kind(c: str) -> str:
    if _MORPH.search(c):
        return "morphism"
    if _CONSTR.search(c):
        return "construction"
    if _PROP.search(c):
        return "property"
    return "object"


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-") or "x"


_GENUS = re.compile(
    r"\bis\s+(?:a|an|the)\s+([a-z][a-z -]{2,40}?)"
    r"(?:\s+(?:where|such that|satisfying|with|in which|whose|that|if|for which|"
    r"together with)\b|[.,;])", re.I)
_DIFF_HEAD = re.compile(
    r"\b(?:where|such that|satisfying|with the property that|in which|if|"
    r"for which|together with|whose)\b", re.I)


# split differentiae before a quantifier-led NEW condition, NOT inside a noun
# list ("a kernel and a cokernel" stays whole; "… and every monic is …" splits).
_DIFF_SPLIT = re.compile(r";\s*|\s+and\s+(?=(?:every|each|all|any|no|there|for\s|"
                         r"the\s+\w+\s+(?:is|are|has|have))\b)", re.I)


def _refs(clause, vocab, c):
    return [t for t in vocab if t != c and len(t) >= 4
            and re.search(r"(?<![a-z])" + re.escape(t) + r"(?![a-z])", clause, re.I)]


def _components(c, gloss, vocab, ref_vocab):
    """Cheap structural breakdown of the GLOSS into genus + differentiae (the
    APM-Xi component form). Each differentia is a clause + the concepts it refs.
    The deep ∀/∃ formalisation (typed-item form) is left as a hole."""
    genus = None
    m = _GENUS.search(gloss)
    if m:
        g = _clean(m.group(1)).lower().strip(" .")
        cand = [t for t in vocab if g == t or g.endswith(" " + t) or g == t + "s"]
        genus = max(cand, key=len) if cand else g
    diff = []
    md = _DIFF_HEAD.search(gloss)
    if md:
        for cl in _DIFF_SPLIT.split(gloss[md.end():]):
            cl = _clean(cl or "").strip(" .")
            if len(cl) >= 6:
                diff.append({"clause": cl[:160], "refs": _refs(cl, ref_vocab, c)[:6]})
    return {"genus": genus, "differentiae": diff[:6]}


def formalisation_hole() -> dict:
    return {
        "kind": "hole",
        "type": "formalise-structure",
        "wanted": "render differentiae as typed forall/exists conditions (APM-Xi item form)",
    }


def audit_entries(entries, *, sample_size: int = 100) -> dict:
    """Completeness audit for structure-first concept encyclopedia entries."""
    sample = list(entries)[:sample_size]
    fields = {
        "def_passage": lambda e: bool((e.get("gloss") or {}).get("text")),
        "provenance": lambda e: bool(e.get("provenance")),
        "dep_edge": lambda e: bool(e.get("depends_on") or e.get("depends-on")),
        "centrality": lambda e: e.get("pagerank") is not None,
        "typed_hole": lambda e: any(
            h.get("kind") == "hole" and h.get("type")
            for h in e.get("holes", [])
        ),
    }
    counts = {name: sum(1 for e in sample if pred(e)) for name, pred in fields.items()}
    return {
        "sample_size": len(sample),
        "counts": counts,
        "rates": {name: (count / len(sample) if sample else 1.0)
                  for name, count in counts.items()},
    }


def _edn(o) -> str:
    """Minimal EDN serializer matching the IATC `.edn` style: keyword keys,
    keyword :kind/:concept-id/:depends-on refs, strings quoted, lists -> vectors."""
    if o is None:
        return "nil"
    if isinstance(o, bool):
        return "true" if o else "false"
    if isinstance(o, (int, float)):
        return repr(o)
    if isinstance(o, _Kw):
        return o.s
    if isinstance(o, str):
        return json.dumps(o, ensure_ascii=False)
    if isinstance(o, (list, tuple)):
        return "[" + " ".join(_edn(x) for x in o) + "]"
    if isinstance(o, dict):
        return "{" + " ".join(f"{(':' + k) if isinstance(k, str) else _edn(k)} {_edn(v)}"
                              for k, v in o.items()) + "}"
    raise TypeError(type(o))


class _Kw:
    __slots__ = ("s",)

    def __init__(self, name):
        self.s = name if name.startswith(":") else ":" + name


def _definition(c, snips):
    """Pick the most DEFINITIONAL passage and return (paper, the definition
    sentence) — prefer "(a|an) C is …" / "\\emph{C} is …" / "C is called …"
    framings over a mere mention, and centre on that sentence."""
    surf = re.escape(c)
    frame = re.compile(
        r"(?:\\(?:emph|textit|textbf|dfn|def|defn)\{)?(?:an?\s+|the\s+)?" + surf
        + r"\}?\s+(?:is|are|is\s+called|will\s+be\s+called|denotes?|means?|consists?)\b",
        re.I)
    mention = re.compile(r"(?:an?\s+)" + surf + r"\b", re.I)
    best = None  # (score, paper, sentence)
    for s in snips:
        t = s.get("snippet", "")
        # skip bibliography / reference snippets — they mention the term in a
        # citation, not a definition.
        if re.search(r"\\bibitem|\\bibliography|\bpp\.\s*\d|\bvol\.|\bArch\s+Math|"
                     r"\d{4}[a-z]?[.,)]\s*$|\\emph\{[A-Z][a-z]+ [A-Z]", t):
            continue
        m = frame.search(t) or mention.search(t)
        if not m:
            continue
        s0 = t.rfind(". ", 0, m.start())
        s0 = s0 + 2 if s0 != -1 else max(0, m.start() - 30)
        e0 = t.find(". ", m.start())
        e0 = e0 + 1 if e0 != -1 else min(len(t), m.end() + 240)
        cand = _clean(t[s0:e0])[:480]
        low = cand.lower()
        score = (3 if frame.search(t) else 0) + (2 if " is a" in low or "is called" in low
                                                 or " is the" in low or "is defined" in low else 0)
        if best is None or score > best[0]:
            best = (score, s.get("paper"), cand)
    if best and best[0] > 0:
        return best[1], best[2]
    nonbib = [s for s in snips
              if not re.search(r"\\bibitem|\\bibliography|\bpp\.\s*\d|\bvol\.|"
                               r"\bArch\s+Math|\d{4}[a-z]?[.,)]\s*$", s.get("snippet", ""))]
    pool = nonbib or snips
    b = max(pool, key=lambda s: len(s.get("snippet", "")))
    return b.get("paper"), _clean(b.get("snippet", ""))[:300]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--msc", default="ct")
    ap.add_argument("--out", type=Path, default=DATA / "concept-encyclopedia-ct.json")
    a = ap.parse_args(argv)

    df = json.loads((DATA / f"term-prior-{a.msc}.json").read_text())["df"]
    idx = json.loads((DATA / "background-corpus-index.json").read_text())["terms"]
    snippets = json.loads((WARP / "def-snippets.json").read_text())["snippets"]
    auth = {e["concept"]: e for e in
            json.loads((WARP / "concept-graph.json").read_text()).get("authority", [])}
    c2p = json.loads((WARP / "defined-index.json").read_text())["concept_to_papers"]

    vocab = set(snippets)                       # the concept lexicon for dep edges
    # broader vocab for differentia refs: add common single-word terms (kernel,
    # cokernel, functor, monomorphism …) the def-snippet set alone misses.
    ref_vocab = vocab | {t for t, n in df.items()
                         if " " not in t and len(t) >= 4 and n >= 80}
    # rank concepts that HAVE a real definition snippet, by corpus df
    ranked = sorted((c for c in snippets if df.get(c, 0) >= 4),
                    key=lambda c: -df.get(c, 0))[:a.n]

    entries = []
    for c in ranked:
        snips = snippets[c]
        defpaper, deftext = _definition(c, snips)
        # concept-dependency edges: (a) STRUCTURAL — a vocab concept that is a
        # proper sub-phrase of c ("abelian category" -> "category"); (b) NAMED in
        # the definition sentence. Structural deps first (most reliable).
        struct = {t for t in vocab if t != c and len(t) >= 4
                  and re.search(r"(?<![a-z])" + re.escape(t) + r"(?![a-z])", c)}
        named = {t for t in vocab if t != c and len(t) >= 5 and t not in struct
                 and re.search(r"(?<![a-z])" + re.escape(t) + r"(?![a-z])", deftext, re.I)}
        deps = sorted(struct, key=lambda t: -df.get(t, 0)) + \
            sorted(named, key=lambda t: -df.get(t, 0))[:10]
        prov = (idx.get(c) or [{}])[0]
        au = auth.get(c, {})
        papers = c2p.get(c, [])
        entries.append({
            "concept": c, "msc": a.msc, "kind": _kind(c),
            "df": df.get(c, 0),
            "pagerank": au.get("pagerank"), "used_papers": au.get("used_papers"),
            "depends_on": deps,                 # concept-import edges
            "gloss": {"paper": defpaper, "text": deftext},   # the prose definition
            "components": _components(c, deftext, vocab, ref_vocab),  # genus + diff
            "defined_in": {"n_papers": len(papers), "sample": papers[:5]},
            "provenance": {k: prov.get(k) for k in ("target", "resolution-kind", "msc", "domains")
                           if prov.get(k) is not None},
            # the deep formalisation (differentiae as ∀/∃ typed-item conditions,
            # APM-Xi form) is the mark3/superpod job:
            "holes": [formalisation_hole()],
        })

    out = {"schema": "concept-encyclopedia-v0", "msc": a.msc,
           "n_concepts": len(entries), "note": "cheap structure-first scaffold; "
           "definitions are real in-corpus passages; :structure is a mark3 hole",
           "audit": audit_entries(entries, sample_size=min(100, len(entries))),
           "entries": entries}
    a.out.write_text(json.dumps(out, indent=1))

    # EDN per-concept files (the superpod handoff units, like the IATC graphs):
    # keyword-slug :concept/id + :depends-on so the concept-dependency graph is
    # machine-traversable; string :name + :text preserved.
    edn_dir = a.out.parent / "concept-encyclopedia" / a.msc
    edn_dir.mkdir(parents=True, exist_ok=True)
    for e in entries:
        edn = {
            "concept/id": _Kw(_slug(e["concept"])), "name": e["concept"],
            "msc": e["msc"], "kind": _Kw(e["kind"]), "df": e["df"],
            "pagerank": e["pagerank"], "used-papers": e["used_papers"],
            "depends-on": [_Kw(_slug(t)) for t in e["depends_on"]],
            "gloss": {"paper": e["gloss"]["paper"], "text": e["gloss"]["text"]},
            "components": {
                "genus": _Kw(_slug(e["components"]["genus"])) if e["components"]["genus"] else None,
                "differentiae": [{"clause": d["clause"],
                                  "refs": [_Kw(_slug(r)) for r in d["refs"]]}
                                 for d in e["components"]["differentiae"]]},
            "defined-in": {"n-papers": e["defined_in"]["n_papers"],
                           "sample": e["defined_in"]["sample"]},
            "provenance": e["provenance"],
            "holes": [{"kind": _Kw("hole"), "type": _Kw("formalise-structure"),
                       "wanted": "render differentiae as typed forall/exists conditions (APM-Xi item form)"}],
        }
        (edn_dir / f"{_slug(e['concept'])}.edn").write_text(_edn(edn) + "\n",
                                                            encoding="utf8")
    print(f"wrote {a.out}  ({len(entries)} concepts) + {len(entries)} .edn in {edn_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
