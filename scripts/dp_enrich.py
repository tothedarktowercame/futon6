#!/usr/bin/env python3
"""Post-annotation enrichment over an already-mined DP artifact.

The DP detector (dp_paper_view) descends into MATH MODE — every control
sequence inside every $-span. The PROSE-CONCEPT layer (named math terms in
running text) is mined by a separate detector (build_golden_paper) and never
merged into the dp marks, so the dp-demo shows one of two layers and "terms not
noticed" is the dominant visible defect (DC-1, holes/dp-defect-catalogue.md).

This module closes that gap WITHOUT re-running the detector — it reads the
stored `text + marks` and adds a `concept` mark layer. That is exactly the
"re-mining runs over the *annotated* texts" idea: a feature noticed later is
folded in by a pass over the annotated artifact, not a fresh parse of the TeX.
It is idempotent and read-only w.r.t. golden/ (safe while the live mining run
is still writing there).

    dp_enrich.py <paper-id>          # writes <pid>-enriched.json beside the demo
    # or call enrich(text, marks) -> marks  (used by dp_anatomy_html at render time)
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import build_golden_paper as gp
import anatomy_v0_sweep as sweep  # SHARED math-span tokenizer — the same one the
# checker uses, so detector/checker agree exactly on "where math is" (runbook).

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "showcases" / "ct-anatomy" / "golden"

# Leading words that are never part of the noun phrase — articles, determiners,
# quantifiers, conjunctions, prepositions, light verbs, discourse glue. We trim
# these off the LEFT of a hole-phrase so "fact that the canonical functor"
# becomes "canonical functor". Adjectives (canonical/small/dense/weak/standard…)
# are NOT here, so real multi-word terms survive.
_LEAD_STOP = {
    "the", "a", "an", "this", "that", "these", "those", "each", "every", "all",
    "both", "some", "any", "such", "no", "and", "or", "of", "with", "to", "for",
    "from", "as", "by", "in", "on", "is", "are", "be", "being", "been", "has",
    "have", "fact", "course", "other", "there", "thus", "hence", "given", "let",
    "consists", "consisting", "class", "classes", "works", "work", "standardly",
    "three", "two", "together", "if", "then", "it", "its", "we", "one", "first",
    "second", "example", "following", "same", "where", "which", "when", "while",
    "yields", "choice", "only", "also", "just", "very", "more", "most", "so",
    "defined", "add", "adds", "presents", "present", "contains", "containing",
    "consider", "considered", "obtain", "obtained", "call", "called", "denote",
    "denoted", "introduce", "introduced", "using", "use", "via", "corresponding",
    "assume", "suppose", "note", "recall", "observe", "show", "see", "prove",
    "clearly", "indeed", "moreover", "however", "therefore",
    # quantifiers + nominalization FRAMES ("neither the …", "the existence of …",
    # "the notion of …") — the real concept is what follows, so trim the frame.
    "neither", "either", "existence", "notion", "presence", "absence", "failure",
    "validity", "lack", "collection", "whose", "their", "our", "now", "since",
    "because", "still", "again", "here", "hence",
}
_WORD = re.compile(r"\s*([A-Za-z][A-Za-z-]*)")
# DC-2: author emphasis = an authoritative named term (a definiendum, usually at
# its first occurrence). MUST match check_invariants.EMPH_RE exactly, so every
# term the checker locates gets a detector mark — that is how C-TERM-COVERAGE
# converges rather than staying a permanent debt.
_EMPH_RE = re.compile(r"\\(?:emph|textit|textbf|textsl|textsc|dfn)\s*\{([^{}]*)\}")


_CLAUSE_CUT = re.compile(
    r"\b(?:we|that|which|where|is|are|was|were|has|have|had|be|been|being"
    r"|produces?|gives?|shows?|yields?|implies|means|consists?|such an?"
    r"|construct(?:s|ed)?|prov(?:e|es|ed|ing)|need(?:s)?|exists?|satisf(?:y|ies)"
    r"|requires?|admits?|becomes?|denote[sd]?|view(?:s|ed)?|regard(?:s|ed)?"
    r"|seen|restrict|equivalent|isomorphic|associated)\b", re.I)


def _trim_phrase(text: str, start: int, end: int) -> tuple[int, int]:
    """Drop leading stopwords / stray LaTeX commands, AND collapse a clause to
    its trailing noun phrase: a hole-phrase ending in a concept word can still be
    a clause ("short note we produce such a category"); cut everything up to the
    last clause-marker so only the NP remains ("category", then dropped by the
    multi-word filter)."""
    prev = -1
    while prev != start:                 # iterate: each cut can re-expose the other
        prev = start
        last = None                      # clause-cut: past the last clause marker
        for cm in _CLAUSE_CUT.finditer(text, start, end):
            last = cm
        if last:
            start = last.end()
        while start < end:               # leading stopwords / \commands
            m = _WORD.match(text, start, end)
            if not m:
                break
            tok_at = m.start(1)
            is_cmd = tok_at > 0 and text[tok_at - 1] == "\\"
            if m.group(1).lower() in _LEAD_STOP or is_cmd:
                start = m.end(1)
                continue
            break
        while start < end and text[start] in " \n\t":
            start += 1
    return start, end


# --- lexicon-grounded term spotting (NNexus-style: first-word index + longest
# match). The prose spotter above (hole_marks) is a SUFFIX heuristic; this layer
# instead matches the bootstrap lexicons (nLab + NNexus + CT-prior, 130 960
# terms) directly, so a known term whose head-noun isn't a hardcoded ending —
# "left adjoint", "enough projectives", "long exact sequences" — is still found
# AND grounded (cites its authority entry). Multi-word terms + a curated
# single-word set (the basic level; "learn new terms as we go" is the follow-on).
_SINGLE_TERMS = {
    "functor", "adjoint", "monad", "comonad", "topos", "groupoid", "presheaf",
    "sheaf", "coproduct", "colimit", "pullback", "pushout", "equalizer",
    "coequalizer", "morphism", "isomorphism", "epimorphism", "monomorphism",
    "endomorphism", "automorphism", "bicategory", "coalgebra", "bialgebra",
    "cohomology", "homology", "homotopy", "localization", "colocalization",
    "adjunction", "localizing", "colocalizing", "cokernel", "idempotent",
    "tricategory", "operad", "comonoid", "monoid", "pseudofunctor", "biequivalence",
    "coend", "profunctor", "dinatural", "flock", "herd", "promonoidal",
}
# common adjective / verb / generic / discourse words that are in NNexus but are
# NOT useful single-word concept tags — never admit via provenance (≥7-char gate
# lets these through otherwise). Real noun terms (convolution, functor…) survive.
_SINGLE_STOP = {
    "mapping", "implies", "surjective", "injective", "bijective", "canonical",
    "following", "mathematics", "whenever", "therefore", "property", "standard",
    "natural", "general", "ordinary", "particular", "arbitrary", "respect",
    "because", "however", "moreover", "consider", "obtained", "example",
    "definition", "equation", "section", "theorem", "corollary", "elements",
    "respectively", "essentially", "explicitly", "satisfies", "denotes",
    "consists", "contains", "suppose", "finite", "infinite", "nonzero",
}
_LEXICON = None


def _lexicon():
    """First-word index of (words, key, target) for multi-word + curated
    single-word lexicon terms."""
    global _LEXICON
    if _LEXICON is not None:
        return _LEXICON
    by_first = {}

    def add(key, target):
        ws = key.split()
        if ws:
            by_first.setdefault(ws[0], []).append((ws, key, target))
    try:
        terms = json.loads(
            (ROOT / "data" / "background-corpus-index.json").read_text())["terms"]
    except Exception:
        terms = {}
    def _admit_single(k, v):
        # admit a SINGLE-word index term when it has strong multi-source
        # provenance (>=2 authoritative sources) AND looks like a technical NOUN
        # — long enough (>=7) and not a common adjective/verb/generic word. This
        # lets real terms through (convolution, antipode, functor, duality) while
        # excluding the common-word noise (finite, center, so, implies, mapping)
        # the bare provenance gate over-tagged. Suffix-less short nouns (monad,
        # topos, sheaf, flock) come in via the curated _SINGLE_TERMS set instead.
        if not (isinstance(v, list)
                and any(isinstance(d, dict) and (d.get("domain-count", 0) or 0) >= 2
                        for d in v)):
            return False
        return len(k) >= 7 and k not in _SINGLE_STOP

    for k, v in terms.items():
        if not isinstance(k, str):
            continue
        ws = k.split()
        if len(ws) >= 2 or (len(ws) == 1 and (k in _SINGLE_TERMS or _admit_single(k, v))):
            tgt = (v.get("target") if isinstance(v, dict) else None) or f"lexicon:{k}"
            add(k, tgt)
    for s in _SINGLE_TERMS:
        if not any(w == [s] for w, _, _ in by_first.get(s, [])):
            add(s, f"lexicon:{s}")
    for f in by_first:
        by_first[f].sort(key=lambda t: -len(t[0]))  # longest match first
    _LEXICON = by_first
    return _LEXICON


_LEX_WORD = re.compile(r"[A-Za-z][A-Za-z-]*")


def _singular(w):
    """NNexus-style depluralization so PLURAL prose matches SINGULAR lexicon
    entries ("bicategories"→"bicategory", "functors"→"functor"). Conservative:
    guards -ss/-us/-is/-os so "class", "status", "basis", "topos" are untouched."""
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"                       # categories -> category
    if len(w) > 4 and w.endswith(("ses", "xes", "zes", "ches", "shes")):
        return w[:-2]                             # classes -> class
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is", "os")):
        return w[:-1]                             # functors -> functor
    return w


def lexicon_marks(text, in_math):
    """Longest-match lexicon terms over prose → (start, end, key, target).
    Matching is morphology-aware: a plural token matches its singular lexicon
    entry (and vice versa) via _singular()."""
    lex = _lexicon()
    toks = [(m.group(0).lower(), m.start(), m.end())
            for m in _LEX_WORD.finditer(text)]
    out, i = [], 0
    while i < len(toks):
        hit = None
        t0 = toks[i][0]
        s0 = _singular(t0)
        cands = list(lex.get(t0, ()))
        if s0 != t0:
            cands += [c for c in lex.get(s0, ()) if c not in cands]
        cands.sort(key=lambda c: -len(c[0]))  # longest match first
        for ws, key, tgt in cands:
            n = len(ws)
            if i + n <= len(toks) and all(
                    _singular(toks[i + j][0]) == _singular(ws[j]) for j in range(n)):
                s, e = toks[i][1], toks[i + n - 1][2]
                # skip PROPER NAMES: a multi-word match whose every word is
                # Title-Case in the source is a person (nLab has mathematician
                # pages), not a concept. Eponymous terms ("Bousfield localization")
                # have a lowercase head-noun, so they survive.
                proper = n >= 2 and all(w[:1].isupper() for w in text[s:e].split())
                if not in_math(s, e) and not proper:
                    hit = (s, e, key, tgt, n)
                break
        if hit:
            out.append(hit[:4])
            i += hit[4]
        else:
            i += 1
    return out


# --- prose-term base-rate prior (E-prior-over-terms) -----------------------
# A real math term RECURS across the corpus; a hapax ("interesting abelian
# category", in 1/900 papers) does not. The document-frequency index built by
# build_term_prior.py is the discriminator — the prose-term analogue of the
# macro recognizer-registry, and the learn-and-promote signal (df = promotion
# criterion). It lets us TRIM overfed phrases to their recurring core, EXTEND
# hungry ones to a recurring fuller form, and DROP hapax junk — all without a
# curated stopword list. MSC-repeatable: DP_TERM_PRIOR (or default path) selects
# which class's index is active. If no index is present the pass is a NO-OP, so
# gates never regress on a fresh checkout.
_PRIOR = None
# df floor to call a phrase a "real, recurring term". Env-tunable per corpus.
_PRIOR_FLOOR = int(os.environ.get("DP_TERM_PRIOR_FLOOR", "4"))
# a normalized term may not END on one of these (preposition / light tail) — so
# a hungry extension never stops on "category of modules over".
_BAD_TAIL = {"of", "over", "in", "on", "to", "for", "with", "and", "or", "the",
             "a", "an", "from", "by", "as", "at", "into", "between", "such"}
_PTOK = re.compile(r"[A-Za-z][A-Za-z-]*")


def _prior() -> dict:
    """Lazy-load the active prose-term df index → {term: papers}."""
    global _PRIOR
    if _PRIOR is None:
        path = os.environ.get("DP_TERM_PRIOR") or str(ROOT / "data" / "term-prior-ct.json")
        try:
            _PRIOR = json.loads(Path(path).read_text()).get("df", {})
        except Exception:
            _PRIOR = {}
    return _PRIOR


def _df(phrase: str) -> int:
    return _prior().get(phrase.lower(), 0)


def _prior_normalize(text: str, s: int, e: int, authoritative: bool):
    """Resolve overfed/hungry/hapax against the corpus df prior.

    Returns a normalized (start, end), or None to DROP the candidate.
    `authoritative` (lexicon / author-emphasis) is never trimmed or dropped —
    only extended — so DC-2 / DC-11 / C-TERM-COVERAGE stay satisfied."""
    if not _prior():
        return (s, e)  # no index → no-op (graceful degradation)
    toks = [(m.group(0), s + m.start(), s + m.end()) for m in _PTOK.finditer(text[s:e])]
    if not toks:
        return (s, e)
    floor = _PRIOR_FLOOR

    # HUNGRY: extend with following words if a longer form recurs at least as
    # well and ends on a content word (caps at the prior's max n-gram length).
    base_df = _df(" ".join(w for w, _, _ in toks))
    tail = list(_PTOK.finditer(text[toks[-1][2]: toks[-1][2] + 60]))
    cur = [w for w, _, _ in toks]
    best_ext = None
    for m in tail:
        cur = cur + [m.group(0)]
        if len(cur) > 6:
            break
        d = _df(" ".join(cur))
        if d == 0:
            continue  # gap not in index; no point extending further blindly
        if d >= floor and d >= base_df * 0.4 and cur[-1].lower() not in _BAD_TAIL:
            best_ext = (toks[0][1], toks[-1][2] + m.end())
    if best_ext:
        return best_ext

    if authoritative:
        return (toks[0][1], toks[-1][2])  # trust author/lexicon; just tidy bounds

    n = len(toks)
    if base_df >= floor and toks[-1][0].lower() not in _BAD_TAIL:
        return (toks[0][1], toks[-1][2])  # already a recurring term
    # OVERFED / HAPAX: longest contiguous subphrase with df>=floor, head-anchored
    # first (so "interesting abelian category" -> "abelian category", not
    # "interesting abelian").
    for L in range(n - 1, 0, -1):
        for st in range(n - L, -1, -1):
            seg = toks[st:st + L]
            if seg[-1][0].lower() in _BAD_TAIL:
                continue
            if _df(" ".join(w for w, _, _ in seg)) >= floor:
                return (seg[0][1], seg[-1][2])
    return None  # no recurring subphrase → not a real term, drop


def concept_marks(text: str, marks: list) -> list:
    """Prose terminology marks (kind=`concept`) from build_golden_paper's
    detectors, run over the stored text. Excludes anything overlapping a math
    span (W-ATOMIC: math spans stay atomic — a prose term never reaches inside)."""
    # "where math is" via the SHARED tokenizer (not the math marks) — identical
    # to the checker, so emphasis inside \[...\]/$...$ is judged the same way.
    math_ranges = [(s, e) for s, e, _d, _b in sweep.math_spans(text)]

    def in_math(s, e):
        return any(ms < e and me > s for ms, me in math_ranges)

    cand = []  # (start, end, term, source, target|None)
    # lexicon-grounded terms (NNexus-style) — these are GROUNDED (cite an authority)
    for s, e, key, tgt in lexicon_marks(text, in_math):
        cand.append((s, e, re.sub(r"\s+", " ", text[s:e]).strip(), "lexicon", tgt))
    # DC-2 — author emphasis: authoritative, kept even when single-word ("dense"),
    # exact braces give clean boundaries (no trimming). This is the layer that
    # answers C-TERM-COVERAGE. Gate MUST match check_invariants.C-TERM-COVERAGE.
    for em in _EMPH_RE.finditer(text):
        s, e = em.start(1), em.end(1)
        term = re.sub(r"\s+", " ", text[s:e]).strip()
        if e - s < 3 or not any(c.isalpha() for c in term) or in_math(s, e):
            continue
        if term.endswith(".") or ". " in term:
            continue  # an emphasised SENTENCE (stress), not a named term
        cand.append((s, e, term, "emphasis", None))
    # defined-in-paper + concept-phrase (build_golden_paper's prose detectors).
    # Multi-word, non-trivial only — single residues after trimming are noise.
    defs = gp.mine_definitions(text)
    for mk in gp.select_non_overlapping(
            gp.definition_marks(text, defs) + gp.hole_marks(text, defs)):
        defined = mk.kind == "defined"
        s, e = (mk.start, mk.end) if defined else _trim_phrase(text, mk.start, mk.end)
        term = re.sub(r"\s+", " ", text[s:e]).strip()
        if e <= s or len(term) < 6 or " " not in term or in_math(s, e):
            continue
        cand.append((s, e, term, "defined-in-paper" if defined else "concept-phrase",
                     None))
    # base-rate normalization (E-prior-over-terms): trim overfed phrases to
    # their recurring core, extend hungry ones, drop hapax junk. Author-emphasis
    # and lexicon hits are authoritative (extend-only, never dropped).
    normed = []
    for s, e, term, source, target in cand:
        span = _prior_normalize(text, s, e, source in ("lexicon", "emphasis"))
        if span is None:
            continue
        ns, ne = span
        if ne <= ns:
            continue
        normed.append((ns, ne, re.sub(r"\s+", " ", text[ns:ne]).strip(),
                       source, target))
    cand = normed
    # de-nest: drop a concept fully contained in a longer accepted one (longest
    # wins; a lexicon match ties broken toward grounding via stable order).
    cand.sort(key=lambda c: (c[0], -(c[1] - c[0]), c[3] != "lexicon"))
    _SRC_TIP = {"emphasis": "author-emphasised term",
                "defined-in-paper": "defined in this paper",
                "concept-phrase": "concept (canon link)",
                "lexicon": "grounded term"}
    out, occupied = [], []
    for s, e, term, source, target in cand:
        if any(os <= s and e <= oe for os, oe in occupied):
            continue
        occupied.append((s, e))
        fields = [["term", term], ["source", source]]
        if target:
            fields.append(["grounded", target])
        out.append({
            "start": s, "end": e, "layer": "dp", "kind": "concept",
            "tip": f"term: {term} — {_SRC_TIP[source]}"
                   + (f" [{target}]" if target else ""),
            "fields": fields,
        })
    return out


def enrich(text: str, marks: list) -> list:
    """Return marks + the enrichment layers. Pure; idempotent; safe to call at
    render time AND from the detector (dp_paper_view persists it). If the marks
    already carry a `concept` layer (a re-mined paper), leave them untouched."""
    if any(m.get("kind") == "concept" for m in marks):
        return marks
    try:
        return marks + concept_marks(text, marks)
    except Exception as exc:  # never let enrichment break the render or the mine
        print(f"dp_enrich: concept pass skipped ({exc})", file=sys.stderr)
        return marks


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("usage: dp_enrich.py <paper-id>")
        return 2
    pid = argv[0]
    data = json.loads((GOLD / f"fable-{pid}-dp-emacs.json").read_text())
    before = len(data["marks"])
    data["marks"] = enrich(data["text"], data["marks"])
    added = len(data["marks"]) - before
    out = GOLD / f"fable-{pid}-dp-emacs.enriched.json"
    out.write_text(json.dumps(data))
    print(f"{pid}: +{added} concept marks -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
