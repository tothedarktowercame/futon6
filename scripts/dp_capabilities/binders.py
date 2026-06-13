"""Binder and definiens detection for DP paper views."""

from __future__ import annotations

import importlib.util as _ilu
import re
from pathlib import Path

BINDER_RE = re.compile(
    r"\b(?:Let|let)\s+(\$[^$]+\$)\s+(?:be|denote)\s+(?:an?\s+|the\s+)?"
    r"([^.,;:]+?)(?=[.,;:]|\s+such that|\s+and\s+\$|\s+in\s+\$|$)")
CONJUNCT_RE = re.compile(
    r"\band\s+(\$[^$]+\$)\s+(?:be\s+)?(?:an?\s+|the\s+)?"
    r"([^.,;:]+?)(?=[.,;:]|\s+such that|\s+and\s+\$|$)")
# "$M$ is a right $A$-module" — a hypothesis binding (Joe). Article REQUIRED
# (a/an/the) so a predicate like "$f$ is continuous" / "$X$ is closed" does NOT
# match — only genuine type assignments "$X$ is a/an/the <type>".
# NB: no leading \b — a word boundary never holds between a space and the "$"
# delimiter (both non-word chars), so \b would make this never match.
IS_RE = re.compile(
    r"(?:^|(?<=\s))(\$[^$]+\$)\s+(?:is|are)\s+(?:an?|the)\s+"
    r"([^.,;:]+?)(?=[.,;:]|\s+such that|\s+and\s+\$|$)")

# DEFINED-IN-PAPER (defined-in-paper capability): in-paper definienda introduced
# by definition prose ground their later uses — the C-SYM-GROUND debt the
# Let/is-a binders miss (Joe's "non-Let forms the grounding never consulted").
# High-precision forms only; group(1)=$symbol$, group(2)=definiens phrase, fed
# into the same binding harvest as BINDER_RE/IS_RE.
DEFINE_RES = [
    # "(we) define $X$ to be/as/by <phrase>"
    re.compile(r"\b(?:[Ww]e\s+)?[Dd]efine\s+(\$[^$]+\$)\s+(?:to\s+be|as|by)\s+"
               r"([^.,;:]+?)(?=[.,;:]|\s+such that|$)"),
    # "$X$ is defined to be/as/by <phrase>"
    re.compile(r"(\$[^$]+\$)\s+is\s+defined\s+(?:to\s+be|as|by)\s+"
               r"([^.,;:]+?)(?=[.,;:]|\s+such that|$)"),
    # "(we) denote by $X$ (the/a) <phrase>"
    re.compile(r"\b(?:[Ww]e\s+)?[Dd]enote\s+by\s+(\$[^$]+\$)\s+(?:the\s+|an?\s+)?"
               r"([^.,;:]+?)(?=[.,;:]|\s+such that|$)"),
    # "(we) write $X$ for <phrase>"
    re.compile(r"\b(?:[Ww]e\s+)?[Ww]rite\s+(\$[^$]+\$)\s+for\s+"
               r"([^.,;:]+?)(?=[.,;:]|\s+such that|$)"),
]

# QUANTIFIER + WHERE-BINDING (Joe's tail): the C-SYM-GROUND residue the
# Let/is-a/define binders never consult. A quantifier INTRODUCES a bound symbol
# with no type phrase ("for all $x$") — the scope-manifest harvest above drops
# it (`sym and typ` fails when there is no type). A where/with clause is a
# post-hoc gloss that DOES carry a type phrase, like IS_RE but lead by
# where/with. Both feed the same _add_binding harvest, so they only ADD bindings
# (symbol -> symbol-grounded); they emit no scope/coverage marks (zero wf impact,
# denominator unchanged) and can only make `grounded` rise.
#   (rx, label) — label is the binding occasion when there is no concept type.
QUANT_RES = [
    # "for all / for every / for each / for some / for any  $x$"
    (re.compile(r"\bfor\s+(?:all|every|each|some|any)\s+(\$[^$]+\$)"),
     "quantified variable (for all/every/some)"),
    # "there exist(s) (a/an/some) $x$"
    (re.compile(r"\bthere\s+exists?\s+(?:an?\s+|some\s+)?(\$[^$]+\$)"),
     "existentially quantified variable"),
    # bare \forall x / \exists x inside math (single Latin letter; high-precision)
    (re.compile(r"\\(?:forall|exists)\s*\$?\s*([A-Za-z])\b"),
     "quantified variable ($\\forall$/$\\exists$)"),
]
# "where $x$ is/denotes <phrase>" / "with $x$ a/an/the <phrase>" — typed gloss.
WHERE_RES = [
    re.compile(r"\bwhere\s+(\$[^$]+\$)\s+(?:is|are|denotes?|stands?\s+for)\s+"
               r"(?:an?\s+|the\s+)?([^.,;:]+?)(?=[.,;:]|\s+and\s+\$|$)"),
    re.compile(r"\bwith\s+(\$[^$]+\$)\s+(?:being\s+)?(?:an?|the)\s+"
               r"([^.,;:]+?)(?=[.,;:]|\s+and\s+\$|$)"),
]

# APPOSITIVE TYPING (claude-2's residue analysis: ~78% of the ungrounded tail).
# "<determiner> <qualifiers> <TYPE-NOUN> $X$" introduces X WITH its type by
# apposition — "a Hopf algebra $H$", "the monoidal category $\C$", "an
# $H$-comodule algebra $A$" — the type-THEN-symbol direction the Let/is-a binders
# (symbol-THEN-type) structurally miss. HIGH PRECISION (claude-1), three anchors:
#   (1) a determiner leads (a/an/the/any/some/every/each) — never bare adjacency;
#   (2) the noun IMMEDIATELY before the symbol is a STRUCTURAL TYPE-NOUN from the
#       lexicon below (not "proof"/"number"/"diagram"/"case" etc.);
#   (3) the $symbol$ immediately follows that type-noun.
# group(1)=type phrase (ends in the type-noun), group(2)=$symbol$; fed into the
# same _add_binding harvest as the other binders (binding-contributor only — it
# adds no marks and shrinks no denominator, so it can only make `grounded` rise).
TYPE_NOUNS = (
    r"algebras?|coalgebras?|bialgebras?|subalgebras?|superalgebras?|"
    r"modules?|comodules?|bimodules?|submodules?|"
    r"categor(?:y|ies)|functors?|morphisms?|homomorphisms?|isomorphisms?|"
    r"objects?|groups?|groupoids?|subgroups?|monoids?|semigroups?|"
    r"rings?|fields?|sets?|subsets?|spaces?|maps?|mappings?|functions?|"
    r"ideals?|operads?|sheaves|schemes?|varieties|manifolds?|complexes|"
    r"transformations?|representations?|extensions?|monomials?|"
    # conservative extension (claude-2's final lever): high-confidence,
    # unambiguous type-nouns only. EXCLUDES form/relation/theory/number — those
    # frequently do NOT type a single symbol ("a relation $R$ on", "number $n$ of
    # elements"). All plurals end in "s", so the plural-conjunct gate stays valid.
    r"points?|elements?|pairs?|vectors?|sequences?|famil(?:y|ies)|bundles?|"
    r"lattices?")
APPOSITIVE_RE = re.compile(
    r"\b(?:an?|the|any|some|every|each)\s+"
    r"([^.,;:]{0,40}?\b(?:" + TYPE_NOUNS + r"))\s+(\$[^$]+\$)")
# Conjuncts sharing ONE plural appositive type: "objects $K$ and $L$", "spaces
# $(V,q_V)$ and $(W,q_W)$" — each extra $symbol$ inherits the same type-noun.
# Anchored (re.match from the previous symbol's end) so only a CONTIGUOUS
# ", "/" and " chain attaches; gated to PLURAL type-nouns by the caller (a
# singular "the map $f$ and $g$" must NOT drag $g$ in — $g$ may be unrelated).
APPOS_CONJ_RE = re.compile(r"\s*(?:,\s*and\s+|,\s*|\s+and\s+)(\$[^$]+\$)")

# DEF-EQUATION / NAME-VERB (claude-2's sequence). A definitional LEAD-IN verb
# (set/put/define/let/write/denote) before "$X = ...$" makes even a BARE "="
# definitional and grounds the LHS symbol X — WITHOUT the lead-in, a bare
# "$X = Y$" is an assertional equation and is deliberately NOT grounded (the
# precision gate: lead-in or := required, never a bare equality). group(1) = the
# LHS bare symbol; ":?=(?!=)" accepts "=" or ":=" but not "==".
DEF_EQ_RE = re.compile(
    r"\b(?:[Ww]e\s+)?(?:[Ss]et|[Pp]ut|[Dd]efin\w+|[Ll]et|[Ww]rite|[Dd]enote)\b\s+"
    r"\$\\?([A-Za-z][A-Za-z0-9]*)\s*:?=(?!=)")
# "we call $X$ [and $Y$ ...] <name>" — a naming; bind the symbol(s) to the name.
CALL_RE = re.compile(
    r"\b[Ww]e\s+call\s+(\$[^$]+\$(?:\s+and\s+\$[^$]+\$)*)\s+"
    r"(?:an?\s+|the\s+)?([^.,;:]+?)(?=[.,;:]|$)")

# INFORMAL PROOF MOVES (Joe). Not the strategies an author *executes* (those
# are the futon3 math-informal flexiargs) but the *rhetoric of the proof*: the
# discourse gestures that assert a step while declining to carry it out — "it
# is not difficult to check", "left to the reader", "clearly". The

def _load_xref():
    """Shuttle cross-ref components: mathlib names, PlanetMath finder."""
    import json as _j
    mathlib_names = []
    mj = Path("/home/joe/code/futon6/data/mathlib-defs.json")
    if not mj.exists():
        mj = Path("/home/joe/code/futon6/data/mathlib-defs-monoidal.json")
    if mj.exists():
        mathlib_names = [d["name"] for d in _j.loads(mj.read_text())]
    pd = _ilu.spec_from_file_location("mpd", Path(__file__).resolve().parents[1] / "mine_prose_def.py")
    mpd = _ilu.module_from_spec(pd); pd.loader.exec_module(mpd)
    return mathlib_names, mpd


def _xref_fields(phrase, head, mathlib_names, mpd, ca):
    """Lean / PlanetMath / nLab cross-references + coverage verdict for a
    definiens — the three-Norn shuttle, surfaced as annotation fields."""
    import re as _re
    cam = "".join(w.capitalize() for w in _re.split(r"[\s-]+", (head or phrase).strip()))
    lean = next((n for n in mathlib_names
                 if cam.lower() in (n.lower(), n.lower().rstrip("_"), n.lower().rstrip("obj"))), None)
    pm = mpd.find_planetmath(head or phrase) if (head or phrase) else None
    pm = pm.name if pm else None
    hit = ca.resolve(head) if (ca and head) else None
    nlab = hit.get("target") if hit else None
    cov = ("fully covered" if (lean and pm) else
           "formalise: prose only" if (pm and not lean) else
           "Lean" if lean else "DEBT: undefined" if not nlab else "pointer only")
    return [["Lean", lean or "–"], ["PlanetMath", pm or "–"],
            ["nLab/NNexus", nlab or "–"], ["coverage", cov]]


def _concept_head(phrase: str) -> str:
    """Last 1-3 words of a concept phrase, math/markup stripped, for lookup."""
    words = re.findall(r"[A-Za-z][A-Za-z-]+", re.sub(r"\$[^$]*\$|[\\{}]", " ", phrase))
    return " ".join(words[-3:]) if words else ""


def detect_binders(ftext, base, ca, xref=None):
    """Emit Let-binder marks with explicit definiendum/definiens structure.
    Per binder, three marks within the blue Let scope:
      - the scope (kind let-binder),
      - the definiendum (the $symbol$; bold, term-indexed),
      - the definiens (the type phrase; underlined, same term-index).
    term-index distinguishes multiple defined terms in one Let sentence
    (Joe: graded gray, <=10/sentence)."""
    out = []
    # "Let $H$ be a Hopf algebra [and $A$ ...]" + "$M$ is a right $A$-module".
    # Each entry = the binders sharing one sentence (for term-index grading).
    sentences = []
    for m in BINDER_RE.finditer(ftext):
        grp = [(m.start(1), m.end(1), m.start(2), m.end(2), m.start(), m.end())]
        for cm in CONJUNCT_RE.finditer(ftext, m.end(), m.end() + 160):
            grp.append((cm.start(1), cm.end(1), cm.start(2), cm.end(2),
                        cm.start(), cm.end()))
        sentences.append(grp)
    let_spans = [(g[0][4], g[-1][5]) for g in sentences]
    for m in IS_RE.finditer(ftext):
        # skip if this "is a" sits inside a Let sentence already captured
        if any(s <= m.start() < e for s, e in let_spans):
            continue
        sentences.append([(m.start(1), m.end(1), m.start(2), m.end(2),
                           m.start(), m.end())])
    for binders in sentences:
        for term_i, (ds, de, ps, pe, ss, se) in enumerate(binders):
            subj = ftext[ds:de]
            phrase = ftext[ps:pe].strip()
            concept = None
            if ca is not None:
                hit = ca.resolve(_concept_head(phrase)) if phrase else None
                if hit:
                    concept = f"{hit.get('term')} [{hit.get('target')}]"
            # the blue Let scope
            out.append({
                "start": base + ss, "end": base + se,
                "layer": "dp", "kind": "let-binder",
                "tip": f"binds {subj} : {phrase[:60]}"
                       + (f" · concept: {concept}" if concept else ""),
                "fields": [["binds", subj], ["as", phrase[:70]],
                           ["canon", concept or "— (unresolved)"]],
            })
            # definiendum: the $symbol$ (bold, term-indexed)
            out.append({
                "start": base + ds, "end": base + de,
                "layer": "dp", "kind": "definiendum", "term-index": term_i,
                "tip": f"definiendum #{term_i}: {subj}",
            })
            # definiens: the type phrase (underlined, same term-index),
            # carrying the three-Norn cross-references as annotation fields
            if pe > ps:
                fields = None
                if xref is not None:
                    mathlib_names, mpd = xref
                    fields = _xref_fields(phrase, _concept_head(phrase),
                                          mathlib_names, mpd, ca)
                out.append({
                    "start": base + ps, "end": base + pe,
                    "layer": "dp", "kind": "definiens", "term-index": term_i,
                    "tip": f"definiens #{term_i}: {phrase[:60]}"
                           + (f" · {concept}" if concept else ""),
                    "fields": fields,
                })
    return out
