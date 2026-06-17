#!/usr/bin/env python3
"""Render a paper's DP classification as a paper-anatomy overlay.

Reuses anatomy_v0_sweep's own classifier so the overlay shows the EXACT
current Distributed-Proofreaders run: every control sequence inside a math
span coloured by class — recognised (classified), role-gap (recognised but
un-typed), or genuine unknown. Emits fable-<paper>-dp-emacs.json in the
golden dir, so `M-x paper-anatomy-open` on "<paper>-dp" overlays it.

    dp_paper_view.py 0809.2517
"""
from __future__ import annotations

import json
import re
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import anatomy_v0_sweep as sweep
import importlib.util as _ilu


def _load_nlab_wiring():
    """Import the superpod scope detector (filename has a hyphen)."""
    p = Path(__file__).resolve().parent / "nlab-wiring.py"
    spec = _ilu.spec_from_file_location("nlab_wiring", p)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

EPRINTS = sweep.DEFAULT_EPRINTS
GOLDEN_DIR = Path("/home/joe/code/futon6/data/showcases/ct-anatomy/golden")
from dp_capabilities.binders import (
    APPOS_CONJ_RE,
    APPOSITIVE_RE,
    BINDER_RE,
    CALL_RE,
    CONJUNCT_RE,
    DEF_EQ_RE,
    DEFINE_RES,
    IS_RE,
    QUANT_RES,
    WHERE_RES,
    _concept_head,
    _load_xref,
    detect_binders,
)
from dp_capabilities.math_envelope import (
    CSEQ_RE,
    DISPLAY_RE,
    _layout_regions,
    _nonsym_kind,
    _textmode_regions,
    display_assign_grounding,
    harvest_display_assigns,
    inside_regions,
    is_script_run,
    mathalpha_regions,
    script_base_grounding,
)
from dp_capabilities.proof_moves import detect_proof_moves
from dp_capabilities.references import _harvest_labels, detect_references
from dp_capabilities.scopes import detect_scope_manifest
from dp_capabilities.wellformed import (
    BINDER_KINDS,
    _clamp_structural_sentence_markers,
    _reconcile_structural_crossings,
    _snap_marks_to_math_atoms,
    reconcile_all_scopes,
)

# DC-3: Let–Then implications. A theorem "Let X . Then Y ." is one logical unit
# (hypothesis ⟹ conclusion) that legitimately crosses the sentence boundary — the
# layer ABOVE binding. W-SENTENCE correctly clamps the inner let-binder; this
# scope re-joins the two halves. Kind `implies` is non-structural, so it is exempt
# from the sentence/atomic clamps (it is meant to span them).
_IMPL_RE = re.compile(
    r"\b(?P<hyp>(?P<hk>Let|Given|Suppose|Assume)\b[^.]*?\.)\s+"
    r"(?P<con>(?P<ck>Then|Hence|Thus|Therefore|It follows)\b[^.]*?\.)", re.S)


def detect_implications(ftext, base):
    marks = []
    for m in _IMPL_RE.finditer(ftext):
        hyp = re.sub(r"\s+", " ", m.group("hyp")).strip()
        con = re.sub(r"\s+", " ", m.group("con")).strip()
        marks.append({
            "start": base + m.start(), "end": base + m.end(),
            "layer": "dp", "kind": "implies",
            "tip": f"implication: {hyp[:48]} ⟹ {con[:48]}",
            "fields": [["hypothesis", hyp[:90]], ["conclusion", con[:90]]],
        })
        # the hypothesis/conclusion KEYWORDS, styled by their syntax class:
        for grp, kind, role in (("hk", "kw-hyp", "hypothesis"),
                                ("ck", "kw-con", "conclusion")):
            marks.append({
                "start": base + m.start(grp), "end": base + m.end(grp),
                "layer": "dp", "kind": kind,
                "tip": f"{m.group(grp)} — {role} keyword",
            })
    return marks


# DC-9: LaTeX environments as scopes (rails + highlight). The nLab-wiki scope
# detector truncated long proofs (a pos+400 clamp) and missed custom env names
# (coro/propo). This matches \begin{NAME}…\end{NAME} directly — exact spans,
# every env name, DELIMITERS INCLUDED (Joe's call, 2026-06-15). Display-math
# envs (equation/align/…) are skipped: they are already marked as math scopes
# (DISPLAY_RE), so including them here would double-mark.
_ENV_RE = re.compile(r"\\(begin|end)\s*\{([A-Za-z]+\*?)\}")
# skipped: display-math envs (already math scopes) + `document`/wrappers whose
# extent is the whole paper (a rail around everything conveys no nesting).
_DISPLAY_MATH_ENVS = {"equation", "eqnarray", "align", "alignat", "flalign",
                      "displaymath", "gather", "multline", "math", "split",
                      "document", "abstract"}
_ENV_CANON = {
    "proof": "proof", "lemma": "lemma", "lem": "lemma",
    "theorem": "theorem", "thm": "theorem", "theo": "theorem",
    "prop": "proposition",
    "propo": "proposition", "proposition": "proposition", "cor": "corollary",
    "coro": "corollary", "corollary": "corollary", "defn": "definition",
    "def": "definition", "definition": "definition", "remark": "remark",
    "rem": "remark", "example": "example", "ex": "example", "note": "note",
    "claim": "claim", "conjecture": "conjecture", "conj": "conjecture",
    # author-defined / multilingual proof environment names (0807.1872 uses
    # \newenvironment{beweis}{…Proof…} as its proof env).
    "beweis": "proof", "pf": "proof", "demo": "proof", "preuve": "proof",
}

# Author proof-DELIMITER macro pairs used in the body in place of an env
# (\prf…\eprf in 0807.1872). Recognised as proof regions so the in-proof
# reasoning layer fires; learned-from-preamble resolution is a later refinement.
_PROOF_MACRO_PAIRS = [("prf", "eprf"), ("bpf", "epf"), ("bpr", "epr"),
                      ("beginproof", "endproof"), ("proofof", "endproof"),
                      ("proof", "qed")]
_TEXT_PROOF_START_RE = re.compile(
    r"(?<![A-Za-z])(?:\\(?:emph|textit|textbf)\s*\{\s*)?Proof\.(?:\s*\})?",
    re.I)
_TEXT_PROOF_END_RE = re.compile(
    r"\\qed\b|\\end\{(?:proof|Proof|thm|theorem|lemma|lem|prop|proposition|"
    r"cor|coro|corollary)\}|\\begin\{(?:thm|theorem|lemma|lem|prop|"
    r"proposition|cor|coro|corollary|defn|definition|remark|section)\}|"
    r"\\section\b|\\subsection\b|\\paragraph\b|□|\\Box\b")


def detect_proof_macros(text):
    bd = text.find("\\begin{document}")
    bs = bd if bd != -1 else 0
    marks = []
    for b, e in _PROOF_MACRO_PAIRS:
        for m in re.finditer(r"\\" + b + r"\b(.*?)\\" + e + r"\b", text[bs:], re.S):
            if m.end() - m.start() < 6000:
                marks.append({"start": bs + m.start(), "end": bs + m.end(),
                              "layer": "dp", "kind": "env/proof",
                              "tip": f"proof (\\{b}…\\{e})"})
    return marks


def detect_text_proofs(text):
    """Bare/text-style proof delimiters, e.g. ``Proof. ... \\qed``.

    Some journal classes use a text macro or heading instead of a LaTeX proof
    environment. Treat the resulting region as an env/proof so downstream
    reasoning marks get the deductive register.
    """
    bd = text.find("\\begin{document}")
    bs = bd if bd != -1 else 0
    marks = []
    for m in _TEXT_PROOF_START_RE.finditer(text, bs):
        tail = text[m.end():m.end() + 6000]
        end = _TEXT_PROOF_END_RE.search(tail)
        if not end:
            continue
        ee = m.end() + end.end()
        if 40 <= ee - m.start() <= 6000:
            marks.append({"start": m.start(), "end": ee,
                          "layer": "dp", "kind": "env/proof",
                          "tip": "proof (text-style Proof.)"})
    return marks


def detect_tex_environments(ftext, base):
    """\\begin{NAME}…\\end{NAME} scopes, delimiters included, nesting-safe."""
    marks, stacks = [], {}
    for m in _ENV_RE.finditer(ftext):
        kw, name = m.group(1), m.group(2)
        key = name.rstrip("*").lower()
        if key in _DISPLAY_MATH_ENVS:
            continue
        if kw == "begin":
            stacks.setdefault(name, []).append(m.start())
        elif stacks.get(name):
            s = stacks[name].pop()
            canon = "env/" + _ENV_CANON.get(key, key)
            marks.append({"start": base + s, "end": base + m.end(),
                          "layer": "dp", "kind": canon,
                          "tip": f"environment: {name}"})
    return marks


# DC-10 / IATC (first crack): the REASONING layer. Inside proofs, illative
# connectives (implies / means / follows from / consequently) anchor an inference
# between two CLAIM clauses — an (subject, relation, object) triple, IAT's RA
# edge. We mark the claims (clause-level proposition spans Joe asked for) and the
# inference (the illative), carrying the triple in fields for an inline table.
_BREAK_RE = re.compile(r"\.\s|\n\s*\n|\\begin\{[^}]*\}|\\end\{[^}]*\}")
_FILLER_RE = re.compile(
    r"^(that|the fact that|but it|but|it|and|so|then|hence|thus|therefore"
    r"|in fact|clearly|consequently|since|because"
    r"|we (?:have|get|obtain|see|know|conclude))"
    r"\b[,\s]*", re.I)
_META_RE = re.compile(r"\b(we will|we do not|one does not|moreover|necessity"
                      r"|sufficiency)\b", re.I)
# a clause that is only a deictic/filler ("the following", "this again") carries
# no proposition; and one containing a structural delimiter has crossed a
# sentence/environment boundary. Either makes a bad inference operand — reject.
_VAGUE_RE = re.compile(
    r"^(?:the following(?: result)?|this(?: again)?|that|the above|such|it"
    r"|the same|the result|the claim|here)\W*$", re.I)
_DELIM_RE = re.compile(r"\\(?:begin|end)\b|Proof\.\\|\\item\b")


def _bad_operand(s):
    return bool(_META_RE.search(s) or _VAGUE_RE.match(s) or _DELIM_RE.search(s))
_CLAUSE_SPLIT = re.compile(r",(?![^{}]*\})")
# the clause nearest the illative: break on " and " and on any inner illative
# ("X means Y and consequently Z" -> subject of consequently is "Y", not "X means Y"),
# but NOT on plain commas — the LHS premise can span commas ("following (1), δ is a
# cocone, (2)" is one premise, not just "(2)").
_LEFT_SPLIT = re.compile(
    r"\sand\s|\b(?:implies that|implies|means that|means|"
    r"follows from(?: the fact that)?)\b")
_OBJ_CUT = re.compile(r"\s+and[, ]|,\s*consequently|;\s")
_ILLATIVES = [("implies that", "implies", "split"),
              ("means that", "means", "split"),
              ("follows from the fact that", "follows from", "rev"),
              ("follows from", "follows from", "rev"),
              ("implies", "implies", "split"), ("means", "means", "split"),
              ("is equivalent to", "equivalent", "split"),
              ("equivalent to", "equivalent", "split"),
              ("if and only if", "equivalent", "split"),
              ("consequently,?", "consequently", "split")]
_PROOF_KINDS = ("env/proof", "env/theorem", "env/lemma", "env/proposition",
                "env/corollary")

# Named operators / categories: a multi-letter run that is ONE standard object,
# not juxtaposed variables. Without this, the DC-6 juxtaposition split shreds
# "Vect" -> V·e·c·t etc. We ground these to their standard meaning (so they read
# as units AND count as grounded), inverting the split for known names. Keyed by
# exact (usually capitalised) spelling to avoid catching ordinary variables.
_OPERATOR_NAMES = {
    "Vect": "category of vector spaces", "Set": "category of sets",
    "Cat": "category of (small) categories", "Grp": "category of groups",
    "Ab": "category of abelian groups", "Mod": "category of modules",
    "Top": "category of topological spaces", "Ring": "category of rings",
    "Grpd": "category of groupoids", "Sh": "category of sheaves",
    "PSh": "category of presheaves", "Fun": "functor category",
    "Func": "functor category", "Nat": "natural transformations",
    "Hom": "hom-functor / hom-set", "End": "endomorphism object",
    "Aut": "automorphism group", "Ob": "objects of a category",
    "Mor": "morphisms of a category", "Spec": "spectrum",
    "Ext": "ext groups", "Tor": "tor groups", "Sym": "symmetric algebra/group",
    "GL": "general linear group", "SL": "special linear group",
    "id": "identity morphism", "im": "image", "ker": "kernel",
    "coker": "cokernel", "colim": "colimit", "Lan": "left Kan extension",
    "Ran": "right Kan extension", "Bicat": "bicategory of …",
}


_TRAIL_RE = re.compile(r"[,\s]+(and|but it|but|so|then|that|which|is|are|in)$", re.I)


def _clean_clause(s):
    s = re.sub(r"\s+", " ", s).strip(" ,.;")
    prev = None
    while prev != s:
        prev = s
        s = _FILLER_RE.sub("", s).strip(" ,.;")
        s = _TRAIL_RE.sub("", s).strip(" ,.;")
    return s


def _last_clause(raw, base):
    """Last non-trivial comma-segment of RAW (the clause nearest the illative),
    returned as (clean_text, global_start, global_end)."""
    parts, idx = [], 0
    for mm in _LEFT_SPLIT.finditer(raw):
        parts.append((idx, mm.start()))
        idx = mm.end()
    parts.append((idx, len(raw)))
    for a, b in reversed(parts):
        c = _clean_clause(raw[a:b])
        if len(c) > 2:
            return c, base + a, base + b
    return "", base, base


def detect_inferences(text, marks):
    proofs = [(m["start"], m["end"]) for m in marks
              if str(m.get("kind", "")) in _PROOF_KINDS]
    in_proof = lambda p: any(a <= p < b for a, b in proofs)
    # Illatives fire in FREE BODY too, not only inside proof/theorem envs — a
    # paper can carry its whole argument in running prose with no \begin{proof}
    # (e.g. 1005.2653). We no longer gate on in_proof; instead we tag each
    # inference with its REGISTER — "deductive" when it sits in a formal proof
    # env, "body" otherwise — so the deductive vs motivational distinction is
    # recorded rather than used to drop the mark.
    register = lambda p: "deductive" if in_proof(p) else "body"

    def sent_bounds(p):
        lo = 0
        for b in _BREAK_RE.finditer(text[:p]):
            lo = b.end()
        m = _BREAK_RE.search(text, p)
        return lo, (m.start() if m else len(text))

    def _trim_ws(s, e):  # boundary whitespace is not contentful — exclude it
        while s < e and text[s] in " \n\t":
            s += 1
        while e > s and text[e - 1] in " \n\t":
            e -= 1
        return s, e

    def _prior_sentence(p):
        """Nearest CONTENTFUL sentence ending at/before p — skips layout-only
        spans (\\newline, \\par, \\pagebreak) that sit between real sentences,
        so a consequent premise never resolves to a bare layout macro."""
        e = p
        for _ in range(4):
            lo, _hi = sent_bounds(max(0, e - 3))
            s2, e2 = _trim_ws(lo, e)
            c = _clean_clause(text[s2:e2])
            if len(c) >= 4 and not re.fullmatch(r"\\[a-zA-Z]+\*?", c):
                return s2, e2
            if lo <= 0 or lo >= e:
                break
            e = lo
        return _trim_ws(max(0, p - 240), p)

    out, taken, seen_claims = [], [], set()
    for pat, rel, dirn in _ILLATIVES:
        for m in re.finditer(r"\b" + pat + r"\b", text):
            s, e = m.start(), m.end()
            if any(a < e and b > s for a, b in taken):
                continue
            lo, hi = sent_bounds(s)
            left, lcs, lce = _last_clause(text[lo:s], lo)
            cut = _OBJ_CUT.search(text, e, hi)
            rce = cut.start() if cut else hi
            right = _clean_clause(text[e:rce])
            if dirn == "rev":
                subj, subj_sp = right, (e, rce)
                obj, obj_sp = left, (lcs, lce)
            else:
                subj, subj_sp = left, (lcs, lce)
                obj, obj_sp = right, (e, rce)
            if len(subj) < 3 or len(obj) < 3 \
                    or _bad_operand(subj) or _bad_operand(obj):
                continue
            taken.append((s, e))
            for cs, ce in (subj_sp, obj_sp):
                cs, ce = _trim_ws(cs, ce)
                if ce > cs and (cs, ce) not in seen_claims:
                    seen_claims.add((cs, ce))
                    out.append({"start": cs, "end": ce, "layer": "dp",
                                "kind": "claim",
                                "tip": f"claim: {_clean_clause(text[cs:ce])[:70]}"})
            ie = s + len(m.group(0).rstrip(", "))  # arrow span, no trailing comma
            out.append({"start": s, "end": ie, "layer": "dp", "kind": "inference",
                        "subj_span": list(subj_sp), "obj_span": list(obj_sp),
                        "nest": 0, "register": register(s),
                        "tip": f"{subj[:42]} —{rel}→ {obj[:42]}",
                        "fields": [["subject", subj[:140]], ["relation", rel],
                                   ["object", obj[:140]], ["register", register(s)]]})
    # "following (P), C" / "by (P), C": a small embedded inference — by premise
    # reference P, conclusion C (L185 "following (1), δ is a colimit cocone").
    # The reference rides with the arrow; it nests inside any larger inference
    # whose operand contains it.
    for m in re.finditer(
            r"\b(?:following|by)\s+(\([^)]{1,24}\)|\\cite\{[^}]{1,40}\})\s*,\s+",
            text):
        oc = re.match(r"[^.,;]{4,140}", text[m.end():])
        if not oc:
            continue
        a_s, a_e = m.start(), m.end(1)             # arrow phrase "following (1)"
        o_s, o_e = _trim_ws(m.end(), m.end() + oc.end())
        obj = _clean_clause(text[o_s:o_e])
        if len(obj) < 4 or _bad_operand(obj):
            continue
        if (o_s, o_e) not in seen_claims:
            seen_claims.add((o_s, o_e))
            out.append({"start": o_s, "end": o_e, "layer": "dp", "kind": "claim",
                        "tip": f"claim: {obj[:70]}"})
        out.append({"start": a_s, "end": a_e, "layer": "dp", "kind": "inference",
                    "subj_span": [a_s, a_e], "obj_span": [o_s, o_e],  # ref rides
                    "nest": 0, "register": register(a_s),             # in the arrow
                    "tip": f"by {text[m.start(1):m.end(1)]} ⟹ {obj[:42]}",
                    "fields": [["subject", text[m.start(1):m.end(1)]],
                               ["relation", "by"], ["object", obj[:140]],
                               ["register", register(a_s)]]})

    # Consequent markers ("X. Thus Y." / "… . Hence Y."): the conclusion is the
    # clause after the marker; the premise is the PRIOR sentence — this builds
    # the proof chain. Sentence-initial only (capitalised), inside proofs.
    for m in re.finditer(
            r"(?:(?<=\.\s)|(?<=\.\n)|(?<=\}\n)|(?<=\n\n))(Thus|Hence|Therefore|So"
            r"|Whence|Accordingly|Consequently)\b\s*,?\s+", text):
        ms, me = m.start(1), m.end()
        if any(a < me and b > ms for a, b in taken):
            continue
        # conclusion = the marker's own sentence; premise = the PRIOR sentence.
        # sent_bounds' _BREAK_RE handles ".\n" and blank lines, so newline-
        # separated prose (1005.2653) segments correctly.
        _, co_e = sent_bounds(me)
        co_s, co_e = _trim_ws(me, co_e)
        pr_s, pr_e = _prior_sentence(ms)
        subj, obj = _clean_clause(text[pr_s:pr_e]), _clean_clause(text[co_s:co_e])
        if len(subj) < 4 or len(obj) < 4 \
                or _bad_operand(subj) or _bad_operand(obj):
            continue
        taken.append((ms, me))
        for cs, ce2 in ((pr_s, pr_e), (co_s, co_e)):
            if ce2 > cs and (cs, ce2) not in seen_claims:
                seen_claims.add((cs, ce2))
                out.append({"start": cs, "end": ce2, "layer": "dp", "kind": "claim",
                            "tip": f"claim: {_clean_clause(text[cs:ce2])[:70]}"})
        out.append({"start": ms, "end": m.end(1), "layer": "dp", "kind": "inference",
                    "subj_span": [pr_s, pr_e], "obj_span": [co_s, co_e], "nest": 0,
                    "register": register(ms),
                    "tip": f"{subj[:40]} —{m.group(1).lower()}→ {obj[:40]}",
                    "fields": [["subject", subj[:140]],
                               ["relation", m.group(1).lower()], ["object", obj[:140]],
                               ["register", register(ms)]]})

    # If–Then implications ("If X, then Y"): premise X, conclusion Y; arrow = then.
    for m in re.finditer(r"\bIf\b\s+([^.]{3,180}?)\s*,?\s+then\b\s+", text):
        ms, ae = m.start(), m.end()
        if any(a < ae and b > ms for a, b in taken):
            continue
        tp = text.rfind("then", ms, ae)
        su_s, su_e = _trim_ws(ms, tp)
        ce = text.find(". ", ae)
        ob_s, ob_e = _trim_ws(ae, ce if ce != -1 else min(len(text), ae + 200))
        subj, obj = _clean_clause(text[su_s:su_e]), _clean_clause(text[ob_s:ob_e])
        if len(subj) < 4 or len(obj) < 4 \
                or _bad_operand(subj) or _bad_operand(obj):
            continue
        taken.append((ms, ae))
        for cs, ce2 in ((su_s, su_e), (ob_s, ob_e)):
            if ce2 > cs and (cs, ce2) not in seen_claims:
                seen_claims.add((cs, ce2))
                out.append({"start": cs, "end": ce2, "layer": "dp", "kind": "claim",
                            "tip": f"claim: {_clean_clause(text[cs:ce2])[:70]}"})
        out.append({"start": tp, "end": tp + 4, "layer": "dp", "kind": "inference",
                    "subj_span": [su_s, su_e], "obj_span": [ob_s, ob_e], "nest": 0,
                    "register": register(ms),
                    "tip": f"if {subj[:38]} —then→ {obj[:38]}",
                    "fields": [["subject", subj[:140]], ["relation", "then"],
                               ["object", obj[:140]], ["register", register(ms)]]})

    # IATC nesting (subgraph-in-a-statement-slot). Two ways an inference B nests
    # one level inside A: CONTAINMENT — B's arrow sits inside A's subject/object
    # operand ("following (1)" inside the implies premise); or CHAIN — B's
    # premise IS A's conclusion ("X means [Y and, consequently, Z]"). Iterate to
    # a fixpoint so depth propagates through stacked nesting.
    # Balance parentheses: a clause/claim cut mid-parenthetical — e.g.
    # "we obtain a functor (i.e. antipode)" sliced at "i.e." because its period
    # reads as a sentence end — gets extended to its closing ")" so the scope
    # spans the whole parenthetical (Joe: "not extending through the parenthetical").
    def _bal(s, e):
        depth = text.count("(", s, e) - text.count(")", s, e)
        j = e
        while depth > 0 and j < len(text) and j < e + 90:
            if text[j] == "(":
                depth += 1
            elif text[j] == ")":
                depth -= 1
            j += 1
        return j if depth == 0 else e
    for m in out:
        if m["kind"] == "claim":
            m["end"] = _bal(m["start"], m["end"])
        elif m["kind"] == "inference":
            for key in ("subj_span", "obj_span"):
                sp = m.get(key)
                if sp:
                    sp[1] = _bal(sp[0], sp[1])

    infs = sorted((m for m in out if m["kind"] == "inference"),
                  key=lambda m: m["start"])

    def _arrow_in(outer, x):
        mid = (x["start"] + x["end"]) // 2
        return any(sp[0] <= mid < sp[1]
                   for sp in (outer["subj_span"], outer["obj_span"]))

    for _ in range(4):
        for B in infs:
            for A in infs:
                if A is B:
                    continue
                contained = _arrow_in(A, B)
                chain = (not _arrow_in(B, A) and A["start"] < B["start"]
                         and A["obj_span"][0] < B["subj_span"][1]
                         and B["subj_span"][0] < A["obj_span"][1])
                if contained or chain:
                    B["nest"] = max(B["nest"], A["nest"] + 1)
    return out


# Enumerate-item anaphora: \item[(1)]/\item[(2)] BIND the labels (1)/(2) to
# their content; later "(1)"/"(2)" in prose are short-range anaphoric references
# back to them — the IATC reference layer. We mark the antecedents and resolve
# each anaphor to its bound item (surfaced on hover).
_ITEM_RE = re.compile(r"\\item\s*\[\s*(\([^\]\n]{1,12}\))\s*\]")


def detect_enumerate_anaphora(text):
    marks, antecedents, item_spans = [], [], []
    for em in re.finditer(r"\\begin\{enumerate\}(.*?)\\end\{enumerate\}", text, re.S):
        base = em.start(1)
        its = list(_ITEM_RE.finditer(em.group(1)))
        for k, im in enumerate(its):
            label = im.group(1)
            ce = its[k + 1].start() if k + 1 < len(its) else len(em.group(1))
            content = re.sub(r"\s+", " ", em.group(1)[im.end():ce]).strip(" ,.;")
            ls, le = base + im.start(1), base + im.end(1)
            antecedents.append((ls, label, content))
            item_spans.append((ls, le))
            marks.append({"start": ls, "end": le, "layer": "dp", "kind": "label",
                          "tip": f"binds {label} — item: {content[:100]}"})
    # resolve each "(N)" anaphor to its NEAREST PRECEDING same-label item (short
    # range); skip the antecedent labels themselves.
    for am in re.finditer(r"\(\d{1,3}\)", text):
        gs, ge, label = am.start(), am.end(), am.group(0)
        if any(s <= gs < e for s, e in item_spans):
            continue
        best = None
        for ls, lab, content in antecedents:
            if lab == label and ls < gs:
                best = (ls, content)
        if best and gs - best[0] < 2500:
            marks.append({"start": gs, "end": ge, "layer": "dp", "kind": "anaphor",
                          "tip": f"↑ {label}: {best[1][:110]}"})
    return marks


def build(paper: str, with_ca: bool = False, with_binders: bool = False,
          with_scopes: bool = False, with_xref: bool = False) -> dict:
    ca = None
    if with_ca or with_binders or with_scopes or with_xref:
        from concept_authority import ConceptAuthority
        ca = ConceptAuthority()
    nw = _load_nlab_wiring() if with_scopes else None
    xref = _load_xref() if with_xref else None
    eprint = None
    for suffix in (".tar.gz", ".tex.gz", ".gz", ".tar", ".tex"):
        cand = EPRINTS / f"{paper}{suffix}"
        if cand.exists():
            eprint = cand
            break
    if eprint is None:
        raise SystemExit(f"no eprint for {paper} under {EPRINTS}")

    files, _meta = sweep.read_eprint_files(eprint)
    roles = sweep.load_latexml_roles(sweep.ROLE_TSV)
    plain = sweep.load_plain_cseq(sweep.PLAIN_CSEQ)
    macros = sweep.collect_macros(files, roles)

    # Concatenate .tex files into one display text; track each file's base.
    tex_files = [f for f in files if f["file"].endswith(".tex")] or files
    parts, bases, cursor = [], {}, 0
    for f in tex_files:
        header = f"% ===== {f['file']} =====\n"
        parts.append(header)
        cursor += len(header)
        bases[f["file"]] = cursor
        parts.append(f["text"])
        cursor += len(f["text"]) + 1
        parts.append("\n")
    text = "".join(parts)
    # R6 (claude-3): bare symbols display-defined by "X := ..." ground to that
    # definition. Harvested once over the whole text (global offsets), consumed
    # as a fallback at the ground() seam below.
    assign_defs = harvest_display_assigns(text)

    marks, counts = [], {"classified": 0, "role-gap": 0, "unknown": 0,
                         "concept-typed": 0, "let-binder": 0}
    # PRE-PASS (Joe's grounding point): the "Let $H$ be a Hopf algebra"
    # stanza grounds the symbol H. Collect symbol → (global-pos, concept)
    # bindings so each later symbol occurrence resolves to its binder (R4,
    # the use→binder edge) instead of leaving an ungrounded "symbol" debt.
    bindings = {}  # bare symbol -> sorted [(global_pos, label)]

    def _add_binding(sym_text, label, gpos):
        s = re.search(r"[A-Za-z]+", sym_text or "")
        if not s or not label:
            return
        bindings.setdefault(s.group(0), []).append((gpos, label.strip()[:50]))

    if with_binders or with_scopes:
        for f in tex_files:
            base = bases[f["file"]]
            # (1) Let-binders + "$M$ is a right $A$-module" (symbol+phrase)
            pairs = []
            for m in BINDER_RE.finditer(f["text"]):
                pairs.append((m.group(1), m.group(2), m.start()))
                for cm in CONJUNCT_RE.finditer(f["text"], m.end(), m.end() + 160):
                    pairs.append((cm.group(1), cm.group(2), cm.start()))
            for m in IS_RE.finditer(f["text"]):
                pairs.append((m.group(1), m.group(2), m.start()))
            # defined-in-paper: definition-prose definienda (define/denote/write)
            for rx in DEFINE_RES:
                for m in rx.finditer(f["text"]):
                    pairs.append((m.group(1), m.group(2), m.start()))
            # quantifier + where/with binding (Joe's QUANTIFIER/POSITIONAL tail)
            for rx, qlabel in QUANT_RES:
                for m in rx.finditer(f["text"]):
                    pairs.append((m.group(1), qlabel, m.start()))
            for rx in WHERE_RES:
                for m in rx.finditer(f["text"]):
                    pairs.append((m.group(1), m.group(2), m.start()))
            # def-equation: a lead-in verb (set/put/let/...) + "$X = ...$" makes
            # a bare = definitional -> ground X (bare assertional = is rejected).
            for m in DEF_EQ_RE.finditer(f["text"]):
                pairs.append((m.group(1), "definitional equation", m.start()))
            # naming: "we call $X$ [and $Y$] <name>" -> bind each symbol to name.
            for m in CALL_RE.finditer(f["text"]):
                for sm in re.finditer(r"\$[^$]+\$", m.group(1)):
                    pairs.append((sm.group(0), m.group(2), m.start()))
            # appositive typing: "<det> <type-noun-phrase> $X$" (a Hopf algebra
            # $H$) — the type-THEN-symbol direction the binders above miss; the
            # biggest remaining grounding lever (~78% of the ungrounded tail).
            for m in APPOSITIVE_RE.finditer(f["text"]):
                pairs.append((m.group(2), m.group(1), m.start()))
                # plural type-noun ("objects $K$ and $L$") -> the contiguous
                # ", "/" and " conjunct chain shares the type; singular does not.
                if m.group(1).rstrip().endswith("s"):
                    pos = m.end()
                    while True:
                        cj = APPOS_CONJ_RE.match(f["text"], pos)
                        if not cj:
                            break
                        pairs.append((cj.group(1), m.group(1), cj.start(1)))
                        pos = cj.end()
            for subj, phrase, pos in pairs:
                label = phrase
                if ca is not None:
                    hit = ca.resolve(_concept_head(phrase))
                    if hit:
                        label = hit.get("term")
                _add_binding(subj, label, base + pos)
            # (2) the FULL scope manifest: every binding-like scope grounds its
            # symbol — bind/typed, assume ("If $M$ is a right $A$-module"),
            # quant, where-binding — not just Let (Joe: the 4029 ungrounded are
            # mostly bound by non-Let forms the grounding never consulted).
            if nw is not None:
                for s in nw.detect_scopes(f"arxiv-{paper}", f["text"]):
                    pos = base + (s.get("hx/content", {}).get("position") or 0)
                    ends = s.get("hx/ends", [])
                    sym = next((e.get("latex") or e.get("text") for e in ends
                                if e.get("role") in ("symbol", "condition")), None)
                    typ = next((e.get("text") or e.get("latex") for e in ends
                                if e.get("role") in ("type", "relation", "value")), None)
                    if sym and not typ:
                        # assume/if: the scope match is often just "If $M$" —
                        # pull the type from the FILE TEXT at the scope position,
                        # not the truncated match.
                        raw = s.get("hx/content", {}).get("position") or 0
                        mm = re.search(r"\bis\s+(?:a|an|the)\s+([^.,;]+)",
                                       f["text"][raw:raw + 120])
                        typ = mm.group(1) if mm else None
                    if sym and typ:
                        _add_binding(sym, typ, pos)
        for k in bindings:
            bindings[k].sort()

    def ground(sym, g):
        """Latest binding of SYM at-or-before global offset G (its scope)."""
        cand = [(p, lab) for p, lab in bindings.get(sym, []) if p <= g]
        return cand[-1][1] if cand else None

    # REFERENCE GRAPH pre-pass: collect every \label key across the WHOLE paper
    # first, so a \ref to a forward label (later section/file) still resolves.
    label_keys = set()
    for f in tex_files:
        for _s, _e, key in _harvest_labels(f["text"]):
            label_keys.add(key)

    for f in tex_files:
        base = bases[f["file"]]
        ftext = f["text"]
        bm = []
        sm = []
        if with_binders:
            bm = detect_binders(ftext, base, ca, xref=xref)
        if with_scopes:
            sm = detect_scope_manifest(ftext, base, f"arxiv-{paper}", nw, ca)
            # DC-9: drop the nLab env scopes (truncated/clamped, missed custom
            # env names); the LaTeX-env detector below is the env source instead.
            sm = [m for m in sm if not str(m.get("kind", "")).startswith("env/")]
        if bm or sm:
            # SNAP (Joe): math spans are atomic. Generalize the old
            # scope-manifest-only pass to binder marks too, then reconcile the
            # two structural layers so manifest-vs-binder scopes never cross.
            spans = [(base + s, base + e) for s, e, _d, _b in sweep.math_spans(ftext)]
            spans += [(base + dm.start(), base + dm.end()) for dm in DISPLAY_RE.finditer(ftext)]
            structural = _snap_marks_to_math_atoms(bm + sm, spans)
            structural = _clamp_structural_sentence_markers(structural, text)
            structural = _snap_marks_to_math_atoms(structural, spans)
            structural = _reconcile_structural_crossings(structural)
            counts["let-binder"] += sum(1 for m in structural
                                         if m.get("layer") == "dp"
                                         and m.get("kind") in BINDER_KINDS)
            counts["scope"] = counts.get("scope", 0) + sum(
                1 for m in structural if m.get("layer") == "scope")
            marks.extend(structural)
        # DISPLAY EQUATIONS (Joe): \begin{eqnarray}/\[...\] are math scopes
        # too — math_spans only sees $-delims. Don't parse the GrCalc layout
        # this pass; DO mark the whole thing a display-math scope and surface
        # the variables it relates (+ := => a definition). "Going inside" the
        # mess: at minimum, this is a display equation relating these symbols.
        for dm in DISPLAY_RE.finditer(ftext):
            envname = dm.group(1) or "\\[ \\]"
            dbody = dm.group(2) or dm.group(3) or ""
            counts["display"] = counts.get("display", 0) + 1
            # PRINCIPLE (Joe): a display environment IS a math scope — by virtue
            # of being a math environment, independently of whether we can read
            # its display semantics. Same status as $...$ (R1), so it's emitted
            # UNCONDITIONALLY. The definition/relating-variables below is
            # ENRICHMENT that rides on the math scope; it does not constitute it.
            dvars = sorted({s for s in re.findall(r"(?<!\\)(?<![A-Za-z])[A-Za-z]", dbody)
                            if s not in "et"})[:8]  # the cell variables
            is_def = ":=" in dbody or "\\stackrel{def}" in dbody
            tip = f"math scope · display ({envname})"
            if is_def:
                tip += " · DEFINITION (:=)"
            if dvars:
                tip += f" · relates {', '.join(dvars)}"
            marks.append({
                "start": base + dm.start(), "end": base + dm.end(),
                "layer": "dp", "kind": "math",
                "tip": tip,
                "fields": [["scope", f"display math ({envname})"],
                           ["semantics", ("definition (:=)" if is_def else "equation")
                                         if (dvars or is_def) else "unread (math scope only)"],
                           ["relates", ", ".join(dvars) or "—"]],
            })
            # mark + ground the variable symbols inside the display
            tm_regions = _textmode_regions(dbody)
            lay_regions = _layout_regions(dbody)
            for sm in re.finditer(r"(?<!\\)(?<![A-Za-z])[A-Za-z][A-Za-z0-9]*", dbody):
                sym = sm.group(0)
                if sym in ("got", "gcl", "gbeg", "gend", "gnl", "gvac", "gob",
                           "grm", "gmu", "gbr", "gcn", "grcm", "gcmu", "hspace",
                           "scalebox", "ot", "label"):
                    continue  # GrCalc / layout macro names, not variables
                g = base + dm.start() + (dm.start(2) - dm.start() if dm.group(2) else
                                         dm.start(3) - dm.start()) + sm.start()
                nonsym = _nonsym_kind(dbody, sm.start(), sym, tm_regions, lay_regions)
                if nonsym is not None:
                    counts[nonsym] = counts.get(nonsym, 0) + 1
                    marks.append({
                        "start": g, "end": g + len(sym), "layer": "dp",
                        "kind": nonsym,
                        "tip": f"{nonsym}: {sym} (non-math token, excluded)"})
                    continue
                bound = ground(sym, g)
                k = "symbol-grounded" if bound else "symbol"
                counts[k] = counts.get(k, 0) + 1
                mk = {"start": g, "end": g + len(sym), "layer": "dp", "kind": k,
                      "tip": (f"{sym} : {bound}" if bound else f"symbol {sym} (in display)")}
                if bound:
                    mk["fields"] = [["symbol", sym], ["bound", bound]]
                marks.append(mk)
        # INFORMAL PROOF MOVES (Joe): the proof's discourse gestures — "it is
        # not difficult to check" and kin — asserting a step while declining to
        # carry it out. A layer distinct from the structural scopes.
        for pmark in detect_proof_moves(ftext, base):
            counts["proof-move"] = counts.get("proof-move", 0) + 1
            marks.append(pmark)
        # DC-3: Let–Then implication scopes (hypothesis ⟹ conclusion).
        for imark in detect_implications(ftext, base):
            counts["implies"] = counts.get("implies", 0) + 1
            marks.append(imark)
        # DC-9: LaTeX environment scopes (\begin..\end, delimiters included).
        for emark in detect_tex_environments(ftext, base):
            counts[emark["kind"]] = counts.get(emark["kind"], 0) + 1
            marks.append(emark)
        # REFERENCE GRAPH (Joe): \label/\ref/\cite harvest — the in-paper
        # reference graph + outbound citations. Unconditional, like proof-moves;
        # lives in prose, so the coverage/well-formedness invariants never see it.
        for rmark in detect_references(ftext, base, label_keys):
            counts[rmark["kind"]] = counts.get(rmark["kind"], 0) + 1
            if rmark["kind"] == "ref" and any(
                    k == "target" and v == "dangling" for k, v in rmark["fields"]):
                counts["ref-dangling"] = counts.get("ref-dangling", 0) + 1
            marks.append(rmark)
        for start, end, delim, body in sweep.math_spans(ftext):
            body_off = start + len(delim)
            # RULE (Joe): anything between dollar signs IS a math scope —
            # never null. Even a bare $H$ (no control sequence) gets an
            # envelope; LaTeX itself is telling us it's mathematics.
            counts["math"] = counts.get("math", 0) + 1
            marks.append({
                "start": base + start, "end": base + end,
                "layer": "dp", "kind": "math",
                "tip": f"math: {(body.strip() or 'empty math span')[:60]}",
                "fields": [["math", (body.strip() or "— empty span —")[:70]],
                           ["delim", "display ($$)" if delim == "$$" else "inline ($)"]],
            })
            # SATIETY (Joe): the $...$ scope is hungry for content — annotate
            # the symbols inside, not just control sequences. A bare $H$ whose
            # H is unmarked is a hungry envelope (a violation). Mark each
            # symbol (letter/identifier run not part of a \cseq name).
            tm_regions = _textmode_regions(body)
            lay_regions = _layout_regions(body)
            ma_regions = mathalpha_regions(body)
            for sm in re.finditer(r"[A-Za-z][A-Za-z0-9]*", body):
                if sm.start() > 0 and body[sm.start() - 1] == "\\":
                    continue  # it's a control-sequence name, handled below
                sym = sm.group(0)
                g = base + body_off + sm.start()
                # NON-MATH token (length unit / env-name / text-mode prose)? Tag
                # it so the checker excludes it from the symbol denominator —
                # never could be grounded, so it is not a symbol (claude-3).
                nonsym = _nonsym_kind(body, sm.start(), sym, tm_regions, lay_regions)
                if nonsym is not None:
                    counts[nonsym] = counts.get(nonsym, 0) + 1
                    marks.append({
                        "start": g, "end": g + len(sym), "layer": "dp",
                        "kind": nonsym,
                        "tip": f"{nonsym}: {sym} (non-math token, excluded)"})
                    continue
                bound = ground(sym, g)
                if not bound:
                    # sub/superscript of a grounded base grounds to it (claude-4)
                    bound = script_base_grounding(body, sm.start(), ground,
                                                  base + body_off)
                if not bound:
                    # symbol display-defined by "X := ..." grounds to it (R6, claude-3)
                    bound = display_assign_grounding(sym, g, assign_defs)
                if not bound and sym in _OPERATOR_NAMES:
                    # named operator / category (Vect, Set, Hom, …): ONE standard
                    # object, not juxtaposed variables — ground to its standard
                    # meaning so it stays whole (skips the DC-6 split below) and
                    # counts as grounded. Fixes "Vect" -> V·e·c·t shredding.
                    bound = _OPERATOR_NAMES[sym]
                # DC-6: a BARE italic multi-letter run is a JUXTAPOSITION (TeX
                # sets "gf" as g·f, "QR" as Q·R), not one identifier. Split into
                # single-letter symbols so each grounds on its own — UNLESS it is
                # an operator name (\mathrm{Hom}, ma_regions) or a script modifier
                # ("op" in A^{op}). Digit-bearing runs (x0) are left intact.
                # CONSERVATIVE: only split an UNGROUNDED whole — splitting a run
                # already grounded as a unit could drop grounding (runbook bar:
                # grounded must not fall). Grounded units stay, but the checker
                # still flags them W-SYM-JUXTAPOSITION (surfaced, not hidden).
                # (DP_NO_JUXT_SPLIT=1 disables it — for the controlled A/B only.)
                if (not bound and os.environ.get("DP_NO_JUXT_SPLIT") != "1"
                        and sym.isalpha() and len(sym) > 1
                        and not inside_regions(sm.start(), sm.end(), ma_regions)
                        and not is_script_run(body, sm.start())):
                    for k, ch in enumerate(sym):
                        gk = g + k
                        b = ground(ch, gk) or display_assign_grounding(
                            ch, gk, assign_defs)
                        kk = "symbol-grounded" if b else "symbol"
                        counts[kk] = counts.get(kk, 0) + 1
                        mk = {"start": gk, "end": gk + 1, "layer": "dp", "kind": kk,
                              "tip": (f"{ch} : {b}" if b else f"symbol {ch} "
                                      f"(ungrounded)") + f"  ·  split from {sym!r}",
                              "fields": [["symbol", ch], ["juxtaposition", sym]]
                              + ([["bound", b]] if b else [])}
                        marks.append(mk)
                    continue
                kind = "symbol-grounded" if bound else "symbol"
                counts[kind] = counts.get(kind, 0) + 1
                mark = {
                    "start": g, "end": g + len(sym),
                    "layer": "dp", "kind": kind,
                    "tip": (f"{sym} : {bound}" if bound else f"symbol {sym} (ungrounded)"),
                }
                if bound:
                    mark["fields"] = [["symbol", sym], ["bound", bound]]
                marks.append(mark)
            for m in CSEQ_RE.finditer(body):
                cs = m.group(1) or m.group(2)
                cls = sweep.classify_cseq(cs, macros, roles, plain)
                concept = None
                if cls["class"] == "UNKNOWN":
                    kind = "unknown"
                elif cls["role"] == "UNKNOWN":
                    kind = "role-gap"
                    # concept-typing fold: resolve the role-gap against the
                    # authority. Guard single-char surfaces (\C->"c" junk).
                    if ca is not None and len(cs.lstrip("\\")) >= 2:
                        hit = ca.resolve(cs)
                        if hit:
                            kind = "concept-typed"
                            concept = f"{hit.get('term')} [{hit.get('target')}]"
                else:
                    kind = "classified"
                counts[kind] += 1
                g = base + body_off + m.start()
                tip = (f"\\{cs} · {cls['class']} · {cls['role']}"
                       + (f" · {cls.get('source','')}" if cls.get("source") else ""))
                if concept:
                    tip += f" · concept: {concept}"
                marks.append({
                    "start": g,
                    "end": g + (m.end() - m.start()),
                    "layer": "dp",
                    "kind": kind,
                    "tip": tip,
                })
    # DC-1 (2026-06-15): persist the PROSE-CONCEPT layer (named math terms in
    # running text) so it ships in the mined JSON and the checker can enforce
    # C-TERM-COVERAGE corpus-wide. Detector half; defensive — a fault here must
    # never break the mine, so it degrades to "no concept marks", not a crash.
    # Author proof-delimiter macros (\prf…\eprf) → proof regions, BEFORE the
    # inference pass (which is proof-restricted).
    for pm in detect_proof_macros(text):
        counts["env/proof"] = counts.get("env/proof", 0) + 1
        marks.append(pm)
    for pm in detect_text_proofs(text):
        counts["env/proof"] = counts.get("env/proof", 0) + 1
        marks.append(pm)
    # IATC reference layer: enumerate-item bindings + short-range anaphors.
    for am in detect_enumerate_anaphora(text):
        counts[am["kind"]] = counts.get(am["kind"], 0) + 1
        marks.append(am)
    # DC-10 / IATC: reasoning triples + claim spans (after env marks exist).
    for im in detect_inferences(text, marks):
        counts[im["kind"]] = counts.get(im["kind"], 0) + 1
        marks.append(im)

    try:
        import dp_enrich
        cmarks = dp_enrich.concept_marks(text, marks)
        for cm in cmarks:
            counts["concept"] = counts.get("concept", 0) + 1
        marks.extend(cmarks)
    except Exception as exc:
        print(f"  concept layer skipped ({exc})", file=sys.stderr)

    # EXPOSITORY SCOPES (claude-3's expository_region_extract): the prose regions
    # — leaf-section / inflight — where the paper explains, motivates, or
    # connects, the layer BENEATH the IATC reasoning. Emitted as `exposition`
    # scope marks. Unclassified here; the minted-kind labels (connection/…,
    # rationale/…) ride the gh200 agent votes and are attached at the 200-run
    # wiring step. Scaffold-less papers (no \section) are covered via the
    # whole-body fallback.
    try:
        import expository_region_extract as expo
        for r in expo.extract_regions(paper, text).get("regions", []):
            counts["exposition"] = counts.get("exposition", 0) + 1
            marks.append({"start": r["char_start"], "end": r["char_end"],
                          "layer": "dp", "kind": "exposition",
                          "tip": f"expository · {r['type']}",
                          "fields": [["region", r["type"]],
                                     ["section", r.get("section_title", "")]]})
    except Exception as exc:
        print(f"  exposition layer skipped ({exc})", file=sys.stderr)

    # PREAMBLE is definitions, not content: everything before \begin{document}
    # is \newcommand/\def/\newenvironment/\newtheorem etc. The env detector was
    # matching the \begin{...} inside \newenvironment{...}{...} DEFINITIONS as
    # environment uses, and the prose detectors were tagging macro bodies as
    # content. Drop every mark that starts in the preamble. (Corpus-wide: a big
    # preamble, e.g. 0807.1872 at 48% of the file, otherwise floods the marks.)
    doc_start = text.find("\\begin{document}")
    if doc_start != -1:
        marks = [m for m in marks if m.get("start", 0) >= doc_start]

    # FINAL nesting pass: make ALL extent scopes (env / manifest / binder /
    # implies / claim) nest cleanly or be disjoint — no environment×scope or
    # claim×scope crossings reach the output (Joe's lint gate).
    marks = reconcile_all_scopes(marks)

    return {"paper": f"{paper}-dp", "text": text, "marks": marks, "_counts": counts}


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("usage: dp_paper_view.py <paper-id>")
        return 2
    paper = argv[0]
    with_ca = "--with-concept-authority" in argv[1:]
    with_binders = "--with-binders" in argv[1:]
    with_scopes = "--with-scopes" in argv[1:]
    with_xref = "--with-xref" in argv[1:]
    data = build(paper, with_ca=with_ca, with_binders=with_binders,
                 with_scopes=with_scopes, with_xref=with_xref)
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    out = GOLDEN_DIR / f"fable-{paper}-dp-emacs.json"
    out.write_text(json.dumps({k: v for k, v in data.items() if k != "_counts"}))
    c = data["_counts"]
    tot = sum(c.values()) or 1
    print(f"{paper}: {len(data['marks'])} marks — "
          f"classified {c['classified']} ({100*c['classified']//tot}%), "
          f"role-gap {c['role-gap']} ({100*c['role-gap']//tot}%), "
          f"concept-typed {c['concept-typed']} ({100*c['concept-typed']//tot}%), "
          f"unknown {c['unknown']} ({100*c['unknown']//tot}%), "
          f"let-binders {c['let-binder']}, scopes {c.get('scope', 0)}")
    print(f"  reference graph: {c.get('label', 0)} labels, "
          f"{c.get('ref', 0)} refs ({c.get('ref-dangling', 0)} dangling), "
          f"{c.get('cite', 0)} cites")
    print(f"wrote {out}  →  M-x paper-anatomy-open  RET  {paper}-dp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
