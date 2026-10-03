#!/usr/bin/env python3
"""Standalone invariant checker over DP paper-anatomy markup — the convergence
engine for non-interactive structure mining (Joe, 2026-06-13).

Detectors PROPOSE structure (dp_paper_view); this checker DISPOSES — it emits a
typed violation list against the markup invariants. The two are adversaries:
COVERAGE invariants push tagging UP (every symbol, every $-span, every entity);
WELL-FORMEDNESS invariants punish sloppy tagging (atomic math, proper nesting,
no straddle). You can't satisfy coverage by spraying tags (it trips
well-formedness) or well-formedness by tagging nothing (it trips coverage); passing both is evidence for these checks, not a proof of semantic correctness. Author != reviewer even in
code: the checker NEVER imports the detector — it reads only the emitted JSON.

Violation schema — the unit the dispatch pool consumes:
    {inv, severity, start, end, msg, fix}
    severity: "error" (well-formedness; must fix) | "debt" (coverage gap)
    fix:      "tighten-detector" | "extend-coverage" | "irreducible-debt"

A violation with fix=tighten-detector/extend-coverage is dispatchable work
(route to a Codex agent). fix=irreducible-debt is honest residue, NOT a bug —
a symbol used but undefined anywhere is a real definition hole (the shuttle's
Skuld/DEBT cell, a prose `sorry`); it is recorded, not dispatched.

    check_invariants.py <paper-id>     # one paper  -> data/loss/<paper>.json
    check_invariants.py --corpus       # aggregate   -> data/loss/dashboard.json
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[0]))
import futon6_config as config


from bisect import bisect_left, bisect_right
from functools import partial
import json
import re
import sys
from pathlib import Path

from dp_capabilities.math_envelope import is_script_run, mathalpha_regions  # SHARED
from dp_capabilities.wellformed import scope_crossings  # SHARED nesting lint
from dp_capabilities.intervals import Extents, Stabbing  # SHARED interval queries (no detector logic)
# "where math-alphabet groups / scripts are" locators (DC-6) — same status as the
# span tokenizer: agreeing where things are is not an author≠reviewer breach.
import paper_pool
import anatomy_v0_sweep as sweep  # SHARED math-span tokenizer (delimiter parity,
# \$ / $$ handling) — the same ground-truth the detector uses for "where math
# is". Agreeing on the span tokenizer is not an author≠reviewer breach (it's
# agreeing what a "line" is); the invariant LOGIC below stays independent.

ROOT = config.ROOT
GOLDEN_DIR = config.marks()
LOSS_DIR = config.path("FUTON6_LOSS", ROOT / "data" / "loss")

# marks whose extent is a structural scope (must not straddle math, must nest):
STRUCTURAL_SCOPE = {"let-binder"}          # dp layer
STRUCTURAL_ENV_PREFIXES = ("env/",)        # legitimately multi-sentence
SYMBOL_KINDS = {"symbol", "symbol-grounded", "classified", "concept-typed",
                "role-gap", "unknown"}      # any tag that "covers" a letter-run
# Detector layout/text-mode labels are proposals, never denominator authority.
NON_SYMBOL_KINDS = {"layout", "text-mode"}
MATH_KINDS = {"math"}
LETTER_RUN = re.compile(r"(?<!\\)(?<![A-Za-z])[A-Za-z][A-Za-z0-9]*")

# C-TERM-COVERAGE locator. The checker's INDEPENDENT ground truth for "where a
# named term is" = AUTHOR EMPHASIS (\textit/\emph/\textbf/...). The detector
# (dp_paper_view's concept layer) keys on concept-endings + definition verbs,
# NOT on emphasis — so emphasised phrases are a test set the detector did not
# get to define, and coverage over them is a real measurement, not self-grading.
TERM_NOTICE_KINDS = {"concept", "definiendum", "definiens"}
EMPH_RE = re.compile(r"\\(?:emph|textit|textbf|textsl|textsc|dfn)\s*\{([^{}]*)\}")


def _math_spans(text):
    """(start, end) of every math span — the atomic math regions — via the
    SHARED tokenizer (proper $ parity / \\$ / $$), NOT a naive $[^$]+$ regex
    (which mis-pairs a closing $ with the next opening $ and swallows the prose
    between two spans). math_spans yields (start, end, delim, body)."""
    return [(s, e) for s, e, _d, _b in sweep.math_spans(text)]


def _is_structural(m):
    k = m.get("kind", "")
    if m.get("layer") == "scope":
        return not k.startswith(STRUCTURAL_ENV_PREFIXES)
    return m.get("layer") == "dp" and k in STRUCTURAL_SCOPE


def _source_nonsymbol_regions(text):
    """Recognize explicit TeX text/layout arguments without consulting marks.

    Deliberately conservative: unknown macros do not remove denominator terms.
    Nested braces are matched; unterminated arguments are not exemptions.
    """
    regions = []
    commands = re.compile(r"\\(?:text|mbox|textrm|textnormal|label|ref|eqref|cite|begin|end)\s*\{")
    for match in commands.finditer(text):
        start, pos, depth = match.end(), match.end(), 1
        while pos < len(text) and depth:
            if text[pos] in "{}" and (pos == 0 or text[pos - 1] != "\\"):
                depth += 1 if text[pos] == "{" else -1
            pos += 1
        if depth == 0:
            regions.append((start, pos - 1))
    for match in re.finditer(r"\\(?:hspace|vspace)\*?\s*\{\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)\s*(cm|mm|pt|em|ex|in|pc)\s*\}", text):
        regions.append(match.span(1))
    return regions


_DEFINED_AS = re.compile(r"\s*\}?\s+is\s+defined\s+(?:as|to be)\b")
_LET = re.compile(r"\blet", re.I)  # every place the binder pattern can begin


class _Grounding:
    """One paper's source-grounding lookups, shared by all of its marks.

    The blank lines are found once, so a paragraph's edges come from a bisection
    instead of str.rfind/str.find scanning back and forth for every mark; a
    paragraph's "Let" positions are found once; and each (symbol, paragraph)
    binder answer is kept. Repeating all of that per mark was quadratic in a
    paper with long paragraphs.
    """

    def __init__(self, text):
        self.text = text
        self.blank_lines = []
        i = text.find("\n\n")
        while i != -1:
            self.blank_lines.append(i)
            i = text.find("\n\n", i + 1)
        self.lets = {}
        self.binders = {}

    def paragraph_edges(self, start):
        """(text.rfind("\n\n", 0, start), text.find("\n\n", start))"""
        k = bisect_right(self.blank_lines, start - 2)
        before = self.blank_lines[k - 1] if k else -1
        k = bisect_left(self.blank_lines, start)
        after = self.blank_lines[k] if k < len(self.blank_lines) else -1
        return before, after

    def lets_in(self, lo, hi):
        if (lo, hi) not in self.lets:
            self.lets[(lo, hi)] = [m.start() for m in _LET.finditer(self.text, lo, hi)]
        return self.lets[(lo, hi)]


def _source_grounded(text, mark, grounding=None):
    """Check a narrow, explicit source declaration, not the proposed label.

    Current evidence scope: local Let $x$ be ... binders in the same paragraph,
    and a term immediately followed by 'is defined as/to be'. Other authority
    claims remain unverified debt until an independent authority reader exists.

    GROUNDING, a _Grounding for this TEXT, shares the paragraph and binder
    lookups between the paper's marks; without it each call works them out.
    """
    start, end = mark["start"], mark["end"]
    surface = text[start:end]
    if mark.get("kind") == "concept":
        fields = dict(mark.get("fields", []))
        claim = fields.get("grounded")
        if not isinstance(claim, str) or not claim.strip():
            return False
        # matched in place: text[end:] copied the rest of the paper for every term
        return bool(re.fullmatch(r"[A-Za-z][A-Za-z -]*", surface)
                    and _DEFINED_AS.match(text, end))
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*|\\[A-Za-z]+", surface):
        return False
    grounding = grounding or _Grounding(text)
    before, segment_end = grounding.paragraph_edges(start)
    paragraph = before + 2
    paragraph = max(0, paragraph if paragraph >= 2 else 0)
    # Include the binder's own symbol as well as subsequent uses.
    limit = segment_end if segment_end >= 0 else len(text)
    key = (surface, paragraph, limit)
    if key not in grounding.binders:
        binder = re.compile(r"\bLet\s+\$\s*(" + re.escape(surface) + r")\s*\$\s+be\s+(?:an?\s+)?[A-Za-z][A-Za-z -]*", re.I)
        # Binder matches come left to right with rising group-1 starts, so the
        # first one alone decides whether any sits at or before START. It can
        # only begin at a "Let"; matching there with the paragraph's end as the
        # text's end reads exactly what a search of the paragraph copy read.
        first = None
        for at in grounding.lets_in(paragraph, limit):
            found = binder.match(text, at, limit)
            if found:
                first = found.start(1)
                break
        grounding.binders[key] = first
    first = grounding.binders[key]
    return first is not None and first <= start


def check_paper(paper, data=None, golden_dir=GOLDEN_DIR):
    """Return {paper, coverage{...}, violations[...], counts{...}}."""
    if data is None:
        f = Path(golden_dir) / f"fable-{paper}-dp-emacs.json"
        data = json.loads(f.read_text())
    text, marks = data["text"], data["marks"]
    spans = _math_spans(text)
    atoms, span_cover = Stabbing(spans), Extents(spans)
    grounding = _Grounding(text)  # _source_grounded's lookups, shared across this paper's marks
    V = []  # violations

    def add(inv, sev, s, e, msg, fix):
        V.append({"inv": inv, "severity": sev, "start": s, "end": e,
                  "msg": msg, "fix": fix})

    struct = [m for m in marks if _is_structural(m)]
    math_marks = [m for m in marks if m.get("kind") in MATH_KINDS]

    # --- WELL-FORMEDNESS ------------------------------------------------
    # W-ATOMIC: no structural-scope boundary strictly inside a $...$ span.
    # (A scope may CONTAIN a span or sit outside it; it may not split one.)
    for m in struct:
        # the first span, in span order, with either boundary strictly inside it
        inside = atoms.containing(m["start"]) + atoms.containing(m["end"])
        if inside:
            s, e = spans[min(inside)]
            add("W-ATOMIC", "error", m["start"], m["end"],
                f'{m.get("kind")} boundary inside math span '
                f'{text[s:e][:30]!r} — math is atomic', "tighten-detector")
    # W-NEST: structural scopes nest or are disjoint, never partially overlap.
    ss = sorted(struct, key=lambda m: (m["start"], -m["end"]))
    for i, a in enumerate(ss):
        for b in ss[i + 1:]:
            if b["start"] >= a["end"]:
                break  # sorted: no later scope can overlap a
            if b["end"] > a["end"]:  # a.start <= b.start < a.end < b.end
                add("W-NEST", "error", b["start"], b["end"],
                    f'{b.get("kind")} partially overlaps {a.get("kind")} '
                    f'(crossing, not nesting)', "tighten-detector")
    # W-NEST-SCOPE: the BROAD nesting lint — every extent scope (environments,
    # manifest/binder scopes, Let–Then implications, IATC claims) must nest or be
    # disjoint, never cross. Catches env×scope / claim×scope crossings the
    # struct-only W-NEST above misses (Joe's final linting gate).
    for a, b in scope_crossings(marks):
        add("W-NEST-SCOPE", "error", b["start"], b["end"],
            f'{b.get("kind")} crosses {a.get("kind")} '
            f'(partial overlap, not nesting)', "tighten-detector")
    # W-SENTENCE: a non-env structural scope must not cross ". ".
    for m in struct:
        seg = text[m["start"]:m["end"]]
        if ". " in seg:
            add("W-SENTENCE", "error", m["start"], m["end"],
                f'{m.get("kind")} crosses a sentence boundary (". ")',
                "tighten-detector")

    # --- COVERAGE -------------------------------------------------------
    # C-MATH-NONNULL: every $...$ span carries a math mark (R1, hungry-$).
    math_extents = [(m["start"], m["end"]) for m in math_marks]
    math_cover = Extents(math_extents)
    null_spans = 0
    for s, e in spans:
        if not math_cover.contain(s, e):
            null_spans += 1
            add("C-MATH-NONNULL", "debt", s, e,
                f'$-span {text[s:e][:30]!r} has no math mark', "extend-coverage")
    # C-SYM-TAGGED / C-SYM-GROUND: every letter-run inside math is tagged;
    # ungrounded tagged symbols are explicit (countable) debt.
    sym_marks = [m for m in marks if m.get("kind") in SYMBOL_KINDS]
    nonsym_extents = _source_nonsymbol_regions(text)
    nonsym_cover = Extents(nonsym_extents)
    for m in marks:
        if m.get("kind") in NON_SYMBOL_KINDS and not nonsym_cover.contain(m["start"], m["end"]):
            add("W-NONSYM-WITNESS", "error", m["start"], m["end"],
                "non-symbol claim has no independently recognized source context",
                "tighten-detector")
    # PIECEWISE coverage (DC-6): a letter-run may be tiled by SEVERAL symbol marks
    # (a split juxtaposition "gf" -> g + f). The run is tagged if every char is
    # under some symbol mark, grounded if every char is under a grounded one. The
    # LETTER_RUN denominator is unchanged, so corpus symbol counts don't move.
    GROUNDED = {"symbol-grounded", "concept-typed", "classified"}
    tagged_pos, grounded_pos = set(), set()
    for m in sym_marks:
        rng = range(m["start"], m["end"])
        tagged_pos.update(rng)
        if m["kind"] in GROUNDED:
            if _source_grounded(text, m, grounding):
                grounded_pos.update(rng)
            else:
                add("C-GROUND-WITNESS", "debt", m["start"], m["end"],
                    "grounding claim lacks a verified source declaration",
                    "extend-coverage")
                if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*|\\[A-Za-z]+", text[m["start"]:m["end"]]):
                    add("W-GROUND-SPAN", "error", m["start"], m["end"],
                        "grounding mark is not one symbol", "tighten-detector")
    total_syms = tagged = grounded = nonsym = 0
    for s, e in spans:
        for lm in LETTER_RUN.finditer(text[s:e]):
            ls, le = s + lm.start(), s + lm.end()
            # Only independently parsed source context can exclude a run.
            if nonsym_cover.contain(ls, le):
                nonsym += 1
                continue
            total_syms += 1
            run = range(ls, le)
            if all(p in tagged_pos for p in run):
                tagged += 1
                if all(p in grounded_pos for p in run):
                    grounded += 1
                else:
                    add("C-SYM-GROUND", "debt", ls, le,
                        f'symbol {text[ls:le]!r} tagged but ungrounded',
                        "extend-coverage")
            else:
                add("C-SYM-TAGGED", "debt", ls, le,
                    f'letter-run {text[ls:le]!r} in math is untagged',
                    "extend-coverage")
    # W-SYM-JUXTAPOSITION (DC-6): a single symbol mark over a BARE multi-letter
    # italic run is a mis-tokenised product ("gf"=g·f) the detector should split.
    # Independent test on the SOURCE: not inside a \mathrm/\operatorname group and
    # not a sub/superscript modifier. Debt (not error) — never inflates wf errors.
    ma_global = []
    for s, e in spans:
        for rs, re_ in mathalpha_regions(text[s:e]):
            ma_global.append((s + rs, s + re_))
    ma_cover = Extents(ma_global)
    # Only UNGROUNDED bare runs (kind "symbol"): a grounded multi-letter unit is
    # a NAME the binder resolved (a sloppy bare "Ab" = the category, not A·b) —
    # not a product to split, so not a juxtaposition defect.
    for m in sym_marks:
        if m["kind"] != "symbol":
            continue
        # only within MEASURED math spans — consistent with C-SYM coverage, and
        # avoids flagging marks the detector placed via per-file span boundaries
        # that differ from the concatenated-text tokenizer (e.g. xy-pic displays).
        if not span_cover.contain(m["start"], m["end"]):
            continue
        surf = text[m["start"]:m["end"]]
        if len(surf) >= 2 and surf.isalpha() \
                and not ma_cover.contain(m["start"], m["end"]) \
                and not is_script_run(text, m["start"]):
            add("W-SYM-JUXTAPOSITION", "debt", m["start"], m["end"],
                f'ungrounded bare multi-letter run {surf!r} should split into '
                f'single symbols (TeX sets it as a product)', "tighten-detector")
    # C-DEFINIENS-DEBT: a definiens that resolves nowhere = irreducible debt.
    for m in marks:
        if m.get("kind") == "definiens" and m.get("fields"):
            cov = next((v for k, v in m["fields"] if k == "coverage"), "")
            if "DEBT" in cov:
                add("C-DEFINIENS-DEBT", "debt", m["start"], m["end"],
                    f'definiens {text[m["start"]:m["end"]][:40]!r} undefined '
                    f'(Lean/PM/nLab all miss)', "irreducible-debt")
    # C-TERM-COVERAGE: every AUTHOR-EMPHASISED prose phrase should be NOTICED —
    # carry a concept/definiendum/definiens mark. Emphasis is the independent
    # locator (the detector does not use it); coverage over it measures the
    # "terms not noticed" defect class (DC-1) corpus-wide. Emphasis wrapping a
    # formula (content overlaps a math span) is skipped — not a prose term.
    notice_extents = [(m["start"], m["end"]) for m in marks
                      if m.get("kind") in TERM_NOTICE_KINDS]
    notice_cover = Extents(notice_extents)
    terms_located = terms_covered = 0
    for em in EMPH_RE.finditer(text):
        cs, ce = em.start(1), em.end(1)
        term = text[cs:ce].strip()
        if ce - cs < 3 or not any(c.isalpha() for c in term):
            continue
        if term.endswith(".") or ". " in term:
            continue  # an emphasised SENTENCE (stress), not a named term
        if span_cover.overlap(cs, ce):
            continue  # emphasis around math, not a term
        terms_located += 1
        if notice_cover.overlap(cs, ce):
            terms_covered += 1
        else:
            add("C-TERM-COVERAGE", "debt", cs, ce,
                f'emphasised term {term[:40]!r} not noticed (no concept mark)',
                "extend-coverage")

    # C-TERM-GROUND: a concept term grounded to an authority (nLab/NNexus/CT-prior
    # via the lexicon spotter) vs only heuristically spotted. Mirrors
    # C-SYM-GROUND for the prose-term layer; ungrounded = extend-coverage debt.
    concept_marks_ = [m for m in marks if m.get("kind") == "concept"]
    terms_grounded = 0
    for m in concept_marks_:
        if _source_grounded(text, m, grounding):
            terms_grounded += 1
        else:
            add("C-TERM-GROUND", "debt", m["start"], m["end"],
                f'concept {text[m["start"]:m["end"]][:40]!r} not grounded to an '
                f'authority', "extend-coverage")

    # C-IMPL-PAIR (DC-3): a sentence-initial Then/Hence/Thus with a Let/Given/
    # Suppose hypothesis just before it should sit inside an `implies` scope that
    # links the two. Independent locator (consequent connective + nearby
    # hypothesis); uncovered = the implication structure was not captured.
    impl_extents = [(m["start"], m["end"]) for m in marks if m.get("kind") == "implies"]
    impl_cover = Extents(impl_extents)
    for cm in re.finditer(r"(?:^|\.\s+)(Then|Hence|Thus|Therefore)\b", text):
        cs = cm.start(1)
        if re.search(r"\b(?:Let|Given|Suppose|Assume)\b", text[max(0, cs - 400):cs]):
            if not impl_cover.hold(cs):
                add("C-IMPL-PAIR", "debt", cs, cs + len(cm.group(1)),
                    f'"{cm.group(1)}" consequent not linked to its hypothesis '
                    f'(no implication scope)', "extend-coverage")

    # --- COVERAGE BEST-GUESS (per-paper scalars) ------------------------
    def rate(n, d):
        return round(n / d, 4) if d else 1.0
    errors = [v for v in V if v["severity"] == "error"]
    coverage = {
        "math_spans": len(spans),
        "math_null": null_spans,
        "math_coverage": rate(len(spans) - null_spans, len(spans)),
        "symbols": total_syms,
        "nonsym_excluded": nonsym,   # length-units/env-names/text-mode (not math)
        "symbol_tagged": rate(tagged, total_syms),
        "symbol_grounded": rate(grounded, total_syms),
        "terms_emphasised": terms_located,
        "term_coverage": rate(terms_covered, terms_located),
        "terms_concept": len(concept_marks_),
        "term_grounded": rate(terms_grounded, len(concept_marks_)),
        "wellformed_errors": len(errors),
        # one headline number: grounded-symbol rate is the live convergence dial
        "best_guess": rate(grounded, total_syms),
    }
    counts = {}
    for v in V:
        counts[v["inv"]] = counts.get(v["inv"], 0) + 1
    return {"paper": paper,
            "grounding_evidence_policy": "source-declarations-v1; unsupported authority claims remain debt",
            "coverage": coverage, "counts": counts,
            "violations": V}


def _print_paper(rep):
    c, counts = rep["coverage"], rep["counts"]
    print(f"\n{rep['paper']}  —  best-guess coverage {c['best_guess']:.0%} "
          f"(grounded {c['symbol_grounded']:.0%}, tagged {c['symbol_tagged']:.0%} "
          f"of {c['symbols']} symbols; math {c['math_coverage']:.0%} of "
          f"{c['math_spans']} spans)")
    err = c["wellformed_errors"]
    print(f"  well-formedness: {'CLEAN' if err == 0 else str(err)+' ERRORS'}"
          f"   debt: {sum(1 for v in rep['violations'] if v['severity']=='debt')}")
    for inv in sorted(counts):
        sev = next(v["severity"] for v in rep["violations"] if v["inv"] == inv)
        fix = next(v["fix"] for v in rep["violations"] if v["inv"] == inv)
        flag = "✗" if sev == "error" else "·"
        print(f"    {flag} {inv:18} {counts[inv]:5}   [{fix}]")


def _check_marks_file(path, loss_dir):
    """One paper's report, written to loss_dir; runs in a worker process."""
    f = Path(path)
    paper = f.name[len("fable-"):-len("-dp-emacs.json")]
    rep = check_paper(paper, json.loads(f.read_text()))
    (Path(loss_dir) / f"{paper}.json").write_text(json.dumps(rep))
    return rep


def corpus(golden_dir=GOLDEN_DIR, loss_dir=LOSS_DIR):
    golden_dir = Path(golden_dir)
    loss_dir = Path(loss_dir)
    loss_dir.mkdir(parents=True, exist_ok=True)
    reps = []
    # Papers are checked on every CPU the job may use; reported and aggregated
    # here in file order, as the serial loop did.
    files = [str(f) for f in sorted(golden_dir.glob("fable-*-dp-emacs.json"))]
    for _, rep in paper_pool.map_papers(partial(_check_marks_file, loss_dir=str(loss_dir)), files):
        reps.append(rep)
        _print_paper(rep)
    # aggregate
    agg = {"papers": len(reps), "by_invariant": {}, "totals": {
        "errors": 0, "debt": 0, "symbols": 0, "grounded": 0}}
    for rep in reps:
        for inv, n in rep["counts"].items():
            agg["by_invariant"][inv] = agg["by_invariant"].get(inv, 0) + n
        agg["totals"]["errors"] += rep["coverage"]["wellformed_errors"]
        agg["totals"]["debt"] += sum(1 for v in rep["violations"]
                                     if v["severity"] == "debt")
        agg["totals"]["symbols"] += rep["coverage"]["symbols"]
        agg["totals"]["grounded"] += round(rep["coverage"]["symbol_grounded"]
                                           * rep["coverage"]["symbols"])
    t = agg["totals"]
    agg["corpus_best_guess"] = round(t["grounded"] / t["symbols"], 4) \
        if t["symbols"] else 1.0
    (loss_dir / "dashboard.json").write_text(json.dumps(agg, indent=1))
    print(f"\n{'='*60}\nCORPUS  ({agg['papers']} papers)  "
          f"best-guess {agg['corpus_best_guess']:.0%}   "
          f"errors {t['errors']}   debt {t['debt']}")
    for inv in sorted(agg["by_invariant"]):
        print(f"    {inv:18} {agg['by_invariant'][inv]:6}")
    print(f"wrote {loss_dir}/dashboard.json")
    return agg


def self_test_grounding():
    """F1 regression controls using only temporary, source-backed fixtures."""
    import tempfile

    def mark(kind, start, end, **fields):
        return dict(kind=kind, start=start, end=end, layer="dp", **fields)

    with tempfile.TemporaryDirectory(prefix="invariants-f1-") as root:
        def check(text, marks):
            data = {"text": text, "marks": [mark("math", a, b)
                    for a, b in _math_spans(text)] + marks}
            (Path(root) / "fable-probe-dp-emacs.json").write_text(json.dumps(data))
            return check_paper("probe", golden_dir=Path(root))

        # Exact audit attack: one classified mark spanning all of $x+y$.
        text = "$x+y$"
        baseline = check(text, [])
        attack = [mark("classified", 0, len(text))]
        assert attack != []
        bad = check(text, attack)
        assert baseline["coverage"]["symbols"] == bad["coverage"]["symbols"] == 2
        assert baseline["coverage"]["best_guess"] == bad["coverage"]["best_guess"] == 0
        assert bad["counts"].get("W-GROUND-SPAN") == 1
        print("F1 blanket classified: PASS (grounding 0 -> 0; 2 symbols; W-GROUND-SPAN)")

        # A detector cannot choose its denominator; real TeX text still excludes.
        for kind in ("layout", "text-mode"):
            bad = check(text, [mark(kind, 0, len(text))])
            assert bad["coverage"]["symbols"] == 2
            assert bad["coverage"]["best_guess"] == 0
            assert bad["counts"].get("W-NONSYM-WITNESS") == 1
        layout = r"$x+\text{hello}+\hspace{2cm}$"
        start = layout.index("hello")
        good = check(layout, [mark("text-mode", start, start + 5)])
        assert good["coverage"]["symbols"] == 1
        assert good["coverage"]["nonsym_excluded"] == 2
        assert good["coverage"]["wellformed_errors"] == 0
        print("F1 layout: PASS (blanket cannot exclude; source-backed exclusions pass)")

        # Positive control: this is not a checker that simply rejects everything.
        text = "Let $x$ be a set. Then $x$."
        good = check(text, [mark("symbol-grounded", i, i + 1)
                           for i, ch in enumerate(text) if ch == "x"])
        assert good["coverage"]["symbols"] == 2
        assert good["coverage"]["best_guess"] == 1.0
        assert good["coverage"]["wellformed_errors"] == 0
        print("F1 source binder: PASS (grounding 1.0; no well-formedness errors)")

        text = "A group is defined as a set with a multiplication."
        start = text.index("group")
        for value, expected in ((False, 0.0), ("source:definition", 1.0)):
            rep = check(text, [mark("concept", start, start + 5,
                                   fields=[["grounded", value]])])
            assert rep["coverage"]["term_grounded"] == expected
        print("F1 concept claim: PASS (false earns 0; source-backed positive earns 1)")
    print("self-test-grounding PASS: 4 controls; no repository data required")
    return 0


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if argv == ["--self-test-grounding"]:
        return self_test_grounding()
    if not argv:
        print("usage: check_invariants.py <paper-id> | --corpus | --self-test-grounding "
              "[--golden-dir DIR] [--loss-dir DIR]")
        return 2
    golden_dir = GOLDEN_DIR
    loss_dir = LOSS_DIR
    rest = []
    i = 0
    while i < len(argv):
        if argv[i] == "--golden-dir" and i + 1 < len(argv):
            golden_dir = Path(argv[i + 1])
            i += 2
        elif argv[i] == "--loss-dir" and i + 1 < len(argv):
            loss_dir = Path(argv[i + 1])
            i += 2
        else:
            rest.append(argv[i])
            i += 1
    argv = rest
    if not argv:
        print("usage: check_invariants.py <paper-id> | --corpus | --self-test-grounding "
              "[--golden-dir DIR] [--loss-dir DIR]")
        return 2
    if argv[0] == "--corpus":
        corpus(golden_dir=golden_dir, loss_dir=loss_dir)
        return 0
    rep = check_paper(argv[0], golden_dir=golden_dir)
    Path(loss_dir).mkdir(parents=True, exist_ok=True)
    (Path(loss_dir) / f"{argv[0]}.json").write_text(json.dumps(rep))
    _print_paper(rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
