#!/usr/bin/env python3
"""Skolem audit over the nLab wiring extraction: is the content in scope?

The mission-side Skolem audit (mission_scope_bindings.py) self-applied to
mathematics (Joe, 2026-06-11): *named entities that are not in a scope are
suspect, and scopes without named entities inside are also suspect.*

Entities here are the two things the corpus actually names:
  - symbolic expressions ($...$ / $$...$$) — the layer whose mining is
    currently "trust in GPU"; this audit gives the CPU-side baseline
  - wiki-links ([[...]]) — nLab's native named entities, gold-standard

Scope, two grades (strict / weak), mirroring the mission audit's grades:
  - strict: inside a typed environment span (env/definition, env/proof, ...)
  - weak:   in a paragraph that carries at least one discourse binder
            (bind/typed, bind/let, quant/universal, ...)
  - floating: neither — the suspect class

The dual class: a typed environment whose span contains NO symbolic
expression and NO wiki-link is a vacuous scope (`∀x:` with nothing bound).

Symbolic expressions are additionally TYPED with a lexical classifier ported
from the First Proof Sprint .md→.tex workup (math-proofread-style.sty's
taxonomy: quantifier / arrow / relation / large-operator / named-operator /
greek / number / variable / text). The in-scope vs floating split per type is
the quality signal for future GPU mining runs.

Reads data/nlab-wiring/pages.json + the raw content.md files.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WIRING = ROOT / "data" / "nlab-wiring" / "pages.json"
NLAB_PAGES = Path(os.environ.get(
    "NLAB_PAGES", str(ROOT.parent / "nlab-content" / "pages")))

MATH_RE = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$", re.S)
# [[!include ...]] / [[!redirects ...]] are wiki directives (nav/plumbing),
# not named entities — excluded from the audit.
WIKILINK_RE = re.compile(r"\[\[(?!!)(.+?)\]\]")
PARA_RE = re.compile(r"\n\s*\n")
HEADING_RE = re.compile(r"^#{1,3}\s*(.+?)\s*#*\s*$", re.M)

# The mini-mission reading (Joe + ni17003, 2026-06-11): an nLab page is a
# small mission. The Idea section is its HEAD — expository register, IATC
# moves, the ∃ the body must discharge. Entities there are not "floating";
# their Skolem obligation is discharge downstream, not enclosure in a binder.
SECTION_PHASE = {
    "idea": "head",
    "motivation": "head",
    "definition": "body", "definitions": "body",
    "statement": "body", "statements": "body",
    "properties": "body",
    "theorem": "body", "theorems": "body",
    "examples": "body", "example": "body",
    "applications": "body",
    "related concepts": "relates", "related entries": "relates",
    "related pages": "relates",
    "references": "sources", "literature": "sources",
}


def section_spans(text: str) -> list[tuple[int, int, str]]:
    """(start, end, phase) for canonical nLab sections; phase ∈
    head|body|relates|sources|other."""
    spans = []
    matches = list(HEADING_RE.finditer(text))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        title = re.sub(r"\{#.*?\}", "", m.group(1)).strip().lower()
        phase = SECTION_PHASE.get(title, "other")
        spans.append((m.start(), end, phase))
    return spans

# Lexical expression typer — the math-proofread-style.sty taxonomy, as
# priority-ordered detectors over raw LaTeX. First hit wins: an expression
# containing a quantifier IS a quantified statement even though it also
# contains variables; an arrow makes it a morphism/map expression; etc.
# NB: (?![a-zA-Z]) rather than \b — "_" is a word char, so \b fails right
# where these commands most often appear (\sum_{i}, \lim_{n}, ...).
EXPR_TYPE_RULES = [
    ("quantifier", re.compile(r"\\(forall|exists)(?![a-zA-Z])")),
    ("arrow", re.compile(r"\\(to|rightarrow|longrightarrow|mapsto|hookrightarrow|twoheadrightarrow|leftarrow|Rightarrow|xrightarrow|underoverset|overset|underset)(?![a-zA-Z])|\\stackrel")),
    ("large-operator", re.compile(r"\\(sum|prod|int|bigcup|bigcap|coprod|lim|colim|varinjlim|varprojlim|holim|hocolim|otimes|oplus)(?![a-zA-Z])")),
    ("relation", re.compile(r"=|\\(le|leq|ge|geq|neq|cong|simeq|equiv|subset|subseteq|supset|in|ni|sim|approx|vdash|models|prec|succ)(?![a-zA-Z])|<|>")),
    ("named-operator", re.compile(r"\\(mathrm|operatorname|mathop|Hom|End|Aut|Map|Spec|Ext|Tor)(?![a-zA-Z])|\\(sin|cos|log|exp|ker|coker|im|dim|rank)(?![a-zA-Z])")),
    ("greek", re.compile(r"\\(alpha|beta|gamma|delta|epsilon|varepsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|pi|rho|sigma|tau|upsilon|phi|varphi|chi|psi|omega|Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Phi|Psi|Omega)(?![a-zA-Z])")),
    ("number", re.compile(r"^\s*-?\d+(\.\d+)?\s*$")),
    ("text", re.compile(r"\\(text|mathrm)\{[^}]*\s[^}]*\}")),
]


def classify_expr(latex: str) -> str:
    for name, rule in EXPR_TYPE_RULES:
        if rule.search(latex):
            return name
    return "variable"


def build_page_index(pages_dir: Path) -> dict[str, Path]:
    """page basename (e.g. '494') -> content.md path."""
    index = {}
    for content in pages_dir.glob("*/*/*/*/*/content.md"):
        index[content.parent.name] = content
    return index


def paragraph_spans(text: str) -> list[tuple[int, int]]:
    spans, start = [], 0
    for m in PARA_RE.finditer(text):
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(text)))
    return spans


def audit_page(page: dict, text: str) -> dict:
    env_spans = []
    for env in page.get("environments", []):
        c = env.get("hx/content", {})
        pos, length = c.get("position"), c.get("length")
        if isinstance(pos, int) and isinstance(length, int):
            env_spans.append((pos, pos + length, env.get("hx/type")))

    binder_positions = [
        s["hx/content"]["position"]
        for s in page.get("discourse", [])
        if isinstance(s.get("hx/content", {}).get("position"), int)
    ]

    paras = paragraph_spans(text)

    def para_of(pos: int) -> int:
        lo, hi = 0, len(paras) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if paras[mid][1] < pos:
                lo = mid + 1
            else:
                hi = mid
        return lo

    binder_paras = {para_of(p) for p in binder_positions}

    def scope_grade(pos: int) -> str:
        for start, end, _ in env_spans:
            if start <= pos < end:
                return "strict"
        if para_of(pos) in binder_paras:
            return "weak"
        return "floating"

    sections = section_spans(text)

    def section_phase(pos: int) -> str:
        for start, end, phase in sections:
            if start <= pos < end:
                return phase
        return "other"

    # Entities channel 1: symbolic expressions, typed. A floating expression
    # in a HEAD-register section is re-graded head-register: its obligation
    # is discharge, not enclosure.
    expr_total = Counter()
    expr_floating = Counter()
    exprs = []
    for m in MATH_RE.finditer(text):
        latex = (m.group(1) or m.group(2) or "").strip()
        if not latex:
            continue
        etype = classify_expr(latex)
        grade = scope_grade(m.start())
        if grade == "floating" and section_phase(m.start()) == "head":
            grade = "head-register"
        expr_total[etype] += 1
        if grade == "floating":
            expr_floating[etype] += 1
        exprs.append((m.start(), grade))

    # Entities channel 2: wiki-links (named entities), same re-grading; plus
    # the mini-mission flow: HEAD links discharged (re-touched by a body
    # section) vs undischarged (promised in the Idea, never picked up).
    link_grades = Counter()
    head_links: dict[str, bool] = {}
    body_targets = set()
    for m in WIKILINK_RE.finditer(text):
        target = m.group(1).split("|")[0].strip().lower()
        phase = section_phase(m.start())
        grade = scope_grade(m.start())
        if grade == "floating" and phase == "head":
            grade = "head-register"
        link_grades[grade] += 1
        if phase == "head":
            head_links.setdefault(target, False)
        elif phase in ("body", "other"):
            # 'other' counts for discharge: many pages put their substance
            # under idiosyncratic headings ("Abstract approach", "In higher
            # category theory"); only head/relates/sources are non-body.
            body_targets.add(target)
    # Discharge channel 2: nLab convention links a term ONCE (usually at
    # first mention, i.e. in the Idea), so link-retouch alone badly
    # undercounts. A plain-text body mention discharges too.
    body_text = "\n".join(
        text[start:end] for start, end, phase in sections
        if phase in ("body", "other")
    ).lower()
    for target in head_links:
        head_links[target] = (
            target in body_targets
            or re.search(r"(?<![a-z0-9])" + re.escape(target) + r"(?![a-z0-9])",
                         body_text) is not None
        )
    undischarged = sorted(t for t, ok in head_links.items() if not ok)

    expr_grades = Counter(g for _, g in exprs)

    # Dual class: environments binding nothing.
    vacuous_envs = []
    for start, end, etype in env_spans:
        span_text = text[start:end]
        if not MATH_RE.search(span_text) and not WIKILINK_RE.search(span_text):
            vacuous_envs.append({"type": etype, "at": start,
                                 "preview": " ".join(span_text.split())[:80]})

    return {
        "page": page.get("page_name"),
        "page_id": page.get("page_id"),
        "exprs": sum(expr_total.values()),
        "expr-grades": dict(expr_grades),
        "expr-types": dict(expr_total),
        "expr-types-floating": dict(expr_floating),
        "links": sum(link_grades.values()),
        "link-grades": dict(link_grades),
        "envs": len(env_spans),
        "vacuous-envs": vacuous_envs,
        "head-links": len(head_links),
        "head-discharged": sum(1 for ok in head_links.values() if ok),
        "head-undischarged": undischarged,
    }


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wiring", type=Path, default=WIRING)
    ap.add_argument("--pages-dir", type=Path, default=NLAB_PAGES)
    ap.add_argument("--limit", type=int, default=0, help="Audit only the first N pages.")
    ap.add_argument("--page", help="Detail report for one page name.")
    ap.add_argument("--json", type=Path, help="Write per-page results to this path.")
    return ap.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    pages = json.loads(args.wiring.read_text(encoding="utf-8"))
    if args.limit:
        pages = pages[: args.limit]
    index = build_page_index(args.pages_dir)

    totals = {
        "expr-grades": Counter(), "link-grades": Counter(),
        "expr-types": Counter(), "expr-types-floating": Counter(),
    }
    vacuous_total = 0
    env_total = 0
    head_links_total = 0
    head_discharged_total = 0
    results = []
    missing = 0

    for page in pages:
        basename = str(page.get("page_id", "")).replace("nlab-", "")
        content = index.get(basename)
        if content is None:
            missing += 1
            continue
        text = content.read_text(encoding="utf-8", errors="ignore")
        r = audit_page(page, text)
        results.append(r)
        totals["expr-grades"].update(r["expr-grades"])
        totals["link-grades"].update(r["link-grades"])
        totals["expr-types"].update(r["expr-types"])
        totals["expr-types-floating"].update(r["expr-types-floating"])
        vacuous_total += len(r["vacuous-envs"])
        env_total += r["envs"]
        head_links_total += r["head-links"]
        head_discharged_total += r["head-discharged"]

    if args.page:
        for r in results:
            if r["page"] == args.page:
                print(json.dumps(r, indent=2))
                return
        print(f"page not found: {args.page}")
        return

    n_expr = sum(totals["expr-grades"].values())
    n_link = sum(totals["link-grades"].values())

    def pct(counter, key, total):
        return f"{counter.get(key, 0)} ({100 * counter.get(key, 0) / total:.1f}%)" if total else "0"

    print(f"pages audited: {len(results)} (missing content: {missing})")
    print(f"\nsymbolic expressions: {n_expr}")
    for grade in ("strict", "weak", "head-register", "floating"):
        print(f"  {grade:13} {pct(totals['expr-grades'], grade, n_expr)}")
    print(f"\nwiki-link named entities: {n_link}")
    for grade in ("strict", "weak", "head-register", "floating"):
        print(f"  {grade:13} {pct(totals['link-grades'], grade, n_link)}")
    print(f"\nvacuous environments: {vacuous_total} / {env_total}")
    if head_links_total:
        print(f"\nmini-mission HEAD discharge: {head_discharged_total}/{head_links_total} "
              f"Idea-section links re-touched by a body section "
              f"({100 * head_discharged_total / head_links_total:.1f}%)")
    print("\nexpression types (total → floating):")
    for etype, n in totals["expr-types"].most_common():
        fl = totals["expr-types-floating"].get(etype, 0)
        print(f"  {etype:15} {n:8} → {fl:8} floating ({100 * fl / n:.1f}%)")

    if args.json:
        args.json.write_text(json.dumps(results, indent=1), encoding="utf-8")
        print(f"\nper-page results: {args.json}")


if __name__ == "__main__":
    main()
