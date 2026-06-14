#!/usr/bin/env python3
"""Scope audit for First Proof writeups.

This is the E-anatomy-of-a-proof first-cut instrument: adapt the existing
nLab Skolem audit to the proof writeup register without modifying the writeups.
It tokenizes ASCII math and indented display blocks, reuses nlab-wiring's
scope records, and reports binding discipline against the nLab baseline.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import background_corpus_index as bg
from nlab_skolem_audit import classify_expr, paragraph_spans

ROOT = Path(__file__).resolve().parent.parent
WRITEUP_DIR = Path("/home/joe/code/storage/futon6/data/first-proof")
OUT_JSON = ROOT / "data" / "first-proof-scope-audit.json"
OUT_SUMMARY = ROOT / "data" / "first-proof-scope-summary.json"
BACKGROUND_INDEX = ROOT / "data" / "background-corpus-index.json"

NLAB_FLOATING_EXPR_BASELINE = 18.3
NLAB_VACUOUS_BASELINE = {"vacuous": 797, "envs": 30154}

ASCII_EXPR_RE = re.compile(
    r"""
    (?:\b[A-Za-z][A-Za-z0-9_]*\([^\n)]{0,100}\))
    |(?:\b(?:sum|prod|int|lim)_[A-Za-z0-9]+(?:\s*[^.,;\n]{0,80})?)
    |(?:\b[A-Za-z][A-Za-z0-9_]*(?:_[A-Za-z0-9]+)+(?:\([^\n)]{0,80}\))?)
    |(?:\b[A-Za-z][A-Za-z0-9_]*(?:\s*(?:>=|<=|!=|=|->|\|->|<|>)\s*[^.,;\n]{1,100}))
    """,
    re.X,
)
SYMBOL_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_]*\b")
BOLD_RE = re.compile(r"\*\*([^*\n]{2,120})\*\*")
CAP_PHRASE_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9_'-]+(?:\s+|[-])){1,5}[A-Z][A-Za-z0-9_'-]+\b"
)
STOP_SYMBOLS = {
    "A", "An", "By", "For", "If", "In", "Let", "QED", "Since", "Step", "The",
    "Then", "This", "We", "WLOG", "Yes", "No", "Proof", "Problem", "Answer",
    "Conclusion", "References", "where", "with", "and", "or", "the", "of", "in",
    "is", "are", "be", "to", "from", "for", "all", "any", "some", "there",
    "exists", "defined", "finite", "nonzero", "smooth", "real", "complex",
}
STOP_CONCEPTS = {
    "Problem Statement", "Problem", "Answer", "Proof", "References",
    "Conclusion", "Step", "Status", "Question",
}


def _load_nlab_wiring():
    path = ROOT / "scripts" / "nlab-wiring.py"
    spec = importlib.util.spec_from_file_location("nlab_wiring", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_NLAB_WIRING = None


def detect_scopes(entity_id: str, text: str) -> list[dict[str, Any]]:
    global _NLAB_WIRING
    if _NLAB_WIRING is None:
        _NLAB_WIRING = _load_nlab_wiring()
    return _NLAB_WIRING.detect_scopes(entity_id, text)


def indented_line_spans(text: str) -> tuple[list[tuple[int, str]], list[tuple[int, int]]]:
    """Display-math lines at their TRUE text offsets, plus block intervals.

    Review fix (fable-2): the first cut tokenized a stripped/re-joined copy of
    each block, so within-block positions drifted from the original text —
    every display expression was then ALSO found by the inline pass at its
    true offset and double-counted, with the drifted copy mis-graded against
    scope spans. Positions here index the original text; the block intervals
    let the inline pass skip block interiors entirely.
    """
    lines: list[tuple[int, str]] = []
    blocks: list[tuple[int, int]] = []
    offset = 0
    cur_start = None
    for raw in text.splitlines(keepends=True):
        line = raw.rstrip("\n")
        if re.match(r"^(?: {4,}|\t)\S", line):
            if cur_start is None:
                cur_start = offset
            indent = len(line) - len(line.lstrip())
            lines.append((offset + indent, line.strip()))
        else:
            if cur_start is not None:
                blocks.append((cur_start, offset))
                cur_start = None
        offset += len(raw)
    if cur_start is not None:
        blocks.append((cur_start, offset))
    return lines, blocks


def _mathy(token: str) -> bool:
    return bool(re.search(r"[=<>]|->|\|->|_[A-Za-z0-9]+|\([^)]*\)|\b(sum|prod|int|Phi|lambda|mu|pi|omega|tau|disc|det|rank|ker)\b", token))


def expression_records(text: str) -> list[dict[str, Any]]:
    records = []
    seen = set()

    def add(expr: str, pos: int, source: str):
        expr = " ".join(expr.strip().split())
        if not expr or not _mathy(expr):
            return
        key = (pos, expr)
        if key in seen:
            return
        seen.add(key)
        records.append({"expr": expr, "position": pos, "type": classify_expr(expr), "source": source})

    display_lines, blocks = indented_line_spans(text)
    for pos, line in display_lines:
        add(line, pos, "display-line")

    def in_block(p: int) -> bool:
        return any(s <= p < e for s, e in blocks)

    # One expression record per display line; the inline token pass covers
    # prose math only — block interiors are skipped so nothing counts twice.
    for m in ASCII_EXPR_RE.finditer(text):
        if not in_block(m.start()):
            add(m.group(), m.start(), "ascii-inline")
    return sorted(records, key=lambda r: (r["position"], r["expr"]))


def _span(scope: dict[str, Any]) -> tuple[int, int]:
    c = scope.get("hx/content", {})
    pos = c.get("position")
    end = c.get("end")
    if not isinstance(pos, int):
        pos = 0
    if not isinstance(end, int) or end <= pos:
        end = pos + len(c.get("match", ""))
    return pos, end


def _bound_symbols(scopes: list[dict[str, Any]]) -> set[str]:
    out = set()
    for s in scopes:
        for e in s.get("hx/ends", []):
            if e.get("role") == "symbol":
                value = e.get("latex") or e.get("text") or ""
                for sym in SYMBOL_RE.findall(value):
                    out.add(sym)
    return out


def _symbols_in_expr(expr: str) -> set[str]:
    return {s for s in SYMBOL_RE.findall(expr) if s not in STOP_SYMBOLS and not s[0].isdigit()}


LEADING_SENTENCE_WORDS = {
    "The", "A", "An", "In", "On", "By", "So", "If", "For", "Let", "Then",
    "Define", "Step", "This", "We", "Since", "Thus", "Hence", "Note",
}

# ProperName(s) + one or two lowercase nouns — how mathematics names its
# concepts (Kirillov model, Whittaker function, Jacquet module). The plain
# consecutive-capitals regex cannot see these (review fix, fable-2: p2
# yielded "The Kirillov" and never "Kirillov model").
CAP_COMPOUND_RE = re.compile(
    r"\b([A-Z][a-zA-Z-]{2,}(?:\s+[A-Z][a-zA-Z-]{2,})?)\s+([a-z][a-z-]{3,}(?:\s+[a-z][a-z-]{3,})?)\b"
)


def concept_terms(text: str, free_symbols: set[str]) -> list[str]:
    out: set[str] = set()
    for m in BOLD_RE.finditer(text):
        term = " ".join(m.group(1).split())
        # a bold SENTENCE is emphasis, not a concept
        if term and term not in STOP_CONCEPTS and len(term.split()) <= 6 \
           and ". " not in term:
            out.add(term)
    for m in CAP_PHRASE_RE.finditer(text):
        term = " ".join(m.group().split())
        if term and term not in STOP_CONCEPTS and not term.startswith("Problem "):
            out.add(term)
    for m in CAP_COMPOUND_RE.finditer(text):
        head, tail = m.group(1), m.group(2)
        head_parts = head.split()
        if head_parts[0] in LEADING_SENTENCE_WORDS:
            if len(head_parts) == 1:
                continue
            head = " ".join(head_parts[1:])
        out.add(f"{head} {tail}")
        tail_parts = tail.split()
        if len(tail_parts) > 1:
            out.add(f"{head} {tail_parts[0]}")
    for sym in free_symbols:
        if len(sym) >= 4 and sym not in STOP_SYMBOLS:
            out.add(sym)
    return sorted(out, key=lambda s: (s.lower(), s))


def load_background_index(candidates: list[str] | None, index_path: Path = BACKGROUND_INDEX) -> dict[str, Any]:
    if index_path.exists():
        index = bg.load_index(index_path)
        if index.get("schema-version") != 2 or "nnexus-domain-counts" not in index:
            return bg.build_index(candidates, output=index_path)
        if candidates:
            covered = set(index.get("candidate-terms", []))
            requested = {bg.normalize_term(c) for c in candidates if bg.normalize_term(c)}
            # The nLab name set is stable, but ct-prior materialization is
            # candidate-filtered. Rebuild only if this run has candidates the
            # persisted index was not built against; unresolved candidates are
            # legitimate orphans, not a reason to rebuild forever.
            if requested.issubset(covered):
                return index
        else:
            return index
    return bg.build_index(candidates, output=index_path)


def resolve_concepts(candidates: list[str], index: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    resolved = []
    orphans = []
    for term in candidates:
        # Resolution gate (review fix, fable-2): two-letter symbols and
        # bare common words resolving against encyclopedia entries (I →
        # imaginary unit, pi → Pi, gives → give) is the W7 keyword-grab
        # failure mode at the resolution layer. Short single tokens never
        # resolve; multiword phrases carry their own specificity.
        if len(term.replace("_", "")) < 4 and " " not in term:
            orphans.append(term)
            continue
        hit = bg.resolve(index, term)
        if hit:
            item = {
                "term": term,
                "resolution-kind": hit["resolution-kind"],
                "target": hit["target"],
                "matched-term": hit["term"],
            }
            for key in ("domains", "domain-count", "msc", "urls"):
                if key in hit:
                    item[key] = hit[key]
            resolved.append(item)
        else:
            orphans.append(term)
    return resolved, orphans


def audit_writeup(path: Path, background_index: dict[str, Any] | None = None) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    entity_id = path.stem.replace("-writeup", "")
    scopes = detect_scopes(entity_id, text)
    exprs = expression_records(text)

    scope_spans = [(s, *_span(s)) for s in scopes]

    # Weak grade is PARAGRAPH-grain, matching nlab_skolem_audit's semantics —
    # the printed nLab baseline is only comparable if both sides scope a
    # binder over its whole paragraph, not just its matched phrase (review
    # fix, fable-2).
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

    binder_paras = {para_of(start) for _, start, _ in scope_spans}

    def grade(pos: int) -> str:
        env_hits = [s for s, start, end in scope_spans if start <= pos < end and str(s.get("hx/type", "")).startswith("env/")]
        if env_hits:
            return "strict"
        if any(start <= pos < end for _, start, end in scope_spans) or para_of(pos) in binder_paras:
            return "weak"
        return "floating"

    expr_types = Counter()
    expr_grades = Counter()
    used_symbols = set()
    for e in exprs:
        g = grade(e["position"])
        e["grade"] = g
        expr_types[e["type"]] += 1
        expr_grades[g] += 1
        used_symbols.update(_symbols_in_expr(e["expr"]))

    bound_symbols = _bound_symbols(scopes)
    free_symbols = used_symbols - bound_symbols
    concepts = concept_terms(text, free_symbols)
    if background_index is None:
        background_index = load_background_index(concepts)
    external_concepts, orphan_concepts = resolve_concepts(concepts, background_index)
    vacuous = []
    for s, start, end in scope_spans:
        if not any(start <= e["position"] < end for e in exprs):
            vacuous.append({
                "scope-id": s.get("hx/id"),
                "type": s.get("hx/type"),
                "match": s.get("hx/content", {}).get("match", "")[:120],
            })

    floating_exprs = [e for e in exprs if e["grade"] == "floating"]
    return {
        "writeup": path.name,
        "expr-count": len(exprs),
        "scope-count": len(scopes),
        "expr-types": dict(expr_types),
        "scope-grades": dict(expr_grades),
        "floating-expr-count": len(floating_exprs),
        "floating-expr-pct": (100.0 * len(floating_exprs) / len(exprs)) if exprs else 0.0,
        "bound-symbols": sorted(bound_symbols),
        "free-symbols": sorted(free_symbols),
        "candidate-concepts": concepts,
        "external-concepts": external_concepts,
        "orphan-concepts": orphan_concepts,
        "externally-bound-count": len(external_concepts),
        "orphan-count": len(orphan_concepts),
        "vacuous-scopes": vacuous,
        "vacuous-count": len(vacuous),
        "scopes": scopes,
        "expressions": exprs,
    }


def writeups(root: Path = WRITEUP_DIR) -> list[Path]:
    return [root / f"problem{i}-writeup.md" for i in range(1, 11)]


def run_audit(root: Path = WRITEUP_DIR) -> list[dict[str, Any]]:
    paths = [p for p in writeups(root) if p.exists()]
    seed_candidates: list[str] = []
    for p in paths:
        text = p.read_text(encoding="utf-8", errors="ignore")
        scopes = detect_scopes(p.stem.replace("-writeup", ""), text)
        exprs = expression_records(text)
        bound = _bound_symbols(scopes)
        used = set()
        for e in exprs:
            used.update(_symbols_in_expr(e["expr"]))
        seed_candidates.extend(concept_terms(text, used - bound))
    index = load_background_index(seed_candidates)
    return [audit_writeup(p, index) for p in paths]


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    expr_total = sum(r["expr-count"] for r in results)
    scope_total = sum(r["scope-count"] for r in results)
    floating = sum(r["floating-expr-count"] for r in results)
    vacuous = sum(r["vacuous-count"] for r in results)
    external = sum(r["externally-bound-count"] for r in results)
    orphan = sum(r["orphan-count"] for r in results)
    external_domains = Counter()
    for r in results:
        for concept in r.get("external-concepts", []):
            for domain in concept.get("domains", []):
                external_domains[domain] += 1
    return {
        "writeups": len(results),
        "expr-total": expr_total,
        "scope-total": scope_total,
        "floating-expr-count": floating,
        "floating-expr-pct": (100.0 * floating / expr_total) if expr_total else 0.0,
        "vacuous-scope-count": vacuous,
        "externally-bound-count": external,
        "orphan-count": orphan,
        "external-resolution-domains": dict(sorted(external_domains.items())),
        "nlab-baseline": {
            "floating-expr-pct": NLAB_FLOATING_EXPR_BASELINE,
            "vacuous-envs": NLAB_VACUOUS_BASELINE,
        },
        "per-writeup": [
            {
                "writeup": r["writeup"],
                "expr-count": r["expr-count"],
                "scope-count": r["scope-count"],
                "floating-expr-pct": round(r["floating-expr-pct"], 1),
                "free-symbols": len(r["free-symbols"]),
                "externally-bound": r["externally-bound-count"],
                "orphan": r["orphan-count"],
                "vacuous-count": r["vacuous-count"],
            }
            for r in results
        ],
    }


def print_table(summary: dict[str, Any]) -> None:
    print("writeup,exprs,scopes,floating%,free-symbols,externally-bound,orphan,vacuous")
    for row in summary["per-writeup"]:
        print(f"{row['writeup']},{row['expr-count']},{row['scope-count']},{row['floating-expr-pct']},{row['free-symbols']},{row['externally-bound']},{row['orphan']},{row['vacuous-count']}")
    print(
        f"TOTAL,{summary['expr-total']},{summary['scope-total']},"
        f"{summary['floating-expr-pct']:.1f},,"
        f"{summary['externally-bound-count']},{summary['orphan-count']},"
        f"{summary['vacuous-scope-count']}"
    )
    print(
        "nLab baseline: "
        f"{NLAB_FLOATING_EXPR_BASELINE}% floating expressions; "
        f"{NLAB_VACUOUS_BASELINE['vacuous']}/{NLAB_VACUOUS_BASELINE['envs']} vacuous envs"
    )


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--writeup-dir", type=Path, default=WRITEUP_DIR)
    ap.add_argument("--json", type=Path, default=OUT_JSON)
    ap.add_argument("--summary-json", type=Path, default=OUT_SUMMARY)
    args = ap.parse_args(argv)

    results = run_audit(args.writeup_dir)
    summary = summarize(results)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(results, indent=1), encoding="utf-8")
    args.summary_json.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print_table(summary)


if __name__ == "__main__":
    main()
