#!/usr/bin/env python3
"""Audit the full First Proof TeX register.

This complements `proof_scope_audit.py` without modifying it. The full TeX
proofs use `\\(...\\)` / `\\[...\\]` math and inline math-proofread `\\m*`
macros. Those macros are harvested as a gold expression-type channel before
being stripped for `classify_expr`.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import proof_scope_audit


ROOT = Path(__file__).resolve().parent.parent
FULL_TEX_DIR = Path("/home/joe/code/storage/futon6/data/first-proof/latex/full")

MACRO_TYPES = {
    "mNumber": "number",
    "mGreek": "greek",
    "mRelation": "relation",
    "mCompare": "relation",
    "mArrow": "arrow",
    "mOperator": "operator",
    "mBin": "operator",
    "mBridgeOperator": "operator",
    "mLargeOperator": "large-operator",
    "mOpName": "named-operator",
    "mFunction": "function",
    "mDelimiter": "delimiter",
    "mMathText": "text",
    "mMathItalic": "variable",
    "mDualStar": "operator",
    "mConstant": "number",
}

MATH_OPEN = re.compile(r"\\\(|\\\[")
SECTION_RE = re.compile(r"\\(?P<kind>section|subsection|subsubsection)(?:\[[^\]]*\])?\{")
SYMBOL_RE = proof_scope_audit.SYMBOL_RE
STOP_SYMBOLS = proof_scope_audit.STOP_SYMBOLS


def matching_brace(text: str, open_idx: int) -> int | None:
    depth = 0
    for i in range(open_idx, len(text)):
        ch = text[i]
        if ch == "{" and (i == 0 or text[i - 1] != "\\"):
            depth += 1
        elif ch == "}" and (i == 0 or text[i - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return i
    return None


def strip_tex_commands(s: str) -> str:
    s = re.sub(r"\\texorpdfstring\{([^{}]*)\}\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\[A-Za-z]+\*?(?:\[[^\]]*\])?\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\[A-Za-z]+\*?", "", s)
    return " ".join(s.replace("{", " ").replace("}", " ").split())


def section_title_at(text: str, brace_idx: int) -> tuple[str, int] | None:
    end = matching_brace(text, brace_idx)
    if end is None:
        return None
    return text[brace_idx + 1:end], end + 1


def section_phase(title: str) -> str:
    clean = strip_tex_commands(title).lower()
    if "problem statement" in clean or clean == "answer":
        return "head"
    if "reference" in clean:
        return "loose"
    return "body"


def section_scopes(entity_id: str, text: str) -> list[dict[str, Any]]:
    starts = []
    for m in SECTION_RE.finditer(text):
        parsed = section_title_at(text, m.end() - 1)
        if not parsed:
            continue
        title, title_end = parsed
        starts.append((m.start(), title_end, m.group("kind"), title, section_phase(title)))
    scopes = []
    for idx, (start, title_end, kind, title, phase) in enumerate(starts):
        end = starts[idx + 1][0] if idx + 1 < len(starts) else len(text)
        clean_title = strip_tex_commands(title)
        scopes.append({
            "hx/id": f"{entity_id}:section-{idx:03d}",
            "hx/role": "section-spine",
            "hx/type": f"section/{phase}",
            "hx/parent": None,
            "hx/ends": [
                {"role": "entity", "ident": entity_id},
                {"role": "environment", "phase": phase, "name": clean_title},
                {"role": "heading", "level": kind, "title": clean_title},
            ],
            "hx/content": {
                "match": text[start:title_end],
                "position": start,
                "end": end,
            },
            "hx/labels": ["scope", "section-spine", phase],
        })
    return scopes


def math_spans(text: str) -> list[tuple[int, int, str]]:
    spans = []
    pos = 0
    while True:
        m = MATH_OPEN.search(text, pos)
        if not m:
            break
        display = m.group() == "\\["
        close = "\\]" if display else "\\)"
        end = text.find(close, m.end())
        if end == -1:
            pos = m.end()
            continue
        spans.append((m.end(), end, "display" if display else "inline"))
        pos = end + len(close)
    return spans


def harvest_and_strip_macros(latex: str) -> tuple[str, list[dict[str, str]]]:
    annotations = []

    def walk(s: str) -> str:
        out = []
        i = 0
        while i < len(s):
            if s[i] == "\\":
                m = re.match(r"\\(m[A-Za-z]+)\{", s[i:])
                if m:
                    macro = m.group(1)
                    open_idx = i + len(m.group(0)) - 1
                    close_idx = matching_brace(s, open_idx)
                    if close_idx is not None:
                        inner = s[open_idx + 1:close_idx]
                        cleaned_inner = walk(inner)
                        if macro in MACRO_TYPES:
                            annotations.append({
                                "macro": macro,
                                "type": MACRO_TYPES[macro],
                                "text": cleaned_inner,
                            })
                        out.append(cleaned_inner)
                        i = close_idx + 1
                        continue
                if s.startswith("\\mDualStar", i):
                    annotations.append({"macro": "mDualStar", "type": MACRO_TYPES["mDualStar"], "text": "*"})
                    out.append("\\ast")
                    i += len("\\mDualStar")
                    continue
            out.append(s[i])
            i += 1
        return "".join(out)

    return walk(latex), annotations


def expression_records(text: str) -> list[dict[str, Any]]:
    records = []
    for start, end, source in math_spans(text):
        raw = text[start:end]
        clean, annotations = harvest_and_strip_macros(raw)
        clean = " ".join(clean.strip().split())
        if not clean:
            continue
        classified = proof_scope_audit.classify_expr(clean)
        gold_types = sorted({a["type"] for a in annotations})
        record = {
            "expr": clean,
            "raw-expr": raw,
            "position": start,
            "end": end,
            "type": classified,
            "source": source,
            "gold-types": gold_types,
            "gold-annotations": annotations,
        }
        records.append(record)
    return records


def _span(scope: dict[str, Any]) -> tuple[int, int]:
    content = scope.get("hx/content", {})
    start = content.get("position")
    end = content.get("end")
    if not isinstance(start, int):
        start = 0
    if not isinstance(end, int) or end <= start:
        end = start + len(str(content.get("match", "")))
    return start, end


def _symbols_in_expr(expr: str) -> set[str]:
    return {s for s in SYMBOL_RE.findall(expr) if s not in STOP_SYMBOLS and not s[0].isdigit()}


def _bound_symbols(scopes: list[dict[str, Any]]) -> set[str]:
    out = set()
    for scope in scopes:
        if str(scope.get("hx/type", "")).startswith("section/"):
            continue
        for end in scope.get("hx/ends", []):
            if end.get("role") == "symbol":
                value = end.get("latex") or end.get("text") or ""
                for sym in SYMBOL_RE.findall(value):
                    if sym not in STOP_SYMBOLS:
                        out.add(sym)
    return out


def audit_tex(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    entity_id = path.stem.replace("-solution-full", "")
    detector_scopes = proof_scope_audit.detect_scopes(entity_id, text)
    sections = section_scopes(entity_id, text)
    scopes = sections + detector_scopes
    expressions = expression_records(text)

    detector_spans = [(scope, *_span(scope)) for scope in detector_scopes]
    section_spans = [(scope, *_span(scope)) for scope in sections]
    paras = proof_scope_audit.paragraph_spans(text)

    def para_of(pos: int) -> int:
        if not paras:
            return 0
        lo, hi = 0, len(paras) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if paras[mid][1] < pos:
                lo = mid + 1
            else:
                hi = mid
        return lo

    binder_paras = {para_of(start) for _, start, _ in detector_spans}

    def in_body_section(pos: int) -> bool:
        return any(start <= pos < end and scope.get("hx/type") == "section/body"
                   for scope, start, end in section_spans)

    def grade(pos: int) -> str:
        if any(start <= pos < end for _, start, end in detector_spans):
            return "strict"
        if para_of(pos) in binder_paras or in_body_section(pos):
            return "weak"
        return "floating"

    expr_types = Counter()
    expr_grades = Counter()
    used_symbols = set()
    disagreements = []
    gold_total = 0
    gold_agree = 0
    for expr in expressions:
        expr["grade"] = grade(expr["position"])
        expr_types[expr["type"]] += 1
        expr_grades[expr["grade"]] += 1
        used_symbols.update(_symbols_in_expr(expr["expr"]))
        # Gold diff at TOKEN grain (review fix, fable-2): the \m* macros
        # annotate individual lexemes, while expr["type"] is the whole
        # expression's dominant class — comparing across that grain gave
        # 0% by construction. Classify each annotated token itself, within
        # the vocabulary the two channels share.
        comparable = {"number", "greek", "relation", "arrow",
                      "large-operator", "named-operator", "text"}
        for ann in expr.get("gold-annotations", []):
            if ann["type"] not in comparable:
                continue
            gold_total += 1
            token_type = proof_scope_audit.classify_expr(ann["text"])
            if token_type == ann["type"]:
                gold_agree += 1
            else:
                disagreements.append({
                    "token": ann["text"],
                    "in-expr": expr["expr"][:60],
                    "position": expr["position"],
                    "gold-type": ann["type"],
                    "classified-type": token_type,
                })

    bound_symbols = _bound_symbols(scopes)
    vacuous = []
    for scope, start, end in detector_spans:
        if not any(start <= expr["position"] < end for expr in expressions):
            vacuous.append({
                "scope-id": scope.get("hx/id"),
                "type": scope.get("hx/type"),
                "match": scope.get("hx/content", {}).get("match", "")[:120],
            })

    floating = [expr for expr in expressions if expr["grade"] == "floating"]
    return {
        "writeup": path.name,
        "register": "full-tex",
        "expr-count": len(expressions),
        "scope-count": len(scopes),
        "detector-scope-count": len(detector_scopes),
        "section-count": len(sections),
        "expr-types": dict(expr_types),
        "scope-grades": dict(expr_grades),
        "floating-expr-count": len(floating),
        "floating-expr-pct": (100.0 * len(floating) / len(expressions)) if expressions else 0.0,
        "bound-symbols": sorted(bound_symbols),
        "free-symbols": sorted(used_symbols - bound_symbols),
        "vacuous-scopes": vacuous,
        "vacuous-count": len(vacuous),
        "gold-annotated-count": gold_total,
        "gold-agree-count": gold_agree,
        "gold-agreement-rate": (100.0 * gold_agree / gold_total) if gold_total else 0.0,
        "gold-disagreements": disagreements,
        "scopes": scopes,
        "expressions": expressions,
    }


def tex_files(root: Path = FULL_TEX_DIR) -> list[Path]:
    return [root / f"problem{i}-solution-full.tex" for i in range(1, 11)]


def run_audit(root: Path = FULL_TEX_DIR) -> list[dict[str, Any]]:
    return [audit_tex(path) for path in tex_files(root) if path.exists()]


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    expr_total = sum(r["expr-count"] for r in results)
    scope_total = sum(r["scope-count"] for r in results)
    floating = sum(r["floating-expr-count"] for r in results)
    gold_total = sum(r["gold-annotated-count"] for r in results)
    gold_agree = sum(r["gold-agree-count"] for r in results)
    return {
        "proofs": len(results),
        "expr-total": expr_total,
        "scope-total": scope_total,
        "floating-expr-count": floating,
        "floating-expr-pct": (100.0 * floating / expr_total) if expr_total else 0.0,
        "vacuous-scope-count": sum(r["vacuous-count"] for r in results),
        "gold-annotated-count": gold_total,
        "gold-agree-count": gold_agree,
        "gold-agreement-rate": (100.0 * gold_agree / gold_total) if gold_total else 0.0,
        "per-proof": [
            {
                "writeup": r["writeup"],
                "expr-count": r["expr-count"],
                "scope-count": r["scope-count"],
                "floating-expr-pct": round(r["floating-expr-pct"], 1),
                "free-symbols": len(r["free-symbols"]),
                "vacuous-count": r["vacuous-count"],
                "gold-agreement-rate": round(r["gold-agreement-rate"], 1),
            }
            for r in results
        ],
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex-dir", type=Path, default=FULL_TEX_DIR)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    results = run_audit(args.tex_dir)
    summary = summarize(results)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("proof,exprs,scopes,floating%,gold-agree%,vacuous")
    for row in summary["per-proof"]:
        print(f"{row['writeup']},{row['expr-count']},{row['scope-count']},{row['floating-expr-pct']},{row['gold-agreement-rate']},{row['vacuous-count']}")
    print(f"TOTAL,{summary['expr-total']},{summary['scope-total']},{summary['floating-expr-pct']:.1f},{summary['gold-agreement-rate']:.1f},{summary['vacuous-scope-count']}")


if __name__ == "__main__":
    main()
