#!/usr/bin/env python3
"""Audit the APM informal-proof register.

The APM register is markdown prose with Unicode mathematics rather than the
First Proof ASCII/TeX registers. This extends the existing proof instruments:
expression typing still goes through nlab_skolem_audit.classify_expr, scope
records use the same hx/ shape, and external concept resolution reuses
proof_scope_audit's background corpus index.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import proof_scope_audit as psa
from nlab_skolem_audit import classify_expr, paragraph_spans

ROOT = Path(__file__).resolve().parent.parent
APM_PROBLEMS = Path("/home/joe/code/apm-lean/problems")
APM_LEAN = Path("/home/joe/code/apm-lean/lean-proofs")
OUT_JSON = ROOT / "data" / "apm-proof-scope-audit.json"
OUT_SUMMARY = ROOT / "data" / "apm-proof-scope-summary.json"

MATH_SPAN_RE = re.compile(r"\$\$?(?P<body>[^$]{1,1000})\$\$?|\\\((?P<paren>.*?)\\\)", re.S)
UNICODE_EXPR_RE = re.compile(
    r"[^.\n;]{0,80}[‖∫∑∏→←↦⇒≤≥≠≈∼≃≅∈⊂⊆⊃⊇∞αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΦΨΩ][^.\n;]{0,120}"
)
UNICODE_SYMBOL_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*|[αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΦΨΩ]")
BOLD_ENV_RE = re.compile(
    r"\*\*(Definition|Claim|Lemma|Theorem|Proposition|Corollary|Proof|Upper bound|Lower bound|Step\s+[^*.]+|[^*\n]{1,80})\.?:?\*\*",
    re.I,
)
HEADING_RE = re.compile(r"(?m)^#\s+(.+)$")
UNICODE_BINDER_RE = re.compile(
    r"\b(Let|Fix)\s+([^\n.]{1,120}?(?:∈|≤|≥|<|>|=)[^\n.]{0,160})",
    re.I,
)

ENV_KIND = {
    "definition": "env/definition",
    "claim": "env/lemma",
    "lemma": "env/lemma",
    "theorem": "env/theorem",
    "proposition": "env/proposition",
    "corollary": "env/corollary",
    "proof": "env/proof",
    "upper bound": "env/lemma",
    "lower bound": "env/lemma",
}


def problem_files(root: Path = APM_PROBLEMS) -> list[Path]:
    return sorted(root.glob("*/informal-solution.md"), key=lambda p: p.parent.name)


def problem_id(path: Path) -> str:
    return path.parent.name


def _mathy(expr: str) -> bool:
    return bool(re.search(r"[=<>≤≥≠∈⊂⊆→←↦⇒‖∫∑∏∞]|[αβγδεζηθικλμνξπρστυφχψω]", expr))


def expression_records(text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()

    def add(expr: str, pos: int, source: str) -> None:
        expr = " ".join(expr.strip().split())
        if not expr or not _mathy(expr):
            return
        key = (pos, expr)
        if key in seen:
            return
        seen.add(key)
        records.append({"expr": expr, "position": pos, "type": classify_expr(expr), "source": source})

    for m in MATH_SPAN_RE.finditer(text):
        body = m.group("body") if m.group("body") is not None else m.group("paren")
        add(body or "", m.start(), "markdown-math")
    for m in UNICODE_EXPR_RE.finditer(text):
        add(m.group(), m.start(), "unicode-inline")
    if not records:
        # Some imported APM files are register stubs with no worked proof yet.
        # Keep the corpus row live without pretending this is math: the source
        # and type mark it as register metadata, not field content.
        if m := HEADING_RE.search(text):
            records.append({"expr": m.group(1).strip(), "position": m.start(),
                            "type": "text", "source": "register-heading"})
    return sorted(records, key=lambda r: (r["position"], r["expr"]))


def _env_type(label: str) -> str:
    clean = label.lower().rstrip(".:")
    if clean.startswith("step"):
        return "env/proof-step"
    return ENV_KIND.get(clean, "env/proof-step")


def bold_environment_scopes(entity_id: str, text: str) -> list[dict[str, Any]]:
    matches = list(BOLD_ENV_RE.finditer(text))
    scopes = []
    for idx, m in enumerate(matches):
        label = m.group(1).strip().rstrip(":.")
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        scopes.append({
            "hx/id": f"{entity_id}:bold-env-{idx:03d}",
            "hx/role": "component",
            "hx/type": _env_type(label),
            "hx/parent": None,
            "hx/ends": [
                {"role": "entity", "ident": entity_id},
                {"role": "environment", "name": label},
            ],
            "hx/content": {"match": m.group(), "position": m.start(), "end": end},
            "hx/labels": ["scope", "bold-environment", label.lower()],
        })
    return scopes


def _first_symbol(text: str) -> str | None:
    m = UNICODE_SYMBOL_RE.search(text)
    return m.group() if m else None


def unicode_binder_scopes(entity_id: str, text: str, start_idx: int = 0) -> list[dict[str, Any]]:
    scopes = []
    for idx, m in enumerate(UNICODE_BINDER_RE.finditer(text), start_idx):
        body = m.group(2).strip()
        sym = _first_symbol(body)
        ends = [{"role": "entity", "ident": entity_id},
                {"role": "binder", "text": m.group(1)}]
        if sym:
            ends.append({"role": "symbol", "text": sym})
        ends.append({"role": "condition", "text": body[:160]})
        scopes.append({
            "hx/id": f"{entity_id}:unicode-bind-{idx:03d}",
            "hx/role": "component",
            "hx/type": "bind/let",
            "hx/parent": None,
            "hx/ends": ends,
            "hx/content": {"match": m.group()[:120], "position": m.start(), "end": _sentence_end(text, m.end())},
            "hx/labels": ["scope", "unicode-binder"],
        })
    return scopes


def _sentence_end(text: str, pos: int) -> int:
    m = re.search(r"[.\n]", text[pos:])
    return pos + m.end() if m else len(text)


def detect_apm_scopes(entity_id: str, text: str) -> list[dict[str, Any]]:
    scopes = psa.detect_scopes(entity_id, text)
    scopes.extend(bold_environment_scopes(entity_id, text))
    scopes.extend(unicode_binder_scopes(entity_id, text, len(scopes)))
    if not scopes:
        if m := HEADING_RE.search(text):
            scopes.append({
                "hx/id": f"{entity_id}:register-heading",
                "hx/role": "component",
                "hx/type": "env/register-heading",
                "hx/parent": None,
                "hx/ends": [
                    {"role": "entity", "ident": entity_id},
                    {"role": "environment", "name": "register-heading"},
                ],
                "hx/content": {"match": m.group(), "position": m.start(), "end": _sentence_end(text, m.end())},
                "hx/labels": ["scope", "register-heading"],
            })
    return sorted(scopes, key=lambda s: s.get("hx/content", {}).get("position", 0))


def _span(scope: dict[str, Any]) -> tuple[int, int]:
    c = scope.get("hx/content", {})
    start = c.get("position") if isinstance(c.get("position"), int) else 0
    end = c.get("end") if isinstance(c.get("end"), int) else start + len(str(c.get("match", "")))
    return start, max(end, start)


def _symbols_in_expr(expr: str) -> set[str]:
    return {s for s in UNICODE_SYMBOL_RE.findall(expr) if s not in psa.STOP_SYMBOLS and not s[0].isdigit()}


def lean_status(pid: str, lean_root: Path = APM_LEAN) -> dict[str, Any]:
    path = lean_root / pid / "Main.lean"
    if not path.exists():
        return {"status": "no-lean", "path": str(path), "sorry-count": None, "sorry-lines": []}
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = [i for i, line in enumerate(text.splitlines(), 1) if "sorry" in line]
    return {
        "status": "sorry-free" if not lines else "sorry-carrying",
        "path": str(path),
        "sorry-count": len(lines),
        "sorry-lines": lines,
    }


def audit_apm(path: Path, background_index: dict[str, Any] | None = None) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    pid = problem_id(path)
    scopes = detect_apm_scopes(pid, text)
    exprs = expression_records(text)
    scope_spans = [(s, *_span(s)) for s in scopes]
    paras = paragraph_spans(text)

    def para_of(pos: int) -> int:
        for i, (_s, e) in enumerate(paras):
            if pos <= e:
                return i
        return max(0, len(paras) - 1)

    binder_paras = {para_of(start) for _, start, _ in scope_spans}

    def grade(pos: int) -> str:
        if any(start <= pos < end and str(scope.get("hx/type", "")).startswith("env/")
               for scope, start, end in scope_spans):
            return "strict"
        if any(start <= pos < end for _, start, end in scope_spans) or para_of(pos) in binder_paras:
            return "weak"
        return "floating"

    expr_types = Counter()
    expr_grades = Counter()
    used_symbols = set()
    for expr in exprs:
        expr["grade"] = grade(expr["position"])
        expr_types[expr["type"]] += 1
        expr_grades[expr["grade"]] += 1
        used_symbols.update(_symbols_in_expr(expr["expr"]))

    bound = psa._bound_symbols(scopes)
    free = used_symbols - bound
    concepts = psa.concept_terms(text, free)
    if background_index is None:
        background_index = psa.load_background_index(concepts)
    external, orphan = psa.resolve_concepts(concepts, background_index)
    vacuous = []
    for scope, start, end in scope_spans:
        if not any(start <= expr["position"] < end for expr in exprs):
            vacuous.append({
                "scope-id": scope.get("hx/id"),
                "type": scope.get("hx/type"),
                "match": scope.get("hx/content", {}).get("match", "")[:120],
            })
    floating = [e for e in exprs if e["grade"] == "floating"]
    return {
        "problem": pid,
        "writeup": str(path),
        "register": "apm-informal",
        "expr-count": len(exprs),
        "scope-count": len(scopes),
        "expr-types": dict(expr_types),
        "scope-grades": dict(expr_grades),
        "floating-expr-count": len(floating),
        "floating-expr-pct": (100.0 * len(floating) / len(exprs)) if exprs else 0.0,
        "bound-symbols": sorted(bound),
        "free-symbols": sorted(free),
        "candidate-concepts": concepts,
        "external-concepts": external,
        "orphan-concepts": orphan,
        "externally-bound-count": len(external),
        "orphan-count": len(orphan),
        "vacuous-scopes": vacuous,
        "vacuous-count": len(vacuous),
        "lean": lean_status(pid),
        "scopes": scopes,
        "expressions": exprs,
    }


def run_audit(root: Path = APM_PROBLEMS) -> list[dict[str, Any]]:
    paths = problem_files(root)
    seed: list[str] = []
    for p in paths:
        text = p.read_text(encoding="utf-8", errors="ignore")
        scopes = detect_apm_scopes(problem_id(p), text)
        exprs = expression_records(text)
        bound = psa._bound_symbols(scopes)
        used = set()
        for expr in exprs:
            used.update(_symbols_in_expr(expr["expr"]))
        seed.extend(psa.concept_terms(text, used - bound))
    index = psa.load_background_index(seed)
    return [audit_apm(path, index) for path in paths]


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    expr_total = sum(r["expr-count"] for r in results)
    floating = sum(r["floating-expr-count"] for r in results)
    label_counts = Counter(r["lean"]["status"] for r in results)
    return {
        "problems": len(results),
        "expr-total": expr_total,
        "scope-total": sum(r["scope-count"] for r in results),
        "floating-expr-count": floating,
        "floating-expr-pct": (100.0 * floating / expr_total) if expr_total else 0.0,
        "externally-bound-count": sum(r["externally-bound-count"] for r in results),
        "orphan-count": sum(r["orphan-count"] for r in results),
        "lean-label-counts": dict(label_counts),
        "per-problem": [
            {"problem": r["problem"],
             "expr-count": r["expr-count"],
             "scope-count": r["scope-count"],
             "floating-expr-pct": round(r["floating-expr-pct"], 1),
             "free-symbols": len(r["free-symbols"]),
             "externally-bound": r["externally-bound-count"],
             "orphan": r["orphan-count"],
             "lean-status": r["lean"]["status"],
             "sorry-count": r["lean"].get("sorry-count")}
            for r in results
        ],
    }


def print_table(summary: dict[str, Any]) -> None:
    print("problem,exprs,scopes,floating%,free-symbols,externally-bound,orphan,lean-status,sorries")
    for row in summary["per-problem"]:
        print(f"{row['problem']},{row['expr-count']},{row['scope-count']},{row['floating-expr-pct']},{row['free-symbols']},{row['externally-bound']},{row['orphan']},{row['lean-status']},{row['sorry-count']}")
    print(f"TOTAL,{summary['expr-total']},{summary['scope-total']},{summary['floating-expr-pct']:.1f},,{summary['externally-bound-count']},{summary['orphan-count']},,")
    print(f"Lean labels: {summary['lean-label-counts']}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apm-root", type=Path, default=APM_PROBLEMS)
    ap.add_argument("--json", type=Path, default=OUT_JSON)
    ap.add_argument("--summary-json", type=Path, default=OUT_SUMMARY)
    args = ap.parse_args(argv)
    results = run_audit(args.apm_root)
    summary = summarize(results)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(results, indent=1), encoding="utf-8")
    args.summary_json.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print_table(summary)


if __name__ == "__main__":
    main()
