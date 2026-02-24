#!/usr/bin/env python3
"""Pilot distinctor-style analysis on sampled PlanetMath entries.

This script reuses scope detection and estimates where explicit distinctness
assumptions (x != y style) might be useful by examining co-bound symbols inside
math expressions.
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path


DEFAULT_DOMAINS = [
    "18",  # category theory / homological algebra
    "11",  # number theory
    "54",  # general topology
    "68",  # computer science
]

BINDERISH_PREFIXES = ("bind/", "quant/")
SINGLE_VAR_RE = re.compile(r"(?<![A-Za-z\\])([A-Za-z])(?![A-Za-z])")
PAIR_REL_RE = re.compile(
    r"(?<![A-Za-z\\])([A-Za-z])(?![A-Za-z])\s*(=|\\neq|\\ne|\\not=)\s*"
    r"(?<![A-Za-z\\])([A-Za-z])(?![A-Za-z])"
)
PLAIN_SYMBOL_LIST_RE = re.compile(r"^\(?\s*[A-Za-z](?:\s*,\s*[A-Za-z]){1,8}\s*\)?$")
RELATIONISH_RE = re.compile(
    r"(=|\\neq|\\ne|\\not=|\\in|\\subset|\\subseteq|\\to|\\rightarrow|\\mapsto|\\times|\\circ|\\hom)"
)
DISTINCTION_WORD_RE = re.compile(r"\b(distinct|different|unequal|not equal)\b", re.IGNORECASE)


def resolve_domain_edn(root: Path, domain: str) -> Path | None:
    cand = Path(domain)
    if cand.exists() and cand.suffix == ".edn":
        return cand

    if domain.isdigit():
        hits = sorted(root.glob(f"{domain}_*.edn"))
        return hits[0] if hits else None

    if domain.endswith(".edn"):
        p = root / domain
        return p if p.exists() else None

    hits = sorted(root.glob(f"{domain}*.edn"))
    return hits[0] if hits else None


def extract_math_expressions(text: str) -> list[dict]:
    out = []
    blocked = []

    for m in re.finditer(r"\$\$(.+?)\$\$", text, re.DOTALL):
        tex = m.group(1).strip()
        if not tex:
            continue
        blocked.append((m.start(), m.end()))
        out.append({"latex": tex, "position": m.start(1), "end": m.end(1)})

    for m in re.finditer(r"\\\[(.+?)\\\]", text, re.DOTALL):
        tex = m.group(1).strip()
        if not tex:
            continue
        blocked.append((m.start(), m.end()))
        out.append({"latex": tex, "position": m.start(1), "end": m.end(1)})

    for m in re.finditer(r"(?<!\$)\$([^$\n]+?)\$(?!\$)", text):
        if any(a <= m.start() < b for a, b in blocked):
            continue
        tex = m.group(1).strip()
        if not tex:
            continue
        out.append({"latex": tex, "position": m.start(1), "end": m.end(1)})

    for m in re.finditer(r"\\\((.+?)\\\)", text, re.DOTALL):
        tex = m.group(1).strip()
        if not tex:
            continue
        out.append({"latex": tex, "position": m.start(1), "end": m.end(1)})

    out.sort(key=lambda r: r["position"])
    return out


def parse_relation_pairs(tex: str) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    equal_pairs = set()
    neq_pairs = set()
    for m in PAIR_REL_RE.finditer(tex):
        a, op, b = m.group(1), m.group(2), m.group(3)
        if a == b:
            continue
        pair = tuple(sorted((a, b)))
        if op == "=":
            equal_pairs.add(pair)
        else:
            neq_pairs.add(pair)
    return equal_pairs, neq_pairs


def is_binderish(scope: dict) -> bool:
    stype = scope.get("hx/type", "")
    return isinstance(stype, str) and stype.startswith(BINDERISH_PREFIXES)


def _symbol_kind(tok: str) -> str:
    if tok.islower():
        return "lower"
    if tok.isupper():
        return "upper"
    return "other"


def _pair_compatible(a: str, b: str) -> bool:
    return _symbol_kind(a) == _symbol_kind(b) and _symbol_kind(a) != "other"


def _extract_symbols_from_field(raw: str) -> list[str]:
    if not raw:
        return []
    if len(raw) > 80:
        return []
    if RELATIONISH_RE.search(raw):
        # Looks like a full formula, not a declaration list.
        return []

    syms = [m.group(1) for m in SINGLE_VAR_RE.finditer(raw)]
    if len(syms) > 6:
        return []

    out = []
    seen = set()
    for s in syms:
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def extract_scope_symbols(scope: dict) -> list[str]:
    syms = []
    seen = set()
    for end in scope.get("hx/ends", []):
        if end.get("role") != "symbol":
            continue
        raw = (end.get("latex") or "").strip()
        for sym in _extract_symbols_from_field(raw):
            if sym in seen:
                continue
            seen.add(sym)
            syms.append(sym)
    return syms


def _scope_interval(scope: dict) -> tuple[int, int] | None:
    c = scope.get("hx/content", {})
    start = c.get("position")
    end = c.get("end")
    if isinstance(start, int) and isinstance(end, int) and end > start:
        return start, end
    return None


def _scope_expressions(scope: dict, exprs: list[dict]) -> list[dict]:
    interval = _scope_interval(scope)
    if not interval:
        return []
    a, b = interval
    return [e for e in exprs if not (e["end"] <= a or e["position"] >= b)]


def _pair_occurs_in_expr(pair: tuple[str, str], latex: str) -> bool:
    a, b = pair
    has_a = bool(re.search(rf"(?<![A-Za-z\\]){re.escape(a)}(?![A-Za-z])", latex))
    has_b = bool(re.search(rf"(?<![A-Za-z\\]){re.escape(b)}(?![A-Za-z])", latex))
    return has_a and has_b


def _is_nontrivial_pair_context(pair: tuple[str, str], latex: str) -> bool:
    if not _pair_occurs_in_expr(pair, latex):
        return False
    if PLAIN_SYMBOL_LIST_RE.fullmatch(latex.strip()):
        return False
    return bool(RELATIONISH_RE.search(latex))


def _best_supporting_expr(pair: tuple[str, str], exprs: list[dict]) -> dict | None:
    preferred = None
    fallback = None
    for ex in exprs:
        latex = ex["latex"]
        if not _pair_occurs_in_expr(pair, latex):
            continue
        if fallback is None:
            fallback = ex
        if _is_nontrivial_pair_context(pair, latex):
            preferred = ex
            break
    return preferred or fallback


def _context_excerpt(text: str, start: int | None, end: int | None, window: int = 220) -> str:
    if not isinstance(start, int) or not isinstance(end, int) or end <= start:
        return ""
    lo = max(0, start - window)
    hi = min(len(text), end + window)
    snippet = text[lo:hi]
    rel_start = max(0, start - lo)
    rel_end = max(rel_start, min(len(snippet), end - lo))
    marked = f"{snippet[:rel_start]}<<{snippet[rel_start:rel_end]}>>{snippet[rel_end:]}"
    marked = " ".join(marked.split())
    if lo > 0:
        marked = "... " + marked
    if hi < len(text):
        marked = marked + " ..."
    return marked


def _sym_pat(sym: str) -> str:
    return rf"(?<![A-Za-z\\]){re.escape(sym)}(?![A-Za-z])"


def _pair_relation_present(text: str, a: str, b: str, op_re: str) -> bool:
    if not text:
        return False
    ap = _sym_pat(a)
    bp = _sym_pat(b)
    pat = re.compile(rf"(?:{ap}\s*{op_re}\s*{bp}|{bp}\s*{op_re}\s*{ap})")
    return bool(pat.search(text))


def _pair_tokens_present(text: str, a: str, b: str) -> bool:
    if not text:
        return False
    return bool(re.search(_sym_pat(a), text) and re.search(_sym_pat(b), text))


def _pair_distinct_lexeme_present(text: str, a: str, b: str) -> bool:
    """Pair-specific distinctness phrasing (avoid broad false positives)."""
    if not text:
        return False
    ap = _sym_pat(a)
    bp = _sym_pat(b)
    patterns = [
        rf"{ap}\s*(?:,|and)\s*{bp}\s*(?:are\s+)?distinct",
        rf"{bp}\s*(?:,|and)\s*{ap}\s*(?:are\s+)?distinct",
        rf"distinct\s+{ap}\s*(?:and|,)\s*{bp}",
        rf"distinct\s+{bp}\s*(?:and|,)\s*{ap}",
    ]
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def assess_mit_for_hit(hit: dict) -> dict:
    """Machine pass: classify whether pair can be equal in this scope."""
    pair = hit.get("pair") or []
    if len(pair) != 2:
        return {
            "mit_decision": "unclear",
            "mit_label": "unclear",
            "mit_can_equal": None,
            "mit_confidence": 0.5,
            "mit_rationale": ["pair-missing"],
        }

    a, b = pair[0], pair[1]
    status = hit.get("status", "")
    scope_type = hit.get("scope_type", "")
    scope_match = hit.get("scope_match", "")
    support_latex = hit.get("support_latex", "")
    scope_context = hit.get("scope_context", "")
    support_context = hit.get("support_context", "")

    joined = " ".join([scope_match, support_latex, scope_context, support_context])
    lower_joined = joined.lower()

    if status == "explicit-distinct":
        return {
            "mit_decision": "must-distinct",
            "mit_label": "likely-distinctor",
            "mit_can_equal": False,
            "mit_confidence": 0.99,
            "mit_rationale": ["explicit-neq"],
        }
    if status == "explicit-equal":
        return {
            "mit_decision": "can-equal",
            "mit_label": "benign-cooccurrence",
            "mit_can_equal": True,
            "mit_confidence": 0.99,
            "mit_rationale": ["explicit-eq"],
        }

    distinct_score = 0
    can_score = 0
    reasons = []

    if _pair_relation_present(joined, a, b, r"(?:\\neq|\\ne|\\not=|!=|≠)"):
        distinct_score += 7
        reasons.append("pair-neq-relation")
    if _pair_relation_present(joined, a, b, r"(?:=)"):
        can_score += 6
        reasons.append("pair-eq-relation")

    if _pair_distinct_lexeme_present(joined, a, b):
        distinct_score += 3
        reasons.append("pair-distinct-lexeme")

    if re.search(rf"\\hom\s*\(\s*{_sym_pat(a)}\s*,\s*{_sym_pat(b)}\s*\)", support_latex):
        can_score += 4
        reasons.append("hom-parameterization")

    if "\\to" in support_latex or "\\rightarrow" in support_latex:
        if _pair_tokens_present(support_latex, a, b):
            can_score += 2
            reasons.append("typed-arrow-context")
    if "\\longrightarrow" in support_latex or "\\hookrightarrow" in support_latex or "\\mapsto" in support_latex:
        if _pair_tokens_present(support_latex, a, b):
            can_score += 2
            reasons.append("extended-arrow-context")

    if "(" in support_latex and "," in support_latex and ")" in support_latex and _pair_tokens_present(support_latex, a, b):
        can_score += 1
        reasons.append("tuple-context")

    if "=" in support_latex and _pair_tokens_present(support_latex, a, b):
        can_score += 2
        reasons.append("equation-with-pair")

    if any(op in support_latex for op in ("\\subset", "\\subseteq", "\\in", "\\cap", "\\cup")) and _pair_tokens_present(support_latex, a, b):
        can_score += 2
        reasons.append("set-relation-with-pair")

    if re.search(rf"{_sym_pat(a)}\s*\^\s*{_sym_pat(b)}|{_sym_pat(b)}\s*\^\s*{_sym_pat(a)}", support_latex):
        can_score += 2
        reasons.append("power-index-parameterization")
    if re.search(rf"_\{{[^}}]*{re.escape(a)}[^}}]*{re.escape(b)}[^}}]*\}}|_\{{[^}}]*{re.escape(b)}[^}}]*{re.escape(a)}[^}}]*\}}", support_latex):
        can_score += 2
        reasons.append("subscript-pair-indexing")

    if "ordered pair" in lower_joined and _pair_tokens_present(joined, a, b):
        can_score += 3
        reasons.append("ordered-pair-quantification")

    if "pair of adjacent" in lower_joined and _pair_tokens_present(joined, a, b):
        can_score += 2
        reasons.append("adjacent-pair-parameterization")

    if ("for any" in lower_joined or "for each" in lower_joined) and _pair_tokens_present(joined, a, b):
        can_score += 1
        reasons.append("universal-quantification")

    if scope_type in {"bind/typed", "bind/let", "quant/universal"}:
        can_score += 1
        reasons.append(f"scope-type:{scope_type}")

    if not scope_context and not support_context:
        reasons.append("missing-context")

    gap = can_score - distinct_score
    if distinct_score >= 4 and distinct_score >= can_score + 2:
        conf = min(0.98, 0.56 + 0.07 * distinct_score + 0.02 * max(0, distinct_score - can_score))
        return {
            "mit_decision": "must-distinct",
            "mit_label": "likely-distinctor",
            "mit_can_equal": False,
            "mit_confidence": round(conf, 3),
            "mit_rationale": reasons[:8],
        }
    if can_score >= 3 and can_score >= distinct_score + 1:
        conf = min(0.97, 0.52 + 0.06 * can_score + 0.015 * max(0, gap))
        return {
            "mit_decision": "can-equal",
            "mit_label": "benign-cooccurrence",
            "mit_can_equal": True,
            "mit_confidence": round(conf, 3),
            "mit_rationale": reasons[:8],
        }
    conf = min(0.79, 0.48 + 0.03 * max(can_score, distinct_score))
    return {
        "mit_decision": "unclear",
        "mit_label": "unclear",
        "mit_can_equal": None,
        "mit_confidence": round(conf, 3),
        "mit_rationale": reasons[:8],
    }


def pilot_domain(edn_path: Path, sample_size: int, seed: int, detect_scopes, pm_mod, max_hits: int = 120) -> dict:
    entries = pm_mod.load_edn(str(edn_path))
    tex_dir = edn_path.with_suffix("")
    if tex_dir.exists() and tex_dir.is_dir():
        tex_data = pm_mod.load_tex_dir(str(tex_dir))
        entries = pm_mod.merge_tex_bodies(entries, tex_data)

    rows = [e for e in entries if (e.get("body") or "").strip()]
    rnd = random.Random(seed)
    if sample_size > 0 and len(rows) > sample_size:
        rows = rnd.sample(rows, sample_size)

    total_entries = len(rows)
    entries_with_binder_scopes = 0
    total_binder_scopes = 0

    candidate_pairs = Counter()
    unresolved_pairs = Counter()
    explicit_equal_pairs = Counter()
    explicit_neq_pairs = Counter()
    shadow_rebinding_tokens = Counter()
    status_counts = Counter()

    example_rows = []
    hits = []

    for row in rows:
        entry_id = row.get("id", "?")
        title = row.get("title", "")
        text = row.get("body", "")

        scopes = detect_scopes(f"pm:{entry_id}", text)
        binder_scopes = [s for s in scopes if is_binderish(s)]
        if binder_scopes:
            entries_with_binder_scopes += 1
        total_binder_scopes += len(binder_scopes)

        exprs = extract_math_expressions(text)
        scope_membership = Counter()

        for scope in binder_scopes:
            scope_symbols = extract_scope_symbols(scope)
            for sym in scope_symbols:
                scope_membership[sym] += 1

            if len(scope_symbols) < 2:
                continue

            scope_exprs = _scope_expressions(scope, exprs)
            eq_pairs = set()
            neq_pairs = set()
            for expr in scope_exprs:
                e_eq, e_neq = parse_relation_pairs(expr["latex"])
                eq_pairs.update(e_eq)
                neq_pairs.update(e_neq)

            for a, b in itertools.combinations(sorted(scope_symbols), 2):
                if not _pair_compatible(a, b):
                    continue
                pair = tuple(sorted((a, b)))

                has_nontrivial_context = any(
                    _is_nontrivial_pair_context(pair, ex["latex"]) for ex in scope_exprs
                )
                if not has_nontrivial_context and pair not in eq_pairs and pair not in neq_pairs:
                    continue

                status = "unresolved"
                candidate_pairs[pair] += 1
                if pair in neq_pairs:
                    status = "explicit-distinct"
                    explicit_neq_pairs[pair] += 1
                elif pair in eq_pairs:
                    status = "explicit-equal"
                    explicit_equal_pairs[pair] += 1
                else:
                    unresolved_pairs[pair] += 1
                    if len(example_rows) < 20:
                        support_ex = _best_supporting_expr(pair, scope_exprs)
                        example_expr = support_ex["latex"] if support_ex else ""
                        example_rows.append({
                            "entry_id": entry_id,
                            "title": title,
                            "pair": list(pair),
                            "scope_type": scope.get("hx/type", "?"),
                            "latex": example_expr[:280],
                        })
                status_counts[status] += 1

                if len(hits) < max_hits:
                    support_ex = _best_supporting_expr(pair, scope_exprs)
                    c = scope.get("hx/content", {})
                    support_start = support_ex.get("position") if support_ex else None
                    support_end = support_ex.get("end") if support_ex else None
                    scope_start = c.get("position")
                    scope_end = c.get("end")
                    hits.append({
                        "hit_id": f"{entry_id}:{scope.get('hx/id', '')}:{pair[0]}:{pair[1]}:{len(hits) + 1}",
                        "status": status,
                        "entry_id": entry_id,
                        "title": title,
                        "pair": [pair[0], pair[1]],
                        "scope_id": scope.get("hx/id", ""),
                        "scope_type": scope.get("hx/type", ""),
                        "scope_start": scope_start,
                        "scope_end": scope_end,
                        "scope_match": (c.get("match") or "")[:140],
                        "scope_symbols": scope_symbols,
                        "support_latex": (support_ex["latex"] if support_ex else "")[:300],
                        "support_expr_start": support_start,
                        "support_expr_end": support_end,
                        "scope_context": _context_excerpt(text, scope_start, scope_end),
                        "support_context": _context_excerpt(text, support_start, support_end),
                        "has_nontrivial_context": bool(
                            support_ex and _is_nontrivial_pair_context(pair, support_ex["latex"])
                        ),
                    })
                    hits[-1].update(assess_mit_for_hit(hits[-1]))

        for tok, n in scope_membership.items():
            if n > 1:
                shadow_rebinding_tokens[tok] += n

    total_candidate = sum(candidate_pairs.values())
    total_unresolved = sum(unresolved_pairs.values())
    total_eq = sum(explicit_equal_pairs.values())
    total_neq = sum(explicit_neq_pairs.values())
    mit_counts = Counter(h.get("mit_label", "unclear") for h in hits)
    mit_avg_conf = sum(float(h.get("mit_confidence", 0.0)) for h in hits) / len(hits) if hits else 0.0

    return {
        "domain": edn_path.stem,
        "edn_path": str(edn_path),
        "sampled_entries": total_entries,
        "entries_with_binder_scopes": entries_with_binder_scopes,
        "entry_binder_coverage": round(entries_with_binder_scopes / total_entries, 4) if total_entries else 0.0,
        "total_binder_scopes": total_binder_scopes,
        "candidate_pair_events": total_candidate,
        "unresolved_pair_events": total_unresolved,
        "explicit_equal_pair_events": total_eq,
        "explicit_distinct_pair_events": total_neq,
        "unresolved_ratio": round(total_unresolved / total_candidate, 4) if total_candidate else 0.0,
        "top_candidate_pairs": [[a, b, n] for (a, b), n in candidate_pairs.most_common(20)],
        "top_unresolved_pairs": [[a, b, n] for (a, b), n in unresolved_pairs.most_common(20)],
        "top_explicit_equal_pairs": [[a, b, n] for (a, b), n in explicit_equal_pairs.most_common(20)],
        "top_explicit_distinct_pairs": [[a, b, n] for (a, b), n in explicit_neq_pairs.most_common(20)],
        "top_shadow_rebinding_tokens": shadow_rebinding_tokens.most_common(20),
        "hit_status_counts": dict(status_counts),
        "mit_counts": dict(mit_counts),
        "mit_avg_confidence": round(mit_avg_conf, 4),
        "examples": example_rows,
        "hits": hits,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Pilot distinctor-style analysis on PlanetMath sample")
    parser.add_argument("--planetmath-root", default="~/code/planetmath",
                        help="PlanetMath root with *.edn domain files")
    parser.add_argument("--domains", nargs="*", default=DEFAULT_DOMAINS,
                        help="Domain prefixes or .edn paths")
    parser.add_argument("--sample-per-domain", type=int, default=80,
                        help="Entries sampled per domain (0 = full domain)")
    parser.add_argument("--seed", type=int, default=13,
                        help="Random seed")
    parser.add_argument("--out", default="data/ct-validation/planetmath-distinctor-pilot.json",
                        help="Output report path")
    parser.add_argument("--hits-out", default="data/ct-validation/planetmath-distinctor-pilot-hits.jsonl",
                        help="Output JSONL path for inspectable candidate hits")
    parser.add_argument("--max-hits-per-domain", type=int, default=200,
                        help="Maximum hit rows captured per domain")
    parser.add_argument("--mit-out", default="data/ct-validation/planetmath-distinctor-mit-findings.json",
                        help="Output JSON report for machine MIT findings")
    parser.add_argument("--mit-md-out", default="data/ct-validation/planetmath-distinctor-mit-findings.md",
                        help="Output markdown report for machine MIT findings")
    args = parser.parse_args()

    root = Path(args.planetmath_root).expanduser()
    if not root.exists():
        print(f"PlanetMath root not found: {root}")
        return 1

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    detect_scopes = importlib.import_module("nlab-wiring").detect_scopes

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    pm_mod = importlib.import_module("futon6.planetmath")

    reports = []
    missing = []
    for i, dom in enumerate(args.domains):
        edn_path = resolve_domain_edn(root, dom)
        if not edn_path:
            missing.append(dom)
            continue
        rep = pilot_domain(
            edn_path=edn_path,
            sample_size=args.sample_per_domain,
            seed=args.seed + i,
            detect_scopes=detect_scopes,
            pm_mod=pm_mod,
            max_hits=args.max_hits_per_domain,
        )
        reports.append(rep)

    if not reports:
        print("No domains resolved; nothing to process.")
        if missing:
            print("Missing:", ", ".join(missing))
        return 1

    agg_candidate = sum(r["candidate_pair_events"] for r in reports)
    agg_unresolved = sum(r["unresolved_pair_events"] for r in reports)
    agg_eq = sum(r["explicit_equal_pair_events"] for r in reports)
    agg_neq = sum(r["explicit_distinct_pair_events"] for r in reports)
    agg_entries = sum(r["sampled_entries"] for r in reports)
    agg_binder_entries = sum(r["entries_with_binder_scopes"] for r in reports)
    agg_scopes = sum(r["total_binder_scopes"] for r in reports)

    out = {
        "planetmath_root": str(root),
        "sample_per_domain": args.sample_per_domain,
        "seed": args.seed,
        "missing_domains": missing,
        "aggregate": {
            "domains": [r["domain"] for r in reports],
            "sampled_entries": agg_entries,
            "entries_with_binder_scopes": agg_binder_entries,
            "entry_binder_coverage": round(agg_binder_entries / agg_entries, 4) if agg_entries else 0.0,
            "total_binder_scopes": agg_scopes,
            "candidate_pair_events": agg_candidate,
            "unresolved_pair_events": agg_unresolved,
            "explicit_equal_pair_events": agg_eq,
            "explicit_distinct_pair_events": agg_neq,
            "unresolved_ratio": round(agg_unresolved / agg_candidate, 4) if agg_candidate else 0.0,
        },
        "domains": reports,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    hits_out = Path(args.hits_out)
    hits_out.parent.mkdir(parents=True, exist_ok=True)
    all_hits = []
    with hits_out.open("w", encoding="utf-8") as f:
        for rep in reports:
            domain = rep["domain"]
            for hit in rep.get("hits", []):
                row = {"domain": domain, **hit}
                all_hits.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    mit_counts = Counter(h.get("mit_label", "unclear") for h in all_hits)
    mit_avg_conf = (
        sum(float(h.get("mit_confidence", 0.0)) for h in all_hits) / len(all_hits)
        if all_hits else 0.0
    )
    out["aggregate"]["mit_counts"] = dict(mit_counts)
    out["aggregate"]["mit_avg_confidence"] = round(mit_avg_conf, 4)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    mit_out_obj = {
        "generated_from": str(hits_out),
        "total_hits": len(all_hits),
        "mit_counts": dict(mit_counts),
        "mit_avg_confidence": round(mit_avg_conf, 4),
        "top_likely_distinctor": [
            h for h in all_hits if h.get("mit_label") == "likely-distinctor"
        ][:60],
        "top_unclear": [
            h for h in all_hits if h.get("mit_label") == "unclear"
        ][:60],
    }
    mit_out = Path(args.mit_out)
    mit_out.parent.mkdir(parents=True, exist_ok=True)
    mit_out.write_text(json.dumps(mit_out_obj, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = [
        "# PlanetMath Distinctor MIT Findings",
        "",
        f"- total_hits: {len(all_hits)}",
        f"- benign-cooccurrence: {mit_counts.get('benign-cooccurrence', 0)}",
        f"- likely-distinctor: {mit_counts.get('likely-distinctor', 0)}",
        f"- unclear: {mit_counts.get('unclear', 0)}",
        f"- avg_confidence: {round(mit_avg_conf, 4)}",
        "",
        "## Likely Distinctor Candidates",
    ]
    likely_rows = [h for h in all_hits if h.get("mit_label") == "likely-distinctor"]
    if likely_rows:
        for i, h in enumerate(likely_rows[:50], start=1):
            md_lines.append(
                f"{i}. {h.get('domain', '')} :: {h.get('entry_id', '')} "
                f"pair=({', '.join(h.get('pair', []))}) "
                f"scope={h.get('scope_type', '')} "
                f"conf={h.get('mit_confidence', 0.0)} "
                f"rationale={','.join(h.get('mit_rationale', []))}"
            )
    else:
        md_lines.append("- none")
    md_lines.append("")
    md_lines.append("## Unclear Candidates")
    unclear_rows = [h for h in all_hits if h.get("mit_label") == "unclear"]
    if unclear_rows:
        for i, h in enumerate(unclear_rows[:50], start=1):
            md_lines.append(
                f"{i}. {h.get('domain', '')} :: {h.get('entry_id', '')} "
                f"pair=({', '.join(h.get('pair', []))}) "
                f"scope={h.get('scope_type', '')} "
                f"conf={h.get('mit_confidence', 0.0)} "
                f"rationale={','.join(h.get('mit_rationale', []))}"
            )
    else:
        md_lines.append("- none")
    md_out = Path(args.mit_md_out)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[pilot] wrote {out_path}")
    print(f"[pilot] wrote {hits_out}")
    print(f"[pilot] wrote {mit_out}")
    print(f"[pilot] wrote {md_out}")
    print("[pilot] aggregate:", json.dumps(out["aggregate"], ensure_ascii=False))
    print("[pilot] mit:", json.dumps({
        "total_hits": len(all_hits),
        "mit_counts": dict(mit_counts),
        "mit_avg_confidence": round(mit_avg_conf, 4),
    }, ensure_ascii=False))
    for rep in reports:
        print(
            f"[pilot] {rep['domain']}: entries={rep['sampled_entries']} "
            f"binder_cov={rep['entry_binder_coverage']:.3f} "
            f"candidate={rep['candidate_pair_events']} "
            f"unresolved={rep['unresolved_pair_events']} "
            f"neq={rep['explicit_distinct_pair_events']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
