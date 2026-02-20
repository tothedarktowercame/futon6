#!/usr/bin/env python3
"""Scope-aware PlanetMath sample audit across multiple domains.

Evaluates scope coverage on sampled entries by measuring whether variable-like
tokens inside math expressions fall within at least one detected scope span.
"""

from __future__ import annotations

import argparse
import importlib
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

GREEK_OR_SYMBOL_COMMANDS = {
    "alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", "zeta",
    "eta", "theta", "vartheta", "iota", "kappa", "lambda", "mu", "nu",
    "xi", "pi", "varpi", "rho", "varrho", "sigma", "varsigma", "tau",
    "upsilon", "phi", "varphi", "chi", "psi", "omega",
    "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Upsilon", "Phi", "Psi", "Omega",
}


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
        out.append({"latex": tex, "position": m.start(1)})

    for m in re.finditer(r"\\\[(.+?)\\\]", text, re.DOTALL):
        tex = m.group(1).strip()
        if not tex:
            continue
        blocked.append((m.start(), m.end()))
        out.append({"latex": tex, "position": m.start(1)})

    for m in re.finditer(r"(?<!\$)\$([^$\n]+?)\$(?!\$)", text):
        if any(a <= m.start() < b for a, b in blocked):
            continue
        tex = m.group(1).strip()
        if not tex:
            continue
        out.append({"latex": tex, "position": m.start(1)})

    for m in re.finditer(r"\\\((.+?)\\\)", text, re.DOTALL):
        tex = m.group(1).strip()
        if not tex:
            continue
        out.append({"latex": tex, "position": m.start(1)})

    out.sort(key=lambda r: r["position"])
    return out


def _overlaps_ranges(start: int, end: int, ranges: list[tuple[int, int]]) -> bool:
    return any(not (end <= a or start >= b) for a, b in ranges)


def _find_nonsemantic_exclusion_ranges(tex: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for m in re.finditer(r"\\begin\{(?:array|tabular|tabularx)\}\{([^}]*)\}", tex):
        spec = m.group(1)
        if len(spec) <= 48:
            ranges.append((m.start(1), m.end(1)))
    return ranges


def find_variable_spans(tex: str) -> list[dict]:
    spans = []
    excluded_ranges = _find_nonsemantic_exclusion_ranges(tex)

    for m in re.finditer(r"\\mathcal\{([A-Za-z])\}", tex):
        spans.append({"token": f"\\mathcal{{{m.group(1)}}}", "start": m.start(), "end": m.end()})

    for m in re.finditer(r"\\([A-Za-z]+)", tex):
        cmd = m.group(1)
        if cmd not in GREEK_OR_SYMBOL_COMMANDS:
            continue
        if _overlaps_ranges(m.start(), m.end(), excluded_ranges):
            continue
        spans.append({"token": f"\\{cmd}", "start": m.start(), "end": m.end()})

    for m in re.finditer(r"(?<![A-Za-z\\])([A-Za-z])(?![A-Za-z])", tex):
        s, e = m.start(1), m.end(1)
        if _overlaps_ranges(s, e, excluded_ranges):
            continue
        spans.append({"token": m.group(1), "start": s, "end": e})

    spans.sort(key=lambda d: (d["start"], d["end"]))

    dedup = []
    seen = set()
    for sp in spans:
        k = (sp["start"], sp["end"], sp["token"])
        if k in seen:
            continue
        seen.add(k)
        dedup.append(sp)
    return dedup


def build_scope_ranges(scopes: list[dict]) -> list[tuple[int, int]]:
    ranges = []
    for s in scopes:
        c = s.get("hx/content", {})
        pos = c.get("position")
        if not isinstance(pos, int):
            continue
        end = c.get("end")
        if not isinstance(end, int):
            match = c.get("match", "")
            if not match:
                continue
            end = pos + len(match)
        if end > pos:
            ranges.append((pos, end))
    ranges.sort()
    return ranges


def is_pos_in_any_scope(pos: int, ranges: list[tuple[int, int]]) -> bool:
    return any(a <= pos < b for a, b in ranges)


def audit_domain(edn_path: Path, sample_size: int, seed: int, detect_scopes, pm_mod) -> dict:
    domain_name = edn_path.stem
    tex_dir = edn_path.with_suffix("")

    entries = pm_mod.load_edn(str(edn_path))
    if tex_dir.exists() and tex_dir.is_dir():
        tex_data = pm_mod.load_tex_dir(str(tex_dir))
        entries = pm_mod.merge_tex_bodies(entries, tex_data)

    rows = [e for e in entries if (e.get("body") or "").strip()]
    rnd = random.Random(seed)
    if sample_size > 0 and len(rows) > sample_size:
        rows = rnd.sample(rows, sample_size)

    scope_type_counts = Counter()
    unscoped_token_counts = Counter()

    entry_count = len(rows)
    entries_with_scopes = 0
    total_scopes = 0
    expr_count = 0
    total_vars = 0
    scoped_vars = 0

    worst_entries = []

    for row in rows:
        eid = row.get("id", "?")
        text = row.get("body", "")
        scopes = detect_scopes(f"pm:{eid}", text)
        if scopes:
            entries_with_scopes += 1
            total_scopes += len(scopes)
        for s in scopes:
            scope_type_counts[s.get("hx/type", "?")] += 1

        scope_ranges = build_scope_ranges(scopes)
        entry_total_vars = 0
        entry_scoped_vars = 0
        entry_unscoped = Counter()

        for expr in extract_math_expressions(text):
            expr_count += 1
            latex = expr["latex"]
            expr_pos = expr["position"]
            for sp in find_variable_spans(latex):
                total_vars += 1
                entry_total_vars += 1
                gpos = expr_pos + sp["start"]
                if is_pos_in_any_scope(gpos, scope_ranges):
                    scoped_vars += 1
                    entry_scoped_vars += 1
                else:
                    unscoped_token_counts[sp["token"]] += 1
                    entry_unscoped[sp["token"]] += 1

        unscoped_n = entry_total_vars - entry_scoped_vars
        if unscoped_n > 0:
            worst_entries.append({
                "id": eid,
                "title": row.get("title", ""),
                "unscoped_vars": unscoped_n,
                "total_vars": entry_total_vars,
                "top_unscoped": entry_unscoped.most_common(8),
            })

    worst_entries.sort(key=lambda x: (-x["unscoped_vars"], -x["total_vars"], x["id"]))

    return {
        "domain": domain_name,
        "edn_path": str(edn_path),
        "sampled_entries": entry_count,
        "entries_with_scopes": entries_with_scopes,
        "entry_scope_coverage": round(entries_with_scopes / entry_count, 4) if entry_count else 0.0,
        "total_scopes": total_scopes,
        "scope_types_top": scope_type_counts.most_common(12),
        "expr_count": expr_count,
        "total_vars": total_vars,
        "scoped_vars": scoped_vars,
        "unscoped_vars": total_vars - scoped_vars,
        "var_scope_coverage": round(scoped_vars / total_vars, 4) if total_vars else 1.0,
        "unscoped_tokens_top": unscoped_token_counts.most_common(20),
        "worst_entries": worst_entries[:12],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit scope-aware variable coverage on PlanetMath samples")
    parser.add_argument("--planetmath-root", default="~/code/planetmath",
                        help="PlanetMath root with *.edn domain files")
    parser.add_argument("--domains", nargs="*", default=DEFAULT_DOMAINS,
                        help="Domain prefixes or .edn paths (default: 18 11 54 68)")
    parser.add_argument("--sample-per-domain", type=int, default=120,
                        help="Entries sampled per domain (0 = full domain)")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed for deterministic sampling")
    parser.add_argument("--out", default="data/ct-validation/planetmath-scope-sample.json",
                        help="Output report JSON path")
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
        rep = audit_domain(
            edn_path=edn_path,
            sample_size=args.sample_per_domain,
            seed=args.seed + i,
            detect_scopes=detect_scopes,
            pm_mod=pm_mod,
        )
        reports.append(rep)

    if not reports:
        print("No domains resolved; nothing to audit.")
        if missing:
            print("Missing:", ", ".join(missing))
        return 1

    agg_total_vars = sum(r["total_vars"] for r in reports)
    agg_scoped_vars = sum(r["scoped_vars"] for r in reports)
    agg_entries = sum(r["sampled_entries"] for r in reports)
    agg_entries_with_scopes = sum(r["entries_with_scopes"] for r in reports)

    aggregate = {
        "domains": [r["domain"] for r in reports],
        "sampled_entries": agg_entries,
        "entries_with_scopes": agg_entries_with_scopes,
        "entry_scope_coverage": round(agg_entries_with_scopes / agg_entries, 4) if agg_entries else 0.0,
        "total_vars": agg_total_vars,
        "scoped_vars": agg_scoped_vars,
        "unscoped_vars": agg_total_vars - agg_scoped_vars,
        "var_scope_coverage": round(agg_scoped_vars / agg_total_vars, 4) if agg_total_vars else 1.0,
    }

    out = {
        "planetmath_root": str(root),
        "sample_per_domain": args.sample_per_domain,
        "seed": args.seed,
        "missing_domains": missing,
        "aggregate": aggregate,
        "domains": reports,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[audit] wrote {out_path}")
    print("[audit] aggregate:", json.dumps(aggregate, ensure_ascii=False))
    for rep in reports:
        print(
            f"[audit] {rep['domain']}: entries={rep['sampled_entries']} "
            f"entry_cov={rep['entry_scope_coverage']:.3f} "
            f"var_cov={rep['var_scope_coverage']:.3f} "
            f"vars={rep['scoped_vars']}/{rep['total_vars']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
