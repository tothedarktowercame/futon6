#!/usr/bin/env python3
"""Structure-first concept definition coverage.

Invert the warp concept-usage index into a de-noised, corpus-wide concept
ranking, then measure whether the top-N genuine concepts have definition
evidence in the existing definition substrates.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import build_term_prior  # noqa: E402

DEFAULT_USAGE = ROOT / "data" / "warp" / "concept-usage.json"
DEFAULT_SNIPPETS = ROOT / "data" / "warp" / "def-snippets.json"
DEFAULT_DEFINED = ROOT / "data" / "warp" / "defined-index.json"
DEFAULT_ENCYCLOPEDIA = ROOT / "data" / "concept-encyclopedia-ct.json"
DEFAULT_GRAPH = ROOT / "data" / "warp" / "concept-graph.json"
DEFAULT_REPORT = ROOT / "holes" / "excursions" / "sfc-concept-coverage.md"

GENERIC_PHRASES = {
    "all morphisms",
    "all objects",
    "any object",
    "any two",
    "category theoretic",
    "category underlying",
    "closed under",
    "cm cm",
    "each other",
    "finitely many",
    "left hand side",
    "main theorem",
    "more generally",
    "non empty",
    "non negative",
    "non trivial",
    "non zero",
    "object finite",
    "one has",
    "special case",
    "there exist",
    "there exists",
    "unique natural",
    "well defined",
}

GENERIC_BOUNDARY = set(build_term_prior._STOP) | {
    "another",
    "between",
    "dimensional",
    "many",
    "more",
    "several",
}

GENERIC_INTERIOR = {
    "whose",
}

EXACT_CONCEPT_MERGES = {
    "algebra topology": "algebraic topology",
    "hom space": "hom-spaces",
    "hom spaces": "hom-spaces",
    "n categories": "n-categories",
    "non commutative": "non-commutative",
    "quasi inverse": "quasi-inverse",
    "quasi inverses": "quasi-inverse",
    "quasi isomorphic": "quasi-isomorphism",
    "quasi isomorphism": "quasi-isomorphism",
    "quasi isomorphisms": "quasi-isomorphism",
    "semi direct product": "semidirect product",
    "semi simple": "semisimple",
    "sub category": "subcategory",
    "unit counit": "unit-counit",
}

LAST_WORD_SINGULARS = {
    "algebras": "algebra",
    "categories": "category",
    "classes": "class",
    "cofibrations": "cofibration",
    "equivalences": "equivalence",
    "functors": "functor",
    "groups": "group",
    "isomorphisms": "isomorphism",
    "modules": "module",
    "morphisms": "morphism",
    "objects": "object",
    "operations": "operation",
    "representations": "representation",
    "sets": "set",
    "spaces": "space",
    "transformations": "transformation",
}

NAMED_NORMALIZATION_EXAMPLES = [
    "non commutative",
    "unit counit",
    "algebra topology",
    "n categories",
    "quasi inverse",
    "quasi isomorphisms",
    "hom spaces",
    "natural transformations",
]


@dataclass(frozen=True)
class RankedConcept:
    rank: int
    concept: str
    df: int
    score: float
    pagerank: float
    defined: bool
    sources: tuple[str, ...]
    resolution_action: str
    input_examples: tuple[str, ...]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def basic_normalize_concept(concept: str) -> str:
    return " ".join(concept.lower().strip().split())


def singularize_last_word(phrase: str) -> str:
    words = phrase.split()
    if not words:
        return phrase
    replacement = LAST_WORD_SINGULARS.get(words[-1])
    if not replacement:
        return phrase
    return " ".join([*words[:-1], replacement])


def normalize_concept(concept: str) -> str:
    phrase = basic_normalize_concept(concept)
    phrase = EXACT_CONCEPT_MERGES.get(phrase, phrase)
    phrase = singularize_last_word(phrase)
    phrase = EXACT_CONCEPT_MERGES.get(phrase, phrase)
    return phrase


def normalization_diagnostics(paper_concepts: dict[str, list[str]]) -> dict[str, Any]:
    raw: Counter[str] = Counter()
    for concepts in paper_concepts.values():
        raw.update(basic_normalize_concept(c) for c in concepts if basic_normalize_concept(c))
    changed = {
        phrase: normalize_concept(phrase)
        for phrase in raw
        if normalize_concept(phrase) != phrase
    }
    by_canonical: dict[str, list[str]] = {}
    for phrase, canonical in changed.items():
        by_canonical.setdefault(canonical, []).append(phrase)
    return {
        "changed_phrase_types": len(changed),
        "changed_phrase_mentions": sum(raw[p] for p in changed),
        "canonical_merge_targets": sum(1 for xs in by_canonical.values() if len(xs) > 1),
        "examples": {
            phrase: {
                "canonical": normalize_concept(phrase),
                "raw_mentions": raw.get(phrase, 0),
            }
            for phrase in NAMED_NORMALIZATION_EXAMPLES
        },
    }


def invert_usage(paper_concepts: dict[str, list[str]]) -> Counter[str]:
    df: Counter[str] = Counter()
    for concepts in paper_concepts.values():
        normalized = {normalize_concept(c) for c in concepts if normalize_concept(c)}
        df.update(normalized)
    return df


def legacy_invert_usage(paper_concepts: dict[str, list[str]]) -> Counter[str]:
    df: Counter[str] = Counter()
    for concepts in paper_concepts.values():
        normalized = {basic_normalize_concept(c) for c in concepts if basic_normalize_concept(c)}
        df.update(normalized)
    return df


def boilerplate_phrase(phrase: str) -> bool:
    words = phrase.split()
    if not words:
        return True
    if phrase in GENERIC_PHRASES:
        return True
    if words[0] in GENERIC_BOUNDARY or words[-1] in GENERIC_BOUNDARY:
        return True
    single_letters = [w for w in words if len(w) == 1]
    if single_letters and phrase not in {"k theory", "n category", "n categories"}:
        return True
    if any(w in GENERIC_INTERIOR for w in words):
        return True
    if len(words) == 1 and len(words[0]) < 4:
        return True
    if len(words) >= 2 and len(set(words)) == 1:
        return True
    return False


def resolved_genuine_concept(
    phrase: str,
    df: Counter[str],
    *,
    min_papers: int,
    normalizer=normalize_concept,
) -> tuple[str | None, str]:
    """Return the canonical genuine concept and resolve action.

    `build_term_prior.resolve_phrase` is the first pass. OVERFED candidates
    collapse to its chosen core; HUNGRY candidates keep their high-frequency
    observed phrase so common concepts like "natural transformation" are not
    replaced by a narrow superphrase.
    """
    phrase = normalizer(phrase)
    if boilerplate_phrase(phrase):
        return None, "BOILERPLATE"

    resolved = build_term_prior.resolve_phrase(phrase, df, min_papers=min_papers)
    action = str(resolved.get("action", "UNKNOWN"))
    if action == "HAPAX":
        return None, action
    if action == "OVERFED":
        candidate = normalizer(str(resolved.get("resolution") or ""))
    else:
        candidate = phrase

    if boilerplate_phrase(candidate):
        return None, f"{action}+BOILERPLATE"
    return candidate, action


def genuine_ranking(
    df: Counter[str],
    *,
    pagerank: dict[str, float] | None = None,
    pagerank_weight: float = 0.0,
    min_papers: int = 3,
    normalizer=normalize_concept,
) -> list[dict[str, Any]]:
    pagerank = pagerank or {}
    aggregate: dict[str, dict[str, Any]] = {}
    for phrase, count in sorted(df.items()):
        concept, action = resolved_genuine_concept(
            phrase,
            df,
            min_papers=min_papers,
            normalizer=normalizer,
        )
        if concept is None:
            continue
        entry = aggregate.setdefault(
            concept,
            {"concept": concept, "df": 0, "resolution_actions": Counter(), "input_examples": []},
        )
        entry["df"] += count
        entry["resolution_actions"][action] += 1
        if len(entry["input_examples"]) < 5 and phrase != concept:
            entry["input_examples"].append(phrase)

    ranked = []
    for concept, entry in aggregate.items():
        pr = float(pagerank.get(concept, 0.0))
        score = float(entry["df"]) * (1.0 + pagerank_weight * pr)
        actions = entry["resolution_actions"]
        ranked.append(
            {
                "concept": concept,
                "df": int(entry["df"]),
                "score": score,
                "pagerank": pr,
                "resolution_action": ",".join(
                    f"{name}:{actions[name]}" for name in sorted(actions)
                ),
                "input_examples": tuple(entry["input_examples"]),
            }
        )
    ranked.sort(key=lambda x: (-x["score"], -x["df"], x["concept"]))
    return ranked


def concept_graph_pagerank(graph: dict[str, Any]) -> dict[str, float]:
    rows: Iterable[dict[str, Any]] = graph.get("authority") or graph.get("nodes") or []
    return {
        normalize_concept(str(row["concept"])): float(row.get("pagerank", 0.0))
        for row in rows
        if isinstance(row, dict) and row.get("concept")
    }


def definition_sets(
    snippets: dict[str, Any],
    defined_index: dict[str, Any],
    encyclopedia: dict[str, Any],
    *,
    normalizer=normalize_concept,
) -> dict[str, set[str]]:
    sources: dict[str, set[str]] = {}

    for concept, rows in (snippets.get("snippets") or {}).items():
        if rows:
            sources.setdefault(normalizer(concept), set()).add("def-snippets")

    for concept, papers in (defined_index.get("concept_to_papers") or {}).items():
        if papers:
            sources.setdefault(normalizer(concept), set()).add("defined-index")

    for entry in encyclopedia.get("entries") or []:
        if not isinstance(entry, dict) or not entry.get("concept"):
            continue
        concept = normalizer(str(entry["concept"]))
        if entry.get("gloss") or (entry.get("defined_in") or {}).get("n_papers", 0):
            sources.setdefault(concept, set()).add("concept-encyclopedia")

    return sources


def attach_coverage(
    ranked: list[dict[str, Any]],
    definition_sources: dict[str, set[str]],
) -> list[RankedConcept]:
    covered = []
    for i, row in enumerate(ranked, start=1):
        sources = tuple(sorted(definition_sources.get(row["concept"], set())))
        covered.append(
            RankedConcept(
                rank=i,
                concept=row["concept"],
                df=row["df"],
                score=row["score"],
                pagerank=row["pagerank"],
                defined=bool(sources),
                sources=sources,
                resolution_action=row["resolution_action"],
                input_examples=tuple(row["input_examples"]),
            )
        )
    return covered


def coverage_summary(ranked: list[RankedConcept], n: int) -> dict[str, Any]:
    top = ranked[:n]
    n_defined = sum(1 for row in top if row.defined)
    undefined = [row for row in top if not row.defined]
    return {
        "n": n,
        "defined": n_defined,
        "total": len(top),
        "coverage": (n_defined / len(top)) if top else 0.0,
        "undefined": undefined,
    }


def find_rank(ranked: list[RankedConcept], needle: str) -> int | None:
    needle = normalize_concept(needle)
    for row in ranked:
        if row.concept == needle:
            return row.rank
    return None


def find_first_containing(ranked: list[RankedConcept], token: str) -> RankedConcept | None:
    token = normalize_concept(token)
    return next((row for row in ranked if token in row.concept.split()), None)


def render_report(
    *,
    ranked: list[RankedConcept],
    legacy_ranked: list[RankedConcept],
    summaries: list[dict[str, Any]],
    legacy_summaries: list[dict[str, Any]],
    top_k_undefined: int,
    usage_meta: dict[str, Any],
    pagerank_weight: float,
    min_papers: int,
    normalization: dict[str, Any],
) -> str:
    lines = [
        "# SFC Concept Coverage",
        "",
        "Generated by `scripts/sfc_concept_coverage.py`.",
        "",
        "## Inputs",
        "",
        f"- Papers scanned: `{usage_meta.get('papers_scanned')}`",
        f"- Papers with concepts: `{usage_meta.get('papers_with_concepts')}`",
        f"- De-noise: `build_term_prior.resolve_phrase`, `min_papers={min_papers}`",
        f"- Pagerank weight: `{pagerank_weight}`",
        "",
        "## Coverage",
        "",
        "| Top N | Defined | Coverage | Undefined |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        legacy = next((s for s in legacy_summaries if s["n"] == summary["n"]), None)
        undefined_count = summary["total"] - summary["defined"]
        delta = ""
        if legacy:
            legacy_undefined = legacy["total"] - legacy["defined"]
            delta = f" ({legacy_undefined - undefined_count:+d} undefined vs before)"
        lines.append(
            f"| {summary['n']} | {summary['defined']}/{summary['total']} | "
            f"{summary['coverage']:.1%} | {undefined_count}{delta} |"
        )

    legacy_top = [row.concept for row in legacy_ranked[:30]]
    current_top = [row.concept for row in ranked[:30]]
    moved_out = [c for c in legacy_top if c not in current_top][:8]
    moved_in = [c for c in current_top if c not in legacy_top][:8]
    lines.extend(
        [
            "",
            "## Normalization Before/After",
            "",
            f"- Changed phrase types: `{normalization['changed_phrase_types']}`",
            f"- Changed phrase mentions: `{normalization['changed_phrase_mentions']}`",
            f"- Canonical targets with multiple merged inputs: `{normalization['canonical_merge_targets']}`",
            "- Top-30 concepts moved out after normalization: "
            + (", ".join(f"`{x}`" for x in moved_out) if moved_out else "`none`"),
            "- Top-30 concepts moved in after normalization: "
            + (", ".join(f"`{x}`" for x in moved_in) if moved_in else "`none`"),
            "",
            "| Raw phrase | Canonical | Raw mentions | Before undefined rank | After undefined rank |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    legacy_undefined_rank = {
        row.concept: row.rank
        for row in legacy_ranked
        if not row.defined
    }
    undefined_rank = {
        row.concept: row.rank
        for row in ranked
        if not row.defined
    }
    for phrase, row in normalization["examples"].items():
        canonical = row["canonical"]
        lines.append(
            f"| `{phrase}` | `{canonical}` | {row['raw_mentions']} | "
            f"{legacy_undefined_rank.get(phrase, '-')} | {undefined_rank.get(canonical, '-')} |"
        )

    lines.extend(
        [
            "",
            "## Top Genuine Concepts",
            "",
            "| Rank | Concept | DF | Defined | Sources |",
            "| ---: | --- | ---: | --- | --- |",
        ]
    )
    for row in ranked[:30]:
        sources = ", ".join(row.sources) if row.sources else "-"
        lines.append(
            f"| {row.rank} | `{row.concept}` | {row.df} | "
            f"{'yes' if row.defined else 'no'} | {sources} |"
        )

    lines.extend(
        [
            "",
            "## Undefined Priorities",
            "",
            "| Rank | Concept | DF | Pagerank |",
            "| ---: | --- | ---: | ---: |",
        ]
    )
    undefined_rows = [row for row in ranked if not row.defined][:top_k_undefined]
    for row in undefined_rows:
        lines.append(f"| {row.rank} | `{row.concept}` | {row.df} | {row.pagerank:.6f} |")

    natural_rank = find_rank(ranked, "natural transformation")
    adjoint_row = find_first_containing(ranked, "adjoint")
    filtered_examples = [
        phrase for phrase in ("there exists", "more generally")
        if find_rank(ranked, phrase) is None
    ]
    lines.extend(
        [
            "",
            "## De-noising Checks",
            "",
            f"- `natural transformation` rank: `{natural_rank}`",
            "- First ranked concept containing `adjoint`: "
            + (
                f"`{adjoint_row.concept}` at rank `{adjoint_row.rank}`"
                if adjoint_row
                else "`not found`"
            ),
            "- Filtered boilerplate examples: "
            + ", ".join(f"`{x}`" for x in filtered_examples),
            "",
            "## Remaining Gaps",
            "",
            "- Coverage is lexical over existing definition substrates; it does not prove that a definition is high quality.",
            "- Concept singular/plural variants are not fully lemmatized; they remain visible as separate ranked concepts when both are used.",
            "- `defined-index.json` is broad and may count weak definition-like evidence; the undefined list is the safer action queue.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> tuple[list[RankedConcept], list[dict[str, Any]], str]:
    usage = load_json(args.usage)
    normalization = normalization_diagnostics(usage["paper_concepts"])
    legacy_raw_df = legacy_invert_usage(usage["paper_concepts"])
    raw_df = invert_usage(usage["paper_concepts"])
    pagerank = concept_graph_pagerank(load_json(args.concept_graph))
    legacy_ranked_raw = genuine_ranking(
        legacy_raw_df,
        pagerank=pagerank,
        pagerank_weight=args.pagerank_weight,
        min_papers=args.min_papers,
        normalizer=basic_normalize_concept,
    )
    ranked_raw = genuine_ranking(
        raw_df,
        pagerank=pagerank,
        pagerank_weight=args.pagerank_weight,
        min_papers=args.min_papers,
    )
    legacy_definition_sources = definition_sets(
        load_json(args.def_snippets),
        load_json(args.defined_index),
        load_json(args.concept_encyclopedia),
        normalizer=basic_normalize_concept,
    )
    definition_sources = definition_sets(
        load_json(args.def_snippets),
        load_json(args.defined_index),
        load_json(args.concept_encyclopedia),
    )
    legacy_ranked = attach_coverage(legacy_ranked_raw, legacy_definition_sources)
    ranked = attach_coverage(ranked_raw, definition_sources)
    legacy_summaries = [coverage_summary(legacy_ranked, n) for n in args.top_n]
    summaries = [coverage_summary(ranked, n) for n in args.top_n]
    report = render_report(
        ranked=ranked,
        legacy_ranked=legacy_ranked,
        summaries=summaries,
        legacy_summaries=legacy_summaries,
        top_k_undefined=args.top_k_undefined,
        usage_meta=usage,
        pagerank_weight=args.pagerank_weight,
        min_papers=args.min_papers,
        normalization=normalization,
    )
    return ranked, summaries, report


def print_console(ranked: list[RankedConcept], summaries: list[dict[str, Any]], top_k: int) -> None:
    print("coverage:")
    for summary in summaries:
        print(
            f"  top-{summary['n']}: {summary['defined']}/{summary['total']} "
            f"= {summary['coverage']:.1%}"
        )
    print()
    print("top genuine concepts:")
    for row in ranked[:20]:
        print(
            f"  {row.rank:>3} {row.concept:<40} df={row.df:<5} "
            f"defined={'yes' if row.defined else 'no'}"
        )
    print()
    print(f"top {top_k} undefined genuine concepts:")
    for row in [r for r in ranked if not r.defined][:top_k]:
        print(f"  {row.rank:>3} {row.concept:<40} df={row.df:<5}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--usage", type=Path, default=DEFAULT_USAGE)
    parser.add_argument("--def-snippets", type=Path, default=DEFAULT_SNIPPETS)
    parser.add_argument("--defined-index", type=Path, default=DEFAULT_DEFINED)
    parser.add_argument("--concept-encyclopedia", type=Path, default=DEFAULT_ENCYCLOPEDIA)
    parser.add_argument("--concept-graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--top-n", type=int, nargs="+", default=[100, 500])
    parser.add_argument("--top-k-undefined", type=int, default=50)
    parser.add_argument("--min-papers", type=int, default=3)
    parser.add_argument("--pagerank-weight", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ranked, summaries, report = run(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report)
    print_console(ranked, summaries, args.top_k_undefined)
    print()
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
