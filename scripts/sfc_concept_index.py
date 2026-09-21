#!/usr/bin/env python3
"""Build/query the structure-first concept -> papers shuffle index.

This is D3 only: invert paper -> concepts into concept -> paper lists and attach
the already-existing SFC1 genuine/definition flags. It deliberately does not
build per-paper grounded instances or the genus/variant reduce.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import sfc_concept_coverage as sfc  # noqa: E402

DEFAULT_INDEX = ROOT / "data" / "warp" / "concept-index.json"
DEFAULT_REPORT = ROOT / "holes" / "excursions" / "sfc-concept-index.md"


def collect_papers_by_concept(paper_concepts: dict[str, list[str]]) -> dict[str, list[str]]:
    papers_by_concept: dict[str, set[str]] = {}
    for paper, concepts in paper_concepts.items():
        for concept in {sfc.normalize_concept(c) for c in concepts if sfc.normalize_concept(c)}:
            papers_by_concept.setdefault(concept, set()).add(paper)
    return {concept: sorted(papers) for concept, papers in sorted(papers_by_concept.items())}


def definition_provenance(def_snippets, defined_index, encyclopedia):
    """Keep definition-bearing papers separate from papers merely using a term.

    A global gloss without a paper remains global evidence, never an invented
    per-paper definition. Encyclopedia samples are partial, not all n_papers.
    """
    evidence = {}

    def add(concept, paper, source):
        concept = sfc.normalize_concept(concept)
        if concept and isinstance(paper, str) and paper:
            evidence.setdefault(concept, set()).add((paper, source))

    for concept, papers in (defined_index.get("concept_to_papers") or {}).items():
        for paper in papers:
            add(concept, paper, "defined-index")
    for concept, rows in (def_snippets.get("snippets") or {}).items():
        for row in rows:
            if isinstance(row, dict) and row.get("snippet"):
                add(concept, row.get("paper"), "def-snippets")
    for entry in encyclopedia.get("entries") or []:
        if not isinstance(entry, dict) or not entry.get("concept"):
            continue
        gloss = entry.get("gloss")
        if isinstance(gloss, dict) and gloss.get("text"):
            add(entry["concept"], gloss.get("paper"), "concept-encyclopedia:gloss")
        defined_in = entry.get("defined_in") or {}
        for paper in defined_in.get("papers", []):
            add(entry["concept"], paper, "concept-encyclopedia:defined-in")
        for paper in defined_in.get("sample", []):
            add(entry["concept"], paper, "concept-encyclopedia:sample")
    return {concept: [{"paper": paper, "source": source}
                      for paper, source in sorted(rows)]
            for concept, rows in evidence.items()}


def definitions_for_papers(index, papers):
    """Definition-ingestion metric: credit only observed defining-paper IDs.

    Consumers must use definition_papers, not the global `defined` flag.
    Legacy indices lack the evidence needed for a prefix claim and refuse.
    """
    papers = set(papers)
    if any("definition_papers" not in row for row in index.values()):
        raise ValueError("definition-provenance-missing: rebuild concept index")
    return {concept for concept, row in index.items()
            if papers.intersection(row["definition_papers"])}


def build_index(
    *,
    usage: dict[str, Any],
    def_snippets: dict[str, Any],
    defined_index: dict[str, Any],
    encyclopedia: dict[str, Any],
    min_papers: int = 3,
) -> tuple[dict[str, dict[str, Any]], list[sfc.RankedConcept]]:
    paper_concepts = usage["paper_concepts"]
    df = sfc.invert_usage(paper_concepts)
    papers_by_concept = collect_papers_by_concept(paper_concepts)
    ranked_raw = sfc.genuine_ranking(df, min_papers=min_papers)
    definition_sources = sfc.definition_sets(def_snippets, defined_index, encyclopedia)
    ranked = sfc.attach_coverage(ranked_raw, definition_sources)
    genuine = {row.concept for row in ranked}

    provenance = definition_provenance(def_snippets, defined_index, encyclopedia)
    index: dict[str, dict[str, Any]] = {}
    for concept in sorted(df):
        papers = papers_by_concept.get(concept, [])
        sources = sorted(definition_sources.get(concept, set()))
        index[concept] = {
            "df": int(df[concept]),
            "papers": papers,
            "genuine": concept in genuine,
            # Global availability only; never a prefix-ingestion assertion.
            "defined": bool(sources),
            "definition_papers": sorted({r["paper"] for r in provenance.get(concept, [])}),
            "definition_evidence": provenance.get(concept, []),
            "definition_provenance_status": (
                "paper-attributed" if provenance.get(concept) else
                "global-only" if sources else "absent"),
            "sources": sources,
        }
    return index, ranked


def validate_index(index: dict[str, dict[str, Any]], usage: dict[str, Any]) -> None:
    df = sfc.invert_usage(usage["paper_concepts"])
    if set(index) != set(df):
        missing = sorted(set(df) - set(index))[:10]
        extra = sorted(set(index) - set(df))[:10]
        raise ValueError(f"concept key mismatch missing={missing} extra={extra}")
    for concept, count in df.items():
        row = index[concept]
        if "definition_papers" not in row:
            raise ValueError(f"definition-provenance-missing for {concept}")
        if set(row["definition_papers"]) != {r["paper"] for r in row["definition_evidence"]}:
            raise ValueError(f"definition provenance mismatch for {concept}")
        if row["df"] != count:
            raise ValueError(f"df mismatch for {concept}: {row['df']} != {count}")
        if len(row["papers"]) != count:
            raise ValueError(f"paper-list mismatch for {concept}: {len(row['papers'])} != {count}")


def coverage_from_index(index: dict[str, dict[str, Any]], ranked: list[sfc.RankedConcept], n: int) -> dict[str, Any]:
    top = ranked[:n]
    defined = sum(1 for row in top if index[row.concept]["defined"])
    return {
        "n": n,
        "defined": defined,
        "total": len(top),
        "coverage": (defined / len(top)) if top else 0.0,
    }


def render_report(
    *,
    index: dict[str, dict[str, Any]],
    ranked: list[sfc.RankedConcept],
    usage: dict[str, Any],
    sample_concept: str,
) -> str:
    concept = sfc.normalize_concept(sample_concept)
    sample = index.get(concept)
    top_100 = coverage_from_index(index, ranked, 100)
    top_500 = coverage_from_index(index, ranked, 500)
    genuine_count = sum(1 for row in index.values() if row["genuine"])
    defined_count = sum(1 for row in index.values() if row["defined"])
    lines = [
        "# SFC Concept Index",
        "",
        "Generated by `scripts/sfc_concept_index.py`.",
        "",
        "## Inputs",
        "",
        f"- Papers scanned: `{usage.get('papers_scanned')}`",
        f"- Papers with concepts: `{len(usage.get('paper_concepts', {}))}`",
        f"- Indexed concepts: `{len(index)}`",
        f"- Genuine concepts: `{genuine_count}`",
        f"- Concepts with definition evidence: `{defined_count}`",
        "",
        "## SFC1 Consistency",
        "",
        "| Top N genuine | Defined | Coverage |",
        "| ---: | ---: | ---: |",
        f"| 100 | {top_100['defined']}/{top_100['total']} | {top_100['coverage']:.1%} |",
        f"| 500 | {top_500['defined']}/{top_500['total']} | {top_500['coverage']:.1%} |",
        "",
        "## Sample Query",
        "",
    ]
    if sample:
        lines.extend(
            [
                f"- Concept: `{concept}`",
                f"- DF: `{sample['df']}`",
                f"- Genuine: `{sample['genuine']}`",
                f"- Defined: `{sample['defined']}`",
                f"- Sources: `{', '.join(sample['sources']) if sample['sources'] else '-'}`",
                f"- First papers: `{', '.join(sample['papers'][:20])}`",
            ]
        )
    else:
        lines.append(f"- Concept `{concept}` not found.")
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "This is the D3 shuffle only: concept -> paper-list plus SFC1 flags. "
            "Definition paper provenance is retained separately from usage. The `defined` flag "
            "means global availability; prefix-ingestion consumers must use `definition_papers`. "
            "Encyclopedia samples provide only the listed paper IDs, not all reported n_papers. "
            "It does not build per-paper grounded instances or the genus/variant-axis reduce.",
            "",
        ]
    )
    return "\n".join(lines)


def load_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    return (
        sfc.load_json(args.usage),
        sfc.load_json(args.def_snippets),
        sfc.load_json(args.defined_index),
        sfc.load_json(args.concept_encyclopedia),
    )


def print_concept(index: dict[str, dict[str, Any]], concept: str) -> None:
    key = sfc.normalize_concept(concept)
    row = index.get(key)
    if row is None:
        print(json.dumps({"concept": key, "found": False}, indent=2))
        return
    print(json.dumps({"concept": key, "found": True, **row}, indent=2))


def print_paper(index: dict[str, dict[str, Any]], usage: dict[str, Any], paper: str) -> None:
    concepts = [
        sfc.normalize_concept(c)
        for c in usage.get("paper_concepts", {}).get(paper, [])
        if sfc.normalize_concept(c)
    ]
    rows = [
        {"concept": concept, **{k: v for k, v in index[concept].items() if k != "papers"}}
        for concept in sorted(set(concepts))
        if concept in index
    ]
    print(json.dumps({"paper": paper, "concept_count": len(rows), "concepts": rows}, indent=2))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usage", type=Path, default=sfc.DEFAULT_USAGE)
    parser.add_argument("--def-snippets", type=Path, default=sfc.DEFAULT_SNIPPETS)
    parser.add_argument("--defined-index", type=Path, default=sfc.DEFAULT_DEFINED)
    parser.add_argument("--concept-encyclopedia", type=Path, default=sfc.DEFAULT_ENCYCLOPEDIA)
    parser.add_argument("--out", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--min-papers", type=int, default=3)
    parser.add_argument("--sample-concept", default="natural transformation")
    parser.add_argument("--concept", help="Query concept -> papers after building/loading the index")
    parser.add_argument("--paper", help="Query paper -> concepts after building/loading the index")
    parser.add_argument("--no-write", action="store_true", help="Build/query without writing index/report")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild even when querying an existing index")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    query_mode = bool(args.concept or args.paper)
    if query_mode and args.out.exists() and not args.rebuild:
        index = sfc.load_json(args.out)
        usage = sfc.load_json(args.usage) if args.paper else {"paper_concepts": {}}
        ranked: list[sfc.RankedConcept] = []
    else:
        usage, def_snippets, defined_index, encyclopedia = load_inputs(args)
        index, ranked = build_index(
            usage=usage,
            def_snippets=def_snippets,
            defined_index=defined_index,
            encyclopedia=encyclopedia,
            min_papers=args.min_papers,
        )
        validate_index(index, usage)

    if not args.no_write and not (query_mode and args.out.exists() and not args.rebuild):
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(index, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            render_report(index=index, ranked=ranked, usage=usage, sample_concept=args.sample_concept),
            encoding="utf-8",
        )

    if args.concept:
        print_concept(index, args.concept)
    elif args.paper:
        print_paper(index, usage, args.paper)
    else:
        print(
            f"indexed {len(index)} concepts over {len(usage.get('paper_concepts', {}))} papers; "
            f"wrote {args.out if not args.no_write else '(no-write)'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
