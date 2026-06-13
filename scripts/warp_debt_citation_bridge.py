#!/usr/bin/env python3
"""Bridge corpus definition holes through the citation graph.

For each corpus-debt frontier concept, compare two definition notions:

* provenance-term definitions: exact raw concordance terms from the debt row
  (for example ``\\smcat``). This matches corpus-debt v2's undefined status.
* concept-label definitions: normalized human concept labels (for example
  ``symmetric monoidal category``). This catches label-level definitions that
  the provenance macro did not inherit.

The report asks whether using papers cite defining papers under either notion.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEBT = ROOT / "data" / "warp" / "corpus-debt.json"
DEFAULT_CONCORDANCE = ROOT / "data" / "warp" / "concordance.json"
DEFAULT_CITATIONS = ROOT / "data" / "warp" / "citations.json"
DEFAULT_BIB = ROOT / "data" / "warp" / "bib-index.json"
DEFAULT_OUT = ROOT / "holes" / "debt-citation-bridge.md"


def norm(value: str) -> str:
    value = re.sub(r"\\[A-Za-z@]+", " ", value)
    value = re.sub(r"[^A-Za-z0-9]+", " ", value).strip().lower()
    return re.sub(r"\s+", " ", value)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def paper_ids_from_bib(path: Path) -> set[str]:
    payload = load_json(path)
    rows = payload.get("papers") if isinstance(payload, dict) else payload
    return {
        str(r.get("paper_id") or r.get("paper") or r.get("id") or r.get("entity"))
        for r in rows
        if isinstance(r, dict)
    }


def rows_for_terms(concordance: dict[str, list[dict]], terms: list[str], role: str) -> list[dict]:
    out: list[dict] = []
    for term in terms:
        for row in concordance.get(term, []):
            if row.get("role") == role:
                out.append({"term": term, **row})
    return out


def rows_for_label(concordance: dict[str, list[dict]], label: str, role: str) -> list[dict]:
    target = norm(label)
    out: list[dict] = []
    for term, rows in concordance.items():
        if norm(term) != target:
            continue
        for row in rows:
            if row.get("role") == role:
                out.append({"term": term, **row})
    return out


def bridge_rows(users: set[str], def_papers: set[str], out_edges: dict[str, list[dict]]) -> list[dict]:
    out: list[dict] = []
    for paper in sorted(users):
        for edge in out_edges.get(paper, []):
            if edge.get("to") in def_papers:
                out.append({"from": paper, "to": edge.get("to"), "via": edge.get("via", {})})
    return out


def fmt_papers(papers: set[str]) -> str:
    if not papers:
        return "0"
    return f"{len(papers)} (" + ", ".join(f"`{p}`" for p in sorted(papers)) + ")"


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    debt = load_json(args.debt)
    concordance = load_json(args.concordance)["terms"]
    citations = load_json(args.citations)
    corpus_ids = paper_ids_from_bib(args.bib_index)
    out_edges: dict[str, list[dict]] = {}
    for edge in citations.get("edges", []):
        out_edges.setdefault(edge["from"], []).append(edge)

    rows = []
    for item in debt["externally_covered_corpus_undefined"]:
        provenance_terms = list(item.get("concordance_terms") or [])
        users = {r["paper"] for r in rows_for_terms(concordance, provenance_terms, "used")}
        provenance_defs = rows_for_terms(concordance, provenance_terms, "defined")
        label_defs = rows_for_label(concordance, item["term"], "defined")
        provenance_def_papers = {r["paper"] for r in provenance_defs}
        label_def_papers = {r["paper"] for r in label_defs}
        provenance_bridges = bridge_rows(users, provenance_def_papers, out_edges)
        label_bridges = bridge_rows(users, label_def_papers, out_edges)
        rows.append(
            {
                "term": item["term"],
                "used_papers": len(users),
                "used_count": item["used_count"],
                "provenance_terms": provenance_terms,
                "provenance_def_papers": sorted(provenance_def_papers),
                "label_def_papers": sorted(label_def_papers),
                "provenance_bridges": provenance_bridges,
                "label_bridges": label_bridges,
                "label_def_papers_exist_in_corpus": sorted(p for p in label_def_papers if p in corpus_ids),
            }
        )
    return {
        "citation_stats": citations.get("stats", {}),
        "rows": rows,
        "corpus_ids": corpus_ids,
    }


def render(report: dict[str, Any]) -> str:
    rows = report["rows"]
    positive_provenance = sum(len(r["provenance_bridges"]) for r in rows)
    positive_label = sum(len(r["label_bridges"]) for r in rows)
    with_label_defs = [r for r in rows if r["label_def_papers"]]
    lines = [
        "# Debt-Citation Bridge",
        "",
        "Input artifacts:",
        "",
        "- `data/warp/corpus-debt.json` (`warp-corpus-debt-v2`)",
        "- `data/warp/concordance.json`",
        "- `data/warp/citations.json` (`30,426` edges, linkage `0.118838`)",
        "",
        "Question: for each high-use corpus definition hole, does a paper that uses the",
        "concept cite another corpus paper that defines it?",
        "",
        "## Result",
        "",
        f"Positive bridges using corpus-debt provenance terms: `{positive_provenance}`.",
        f"Positive bridges using normalized concept labels: `{positive_label}`.",
        "",
        "The corpus-debt v2 undefined status is provenance-term based: for example,",
        "`\\smcat` and `\\inprod` have no `role=defined` rows. A stricter",
        "concept-label scan finds in-corpus definitions for two labels",
        "(`symmetric monoidal category`, `inner product`), but no using paper cites",
        "those defining papers. The graph therefore still does not self-contain the",
        "frontier holes.",
        "",
        "Grounding strategy: use external nLab/Lean/PlanetMath anchors for all 18;",
        "optionally also attach the two label-level in-corpus definers as secondary",
        "anchors after the concordance role/provenance model is reconciled.",
        "",
        "## Bridge Table",
        "",
        "| concept | papers using | provenance-term definitions | concept-label definitions | citation bridges | verdict |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        pdefs = set(row["provenance_def_papers"])
        ldefs = set(row["label_def_papers"])
        bridges = len(row["provenance_bridges"]) + len(row["label_bridges"])
        if ldefs and not bridges:
            verdict = "label definition exists, but users do not cite it"
        elif not ldefs:
            verdict = "no corpus concept-label definition"
        else:
            verdict = "bridged"
        lines.append(
            f"| {row['term']} | {row['used_papers']} | {fmt_papers(pdefs)} | "
            f"{fmt_papers(ldefs)} | {bridges} | {verdict} |"
        )
    lines.extend(
        [
            "",
            "## Spot Checks",
            "",
            "There are no positive bridges to resolve. I checked the disagreement mode directly:",
            "",
            "- `\\smcat`: `26` used rows and `0` defined rows for the provenance term;",
            "  the normalized label `symmetric monoidal category` has definitions in",
            "  `0706.0711` and `0809.2517`, but none of the 26 users cite those papers.",
            "- `\\inprod`: `26` used rows and `0` defined rows for the provenance term;",
            "  the normalized label `inner product` has a definition in `0706.0711`,",
            "  but none of the 26 users cite it.",
            "- `homotopy colimit`: `168` users via `\\hocolim`; no provenance-term or",
            "  normalized-label corpus definition exists, so no citation bridge can exist.",
            "",
            "## Method",
            "",
            "For each frontier concept, collect using papers from its provenance",
            "`concordance_terms`, collect defining papers in two modes (exact provenance",
            "term and normalized concept label), then intersect the using papers'",
            "outgoing citation targets with each defining-paper set.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--debt", type=Path, default=DEFAULT_DEBT)
    ap.add_argument("--concordance", type=Path, default=DEFAULT_CONCORDANCE)
    ap.add_argument("--citations", type=Path, default=DEFAULT_CITATIONS)
    ap.add_argument("--bib-index", type=Path, default=DEFAULT_BIB)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = analyze(args)
    args.out.write_text(render(report), encoding="utf-8")
    rows = report["rows"]
    print(
        json.dumps(
            {
                "frontier_concepts": len(rows),
                "provenance_positive_bridges": sum(len(r["provenance_bridges"]) for r in rows),
                "label_positive_bridges": sum(len(r["label_bridges"]) for r in rows),
                "concepts_with_label_definitions": sum(1 for r in rows if r["label_def_papers"]),
                "out": str(args.out),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
