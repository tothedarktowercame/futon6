#!/usr/bin/env python3
"""Search arXiv for Problem 7 gap-closing literature.

Focus: G1/G2/G3 and U1-U3
- G1: rational PD / Poincare complex structure
- G2: Spivak normal fibration reduction / normal maps / rational surgery
- G3: transfer compatibility of surgery obstruction
- U1-U3: equivariant splitting, Thom isomorphism, twisted L-theory coefficients

Outputs:
  - JSON (default): data/first-proof/problem7-gap-arxiv-results.json
  - Markdown short-list (optional): data/first-proof/problem7-gap-arxiv-top20.md
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

NS = {"a": "http://www.w3.org/2005/Atom"}


@dataclass(frozen=True)
class Query:
    id: str
    text: str
    gap: str


QUERIES: list[Query] = [
    Query("g1-01", 'all:"rational Poincare duality" AND all:"finite CW"', "G1"),
    Query("g1-02", 'all:"Bredon Poincare duality" AND all:"group"', "G1"),
    Query("g1-03", 'all:"orbifold" AND all:"Poincare duality" AND all:"rational"', "G1"),
    Query("g1-04", 'all:"rationally acyclic" AND all:"universal cover" AND all:"Poincare"', "G1"),
    Query("g2-01", 'all:"Spivak normal fibration" AND all:"Poincare complex"', "G2"),
    Query("g2-02", 'all:"degree one normal map" AND all:"Poincare complex"', "G2"),
    Query("g2-03", 'all:"rational surgery" AND all:"Poincare duality group"', "G2"),
    Query("g2-04", 'all:"surgery obstruction" AND all:"normal invariant"', "G2"),
    Query("g3-01", 'all:"transfer" AND all:"surgery obstruction" AND all:"finite index"', "G3"),
    Query("u1-01", 'all:"equivariant homology" AND all:"isotropy filtration" AND all:"splitting"', "U1"),
    Query("u2-01", 'all:"equivariant Thom isomorphism" AND all:"L-theory"', "U2"),
    Query("u2-02", 'all:"equivariant L-theory" AND all:"Thom"', "U2"),
    Query("u3-01", 'all:"twisted L-theory" AND all:"orientation character"', "U3"),
    Query("u3-02", 'all:"L-groups" AND all:"orientation character" AND all:"Q"', "U3"),
    Query("fj-01", 'all:"Farrell-Jones" AND all:"lattices" AND all:"Lie groups"', "FJ"),
    Query("fj-02", 'all:"UNil" AND all:"infinite dihedral" AND all:"L-theory"', "FJ"),
    Query("rat-01", 'all:"Sullivan" AND all:"rational homotopy" AND all:"fundamental group"', "G3"),
    Query("rat-02", 'all:"non-nilpotent" AND all:"rational homotopy"', "G3"),
    Query("geom-01", 'all:"reflection" AND all:"hyperbolic manifold" AND all:"fixed set"', "E2"),
    Query("geom-02", 'all:"equivariant surgery" AND all:"Z/2"', "G2"),
]


def query_arxiv(query: str, max_results: int) -> list[dict]:
    q = urllib.parse.quote(query)
    url = (
        "http://export.arxiv.org/api/query"
        f"?search_query={q}&start=0&max_results={max_results}"
        "&sortBy=relevance&sortOrder=descending"
    )
    data = urllib.request.urlopen(url, timeout=30).read()
    root = ET.fromstring(data)
    out: list[dict] = []
    for entry in root.findall("a:entry", NS):
        aid = entry.findtext("a:id", "", NS)
        m = re.search(r"arxiv\\.org/abs/([^\\s]+)", aid or "")
        raw = m.group(1) if m else (aid.rsplit("/", 1)[-1] if aid else "")
        arx = re.sub(r"v\d+$", "", raw)
        title = " ".join((entry.findtext("a:title", "", NS) or "").split())
        date = (entry.findtext("a:published", "", NS) or "")[:10]
        authors = [x.findtext("a:name", "", NS) for x in entry.findall("a:author", NS)]
        cats = [x.attrib.get("term", "") for x in entry.findall("a:category", NS)]
        summary = " ".join((entry.findtext("a:summary", "", NS) or "").split())
        out.append(
            {
                "id": arx,
                "title": title,
                "date": date,
                "authors": authors,
                "cats": cats,
                "summary": summary,
                "url": f"https://arxiv.org/abs/{arx}",
            }
        )
    return out


def score_row(row: dict, query_hits: int, gap_hits: int) -> int:
    text = (row["title"] + " " + row["summary"]).lower()
    keys = [
        "poincare",
        "duality",
        "spivak",
        "normal map",
        "surgery",
        "farrell-jones",
        "l-theory",
        "equivariant",
        "thom",
        "orientation character",
        "unil",
        "rational homotopy",
        "bredon",
        "orbifold",
        "fixed set",
    ]
    score = 3 * gap_hits + 2 * query_hits
    for k in keys:
        if k in text:
            score += 1
    cat_text = ",".join(c.lower() for c in row.get("cats", []))
    for c in ("math.gt", "math.at", "math.gr", "math.kt"):
        if c in cat_text:
            score += 1
    # Prefer older foundational + modern survey-ish papers slightly
    year = 0
    try:
        year = int((row.get("date", "") or "")[:4])
    except Exception:
        year = 0
    if 1990 <= year <= 2016:
        score += 1
    return score


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="data/first-proof/problem7-gap-arxiv-results.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--top-markdown",
        default="data/first-proof/problem7-gap-arxiv-top20.md",
        help="Output markdown shortlist path",
    )
    parser.add_argument("--max-results", type=int, default=15)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    rows: list[dict] = []
    for q in QUERIES:
        try:
            hits = query_arxiv(q.text, args.max_results)
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "query_id": q.id,
                    "query_text": q.text,
                    "gap": q.gap,
                    "error": str(exc),
                }
            )
            continue
        for hit in hits:
            hit["query_id"] = q.id
            hit["query_text"] = q.text
            hit["gap"] = q.gap
            rows.append(hit)

    by: dict[str, dict] = {}
    for row in rows:
        if "id" not in row:
            continue
        rid = row["id"]
        merged = by.setdefault(
            rid,
            {
                "id": row["id"],
                "title": row["title"],
                "date": row["date"],
                "authors": row["authors"],
                "cats": row["cats"],
                "summary": row["summary"],
                "url": row["url"],
                "query_ids": set(),
                "query_texts": set(),
                "gaps": set(),
            },
        )
        merged["query_ids"].add(row["query_id"])
        merged["query_texts"].add(row["query_text"])
        merged["gaps"].add(row["gap"])

    merged_rows = []
    for row in by.values():
        query_ids = sorted(row["query_ids"])
        query_texts = sorted(row["query_texts"])
        gaps = sorted(row["gaps"])
        merged_rows.append(
            {
                **{k: v for k, v in row.items() if k not in {"query_ids", "query_texts", "gaps"}},
                "query_ids": query_ids,
                "query_texts": query_texts,
                "gaps": gaps,
                "query_hits": len(query_ids),
                "gap_hits": len(gaps),
            }
        )

    for row in merged_rows:
        row["score"] = score_row(row, row["query_hits"], row["gap_hits"])

    merged_rows.sort(key=lambda x: (-x["score"], -x["gap_hits"], -x["query_hits"], x["date"]))

    top = merged_rows[: args.top_n]

    out = {
        "scope": "problem7-gaps",
        "queries": [{"id": q.id, "text": q.text, "gap": q.gap} for q in QUERIES],
        "unique_results": len(merged_rows),
        "top_n": args.top_n,
        "top_results": top,
        "results": merged_rows,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    md_lines = [
        "# Problem 7 Gap Papers (arXiv Top 20)",
        "",
        "Generated by `scripts/search-arxiv-p7-gaps.py`.",
        "",
        "Gap tags:",
        "- `G1`: rational PD / Poincare-complex bridge",
        "- `G2`: normal map / Spivak reduction / rational surgery",
        "- `G3`: transfer compatibility",
        "- `U1-U3`: equivariant splitting, Thom, twisted coefficients",
        "- `FJ`: Farrell-Jones / UNil",
        "- `E2`: reflection/fixed-set context",
        "",
    ]
    for i, row in enumerate(top, start=1):
        gaps = ", ".join(row["gaps"])
        md_lines.append(
            f"{i}. `{row['id']}` — {row['title']}  "
            f"(date: {row['date']}, score: {row['score']}, gaps: {gaps})  "
            f"{row['url']}"
        )
    md_lines.append("")
    Path(args.top_markdown).write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Wrote {out_path}")
    print(f"Wrote {args.top_markdown}")
    print(f"unique_results={len(merged_rows)}")
    for row in top:
        print(
            f"{row['score']:2d} s | {row['gap_hits']} g | {row['query_hits']} q | "
            f"{row['date']} | {row['id']} | {row['title']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
