#!/usr/bin/env python3
"""Search arXiv for Problem 7 open-approach targets from commit 0fa4e82.

Focus:
- Approach II: equivariant surgery to eliminate codim-1 fixed sets
- Approach III: orbifold resolution with pi_1 control
- Cross-cutting E2-alt: odd-dim constructions with even-dim fixed sets and chi=0

Outputs:
  - JSON: data/first-proof/problem7-approach23-arxiv-results.json
  - MD:   data/first-proof/problem7-approach23-top20.md
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
    focus: str


QUERIES: list[Query] = [
    Query("a2-01", 'all:"equivariant surgery" AND all:"fixed point set"', "Approach-II"),
    Query("a2-02", 'all:"equivariant surgery" AND all:"codimension one"', "Approach-II"),
    Query("a2-03", 'all:"equivariant surgery" AND all:"involution"', "Approach-II"),
    Query("a2-04", 'all:"Dovermann" AND all:"Schultz" AND all:"equivariant surgery"', "Approach-II"),
    Query("a2-05", 'all:"equivariant Spivak" AND all:"surgery"', "Approach-II"),
    Query("a2-06", 'all:"equivariant bordism" AND all:"fixed points"', "Approach-II"),
    Query("a2-07", 'all:"G-manifold" AND all:"fixed set" AND all:"surgery"', "Approach-II"),
    Query("a3-01", 'all:"orbifold resolution" AND all:"fundamental group"', "Approach-III"),
    Query("a3-02", 'all:"resolution of orbifold singularities" AND all:"fundamental group"', "Approach-III"),
    Query("a3-03", 'all:"hyperbolic orbifold" AND all:"resolution"', "Approach-III"),
    Query("a3-04", 'all:"aspherical orbifold" AND all:"manifold"', "Approach-III"),
    Query("a3-05", 'all:"orbifold fundamental group" AND all:"manifold" AND all:"surgery"', "Approach-III"),
    Query("a3-06", 'all:"Borel conjecture" AND all:"orbifold"', "Approach-III"),
    Query("e2o-01", 'all:"hyperbolic manifold" AND all:"involution" AND all:"fixed hypersurface"', "E2-odd-alt"),
    Query("e2o-02", 'all:"fixed set" AND all:"Euler characteristic" AND all:"hyperbolic manifold"', "E2-odd-alt"),
    Query("e2o-03", 'all:"reflection" AND all:"hyperbolic manifold" AND all:"fixed"', "E2-odd-alt"),
    Query("e2o-04", 'all:"totally geodesic hypersurface" AND all:"hyperbolic manifold"', "E2-odd-alt"),
    Query("e2o-05", 'all:"nonseparating fixed point set" AND all:"hyperbolic"', "E2-odd-alt"),
    Query("e2o-06", 'all:"finite group action" AND all:"hyperbolic manifold" AND all:"fixed set"', "E2-odd-alt"),
    Query("x-01", 'all:"Smith theory" AND all:"aspherical" AND all:"periodic diffeomorphisms"', "Cross"),
    Query("x-02", 'all:"orbifold" AND all:"equivariant surgery"', "Cross"),
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
        m = re.search(r"arxiv\.org/abs/([^\s]+)", aid or "")
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


def score_row(row: dict, query_hits: int, focus_hits: int) -> int:
    text = (row["title"] + " " + row["summary"]).lower()
    keys = [
        "equivariant surgery",
        "spivak",
        "fixed point",
        "involution",
        "codimension",
        "orbifold",
        "resolution",
        "fundamental group",
        "hyperbolic",
        "totally geodesic",
        "euler characteristic",
        "reflection",
        "smith theory",
    ]
    score = 3 * focus_hits + 2 * query_hits
    for k in keys:
        if k in text:
            score += 1
    cat_text = ",".join(c.lower() for c in row.get("cats", []))
    for c in ("math.gt", "math.at", "math.dg", "math.gr"):
        if c in cat_text:
            score += 1
    return score


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/first-proof/problem7-approach23-arxiv-results.json")
    parser.add_argument("--top-markdown", default="data/first-proof/problem7-approach23-top20.md")
    parser.add_argument("--max-results", type=int, default=25)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    rows: list[dict] = []
    for q in QUERIES:
        try:
            hits = query_arxiv(q.text, args.max_results)
        except Exception as exc:  # noqa: BLE001
            rows.append({"query_id": q.id, "query_text": q.text, "focus": q.focus, "error": str(exc)})
            continue
        for hit in hits:
            hit["query_id"] = q.id
            hit["query_text"] = q.text
            hit["focus"] = q.focus
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
                "focuses": set(),
            },
        )
        merged["query_ids"].add(row["query_id"])
        merged["query_texts"].add(row["query_text"])
        merged["focuses"].add(row["focus"])

    merged_rows = []
    for row in by.values():
        qids = sorted(row["query_ids"])
        qtxts = sorted(row["query_texts"])
        foc = sorted(row["focuses"])
        merged_rows.append(
            {
                **{k: v for k, v in row.items() if k not in {"query_ids", "query_texts", "focuses"}},
                "query_ids": qids,
                "query_texts": qtxts,
                "focuses": foc,
                "query_hits": len(qids),
                "focus_hits": len(foc),
            }
        )

    for row in merged_rows:
        row["score"] = score_row(row, row["query_hits"], row["focus_hits"])

    merged_rows.sort(key=lambda x: (-x["score"], -x["focus_hits"], -x["query_hits"], x["date"]))
    top = merged_rows[: args.top_n]

    out = {
        "scope": "problem7-approach23",
        "queries": [{"id": q.id, "text": q.text, "focus": q.focus} for q in QUERIES],
        "unique_results": len(merged_rows),
        "top_n": args.top_n,
        "top_results": top,
        "results": merged_rows,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = [
        "# Problem 7 Approach II/III Focused arXiv Shortlist",
        "",
        "Derived from the 0fa4e82 guidance:",
        "- Approach II: equivariant surgery with codim-1 fixed sets",
        "- Approach III: orbifold resolution with pi_1 control",
        "- Cross-cutting: odd-dim E2 alternatives (even-dim fixed sets, chi=0)",
        "",
    ]
    for i, row in enumerate(top, start=1):
        foc = ", ".join(row["focuses"])
        lines.append(
            f"{i}. `{row['id']}` — {row['title']}  "
            f"(date: {row['date']}, score: {row['score']}, focuses: {foc})  "
            f"{row['url']}"
        )
    lines.append("")
    Path(args.top_markdown).write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {out_path}")
    print(f"Wrote {args.top_markdown}")
    print(f"unique_results={len(merged_rows)}")
    for row in top:
        print(
            f"{row['score']:2d} s | {row['focus_hits']} f | {row['query_hits']} q | "
            f"{row['date']} | {row['id']} | {row['title']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
