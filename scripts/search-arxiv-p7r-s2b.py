#!/usr/bin/env python3
"""Search arXiv for reduced-P7 node p7r-s2b (order-2 lattice instantiation).

Outputs:
  - JSON: data/first-proof/problem7r-s2b-arxiv-results.json
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


QUERIES: list[Query] = [
    Query("core-01", 'all:"Finite Groups and Hyperbolic Manifolds"'),
    Query("core-02", 'all:"Belolipetsky" AND all:"Lubotzky"'),
    Query("core-03", 'all:"uniform lattice" AND all:"order 2" AND all:"SO(n,1)"'),
    Query("core-04", 'all:"arithmetic lattice" AND all:"SO(n,1)" AND (all:"involution" OR all:"Z/2")'),
    Query("geom-01", 'all:"finite group actions" AND all:"hyperbolic manifolds"'),
    Query("geom-02", 'all:"involution" AND all:"hyperbolic manifold" AND all:"fixed points"'),
    Query("geom-03", 'all:"compact hyperbolic" AND all:"involution"'),
    Query("geom-04", 'all:"reflection" AND all:"hyperbolic manifold" AND all:"fixed"'),
    Query("orb-01", 'all:"orbifold fundamental group" AND all:"finite group action" AND all:"Euler characteristic"'),
    Query("orb-02", 'all:"uniform lattice" AND all:"torsion" AND all:"orbifold"'),
    Query("aux-01", 'all:"On the number of finite subgroups of a lattice"'),
    Query("aux-02", 'all:"Smith theory" AND all:"locally symmetric manifolds"'),
]


def query_arxiv(query: str, max_results: int) -> list[dict]:
    q = urllib.parse.quote(query)
    url = (
        "http://export.arxiv.org/api/query"
        f"?search_query={q}&start=0&max_results={max_results}"
        "&sortBy=relevance&sortOrder=descending"
    )
    data = urllib.request.urlopen(url, timeout=25).read()
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


def score_row(row: dict, query_hits: int) -> int:
    text = (row["title"] + " " + row["summary"]).lower()
    keys = [
        "hyperbolic manifold",
        "finite group",
        "fixed set",
        "fixed point",
        "involution",
        "z/2",
        "arithmetic",
        "lattice",
        "orbifold",
        "euler characteristic",
    ]
    score = 2 * query_hits
    for k in keys:
        if k in text:
            score += 1
    cat_text = ",".join(c.lower() for c in row.get("cats", []))
    if "math.gt" in cat_text:
        score += 1
    if "math.gr" in cat_text:
        score += 1
    return score


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="data/first-proof/problem7r-s2b-arxiv-results.json",
        help="Output JSON path",
    )
    parser.add_argument("--max-results", type=int, default=20)
    args = parser.parse_args()

    rows: list[dict] = []
    for q in QUERIES:
        try:
            hits = query_arxiv(q.text, args.max_results)
        except Exception as exc:  # noqa: BLE001
            rows.append({"query_id": q.id, "query_text": q.text, "error": str(exc)})
            continue
        for hit in hits:
            hit["query_id"] = q.id
            hit["query_text"] = q.text
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
            },
        )
        merged["query_ids"].add(row["query_id"])
        merged["query_texts"].add(row["query_text"])

    merged_rows = []
    for row in by.values():
        query_ids = sorted(row["query_ids"])
        query_texts = sorted(row["query_texts"])
        merged_rows.append(
            {
                **{k: v for k, v in row.items() if k not in {"query_ids", "query_texts"}},
                "query_ids": query_ids,
                "query_texts": query_texts,
                "query_hits": len(query_ids),
            }
        )

    for row in merged_rows:
        row["score"] = score_row(row, row["query_hits"])

    merged_rows.sort(key=lambda x: (-x["score"], -x["query_hits"], x["date"]))

    out = {
        "scope": "p7r-s2b",
        "queries": [{"id": q.id, "text": q.text} for q in QUERIES],
        "unique_results": len(merged_rows),
        "results": merged_rows,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"Wrote {out_path}")
    print(f"unique_results={len(merged_rows)}")
    for row in merged_rows[:20]:
        print(
            f"{row['score']:2d} s | {row['query_hits']} q | "
            f"{row['date']} | {row['id']} | {row['title']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
