#!/usr/bin/env python3
"""Build a curated paper knowledge base for Problem 7 gaps."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

NS = {"a": "http://www.w3.org/2005/Atom"}

PAPERS = [
    {"id": "1204.4667", "tags": ["G1", "E2"], "why": "FH(Q) criterion and fixed-set Euler input."},
    {"id": "1506.06293", "tags": ["G2"], "why": "Rational manifold models and rational surgery pipeline."},
    {"id": "1311.7629", "tags": ["G1"], "why": "Bredon-Poincare duality framework with torsion."},
    {"id": "0705.3249", "tags": ["G1"], "why": "Orbifold/groupoid cohomological infrastructure."},
    {"id": "math/0312378", "tags": ["G1", "FJ"], "why": "Classifying spaces for families, E_FIN/E_VCYC context."},
    {"id": "math/0510602", "tags": ["FJ"], "why": "Assembly-map and K/L-theory framework survey."},
    {"id": "1003.5002", "tags": ["FJ"], "why": "K/L-theory overview with Farrell-Jones context."},
    {"id": "1007.0845", "tags": ["FJ", "U3"], "why": "Computational K/L-theory patterns for virtually cyclic actions."},
    {"id": "1805.00226", "tags": ["FJ"], "why": "Modern isomorphism-conjecture survey and assembly interpretations."},
    {"id": "1101.0469", "tags": ["FJ"], "why": "FJ for cocompact lattices in virtually connected Lie groups."},
    {"id": "1401.0876", "tags": ["FJ"], "why": "FJ for arbitrary lattices in virtually connected Lie groups."},
    {"id": "0905.0104", "tags": ["G3"], "why": "Closed-manifold subgroup and surgery-obstruction interface."},
    {"id": "math/0306054", "tags": ["U3"], "why": "Explicit L-group computations with torsion/orientation characters."},
    {"id": "1707.07960", "tags": ["G1"], "why": "Finiteness obstruction background and Wall-style perspectives."},
    {"id": "math/0008070", "tags": ["G1"], "why": "Finiteness/PD obstruction background used in reduced wiring."},
    {"id": "1705.10909", "tags": ["G2", "G3"], "why": "Equivariant Spivak normal bundle and equivariant surgery."},
    {"id": "2304.00880", "tags": ["G1", "G3"], "why": "Non-simply-connected rational homotopy models; reference candidate."},
    {"id": "1106.1704", "tags": ["E2"], "why": "Smith-theory constraints for periodic actions on locally symmetric spaces."},
    {"id": "2506.23994", "tags": ["E2"], "why": "Reflection congruence-manifold construction with fixed hypersurfaces."},
    {"id": "math/0406607", "tags": ["E2"], "why": "Finite-group realization in compact hyperbolic manifolds."},
    {"id": "1209.2484", "tags": ["E2"], "why": "Finite subgroup/isotropy complexity bounds for lattices."},
    {"id": "2012.15322", "tags": ["E2", "FJ"], "why": "Hyperbolic orbifold homology bounds and volume-scaling controls."},
]


def normalize_id(raw: str) -> str:
    return re.sub(r"v\d+$", "", raw.strip())


def _fetch(url: str, retries: int = 3) -> bytes:
    last_exc: Exception | None = None
    for i in range(retries):
        try:
            return urllib.request.urlopen(url, timeout=30).read()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(1 + i)
    assert last_exc is not None
    raise last_exc


def fetch_by_ids(ids: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for rid in ids:
        rid = normalize_id(rid)
        query = rid if "/" in rid else f"id:{rid}"
        q = urllib.parse.quote(query, safe="/")
        url = (
            "http://export.arxiv.org/api/query"
            f"?search_query={q}&start=0&max_results=1"
            "&sortBy=relevance&sortOrder=descending"
        )
        data = _fetch(url)
        root = ET.fromstring(data)
        hit = False
        for entry in root.findall("a:entry", NS):
            hit = True
            aid = entry.findtext("a:id", "", NS)
            m = re.search(r"arxiv\.org/abs/([^\s]+)", aid or "")
            raw_id = m.group(1) if m else (aid.rsplit("/", 1)[-1] if aid else "")
            arx = normalize_id(raw_id)
            title = " ".join((entry.findtext("a:title", "", NS) or "").split())
            date = (entry.findtext("a:published", "", NS) or "")[:10]
            authors = [x.findtext("a:name", "", NS) for x in entry.findall("a:author", NS)]
            cats = [x.attrib.get("term", "") for x in entry.findall("a:category", NS)]
            summary = " ".join((entry.findtext("a:summary", "", NS) or "").split())
            out[rid] = {
                "id": rid,
                "title": title,
                "date": date,
                "authors": authors,
                "cats": cats,
                "summary": summary,
                "url": f"https://arxiv.org/abs/{arx}",
            }
        if not hit:
            out[rid] = {
                "id": rid,
                "title": "(metadata fetch failed)",
                "date": "",
                "authors": [],
                "cats": [],
                "summary": "",
                "url": f"https://arxiv.org/abs/{rid}",
            }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", default="data/first-proof/problem7-gap-paper-kb.json")
    parser.add_argument("--md-output", default="data/first-proof/problem7-gap-paper-kb.md")
    args = parser.parse_args()

    wanted = [normalize_id(p["id"]) for p in PAPERS]
    meta = fetch_by_ids(wanted)

    rows = []
    for p in PAPERS:
        pid = normalize_id(p["id"])
        m = meta.get(
            pid,
            {
                "id": pid,
                "title": "(metadata fetch failed)",
                "date": "",
                "authors": [],
                "cats": [],
                "summary": "",
                "url": f"https://arxiv.org/abs/{pid}",
            },
        )
        rows.append({**m, "tags": p["tags"], "why": p["why"]})

    by_tag: dict[str, list[dict]] = {}
    for r in rows:
        for t in r["tags"]:
            by_tag.setdefault(t, []).append(r)

    out = {
        "scope": "problem7-gaps-kb",
        "count": len(rows),
        "papers": rows,
        "by_tag_counts": {k: len(v) for k, v in sorted(by_tag.items())},
    }

    jpath = Path(args.json_output)
    jpath.parent.mkdir(parents=True, exist_ok=True)
    jpath.write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = [
        "# Problem 7 Gap Paper KB (circa 20 papers)",
        "",
        "Curated set tied to current gap labels `G1/G2/G3/U1-U3/FJ/E2`.",
        "",
        f"Total papers: {len(rows)}",
        "",
        "## Tag Legend",
        "- `G1`: rational PD / Poincare-complex bridge",
        "- `G2`: normal map / Spivak reduction / rational surgery setup",
        "- `G3`: transfer compatibility for restricted obstructions",
        "- `U1-U3`: conjectural lemmas in equivariant-localization step",
        "- `FJ`: Farrell-Jones / assembly / UNil infrastructure",
        "- `E2`: reflection-lattice and fixed-set input branch",
        "",
        "## Papers",
        "",
    ]

    for i, r in enumerate(rows, start=1):
        tags = ", ".join(r["tags"])
        author_txt = ", ".join([a for a in r["authors"] if a])
        lines.append(f"{i}. `{r['id']}` — {r['title']}")
        lines.append(f"   - Tags: {tags}")
        lines.append(f"   - Why: {r['why']}")
        lines.append(f"   - Date: {r['date']}")
        lines.append(f"   - Authors: {author_txt}")
        lines.append(f"   - URL: {r['url']}")

    mpath = Path(args.md_output)
    mpath.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {jpath}")
    print(f"Wrote {mpath}")
    print(f"count={len(rows)}")
    print("tag-counts:")
    for k, v in sorted(out["by_tag_counts"].items()):
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
