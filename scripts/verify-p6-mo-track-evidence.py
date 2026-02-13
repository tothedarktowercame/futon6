#!/usr/bin/env python3
"""Build independent MathOverflow evidence scorecards for Problem 6 strategy tracks.

Scans local StackExchange XML dump (MathOverflow) and reports per-track keyword hits.
"""

from __future__ import annotations

import argparse
import datetime as dt
import heapq
import html
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple


TRACKS: List[Dict[str, object]] = [
    {
        "id": "A",
        "name": "Strongly Rayleigh / edge-negative-dependence transfer",
        "terms": [
            "strongly rayleigh",
            "real stable",
            "negative dependence",
            "negative association",
            "determinantal point process",
            "determinantal",
            "random spanning tree",
            "uniform spanning tree",
            "matroid basis",
        ],
        "context": [
            "graph",
            "matrix",
            "laplacian",
            "spectral",
            "spanning tree",
            "sparsifier",
        ],
    },
    {
        "id": "B",
        "name": "Hyperbolic barrier / hyperbolicity cone",
        "terms": [
            "hyperbolic polynomial",
            "hyperbolicity cone",
            "garding inequality",
            "self-concordant barrier",
            "log-det barrier",
        ],
        "context": [
            "matrix",
            "eigenvalue",
            "spectral",
            "laplacian",
            "sparsifier",
            "hyperbolic",
        ],
    },
    {
        "id": "C",
        "name": "Interlacing / KS / mixed-characteristic",
        "terms": [
            "interlacing polynomial",
            "interlacing families",
            "mixed characteristic polynomial",
            "kadison-singer",
            "weaver",
            "real-rooted",
            "ramanujan graph",
        ],
        "context": [
            "graph",
            "matrix",
            "operator",
            "spectrum",
            "eigenvalue",
            "polynomial",
        ],
    },
    {
        "id": "D",
        "name": "Near-rank-1 reformulation",
        "terms": [
            "rank one matrix",
            "rank-1 matrix",
            "best rank one approximation",
            "rank one perturbation",
            "principal eigenvector",
            "top eigenvector",
            "effective rank",
            "stable rank",
        ],
        "context": [
            "matrix",
            "eigenvalue",
            "eigenvector",
            "spectral",
            "laplacian",
            "graph",
        ],
    },
    {
        "id": "E",
        "name": "Graph-adaptive transfer constant (expansion/spectral geometry)",
        "terms": [
            "expander",
            "expansion",
            "conductance",
            "cheeger",
            "spectral gap",
            "algebraic connectivity",
            "effective resistance",
            "laplacian eigenvalue",
            "isoperimetric",
        ],
        "context": [
            "graph",
            "laplacian",
            "spectral",
            "eigenvalue",
            "conductance",
            "cheeger",
        ],
    },
    {
        "id": "F",
        "name": "Potential-budget greedy barrier (BSS-like trajectory)",
        "terms": [
            "batson",
            "spielman",
            "srivastava",
            "sparsifier",
            "sparsification",
            "bss sparsifier",
            "sherman morrison",
            "matrix freedman",
            "matrix bernstein",
            "matrix martingale",
        ],
        "context": [
            "graph",
            "laplacian",
            "matrix",
            "sparsifier",
            "sparsification",
            "barrier",
            "potential",
            "martingale",
        ],
    },
]


HTML_TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")


def norm_text(s: str) -> str:
    t = html.unescape(s or "")
    t = HTML_TAG_RE.sub(" ", t)
    t = t.replace("|", " ")
    t = t.lower()
    t = SPACE_RE.sub(" ", t).strip()
    return t


def compile_term_patterns(terms: List[str]) -> Dict[str, re.Pattern[str]]:
    pats: Dict[str, re.Pattern[str]] = {}
    for term in terms:
        parts = [re.escape(p) for p in term.split()]
        if len(parts) == 1:
            expr = r"(?<![a-z0-9])" + parts[0] + r"(?![a-z0-9])"
        else:
            expr = r"(?<![a-z0-9])" + r"\s+".join(parts) + r"(?![a-z0-9])"
        pats[term] = re.compile(expr)
    return pats


def has_any_term(pats: Dict[str, re.Pattern[str]], text: str) -> bool:
    for pat in pats.values():
        if pat.search(text):
            return True
    return False


@dataclass
class Hit:
    post_id: int
    year: int
    score: int
    title: str
    tags: str
    matched_terms: List[str]
    title_or_tags_hits: int


class TopHits:
    def __init__(self, k: int) -> None:
        self.k = k
        self.heap: List[Tuple[Tuple[int, int, int], int, Hit]] = []
        self._counter = 0

    def push(self, hit: Hit) -> None:
        key = (len(hit.matched_terms), hit.title_or_tags_hits, hit.score)
        self._counter += 1
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, (key, self._counter, hit))
            return
        if key > self.heap[0][0]:
            heapq.heapreplace(self.heap, (key, self._counter, hit))

    def sorted_hits(self) -> List[Hit]:
        return [h for _, _, h in sorted(self.heap, key=lambda x: x[0], reverse=True)]


def confidence_label(strong_hits: int) -> str:
    if strong_hits >= 80:
        return "high"
    if strong_hits >= 30:
        return "medium"
    if strong_hits >= 10:
        return "low-medium"
    return "low"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--posts", required=True, type=Path)
    ap.add_argument("--out-md", required=True, type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    compiled = {
        tr["id"]: {
            "terms": compile_term_patterns(tr["terms"]),  # type: ignore[index]
            "context": compile_term_patterns(tr["context"]),  # type: ignore[index]
        }
        for tr in TRACKS
    }

    stats = {
        tr["id"]: {
            "name": tr["name"],
            "terms": tr["terms"],
            "weak_hits": 0,
            "strong_hits": 0,
            "anchored_weak_hits": 0,
            "anchored_strong_hits": 0,
            "top": TopHits(args.top_k),
        }
        for tr in TRACKS
    }

    total_questions = 0
    for _, elem in ET.iterparse(args.posts, events=("end",)):
        if elem.tag != "row":
            continue
        a = elem.attrib
        if a.get("PostTypeId") != "1":
            elem.clear()
            continue

        total_questions += 1
        title_raw = a.get("Title", "")
        body_raw = a.get("Body", "")
        tags_raw = a.get("Tags", "")

        title = norm_text(title_raw)
        body = norm_text(body_raw)
        tags = norm_text(tags_raw)
        full = f"{title} {tags} {body}"

        year = 0
        created = a.get("CreationDate", "")
        if len(created) >= 4 and created[:4].isdigit():
            year = int(created[:4])

        try:
            score = int(a.get("Score", "0"))
        except ValueError:
            score = 0

        try:
            pid = int(a.get("Id", "0"))
        except ValueError:
            pid = 0

        for tr in TRACKS:
            tid = tr["id"]  # type: ignore[index]
            pats = compiled[tid]["terms"]
            matched: List[str] = []
            title_tags_hits = 0

            title_tags_blob = f"{title} {tags}"
            for term, pat in pats.items():
                if pat.search(full):
                    matched.append(term)
                    if pat.search(title_tags_blob):
                        title_tags_hits += 1

            if not matched:
                continue

            anchored = has_any_term(compiled[tid]["context"], full)
            stats[tid]["weak_hits"] += 1
            if anchored:
                stats[tid]["anchored_weak_hits"] += 1
            if len(matched) >= 2:
                stats[tid]["strong_hits"] += 1
                if anchored:
                    stats[tid]["anchored_strong_hits"] += 1

            hit = Hit(
                post_id=pid,
                year=year,
                score=score,
                title=title_raw,
                tags=tags_raw,
                matched_terms=sorted(matched),
                title_or_tags_hits=title_tags_hits,
            )
            stats[tid]["top"].push(hit)

        elem.clear()

    now_utc = dt.datetime.now(dt.timezone.utc)
    json_out = {
        "generated": now_utc.isoformat(),
        "posts_path": str(args.posts),
        "total_questions": total_questions,
        "tracks": [],
    }

    lines: List[str] = []
    lines.append("# Problem 6: MathOverflow Evidence Scan for Strategy Tracks")
    lines.append("")
    lines.append(f"Generated: {now_utc.isoformat()}")
    lines.append(f"Source: `{args.posts}`")
    lines.append(f"Question rows scanned: `{total_questions}`")
    lines.append("")
    lines.append(
        "Method: independent keyword bundles per track (A-F), with `strong hit` = >=2 matched terms in a question."
    )
    lines.append("This is evidence mapping, not proof of applicability.")
    lines.append("")

    for tr in TRACKS:
        tid = tr["id"]  # type: ignore[index]
        st = stats[tid]
        top_hits = st["top"].sorted_hits()

        lines.append(f"## Track {tid}: {st['name']}")
        lines.append("")
        lines.append(f"- Terms: {', '.join('`'+t+'`' for t in st['terms'])}")
        lines.append(f"- Weak hits (>=1 term): `{st['weak_hits']}`")
        lines.append(f"- Strong hits (>=2 terms): `{st['strong_hits']}`")
        lines.append(f"- Anchored weak hits (>=1 term + context): `{st['anchored_weak_hits']}`")
        lines.append(f"- Anchored strong hits (>=2 terms + context): `{st['anchored_strong_hits']}`")
        lines.append(f"- Prior signal: `{confidence_label(st['anchored_strong_hits'])}`")
        lines.append("")
        lines.append("Top MO question hits:")
        lines.append("")
        lines.append("| MO id | Year | Score | #terms | Title |")
        lines.append("|---|---:|---:|---:|---|")
        for h in top_hits[: args.top_k]:
            title = h.title.replace("|", "\\|")
            lines.append(
                f"| [{h.post_id}](https://mathoverflow.net/questions/{h.post_id}) | {h.year} | {h.score} | {len(h.matched_terms)} | {title} |"
            )
        if not top_hits:
            lines.append("| - | - | - | - | No matches |")
        lines.append("")

        json_out["tracks"].append(
            {
                "id": tid,
                "name": st["name"],
                "terms": st["terms"],
                "weak_hits": st["weak_hits"],
                "strong_hits": st["strong_hits"],
                "anchored_weak_hits": st["anchored_weak_hits"],
                "anchored_strong_hits": st["anchored_strong_hits"],
                "prior_signal": confidence_label(st["anchored_strong_hits"]),
                "top_hits": [asdict(h) for h in top_hits],
            }
        )

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.out_json.write_text(json.dumps(json_out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
