#!/usr/bin/env python3
"""H7 per-paper citation resolution for mark3/P6.

This is the per-paper half of citation resolution: combine layer-(a) cite marks
from `fable-<id>-dp-emacs.json`, the existing warp bibliography/citation index,
and the existing CT corpus arXiv id set. It emits one standoff JSON file per
paper. A record is resolved only when the target exists in the corpus id set;
otherwise it carries an honest `hole`.

Output schema: `futon6/h7-cite-resolution/v1`.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GH200 = ROOT / "data" / "warp" / "gh200.txt"
DEFAULT_GOLDEN = ROOT / "data" / "showcases" / "ct-anatomy" / "golden"
DEFAULT_BIB_INDEX = ROOT / "data" / "warp" / "bib-index.json"
DEFAULT_CITATIONS = ROOT / "data" / "warp" / "citations.json"
DEFAULT_CORPUS_INDEX = Path("/home/joe/code/storage/futon6/data/arxiv-math-ct-file-index.jsonl")
DEFAULT_OUT = ROOT / "data" / "warp" / "cite-resolution"
SCHEMA = "futon6/h7-cite-resolution/v1"


ARXIV_PATTERNS = [
    # New-style numeric id (e.g. 2401.14311). Require an arXiv prefix so we never
    # collide with DOIs (10.1006/aima.1993.1055) or volume.page tokens. NB: the
    # separator class uses a real \s — the previous \\s/\\. matched a literal
    # backslash, so new-style ids were never extracted.
    re.compile(r"arxiv[:\s.-]*([0-9]{4}\.[0-9]{4,5})(?:v[0-9]+)?", re.I),
    # Old-style archive/NNNNNNN id (math/9811139, hep-th/9901001, cond-mat/0011001).
    # Unambiguous, so also catch it bare in prose ("Also available as math/9811139").
    re.compile(r"(?:arxiv[:\s.-]*)?\b([a-z][a-z-]+/[0-9]{7})(?:v[0-9]+)?\b", re.I),
]


@dataclass(frozen=True)
class CorpusIds:
    canonical: set[str]
    safe: set[str]
    safe_to_canonical: dict[str, str]
    canonical_to_safe: dict[str, str]
    titles_by_safe: dict[str, str]


def safe_id(arxiv_id: str) -> str:
    return arxiv_id.replace("/", "__")


def canonical_id(arxiv_or_safe: str, ids: CorpusIds) -> str:
    return ids.safe_to_canonical.get(arxiv_or_safe, arxiv_or_safe.replace("__", "/"))


def corpus_safe_id(arxiv_or_safe: str, ids: CorpusIds) -> str:
    if arxiv_or_safe in ids.safe:
        return arxiv_or_safe
    return ids.canonical_to_safe.get(arxiv_or_safe, safe_id(arxiv_or_safe))


def load_corpus_ids(path: Path) -> CorpusIds:
    canonical: set[str] = set()
    safe: set[str] = set()
    safe_to_canonical: dict[str, str] = {}
    canonical_to_safe: dict[str, str] = {}
    titles_by_safe: dict[str, str] = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            cid = row["id"]
            sid = row.get("safe_id") or safe_id(cid)
            canonical.add(cid)
            safe.add(sid)
            safe_to_canonical[sid] = cid
            canonical_to_safe[cid] = sid
            if row.get("title"):
                titles_by_safe[sid] = row["title"]
    return CorpusIds(canonical, safe, safe_to_canonical, canonical_to_safe, titles_by_safe)


def load_bib_index(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text())
    return {paper["paper_id"]: paper for paper in data.get("papers", [])}


def load_citation_edges(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    data = json.loads(path.read_text())
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for edge in data.get("edges", []):
        via = edge.get("via") or {}
        paper = edge.get("from")
        key = via.get("key")
        if paper and key:
            current = by_key.get((paper, key))
            if current is None or float((via or {}).get("score", 1.0)) > float((current.get("via") or {}).get("score", 0.0)):
                by_key[(paper, key)] = edge
    return by_key


def field_map(mark: dict[str, Any]) -> dict[str, str]:
    return {str(k): str(v) for k, v in mark.get("fields", []) if isinstance(k, str)}


def cite_keys(mark: dict[str, Any]) -> list[str]:
    fields = field_map(mark)
    raw = fields.get("cite", "")
    return [key.strip() for key in raw.split(",") if key.strip()]


def cite_marks(marks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        mark for mark in marks
        if mark.get("layer") == "dp" and mark.get("kind") == "cite" and cite_keys(mark)
    ]


def bib_maps(paper: dict[str, Any] | None) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    items = list((paper or {}).get("bibitems") or [])
    by_key = {item.get("key"): item for item in items if item.get("key")}
    markers = {item.get("key"): f"[{i}]" for i, item in enumerate(items, start=1) if item.get("key")}
    return by_key, markers


def arxiv_id_from_bibitem(item: dict[str, Any] | None) -> str | None:
    if not item:
        return None
    if item.get("arxiv_id"):
        return str(item["arxiv_id"]).replace("__", "/")
    raw = str(item.get("raw") or "")
    for pattern in ARXIV_PATTERNS:
        match = pattern.search(raw)
        if match:
            return match.group(1)
    return None


def title_for(target_safe: str | None, via: dict[str, Any] | None, bibitem: dict[str, Any] | None,
              ids: CorpusIds) -> str | None:
    if target_safe and ids.titles_by_safe.get(target_safe):
        return ids.titles_by_safe[target_safe]
    if via and via.get("title"):
        return str(via["title"])
    if bibitem and bibitem.get("title"):
        return str(bibitem["title"])
    return None


def resolve_key(paper_id: str, key: str, citation_edges: dict[tuple[str, str], dict[str, Any]],
                bibitem: dict[str, Any] | None, ids: CorpusIds) -> dict[str, Any]:
    edge = citation_edges.get((paper_id, key))
    via = (edge or {}).get("via") or {}
    candidate = (edge or {}).get("to")
    method = via.get("method")
    confidence = float(via.get("score", 1.0 if candidate else 0.0))

    if candidate is None:
        direct = arxiv_id_from_bibitem(bibitem)
        if direct:
            candidate = direct
            method = "bibitem-arxiv-id"
            confidence = 0.98

    if candidate:
        target_safe = corpus_safe_id(str(candidate), ids)
        target_canonical = canonical_id(str(candidate), ids)
        if target_safe in ids.safe and target_canonical in ids.canonical:
            return {
                "resolved-arxiv-id": target_canonical,
                "resolved-corpus-id": target_safe,
                "title": title_for(target_safe, via, bibitem, ids),
                "confidence": round(confidence, 4),
                "method": method or "citation-index",
                "hole": None,
            }
        return {
            "resolved-arxiv-id": None,
            "resolved-corpus-id": None,
            "title": title_for(None, via, bibitem, ids),
            "confidence": 0.0,
            "method": "hole",
            "hole": {
                "kind": "unresolved-citation",
                "reason": "candidate-not-in-corpus-id-set",
                "candidate": str(candidate),
                "bibitem": (bibitem or {}).get("raw"),
            },
        }

    return {
        "resolved-arxiv-id": None,
        "resolved-corpus-id": None,
        "title": title_for(None, via, bibitem, ids),
        "confidence": 0.0,
        "method": "hole",
        "hole": {
            "kind": "unresolved-citation",
            "reason": "no-corpus-match",
            "bibitem": (bibitem or {}).get("raw"),
        },
    }


def resolve_paper(marks_path: Path, bib_index: dict[str, dict[str, Any]],
                  citation_edges: dict[tuple[str, str], dict[str, Any]], ids: CorpusIds,
                  *, source_paths: dict[str, str]) -> dict[str, Any]:
    data = json.loads(marks_path.read_text())
    # The fable JSON `paper` field is sometimes the render stem (`2311.05789-dp`);
    # the warp bibliography/citation indexes use the arXiv-safe paper id. The
    # filename is the stable join key.
    paper_id = marks_path.name.removeprefix("fable-").removesuffix("-dp-emacs.json")
    text = data.get("text") or ""
    paper_bib = bib_index.get(paper_id)
    by_key, markers = bib_maps(paper_bib)
    records: list[dict[str, Any]] = []
    for mark in cite_marks(data.get("marks") or []):
        start, end = int(mark["start"]), int(mark["end"])
        raw = text[start:end]
        keys = cite_keys(mark)
        group_marker = "[" + ",".join(markers.get(key, "?") for key in keys).replace("[", "").replace("]", "") + "]"
        for key in keys:
            bibitem = by_key.get(key)
            resolved = resolve_key(paper_id, key, citation_edges, bibitem, ids)
            records.append({
                "cite/marker": markers.get(key, "[?]"),
                "cite/group-marker": group_marker,
                "cite/key": key,
                "cite/raw": raw,
                "char-anchor": [start, end],
                **resolved,
            })
    total = len(records)
    resolved_n = sum(1 for row in records if row["resolved-arxiv-id"])
    hole_n = total - resolved_n
    return {
        "schema": SCHEMA,
        "paper-id": paper_id,
        "source": {
            "marks-path": str(marks_path),
            **source_paths,
        },
        "records": records,
        "stats": {
            "total": total,
            "resolved": resolved_n,
            "holes": hole_n,
            "resolution-rate": round((resolved_n / total), 4) if total else 1.0,
        },
    }


def iter_papers(args: argparse.Namespace) -> list[str]:
    if args.paper:
        return args.paper
    ids = [line.strip() for line in args.gh200.read_text().splitlines() if line.strip()]
    if args.sample_size:
        ids = ids[:args.sample_size]
    return ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="append", help="paper id to resolve; repeatable")
    ap.add_argument("--sample-size", type=int, default=20, help="first N gh200 ids when --paper is omitted")
    ap.add_argument("--gh200", type=Path, default=DEFAULT_GH200)
    ap.add_argument("--golden-dir", type=Path, default=DEFAULT_GOLDEN)
    ap.add_argument("--bib-index", type=Path, default=DEFAULT_BIB_INDEX)
    ap.add_argument("--citations", type=Path, default=DEFAULT_CITATIONS)
    ap.add_argument("--corpus-index", type=Path, default=DEFAULT_CORPUS_INDEX)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    ids = load_corpus_ids(args.corpus_index)
    bib = load_bib_index(args.bib_index)
    citation_edges = load_citation_edges(args.citations)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    source_paths = {
        "bib-index": str(args.bib_index),
        "citations-index": str(args.citations),
        "corpus-id-index": str(args.corpus_index),
    }

    summaries = []
    skipped = []
    for paper_id in iter_papers(args):
        marks_path = args.golden_dir / f"fable-{paper_id}-dp-emacs.json"
        if not marks_path.exists():
            skipped.append({"paper-id": paper_id, "reason": "missing-marks-json"})
            continue
        result = resolve_paper(marks_path, bib, citation_edges, ids, source_paths=source_paths)
        out_path = args.out_dir / f"{paper_id}.cite-resolution.json"
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True))
        summaries.append({"paper-id": paper_id, "path": str(out_path), **result["stats"]})

    total = sum(row["total"] for row in summaries)
    resolved_n = sum(row["resolved"] for row in summaries)
    holes = sum(row["holes"] for row in summaries)
    manifest = {
        "schema": "futon6/h7-cite-resolution-run/v1",
        "output-schema": SCHEMA,
        "out-dir": str(args.out_dir),
        "papers": summaries,
        "skipped": skipped,
        "stats": {
            "papers-written": len(summaries),
            "papers-skipped": len(skipped),
            "total": total,
            "resolved": resolved_n,
            "holes": holes,
            "resolution-rate": round((resolved_n / total), 4) if total else 1.0,
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps(manifest["stats"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
