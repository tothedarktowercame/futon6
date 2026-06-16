#!/usr/bin/env python3
"""Checker for H7 citation-resolution JSON outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_INDEX = Path("/home/joe/code/storage/futon6/data/arxiv-math-ct-file-index.jsonl")
DEFAULT_OUT = ROOT / "data" / "warp" / "cite-resolution"
SCHEMA = "futon6/h7-cite-resolution/v1"


def safe_id(arxiv_id: str) -> str:
    return arxiv_id.replace("/", "__")


def load_corpus_ids(path: Path) -> tuple[set[str], set[str]]:
    canonical: set[str] = set()
    safe: set[str] = set()
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            canonical.add(row["id"])
            safe.add(row.get("safe_id") or safe_id(row["id"]))
    return canonical, safe


def output_files(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(p for p in path.glob("*.cite-resolution.json"))
    return [path]


def check_record(file: Path, paper_id: str, idx: int, rec: dict[str, Any],
                 canonical_ids: set[str], safe_ids: set[str]) -> list[str]:
    errors: list[str] = []
    prefix = f"{file}:{paper_id}:records[{idx}]"
    for key in ["cite/marker", "cite/key", "char-anchor", "confidence", "method"]:
        if key not in rec:
            errors.append(f"{prefix}: missing {key}")
    anchor = rec.get("char-anchor")
    if not (isinstance(anchor, list) and len(anchor) == 2 and all(isinstance(x, int) for x in anchor)
            and anchor[0] <= anchor[1]):
        errors.append(f"{prefix}: invalid char-anchor {anchor!r}")
    resolved = rec.get("resolved-arxiv-id")
    corpus = rec.get("resolved-corpus-id")
    hole = rec.get("hole")
    if resolved:
        if resolved not in canonical_ids:
            errors.append(f"{prefix}: resolved-arxiv-id not in corpus canonical id set: {resolved}")
        if not corpus:
            errors.append(f"{prefix}: resolved record missing resolved-corpus-id")
        elif corpus not in safe_ids:
            errors.append(f"{prefix}: resolved-corpus-id not in corpus safe id set: {corpus}")
        if hole is not None:
            errors.append(f"{prefix}: resolved record must have hole=null")
    else:
        if not isinstance(hole, dict):
            errors.append(f"{prefix}: unresolved record must carry hole map")
        elif hole.get("kind") != "unresolved-citation":
            errors.append(f"{prefix}: unexpected hole kind {hole.get('kind')!r}")
    return errors


def check_file(file: Path, canonical_ids: set[str], safe_ids: set[str]) -> list[str]:
    errors: list[str] = []
    data = json.loads(file.read_text())
    if data.get("schema") != SCHEMA:
        errors.append(f"{file}: schema mismatch {data.get('schema')!r}")
    paper_id = data.get("paper-id")
    if not paper_id:
        errors.append(f"{file}: missing paper-id")
    records = data.get("records")
    if not isinstance(records, list):
        errors.append(f"{file}: records must be a list")
        return errors
    for idx, rec in enumerate(records):
        if not isinstance(rec, dict):
            errors.append(f"{file}:{paper_id}:records[{idx}] is not a map")
            continue
        errors.extend(check_record(file, paper_id, idx, rec, canonical_ids, safe_ids))
    stats = data.get("stats") or {}
    resolved_n = sum(1 for rec in records if isinstance(rec, dict) and rec.get("resolved-arxiv-id"))
    holes = len(records) - resolved_n
    if stats.get("total") != len(records):
        errors.append(f"{file}: stats.total {stats.get('total')} != records {len(records)}")
    if stats.get("resolved") != resolved_n:
        errors.append(f"{file}: stats.resolved {stats.get('resolved')} != {resolved_n}")
    if stats.get("holes") != holes:
        errors.append(f"{file}: stats.holes {stats.get('holes')} != {holes}")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--corpus-index", type=Path, default=DEFAULT_CORPUS_INDEX)
    args = ap.parse_args()

    canonical_ids, safe_ids = load_corpus_ids(args.corpus_index)
    files = output_files(args.path)
    errors: list[str] = []
    for file in files:
        errors.extend(check_file(file, canonical_ids, safe_ids))
    if errors:
        for error in errors:
            print(error)
        return 1
    print(f"checked {len(files)} citation-resolution file(s): OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
