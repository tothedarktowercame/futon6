#!/usr/bin/env python3
"""CPU-only scope/binder audit for SE entities and arXiv metadata.

Uses scripts/nlab-wiring.py detect_scopes() to report coverage and top scope types
on lightweight samples.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import Counter
from pathlib import Path



def load_detector():
    scripts_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(scripts_dir))
    module = importlib.import_module("nlab-wiring")
    return module.detect_scopes



def audit_se_entities(path: Path, sample: int, detect_scopes):
    if not path.exists():
        return None

    data = json.loads(path.read_text())
    rows = data[:sample] if sample > 0 else data

    type_counts = Counter()
    with_scopes = 0
    total_scopes = 0

    for i, row in enumerate(rows):
        text = " ".join([
            row.get("title", ""),
            row.get("question-body", ""),
            row.get("answer-body", ""),
        ]).strip()
        scopes = detect_scopes(f"se-{i}", text)
        if scopes:
            with_scopes += 1
            total_scopes += len(scopes)
        for s in scopes:
            type_counts[s.get("hx/type", "?")] += 1

    n = len(rows)
    return {
        "dataset": str(path),
        "sampled": n,
        "with_scopes": with_scopes,
        "scope_coverage": with_scopes / n if n else 0.0,
        "total_scopes": total_scopes,
        "top_types": type_counts.most_common(12),
    }



def audit_arxiv_jsonl(path: Path, sample: int, detect_scopes):
    if not path.exists():
        return None

    rows = []
    with path.open() as f:
        for i, line in enumerate(f):
            if sample > 0 and i >= sample:
                break
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    type_counts = Counter()
    with_scopes = 0
    total_scopes = 0

    for i, row in enumerate(rows):
        text = " ".join([
            row.get("title", ""),
            row.get("abstract", ""),
        ]).strip()
        scopes = detect_scopes(f"arxiv-{i}", text)
        if scopes:
            with_scopes += 1
            total_scopes += len(scopes)
        for s in scopes:
            type_counts[s.get("hx/type", "?")] += 1

    n = len(rows)
    return {
        "dataset": str(path),
        "sampled": n,
        "with_scopes": with_scopes,
        "scope_coverage": with_scopes / n if n else 0.0,
        "total_scopes": total_scopes,
        "top_types": type_counts.most_common(12),
    }



def print_report(report):
    if not report:
        return
    print(f"\n=== {report['dataset']} ===")
    print(f"sampled: {report['sampled']}")
    print(f"with_scopes: {report['with_scopes']}")
    print(f"scope_coverage: {report['scope_coverage']:.3f}")
    print(f"total_scopes: {report['total_scopes']}")
    print("top_types:")
    for t, n in report["top_types"]:
        print(f"  {n:6d}  {t}")



def main() -> int:
    parser = argparse.ArgumentParser(description="Audit scope/binder detection on local corpora")
    parser.add_argument("--se-entities", default="mo-processed/entities.json",
                        help="SE entities.json path (default: mo-processed/entities.json)")
    parser.add_argument("--arxiv-jsonl", default="data/arxiv-math-ct-metadata.jsonl",
                        help="arXiv metadata JSONL path")
    parser.add_argument("--sample", type=int, default=1000,
                        help="Rows to sample from each dataset (0 = full)")
    args = parser.parse_args()

    detect_scopes = load_detector()

    se_report = audit_se_entities(Path(args.se_entities), args.sample, detect_scopes)
    arxiv_report = audit_arxiv_jsonl(Path(args.arxiv_jsonl), args.sample, detect_scopes)

    if not se_report and not arxiv_report:
        print("No datasets found for audit.")
        return 1

    print_report(se_report)
    print_report(arxiv_report)
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
