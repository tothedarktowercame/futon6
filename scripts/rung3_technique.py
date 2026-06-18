#!/usr/bin/env python3
"""Deterministic rung-3 technique coverage detector.

This promotes the rung-3-1 residue spike into per-paper technique gap maps.
It reuses cas_select retrieval/verification and emits bucketed moves suitable
for CAS-CERT's technique grain.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import cas_select  # noqa: E402
from rung3_residue_spike import AUTHOR_DECLARED_GAP_RE, VERIFIABLE_PATTERNS  # noqa: E402

DEFAULT_STEPS_DIR = ROOT / "data" / "cas-select-steps" / "loop-run-70b"
DEFAULT_FIXTURES = ROOT / "tests" / "fixtures" / "cas-select"
DEFAULT_OUT_DIR = ROOT / "data" / "rung3-technique" / "loop-run-70b"
BUCKETS = ("grounded-by-pattern", "grounded-by-citation", "thin", "ungrounded", "conjecture")


def pattern_type(pattern: str | None) -> str:
    if not pattern:
        return "none"
    return "verifiable" if pattern in VERIFIABLE_PATTERNS else "heuristic"


def bucket_for(*, text: str, pattern: str | None, ptype: str) -> str:
    if AUTHOR_DECLARED_GAP_RE.search(text):
        return "conjecture"
    if not pattern:
        return "ungrounded"
    if ptype == "verifiable":
        return "grounded-by-pattern"
    return "thin"


def gap_reason(bucket: str, pattern: str | None) -> str:
    if bucket == "thin":
        return f"matched heuristic pattern {pattern}; needs a verifiable discharge"
    if bucket == "ungrounded":
        return "no technique pattern verified"
    if bucket == "conjecture":
        return "author-declared gap credited as conjecture/open-status"
    return ""


def classify_step(
    step: dict[str, Any],
    patterns: dict[str, cas_select.Pattern],
    *,
    oracle: dict[str, Any] | None = None,
    k: int = 4,
) -> dict[str, Any]:
    candidates = cas_select.retrieve(step["text"], patterns, k=k)
    if oracle is not None:
        verdict = cas_select.verify(step, candidates, patterns, backend="stub", oracle=oracle)
        pattern = verdict.get("pattern")
        slot = verdict.get("slot")
        score = verdict.get("confidence", 0.0)
    else:
        top = candidates[0] if candidates else {}
        pattern = top.get("pattern")
        slot = None
        score = top.get("score", 0.0)
    ptype = pattern_type(pattern)
    bucket = bucket_for(text=step["text"], pattern=pattern, ptype=ptype)
    row = {
        "step": step["id"],
        "text": step["text"],
        "pattern": pattern,
        "type": ptype,
        "bucket": bucket,
        "score": score,
        "slot": slot,
        "candidates": [c["pattern"] for c in candidates],
    }
    if bucket == "conjecture":
        row["credited"] = True
    return row


def gapmap_for_steps(
    steps_doc: dict[str, Any],
    patterns: dict[str, cas_select.Pattern],
    *,
    oracle: dict[str, Any] | None = None,
    k: int = 4,
) -> dict[str, Any]:
    moves = [classify_step(step, patterns, oracle=oracle, k=k) for step in steps_doc["steps"]]
    counts = Counter(row["bucket"] for row in moves)
    gaps = []
    for row in moves:
        if row["bucket"] in {"grounded-by-pattern", "grounded-by-citation"}:
            continue
        gap = {
            "step": row["step"],
            "bucket": row["bucket"],
            "pattern": row["pattern"],
            "why": gap_reason(row["bucket"], row["pattern"]),
        }
        if row.get("credited"):
            gap["credited"] = True
        gaps.append(gap)
    return {
        "paper_id": steps_doc["paper_id"],
        "moves": moves,
        "buckets": {bucket: counts.get(bucket, 0) for bucket in BUCKETS},
        "gaps": gaps,
    }


def gapmap_from_cas_select_result(paper_id: str, result: dict[str, Any]) -> dict[str, Any]:
    moves = []
    for row in result.get("matches") or []:
        pattern = row.get("pattern")
        ptype = pattern_type(pattern)
        bucket = "grounded-by-pattern" if ptype == "verifiable" else "thin"
        moves.append(
            {
                "step": row.get("step"),
                "pattern": pattern,
                "type": ptype,
                "bucket": bucket,
                "score": row.get("score", 0.0),
                "slot": row.get("slot"),
                "candidates": [],
            }
        )
    for row in result.get("induce_queue") or []:
        candidates = row.get("candidates") or []
        pattern = candidates[0] if candidates else None
        ptype = pattern_type(pattern)
        bucket = "thin" if pattern and ptype == "heuristic" else "ungrounded"
        moves.append(
            {
                "step": row.get("step"),
                "pattern": pattern,
                "type": ptype,
                "bucket": bucket,
                "score": 0.0,
                "slot": None,
                "candidates": candidates,
            }
        )
    moves.sort(key=lambda row: str(row.get("step")))
    counts = Counter(row["bucket"] for row in moves)
    gaps = [
        {
            "step": row["step"],
            "bucket": row["bucket"],
            "pattern": row["pattern"],
            "why": gap_reason(row["bucket"], row["pattern"]),
        }
        for row in moves
        if row["bucket"] not in {"grounded-by-pattern", "grounded-by-citation"}
    ]
    return {
        "paper_id": paper_id,
        "moves": moves,
        "buckets": {bucket: counts.get(bucket, 0) for bucket in BUCKETS},
        "gaps": gaps,
    }


def write_gapmap(gapmap: dict[str, Any], out_dir: Path = DEFAULT_OUT_DIR, out_path: Path | None = None) -> Path:
    target = out_path or out_dir / f"{gapmap['paper_id']}.technique.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(gapmap, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    return target


def run_steps_paths(args: argparse.Namespace, paths: list[Path]) -> list[dict[str, Any]]:
    patterns = cas_select.load_patterns(index_path=args.index, library_dir=args.library)
    out = []
    for path in sorted(paths):
        steps_doc = cas_select.load_steps(path)
        out.append(gapmap_for_steps(steps_doc, patterns, k=args.k))
    return out


def run_fixture_dir(args: argparse.Namespace) -> list[dict[str, Any]]:
    patterns = cas_select.load_patterns(index_path=args.index, library_dir=args.library)
    out = []
    for path in sorted(args.fixtures.glob("*.steps.json")):
        steps_doc = cas_select.load_steps(path)
        oracle = cas_select.load_oracle(args.fixtures / f"{steps_doc['paper_id']}.oracle.json")
        out.append(gapmap_for_steps(steps_doc, patterns, oracle=oracle, k=args.k))
    return out


def run_cas_select_payload(args: argparse.Namespace) -> list[dict[str, Any]]:
    payload = json.loads(args.cas_select.read_text())
    return [
        gapmap_from_cas_select_result(str(paper_id), result)
        for paper_id, result in sorted((payload.get("results") or {}).items())
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", type=Path)
    ap.add_argument("--steps", type=Path)
    ap.add_argument("--steps-dir", type=Path, default=None)
    ap.add_argument("--cas-select", type=Path)
    ap.add_argument("--out", type=Path, help="Output path for a single gap-map")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--index", type=Path, default=cas_select.DEFAULT_INDEX)
    ap.add_argument("--library", type=Path, default=cas_select.DEFAULT_LIBRARY)
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args(argv)

    sources = [bool(args.fixtures), bool(args.steps), bool(args.steps_dir), bool(args.cas_select)]
    if sum(sources) != 1:
        ap.error("choose exactly one of --fixtures, --steps, --steps-dir, --cas-select")
    if args.out and not args.steps:
        ap.error("--out is only valid with --steps")

    if args.fixtures:
        gapmaps = run_fixture_dir(args)
    elif args.steps:
        gapmaps = run_steps_paths(args, [args.steps])
    elif args.steps_dir:
        gapmaps = run_steps_paths(args, list(args.steps_dir.glob("*.steps.json")))
    else:
        gapmaps = run_cas_select_payload(args)

    paths = [write_gapmap(gapmap, args.out_dir, args.out) for gapmap in gapmaps]
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
