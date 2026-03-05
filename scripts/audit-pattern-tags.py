#!/usr/bin/env python3
"""A6: Pattern tag precision audit.

Samples 200 entities from pattern-tags.json, displays each entity's question,
answer, and assigned tags for human judgment.  Writes verdicts to a JSONL audit
file, then computes per-pattern precision at the end.

Usage:
    python3 scripts/audit-pattern-tags.py --source math   # interactive audit
    python3 scripts/audit-pattern-tags.py --source math --report  # report only (from existing verdicts)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

STORAGE = {
    "math": Path(os.path.expanduser("~/code/storage/math-processed-gpu")),
    "mo": Path(os.path.expanduser("~/code/storage/mo-processed-gpu")),
}

SAMPLE_SIZE = 200
SEED = 42


def load_entities(source: str) -> dict[str, dict]:
    """Load the parsed QA entities keyed by entry_id."""
    outdir = STORAGE[source]
    entities_path = outdir / "entities.json"
    if not entities_path.exists():
        # Fall back: try to reconstruct from the SE parse output
        entities_path = outdir / "se-parsed.json"
    if not entities_path.exists():
        print(f"Cannot find entities at {entities_path}", file=sys.stderr)
        sys.exit(1)
    with entities_path.open() as f:
        data = json.load(f)
    if isinstance(data, list):
        return {r.get("entry_id", r.get("id", str(i))): r for i, r in enumerate(data)}
    return data


def load_pattern_tags(source: str) -> list[dict]:
    path = STORAGE[source] / "pattern-tags.json"
    with path.open() as f:
        return json.load(f)


def audit_file(source: str) -> Path:
    return STORAGE[source] / "pattern-tag-audit.jsonl"


def load_existing_verdicts(source: str) -> dict[str, dict]:
    path = audit_file(source)
    verdicts = {}
    if path.exists():
        with path.open() as f:
            for line in f:
                rec = json.loads(line)
                verdicts[rec["entry_id"]] = rec
    return verdicts


def sample_entries(tags: list[dict], n: int, seed: int) -> list[dict]:
    """Stratified sample: ensure at least a few entries per pattern."""
    rng = random.Random(seed)

    # Group entries by their patterns
    by_pattern: dict[str, list[dict]] = defaultdict(list)
    for rec in tags:
        for p in rec.get("patterns", []):
            by_pattern[p].append(rec)

    seen_ids = set()
    sampled = []

    # Take at least 3 per pattern (if available)
    per_pattern = max(3, n // len(by_pattern)) if by_pattern else n
    for pat in sorted(by_pattern):
        pool = [r for r in by_pattern[pat] if r["entry_id"] not in seen_ids]
        pick = rng.sample(pool, min(per_pattern, len(pool)))
        for r in pick:
            if r["entry_id"] not in seen_ids:
                seen_ids.add(r["entry_id"])
                sampled.append(r)

    # Fill remainder randomly
    remainder = [r for r in tags if r["entry_id"] not in seen_ids and r.get("patterns")]
    rng.shuffle(remainder)
    for r in remainder:
        if len(sampled) >= n:
            break
        sampled.append(r)
        seen_ids.add(r["entry_id"])

    rng.shuffle(sampled)
    return sampled[:n]


def interactive_audit(source: str) -> None:
    tags = load_pattern_tags(source)
    entities = load_entities(source)
    sample = sample_entries(tags, SAMPLE_SIZE, SEED)
    verdicts = load_existing_verdicts(source)
    out_path = audit_file(source)

    already = len(verdicts)
    remaining = [r for r in sample if r["entry_id"] not in verdicts]
    print(f"Audit: {len(sample)} sampled, {already} already judged, "
          f"{len(remaining)} remaining\n")

    with out_path.open("a") as out:
        for i, rec in enumerate(remaining):
            eid = rec["entry_id"]
            ent = entities.get(eid, {})
            q_title = ent.get("question_title", ent.get("title", ""))
            q_text = ent.get("question_text", ent.get("question", ""))[:400]
            a_text = ent.get("answer_text", ent.get("answer", ""))[:400]
            pats = rec.get("patterns", [])

            print(f"--- [{i+1}/{len(remaining)}] {eid} ---")
            print(f"Title: {q_title}")
            print(f"Q: {q_text}")
            print(f"A: {a_text}")
            print(f"Tags: {pats}")
            print()

            tag_verdicts = {}
            for pat in pats:
                while True:
                    v = input(f"  '{pat}' correct? [y/n/s(kip)/q(uit)] ").strip().lower()
                    if v in ("y", "n", "s", "q"):
                        break
                if v == "q":
                    print("Quitting. Progress saved.")
                    return
                if v == "s":
                    tag_verdicts[pat] = "skip"
                else:
                    tag_verdicts[pat] = v == "y"

            verdict_rec = {
                "entry_id": eid,
                "patterns": pats,
                "verdicts": tag_verdicts,
            }
            out.write(json.dumps(verdict_rec) + "\n")
            out.flush()
            print()

    print("Audit complete.")
    report(source)


def report(source: str) -> None:
    verdicts = load_existing_verdicts(source)
    if not verdicts:
        print("No verdicts found.")
        return

    per_pattern: dict[str, Counter] = defaultdict(Counter)
    total_correct = 0
    total_judged = 0

    for rec in verdicts.values():
        for pat, v in rec.get("verdicts", {}).items():
            if v == "skip":
                continue
            per_pattern[pat]["total"] += 1
            if v is True:
                per_pattern[pat]["correct"] += 1
                total_correct += 1
            total_judged += 1

    print(f"\n=== Pattern Tag Precision Report ({source}) ===")
    print(f"Entries judged: {len(verdicts)}")
    print(f"Tag judgments: {total_judged}")
    if total_judged:
        print(f"Overall precision: {total_correct/total_judged:.1%}")
    print()
    print(f"{'Pattern':<40s} {'Correct':>8s} {'Total':>8s} {'Precision':>10s}")
    print("-" * 68)
    for pat in sorted(per_pattern):
        c = per_pattern[pat]
        t = c["total"]
        cor = c["correct"]
        prec = cor / t if t else 0
        print(f"{pat:<40s} {cor:>8d} {t:>8d} {prec:>9.1%}")


def main():
    parser = argparse.ArgumentParser(description="A6: Pattern tag precision audit")
    parser.add_argument("--source", choices=["math", "mo"], default="math")
    parser.add_argument("--report", action="store_true",
                        help="Show report from existing verdicts only")
    args = parser.parse_args()

    if args.report:
        report(args.source)
    else:
        interactive_audit(args.source)


if __name__ == "__main__":
    main()
