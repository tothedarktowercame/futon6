#!/usr/bin/env python3
"""Prepare ASE work queue for Tickle overnight runs.

Converts the synthetic QA prompts (from generate-synthetic-qa.py) into
a format consumable by Tickle's work queue machinery:

  - entities.json: list of work items with IDs, titles, prompts
  - review-prompts.json: corresponding review prompts for Claude
  - queue-manifest.json: metadata for progress tracking

The output goes to futon6/data/ase-queue/ where tickle_work_queue.clj
can pick it up alongside the CT entities.

Can also generate prompts for Frontier Math problems when wiring
diagrams exist.

Usage:
    # Prepare queue for P7 hotspots
    python3 scripts/prepare-ase-tickle-queue.py --problem 7

    # Prepare for multiple problems
    python3 scripts/prepare-ase-tickle-queue.py --problem 3 7

    # Include all problems that have wiring diagrams
    python3 scripts/prepare-ase-tickle-queue.py --all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ASE_QUEUE_DIR = REPO_ROOT / "data" / "ase-queue"


def load_prompts(problem: int) -> list[dict]:
    """Load synthetic QA prompts for a problem."""
    path = REPO_ROOT / "data" / "synthetic-qa" / f"problem{problem}-prompts.jsonl"
    if not path.exists():
        return []
    prompts = []
    with path.open() as f:
        for line in f:
            prompts.append(json.loads(line))
    return prompts


def build_review_prompt(work_item: dict) -> str:
    """Build a review prompt for Claude to evaluate a generated QA pair."""
    return (
        "Runtime surface contract:\n"
        "- Agent: claude-1 (Tickle ASE work queue — review mode)\n"
        f"- Task: Review synthetic QA pair for proof node: {work_item['node-id']}\n"
        "- Your verdict will be recorded as evidence.\n\n"
        "## Review Criteria\n\n"
        "1. **Mathematical correctness**: Is the question well-posed? "
        "Is the answer mathematically rigorous and correct?\n"
        "2. **Gap targeting**: Does this QA actually address the identified gap, "
        "or is it generic/off-topic?\n"
        "3. **Hypergraph quality**: Are the nodes and edges well-typed? "
        "Do they use the correct type vocabulary (post, term, expression, scope)? "
        "Are edges meaningful (not just connecting everything to everything)?\n"
        "4. **Composability**: Would this thread be useful as retrieval context "
        "for the proof step it targets?\n"
        "5. **LaTeX quality**: Are formulas correct and well-formatted?\n\n"
        "## Verdict\n\n"
        "Reply with a JSON object:\n"
        "```json\n"
        '{"verdict": "accept"|"revise"|"reject",\n'
        ' "correctness": 1-5,\n'
        ' "gap_relevance": 1-5,\n'
        ' "hypergraph_quality": 1-5,\n'
        ' "notes": "free text"}\n'
        "```\n"
    )


def find_problems_with_wiring() -> list[int]:
    """Find all problems that have wiring diagrams."""
    problems = []
    for f in sorted((REPO_ROOT / "data" / "first-proof").glob("problem*-wiring.json")):
        try:
            num = int(f.name.replace("problem", "").replace("-wiring.json", ""))
            problems.append(num)
        except ValueError:
            pass
    return problems


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--problem", type=int, nargs="+", default=[])
    parser.add_argument("--all", action="store_true",
                        help="Include all problems with wiring diagrams")
    parser.add_argument("--output-dir", type=Path, default=ASE_QUEUE_DIR)
    args = parser.parse_args()

    if args.all:
        problems = find_problems_with_wiring()
    elif args.problem:
        problems = args.problem
    else:
        problems = [7]  # default

    print(f"Preparing ASE queue for problems: {problems}")

    all_work_items = []
    all_review_prompts = {}

    for problem in problems:
        prompts = load_prompts(problem)
        if not prompts:
            print(f"  P{problem}: no prompts found — run generate-synthetic-qa.py first")
            continue

        for rec in prompts:
            work_id = rec["thread_id"]
            work_item = {
                "entity-id": work_id,
                "title": f"[ASE P{problem}] {rec['node_id']} — synthetic QA #{rec['instance']}",
                "type": "synthetic-qa",
                "corpus": "ase",
                "problem": problem,
                "node-id": rec["node_id"],
                "instance": rec["instance"],
                "gap-severity": rec["gap_severity"],
                "prompt": rec["prompt"],
                "response-schema": "synthetic-hypergraph",
            }
            all_work_items.append(work_item)
            all_review_prompts[work_id] = build_review_prompt(work_item)

        print(f"  P{problem}: {len(prompts)} work items")

    if not all_work_items:
        print("No work items generated.")
        return 2

    # Write queue files
    args.output_dir.mkdir(parents=True, exist_ok=True)

    entities_path = args.output_dir / "entities.json"
    with entities_path.open("w") as f:
        json.dump(all_work_items, f, indent=1, ensure_ascii=False)

    reviews_path = args.output_dir / "review-prompts.json"
    with reviews_path.open("w") as f:
        json.dump(all_review_prompts, f, indent=1, ensure_ascii=False)

    manifest = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "problems": problems,
        "n_work_items": len(all_work_items),
        "response_schema": "synthetic-hypergraph",
        "estimated_time_minutes": len(all_work_items) * 3,  # ~3 min per item
        "queue_type": "ase",
    }
    manifest_path = args.output_dir / "queue-manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nQueue prepared: {args.output_dir}")
    print(f"  {len(all_work_items)} work items")
    print(f"  Est. time: ~{manifest['estimated_time_minutes']} minutes")
    print(f"\nFiles:")
    print(f"  {entities_path}")
    print(f"  {reviews_path}")
    print(f"  {manifest_path}")
    print(f"\nTo run overnight via Tickle:")
    print(f"  ;; In futon3c REPL:")
    print(f"  (dev/run-ase-batch! :n {len(all_work_items)})")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
