#!/usr/bin/env python3
"""A5: Add confidence tiers to thread-wiring-ct output.

Post-processes thread-wiring-ct.jsonl to add a confidence_tier field:
  - Tier 1 (IATC performatives): 66-72% alignment rate, usable for production
  - Tier 2 (categorical assertions): 3-5% consistency, experimental only

Also adds tier summary stats to each thread.

Usage:
    python scripts/add-confidence-tiers.py /path/to/processed-gpu/

Reads:  thread-wiring-ct.jsonl
Writes: thread-wiring-ct-tiered.jsonl
"""

import argparse
import json
import sys
from pathlib import Path


TIER1_EDGE_TYPES = {
    "assert", "challenge", "query", "clarify", "reform",
    "exemplify", "reference", "agree", "retract",
    "responds-to", "comment-on",
}

# Categorical annotations in nodes — always Tier 2
TIER2_CATEGORICAL_TYPES = {
    "cat/limit", "cat/equivalence", "cat/universal-property",
    "cat/natural-transformation", "cat/adjunction",
    "cat/monad", "cat/fibration", "cat/kan-extension",
}


def tier_thread(thread: dict) -> dict:
    """Add confidence_tier annotations to a thread entry."""
    # Tier edges
    for edge in thread.get("edges", []):
        iatc = edge.get("iatc", "")
        has_ports = len(edge.get("port_matches", [])) > 0

        if iatc in TIER1_EDGE_TYPES:
            edge["confidence_tier"] = 1
            edge["tier_basis"] = "iatc"
        elif has_ports:
            edge["confidence_tier"] = 2
            edge["tier_basis"] = "port_match"
        else:
            edge["confidence_tier"] = 1
            edge["tier_basis"] = "structural"

    # Tier categorical annotations on nodes
    for node in thread.get("nodes", []):
        for cat in node.get("categorical", []):
            cat_type = cat.get("type", "") if isinstance(cat, dict) else str(cat)
            if isinstance(cat, dict):
                cat["confidence_tier"] = 2
                cat["tier_basis"] = "categorical"

    # Add tier summary to stats
    stats = thread.get("stats", {})
    tier1_edges = sum(1 for e in thread.get("edges", [])
                      if e.get("confidence_tier") == 1)
    tier2_edges = sum(1 for e in thread.get("edges", [])
                      if e.get("confidence_tier") == 2)
    n_cat = stats.get("n_categorical", 0)

    stats["tier1_edges"] = tier1_edges
    stats["tier2_edges"] = tier2_edges
    stats["tier2_categorical"] = n_cat
    stats["tier1_fraction"] = (
        tier1_edges / (tier1_edges + tier2_edges)
        if (tier1_edges + tier2_edges) > 0 else 1.0
    )

    return thread


def process_file(data_dir: Path):
    input_path = data_dir / "thread-wiring-ct.jsonl"
    output_path = data_dir / "thread-wiring-ct-tiered.jsonl"

    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {input_path}...")
    total = 0
    tier1_total = 0
    tier2_total = 0
    cat_total = 0

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                thread = json.loads(line)
            except json.JSONDecodeError:
                continue

            thread = tier_thread(thread)
            fout.write(json.dumps(thread, ensure_ascii=False) + "\n")

            stats = thread.get("stats", {})
            tier1_total += stats.get("tier1_edges", 0)
            tier2_total += stats.get("tier2_edges", 0)
            cat_total += stats.get("tier2_categorical", 0)
            total += 1

    all_edges = tier1_total + tier2_total
    print(f"\nProcessed {total:,} threads")
    print(f"  Tier 1 edges (IATC/structural): {tier1_total:,} "
          f"({tier1_total/all_edges*100:.1f}%)" if all_edges else "")
    print(f"  Tier 2 edges (port/categorical): {tier2_total:,} "
          f"({tier2_total/all_edges*100:.1f}%)" if all_edges else "")
    print(f"  Tier 2 categorical annotations:  {cat_total:,}")
    print(f"\nWrote {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Add confidence tiers to thread wiring")
    parser.add_argument("data_dir", type=Path)
    args = parser.parse_args()
    process_file(args.data_dir)


if __name__ == "__main__":
    main()
