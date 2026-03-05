#!/usr/bin/env python3
"""ArSE (Artificial Stack Exchange) hotspot loop.

Weaves together the hotspot stepper, corpus retrieval, and synthetic QA
generation into a single workflow:

  1. Read hotspot manifest → identify unresolved proof nodes
  2. Retrieve relevant real corpus threads per hotspot (FAISS + text)
  3. For each hotspot, generate the questions the proof node needs answered
  4. Produce synthetic QA in hypergraph-native format
  5. Output enriched corpus context for the stepper

This is the ArSE self-play loop applied to proof improvement: the proof
nodes are the Asker (they generate questions via gap analysis), the LLM
is the Answerer, the hypergraph schema is the structure, and the hotspot
stepper is the evaluator.

Usage:
    # Full pipeline: retrieve + generate + prepare stepper
    python3 scripts/run-arse-hotspot-loop.py --problem 7

    # Just retrieve + analyze (no generation)
    python3 scripts/run-arse-hotspot-loop.py --problem 7 --retrieve-only

    # Generate synthetic QA from existing retrieval
    python3 scripts/run-arse-hotspot-loop.py --problem 7 --generate-only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python3"


def find_hotspot_manifest(problem: int) -> Path | None:
    """Find the most recent hotspot stepper manifest."""
    stepper_dir = REPO_ROOT / "data" / "first-proof" / "stepper"
    candidates = sorted(
        stepper_dir.glob(f"problem{problem}-hotspot-stepper-*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_hotspots(manifest_path: Path) -> dict:
    return json.loads(manifest_path.read_text())


def find_hotspot_md(problem: int) -> Path | None:
    stepper_dir = REPO_ROOT / "data" / "first-proof" / "stepper"
    candidates = sorted(
        stepper_dir.glob(f"problem{problem}-hotspots-*.md"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def parse_hotspot_nodes(md_path: Path) -> list[dict]:
    """Parse the hotspot markdown table to get node stats."""
    nodes = []
    lines = md_path.read_text().splitlines()
    in_table = False
    for line in lines:
        if line.startswith("| Node"):
            in_table = True
            continue
        if in_table and line.startswith("|---"):
            continue
        if in_table and line.startswith("|"):
            parts = [p.strip() for p in line.split("|")[1:-1]]
            if len(parts) >= 7:
                nodes.append({
                    "node_id": parts[0],
                    "observations": int(parts[1]),
                    "verified": int(parts[2]),
                    "plausible": int(parts[3]),
                    "gap": int(parts[4]),
                    "error": int(parts[5]),
                    "parse_error": int(parts[6]),
                    "unresolved_pct": float(parts[7].rstrip("%")) if len(parts) > 7 else 100.0,
                })
        elif in_table and not line.strip():
            in_table = False
    return nodes


def run_retrieval(problem: int, python: str) -> Path:
    """Run retrieve-proof-context.py and return the output path."""
    output = REPO_ROOT / "data" / "first-proof" / f"problem{problem}-corpus-context.jsonl"
    cmd = [
        python,
        str(REPO_ROOT / "scripts" / "retrieve-proof-context.py"),
        "--problem", str(problem),
        "--top-k", "5",
        "--structural-expand", "3",
        "--output", str(output),
    ]
    print(f"\n[1/3] Retrieving corpus context...")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Retrieval failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    print(result.stdout)
    print(f"  Retrieved in {time.time()-t0:.1f}s")
    return output


def run_synthetic_generation(problem: int, python: str, node_ids: list[str]) -> Path:
    """Run generate-synthetic-qa.py for specific nodes."""
    output = REPO_ROOT / "data" / "synthetic-qa" / f"problem{problem}-prompts.jsonl"
    cmd = [
        python,
        str(REPO_ROOT / "scripts" / "generate-synthetic-qa.py"),
        "--problem", str(problem),
        "--dry-run",
    ]
    for nid in node_ids:
        cmd.extend(["--node-id", nid])

    print(f"\n[2/3] Generating synthetic QA prompts for {len(node_ids)} hotspot nodes...")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Generation failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    print(result.stdout)
    return output


def build_enriched_stepper_manifest(
    problem: int,
    original_manifest: dict,
    corpus_context_path: Path,
) -> Path:
    """Build a new stepper manifest that uses --corpus-context."""
    output_ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_jsonl = (
        f"data/first-proof/problem{problem}-codex-results-arse-{output_ts}.jsonl"
    )
    prompts_jsonl = (
        f"data/first-proof/problem{problem}-codex-prompts-arse-{output_ts}.jsonl"
    )

    # Take the original intervention command and replace/add --corpus-context
    orig_intervention = original_manifest.get("interventions", [{}])[0]
    orig_cmd = list(orig_intervention.get("command", []))

    # Build new command
    new_cmd = []
    skip_next = False
    removed_math_se = False
    for i, arg in enumerate(orig_cmd):
        if skip_next:
            skip_next = False
            continue
        if arg == "--math-se-dir":
            skip_next = True  # remove --math-se-dir and its value
            removed_math_se = True
            continue
        if arg == "--output" and i + 1 < len(orig_cmd):
            new_cmd.append(arg)
            new_cmd.append(output_jsonl)
            skip_next = True
            continue
        if arg == "--prompts-out" and i + 1 < len(orig_cmd):
            new_cmd.append(arg)
            new_cmd.append(prompts_jsonl)
            skip_next = True
            continue
        new_cmd.append(arg)

    # Add corpus context
    new_cmd.extend(["--corpus-context", str(corpus_context_path)])

    manifest = {
        "experiment_id": f"p{problem}-arse-hotspots-{output_ts}",
        "wiring": original_manifest.get("wiring"),
        "compare_script": original_manifest.get("compare_script"),
        "run": {"execute": False, "stop_on_error": True},
        "baseline": original_manifest.get("baseline"),
        "interventions": [
            {
                "label": "arse-enriched-hotspot",
                "command": new_cmd,
                "output_jsonl": output_jsonl,
                "notes": (
                    "Hotspot-only run with pre-retrieved corpus context from ArSE loop. "
                    "Uses FAISS structural expansion + text embedding reranking "
                    "instead of filesystem browsing."
                ),
                "hypothesis": (
                    "Pre-retrieved, structurally-expanded corpus context should "
                    "improve verification quality and eliminate timeout failures."
                ),
            }
        ],
        "learning_log": {
            "decision_rule": (
                "Compare node-level claim_verified distributions between "
                "legacy baseline and ArSE-enriched run. Adopt if gap rate decreases."
            ),
            "questions": [
                "Did pre-retrieved context eliminate filesystem browsing timeouts?",
                "Did structural FAISS expansion surface relevant threads that keywords missed?",
                "Which hotspot nodes moved from gap/plausible to verified?",
                "What was the quality of synthetic QA threads (if used)?",
            ],
        },
    }

    out_path = (
        REPO_ROOT / "data" / "first-proof" / "stepper"
        / f"problem{problem}-arse-stepper-{output_ts}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--problem", type=int, default=7)
    parser.add_argument("--retrieve-only", action="store_true",
                        help="Only run retrieval, skip generation")
    parser.add_argument("--generate-only", action="store_true",
                        help="Only generate synthetic QA from existing retrieval")
    parser.add_argument("--python", default=str(VENV_PYTHON),
                        help="Python interpreter (default: .venv/bin/python3)")
    args = parser.parse_args()

    # Find hotspot data
    hotspot_md = find_hotspot_md(args.problem)
    manifest_path = find_hotspot_manifest(args.problem)

    if not hotspot_md:
        print(f"No hotspot markdown found for problem {args.problem}")
        print("Run: python3 scripts/prepare-proof-hotspot-stepper.py --problem "
              f"{args.problem}")
        return 2

    # Parse hotspot nodes
    hotspot_nodes = parse_hotspot_nodes(hotspot_md)
    print(f"Problem {args.problem} hotspots ({hotspot_md.name}):")
    print(f"{'Node':<20s} {'Obs':>5s} {'Verified':>9s} {'Gap':>5s} {'Unresolved':>11s}")
    print("-" * 55)
    for n in hotspot_nodes:
        print(f"{n['node_id']:<20s} {n['observations']:>5d} {n['verified']:>9d} "
              f"{n['gap']:>5d} {n['unresolved_pct']:>10.1f}%")

    # Priority: gap nodes first, then all-plausible nodes
    gap_nodes = [n for n in hotspot_nodes if n["gap"] > 0]
    plausible_nodes = [n for n in hotspot_nodes if n["gap"] == 0 and n["verified"] == 0]
    priority_node_ids = [n["node_id"] for n in gap_nodes + plausible_nodes]

    print(f"\nPriority nodes ({len(priority_node_ids)}): {priority_node_ids}")

    # Step 1: Retrieve
    if not args.generate_only:
        context_path = run_retrieval(args.problem, args.python)
    else:
        context_path = (
            REPO_ROOT / "data" / "first-proof"
            / f"problem{args.problem}-corpus-context.jsonl"
        )
        if not context_path.exists():
            print(f"No existing context at {context_path}")
            return 2

    if args.retrieve_only:
        print(f"\nRetrieval complete. Context at: {context_path}")
        return 0

    # Step 2: Generate synthetic QA prompts
    prompts_path = run_synthetic_generation(args.problem, args.python, priority_node_ids)

    # Step 3: Build enriched stepper manifest
    if manifest_path:
        manifest = load_hotspots(manifest_path)
        stepper_path = build_enriched_stepper_manifest(
            args.problem, manifest, context_path,
        )
        print(f"\n[3/3] Enriched stepper manifest: {stepper_path}")
        print(f"\nTo run the ArSE-enriched stepper:")
        print(f"  python3 scripts/run-proof-stepper.py --manifest {stepper_path}")
    else:
        print(f"\nNo stepper manifest found — skipping manifest generation")

    print(f"\n{'='*60}")
    print(f"ArSE hotspot loop complete for problem {args.problem}")
    print(f"  Corpus context:     {context_path}")
    print(f"  Synthetic prompts:  {prompts_path}")
    if manifest_path:
        print(f"  Stepper manifest:   {stepper_path}")
    print(f"\nNext steps:")
    print(f"  1. Review synthetic QA prompts in {prompts_path}")
    print(f"  2. Run prompts through API to generate QA pairs")
    print(f"  3. Run enriched stepper to measure improvement")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
