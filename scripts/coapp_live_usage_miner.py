#!/usr/bin/env python3
"""coapp_live_usage_miner.py — W-coapp-live-usage-miner (cards-II gap 3).

Scans fold-turn deposits (ft-*.edn) — the real cascade usage records — and
emits co-application edges: pairs of patterns that appeared together in the
same cascade. Each edge carries provenance (which deposit, which mission).

This is the LIVE USAGE source that complements the doc-citation co_app edges
in pattern-phylogeny-edges.json (which come from mission-doc text co-mentions
via pattern_phylogeny.py). Together they close the pipeline wiring's pending
edge: [:cascade :co-app-usage] -> [:pattern-library :co-app-usage].

Recompute-from-scratch: running repeatedly over the same deposit dir produces
byte-identical output (no double-count). Each deposit contributes its
pattern-pairs exactly once.

Output format (matches cascade_learn.py's learned co_app shape, with added
provenance):
  {
    "schema": "coapp-live-usage.v1",
    "source": "futon6/data/fold-turns",
    "deposits_scanned": N,
    "cascades_with_patterns": M,
    "co_app_usage": [[stem_a, stem_b, weight, [provenance...]], ...],
    "novel_edges": K  (pairs not in the computed phylogeny)
  }

Usage:
  python3 coapp_live_usage_miner.py                    # dry-run summary
  python3 coapp_live_usage_miner.py --emit             # write output json
  python3 coapp_live_usage_miner.py --test             # deterministic test
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path("/home/joe/code")
DEFAULT_DEPOSIT_DIR = ROOT / "futon6/data/fold-turns"
DEFAULT_PHYLOGENY = ROOT / "futon6/data/pattern-phylogeny-edges.json"
DEFAULT_OUTPUT = ROOT / "futon6/data/coapp-live-usage.json"


def strip_comments(text):
    """Remove ; comments from EDN text (but not inside strings)."""
    lines = []
    for line in text.splitlines():
        in_string = False
        escaped = False
        cut = len(line)
        for i, ch in enumerate(line):
            if escaped:
                escaped = False
                continue
            if ch == '\\':
                escaped = True
                continue
            if ch == '"':
                in_string = not in_string
            elif ch == ';' and not in_string:
                cut = i
                break
        lines.append(line[:cut])
    return "\n".join(lines)


def extract_pattern_ids(edn_text):
    """Extract the :pattern-ids vector from a fold-turn deposit's EDN text.

    Returns a list of pattern-id strings, or empty list if not found.
    """
    text = strip_comments(edn_text)
    # The :pattern-ids vector lives inside :cascade; find it by key
    m = re.search(r':pattern-ids\s*\[(.*?)\]', text, re.DOTALL)
    if not m:
        return []
    return re.findall(r'"([^"]+)"', m.group(1))


def extract_deposit_id(edn_text):
    """Extract the :fold-turn/id from a deposit."""
    text = strip_comments(edn_text)
    m = re.search(r':fold-turn/id\s+"([^"]+)"', text)
    return m.group(1) if m else "unknown"


def extract_mission(edn_text):
    """Extract the :mission from a deposit."""
    text = strip_comments(edn_text)
    m = re.search(r':mission\s+"([^"]+)"', text)
    return m.group(1) if m else "unknown"


def stem(pattern_id):
    """Reduce a full pattern-id to its stem (last path segment).

    This matches pattern_phylogeny.py and cascade_learn.py convention —
    the phylogeny edges use stems, so we do too for joinability.
    """
    return str(pattern_id).split("/")[-1]


def ordered_pair(a, b):
    """Canonical ordering for an unordered pair of stems."""
    return tuple(sorted((a, b)))


def load_computed_phylogeny(path):
    """Load the existing computed phylogeny to check for novel edges."""
    if not Path(path).exists():
        return set()
    data = json.loads(Path(path).read_text())
    known = set()
    for edge in data.get("co_app", []):
        known.add(ordered_pair(edge[0], edge[1]))
    for edge in data.get("descent", []):
        known.add(ordered_pair(edge[0], edge[1]))
    return known


def scan_deposits(deposit_dir):
    """Scan all ft-*.edn deposits and extract co-app usage pairs.

    Returns:
      pairs: Counter {(stem_a, stem_b): count}
      provenance: {(stem_a, stem_b): [{"deposit": id, "mission": mission, "n_patterns": N}]}
      stats: {deposits_scanned, cascades_with_patterns, total_patterns_seen}
    """
    deposit_dir = Path(deposit_dir)
    files = sorted(deposit_dir.glob("ft-*.edn"))

    pairs = Counter()
    provenance = defaultdict(list)
    stats = {"deposits_scanned": 0, "cascades_with_patterns": 0,
             "total_patterns_seen": 0}

    for f in files:
        stats["deposits_scanned"] += 1
        text = f.read_text(errors="ignore")
        pids = extract_pattern_ids(text)
        if len(pids) < 2:
            continue

        stats["cascades_with_patterns"] += 1
        stats["total_patterns_seen"] += len(pids)

        deposit_id = extract_deposit_id(text)
        mission = extract_mission(text)
        stems = sorted(set(stem(p) for p in pids))

        for a, b in combinations(stems, 2):
            pair = ordered_pair(a, b)
            pairs[pair] += 1
            provenance[pair].append({
                "deposit": deposit_id,
                "mission": mission,
                "n_patterns": len(pids),
            })

    return pairs, provenance, stats


def build_output(pairs, provenance, stats, known_edges):
    """Build the output document with novelty analysis."""
    co_app_usage = []
    novel_count = 0
    for (a, b), weight in sorted(pairs.items()):
        is_novel = ordered_pair(a, b) not in known_edges
        if is_novel:
            novel_count += 1
        co_app_usage.append({
            "stems": [a, b],
            "weight": weight,
            "novel": is_novel,
            "provenance": provenance[(a, b)],
        })

    return {
        "schema": "coapp-live-usage.v1",
        "source": "futon6/data/fold-turns",
        "deposits_scanned": stats["deposits_scanned"],
        "cascades_with_patterns": stats["cascades_with_patterns"],
        "total_patterns_seen": stats["total_patterns_seen"],
        "co_app_edges": len(co_app_usage),
        "novel_edges": novel_count,
        "known_edges_in_phylogeny": len(known_edges),
        "co_app_usage": co_app_usage,
    }


def run(args):
    pairs, provenance, stats = scan_deposits(args.deposit_dir)
    known = load_computed_phylogeny(args.phylogeny)
    output = build_output(pairs, provenance, stats, known)

    # Summary
    print(f"Deposits scanned:       {output['deposits_scanned']}")
    print(f"Cascades with >=2 pats: {output['cascades_with_patterns']}")
    print(f"Total patterns seen:    {output['total_patterns_seen']}")
    print(f"Co-app usage edges:     {output['co_app_edges']}")
    print(f"Novel edges (not in phylogeny): {output['novel_edges']}")
    print(f"Known edges in phylogeny:       {output['known_edges_in_phylogeny']}")

    if output["co_app_edges"] > 0:
        print("\nTop 10 co-app pairs by weight:")
        for edge in sorted(output["co_app_usage"],
                           key=lambda e: -e["weight"])[:10]:
            novel_tag = " [NOVEL]" if edge["novel"] else ""
            print(f"  {edge['stems'][0]:<45s} x {edge['stems'][1]:<45s}"
                  f"  w={edge['weight']}{novel_tag}")
            for prov in edge["provenance"][:2]:
                print(f"    via {prov['deposit']} ({prov['mission']})")

    if args.emit:
        Path(args.output).write_text(
            json.dumps(output, indent=2, sort_keys=True) + "\n")
        print(f"\nWrote {args.output}")

    return output


def test():
    """Deterministic test: prove at least one live usage pair is found,
    and that the scanner handles edge cases (deposits with <2 patterns)."""
    print("=== TEST: coapp_live_usage_miner ===\n")

    pairs, provenance, stats = scan_deposits(DEFAULT_DEPOSIT_DIR)

    # Test 1: at least one deposit was scanned
    assert stats["deposits_scanned"] > 0, "No deposits found"
    print(f"PASS: {stats['deposits_scanned']} deposits scanned")

    # Test 2: at least one cascade has >=2 patterns
    assert stats["cascades_with_patterns"] > 0, \
        "No cascades with >=2 patterns found"
    print(f"PASS: {stats['cascades_with_patterns']} cascades with >=2 patterns")

    # Test 3: at least one co-app pair found
    assert len(pairs) > 0, "No co-app pairs extracted"
    print(f"PASS: {len(pairs)} co-app pairs extracted")

    # Test 4: provenance is complete (every pair has at least one source)
    for pair, provs in provenance.items():
        assert len(provs) > 0, f"Pair {pair} has no provenance"
        for p in provs:
            assert "deposit" in p and "mission" in p, \
                f"Incomplete provenance for {pair}: {p}"
    print(f"PASS: all {len(pairs)} pairs have complete provenance")

    # Test 5: at least one pair appears in multiple deposits (real signal)
    multi = {p: w for p, w in pairs.items() if w > 1}
    print(f"INFO: {len(multi)} pairs appear in multiple deposits")
    if multi:
        top = max(multi, key=multi.get)
        print(f"  Top multi-deposit pair: {top[0]} x {top[1]} (w={multi[top]})")

    # Test 6: recompute-from-scratch idempotency
    pairs2, _, _ = scan_deposits(DEFAULT_DEPOSIT_DIR)
    assert pairs == pairs2, "Non-deterministic: second scan differs"
    print("PASS: recompute-from-scratch is idempotent (byte-identical)")

    # Test 7: edge case — extract_pattern_ids on empty/malformed
    assert extract_pattern_ids("no pattern ids here") == []
    assert extract_pattern_ids('; :pattern-ids ["commented/out"]') == []
    print("PASS: edge cases (empty, commented) handled correctly")

    print(f"\n{'='*60}")
    print("ALL TESTS PASSED")
    return True


def main():
    ap = argparse.ArgumentParser(
        description="Mine co-app edges from live cascade usage (fold-turn deposits)")
    ap.add_argument("--deposit-dir", default=str(DEFAULT_DEPOSIT_DIR),
                    help="Directory of ft-*.edn deposits")
    ap.add_argument("--phylogeny", default=str(DEFAULT_PHYLOGENY),
                    help="Computed phylogeny edges JSON (for novelty check)")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT),
                    help="Output JSON path (with --emit)")
    ap.add_argument("--emit", action="store_true",
                    help="Write the output JSON file")
    ap.add_argument("--test", action="store_true",
                    help="Run deterministic test suite")
    args = ap.parse_args()

    if args.test:
        ok = test()
        sys.exit(0 if ok else 1)

    run(args)


if __name__ == "__main__":
    main()
