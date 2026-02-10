#!/usr/bin/env python3
"""Validate superpod pipeline output against golden-50 test suite.

Memory-safe: streams through large pipeline output files, only keeping
the 50 golden entries in memory. Will not OOM on 400MB+ JSON files.

Usage:
    python scripts/validate-golden.py <pipeline-output-dir>
    python scripts/validate-golden.py --self-check

Exit codes:
    0 — all checks pass
    1 — some checks fail (see report)
    2 — missing files or bad input
"""

import json
import sys
from pathlib import Path

GOLDEN = Path("tests/golden/golden-50.json")

# Minimum thresholds for passing
MIN_NER_RECALL = 0.50
MIN_NER_PRECISION = 0.30
MIN_PATTERN_AGREEMENT = 0.40
MIN_ENTITY_COVERAGE = 0.90
MIN_SCOPE_RECALL = 0.30


def load_golden():
    """Load golden-50 — small file, safe to load fully."""
    with open(GOLDEN) as f:
        data = json.load(f)
    return {e["id"]: e for e in data["entries"]}


def stream_entity_ids(path):
    """Stream entity IDs from entities.json without loading full file.

    Uses ijson-style manual parsing: read line-by-line looking for entity/id.
    Falls back to chunked scanning if needed.
    """
    ids = set()
    with open(path) as f:
        # Scan for "entity/id" keys without loading entire JSON
        buf = ""
        for line in f:
            # Fast scan: look for the entity/id pattern in each line
            idx = 0
            while True:
                pos = line.find('"entity/id"', idx)
                if pos == -1:
                    break
                # Find the value after the colon
                rest = line[pos + 11:]
                colon = rest.find(":")
                if colon != -1:
                    rest = rest[colon + 1:].lstrip()
                    if rest.startswith('"'):
                        end = rest.find('"', 1)
                        if end != -1:
                            ids.add(rest[1:end])
                idx = pos + 11
    return ids


def stream_lookup_json_list(path, id_key, wanted_ids):
    """Stream through a JSON array file, extracting only entries with IDs in wanted_ids.

    Reads the file in chunks, parsing one object at a time.
    Returns dict {id: entry} for matched entries only.
    """
    results = {}
    remaining = set(wanted_ids)
    if not remaining:
        return results

    with open(path) as f:
        content = f.read(1)  # skip opening [
        if content != '[':
            # Might be a dict or other format — fall back to small-file load
            f.seek(0)
            data = json.load(f)
            if isinstance(data, list):
                for entry in data:
                    eid = entry.get(id_key)
                    if eid in remaining:
                        results[eid] = entry
                        remaining.discard(eid)
                        if not remaining:
                            break
            return results

        # Stream through JSON array using a bracket-counting parser
        depth = 0
        obj_start = False
        obj_chars = []

        while remaining:
            chunk = f.read(8192)
            if not chunk:
                break
            for ch in chunk:
                if ch == '{':
                    if depth == 0:
                        obj_start = True
                        obj_chars = [ch]
                    else:
                        obj_chars.append(ch)
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    obj_chars.append(ch)
                    if depth == 0 and obj_start:
                        obj_str = ''.join(obj_chars)
                        # Quick check: does this object contain any wanted ID?
                        found_id = None
                        for wid in remaining:
                            if wid in obj_str:
                                found_id = wid
                                break
                        if found_id:
                            try:
                                entry = json.loads(obj_str)
                                eid = entry.get(id_key)
                                if eid in remaining:
                                    results[eid] = entry
                                    remaining.discard(eid)
                            except json.JSONDecodeError:
                                pass
                        obj_chars = []
                        obj_start = False
                elif obj_start:
                    obj_chars.append(ch)

    return results


def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def precision_recall(predicted, golden):
    if not predicted and not golden:
        return 1.0, 1.0
    if not predicted:
        return 0.0, 0.0
    if not golden:
        return 0.0, 1.0
    tp = len(predicted & golden)
    return tp / len(predicted), tp / len(golden)


def validate(outdir):
    golden = load_golden()
    golden_ids = set(golden.keys())
    print(f"Golden test suite: {len(golden)} entries")
    print(f"Pipeline output:   {outdir}/\n")

    checks = []
    warnings = []

    # --- Check 1: Entity coverage (stream, don't load) ---
    ent_path = outdir / "entities.json"
    if not ent_path.exists():
        print("FAIL: entities.json not found")
        return 2

    print("  Scanning entities.json for golden IDs...")
    pipeline_ids = stream_entity_ids(ent_path)
    covered = golden_ids & pipeline_ids
    coverage = len(covered) / len(golden_ids) if golden_ids else 0
    check_pass = coverage >= MIN_ENTITY_COVERAGE
    checks.append(("entity-coverage", check_pass,
                    f"{coverage:.0%} ({len(covered)}/{len(golden_ids)})"))
    print(f"[{'PASS' if check_pass else 'FAIL'}] Entity coverage: {coverage:.0%}")

    # --- Check 2: NER term recall/precision (stream lookup) ---
    ner_path = outdir / "ner-terms.json"
    if ner_path.exists():
        print("  Scanning ner-terms.json for golden entries...")
        ner_matches = stream_lookup_json_list(ner_path, "entity_id", covered)
        all_precision = []
        all_recall = []
        for eid in covered:
            golden_terms = set(golden[eid].get("golden_ner_terms", []))
            entry = ner_matches.get(eid)
            if entry:
                pipeline_terms = set(t["term_lower"] if isinstance(t, dict)
                                     else t for t in entry.get("terms", []))
            else:
                pipeline_terms = set()
            p, r = precision_recall(pipeline_terms, golden_terms)
            all_precision.append(p)
            all_recall.append(r)

        avg_p = sum(all_precision) / len(all_precision) if all_precision else 0
        avg_r = sum(all_recall) / len(all_recall) if all_recall else 0
        p_pass = avg_p >= MIN_NER_PRECISION
        r_pass = avg_r >= MIN_NER_RECALL
        checks.append(("ner-precision", p_pass, f"{avg_p:.0%}"))
        checks.append(("ner-recall", r_pass, f"{avg_r:.0%}"))
        print(f"[{'PASS' if p_pass else 'FAIL'}] NER precision: {avg_p:.0%}")
        print(f"[{'PASS' if r_pass else 'FAIL'}] NER recall:    {avg_r:.0%}")
        del ner_matches  # free memory
    else:
        warnings.append("ner-terms.json not found")
        print("[SKIP] NER terms: not found")

    # --- Check 3: Pattern tag agreement (stream lookup) ---
    pat_path = outdir / "pattern-tags.json"
    if pat_path.exists():
        print("  Scanning pattern-tags.json for golden entries...")
        pat_matches = stream_lookup_json_list(pat_path, "entry_id", covered)
        all_jaccard = []
        for eid in covered:
            golden_patterns = set(golden[eid].get("golden_pattern_names", []))
            entry = pat_matches.get(eid)
            pipeline_patterns = set(entry.get("patterns", [])) if entry else set()
            all_jaccard.append(jaccard(golden_patterns, pipeline_patterns))

        avg_j = sum(all_jaccard) / len(all_jaccard) if all_jaccard else 0
        j_pass = avg_j >= MIN_PATTERN_AGREEMENT
        checks.append(("pattern-agreement", j_pass, f"{avg_j:.0%}"))
        print(f"[{'PASS' if j_pass else 'FAIL'}] Pattern agreement: {avg_j:.0%}")
        del pat_matches
    else:
        warnings.append("pattern-tags.json not found")
        print("[SKIP] Pattern tags: not found")

    # --- Check 4: Scope detection recall (stream lookup) ---
    scp_path = outdir / "scopes.json"
    if scp_path.exists():
        print("  Scanning scopes.json for golden entries...")
        scp_matches = stream_lookup_json_list(scp_path, "entity_id", covered)
        all_recall = []
        for eid in covered:
            golden_types = set(s["type"] for s in golden[eid].get("golden_scopes", []))
            entry = scp_matches.get(eid)
            if entry:
                pipeline_types = set(
                    s.get("hx/type", "").replace("scope/", "")
                    for s in entry.get("scopes", []))
            else:
                pipeline_types = set()
            _, r = precision_recall(pipeline_types, golden_types)
            all_recall.append(r)

        avg_r = sum(all_recall) / len(all_recall) if all_recall else 0
        s_pass = avg_r >= MIN_SCOPE_RECALL
        checks.append(("scope-recall", s_pass, f"{avg_r:.0%}"))
        print(f"[{'PASS' if s_pass else 'FAIL'}] Scope recall: {avg_r:.0%}")
        del scp_matches
    else:
        warnings.append("scopes.json not found")
        print("[SKIP] Scopes: not found")

    # --- Check 5: Manifest ---
    manifest_path = outdir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        stages = manifest.get("stages_completed", [])
        m_pass = "parse" in stages
        checks.append(("manifest-parse", m_pass, f"stages: {stages}"))
        print(f"[{'PASS' if m_pass else 'FAIL'}] Manifest: {stages}")
    else:
        warnings.append("manifest.json not found")
        print("[SKIP] Manifest: not found")

    # --- Report ---
    print(f"\n{'='*60}")
    passed = sum(1 for _, p, _ in checks if p)
    failed = sum(1 for _, p, _ in checks if not p)
    print(f"Results: {passed} passed, {failed} failed, {len(warnings)} skipped")

    if warnings:
        print(f"\nWarnings:")
        for w in warnings:
            print(f"  - {w}")

    if failed:
        print(f"\nFailed checks:")
        for name, p, detail in checks:
            if not p:
                print(f"  - {name}: {detail}")

    print(f"\nPer-stratum summary:")
    for stratum in ["easy", "medium", "hard"]:
        entries = [g for g in golden.values() if g["stratum"] == stratum]
        avg_ner = sum(g["golden_ner_count"] for g in entries) / len(entries)
        avg_pat = sum(len(g["golden_pattern_names"]) for g in entries) / len(entries)
        avg_scp = sum(g["golden_scope_count"] for g in entries) / len(entries)
        print(f"  {stratum:8s}: {len(entries)} entries, "
              f"avg NER={avg_ner:.0f}, patterns={avg_pat:.1f}, scopes={avg_scp:.1f}")

    return 0 if failed == 0 else 1


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <pipeline-output-dir>")
        print(f"       {sys.argv[0]} --self-check")
        sys.exit(2)

    if sys.argv[1] == "--self-check":
        golden = load_golden()
        print(f"Self-check: {len(golden)} golden entries\n")
        for eid, g in golden.items():
            if g["golden_ner_count"] == 0:
                print(f"  WARNING: {eid} has 0 NER terms")
            if not g["golden_pattern_names"]:
                print(f"  WARNING: {eid} has 0 patterns")
        print("\nSelf-check complete.")
        sys.exit(0)

    outdir = Path(sys.argv[1])
    if not outdir.is_dir():
        print(f"Error: {outdir} is not a directory")
        sys.exit(2)

    sys.exit(validate(outdir))


if __name__ == "__main__":
    main()
