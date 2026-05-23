#!/usr/bin/env python3
r"""Audit the NER kernel for garbage annotations.

Joe's diagnosis: "stable" → "StableMarriageProblem" is probably the
tip of the iceberg. Common English words get mapped to specific PM
article titles because the kernel was built from PM page-title
constituents without filtering for ambiguous single-word terms.

Audit dimensions:
  1. STATIC suspicion shape (no corpus needed):
       - term is a single word (no spaces)
       - canon is much longer than term
       - canon starts/contains the term as a constituent word
     Example: term="stable", canon="StableMarriageProblem" — flagged.
              term="abelian group", canon="AbelianGroup" — clean.
  2. DYNAMIC impact (sample of papers):
       - count how often the entry fires in real arxiv text
       - rank by impact = static-suspicion × hit count

Output: ranked list of suspicious entries with sample contexts +
total observed hits. The operator can then decide whether to
remove from kernel, build a blocklist, or rebuild with a stricter
filter.

Usage:
    python scripts/audit-ner-kernel.py \\
        --ner-kernel /home/joe/code/storage/futon6/data/ner-kernel/terms.tsv \\
        --eprint-dir /home/joe/code/storage/futon6/data/arxiv-math-ct-eprints \\
        --max-papers 50 \\
        --out data/ner-kernel-audit.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from futon6 import structure_seed as _ss


def _load_module(name: str, rel_path: str):
    spec = spec_from_file_location(name, ROOT / rel_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SUPERPOD_JOB = _load_module("superpod_job_audit", "scripts/superpod-job.py")
TERM_EVIDENCE = _load_module(
    "build_arxiv_ct_term_evidence_audit",
    "scripts/build-arxiv-ct-term-evidence.py",
)


def static_suspicion_score(term_lower: str, canon: str) -> int:
    """0 (clean) to 4 (very suspicious).

    Heuristics:
      +1: term is a single word (no spaces)
      +1: term is short (≤ 8 chars)
      +1: canon is much longer than term (≥ 2x chars)
      +1: canon starts with the term as a prefix (the term is a
          modifier; canon names a specific compound concept)
    """
    score = 0
    if " " not in term_lower:
        score += 1
    if len(term_lower) <= 8:
        score += 1
    if canon and len(canon) >= 2 * max(len(term_lower), 1):
        score += 1
    canon_lower = (canon or "").lower()
    if canon_lower.startswith(term_lower) and len(canon_lower) > len(term_lower):
        # canon = term + (more); the extra is the disambiguating
        # specifier ("StableMarriageProblem" beyond "stable")
        score += 1
    return score


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ner-kernel", type=Path, required=True)
    parser.add_argument("--eprint-dir", type=Path, default=None,
                        help="Optional: sample arxiv tarballs for dynamic "
                             "hit-frequency. Skip dynamic pass if omitted.")
    parser.add_argument("--max-papers", type=int, default=50)
    parser.add_argument("--out", type=Path,
                        default=Path("ner-kernel-audit.json"))
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--min-static-score", type=int, default=3,
                        help="Static suspicion threshold for the flagged "
                             "list (1-4; default 3)")
    parser.add_argument("--top-n", type=int, default=50,
                        help="Top N flagged entries to keep in report")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print(f"[ner-audit] loading kernel from {args.ner_kernel}")
    singles, multi_index, _ = SUPERPOD_JOB.load_ner_kernel(args.ner_kernel)

    # Walk every kernel entry, compute static suspicion score.
    # singles is dict[term_lower → (term_orig, canon)]
    # multi_index is dict[first_word → list of (term_lower, term_orig, canon)]
    all_entries = []
    for term_lower, (term_orig, canon) in singles.items():
        all_entries.append((term_lower, term_orig, canon, "single"))
    for first_word, rows in multi_index.items():
        for term_lower, term_orig, canon in rows:
            all_entries.append((term_lower, term_orig, canon, "multi"))
    print(f"[ner-audit] {len(all_entries)} kernel entries")

    static_scores = []
    for term_lower, term_orig, canon, kind in all_entries:
        s = static_suspicion_score(term_lower, canon)
        static_scores.append({
            "term_lower": term_lower,
            "term_orig": term_orig,
            "canon": canon,
            "kind": kind,
            "static_score": s,
        })
    flagged_static = [e for e in static_scores
                      if e["static_score"] >= args.min_static_score]
    print(f"[ner-audit] {len(flagged_static)} entries with "
          f"static score >= {args.min_static_score}")

    # Dynamic: count hits across arxiv sample
    hit_counts: Counter[str] = Counter()  # keyed by term_lower
    sample_contexts: dict[str, list[str]] = defaultdict(list)
    n_papers = 0
    if args.eprint_dir:
        rng = random.Random(args.seed)
        all_tarballs = sorted(args.eprint_dir.glob("*.tar.gz"))
        if all_tarballs:
            sample = rng.sample(all_tarballs, min(args.max_papers, len(all_tarballs)))
            print(f"[ner-audit] scanning {len(sample)} arxiv eprints "
                  f"for hit frequencies")
            for tar_path in sample:
                try:
                    text = TERM_EVIDENCE.LOAD_ARXIV._read_payload(tar_path)
                except Exception:
                    continue
                n_papers += 1
                hits = _ss.find_kernel_term_positions(
                    text, SUPERPOD_JOB.spot_terms_entity,
                    singles, multi_index,
                )
                for start, end, term_lower, canon in hits:
                    hit_counts[term_lower] += 1
                    # Capture a context sample (first 3 contexts per term)
                    if len(sample_contexts[term_lower]) < 3:
                        ctx_start = max(0, start - 40)
                        ctx_end = min(len(text), end + 40)
                        ctx = text[ctx_start:ctx_end].replace("\n", " ")
                        sample_contexts[term_lower].append(ctx)
                if n_papers % 10 == 0:
                    print(f"[ner-audit]   ...{n_papers}/{len(sample)} papers")

    # Compose ranked output: only flagged-suspicious entries, sorted by
    # (hit_count desc, static_score desc).
    flagged = []
    for e in flagged_static:
        hits = hit_counts.get(e["term_lower"], 0)
        flagged.append({
            **e,
            "hits_in_sample": hits,
            "impact": hits * e["static_score"],
            "sample_contexts": sample_contexts.get(e["term_lower"], []),
        })
    flagged.sort(key=lambda r: (-r["impact"], -r["hits_in_sample"], -r["static_score"]))
    top = flagged[: args.top_n]

    # Also output overall hit frequency for non-flagged entries
    # (so we can see what the kernel's most-frequent calls are).
    most_frequent = []
    for term_lower, count in hit_counts.most_common(args.top_n):
        # Find the kernel entry
        entry = next(
            (e for e in all_entries if e[0] == term_lower),
            None,
        )
        if entry:
            most_frequent.append({
                "term_lower": term_lower,
                "term_orig": entry[1],
                "canon": entry[2],
                "static_score": static_suspicion_score(term_lower, entry[2]),
                "hits": count,
                "sample_contexts": sample_contexts.get(term_lower, []),
            })

    out = {
        "ner_kernel_path": str(args.ner_kernel),
        "total_kernel_entries": len(all_entries),
        "entries_with_static_score_geq_threshold": len(flagged_static),
        "threshold": args.min_static_score,
        "papers_scanned": n_papers,
        "top_flagged_by_impact": top,
        "top_most_frequent_kernel_hits": most_frequent,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print()
    print(f"[ner-audit] TOP FLAGGED (high static score + high hit count):")
    for r in top[:25]:
        contexts = r["sample_contexts"]
        ctx_show = (contexts[0][:60] + "...") if contexts else ""
        print(
            f"  s={r['static_score']} hits={r['hits_in_sample']:4d}  "
            f"{r['term_lower']!r:18s} → {r['canon']!r:34s}  ctx: {ctx_show!r}"
        )
    print()
    print(f"[ner-audit] TOP MOST FREQUENT KERNEL HITS (any static score):")
    for r in most_frequent[:25]:
        marker = "⚠" if r["static_score"] >= args.min_static_score else " "
        print(
            f"  {marker} static={r['static_score']} hits={r['hits']:4d}  "
            f"{r['term_lower']!r:18s} → {r['canon']!r:34s}"
        )
    print(f"[ner-audit] wrote {args.out}")
    return out


if __name__ == "__main__":
    main()
