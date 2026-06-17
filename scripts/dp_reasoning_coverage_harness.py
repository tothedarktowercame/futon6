#!/usr/bin/env python3
"""Held-out reasoning-layer coverage harness for dp_paper_view.

This intentionally runs the detector, not the checker. The checker gate remains
``check_invariants.py``; this harness only measures whether the reasoning layer
anchors proof regions, claims, and illative inferences on a held-out paper set.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import dp_paper_view as dpv  # noqa: E402

DEFAULT_HELDOUT = [
    "0704.0502", "0704.1378", "0704.1624", "0704.2106", "0704.2207",
    "0704.2576", "0704.3976", "0705.0102", "0705.0452", "0705.0462",
    "0705.2537", "0705.3249", "0705.3485", "0705.4334", "0705.4406",
]
DEFAULT_REQUIRED = ["1012.1220"]
TEXT_PROOF_RE = re.compile(
    r"(?<![A-Za-z])(?:\\(?:emph|textit|textbf)\s*\{\s*)?Proof\.(?:\s*\})?",
    re.I,
)


def paper_counts(pid: str) -> dict:
    data = dpv.build(pid, with_binders=False, with_scopes=False, with_xref=False)
    counts = Counter(m.get("kind") for m in data["marks"])
    doc_start = data["text"].find("\\begin{document}")
    body = data["text"][doc_start if doc_start != -1 else 0:]
    return {
        "paper": pid,
        "proof_regions": counts["env/proof"],
        "inferences": counts["inference"],
        "claims": counts["claim"],
        "has_text_proof_marker": bool(TEXT_PROOF_RE.search(body)),
    }


def summarize(rows: list[dict], required_rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "heldout_papers": n,
        "heldout_proof_region_coverage": (
            sum(1 for r in rows if r["proof_regions"] > 0) / n if n else 0
        ),
        "heldout_inference_coverage": (
            sum(1 for r in rows if r["inferences"] > 0) / n if n else 0
        ),
        "heldout_total_proof_regions": sum(r["proof_regions"] for r in rows),
        "heldout_total_inferences": sum(r["inferences"] for r in rows),
        "heldout_total_claims": sum(r["claims"] for r in rows),
        "required": required_rows,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="append", dest="papers")
    ap.add_argument("--required-paper", action="append", dest="required")
    ap.add_argument("--min-papers", type=int, default=15)
    ap.add_argument("--min-inference-coverage", type=float, default=0.80)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)

    papers = args.papers or DEFAULT_HELDOUT
    required = args.required or DEFAULT_REQUIRED
    rows = [paper_counts(pid) for pid in papers]
    required_rows = [paper_counts(pid) for pid in required]
    summary = summarize(rows, required_rows)
    result = {"summary": summary, "per_paper": rows}
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.out:
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    ok = True
    ok &= len(rows) >= args.min_papers
    ok &= summary["heldout_inference_coverage"] >= args.min_inference_coverage
    for row in required_rows:
        if row["has_text_proof_marker"] and row["proof_regions"] == 0:
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
