#!/usr/bin/env python3
r"""Propose strategy merges using the canon-store fingerprints and
the literature graph.

Loads per-batch fingerprints from a canon-store directory, walks
them by paper to find (strategy, symbol, canon) co-firings, then
asks compute_concordance() how often pairs of strategies agree
either exactly or via graph-adjacent canons. High-concordance
pairs with enough evidence become merge proposals.

Usage:
    python scripts/propose-strategy-merges.py \\
        --store-dir data/canon-store-pm-pw \\
        --ancestry-index data/canon-ancestry-pm.json \\
        --out data/strategy-merge-proposals.json \\
        --min-concordance 0.7 --min-co-firings 30
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from futon6 import canon_store as _cs
from futon6 import strategy_reduction as _sr


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-dir", type=Path, required=True,
                        help="Canon-store dir from build-canon-store.py "
                             "(must contain fingerprints/*.jsonl)")
    parser.add_argument("--ancestry-index", type=Path, required=True,
                        help="Literature graph from build-canon-ancestry-pm.py")
    parser.add_argument("--out", type=Path,
                        default=Path("strategy-merge-proposals.json"))
    parser.add_argument("--min-concordance", type=float, default=0.7)
    parser.add_argument("--min-co-firings", type=int, default=30)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    fp_dir = args.store_dir / "fingerprints"
    batch_paths = sorted(fp_dir.glob("*.jsonl"))
    if not batch_paths:
        raise SystemExit(f"no fingerprint JSONL files in {fp_dir}")
    print(f"[merges] reading {len(batch_paths)} batch files from {fp_dir}")

    ai = json.loads(args.ancestry_index.read_text(encoding="utf-8"))
    graph = {k: set(v) for k, v in ai.get("by_canon", {}).items()}
    print(f"[merges] literature graph: {len(graph)} canons, "
          f"{sum(len(v) for v in graph.values())} edges")

    # Group fingerprints by paper_id so co-firings are per-paper
    per_paper: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    n_fp = 0
    for path in batch_paths:
        for fp in _cs.iter_batch_fingerprints(path):
            if fp.canon is None:
                continue
            per_paper[fp.paper_id].append((fp.strategy, fp.symbol, fp.canon))
            n_fp += 1
    print(f"[merges] {n_fp} fingerprints across {len(per_paper)} papers")

    concordance = _sr.compute_concordance(
        per_paper.values(), graph, max_examples=5,
    )
    proposals = _sr.propose_merges(
        concordance,
        min_concordance=args.min_concordance,
        min_co_firings=args.min_co_firings,
    )

    out = {
        "store_dir": str(args.store_dir),
        "ancestry_index": str(args.ancestry_index),
        "min_concordance": args.min_concordance,
        "min_co_firings": args.min_co_firings,
        "n_fingerprints": n_fp,
        "n_papers": len(per_paper),
        "all_concordance_pairs": [
            {
                "strategy_a": p.strategy_a,
                "strategy_b": p.strategy_b,
                "co_firings": p.n_co_firings,
                "agree_exact": p.n_agree_exact,
                "agree_graph": p.n_agree_graph,
                "disagree": p.n_disagree,
                "concordance": p.concordance,
            }
            for p in sorted(concordance.values(), key=lambda p: -p.n_co_firings)
        ],
        "proposals": [
            {
                "strategy_a": p.strategy_a,
                "strategy_b": p.strategy_b,
                "co_firings": p.n_co_firings,
                "agree_exact": p.n_agree_exact,
                "agree_graph": p.n_agree_graph,
                "concordance": p.concordance,
                "examples_agree": [list(ex) for ex in p.examples_agree],
                "examples_disagree": [list(ex) for ex in p.examples_disagree],
            }
            for p in proposals
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print()
    print(f"[merges] all strategy-pair concordances "
          f"(sorted by co-firings desc):")
    for p in sorted(concordance.values(), key=lambda x: -x.n_co_firings):
        flag = "MERGE" if p in proposals else ""
        print(
            f"  {p.strategy_a:14s} ↔ {p.strategy_b:14s} "
            f"co-firings={p.n_co_firings:5d}  "
            f"agree-exact={p.n_agree_exact:5d}  "
            f"agree-graph={p.n_agree_graph:4d}  "
            f"concordance={p.concordance*100:5.1f}%  {flag}"
        )
    print()
    print(f"[merges] {len(proposals)} merge proposals "
          f"(≥{int(args.min_concordance*100)}% concordance, "
          f"≥{args.min_co_firings} co-firings)")
    for p in proposals:
        print(f"  {p.strategy_a} ↔ {p.strategy_b}")
        if p.examples_agree:
            ex_sym, ex_a, ex_b = p.examples_agree[0]
            print(f"    example: ({ex_sym!r}, {ex_a!r}, {ex_b!r})")
    print(f"[merges] wrote {args.out}")
    return out


if __name__ == "__main__":
    main()
