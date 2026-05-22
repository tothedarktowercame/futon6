#!/usr/bin/env python3
"""Structured eval of the symbol-grounding strategies on a paper set.

Runs `futon6.grounding.detect_grounded_symbols` over every `.tex` file in
an input directory, aggregates the cross-paper strategy meta-learning
table, and emits a sample of N bindings per strategy for manual
spot-check.

Usage:
    python scripts/eval-grounding.py \
        --input-dir /home/joe/code/storage/futon6/data/first-proof/latex \
        --ner-kernel /home/joe/code/storage/futon6/data/ner-kernel/terms.tsv \
        --out report.json \
        --sample-per-strategy 8

The mission success criterion is symbol-grounding precision ≥ 50% on a
30-paper sample. Without a labeled gold set, we use cross-strategy
corroboration_rate as the precision proxy (a binding with the same
canon from two independent strategies is presumed correct). The
manual-spot-check sample lets the operator put eyes on representative
bindings per strategy and override the proxy if needed.
"""

from __future__ import annotations

import argparse
import json
import sys
import random
from collections import defaultdict
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from futon6 import grounding as _grd
from futon6 import symbol_grounding as _sg


def _load_module(name: str, rel_path: str):
    spec = spec_from_file_location(name, ROOT / rel_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SUPERPOD_JOB = _load_module("superpod_job_eval_grounding", "scripts/superpod-job.py")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory containing .tex files to evaluate")
    parser.add_argument("--ner-kernel", type=Path, required=True,
                        help="Path to NER kernel TSV")
    parser.add_argument("--out", type=Path, default=Path("grounding-eval.json"),
                        help="Where to write the JSON report")
    parser.add_argument("--sample-per-strategy", type=int, default=8,
                        help="Bindings to sample per strategy for spot-check")
    parser.add_argument("--learned-vocab", type=Path, default=None,
                        help="Optional learned-newcommand-vocab.json to load")
    parser.add_argument("--max-papers", type=int, default=0,
                        help="Cap paper count (0 = no cap). Mission target: 30.")
    parser.add_argument("--context-chars", type=int, default=120,
                        help="Chars of context around each spot-check sample")
    parser.add_argument("--seed", type=int, default=20260522,
                        help="RNG seed for sampling")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    singles, multi_index, _ = SUPERPOD_JOB.load_ner_kernel(args.ner_kernel)
    learned_vocab = (
        _grd.load_learned_vocab(args.learned_vocab) if args.learned_vocab else []
    )

    paper_paths = sorted(args.input_dir.glob("**/*.tex"))
    if args.max_papers > 0:
        paper_paths = paper_paths[: args.max_papers]

    print(f"[eval-grounding] {len(paper_paths)} papers from {args.input_dir}")
    if learned_vocab:
        print(f"[eval-grounding] learned vocab: {len(learned_vocab)} entries")

    metrics_by_paper: dict[str, dict] = {}
    samples_by_strategy: dict[str, list[dict]] = defaultdict(list)
    per_paper_summary = []
    total_grounded_marks = 0

    for path in paper_paths:
        paper_id = path.stem
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            print(f"[eval-grounding] skipping {path}: {exc}")
            continue
        records, env, summary = _grd.detect_grounded_symbols(
            paper_id, text, singles, multi_index, SUPERPOD_JOB.spot_terms_entity,
            learned_vocab=learned_vocab,
        )
        metrics_by_paper[paper_id] = summary.get("strategy_metrics") or {}
        per_paper_summary.append({
            "paper_id": paper_id,
            "path": str(path),
            "total_bindings_emitted": summary.get("total_bindings_emitted", 0),
            "active_bindings": summary.get("active_bindings", 0),
            "grounded_atom_count": summary.get("grounded_atom_count", 0),
            "strategy_emit_counts": summary.get("strategy_emit_counts", {}),
        })
        total_grounded_marks += len(records)
        # Collect spot-check samples — every binding is a candidate; we
        # reservoir later.
        for b in env.all_bindings:
            ctx_start = max(0, b.evidence_span[0] - args.context_chars)
            ctx_end = min(len(text), b.evidence_span[1] + args.context_chars)
            samples_by_strategy[b.strategy].append({
                "paper_id": paper_id,
                "symbol": b.symbol,
                "canon": b.canon,
                "type_phrase": b.type_phrase,
                "confidence": b.confidence,
                "defeated": b.defeated_by is not None,
                "evidence_span": list(b.evidence_span),
                "context": text[ctx_start:ctx_end].replace("\n", " "),
            })

    aggregate = _sg.aggregate_strategy_metrics(metrics_by_paper)

    # Reservoir-sample per strategy. Bias toward UN-defeated bindings
    # (more interesting for spot-check; defeated ones got overridden by
    # the engine already).
    rng = random.Random(args.seed)
    sampled: dict[str, list[dict]] = {}
    for strat, candidates in samples_by_strategy.items():
        undefeated = [c for c in candidates if not c["defeated"]]
        pool = undefeated if len(undefeated) >= args.sample_per_strategy else candidates
        if not pool:
            sampled[strat] = []
            continue
        if len(pool) <= args.sample_per_strategy:
            sampled[strat] = pool
        else:
            sampled[strat] = rng.sample(pool, args.sample_per_strategy)

    report = {
        "input_dir": str(args.input_dir),
        "ner_kernel": str(args.ner_kernel),
        "learned_vocab_size": len(learned_vocab),
        "paper_count": len(paper_paths),
        "total_grounded_marks": total_grounded_marks,
        "strategy_meta_learning": aggregate,
        "spot_check_samples": sampled,
        "per_paper_summary": per_paper_summary,
    }

    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[eval-grounding] wrote {args.out}")

    # Print headline
    print()
    print("[eval-grounding] Strategy meta-learning across "
          f"{len(metrics_by_paper)} papers ({total_grounded_marks} grounded marks):")
    rows = sorted(aggregate.items(), key=lambda kv: -kv[1].get("emitted", 0))
    for strat, agg in rows:
        emit = agg.get("emitted", 0)
        defeat_pct = agg.get("defeat_rate", 0) * 100
        corr_pct = agg.get("corroboration_rate", 0) * 100
        papers = agg.get("papers_active", 0)
        print(
            f"  {strat:14s} emitted={emit:5d}  papers={papers:3d}  "
            f"defeated={defeat_pct:5.1f}%  corroborated={corr_pct:5.1f}%"
        )
    return report


if __name__ == "__main__":
    main()
