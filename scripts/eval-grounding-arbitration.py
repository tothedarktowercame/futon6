#!/usr/bin/env python3
r"""Evaluate the Bayesian per-binding canon posterior at inference.

Does combining strategy votes via reliability-weighted Bayes
(`combine_strategy_votes`) actually lift held-out precision over
the best single strategy?

Pipeline:
  1. Train/test split of a gold corpus.
  2. From TRAIN: initialize StrategyReliability posteriors (gold-supervised).
  3. From TEST: for each gold (symbol, canon) pair:
       - run engine, collect each strategy's vote on this symbol
       - combine via `combine_strategy_votes` → CanonPosterior
       - check if top-1 canon matches gold
  4. Compare arbitrated precision vs per-strategy precision.

The arbitrated precision should EXCEED the best single strategy
if the posterior is genuinely combining information across votes.
If it just averages them, it should fall between max-strategy and
average-strategy precision.

Usage:
    python scripts/eval-grounding-arbitration.py \\
        --gold data/grounding-gold-proofwiki.json \\
        --ner-kernel /home/joe/code/storage/futon6/data/ner-kernel/terms.tsv \\
        --train-n 600 --test-n 400 \\
        --match-mode ancestry \\
        --ancestry-index data/canon-ancestry-pm.json \\
        --out data/grounding-arbitration-report.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from futon6 import bayesian_grounding as _bg
from futon6 import grounding as _grd


def _load_module(name: str, rel_path: str):
    spec = spec_from_file_location(name, ROOT / rel_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SUPERPOD_JOB = _load_module("superpod_job_arb", "scripts/superpod-job.py")
EVAL_GOLD = _load_module("eval_grounding_gold_arb", "scripts/eval-grounding-gold.py")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--ner-kernel", type=Path, required=True)
    parser.add_argument("--out", type=Path,
                        default=Path("grounding-arbitration-report.json"))
    parser.add_argument("--train-n", type=int, default=600)
    parser.add_argument("--test-n", type=int, default=400)
    parser.add_argument("--match-mode", choices=["loose", "strict", "ancestry"],
                        default="loose")
    parser.add_argument("--ancestry-index", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--disable-strategy", action="append", default=[],
                        dest="disable_strategies")
    return parser.parse_args(argv)


def run_grounding(entry, singles, multi_index, disabled):
    _, env, _ = _grd.detect_grounded_symbols(
        entry["id"], entry["raw_text"], singles, multi_index,
        SUPERPOD_JOB.spot_terms_entity,
        disabled_strategies=disabled,
    )
    return [(b.strategy, b.symbol, b.canon) for b in env.all_bindings]


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    rng = random.Random(args.seed)
    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    entries = list(gold["entries"])
    rng.shuffle(entries)
    train = entries[: args.train_n]
    test = entries[args.train_n : args.train_n + args.test_n]
    print(f"[arbitration] split: train={len(train)}, test={len(test)}")

    singles, multi_index, _ = SUPERPOD_JOB.load_ner_kernel(args.ner_kernel)
    ancestry: dict[str, set[str]] | None = None
    if args.ancestry_index:
        ai = json.loads(args.ancestry_index.read_text(encoding="utf-8"))
        ancestry = {k: set(v) for k, v in ai.get("by_canon", {}).items()}
    disabled = set(args.disable_strategies) if args.disable_strategies else None

    # --- TRAIN: initialise strategy reliability posteriors from gold ---
    train_tp: dict[str, int] = defaultdict(int)
    train_fp: dict[str, int] = defaultdict(int)
    for entry in train:
        gold_pairs = entry.get("gold") or []
        gold_by_sym: dict[str, list[str]] = defaultdict(list)
        for g in gold_pairs:
            gold_by_sym[g["symbol"]].append(g["canon"])
        bindings = run_grounding(entry, singles, multi_index, disabled)
        on_sym: dict[str, list[tuple[str, str | None]]] = defaultdict(list)
        for strat, sym, canon in bindings:
            on_sym[sym].append((strat, canon))
        for sym, canons in gold_by_sym.items():
            for strat, ec in on_sym.get(sym, []):
                if any(EVAL_GOLD.canon_match(ec, gc, args.match_mode, ancestry)
                       for gc in canons):
                    train_tp[strat] += 1
                else:
                    train_fp[strat] += 1
    reliabilities: dict[str, _bg.StrategyReliability] = {}
    for strat in set(train_tp) | set(train_fp):
        reliabilities[strat] = _bg.StrategyReliability(
            name=strat,
            alpha=1.0 + train_tp[strat],
            beta=1.0 + train_fp[strat],
            n_observations=train_tp[strat] + train_fp[strat],
        )
    print("[arbitration] strategy reliabilities (from TRAIN):")
    for s in sorted(reliabilities):
        r = reliabilities[s]
        print(f"  {s:18s} mean={r.mean*100:5.1f}%  n={r.n_observations}")

    # --- TEST: per-strategy AND arbitrated precision ---
    # Per-strategy: each strategy vote on each gold (symbol, canon)
    # Arbitrated: combine all strategies' votes via combine_strategy_votes,
    # use the top-1 canon as the engine's answer
    per_strat_tp: dict[str, int] = defaultdict(int)
    per_strat_fp: dict[str, int] = defaultdict(int)
    arb_tp = 0
    arb_fp = 0
    arb_no_vote = 0
    for entry in test:
        gold_pairs = entry.get("gold") or []
        gold_by_sym: dict[str, list[str]] = defaultdict(list)
        for g in gold_pairs:
            gold_by_sym[g["symbol"]].append(g["canon"])
        bindings = run_grounding(entry, singles, multi_index, disabled)
        on_sym: dict[str, list[tuple[str, str | None]]] = defaultdict(list)
        for strat, sym, canon in bindings:
            on_sym[sym].append((strat, canon))
        for sym, canons in gold_by_sym.items():
            votes = on_sym.get(sym, [])
            # Per-strategy precision (same as eval-grounding-gold)
            for strat, ec in votes:
                if any(EVAL_GOLD.canon_match(ec, gc, args.match_mode, ancestry)
                       for gc in canons):
                    per_strat_tp[strat] += 1
                else:
                    per_strat_fp[strat] += 1
            # Arbitrated precision: combine votes → top-1 canon
            if not votes:
                arb_no_vote += 1
                continue
            posterior = _bg.combine_strategy_votes(
                sym, votes, reliabilities,
            )
            top_canon, top_prob = posterior.top1()
            if top_canon is None:
                arb_no_vote += 1
                continue
            if any(EVAL_GOLD.canon_match(top_canon, gc, args.match_mode, ancestry)
                   for gc in canons):
                arb_tp += 1
            else:
                arb_fp += 1

    arb_precision = arb_tp / (arb_tp + arb_fp) if (arb_tp + arb_fp) else 0.0
    per_strat_precision = {}
    for strat in set(per_strat_tp) | set(per_strat_fp):
        denom = per_strat_tp[strat] + per_strat_fp[strat]
        per_strat_precision[strat] = per_strat_tp[strat] / denom if denom else 0.0

    out = {
        "gold": str(args.gold),
        "match_mode": args.match_mode,
        "split": {"train": len(train), "test": len(test)},
        "strategy_reliabilities": {
            s: {"mean": r.mean, "alpha": r.alpha, "beta": r.beta, "n": r.n_observations}
            for s, r in reliabilities.items()
        },
        "per_strategy_precision_on_test": per_strat_precision,
        "arbitration": {
            "tp": arb_tp,
            "fp": arb_fp,
            "no_vote": arb_no_vote,
            "precision": arb_precision,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print()
    print("[arbitration] held-out TEST precision comparison")
    print(f"  best single strategy:      "
          f"{max(per_strat_precision.values(), default=0)*100:5.1f}% "
          f"({max(per_strat_precision, key=lambda k: per_strat_precision[k]) if per_strat_precision else 'n/a'})")
    print(f"  weighted avg per strategy: "
          f"{(sum(per_strat_precision[s] * (per_strat_tp[s] + per_strat_fp[s]) for s in per_strat_precision) / max(1, sum(per_strat_tp[s] + per_strat_fp[s] for s in per_strat_precision)))*100:5.1f}%")
    print(f"  ARBITRATED (Bayesian):     {arb_precision*100:5.1f}%")
    print(f"  no_vote (no strategy fired on gold symbol): {arb_no_vote}")
    print(f"[arbitration] wrote {args.out}")
    return out


if __name__ == "__main__":
    main()
