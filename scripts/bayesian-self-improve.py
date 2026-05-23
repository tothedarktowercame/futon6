#!/usr/bin/env python3
r"""Channel-2 self-improvement experiment: can unsupervised
cross-strategy agreement updates the per-strategy reliability
posteriors *without* degrading held-out precision?

Pipeline:
  1. Split a gold corpus (e.g. ProofWiki) into TRAIN / UNSUP / TEST.
  2. Initialize strategy reliability posteriors from TRAIN gold
     (supervised: TP/FP updates per the standard eval).
  3. Process UNSUP entries WITHOUT gold — run grounding, apply
     `update_from_agreement` (channel 2: cross-strategy agreement
     weighted by current reliabilities).
  4. Validate held-out precision on TEST gold BEFORE and AFTER the
     unsupervised phase. Report posterior trajectory.

Interpretation:
  - TEST precision INCREASES or HOLDS → Channel 2 is faithful;
    unsupervised updates encode genuine signal. The system can
    learn from its own runs.
  - TEST precision DEGRADES → bootstrap bias is real. Channel 2
    needs constraints (e.g. minimum-strategy-corroboration
    threshold, or only update when one of the trusted strategies
    votes).

The mission framing (M-bayesian-structure-learning.md §3) calls
this "channel 2" of the Bayesian update mechanism — semi-
supervised, runs on any batch, no gold required.

Usage:
    python scripts/bayesian-self-improve.py \\
        --gold data/grounding-gold-proofwiki.json \\
        --ner-kernel /home/joe/code/storage/futon6/data/ner-kernel/terms.tsv \\
        --train-n 600 --unsup-n 800 --test-n 400 \\
        --out data/bayesian-self-improve-report.json
"""

from __future__ import annotations

import argparse
import copy
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


SUPERPOD_JOB = _load_module("superpod_job_selfimprove", "scripts/superpod-job.py")
EVAL_GOLD = _load_module("eval_grounding_gold", "scripts/eval-grounding-gold.py")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True,
                        help="Gold JSON (one source). The entries are "
                             "split into train/unsup/test slices.")
    parser.add_argument("--ner-kernel", type=Path, required=True)
    parser.add_argument("--out", type=Path,
                        default=Path("bayesian-self-improve-report.json"))
    parser.add_argument("--train-n", type=int, default=600,
                        help="Entries used for supervised TRAIN init")
    parser.add_argument("--unsup-n", type=int, default=800,
                        help="Entries used for unsupervised Channel-2 phase")
    parser.add_argument("--test-n", type=int, default=400,
                        help="Held-out entries for TEST validation")
    parser.add_argument("--match-mode", choices=["loose", "strict", "ancestry"],
                        default="loose")
    parser.add_argument("--ancestry-index", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--disable-strategy", action="append", default=[],
                        dest="disable_strategies")
    parser.add_argument("--snapshot-every", type=int, default=100,
                        help="Record posterior snapshot every N unsupervised "
                             "entries (for trajectory plotting later)")
    return parser.parse_args(argv)


def run_grounding(entry, singles, multi_index, disabled):
    """Run the engine on one entry, return list of (strategy, sym, canon)
    tuples for env.all_bindings (channel-2 input shape)."""
    _, env, _ = _grd.detect_grounded_symbols(
        entry["id"], entry["raw_text"], singles, multi_index,
        SUPERPOD_JOB.spot_terms_entity,
        disabled_strategies=disabled,
    )
    return [(b.strategy, b.symbol, b.canon) for b in env.all_bindings]


def evaluate_on_test(test_entries, reliabilities, singles, multi_index,
                     match_mode, ancestry, disabled):
    """Run grounding on TEST entries, compute precision against gold,
    optionally re-weighted by current strategy reliabilities.

    Returns: (overall_precision, per_strategy_precision_dict).
    The per-strategy precision is on gold-symbol bindings (matches the
    eval-grounding-gold convention).
    """
    tp_by_strategy: dict[str, int] = defaultdict(int)
    fp_by_strategy: dict[str, int] = defaultdict(int)
    for entry in test_entries:
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
                if any(EVAL_GOLD.canon_match(ec, gc, match_mode, ancestry)
                       for gc in canons):
                    tp_by_strategy[strat] += 1
                else:
                    fp_by_strategy[strat] += 1
    tp = sum(tp_by_strategy.values())
    fp = sum(fp_by_strategy.values())
    overall = tp / (tp + fp) if (tp + fp) else 0.0
    per_strat = {
        s: tp_by_strategy[s] / (tp_by_strategy[s] + fp_by_strategy[s])
        for s in set(tp_by_strategy) | set(fp_by_strategy)
        if (tp_by_strategy[s] + fp_by_strategy[s]) > 0
    }
    return overall, per_strat


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    rng = random.Random(args.seed)
    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    entries = list(gold["entries"])
    rng.shuffle(entries)
    train = entries[: args.train_n]
    unsup = entries[args.train_n : args.train_n + args.unsup_n]
    test = entries[args.train_n + args.unsup_n :][: args.test_n]
    print(f"[self-improve] split: train={len(train)}, "
          f"unsup={len(unsup)}, test={len(test)}")

    singles, multi_index, _ = SUPERPOD_JOB.load_ner_kernel(args.ner_kernel)
    ancestry: dict[str, set[str]] | None = None
    if args.ancestry_index:
        ai = json.loads(args.ancestry_index.read_text(encoding="utf-8"))
        ancestry = {k: set(v) for k, v in ai.get("by_canon", {}).items()}
    disabled = set(args.disable_strategies) if args.disable_strategies else None

    # ----- STAGE 1: gold-supervised init from TRAIN -----
    print("[self-improve] STAGE 1: gold-supervised init from TRAIN")
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
    init_state = {
        s: {"alpha": r.alpha, "beta": r.beta, "mean": r.mean,
            "ci": list(r.credible_interval())}
        for s, r in reliabilities.items()
    }

    # ----- STAGE 2: held-out TEST eval BEFORE unsupervised phase -----
    print("[self-improve] STAGE 2: TEST eval before unsupervised phase")
    pre_overall, pre_strat = evaluate_on_test(
        test, reliabilities, singles, multi_index,
        args.match_mode, ancestry, disabled,
    )
    print(f"[self-improve]   pre TEST precision: {pre_overall*100:.1f}%")

    # ----- STAGE 3: UNSUP phase, channel-2 updates only -----
    print(f"[self-improve] STAGE 3: unsupervised phase ({len(unsup)} entries)")
    trajectory = []
    for i, entry in enumerate(unsup, start=1):
        bindings = run_grounding(entry, singles, multi_index, disabled)
        _bg.update_from_agreement(reliabilities, [bindings])
        if i % args.snapshot_every == 0 or i == len(unsup):
            snapshot = {"n_processed": i}
            for s, r in reliabilities.items():
                snapshot[s] = {"mean": r.mean,
                               "ci": list(r.credible_interval())}
            trajectory.append(snapshot)
            print(f"[self-improve]   ...processed {i}/{len(unsup)}; "
                  f"sample reliabilities: " + ", ".join(
                      f"{s}={r.mean*100:.1f}%" for s, r in reliabilities.items()
                  ))

    # ----- STAGE 4: held-out TEST eval AFTER unsupervised phase -----
    print("[self-improve] STAGE 4: TEST eval after unsupervised phase")
    post_overall, post_strat = evaluate_on_test(
        test, reliabilities, singles, multi_index,
        args.match_mode, ancestry, disabled,
    )
    print(f"[self-improve]   post TEST precision: {post_overall*100:.1f}%")
    delta_pp = (post_overall - pre_overall) * 100
    print(f"[self-improve]   delta: {delta_pp:+.2f}pp")

    final_state = {
        s: {"alpha": r.alpha, "beta": r.beta, "mean": r.mean,
            "ci": list(r.credible_interval())}
        for s, r in reliabilities.items()
    }

    out = {
        "gold": str(args.gold),
        "ner_kernel": str(args.ner_kernel),
        "match_mode": args.match_mode,
        "split": {"train": len(train), "unsup": len(unsup), "test": len(test)},
        "disabled_strategies": list(disabled) if disabled else [],
        "test_precision_before": pre_overall,
        "test_precision_after": post_overall,
        "test_precision_delta_pp": delta_pp,
        "test_per_strategy_before": pre_strat,
        "test_per_strategy_after": post_strat,
        "posteriors_initial": init_state,
        "posteriors_final": final_state,
        "trajectory": trajectory,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"[self-improve] wrote {args.out}")
    print()
    print("[self-improve] FINAL POSTERIORS (initial → final):")
    for s in sorted(reliabilities.keys()):
        init = init_state[s]
        final = final_state[s]
        print(
            f"  {s:18s} "
            f"{init['mean']*100:5.1f}% [{init['ci'][0]*100:4.1f}-{init['ci'][1]*100:4.1f}] "
            f"→ {final['mean']*100:5.1f}% [{final['ci'][0]*100:4.1f}-{final['ci'][1]*100:4.1f}]"
        )
    return out


if __name__ == "__main__":
    main()
