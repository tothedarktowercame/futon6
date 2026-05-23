#!/usr/bin/env python3
r"""Run the symbol-grounding engine against PlanetMath gold; report P/R/F1.

Loads a gold JSON produced by `build-grounding-gold.py`, runs the
default strategies on each entry's `raw_text` (markup-stripped PM
body), and compares engine bindings to the gold (symbol, canon) pairs.

Metrics:
  - **precision_on_gold_symbols**: of engine bindings whose `symbol`
    appears in this entry's gold, what fraction have a matching canon?
    Avoids punishing the engine for being more thorough than the
    author's link density.
  - **recall**: of gold (symbol, canon) pairs, what fraction did the
    engine emit a matching binding for?
  - **f1**: harmonic mean of the two.

Match semantics (default `--match-mode loose`):
  - Symbols compared verbatim after stripping whitespace.
  - Canons compared case-insensitively and via substring containment
    so that "Topology" matches "TopologicalGroup" (the engine may have
    a coarser/finer kernel label than the PM link target).

Pass `--match-mode strict` for case-insensitive exact match only —
that's a tighter lower bound.

Usage:
    python scripts/eval-grounding-gold.py \
        --gold data/grounding-gold-pm.json \
        --ner-kernel /home/joe/code/storage/futon6/data/ner-kernel/terms.tsv \
        --out data/grounding-gold-eval.json \
        --max-entries 0    # 0 = all
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from futon6 import grounding as _grd


def _load_module(name: str, rel_path: str):
    spec = spec_from_file_location(name, ROOT / rel_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SUPERPOD_JOB = _load_module("superpod_job_eval_gold", "scripts/superpod-job.py")


def canon_match(
    engine_canon: str | None,
    gold_canon: str,
    mode: str,
    ancestry: dict[str, set[str]] | None = None,
) -> bool:
    if engine_canon is None or gold_canon is None:
        return False
    a = engine_canon.strip().lower()
    b = gold_canon.strip().lower()
    if not a or not b:
        return False
    if mode == "strict":
        return a == b
    if a == b or a in b or b in a:
        return True
    # ancestry mode: check related-canon graph (loose + ancestry, since
    # ancestry alone would miss substring-but-not-related cases like
    # "Group" vs "TopologicalGroup" — and those are usually right).
    if mode == "ancestry" and ancestry:
        # ancestry is keyed by ORIGINAL case (PM uses CamelCase canons).
        # Look up both directions; if either is in the other's related set
        # under any casing, count as match.
        a_orig = engine_canon.strip()
        b_orig = gold_canon.strip()
        if a_orig in ancestry and b_orig in ancestry[a_orig]:
            return True
        if b_orig in ancestry and a_orig in ancestry[b_orig]:
            return True
        # Try case-insensitive variant in case the keys disagree on case
        a_keys = {k for k in ancestry if k.lower() == a}
        b_keys = {k for k in ancestry if k.lower() == b}
        for ak in a_keys:
            related_lower = {x.lower() for x in ancestry[ak]}
            if b in related_lower:
                return True
        for bk in b_keys:
            related_lower = {x.lower() for x in ancestry[bk]}
            if a in related_lower:
                return True
    return False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gold", type=Path, required=True, action="append",
        help="Gold JSON path. Repeatable: --gold pm.json --gold wiki.json. "
             "The eval keeps per-source counters so PM vs Wikipedia "
             "precision delta can be inspected.",
    )
    parser.add_argument("--ner-kernel", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("grounding-gold-eval.json"))
    parser.add_argument(
        "--max-entries-per-source", type=int, default=0,
        help="Cap entries per source (0 = no cap). Useful for keeping "
             "Wikipedia's volume balanced against smaller PM-style sources.",
    )
    parser.add_argument(
        "--match-mode", choices=["loose", "strict", "ancestry"], default="loose",
        help="loose = case-insensitive + substring; strict = exact; "
             "ancestry = loose + related-canon graph (requires "
             "--ancestry-index)",
    )
    parser.add_argument(
        "--ancestry-index", type=Path, default=None,
        help="Path to canon-ancestry JSON (build-canon-ancestry-pm.py). "
             "Only used when --match-mode=ancestry.",
    )
    parser.add_argument(
        "--disable-strategy", action="append", default=[], dest="disable_strategies",
        help="Strategy name to omit from the run (repeatable). "
             "Used for Gate P3 to validate that gating noisy strategies "
             "lifts precision. Common choices from P2 baselines: "
             "the-Y-X (3.9%%), section-context (2.8%%), "
             "kernel-ambient (9.7%%).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    # Load every supplied gold file; tag entries with their source so the
    # per-source breakdown can detect corpus-overfitting (gate P2).
    sources: list[dict] = []  # list of {source, entries}
    flat_entries: list[tuple[str, dict]] = []  # (source, entry)
    for gold_path in args.gold:
        data = json.loads(gold_path.read_text(encoding="utf-8"))
        src = data.get("source") or gold_path.stem
        es = data["entries"]
        if args.max_entries_per_source:
            es = es[: args.max_entries_per_source]
        sources.append({
            "source": src,
            "path": str(gold_path),
            "entries": len(es),
            "pairs": sum(len(e["gold"]) for e in es),
        })
        for e in es:
            flat_entries.append((src, e))
        print(f"[gold-eval] {len(es)} entries / "
              f"{sum(len(e['gold']) for e in es)} pairs from {gold_path}")
    print(f"[gold-eval] Combined: {len(flat_entries)} entries across "
          f"{len(sources)} source(s)")
    singles, multi_index, _ = SUPERPOD_JOB.load_ner_kernel(args.ner_kernel)

    # Optional canon-ancestry index — wired into canon_match below.
    ancestry: dict[str, set[str]] | None = None
    if args.ancestry_index:
        ai = json.loads(args.ancestry_index.read_text(encoding="utf-8"))
        ancestry = {k: set(v) for k, v in ai.get("by_canon", {}).items()}
        print(f"[gold-eval] Ancestry index: {len(ancestry)} canons, "
              f"{sum(len(v) for v in ancestry.values())} edges")

    # Per-strategy counters across the corpus.
    tp_by_strategy: dict[str, int] = defaultdict(int)
    fp_by_strategy: dict[str, int] = defaultdict(int)
    # Symbol-level gold recall (per entry): which gold symbols did ANY strategy hit?
    total_gold = 0
    total_gold_hit = 0

    # Per-source counters for Gate P2's per-source-delta check.
    per_source: dict[str, dict] = defaultdict(lambda: {
        "tp": 0, "fp": 0, "gold_total": 0, "gold_hit": 0, "entries": 0,
    })

    # Per-entry view for debugging
    per_entry = []

    # Per-strategy total volume (denominator for "engine bindings on
    # gold symbols")
    engine_on_gold = defaultdict(int)

    miss_samples = []
    hit_samples = []

    for source, entry in flat_entries:
        raw = entry["raw_text"]
        gold_pairs = entry["gold"]
        gold_by_sym: dict[str, list[str]] = defaultdict(list)
        for g in gold_pairs:
            gold_by_sym[g["symbol"]].append(g["canon"])

        records, env, _ = _grd.detect_grounded_symbols(
            entry["id"], raw, singles, multi_index,
            SUPERPOD_JOB.spot_terms_entity,
            disabled_strategies=(set(args.disable_strategies) if args.disable_strategies else None),
        )
        # Engine bindings on each gold symbol
        engine_on_sym: dict[str, list[tuple[str, str | None]]] = defaultdict(list)
        for b in env.all_bindings:
            engine_on_sym[b.symbol].append((b.strategy, b.canon))

        per_source[source]["entries"] += 1
        entry_hit = 0
        for sym, canons in gold_by_sym.items():
            total_gold += 1
            per_source[source]["gold_total"] += 1
            engine_calls = engine_on_sym.get(sym, [])
            # Count engine bindings on this gold symbol per strategy
            for strat, _c in engine_calls:
                engine_on_gold[strat] += 1
            # Look for a matching canon
            matched = False
            for strat, ec in engine_calls:
                if any(canon_match(ec, gc, args.match_mode, ancestry) for gc in canons):
                    tp_by_strategy[strat] += 1
                    per_source[source]["tp"] += 1
                    matched = True
                else:
                    fp_by_strategy[strat] += 1
                    per_source[source]["fp"] += 1
            if matched:
                total_gold_hit += 1
                per_source[source]["gold_hit"] += 1
                entry_hit += 1
                if len(hit_samples) < 6:
                    hit_samples.append({
                        "source": source,
                        "entry": entry["id"],
                        "symbol": sym,
                        "gold_canons": canons,
                        "engine_calls": [(s, c) for s, c in engine_calls],
                    })
            else:
                if len(miss_samples) < 6:
                    miss_samples.append({
                        "source": source,
                        "entry": entry["id"],
                        "symbol": sym,
                        "gold_canons": canons,
                        "engine_calls": [(s, c) for s, c in engine_calls],
                    })

        per_entry.append({
            "source": source,
            "entry_id": entry["id"],
            "gold_count": len(gold_pairs),
            "gold_hit": entry_hit,
        })

    overall_recall = total_gold_hit / total_gold if total_gold else 0.0

    # Per-strategy precision on gold symbols
    strategy_table = {}
    for strat in set(list(tp_by_strategy) + list(fp_by_strategy)):
        tp = tp_by_strategy[strat]
        fp = fp_by_strategy[strat]
        denom = tp + fp
        precision = tp / denom if denom else 0.0
        strategy_table[strat] = {
            "tp": tp,
            "fp_on_gold_symbols": fp,
            "engine_bindings_on_gold_symbols": denom,
            "precision_on_gold_symbols": precision,
        }

    # Overall aggregate
    overall_tp = sum(tp_by_strategy.values())
    overall_fp = sum(fp_by_strategy.values())
    overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) else 0.0
    overall_f1 = (
        2 * overall_precision * overall_recall / (overall_precision + overall_recall)
        if (overall_precision + overall_recall) else 0.0
    )

    # Per-source aggregates for Gate P2's per-source-delta check.
    per_source_summary = {}
    for src, c in per_source.items():
        emit = c["tp"] + c["fp"]
        precision = c["tp"] / emit if emit else 0.0
        recall = c["gold_hit"] / c["gold_total"] if c["gold_total"] else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) else 0.0
        )
        per_source_summary[src] = {
            "entries": c["entries"],
            "gold_total": c["gold_total"],
            "gold_hit": c["gold_hit"],
            "tp": c["tp"],
            "fp": c["fp"],
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    # Max delta across all source pairs (informative for the P2 gate)
    max_precision_delta = 0.0
    src_names = list(per_source_summary.keys())
    for i, a in enumerate(src_names):
        for b in src_names[i + 1:]:
            delta = abs(per_source_summary[a]["precision"] - per_source_summary[b]["precision"])
            max_precision_delta = max(max_precision_delta, delta)

    report = {
        "gold_paths": [str(p) for p in args.gold],
        "sources": sources,
        "ner_kernel": str(args.ner_kernel),
        "match_mode": args.match_mode,
        "entries_evaluated": len(flat_entries),
        "total_gold_pairs": total_gold,
        "total_gold_hit": total_gold_hit,
        "overall_recall": overall_recall,
        "overall_precision": overall_precision,
        "overall_f1": overall_f1,
        "per_source": per_source_summary,
        "max_precision_delta": max_precision_delta,
        "strategy_table": strategy_table,
        "hit_samples": hit_samples,
        "miss_samples": miss_samples,
        "per_entry": per_entry,
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[gold-eval] wrote {args.out}")
    print()
    print(f"[gold-eval] OVERALL: gold={total_gold}, "
          f"hit={total_gold_hit}, recall={overall_recall*100:.1f}%, "
          f"precision={overall_precision*100:.1f}%, "
          f"F1={overall_f1*100:.1f}%")
    print()
    print("[gold-eval] Per-source breakdown:")
    for src, v in per_source_summary.items():
        print(
            f"  {src:20s} entries={v['entries']:5d}  gold={v['gold_total']:5d}  "
            f"P={v['precision']*100:5.1f}%  R={v['recall']*100:5.1f}%  "
            f"F1={v['f1']*100:5.1f}%"
        )
    print(f"  Max per-source precision delta: {max_precision_delta*100:.1f}pp")
    print()
    print("[gold-eval] Per-strategy precision (TP / engine bindings on gold symbols):")
    rows = sorted(
        strategy_table.items(),
        key=lambda kv: -kv[1]["engine_bindings_on_gold_symbols"],
    )
    for strat, v in rows:
        if v["engine_bindings_on_gold_symbols"] == 0:
            continue
        print(
            f"  {strat:18s} {v['tp']:4d}/{v['engine_bindings_on_gold_symbols']:5d} = "
            f"{v['precision_on_gold_symbols']*100:5.1f}%"
        )
    return report


if __name__ == "__main__":
    main()
