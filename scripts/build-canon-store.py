#!/usr/bin/env python3
r"""Build a canon fingerprint store from one or more gold corpora.

Runs the engine on each entry, writes per-binding CanonFingerprint
records to a per-corpus JSONL, then state-merges them into a single
aggregate JSON the per-binding posterior can consume as a prior.

The store can also accept an existing prior aggregate via
--prior-aggregate — the new corpus state-merges into it.

Usage:
    python scripts/build-canon-store.py \\
        --gold data/grounding-gold-pm.json \\
        --gold data/grounding-gold-proofwiki.json \\
        --ner-kernel /home/joe/code/storage/futon6/data/ner-kernel/terms.tsv \\
        --out-dir data/canon-store-pm-pw/ \\
        --disable-strategy the-Y-X \\
        --disable-strategy section-context \\
        --disable-strategy kernel-ambient
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from futon6 import canon_store as _cs
from futon6 import grounding as _grd


def _load_module(name: str, rel_path: str):
    spec = spec_from_file_location(name, ROOT / rel_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SUPERPOD_JOB = _load_module("superpod_job_buildstore", "scripts/superpod-job.py")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True, action="append",
                        help="Gold JSON path (repeatable)")
    parser.add_argument("--ner-kernel", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Output dir: writes fingerprints/<source>.jsonl "
                             "and aggregate.json")
    parser.add_argument("--prior-aggregate", type=Path, default=None,
                        help="Existing aggregate.json to state-merge into")
    parser.add_argument("--max-entries-per-source", type=int, default=0)
    parser.add_argument("--disable-strategy", action="append", default=[],
                        dest="disable_strategies")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    singles, multi_index, _ = SUPERPOD_JOB.load_ner_kernel(args.ner_kernel)
    disabled = set(args.disable_strategies) if args.disable_strategies else None
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fp_dir = args.out_dir / "fingerprints"
    fp_dir.mkdir(parents=True, exist_ok=True)

    batch_paths = []
    for gold_path in args.gold:
        data = json.loads(gold_path.read_text(encoding="utf-8"))
        src = data.get("source") or gold_path.stem
        entries = data["entries"]
        if args.max_entries_per_source:
            entries = entries[: args.max_entries_per_source]
        out_jsonl = fp_dir / f"{src}.jsonl"
        if out_jsonl.exists():
            out_jsonl.unlink()
        n_fp = 0
        for entry in entries:
            _, env, _ = _grd.detect_grounded_symbols(
                entry["id"], entry["raw_text"], singles, multi_index,
                SUPERPOD_JOB.spot_terms_entity,
                disabled_strategies=disabled,
            )
            fps = [
                _cs.CanonFingerprint(
                    symbol=b.symbol, canon=b.canon, paper_id=entry["id"],
                    strategy=b.strategy, confidence=b.confidence,
                    constructor=getattr(b, "constructor", "single"),
                )
                for b in env.all_bindings
            ]
            n_fp += _cs.write_batch_fingerprints(fps, out_jsonl)
        batch_paths.append(out_jsonl)
        print(f"[build-store] {src}: {len(entries)} entries → "
              f"{n_fp} fingerprints in {out_jsonl}")

    prior = None
    if args.prior_aggregate and args.prior_aggregate.exists():
        prior = _cs.load_aggregate(args.prior_aggregate)
        print(f"[build-store] loaded prior aggregate: {len(prior)} entries")

    aggregate = _cs.aggregate_canon_store(batch_paths, prior_aggregate=prior)
    agg_path = args.out_dir / "aggregate.json"
    _cs.save_aggregate(aggregate, agg_path)
    print(f"[build-store] aggregate: {len(aggregate)} (symbol, canon) entries → {agg_path}")
    return aggregate


if __name__ == "__main__":
    main()
