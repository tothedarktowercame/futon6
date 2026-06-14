#!/usr/bin/env python3
"""Append a timestamped loss-trajectory row so Joe can follow the run on return.

Runs the corpus invariant check, then appends ONE row to:
  - holes/loss-ledger.md     (committed — the human-readable trajectory)
  - data/loss/loss-log.jsonl (live, gitignored — the machine record)

    log_loss.py ["note: what landed since last tick"]

claude-1 calls this every loop tick; the committed ledger is the artifact Joe
reads first when he's back (grounding % over time + what changed each tick).
"""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_invariants as ci

ROOT = Path("/home/joe/code/futon6")
LEDGER = ROOT / "holes" / "loss-ledger.md"
LOGJSONL = ROOT / "data" / "loss" / "loss-log.jsonl"

HEADER = """# Loss ledger — DP fleet coverage trajectory

Appended every loop tick by `scripts/log_loss.py` (claude-1). Read top-to-bottom
for the story: grounding % should rise, well-formedness errors should fall to 0,
debt is mostly the dominant ungrounded-symbol count (and irreducible definition
holes). See `holes/dp-fleet-plan.md` for the capability targets.

| time (local) | papers | grounded | wf-errors | debt | note |
|---|---|---|---|---|---|
"""


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    note = argv[0] if argv else ""
    agg = ci.corpus()  # runs the corpus check + writes data/loss/dashboard.json
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    row = {
        "ts": ts,
        "papers": agg["papers"],
        "grounded": agg["corpus_best_guess"],
        "wf_errors": agg["totals"]["errors"],
        "debt": agg["totals"]["debt"],
        "by_invariant": agg["by_invariant"],
        "note": note,
    }
    LOGJSONL.parent.mkdir(parents=True, exist_ok=True)
    with LOGJSONL.open("a") as f:
        f.write(json.dumps(row) + "\n")
    if not LEDGER.exists():
        LEDGER.write_text(HEADER)
    with LEDGER.open("a") as f:
        f.write(f"| {ts} | {agg['papers']} | {agg['corpus_best_guess']:.0%} | "
                f"{agg['totals']['errors']} | {agg['totals']['debt']} | {note} |\n")
    print(f"\nlogged: {ts}  papers={agg['papers']}  "
          f"grounded={agg['corpus_best_guess']:.0%}  "
          f"wf_errors={agg['totals']['errors']}  -> {LEDGER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
