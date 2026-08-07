#!/usr/bin/env python3
"""Record that an optional stage step was skipped, and why — as an artifact.

A stage step that prints "skipping X because Y" and moves on is only as reliable
as the operator reading its stdout, which on an unattended twenty-hour run is
not reliable at all. That is the same defect as H15/H16 (findings printed and
never persisted) wearing different clothes: the information exists exactly once,
in a terminal nobody is watching, and disappears at teardown.

So a skip writes a record:

    <run-dir>/skipped/<step>.json
    {"step": …, "skipped": true, "reason": …, "missing": [...], "run_id": …}

which makes three things possible that a printed message does not: the preflight
can tell an operator BEFORE the window which optional steps will not run; the
replay harness can assert that every optional step either produced a product or
recorded a refusal; and the run directory carries its own account of what it did
not do.

  python scripts/stage_skip.py --run-dir data/runs/mark7z --step apm-structure-match \
      --reason "cross-programme scope inputs absent" --missing data/nlab-wiring/eprint-scopes.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def record(run_dir: str, step: str, reason: str, missing: list[str] | None = None,
           run_id: str = "adhoc", corpus_id: str = "adhoc") -> str:
    out = run_dir if os.path.isabs(run_dir) else os.path.join(ROOT, run_dir)
    d = os.path.join(out, "skipped")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{step}.json")
    with open(path, "w") as fh:
        json.dump({"step": step, "skipped": True, "reason": reason,
                   "missing": missing or [], "run_id": run_id,
                   "corpus_id": corpus_id}, fh, indent=1)
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--step", required=True)
    ap.add_argument("--reason", required=True)
    ap.add_argument("--missing", nargs="*", default=[])
    ap.add_argument("--run-id", default=os.environ.get("RUN_ID", "adhoc"))
    ap.add_argument("--corpus-id", default=os.environ.get("CORPUS", "adhoc"))
    a = ap.parse_args()
    p = record(a.run_dir, a.step, a.reason, a.missing, a.run_id, a.corpus_id)
    print(f"  SKIPPED {a.step}: {a.reason} (recorded in {os.path.relpath(p, ROOT)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
