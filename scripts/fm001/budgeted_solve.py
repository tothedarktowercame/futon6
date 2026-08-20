#!/usr/bin/env python3
"""
FM-001: run one book-Ramsey instance under an EXPLICIT solver budget, and
record an outcome that distinguishes the three things that can happen.

Why this exists
---------------
Every FM-001 solve so far was launched by hand, and the harness directory shows
the cost: `FM001-n5.kissat.log` ends with `s SATISFIABLE`, but
`FM001-n6.kissat.log` and `FM001b-n8.kissat.log` just stop mid-search with no
`s` line and no summary. Read later, an interrupted run is indistinguishable
from a hard instance — n=6 sat for five months looking "unresolved" when its
only attempt had been killed after 5.14 seconds.

So this script always records the budget it gave, and classifies:

  sat      - `s SATISFIABLE`; the witness is decoded AND re-verified with the
             harness's own verify_assignment before the result is written.
             An unverified witness is reported as `sat-unverified`, never as sat.
  unsat    - `s UNSATISFIABLE`; supports R(B_{n-1}, B_n) < 4n-1 for that n.
  unknown  - solver returned without deciding, INSIDE its budget.
  budget-exhausted - solver hit the wall clock we set. Honest "no answer yet".
  interrupted      - no verdict and the budget was NOT reached: the run died.
                     This is the case the old logs could not express.

Usage
-----
    .venv/bin/python scripts/fm001/budgeted_solve.py 6 --budget-seconds 1200 \
        --kissat /home/joe/code/kissat/build/kissat --out-dir <harness-dir>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_harness():
    """Load ramsey_book_sat.py as a module so the encoder, decoder and verifier
    are the SAME code that produced the existing artifacts."""
    spec = importlib.util.spec_from_file_location(
        "ramsey_book_sat", HERE / "ramsey_book_sat.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_model(stdout: str) -> list[int]:
    """DIMACS `v` lines -> literal list."""
    lits: list[int] = []
    for line in stdout.splitlines():
        if line.startswith("v "):
            for tok in line[2:].split():
                val = int(tok)
                if val != 0:
                    lits.append(val)
    return lits


def classify(returncode: int, stdout: str, elapsed: float, budget: float) -> str:
    if re.search(r"^s SATISFIABLE", stdout, re.M):
        return "sat"
    if re.search(r"^s UNSATISFIABLE", stdout, re.M):
        return "unsat"
    if re.search(r"^s UNKNOWN", stdout, re.M):
        # kissat prints UNKNOWN both on its own --time limit and on other
        # non-decisions; the budget comparison is what tells them apart.
        return "budget-exhausted" if elapsed >= budget * 0.95 else "unknown"
    # No verdict line at all.
    return "budget-exhausted" if elapsed >= budget * 0.95 else "interrupted"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("n", type=int, help="book parameter n (builds K_{4n-2})")
    ap.add_argument("--budget-seconds", type=int, required=True,
                    help="wall-clock budget handed to the solver AND recorded")
    ap.add_argument("--kissat", default="kissat", help="path to the kissat binary")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="directory for the cnf, log, witness and result record")
    args = ap.parse_args()

    harness = load_harness()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cnf_path = out_dir / f"FM001-n{args.n}.cnf"
    log_path = out_dir / f"FM001-n{args.n}.budgeted.log"
    result_path = out_dir / f"FM001-n{args.n}.result.json"

    cnf, edges, _pool = harness.build_instance(args.n)
    cnf.to_file(str(cnf_path))
    vertex_count = 4 * args.n - 2

    cmd = [args.kissat, f"--time={args.budget_seconds}", str(cnf_path)]
    started = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.monotonic() - started
    log_path.write_text(proc.stdout + proc.stderr)

    outcome = classify(proc.returncode, proc.stdout, elapsed, args.budget_seconds)

    witness_path = None
    if outcome == "sat":
        assignment = harness.decode_model(parse_model(proc.stdout), edges)
        if harness.verify_assignment(args.n, vertex_count, assignment):
            witness_path = out_dir / f"n{args.n}-witness.json"
            harness.write_witness(witness_path, assignment, args.n)
        else:
            # A solver saying SAT is not a refutation until the colouring is
            # independently checked. Never upgrade this to "sat".
            outcome = "sat-unverified"

    record = {
        "n": args.n,
        "vertex_count": vertex_count,
        "vars": cnf.nv,
        "clauses": len(cnf.clauses),
        "solver": "kissat",
        "budget_seconds": args.budget_seconds,
        "elapsed_seconds": round(elapsed, 2),
        "returncode": proc.returncode,
        "outcome": outcome,
        "witness_verified": outcome == "sat",
        "witness": str(witness_path) if witness_path else None,
        "log": str(log_path),
        "cnf": str(cnf_path),
    }
    result_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    # Non-zero only when the run itself failed to produce an interpretable
    # outcome; a truthful "budget-exhausted" is a successful measurement.
    return 0 if outcome != "interrupted" else 1


if __name__ == "__main__":
    sys.exit(main())
