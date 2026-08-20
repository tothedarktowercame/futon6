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
  solver-error     - the solver exited with a code outside {10, 20, 0}, or its
                     stdout verdict disagrees with its exit code. Nothing was
                     measured, and it must not be mistaken for a hard instance.
  budget-killed    - the solver overran its own --time and WE stopped it at the
                     wall clock. The budget is enforced here, not merely handed
                     to the solver and trusted.

`budget-exhausted` requires the solver to SAY it stopped undecided (`s UNKNOWN`).
A run that printed no verdict at all is `interrupted` no matter how long it ran:
inferring hardness from the clock alone is the error FM001-n6.kissat.log
encoded for five months.

Usage
-----
    .venv/bin/python scripts/fm001/budgeted_solve.py 6 --budget-seconds 1200 \
        --kissat /home/joe/code/kissat/build/kissat --out-dir <harness-dir>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
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


# kissat's documented exit codes. Anything else means the solver itself failed
# (missing/unreadable CNF is 1, signals give 128+n) and its stdout must not be
# read as a measurement.
RC_SAT = 10
RC_UNSAT = 20
RC_INDETERMINATE = 0
# Every branch of classify() checks the exit code against one of the three
# above, so an out-of-range code cannot reach an outcome other than
# "solver-error". An early `not in INTERPRETABLE_RETURNCODES` guard was removed
# once mutation testing showed it unreachable-in-effect: deleting it changed no
# behaviour and no test, which is exactly the kind of branch that rots.
INTERPRETABLE_RETURNCODES = frozenset({RC_SAT, RC_UNSAT, RC_INDETERMINATE})

MIN_BUDGET_SECONDS = 1

# Outcomes that mean "no trustworthy measurement was produced".
FAILED_RUN_OUTCOMES = frozenset(
    {"interrupted", "solver-error", "sat-unverified", "budget-killed"})

MIN_GRACE_SECONDS = 30


def grace_seconds(budget: int) -> int:
    """Slack allowed past the budget before we kill the solver ourselves.

    Generous enough that ordinary shutdown and log flushing never trip it, small
    enough that a solver ignoring its own --time cannot run away."""
    return max(MIN_GRACE_SECONDS, budget // 10)


def validate_budget(budget: int) -> None:
    """A budget must be a positive number of seconds.

    Without this, --budget-seconds 0 (or negative) made the exhaustion test
    `elapsed >= budget * 0.95` vacuously true, so a solver that did nothing at
    all was recorded as a successful `budget-exhausted` MEASUREMENT — the exact
    false-evidence shape this script exists to prevent.
    """
    if budget < MIN_BUDGET_SECONDS:
        raise ValueError(
            f"--budget-seconds must be >= {MIN_BUDGET_SECONDS}, got {budget}: "
            "a non-positive budget cannot produce a measurement")


def classify(returncode: int, stdout: str, elapsed: float, budget: float) -> str:
    """Outcome from (exit code, stdout, timing). The exit code is consulted
    FIRST and can veto the stdout verdict.

    A crashed or failed solver near the end of its budget used to fall through
    to `budget-exhausted`, which reads as "we measured this instance and it is
    hard" when in fact nothing was measured.
    """
    sat_line = bool(re.search(r"^s SATISFIABLE", stdout, re.M))
    unsat_line = bool(re.search(r"^s UNSATISFIABLE", stdout, re.M))
    unknown_line = bool(re.search(r"^s UNKNOWN", stdout, re.M))

    # A verdict that disagrees with the exit code is not trustworthy either way.
    if sat_line and unsat_line:
        return "solver-error"
    if sat_line:
        return "sat" if returncode == RC_SAT else "solver-error"
    if unsat_line:
        return "unsat" if returncode == RC_UNSAT else "solver-error"
    # No verdict: the exit code must be the indeterminate one.
    if returncode != RC_INDETERMINATE:
        return "solver-error"
    if unknown_line:
        # kissat prints UNKNOWN both on its own --time limit and on other
        # non-decisions; the budget comparison is what tells them apart.
        return "budget-exhausted" if elapsed >= budget * 0.95 else "unknown"
    # No verdict line AT ALL. Elapsed time must not upgrade this: a solver that
    # printed no `s` line reported nothing, and inferring "we measured this
    # instance and it is hard" from the clock alone is precisely the mistake
    # FM001-n6.kissat.log encoded for five months. A genuine budget exhaustion
    # says so with `s UNKNOWN`; silence is an interrupted run, however long it
    # ran for.
    return "interrupted"


def run_solver(cmd: list[str], budget: int, grace: int):
    """Run the solver under OUR wall clock.

    `--time=N` is advisory: it is handed to the solver, which is then trusted to
    honour it. Extracted and hard-bounded here so that a solver ignoring its own
    limit cannot run unbounded, and so the enforcement is reachable by a test —
    while this lived inline in main(), deleting the timeout changed no test.

    Returns (stdout, stderr, returncode, elapsed, killed). `returncode` is None
    when we killed it; `killed` is the fact the caller must not paper over.
    """
    started = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=budget + grace)
        return (proc.stdout, proc.stderr, proc.returncode,
                time.monotonic() - started, False)
    except subprocess.TimeoutExpired as exc:
        def _text(stream):
            if stream is None:
                return ""
            return stream.decode() if isinstance(stream, bytes) else stream
        return (_text(exc.stdout), _text(exc.stderr), None,
                time.monotonic() - started, True)


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

    validate_budget(args.budget_seconds)

    solver = shutil.which(args.kissat) or args.kissat
    if not (Path(solver).is_file() and os.access(solver, os.X_OK)):
        raise SystemExit(
            f"solver not executable: {args.kissat!r}. Refusing to run rather "
            "than record an uninterpretable outcome.")

    harness = load_harness()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cnf_path = out_dir / f"FM001-n{args.n}.cnf"
    log_path = out_dir / f"FM001-n{args.n}.budgeted.log"
    result_path = out_dir / f"FM001-n{args.n}.result.json"

    cnf, edges, _pool = harness.build_instance(args.n)
    cnf.to_file(str(cnf_path))
    vertex_count = 4 * args.n - 2

    cmd = [solver, f"--time={args.budget_seconds}", str(cnf_path)]
    grace = grace_seconds(args.budget_seconds)
    stdout, stderr, returncode, elapsed, killed = run_solver(
        cmd, args.budget_seconds, grace)
    log_path.write_text(stdout + stderr)

    if killed:
        outcome = "budget-killed"
    else:
        outcome = classify(returncode, stdout, elapsed, args.budget_seconds)

    witness_path = None
    if outcome == "sat":
        assignment = harness.decode_model(parse_model(stdout), edges)
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
        "returncode": returncode,
        "grace_seconds": grace,
        "outcome": outcome,
        "witness_verified": outcome == "sat",
        "witness": str(witness_path) if witness_path else None,
        "log": str(log_path),
        "cnf": str(cnf_path),
    }
    result_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    # Non-zero whenever the run did NOT produce a trustworthy measurement.
    # A truthful "budget-exhausted" is a successful measurement; a crash, a
    # died-early run, or a SAT verdict whose colouring failed verification are
    # not, and must not exit 0.
    return 0 if outcome not in FAILED_RUN_OUTCOMES else 1


if __name__ == "__main__":
    sys.exit(main())
