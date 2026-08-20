"""Regression tests for scripts/fm001/budgeted_solve.py outcome classification.

The point of that script is that a run which measured nothing must never be
recorded as a measurement. Two ways it previously could be:

  * a non-positive --budget-seconds made the exhaustion test
    `elapsed >= budget * 0.95` vacuously true, so a solver that did nothing
    became a "successful" budget-exhausted result;
  * a crashed solver near the end of its budget fell through to
    budget-exhausted, which reads as "we measured this instance and it is
    hard" when in fact nothing was measured.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "fm001" / "budgeted_solve.py"


def _load():
    spec = importlib.util.spec_from_file_location("budgeted_solve", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bs = _load()

SAT_OUT = "s SATISFIABLE\nv 1 -2 0\n"
UNSAT_OUT = "s UNSATISFIABLE\n"
UNKNOWN_OUT = "s UNKNOWN\n"


@pytest.mark.parametrize("budget", [0, -1, -1200])
def test_non_positive_budget_is_refused(budget):
    # Without this, any run at all satisfies `elapsed >= budget * 0.95`.
    with pytest.raises(ValueError):
        bs.validate_budget(budget)


def test_minimum_budget_is_accepted():
    bs.validate_budget(bs.MIN_BUDGET_SECONDS)


@pytest.mark.parametrize("returncode", [1, 2, 127, 134, 139])
def test_solver_error_beats_timing(returncode):
    """A crash at the end of the budget must NOT look like a hard instance."""
    assert bs.classify(returncode, "", 1200.0, 1200) == "solver-error"


@pytest.mark.parametrize(
    "returncode,stdout",
    [(bs.RC_INDETERMINATE, SAT_OUT),   # verdict says SAT, exit code does not
     (bs.RC_SAT, UNSAT_OUT),           # and the reverse
     (bs.RC_UNSAT, SAT_OUT)],
)
def test_verdict_disagreeing_with_exit_code_is_an_error(returncode, stdout):
    assert bs.classify(returncode, stdout, 1.0, 1200) == "solver-error"


def test_contradictory_verdict_lines_are_an_error():
    assert bs.classify(bs.RC_SAT, SAT_OUT + UNSAT_OUT, 1.0, 1200) == "solver-error"


def test_genuine_outcomes_still_classify():
    assert bs.classify(bs.RC_SAT, SAT_OUT, 0.04, 60) == "sat"
    assert bs.classify(bs.RC_UNSAT, UNSAT_OUT, 5.0, 60) == "unsat"


def test_budget_exhausted_requires_reaching_the_budget():
    # This is the n=6 measurement's shape: indeterminate exit, UNKNOWN, full budget.
    assert bs.classify(bs.RC_INDETERMINATE, UNKNOWN_OUT, 1200.01, 1200) == "budget-exhausted"
    # Same verdict, nowhere near the budget: not a measurement of hardness.
    assert bs.classify(bs.RC_INDETERMINATE, UNKNOWN_OUT, 5.0, 1200) == "unknown"


def test_no_verdict_short_of_budget_is_interrupted():
    # The FM001-n6.kissat.log / FM001b-n8.kissat.log shape: stops mid-search.
    assert bs.classify(bs.RC_INDETERMINATE, "c searching\n", 5.14, 1200) == "interrupted"


def test_failed_run_outcomes_are_the_non_measurements():
    assert bs.FAILED_RUN_OUTCOMES == {"interrupted", "solver-error", "sat-unverified"}
    for good in ("sat", "unsat", "unknown", "budget-exhausted"):
        assert good not in bs.FAILED_RUN_OUTCOMES
