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


@pytest.mark.parametrize("elapsed", [5.14, 1140.0, 1200.01, 5000.0])
def test_verdict_free_run_is_never_budget_exhausted(elapsed):
    """A solver that printed no `s` line reported NOTHING.

    Elapsed time must not upgrade silence into "we measured this instance and
    it is hard" — that inference is exactly what FM001-n6.kissat.log encoded
    for five months. Genuine exhaustion says `s UNKNOWN`.
    """
    outcome = bs.classify(bs.RC_INDETERMINATE, "c searching\n", elapsed, 1200)
    assert outcome == "interrupted"
    assert outcome in bs.FAILED_RUN_OUTCOMES


def test_budget_exhausted_requires_the_solver_to_say_unknown():
    full = bs.classify(bs.RC_INDETERMINATE, UNKNOWN_OUT, 1200.01, 1200)
    silent = bs.classify(bs.RC_INDETERMINATE, "", 1200.01, 1200)
    assert full == "budget-exhausted"
    assert silent == "interrupted"


@pytest.mark.parametrize(
    "budget,expected_at_least",
    [(1, bs.MIN_GRACE_SECONDS), (1200, bs.MIN_GRACE_SECONDS), (36000, 3600)],
)
def test_grace_is_bounded_and_scales(budget, expected_at_least):
    """The wall clock is enforced here, not delegated to the solver's --time.

    Grace must be generous enough that ordinary shutdown never trips it, and
    proportionate so a long run is not killed on a fixed 30 s margin."""
    grace = bs.grace_seconds(budget)
    assert grace >= expected_at_least
    assert grace >= bs.MIN_GRACE_SECONDS


def test_budget_killed_is_not_a_measurement():
    # Set when WE stop a solver that overran its own --time.
    assert "budget-killed" in bs.FAILED_RUN_OUTCOMES


def test_failed_run_outcomes_are_the_non_measurements():
    assert bs.FAILED_RUN_OUTCOMES == {
        "interrupted", "solver-error", "sat-unverified", "budget-killed"}
    for good in ("sat", "unsat", "unknown", "budget-exhausted"):
        assert good not in bs.FAILED_RUN_OUTCOMES


def test_wall_clock_is_enforced_here_not_delegated(tmp_path):
    """A solver that ignores its own --time must be stopped by US.

    While this enforcement lived inline in main(), deleting the subprocess
    timeout changed no test — the classifier suite cannot see it. This drives
    the real code path with a fake solver that sleeps regardless of --time.
    """
    fake = tmp_path / "ignores-its-time-limit.sh"
    # Bounded sleep: long enough that finishing on its own proves nothing,
    # short enough that if the guard is ever deleted this test FAILS in ~20s
    # instead of hanging the suite.
    fake.write_text("#!/bin/bash\nsleep 20\n")
    fake.chmod(0o755)

    stdout, stderr, returncode, elapsed, killed = bs.run_solver(
        [str(fake), "--time=1", "dummy.cnf"], budget=1, grace=1)

    assert killed is True, "the overrunning solver must be killed"
    assert returncode is None, "a killed run has no interpretable exit code"
    assert elapsed < 10, (
        f"must be stopped near budget+grace (2s), ran {elapsed:.1f}s — "
        "the wall clock is not being enforced")


def test_wall_clock_lets_a_well_behaved_solver_finish(tmp_path):
    fake = tmp_path / "prompt-solver.sh"
    fake.write_text("#!/bin/bash\necho 's UNKNOWN'\nexit 0\n")
    fake.chmod(0o755)

    stdout, _stderr, returncode, _elapsed, killed = bs.run_solver(
        [str(fake)], budget=1, grace=1)

    assert killed is False
    assert returncode == bs.RC_INDETERMINATE
    assert "s UNKNOWN" in stdout
