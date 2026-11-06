"""Tests for mark3 coverage-discipline records (M-superpod-mark3 DERIVE).

Each test constructs the exact bad case the coverage record is named for:
the two silent-fail modes found in §MAP (triple-level null analysis vs
hypergraph-level zero claims), the stage-3 parse failure, and slot
collapse. No network, no LLM — the records are pure functions of the
stage outputs.
"""
import os
import sys

_SRC = os.path.join(os.path.dirname(__file__), os.pardir, "src")
if os.path.abspath(_SRC) not in sys.path:
    sys.path.insert(0, os.path.abspath(_SRC))

from futon6.mark3_coverage import (  # noqa: E402
    COVERED,
    FAILED,
    REASON_HYPERGRAPH_ZERO_CLAIMS,
    REASON_SLOT_COLLAPSE,
    REASON_STAGE3_PARSE_ERROR,
    REASON_TRIPLE_NULL_ANALYSIS,
    build_coverage_record,
    summarize_batch,
)

OK_PARSE = {"ok": True, "family": "math-strategy/structure-relation", "leaf": "technique:axis-parallel-box"}
GOOD_TRIPLE = {
    "situation": "count extremal combinatorics question",
    "salience": "axis-parallel box incidence count",
    "constraint": "upper bound on box count must close",
}


def test_covered_paper_gets_explicit_covered_record():
    record = build_coverage_record("2401.00001", OK_PARSE, GOOD_TRIPLE, n_claims=12)
    assert record["status"] == COVERED
    assert record["reason"] is None
    assert record["family"] == OK_PARSE["family"]
    assert record["n_claims"] == 12


def test_stage3_parse_failure_is_named():
    record = build_coverage_record(
        "2401.00002", {"ok": False, "error": "no-json-object"}, GOOD_TRIPLE, n_claims=5
    )
    assert record["status"] == FAILED
    assert record["reason"] == REASON_STAGE3_PARSE_ERROR
    assert record["detail"] == "no-json-object"


def test_null_analysis_is_the_triple_level_silent_fail():
    record = build_coverage_record("2401.00003", OK_PARSE, None, n_claims=7)
    assert record["reason"] == REASON_TRIPLE_NULL_ANALYSIS
    # Even with claims present, a null analysis is a triple-level failure,
    # not zero-claims: the two modes stay distinguishable.
    assert record["reason"] != REASON_HYPERGRAPH_ZERO_CLAIMS


def test_zero_claims_is_the_hypergraph_level_silent_fail():
    record = build_coverage_record("2401.00004", OK_PARSE, GOOD_TRIPLE, n_claims=0)
    assert record["reason"] == REASON_HYPERGRAPH_ZERO_CLAIMS
    # A healthy triple with zero claims is NOT a null-analysis failure.
    assert record["reason"] != REASON_TRIPLE_NULL_ANALYSIS


def test_collapsed_slots_are_rejected():
    collapsed = dict(GOOD_TRIPLE)
    collapsed["salience"] = collapsed["situation"]
    record = build_coverage_record("2401.00005", OK_PARSE, collapsed, n_claims=9)
    assert record["reason"] == REASON_SLOT_COLLAPSE


def test_missing_slot_is_rejected():
    missing = {k: v for k, v in GOOD_TRIPLE.items() if k != "constraint"}
    record = build_coverage_record("2401.00006", OK_PARSE, missing, n_claims=9)
    assert record["reason"] == REASON_SLOT_COLLAPSE


def test_batch_summary_keeps_both_silent_modes_distinct():
    records = [
        build_coverage_record("p1", OK_PARSE, GOOD_TRIPLE, n_claims=3),
        build_coverage_record("p2", OK_PARSE, GOOD_TRIPLE, n_claims=3),
        build_coverage_record("p3", OK_PARSE, None, n_claims=3),
        build_coverage_record("p4", OK_PARSE, GOOD_TRIPLE, n_claims=0),
        build_coverage_record("p5", {"ok": False, "error": "x"}, GOOD_TRIPLE, n_claims=1),
    ]
    summary = summarize_batch(records)
    assert summary["total"] == 5
    assert summary["covered"] == 2
    assert abs(summary["coverage_rate"] - 0.4) < 1e-9
    failures = summary["failure_counts"]
    assert failures[REASON_TRIPLE_NULL_ANALYSIS] == 1
    assert failures[REASON_HYPERGRAPH_ZERO_CLAIMS] == 1
    assert failures[REASON_STAGE3_PARSE_ERROR] == 1
    # No failure is counted under a null or shared reason.
    assert all(isinstance(k, str) and k for k in failures)


def test_unnamed_failure_is_surfaced_not_silently_covered():
    summary = summarize_batch([{"paper_id": "p6", "status": "failed"}])
    assert summary["failure_counts"]["unnamed-failure"] == 1
    assert summary["covered"] == 0
