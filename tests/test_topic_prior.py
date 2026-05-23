"""Tests for topic_prior: MSCTopicPrior, SECorpusPrior, and the
arxiv→MSC mapping that resolves a paper's categories to prior keys."""

from __future__ import annotations

from pathlib import Path

import pytest

from futon6.topic_prior import (
    ARXIV_TO_MSC_PRIMARY,
    MSCTopicPrior,
    SECorpusPrior,
    arxiv_categories_to_msc,
    canon_to_phrase,
)


def test_canon_to_phrase_basic():
    assert canon_to_phrase("StableMarriageProblem") == "stable marriage problem"
    assert canon_to_phrase("RingedSpace") == "ringed space"
    assert canon_to_phrase("Functor") == "functor"


def test_arxiv_categories_to_msc_known():
    assert arxiv_categories_to_msc(["math.CT"]) == ["18"]
    assert arxiv_categories_to_msc(["math.NT", "math.CT"]) == ["11", "18"]


def test_arxiv_categories_to_msc_dedup():
    # math.QA includes 18; if combined with math.CT, 18 appears once
    result = arxiv_categories_to_msc(["math.CT", "math.QA"])
    assert result.count("18") == 1


def test_arxiv_categories_to_msc_unknown_class():
    assert arxiv_categories_to_msc(["hep-th", "cs.AI"]) == []
    assert arxiv_categories_to_msc([]) == []


def test_msc_prior_basic():
    p = MSCTopicPrior()
    p.add("Functor", "18", 5)
    p.add("StableMarriageProblem", "91", 3)
    # Functor is heavily on MSC 18
    assert p.prior("Functor", ["18"]) > 0.9
    # StableMarriageProblem is on MSC 91, not 18
    assert p.prior("StableMarriageProblem", ["18"]) < 0.2
    # Unseen canon returns neutral 1.0
    assert p.prior("RandomNeverSeen", ["18"]) == 1.0


def test_msc_prior_empty_primaries_returns_neutral():
    p = MSCTopicPrior()
    p.add("Functor", "18", 5)
    # No topic supplied: shouldn't down-weight anything
    assert p.prior("Functor", []) == 1.0


def test_msc_prior_multiple_primaries():
    p = MSCTopicPrior()
    p.add("RingedSpace", "18", 1)
    p.add("RingedSpace", "14", 2)
    # math.QA covers 16/17/18/81 — RingedSpace matches 18
    primaries = arxiv_categories_to_msc(["math.QA"])
    val = p.prior("RingedSpace", primaries)
    assert 0.0 < val <= 1.0


def test_msc_prior_save_load_roundtrip(tmp_path: Path):
    p = MSCTopicPrior()
    p.add("Functor", "18", 7)
    p.add("Limit", "26", 3)
    out = tmp_path / "msc.json"
    p.save(out)
    loaded = MSCTopicPrior.load(out)
    assert loaded.totals == p.totals
    assert loaded.counts == p.counts
    assert loaded.grand_total == p.grand_total


def test_msc_prior_load_missing_returns_empty(tmp_path: Path):
    loaded = MSCTopicPrior.load(tmp_path / "does-not-exist.json")
    assert loaded.grand_total == 0
    assert loaded.counts == {}


def test_se_prior_basic_shape():
    p = SECorpusPrior()
    p.add("Functor", 100)
    p.add("StableMarriageProblem", 1)
    # Log-scaled: heavily-mentioned canon near 1, rare canon non-zero
    # but smaller, unseen canon smallest. Spread is compressed vs
    # linear scaling (intentional — we don't want to wipe out
    # rare-but-legitimate canons).
    f = p.prior("Functor")
    s = p.prior("StableMarriageProblem")
    u = p.prior("NeverHeardOfIt")
    assert f > 0.95
    assert 0 < s < f
    assert 0 < u < s


def test_se_prior_empty_returns_neutral():
    p = SECorpusPrior()
    assert p.prior("AnyCanon") == 1.0


def test_se_prior_save_load_roundtrip(tmp_path: Path):
    p = SECorpusPrior()
    p.add("Functor", 50)
    p.add("RingedSpace", 12)
    p.n_documents = 1000
    out = tmp_path / "se.json"
    p.save(out)
    loaded = SECorpusPrior.load(out)
    assert loaded.counts == p.counts
    assert loaded.grand_total == p.grand_total
    assert loaded.n_documents == p.n_documents


def test_msc_prior_update_from_run_filters_low_confidence():
    p = MSCTopicPrior()
    n = p.update_from_run(
        [("Functor", 0.9), ("UnsureCanon", 0.3), ("AnotherSure", 0.7)],
        msc_primaries=["18"],
        min_confidence=0.5,
    )
    # Only the two high-confidence emissions go in, one MSC each
    assert n == 2
    assert p.counts["Functor"]["18"] == 1
    assert p.counts["AnotherSure"]["18"] == 1
    assert "UnsureCanon" not in p.counts


def test_msc_prior_update_from_run_no_msc_is_noop():
    p = MSCTopicPrior()
    n = p.update_from_run(
        [("Functor", 0.9)], msc_primaries=[],
    )
    assert n == 0
    assert p.grand_total == 0


def test_context_factors_compose_with_combine_votes():
    """End-to-end: context_factors actually shifts probability."""
    from futon6.bayesian_grounding import (
        StrategyReliability,
        combine_strategy_votes,
    )
    rels = {"a": StrategyReliability(name="a", alpha=10, beta=2)}
    # Two strategies each voting once for different canons (so each
    # canon has the same likelihood from the engine).
    votes = [("a", "Bad"), ("a", "Good")]
    p_neutral = combine_strategy_votes("X", votes, rels)
    # Without context, the two have similar mass (each strategy votes
    # once for one canon and against the other).
    assert abs(p_neutral.candidates.get("Bad", 0) - p_neutral.candidates.get("Good", 0)) < 0.1

    # With context that down-weights Bad by 100x, Good should win.
    p_ctx = combine_strategy_votes(
        "X", votes, rels,
        context_factors=[lambda c: 0.01 if c == "Bad" else 1.0],
    )
    assert p_ctx.candidates.get("Good", 0) > p_ctx.candidates.get("Bad", 0) * 5
