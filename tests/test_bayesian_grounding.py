"""Tests for futon6.bayesian_grounding."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from futon6.bayesian_grounding import (
    CanonPosterior,
    StrategyReliability,
    combine_strategy_votes,
    expected_batch_info_gain,
    fit_reliabilities_from_eval_report,
    update_from_agreement,
)


def test_default_prior_is_uniform():
    r = StrategyReliability(name="x")
    assert r.alpha == 1.0 and r.beta == 1.0
    assert r.mean == 0.5


def test_update_increments_alpha_on_correct():
    r = StrategyReliability(name="x")
    r.update(True)
    assert r.alpha == 2.0
    assert r.beta == 1.0


def test_update_increments_beta_on_wrong():
    r = StrategyReliability(name="x")
    r.update(False)
    assert r.alpha == 1.0
    assert r.beta == 2.0


def test_mean_after_many_updates_tracks_evidence_ratio():
    r = StrategyReliability(name="x")
    for _ in range(80):
        r.update(True)
    for _ in range(20):
        r.update(False)
    # α=81, β=21, mean ≈ 0.794. With the uniform prior (added to 80/20)
    # the posterior mean is 81/(81+21).
    assert abs(r.mean - 81 / 102) < 1e-6


def test_credible_interval_tightens_with_more_observations():
    r1 = StrategyReliability(name="x", alpha=2, beta=2)
    r2 = StrategyReliability(name="x", alpha=100, beta=100)
    lo1, hi1 = r1.credible_interval()
    lo2, hi2 = r2.credible_interval()
    width1 = hi1 - lo1
    width2 = hi2 - lo2
    # Same mean (0.5) but tighter interval for more observations.
    assert width2 < width1


def test_fit_from_eval_report_uniform_prior_offset():
    table = {
        "let-binding": {"tp": 95, "fp_on_gold_symbols": 230},
        "denotation": {"tp": 32, "fp_on_gold_symbols": 88},
    }
    rels = fit_reliabilities_from_eval_report(table)
    assert rels["let-binding"].alpha == 96  # 1 + 95
    assert rels["let-binding"].beta == 231  # 1 + 230
    assert rels["let-binding"].n_observations == 325


def test_expected_info_gain_is_positive_for_unobserved_strategy():
    r = StrategyReliability(name="x")  # Beta(1,1), n=0
    assert r.expected_info_gain(1000) > 0


def test_expected_info_gain_shrinks_with_observations():
    """Asymptotically tighter posteriors gain less from new data."""
    r1 = StrategyReliability(name="x", alpha=10, beta=10, n_observations=18)
    r2 = StrategyReliability(name="x", alpha=1000, beta=1000, n_observations=1998)
    g1 = r1.expected_info_gain(500)
    g2 = r2.expected_info_gain(500)
    assert g1 > g2 > 0


def test_expected_batch_info_gain_returns_per_strategy_dict():
    rels = {
        "a": StrategyReliability("a", alpha=10, beta=10, n_observations=18),
        "b": StrategyReliability("b", alpha=2, beta=2, n_observations=2),
    }
    proj = expected_batch_info_gain(rels, {"a": 100, "b": 100})
    assert set(proj.keys()) == {"a", "b"}
    # b has higher variance (smaller posterior n), so more to gain.
    assert proj["b"] > proj["a"]


def test_update_from_agreement_corroborates_both_strategies():
    rels = {
        "let": StrategyReliability("let", alpha=10, beta=10, n_observations=18),
        "denote": StrategyReliability("denote", alpha=10, beta=10, n_observations=18),
    }
    bindings = [[("let", "X", "Group"), ("denote", "X", "Group")]]
    a_before = rels["let"].alpha
    update_from_agreement(rels, bindings)
    # both strategies should gain alpha because their canons agreed
    assert rels["let"].alpha > a_before


def test_update_from_agreement_punishes_lower_reliability_on_disagreement():
    rels = {
        "trusted": StrategyReliability("trusted", alpha=100, beta=10, n_observations=108),
        "noisy": StrategyReliability("noisy", alpha=10, beta=100, n_observations=108),
    }
    bindings = [[("trusted", "X", "Group"), ("noisy", "X", "Ring")]]
    noisy_beta_before = rels["noisy"].beta
    trusted_alpha_before = rels["trusted"].alpha
    update_from_agreement(rels, bindings)
    # Noisy strategy takes the hit
    assert rels["noisy"].beta > noisy_beta_before
    # Trusted strategy unchanged
    assert rels["trusted"].alpha == trusted_alpha_before


def test_update_from_agreement_ignores_none_canon():
    """When a strategy has no canon, its vote shouldn't count."""
    rels = {
        "a": StrategyReliability("a"),
        "b": StrategyReliability("b"),
    }
    bindings = [[("a", "X", None), ("b", "X", "Group")]]
    a_alpha = rels["a"].alpha
    b_alpha = rels["b"].alpha
    update_from_agreement(rels, bindings)
    # No update — strategy a doesn't have a canon, so no agreement signal.
    assert rels["a"].alpha == a_alpha
    assert rels["b"].alpha == b_alpha


def test_update_from_agreement_stable_over_many_repeated_agreements():
    """Sanity: when two strategies consistently agree on the same
    binding, both posteriors should stay near their initial mean
    (they get α boost only, but proportionally so the ratio holds)."""
    rels = {
        "a": StrategyReliability("a", alpha=20, beta=20, n_observations=38),
        "b": StrategyReliability("b", alpha=20, beta=20, n_observations=38),
    }
    initial_mean_a = rels["a"].mean
    bindings = [[("a", "X", "Group"), ("b", "X", "Group")]] * 50
    update_from_agreement(rels, bindings)
    # Mean shifts upward (only α grows, not β) — that's expected.
    # But the system shouldn't blow up or go to a weird fixed point.
    assert 0.5 <= rels["a"].mean <= 1.0
    assert rels["a"].mean > initial_mean_a


def test_update_from_agreement_keeps_disagreement_weight_correct():
    """When trusted disagrees with noisy, noisy loses β-weight
    proportional to trusted's mean. After many iterations the
    noisy strategy's mean should drop, trusted should stay."""
    rels = {
        "trusted": StrategyReliability("trusted", alpha=90, beta=10, n_observations=98),
        "noisy": StrategyReliability("noisy", alpha=50, beta=50, n_observations=98),
    }
    trusted_mean_initial = rels["trusted"].mean
    noisy_mean_initial = rels["noisy"].mean
    bindings = [[("trusted", "X", "Group"), ("noisy", "X", "Ring")]] * 100
    update_from_agreement(rels, bindings)
    # Noisy's mean falls; trusted's mean unchanged (no update on it).
    assert rels["noisy"].mean < noisy_mean_initial
    assert rels["trusted"].mean == trusted_mean_initial


# ============================================================
# CanonPosterior + combine_strategy_votes
# ============================================================

def test_combine_votes_unanimous_high_reliability_picks_that_canon():
    rels = {
        "a": StrategyReliability("a", alpha=90, beta=10, n_observations=98),
        "b": StrategyReliability("b", alpha=90, beta=10, n_observations=98),
    }
    votes = [("a", "Group"), ("b", "Group")]
    post = combine_strategy_votes("X", votes, rels)
    top_canon, top_prob = post.top1()
    assert top_canon == "Group"
    assert top_prob > 0.9


def test_combine_votes_high_trust_beats_low_trust_when_disagreeing():
    rels = {
        "trusted": StrategyReliability("trusted", alpha=95, beta=5, n_observations=98),
        "noisy": StrategyReliability("noisy", alpha=20, beta=80, n_observations=98),
    }
    votes = [("trusted", "Group"), ("noisy", "Ring")]
    post = combine_strategy_votes("X", votes, rels)
    top_canon, _ = post.top1()
    assert top_canon == "Group"


def test_combine_votes_no_votes_returns_null_posterior():
    rels = {}
    post = combine_strategy_votes("X", [], rels)
    top_canon, top_prob = post.top1()
    assert top_canon is None
    assert top_prob == 1.0


def test_combine_votes_two_for_one_against_majority_wins():
    rels = {
        "a": StrategyReliability("a", alpha=80, beta=20, n_observations=98),
        "b": StrategyReliability("b", alpha=80, beta=20, n_observations=98),
        "c": StrategyReliability("c", alpha=80, beta=20, n_observations=98),
    }
    votes = [("a", "Group"), ("b", "Group"), ("c", "Ring")]
    post = combine_strategy_votes("X", votes, rels)
    assert post.candidates.get("Group", 0) > post.candidates.get("Ring", 0)


def test_combine_votes_with_explicit_prior_uses_it():
    rels = {
        "a": StrategyReliability("a", alpha=80, beta=20, n_observations=98),
        "b": StrategyReliability("b", alpha=80, beta=20, n_observations=98),
    }
    votes = [("a", "Group"), ("b", "Ring")]
    prior = {"Group": 0.85, "Ring": 0.10}
    post = combine_strategy_votes("X", votes, rels, prior=prior)
    top_canon, _ = post.top1()
    assert top_canon == "Group"


def test_combine_votes_canon_posterior_normalizes_to_one():
    rels = {
        "a": StrategyReliability("a", alpha=70, beta=30, n_observations=98),
        "b": StrategyReliability("b", alpha=70, beta=30, n_observations=98),
    }
    votes = [("a", "Group"), ("b", "Ring")]
    post = combine_strategy_votes("X", votes, rels)
    total = sum(post.candidates.values()) + post.null_mass
    assert abs(total - 1.0) < 1e-9
