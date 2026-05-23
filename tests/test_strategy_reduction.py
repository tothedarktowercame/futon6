"""Tests for futon6.strategy_reduction (F4 of canon-fingerprint-store)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from futon6.strategy_reduction import (
    StrategyMergeProposal,
    _canons_related,
    compute_concordance,
    propose_merges,
)


GRAPH = {
    "Group": {"AbelianGroup", "TopologicalGroup", "Subgroup"},
    "AbelianGroup": {"Group"},
    "TopologicalGroup": {"Group"},
    "Subgroup": {"Group"},
    "Ring": {"Field"},
    "Field": {"Ring"},
}


def test_canons_related_equal_is_related():
    assert _canons_related("Group", "Group", GRAPH)


def test_canons_related_graph_neighbour():
    assert _canons_related("Group", "AbelianGroup", GRAPH)
    assert _canons_related("AbelianGroup", "Group", GRAPH)


def test_canons_related_unrelated_returns_false():
    assert not _canons_related("Group", "Field", GRAPH)


def test_canons_related_case_insensitive_fallback():
    assert _canons_related("group", "AbelianGroup", GRAPH)
    assert _canons_related("GROUP", "abeliangroup", GRAPH)


def test_canons_related_none_returns_false():
    assert not _canons_related(None, "Group", GRAPH)
    assert not _canons_related("Group", None, GRAPH)


def test_concordance_two_strategies_unanimous():
    bindings = [
        [("a", "X", "Group"), ("b", "X", "Group")],
        [("a", "Y", "Ring"), ("b", "Y", "Ring")],
    ]
    c = compute_concordance(bindings, GRAPH)
    pair = c[("a", "b")]
    assert pair.n_co_firings == 2
    assert pair.n_agree_exact == 2
    assert pair.concordance == 1.0


def test_concordance_graph_adjacent_counts_as_agreement():
    bindings = [
        [("a", "X", "Group"), ("b", "X", "AbelianGroup")],
        [("a", "Y", "Group"), ("b", "Y", "Subgroup")],
    ]
    c = compute_concordance(bindings, GRAPH)
    pair = c[("a", "b")]
    assert pair.n_co_firings == 2
    assert pair.n_agree_exact == 0
    assert pair.n_agree_graph == 2
    assert pair.concordance == 1.0


def test_concordance_disagreement_below_one():
    bindings = [
        [("a", "X", "Group"), ("b", "X", "Group")],          # agree
        [("a", "Y", "Group"), ("b", "Y", "Field")],          # disagree
    ]
    c = compute_concordance(bindings, GRAPH)
    pair = c[("a", "b")]
    assert pair.n_co_firings == 2
    assert pair.n_agree_exact == 1
    assert pair.n_disagree == 1
    assert pair.concordance == 0.5


def test_concordance_ignores_none_canons():
    bindings = [
        [("a", "X", None), ("b", "X", "Group")],
    ]
    c = compute_concordance(bindings, GRAPH)
    assert ("a", "b") not in c  # nothing to compare against


def test_concordance_keys_normalised_sorted():
    bindings = [
        [("zz", "X", "Group"), ("aa", "X", "Group")],
    ]
    c = compute_concordance(bindings, GRAPH)
    # Even though input was zz, aa, output key is sorted lexically
    assert ("aa", "zz") in c
    assert ("zz", "aa") not in c


def test_concordance_ignores_same_strategy_pairs():
    """A strategy paired with itself is not a co-firing."""
    bindings = [
        [("a", "X", "Group"), ("a", "X", "Group")],
    ]
    c = compute_concordance(bindings, GRAPH)
    assert ("a", "a") not in c


def test_concordance_collects_examples():
    bindings = [
        [("a", "X", "Group"), ("b", "X", "AbelianGroup")],
        [("a", "Y", "Field"), ("b", "Y", "Ring")],
    ]
    c = compute_concordance(bindings, GRAPH)
    pair = c[("a", "b")]
    assert ("X", "Group", "AbelianGroup") in pair.examples_agree
    assert ("Y", "Field", "Ring") in pair.examples_agree


def test_propose_merges_filters_low_concordance():
    proposals = {
        ("a", "b"): StrategyMergeProposal(
            strategy_a="a", strategy_b="b",
            n_co_firings=100, n_agree_exact=80, n_agree_graph=0,
            n_disagree=20, concordance=0.8,
        ),
        ("c", "d"): StrategyMergeProposal(
            strategy_a="c", strategy_b="d",
            n_co_firings=100, n_agree_exact=30, n_agree_graph=10,
            n_disagree=60, concordance=0.4,
        ),
    }
    result = propose_merges(proposals, min_concordance=0.7, min_co_firings=30)
    assert len(result) == 1
    assert result[0].strategy_a == "a"


def test_propose_merges_filters_low_co_firings():
    proposals = {
        ("a", "b"): StrategyMergeProposal(
            strategy_a="a", strategy_b="b",
            n_co_firings=10, n_agree_exact=10, n_agree_graph=0,
            n_disagree=0, concordance=1.0,
        ),
    }
    # 10 < default min_co_firings=30 → filtered out even though
    # concordance is perfect; small samples are noise.
    result = propose_merges(proposals)
    assert result == []


def test_propose_merges_sorted_by_concordance_desc():
    proposals = {
        ("a", "b"): StrategyMergeProposal(
            strategy_a="a", strategy_b="b",
            n_co_firings=100, n_agree_exact=80, n_agree_graph=0,
            n_disagree=20, concordance=0.8,
        ),
        ("c", "d"): StrategyMergeProposal(
            strategy_a="c", strategy_b="d",
            n_co_firings=100, n_agree_exact=90, n_agree_graph=0,
            n_disagree=10, concordance=0.9,
        ),
    }
    result = propose_merges(proposals, min_concordance=0.7, min_co_firings=30)
    assert [p.strategy_a for p in result] == ["c", "a"]
