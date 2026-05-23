"""Literature-lifted strategy merge proposer — slice F4 of
M-canon-fingerprint-store.md.

The idea (from Joe's reframing of Friston Bayesian model reduction):
when two strategies systematically emit canons that the literature
already says are related (PM `\\pmrelated`, ProofWiki/nLab/Wikipedia
graph adjacencies), those strategies are capturing the same
underlying concept under different framings. They are candidates for
merging.

Why merge rather than just match-via-ancestry at canon-comparison
time? Because at arbitration time we want one vote per concept,
not two. If `let-binding` says "Group" and `inline-is-a` says
"AbelianGroup" for the same X, and the graph says those are
related, that's *one* unified vote for the group-family concept —
the per-binding posterior should treat it as such, not double-count
or worse, treat them as contradictory.

Two phases:
  1. CONCORDANCE — for every strategy pair (A, B), what fraction of
     their co-firings on the same symbol produce graph-adjacent
     canons? High concordance = merge candidate.
  2. PROPOSAL — strategies with concordance above a threshold get
     proposed for merge. Caller validates against held-out gold.

The proposal is data; this module doesn't mutate the strategy
library. The merge action belongs to the caller's discretion.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class StrategyMergeProposal:
    """A candidate merge with the evidence that supports it."""
    strategy_a: str
    strategy_b: str
    n_co_firings: int          # symbols where both A and B emitted
    n_agree_exact: int         # exact canon match
    n_agree_graph: int         # graph-adjacent canons (related-but-not-equal)
    n_disagree: int            # canons not related in graph
    concordance: float         # (n_agree_exact + n_agree_graph) / n_co_firings
    examples_agree: list[tuple[str, str, str]] = field(default_factory=list)
    examples_disagree: list[tuple[str, str, str]] = field(default_factory=list)


def _canons_related(
    canon_a: str,
    canon_b: str,
    literature_graph: dict[str, set[str]],
) -> bool:
    """Return True if canons are equal or share a graph edge.

    The literature_graph is symmetric (edges go both ways), as
    built by build-canon-ancestry-pm.py.
    """
    if not canon_a or not canon_b:
        return False
    if canon_a == canon_b:
        return True
    if canon_a in literature_graph and canon_b in literature_graph[canon_a]:
        return True
    if canon_b in literature_graph and canon_a in literature_graph[canon_b]:
        return True
    # Case-insensitive fallback for caller-supplied graphs that don't
    # follow the same casing convention as the engine.
    a_low, b_low = canon_a.lower(), canon_b.lower()
    for key, neighbours in literature_graph.items():
        if key.lower() == a_low and any(n.lower() == b_low for n in neighbours):
            return True
        if key.lower() == b_low and any(n.lower() == a_low for n in neighbours):
            return True
    return False


def compute_concordance(
    bindings_by_paper: Iterable[list[tuple[str, str, str]]],
    literature_graph: dict[str, set[str]],
    max_examples: int = 5,
) -> dict[tuple[str, str], StrategyMergeProposal]:
    """For each (strategy_a, strategy_b) pair, compute their
    concordance — fraction of co-firings on the same symbol where
    their canons are either equal or graph-related.

    `bindings_by_paper` yields, per paper, a list of
    `(strategy, symbol, canon)` tuples (canon can be None to skip).

    Returns dict keyed by (a, b) with a < b (lexicographic) so each
    pair appears exactly once. Pairs that never co-fired are absent.
    """
    pair_stats: dict[tuple[str, str], dict] = defaultdict(lambda: {
        "co_firings": 0,
        "agree_exact": 0,
        "agree_graph": 0,
        "disagree": 0,
        "examples_agree": [],
        "examples_disagree": [],
    })
    for bindings in bindings_by_paper:
        by_symbol: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for strat, sym, canon in bindings:
            if canon is None:
                continue
            by_symbol[sym].append((strat, canon))
        for sym, votes in by_symbol.items():
            for i, (sa, ca) in enumerate(votes):
                for sb, cb in votes[i + 1 :]:
                    if sa == sb:
                        continue
                    key = tuple(sorted([sa, sb]))
                    stat = pair_stats[key]
                    stat["co_firings"] += 1
                    if ca == cb:
                        stat["agree_exact"] += 1
                        if len(stat["examples_agree"]) < max_examples:
                            stat["examples_agree"].append((sym, ca, cb))
                    elif _canons_related(ca, cb, literature_graph):
                        stat["agree_graph"] += 1
                        if len(stat["examples_agree"]) < max_examples:
                            stat["examples_agree"].append((sym, ca, cb))
                    else:
                        stat["disagree"] += 1
                        if len(stat["examples_disagree"]) < max_examples:
                            stat["examples_disagree"].append((sym, ca, cb))
    out: dict[tuple[str, str], StrategyMergeProposal] = {}
    for (a, b), stat in pair_stats.items():
        n = stat["co_firings"]
        if n == 0:
            continue
        concordance = (stat["agree_exact"] + stat["agree_graph"]) / n
        out[(a, b)] = StrategyMergeProposal(
            strategy_a=a,
            strategy_b=b,
            n_co_firings=n,
            n_agree_exact=stat["agree_exact"],
            n_agree_graph=stat["agree_graph"],
            n_disagree=stat["disagree"],
            concordance=concordance,
            examples_agree=stat["examples_agree"],
            examples_disagree=stat["examples_disagree"],
        )
    return out


def propose_merges(
    concordance: dict[tuple[str, str], StrategyMergeProposal],
    min_concordance: float = 0.7,
    min_co_firings: int = 30,
) -> list[StrategyMergeProposal]:
    """Filter concordance results to merge proposals.

    Defaults: ≥70% concordance, ≥30 co-firings (so the proposal is
    supported by enough evidence to be more than noise). Both are
    tunable per call; the caller decides the threshold for their
    confidence level.
    """
    return sorted(
        (p for p in concordance.values()
         if p.concordance >= min_concordance and p.n_co_firings >= min_co_firings),
        key=lambda p: -p.concordance,
    )
