"""Bayesian posterior layer over the symbol-grounding strategy library.

Following M-bayesian-structure-learning.md: replace the heuristic
per-strategy counters with reliability *posteriors* — Beta(α, β)
distributions whose mean is the strategy's expected precision and
whose credible interval is our uncertainty about that precision.

Two evidence channels feed posteriors:

  1. **Gold-match evidence** (supervised): each engine binding on
     a gold symbol is a TP if its canon matches gold, an FP
     otherwise. Posterior updates are clean Bayesian Beta updates:
     α += TP_count, β += FP_count.

  2. **Cross-strategy agreement** (semi-supervised, optional): when
     two strategies independently emit the same (symbol, canon)
     within a paper, both posteriors gain corroboration evidence.
     Weighted by the *other* strategy's current reliability so a
     low-trust strategy's vote doesn't dominate.

The first artifact (§8 of the mission doc) uses only channel 1 to
produce side-by-side numbers vs the heuristic eval. Channel 2 is
the path forward — it lets us update posteriors on Rob's batch
without needing per-paper gold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

try:
    from scipy import stats
    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    _HAVE_SCIPY = False


@dataclass
class StrategyReliability:
    """Beta(α, β) posterior over a single strategy's precision.

    Default prior Beta(1, 1) is uniform on [0, 1]. After observing
    n_correct successes and n_wrong failures, the posterior is
    Beta(1 + n_correct, 1 + n_wrong).
    """
    name: str
    alpha: float = 1.0
    beta: float = 1.0
    n_observations: int = 0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        s = self.alpha + self.beta
        return (self.alpha * self.beta) / (s * s * (s + 1.0))

    def credible_interval(self, level: float = 0.95) -> tuple[float, float]:
        """Equal-tailed credible interval at the given probability level."""
        if not _HAVE_SCIPY:
            # Fall back to a normal-approximation interval. Acceptable for
            # large α + β; the CLI will warn callers when scipy is missing.
            from math import sqrt
            m = self.mean
            sd = sqrt(self.variance)
            z = 1.959963984540054 if level == 0.95 else 1.6448536269514722
            return (max(0.0, m - z * sd), min(1.0, m + z * sd))
        lo_q = (1.0 - level) / 2.0
        hi_q = 1.0 - lo_q
        lo = float(stats.beta.ppf(lo_q, self.alpha, self.beta))
        hi = float(stats.beta.ppf(hi_q, self.alpha, self.beta))
        return (lo, hi)

    def update(self, correct: bool, weight: float = 1.0) -> None:
        """Single Beta update. `weight` lets semi-supervised evidence
        contribute fractionally (e.g. cross-strategy agreement weighted
        by the *other* strategy's mean reliability)."""
        if weight <= 0.0:
            return
        if correct:
            self.alpha += weight
        else:
            self.beta += weight
        self.n_observations += 1

    def expected_info_gain(self, n_additional_observations: int) -> float:
        """Expected reduction in posterior variance after N more
        observations, assuming the same correct/wrong ratio.

        This is the "epistemic value" of N more papers for THIS
        strategy. Strategies with high variance + low n_observations
        benefit most. Returns a unit-free positive number (variance
        delta).
        """
        if self.n_observations == 0:
            return self.variance  # everything to learn
        # If we assume the next N observations follow the current
        # mean precision, the new α' / β' / variance' is deterministic
        # and we can compute the variance delta directly.
        ratio_correct = self.mean
        new_alpha = self.alpha + n_additional_observations * ratio_correct
        new_beta = self.beta + n_additional_observations * (1.0 - ratio_correct)
        s_new = new_alpha + new_beta
        new_var = (new_alpha * new_beta) / (s_new * s_new * (s_new + 1.0))
        return max(0.0, self.variance - new_var)


def fit_reliabilities_from_eval_report(
    strategy_table: dict,
) -> dict[str, StrategyReliability]:
    """Build StrategyReliability posteriors from an eval-grounding-gold
    `strategy_table`.

    Expected shape per strategy entry:
        {"tp": int, "fp_on_gold_symbols": int, ...}

    Returns dict[strategy_name -> StrategyReliability]. Beta(1, 1)
    prior; α = 1 + tp, β = 1 + fp.
    """
    out: dict[str, StrategyReliability] = {}
    for strat, info in strategy_table.items():
        tp = int(info.get("tp", 0))
        fp = int(info.get("fp_on_gold_symbols", 0))
        rel = StrategyReliability(
            name=strat,
            alpha=1.0 + tp,
            beta=1.0 + fp,
            n_observations=tp + fp,
        )
        out[strat] = rel
    return out


def update_from_agreement(
    reliabilities: dict[str, StrategyReliability],
    bindings_per_paper: Iterable[list[tuple[str, str, str]]],
) -> None:
    """Update posteriors via cross-strategy agreement (channel 2).

    `bindings_per_paper` yields, for each paper, a list of
    `(strategy_name, symbol, canon)` tuples produced by the engine.
    Within a paper, two strategies agreeing on (symbol, canon) is a
    +1 corroboration vote weighted by the *other* strategy's current
    mean reliability. Disagreement (same symbol, different canon)
    is a fractional negative update on whichever strategy has the
    lower current mean — the higher one is presumed correct.

    This is genuinely semi-supervised: it adds evidence without
    requiring gold. The trade-off is bootstrap circularity — at
    init when all reliabilities are equal, agreement updates are
    proportional, but once one strategy pulls ahead it dominates.
    Call this AFTER the gold-match init when possible.
    """
    for bindings in bindings_per_paper:
        # Group by symbol within the paper
        by_symbol: dict[str, list[tuple[str, str]]] = {}
        for strat, sym, canon in bindings:
            if canon is None:
                continue
            by_symbol.setdefault(sym, []).append((strat, canon))
        for sym, votes in by_symbol.items():
            for i, (sa, ca) in enumerate(votes):
                for sb, cb in votes[i + 1 :]:
                    if sa == sb:
                        continue
                    rel_a = reliabilities.get(sa)
                    rel_b = reliabilities.get(sb)
                    if rel_a is None or rel_b is None:
                        continue
                    if ca == cb:
                        rel_a.update(True, weight=rel_b.mean)
                        rel_b.update(True, weight=rel_a.mean)
                    else:
                        # Whichever has lower mean reliability takes the FP hit.
                        if rel_a.mean < rel_b.mean:
                            rel_a.update(False, weight=rel_b.mean)
                        elif rel_b.mean < rel_a.mean:
                            rel_b.update(False, weight=rel_a.mean)
                        # If exactly tied, no update — symmetric
                        # disagreement carries no signal.


def expected_batch_info_gain(
    reliabilities: dict[str, StrategyReliability],
    expected_observations_per_strategy: dict[str, int],
) -> dict[str, float]:
    """Expected per-strategy variance reduction from processing a batch.

    `expected_observations_per_strategy` should be roughly
    `(papers_in_batch) * (mean_emissions_per_paper_for_this_strategy)`,
    derivable from prior runs. Returns dict[strategy_name -> info gain]
    so the operator can see which strategies the batch will tighten.

    Headline number: sum of values is the total info gain a batch
    delivers; comparable to GPU-hour cost in the send-to-Rob decision.
    """
    return {
        name: rel.expected_info_gain(expected_observations_per_strategy.get(name, 0))
        for name, rel in reliabilities.items()
    }
