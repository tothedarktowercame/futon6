"""Topic-aware priors for canon arbitration.

Two priors live here, both supplying *multiplicative factors* that
the caller composes with the existing canon-fingerprint-store prior
inside `bayesian_grounding.combine_strategy_votes`:

  1. **MSC topic prior** — P(canon | MSC code) from PlanetMath's
     MSC-tagged corpus (each PM entry has one or more `:msc-codes`).
     Down-weights canons whose probability mass lives in MSC codes
     unrelated to the paper being processed. For an arxiv math.CT
     paper we expect canons from MSC 18 (CT) to dominate; a
     StableMarriageProblem (MSC 91, operations research) gets a
     low conditional prior and falls out of arbitration.

  2. **SE corpus-frequency prior** — marginal P(canon) over
     math.StackExchange + MathOverflow question/answer bodies.
     Down-weights canons that essentially never appear in real
     mathematical discourse, regardless of topic.

Both priors are JSON-backed and updateable. The MSC prior is the
natural learn-as-you-go target: when we run on the full arxiv (each
paper has known categories), after arbitration we have
(paper_id, msc_codes, accepted_canons) which lets us increment
P(canon | msc) in-flight — same online-EM pattern as the canon store.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


# Arxiv math subject classes → MSC primary codes (2-digit).
# Used at inference time so a paper's `categories: ["math.CT"]`
# resolves to MSC primary "18" for prior lookup.
# Mapping below covers the major math.XX classes; non-math arxiv
# classes (cs.*, hep-th, etc.) are passed through as no-MSC.
ARXIV_TO_MSC_PRIMARY: dict[str, list[str]] = {
    "math.AC": ["13"],            # Commutative algebra
    "math.AG": ["14"],            # Algebraic geometry
    "math.AP": ["35"],            # Analysis of PDEs
    "math.AT": ["55"],            # Algebraic topology
    "math.CA": ["26", "30", "33"],  # Classical analysis
    "math.CO": ["05"],            # Combinatorics
    "math.CT": ["18"],            # Category theory
    "math.CV": ["32"],            # Complex variables
    "math.DG": ["53"],            # Differential geometry
    "math.DS": ["37"],            # Dynamical systems
    "math.FA": ["46"],            # Functional analysis
    "math.GM": ["00"],            # General mathematics
    "math.GN": ["54"],            # General topology
    "math.GR": ["20"],            # Group theory
    "math.GT": ["57"],            # Geometric topology
    "math.HO": ["01"],            # History/overview
    "math.IT": ["94"],            # Information theory
    "math.KT": ["19"],            # K-theory
    "math.LO": ["03"],            # Logic
    "math.MG": ["51", "52"],      # Metric geometry
    "math.MP": ["81", "82"],      # Mathematical physics
    "math.NA": ["65"],            # Numerical analysis
    "math.NT": ["11"],            # Number theory
    "math.OA": ["46", "47"],      # Operator algebras
    "math.OC": ["49", "90", "93"],  # Optimization, control
    "math.PR": ["60"],            # Probability
    "math.QA": ["16", "17", "18", "81"],  # Quantum algebra
    "math.RA": ["16", "17"],      # Rings and algebras
    "math.RT": ["20", "22"],      # Representation theory
    "math.SG": ["53"],            # Symplectic geometry
    "math.SP": ["47"],            # Spectral theory
    "math.ST": ["62"],            # Statistics
}


def arxiv_categories_to_msc(categories: list[str]) -> list[str]:
    """Resolve arxiv `categories` list to MSC primary codes.

    Non-math classes contribute nothing; the union of all matched
    primaries is returned (order preserved, deduplicated)."""
    out: list[str] = []
    for cat in categories or []:
        for code in ARXIV_TO_MSC_PRIMARY.get(cat, []):
            if code not in out:
                out.append(code)
    return out


# ============================================================
# MSC topic prior
# ============================================================

@dataclass
class MSCTopicPrior:
    """P(canon | MSC primary code) over a corpus.

    Stored as `counts[canon][msc_primary] = n`, plus a per-canon
    marginal `totals[canon]`. The marginal is the denominator for
    `prior(canon, msc_primaries)`: we sum the canon's mass in the
    requested MSC primaries and divide by total mass.

    Smoothing: a small additive constant prevents zero priors for
    canons that legitimately exist but happen to never coincide
    with the requested MSC primaries in our corpus.
    """
    counts: dict[str, dict[str, int]] = field(default_factory=dict)
    totals: dict[str, int] = field(default_factory=dict)
    # Total mass across the whole corpus (denominator for unseen
    # canons in the corpus-frequency sense — separate concept from
    # the per-canon conditional).
    grand_total: int = 0

    def add(self, canon: str, msc_primary: str, n: int = 1) -> None:
        slot = self.counts.setdefault(canon, {})
        slot[msc_primary] = slot.get(msc_primary, 0) + n
        self.totals[canon] = self.totals.get(canon, 0) + n
        self.grand_total += n

    def prior(
        self,
        canon: str,
        msc_primaries: Iterable[str],
        smoothing: float = 0.5,
    ) -> float:
        """Return P(canon ∈ requested MSC | seen) with additive smoothing.

        Range: roughly (0, 1]. A canon that lives entirely outside
        the requested MSC primaries gets `smoothing / (total + n_primaries*smoothing)`;
        a canon that lives entirely inside gets `(total + smoothing) /
        (total + n_primaries*smoothing)` ≈ 1 for any non-trivial total.

        Unseen canons (not in the store at all) return 1.0 — we don't
        penalise canons our PM corpus simply doesn't cover; that's the
        SE-corpus prior's job.
        """
        if canon not in self.totals:
            return 1.0
        primaries = list(msc_primaries)
        if not primaries:
            return 1.0  # caller didn't supply a topic → don't down-weight
        matched = sum(
            self.counts.get(canon, {}).get(p, 0) for p in primaries
        )
        total = self.totals[canon]
        n_primaries = len(primaries)
        return (matched + smoothing) / (total + n_primaries * smoothing)

    def update_from_run(
        self,
        emissions: Iterable[tuple[str, float]],
        msc_primaries: Iterable[str],
        min_confidence: float = 0.5,
    ) -> int:
        """Online update from a single paper's arbitration output.

        `emissions` is an iterable of (canon, posterior_probability)
        pairs — the engine's accepted bindings for the paper.
        `msc_primaries` are the MSC primary codes that paper sits in
        (resolved from its arxiv categories).

        Only emissions with `posterior >= min_confidence` are folded
        in — otherwise we'd be folding our own noise back into the
        prior, an online-EM degenerate-collapse failure mode.

        Returns the number of (canon, msc) pairs updated."""
        primaries = list(msc_primaries)
        if not primaries:
            return 0
        n_updates = 0
        for canon, prob in emissions:
            if prob < min_confidence:
                continue
            for p in primaries:
                self.add(canon, p, n=1)
                n_updates += 1
        return n_updates

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({
                "schema_version": "v1",
                "n_canons": len(self.counts),
                "grand_total": self.grand_total,
                "counts": self.counts,
                "totals": self.totals,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "MSCTopicPrior":
        if not path.exists():
            return cls()
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            counts={k: dict(v) for k, v in d.get("counts", {}).items()},
            totals=dict(d.get("totals", {})),
            grand_total=d.get("grand_total", 0),
        )


# ============================================================
# SE corpus-frequency prior
# ============================================================

@dataclass
class SECorpusPrior:
    """P(canon) over math.SE + MathOverflow body text.

    Counts how often each canon's natural-language form
    (e.g. "stable marriage problem") appears in question/answer
    bodies. Sets a floor on plausible canons regardless of MSC.

    Optional `by_tag`: P(canon | SE tag) for the future MO/MSE
    facetised path. v1 keeps only the marginal — facet integration
    is a follow-on.
    """
    counts: dict[str, int] = field(default_factory=dict)
    grand_total: int = 0
    n_documents: int = 0

    def add(self, canon: str, n: int = 1) -> None:
        self.counts[canon] = self.counts.get(canon, 0) + n
        self.grand_total += n

    def prior(self, canon: str, smoothing: float = 1.0) -> float:
        """Multiplicative factor in (0, 1].

        Log-scaled so single-letter or otherwise pathologically-common
        canons in the index (e.g. "A" appearing 94000 times in MO
        body text) don't flatten everything else to near-zero. Shape:
            (log(n+1) + smoothing) / (log(max_seen+1) + smoothing)
        Canons absent from corpus get `smoothing / (log(max_seen+1) +
        smoothing)` — small but non-zero. A 100-mention canon lands
        around 0.55 when max is 5000; a 5000-mention canon lands ~0.95.
        """
        if not self.counts:
            return 1.0
        from math import log
        max_seen = max(self.counts.values())
        n = self.counts.get(canon, 0)
        return (log(n + 1) + smoothing) / (log(max_seen + 1) + smoothing)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({
                "schema_version": "v1",
                "n_canons": len(self.counts),
                "grand_total": self.grand_total,
                "n_documents": self.n_documents,
                "counts": self.counts,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "SECorpusPrior":
        if not path.exists():
            return cls()
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            counts=dict(d.get("counts", {})),
            grand_total=d.get("grand_total", 0),
            n_documents=d.get("n_documents", 0),
        )


# ============================================================
# Helpers: canon name → natural-language form for corpus matching
# ============================================================

_CAMEL_SPLIT = re.compile(r"(?<!^)(?=[A-Z])")


def canon_to_phrase(canon: str) -> str:
    """Split a CamelCase canon into a lowercased space-separated phrase.

    "StableMarriageProblem" -> "stable marriage problem"
    "RingedSpace"          -> "ringed space"
    "C2category"           -> "c2category"   (leaves leading number/word alone)

    Used by SECorpusPrior.build to count canon occurrences in
    natural-language SE corpus text.
    """
    parts = _CAMEL_SPLIT.split(canon)
    return " ".join(p.lower() for p in parts if p)
