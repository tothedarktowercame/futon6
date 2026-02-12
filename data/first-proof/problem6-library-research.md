# Problem 6 Library Research: Grouped Paving and Vertex-Induced Sparsification

Date: 2026-02-12
Author: Claude (Opus 4.6)
Scope: Targeted literature search for the open bridge in Problem 6 (Assumption V / GPL-H).

## What we searched for

The proof attempt (`problem6-proof-attempt.md`) reduces the full theorem to a
single open implication called GPL-H: given hypotheses H1-H4 on a Case-2b
barrier state, prove `min_v ||Y_t(v)|| <= theta < 1` for the barrier-normalized
grouped vertex updates.

This is equivalent to a "grouped paving lemma" where the partition of PSD atoms
into vertex groups is **fixed by the graph structure** (not optimized).

We searched five specific topics across 47 web queries (arXiv, journals,
conference proceedings, author pages) covering 2020-2026.

## Topic 1: Grouped paving lemma / block paving for PSD matrices

**No direct results found.**

The phrase "grouped paving lemma" does not appear in the literature. Classical
paving results (Bourgain-Tzafriri, Tropp 2008) bound the minimum block norm
when the partition is **chosen optimally or randomly**, not when pre-assigned.

- Tropp (2008), "The Random Paving Property for Uniformly Bounded Matrices"
  [arXiv:math/0612070]. Random partition into r blocks; min block norm is small
  w.h.p. But partition is drawn uniformly, not fixed.

**Verdict**: No result for pre-assigned partition minimum norm.

## Topic 2: Matrix discrepancy with fixed partitions of PSD atoms

All results optimize over the signing/partition. None address a fixed partition.

- Dadush, Jiang, Reis (2021/STOC 2022), "A New Framework for Matrix
  Discrepancy: Partial Coloring Bounds via Mirror Descent" [arXiv:2111.03171].
  For symmetric d x d matrices with ||A_i|| <= 1 and rank(A_i) <= r: optimal
  signing gives ||sum x_i A_i|| = O(sqrt(n log(min(rm/n, r)))). Signs optimized.

- Bansal, Jiang, Meka (2022/STOC 2023), "Resolving Matrix Spencer Conjecture
  Up to Poly-logarithmic Rank" [arXiv:2208.11286]. Signs x_i in {-1,+1} with
  ||sum x_i A_i|| = O(sqrt(n)) when rank(A_i) <= n/log^3(n). Signs optimized.

- Hopkins, Raghavendra, Shetty (2021/STOC 2022), "Matrix Discrepancy from
  Quantum Communication" [arXiv:2110.10099]. SDP-based partial coloring.
  Signs optimized.

- Kunisky, Zhang (2023), "Average-Case Matrix Discrepancy" [arXiv:2307.10055].
  Upper/lower bounds for i.i.d. random matrices. Average-case, not fixed
  partition.

**Verdict**: All optimize over signs/partition. None give guarantees for
pre-assigned groups.

## Topic 3: MSS / Kadison-Singer / Weaver KS extensions to grouped settings

### Closest result: Bownik (2024)

- Bownik (2024), "Selector form of Weaver's conjecture, Feichtinger's
  conjecture, and frame sparsification" [arXiv:2405.18235].
  - Block diagonal extension: for block diagonal PSD random matrices with k
    blocks and per-block trace constraint epsilon, bound improves over naive
    union bound.
  - **Selector form of Weaver KS_r**: for block diagonal trace class operators,
    a selector (choosing one element from each group) achieves
    ||sum_{i in J} T_i - (1/2)T|| <= 2*sqrt(epsilon) + epsilon.
  - **Key distinction**: the selector is CHOSEN to achieve the bound, not
    pre-fixed. Paper explicitly notes "selector results are stronger than their
    partition counterparts."
  - **Relevance to GPL-H**: partially relevant. The block diagonal structure
    and grouped atoms are analogous, but GPL-H needs a bound for the
    graph-determined grouping, not an optimized selector.

### Other KS/MSS extensions (none address pre-grouped question)

- Bownik (2023), "On Akemann-Weaver Conjecture" [arXiv:2303.12954]. Extends
  Lyapunov theorem to higher-rank trace class operators. Existence of good
  partitions, not bounds for pre-assigned ones.

- Xu, Xu, Zhu (2021/J. Funct. Anal. 2023), "Improved bounds in Weaver's KS_r
  conjecture for high rank positive semidefinite matrices" [arXiv:2106.09205].
  Introduces (k,m)-characteristic polynomials; sharpens largest root bounds.
  Existential partition, not pre-fixed.

- Zhang, Zhang (ICALP 2023), "A Hyperbolic Extension of Kadison-Singer Type
  Results" [LIPIcs.ICALP.2023.108]. Extends MSS to hyperbolic polynomials and
  Strongly Rayleigh distributions. Any partition, not fixed.

- Branden (2018), "Hyperbolic polynomials and the Kadison-Singer problem"
  [arXiv:1809.03255]. Better bounds for Weaver KS_r via hyperbolic
  polynomials. Existential.

**Verdict**: Bownik 2024 is closest but still optimizes the selector. The
pre-grouped minimum norm question is open.

## Topic 4: BSS barrier method for vertex selection (not edge selection)

- Kozyrev, Osinsky (2025), "Subset selection for matrices in spectral norm"
  [arXiv:2507.20435]. Deterministic greedy column selection using single
  barrier function Phi_l(Y) = tr(Y - lI)^{-1}. Achieves quantitative subset
  size bounds in O(nkm^2) time. **Partially relevant**: column (vertex)
  selection with BSS-type barrier, but for pseudoinverse norm, not graph
  Laplacian context.

- **No result found** for BSS vertex selection controlling induced subgraph
  Laplacian spectral norm.

**Verdict**: BSS applied to vertex/column selection exists (Kozyrev-Osinsky)
but not in the graph Laplacian setting.

## Topic 5: Spectral vertex sparsification with universal constants

**No direct result found.**

The specific question — finding S with |S| = Omega(epsilon * n) such that
L_{G[S]} <= epsilon * L (vertex-induced, no reweighting) — does not appear
in the literature.

Related but distinct:
- Edge sparsification (BSS, Spielman-Srivastava): mature, but edges not
  vertices.
- Vertex sparsification (Moitra 2009): cut preservation for terminals, not
  spectral Laplacian approximation.
- Spectral subspace sparsification (Jiang-Song-Woodruff-Zhang,
  arXiv:1810.03224): top-k eigenspace, edge sampling.
- Spectrahedral geometry of graph sparsifiers (SIAM J. Discrete Math. 2024,
  doi:10.1137/23M1610069): edge-based.

**Verdict**: The vertex-induced spectral sparsification question (Assumption V)
appears to be open.

## Summary table

| # | Topic | Closest result | Pre-grouped? |
|---|-------|---------------|-------------|
| 1 | Grouped paving | Tropp 2008 random paving | No — partition random/optimized |
| 2 | Matrix discrepancy | Bansal-Jiang-Meka 2022 | No — signs optimized |
| 3 | MSS/KS grouped | Bownik 2024 selector KS_r | Closest, but selector chosen |
| 4 | BSS vertex selection | Kozyrev-Osinsky 2025 | Partial — not graph Laplacian |
| 5 | Vertex sparsification | None | No direct result |

## Key conclusion

The "pre-grouped minimum norm" question — where a partition of PSD atoms into
groups is fixed in advance and one asks whether min_i ||Y_i|| < theta < 1 —
is an **open problem** not addressed by any paper found in 2020-2026.

All existing Kadison-Singer / Weaver KS / MSS / discrepancy results prove
**existence** of good partitions (or selectors) but do not guarantee bounds
for an arbitrary pre-assigned partition.

This confirms that GPL-H (the open bridge in Problem 6) requires a genuinely
new theorem, not an application of existing results.

## Directions suggested by the literature

1. **Bownik's selector framework** (2024): closest template. If the
   graph-determined vertex grouping can be shown to satisfy Bownik's block
   diagonal trace conditions, his selector bounds might apply. Key question:
   does the graph structure provide enough "block diagonal" regularity?

2. **Column subset selection** (Kozyrev-Osinsky 2025): the barrier potential
   Phi_l(Y) = tr(Y - lI)^{-1} is exactly the potential in the proof attempt's
   barrier greedy. Their quantitative subset size bounds might transfer to the
   vertex-induced Laplacian setting.

3. **Matrix discrepancy via mirror descent** (Dadush-Jiang-Reis 2021): the
   partial coloring framework handles low-rank atoms. If the vertex groups
   Y_t(v) (rank <= D0/epsilon) can be cast as discrepancy atoms, partial
   coloring might yield the needed bound.

## References

- [Tropp 2008](https://arxiv.org/abs/math/0612070)
- [Dadush-Jiang-Reis 2021](https://arxiv.org/abs/2111.03171)
- [Bansal-Jiang-Meka 2022](https://arxiv.org/abs/2208.11286)
- [Hopkins-Raghavendra-Shetty 2021](https://arxiv.org/abs/2110.10099)
- [Kunisky-Zhang 2023](https://arxiv.org/abs/2307.10055)
- [Bownik 2024](https://arxiv.org/abs/2405.18235)
- [Bownik 2023](https://arxiv.org/abs/2303.12954)
- [Xu-Xu-Zhu 2021](https://arxiv.org/abs/2106.09205)
- [Zhang-Zhang ICALP 2023](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ICALP.2023.108)
- [Branden 2018](https://arxiv.org/abs/1809.03255)
- [Kozyrev-Osinsky 2025](https://arxiv.org/abs/2507.20435)
- [Moitra 2009](https://link.springer.com/chapter/10.1007/978-3-642-15369-3_12)
- [Jiang-Song-Woodruff-Zhang 2018](https://arxiv.org/abs/1810.03224)
