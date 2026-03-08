# FM-001 Strategy: Ramsey Numbers for Book Graphs

## Problem

Construct a graph G on 4n-2 vertices such that G contains no B_{n-1} and
complement(G) contains no B_n, where B_k is the book graph (k triangles
sharing a common K_2 edge). This witnesses R(B_{n-1}, B_n) >= 4n-1.

Tiers: T1 (n=25, 98 vertices), T2 (n=50, 198 vertices), T3 (general <10min).

## Mathematical Landscape

**Known results** (Wesley 2025):
- R(B_{n-1}, B_n) = 4n-1 for all n <= 20 (SAT/SMS + DRAT certificates)
- R(B_{n-1}, B_n) = 4n-1 whenever 2n-1 is a prime power ≡ 1 (mod 4),
  via 2-block-circulant Paley-type colorings
- The Rousseau-Sheehan upper bound gives R(B_{n-1}, B_n) <= 4n-1

**Implication**: The conjecture R(B_{n-1}, B_n) = 4n-1 for all n is very
likely true given Wesley's results. However, F1 must be run with genuine
falsification intent — not as a box-check. The SAT encoding developed for
F1 also becomes the verification backbone for C3/C4.

## Phase 1: FALSIFY

### F1-opposite — Falsification attempt

**Approach**: Computational search for counterexamples at small n.

- **Method**: For n = 3..12, enumerate (or SAT-solve) all 2-colorings of
  K_{4n-2} and check whether any avoids both B_{n-1} (red) and B_n (blue).
  If such a coloring exists on 4n-2 vertices, it witnesses R(B_{n-1},B_n)
  >= 4n-1. If no coloring on 4n-3 vertices avoids both, then R = 4n-1 exactly.
  We need a case where R < 4n-1 to falsify.
- **Tool**: SAT encoding. B_k-freeness can be encoded as: for every (k+2)-subset
  that forms a book, at least one edge is not monochromatic. The variable
  x_{ij} = 1 means edge (i,j) is red.
- **Expected outcome**: All n <= 12 confirm R = 4n-1 (consistent with Wesley).
  FALSIFICATION FAILS.
- **Exit criterion**: Either find a counterexample (n where R < 4n-1) → report
  and halt, or confirm R = 4n-1 for n <= 12 → mark F1 as RESOLVED:no-counterexample.
- **Resource bound**: Max 48 hours compute. If SAT is intractable for n > 8,
  switch to targeted random sampling.

### F2-literature — Literature search

**Approach**: Systematic search for existing results.

- **Method**: (a) Query ArSE (when corpus-1 bridge is up) for: book Ramsey
  constructions, Paley graph B_k-freeness proofs, non-prime-power Ramsey
  witnesses, (b) check Wesley 2025 and Wigderson 2023 for whether T1/T2
  n-values are already covered by block-circulant constructions, (c) check
  whether 2n-1 is prime power ≡ 1 (mod 4) for n=25 and n=50.
- **Key check for n=25**: 2(25)-1 = 49 = 7^2. Is 49 ≡ 1 (mod 4)? 49 mod 4 = 1. YES.
  So the Paley-type construction should work directly for T1.
- **Key check for n=50**: 2(50)-1 = 99 = 9 × 11. NOT a prime power.
  So we need a different approach for T2.
- **Exit criterion**: Produce a table: for each tier n, does the block-circulant
  construction apply? What alternative constructions exist for non-prime-power cases?

## Phase 2: CONSTRUCT

### C1-structure — Characterize B_k-freeness

**Approach**: Translate B_k-freeness into a local graph property.

- B_k has k+2 vertices: an edge (u,v) plus k vertices each adjacent to both
  u and v. So B_k-freeness means: for every edge (u,v) in G, |N(u) ∩ N(v)| < k.
  In other words, no edge has k common neighbors.
- **Deliverable**: A function `max_common_neighbors(G)` → max over all edges of
  |N(u) ∩ N(v)|. G is B_k-free iff max_common_neighbors(G) < k.

### C2-strategy — Construction method

**Primary approach: Paley-type block-circulant** (for n where 2n-1 is prime power ≡ 1 mod 4)

- Construct the Paley graph P(q) on GF(q) where q = 2n-1.
- The 2-block-circulant on 2q vertices: take two copies of GF(q) and connect
  them according to quadratic residue structure.
- This is known to work (Wesley 2025). For T1 (n=25, q=49=7^2), this applies.

**Fallback approach: Modified circulant** (for n where 2n-1 is not prime power)

- T2 needs n=50, so q=99 (composite). Block-circulant doesn't directly apply.
- **Ranked fallbacks** (try in order — QP-2 discipline: survey before committing):
  1. *Cayley graph on Z_99*: cheapest to test. Choose connection set S as
     quadratic residues mod each prime factor (9,11), combine via CRT.
     Quick computational test: generate candidate, run C1 checker.
  2. *Random algebraic construction*: sample Cayley graphs with random
     connection sets, verify B_k-freeness. Broader search, higher variance.
  3. *Optimization search*: simulated annealing / genetic algorithm starting
     from a near-Paley template. Most expensive, last resort.
- **Decision point**: After C1 delivers the max-common-neighbors checker,
  test Cayley Z_99 first (budget: 4 hours). If no candidate found, proceed
  to random algebraic (budget: 12 hours), then optimization (budget: 24 hours).

**Emergency fallback: Brute computation**

- For T1 (98 vertices), direct SAT encoding is feasible.
- For T2 (198 vertices), SAT is likely intractable. Must use algebraic or
  optimization approach.

### C3/C4 — Verification

- C3: check G is B_{n-1}-free (max_common_neighbors(G) < n-1)
- C4: check complement(G) is B_n-free (max_common_neighbors(complement(G)) < n)
- Both reduce to running the checker from C1 on G and its complement.

### T1/T2/T3 — Witness production

- T1 (n=25): Paley construction → verify → output adjacency string
- T2 (n=50): Modified circulant or optimization → verify → output adjacency string
- T3 (general): Parameterize the construction by n, handle both prime-power
  and non-prime-power cases, verify, output. Runtime target <10min for n<=100.

## Conductor Dispatch Plan

| Obligation | Assignee | Approach | Success criterion |
|-----------|----------|---------|-------------------|
| F1-opposite | codex | SAT solver for n=3..10 | All confirm 4n-1, or counterexample found |
| F2-literature | claude/corpus | @corpus queries + manual | Table of which n are covered by existing constructions |
| C1-structure | codex | Implement max_common_neighbors | Function passes on known examples |
| C2-strategy | claude | Analyze prime-power coverage, choose approach per tier | Written decision with fallback criteria |
| C3-verify-G | codex | Wrap C1 checker for G | Passes on Paley graph for n=25 |
| C4-verify-complement | codex | Wrap C1 checker for complement(G) | Passes on Paley graph complement for n=25 |
| T1-warm | codex | Paley construction for n=25 | Valid adjacency string, both checks pass |
| T2-single | codex | Modified circulant/optimization for n=50 | Valid adjacency string, both checks pass |
| T3-general | codex | Parameterized algorithm | All n<=100 in <10min each |

## Key Mathematical Risks

1. **T2 is the hard case**: 2(50)-1 = 99 is not a prime power. No known
   construction template applies directly. This may require genuine
   mathematical innovation or extensive computation.

2. **T3 generality**: A general algorithm must handle both prime-power and
   non-prime-power 2n-1. The non-prime-power case lacks a known template.

3. **B_k-freeness checking at scale**: For n=100, the graph has 398 vertices
   and ~79,000 edges. Common-neighbor counting is O(n * max_degree) per edge,
   feasible but needs efficient implementation.

## Mentor Review (claude-2, 2026-03-08)

- **QP-1 (landscape scout)**: Satisfied — Wesley results mapped, prime-power
  coverage identified, T1 approach clear.
- **QP-2 (technique landscape)**: Now addressed — T2 fallbacks ranked with
  time budgets. Cayley Z_99 first, then random algebraic, then optimization.
- **QP-8 (confidence anticorrelation)**: Flagged and mitigated — F1 framing
  updated to emphasize genuine falsification intent. SAT encoding reuse noted.
- **ArSE integration**: Added to F2-literature methodology.
- **No triggers fire.** Strategy cleared for dispatch.

## References

- Wesley, "Lower bounds for book Ramsey numbers", Discrete Mathematics (2025)
- Wigderson, "Book Ramsey numbers I", arXiv:2208.01630 (2023)
- Rousseau & Sheehan upper bound: R(B_{s}, B_{t}) <= s + 2t - 1
