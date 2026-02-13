# Problem 6 Cycle 5 Codex Handoff: Relaxation Scan

Date: 2026-02-13
Agent: Claude -> Codex
Type: Systematic relaxation analysis + recombination

## Motivation

Cycle 4 produced four blocking results showing that the standard spectral
toolkit cannot close GPL-H directly. But a key observation emerges from
looking at the *threshold*:

**The constant 1/2 in rho_1 < 1/2 is NOT fundamental.** The assembly formula:

  dbar <= dbar^0 * (1 - x(1-c)) / (1 - x)

where dbar^0 <= 2/3 (Foster) and x = ||M||/eps, requires:

  c < (1 + x) / (2x)    (for dbar < 1)

At the K_n horizon x = 1/3: c < 2. At x = 0.375 (Barbell worst case): c < 1.83.
For any x < 1 (guaranteed by barrier): c < (1+x)/(2x) > 1.

**Therefore: ANY c < 1 suffices for the assembly.** We don't need rho_1 < 1/2;
we need rho_1 < 1. This is a factor of 2 weaker, transforming a tight inequality
(8% margin) into a qualitatively different statement (100%+ margin).

The strategy: systematically relax each constraint in GPL-H, prove the relaxed
pieces, and find how they combine (the "pushout").

## Task 1: Verify the Assembly with c < 1

**Prove rigorously** that the assembly dbar < 1 closes with c < 1 (not just
c < 1/2). Specifically:

1. Verify that dbar^0_t <= 2/3 at ALL steps t (not just the horizon).
   - At step t, dbar^0_t = tr(F_t) / (eps * r_t).
   - The current proof uses Foster's theorem: sum_e tau_e = n - 1.
   - Does this give dbar^0_t <= 2/3 for all t, or only at the horizon?
   - If not at all t, what is the tightest bound on dbar^0_t?

2. Compute c_needed = (1-x_t)*(1/dbar^0_t - 1)/x_t at every step t for the
   full test suite. Confirm c_needed >= 1 everywhere.

3. If dbar^0_t can exceed 2/3 at intermediate steps, compute the joint
   constraint surface: what (c, x, dbar^0) triples give dbar < 1?

4. **Key question**: Is there a uniform constant c_0 < 1 such that
   rho_1 < c_0 and dbar^0 * (1-x+c_0*x)/(1-x) < 1 at all steps for all
   graphs? What is the optimal c_0?

**Output**: JSON with c_needed at every step, and a proof or counterexample
for dbar^0 <= 2/3 at all steps.

## Task 2: Prove rho_1 < 1 (the Relaxed Threshold)

The operator bound gives:
  rho_1 <= sum mu_i(1-mu_i) / (mu_max * tr(F))

For rho_1 < 1, we need:
  tr(F) > sum mu_i(1-mu_i) / mu_max

This is HALF the requirement for rho_1 < 1/2 (which needs tr(F) > 2 * ...).
Equivalently: alpha = tr(P_M F)/tr(F) < 1.

**Alpha < 1 means: F has nonzero mass outside col(M).** For graph-derived
M, F, this should follow from:
- Cross-edges (u,v) with v in R contribute z_v components outside col(M)
- v is not in S, so z_v is not captured by S-internal edges
- Specifically: ||P_{col(M)^perp} z_v||^2 > 0 for v in R

Attempt to prove alpha < 1:
1. For each cross-edge (u,v) with v in R, decompose:
   ||P_M(z_u - z_v)||^2 = ||P_M z_u||^2 + ||P_M z_v||^2 - 2<P_M z_u, P_M z_v>
2. The perp part: ||P_perp(z_u - z_v)||^2 >= (||P_perp z_v|| - ||P_perp z_u||)^2
3. Since z_v has mass outside col(M) (v not adjacent to M's edges)...

**Try multiple routes:**
(a) Direct: show ||P_{col(M)^perp} z_v||^2 > 0 for each v in R, then aggregate
(b) Counting: col(M) has dim t-1, range(Pi) has dim n-1, so the "fraction
    of directions" in col(M) is (t-1)/(n-1) << 1. Random z_v should miss col(M).
(c) Foster + dimension: the total leverage in col(M)^perp is at least
    (n-1) - tr(M)/... (some Foster-type argument for the complement)

**Output**: Proof sketch for rho_1 < 1 (or alpha < 1), or precise failure point.

## Task 3: Graph Family Scan

For each graph family, run the barrier greedy and compute at each step:
- x_t = ||M_t||/eps
- dbar^0_t
- c_needed_t
- alpha_t, rho_1_t
- The "relaxed margin" = c_needed_t - rho_1_t (how much room we have)

**Graph families to test** (use larger n than Cycle 4):
- K_n for n in {100, 200, 500}
- d-regular random graphs for d in {3, 10, n/2}, n = 100
- Erdos-Renyi G(100, p) for p in {0.1, 0.3, 0.5}
- Trees: random spanning tree of K_100, path P_100, star S_100
- Bipartite: complete bipartite K_{50,50}, random bipartite
- Expanders: Ramanujan graph if available, or algebraic construction

**Epsilons**: {0.1, 0.2, 0.3, 0.5}

**Key questions**:
1. Which family has the tightest relaxed margin?
2. Is there a family where c_needed < 1? (This would block the c < 1 route.)
3. Does x_t stay bounded away from 1 universally?
4. For which families is dbar^0_t <= 2/3 at all steps?

**Output**: JSON with per-step data for each family/eps combination.

## Task 4: Low-Rank Scan

At each rank of M (rank 1, 2, ..., 20), can we prove alpha < 1?

For rank(M) = k, col(M) is k-dimensional in the (n-1)-dimensional range(Pi).
The fraction k/(n-1) is small when k << n.

**Approach**: At step t (rank(M) = t-1), compute:
- alpha_t as a function of rank only
- Does alpha_t < 1 follow from a pure dimensional argument?
- At what rank does alpha first approach 1? (If ever)

Also test: for rank(M) = 1, can we get an exact formula for alpha?
At t = 2 (first internal edge), M = X_e for one edge e. col(M) = span(z_e).
Then P_M = z_e z_e^T / ||z_e||^2 and:
  alpha = sum_{cross (u,v)} |<z_{uv}, z_e>|^2 / (||z_e||^2 * sum ||z_{uv}||^2)

This has a clean effective-resistance interpretation.

**Output**: Per-rank alpha data, and a proof at rank 1 if achievable.

## Task 5: Edge-Partition Relaxation

Does alpha < 1 (or < 1/2) hold for arbitrary edge partitions, not just
vertex-induced ones?

**Setup**: Take a graph G. Partition edges into "internal" (I) and "cross" (C).
Set M = sum_{e in I} X_e, F = sum_{e in C} X_e. Does M + F <= Pi always hold?
Is alpha = tr(P_M F)/tr(F) < 1?

**Test**: For each graph in the suite, generate random edge partitions
(each edge independently assigned to I with probability p, for p in {0.1, ..., 0.9}).
Compute alpha for each partition.

**Key question**: If alpha < 1 fails for arbitrary edge partitions but holds for
vertex-induced ones, the vertex-induced structure is essential. If alpha < 1 holds
for all edge partitions, the problem is purely about PSD geometry, not vertex structure.

(Note: BR1 shows alpha can be 1 for abstract PSD matrices. But graph-derived
edge-partition matrices have additional structure from the Laplacian.)

**Output**: JSON with alpha data for random edge partitions, and assessment of
whether the vertex-induced constraint is essential.

## Task 6: Random Selection Probe

Instead of the barrier greedy, pick T vertices uniformly at random from I_0.
Compute E[alpha] and concentration.

**Key question**: Does E[alpha] < 1/2 (or < 1) for random selection?
If yes, this gives an existence proof via the probabilistic method.
If no, the greedy's selection criterion is essential.

**Output**: Alpha statistics for random vertex subsets.

## Priority

Task 1 (assembly verification) >> Task 2 (prove rho_1 < 1) >> Task 3 (families)
>= Task 4 (rank) >= Task 5 (edge partition) >= Task 6 (random).

Task 1 is critical: if c < 1 genuinely suffices at all steps, the problem
transforms from "tight inequality" to "nonzero mass outside a subspace."
Task 2 is the new proof attempt under the relaxed threshold.

## File locations

- Blocking results: `data/first-proof/problem6-blocking-results.md`
- Alpha data: `data/first-proof/alpha-rho-analysis.json`
- C4 results: `data/first-proof/problem6-codex-cycle4-results.json`
- C4 verification: `data/first-proof/problem6-codex-cycle4-verification.md`
- Greedy script: `scripts/verify-p6-cycle3-codex.py`
- Alpha script: `scripts/compute-alpha-rho.py`
- Blocking script: `scripts/verify-blocking-results.py`
- Output: `data/first-proof/problem6-codex-cycle5-results.json`
- Verification: `data/first-proof/problem6-codex-cycle5-verification.md`

## Graph suite

Expanded from C4: K_n (n=100,200,500), d-regular random (d=3,10,n/2; n=100),
ER G(100,p) (p=0.1,0.3,0.5), Trees (random spanning tree, path, star; n=100),
Bipartite (K_{50,50}, random), Expander (algebraic if available).
Epsilons: {0.1, 0.2, 0.3, 0.5}.

## The Houseboat Principle

The blocking results say "a proof using only standard tools is impossible."
But the 1/2 threshold was never load-bearing — c < 1 suffices. The relaxed
problem (rho_1 < 1 instead of rho_1 < 1/2) is qualitatively different:

- **rho_1 < 1/2**: tight, needs exact eigenvalue accounting (blocked by BR1-4)
- **rho_1 < 1**: qualitative, needs "F has mass outside col(M)" (should follow
  from graph structure)

The houseboat: combine the relaxed threshold (c < 1) with the assembly bound
(dbar^0 <= 2/3, x <= 1/3) to close GPL-H without ever proving rho_1 < 1/2.
