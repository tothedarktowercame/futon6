# GPL-H Counterexample: Complete Bipartite Graphs

**Date:** 2026-02-12
**Author:** Claude (advisor)
**Severity:** Blocks current proof path

---

## 1. The counterexample

Complete bipartite graphs K_{t,r} with small t and large r violate GPL-H
as formulated (H1-H4 with D0 = 12, c_step = 1/3).

### Concrete instance: K_{2,20}

- n = 22 vertices: left = {a1, a2}, right = {x1,...,x20}
- All 40 edges from left to right, each with τ = 21/40 = 0.525
- ε = 0.525 (just above τ, so H1 holds)
- I0 = all 22 vertices (no heavy edges, H2 passes with D0_bound = 22.86)
- T = floor(εn/3) = floor(0.525 × 22/3) = 3

**Barrier greedy execution:**

- Step 0: Select a1 (or a2), score = 0. Phase 1.
- Step 1: Select a2 (zero score — a2 has no edge to a1 in I0). Phase 1.
- Step 2: Phase 2 entry! Every x_i has 2 neighbors in S_2 = {a1, a2}.
  - S_2 independent (no edge between a1, a2). M_t = 0.
  - Score(x_i) = ||Y_2(x_i)|| = ||(1/ε)(X_{a1,xi} + X_{a2,xi})|| = 1.905
  - ALL vertices have identical score 1.905 (by symmetry)
  - **min score = 1.905 > 1. GPL-H VIOLATED.**

### Score formula for K_{t,r}

In K_{t,r} with unit weights: τ = (t+r-1)/(tr) for all edges.

At Phase 2 entry (step t, S_t = left part, M_t = 0):

    score(x) = (1/ε) · ||Σ_{j=1}^t X_{aj,x}||

The edge vectors X_{a_j,x} are rank-1 PSD with ||X|| = τ. They are nearly
parallel (since they all involve vertex x). By the Gram structure:

    ||Σ X|| = τ + (t-1)α where α = z_j^T z_k for j ≠ k

For K_{t,r}: α → τ as r → ∞ (the edge vectors become parallel). So:

    score → tτ/ε → t · (t+r-1)/(tr·ε) → t/ε for large r

For t = 2, ε ≈ 1/2: score → 4. For t = 5, ε ≈ 1/5: score → 25.

**The score diverges as r → ∞ for fixed t.**

### Verified violations (97 total across K_{t,r} families)

| Graph | ε | Score | H1 | H2 (D0=12) |
|-------|-------|-------|------|------|
| K_{2,10} | 0.825 | 1.21 | ✓ | ✓ |
| K_{2,20} | 0.525 | 1.90 | ✓ | ✓ |
| K_{2,100} | 0.505 | 1.98 | ✓ | ✓ |
| K_{3,30} | 0.391 | 2.56 | ✓ | ✓ |
| K_{4,80} | 0.259 | 3.86 | ✓ | ✓ |
| K_{5,80} | 0.212 | 4.71 | ✓ | ✓ |

---

## 2. Why previous testing missed it

The test suite includes: K_n (complete), C_n (cycle), Barbell, Dumbbell,
DisjCliq, ER, RandReg. None of these are unbalanced bipartite. Complete
bipartite K_{t,r} with t ≪ r was not in the suite.

The K_n (complete graph) is actually K_{n,0} — the balanced extreme where
all vertices have equal leverage degree ≈ 2. The counterexample requires
**asymmetric leverage concentration**: a few hub vertices with high leverage
degree connected to many low-leverage leaves.

---

## 3. Root cause analysis

### The leverage concentration problem

In K_{t,r}: t hub vertices have leverage degree ℓ_hub = (t+r-1)/t ≈ r/t.
The r leaf vertices have ℓ_leaf = (t+r-1)/r ≈ 1.

When ε ≈ τ: H2 allows ℓ ≤ D0/ε ≈ D0/τ ≈ D0 · tr/(t+r). For D0 = 12
and t = 2, r = 20: threshold = 12 × 40/22 ≈ 21.8. But ℓ_hub = 10.5 < 21.8.
So hubs pass H2.

The barrier greedy selects all t hubs (they have zero score in Phase 1,
since they're mutually non-adjacent). Then every leaf has t neighbors in S_t,
giving score ≈ tτ/ε ≈ t, which is >> 1 for t ≥ 2.

### Why the existing graphs don't have this problem

- **K_k (complete):** All vertices equivalent, ℓ ≈ 2. Score = 2/(kε) ≤ 1.
- **Barbell:** Two dense cliques. Each clique is K_k-like internally.
- **ER, RandReg:** No persistent hub structure. Degree distribution concentrates.
- **DisjCliq:** Each clique is K_k. Bridge edges have high τ (heavy in G_H).

The common property: **leverage degree is approximately uniform** (all ≈ 2).
The counterexample has **bimodal leverage**: hubs at r/t, leaves at ≈ 1.

---

## 4. The fix: tighter regularization

### Option A: Tighten H2 to D0 = 2

Set the leverage degree bound to ℓ_v ≤ 2/ε (i.e., D0 = 2). This excludes
hub vertices since ℓ_hub = r/t > 2/ε when r/t > 2/ε.

**Consequences:**
- K_{t,r} hubs excluded. I_0 = leaf vertices only. No internal edges → Case 1.
- K_k: ℓ = 2(k-1)/k < 2 ≤ 2/ε. All vertices pass. ✓
- General: at most ε·n vertices excluded (by Markov on Σℓ = 2(n-1)).

**Problem:** For small I (close to Turán bound), removing εn vertices can
make I_0 < εn/3. The greedy horizon exceeds I_0.

### Option B: Re-check α_{I_0} after regularization

After removing high-leverage vertices: if I_0 has no internal edges (or
α_{I_0} ≤ ε), it's Case 1 or Case 2a. Handle separately.

**Problem:** Doesn't fix cases where I_0 still has hubs (e.g., K_{2,20} at
ε near τ, where hubs pass even the tighter bound).

### Option C: Leverage-aware greedy

Modify the barrier greedy to only select vertices with ℓ_v ≤ 2 + δ. This
prevents hub concentration.

**Advantage:** No change to I_0 construction. The greedy avoids hubs naturally.
**Feasibility:** There are always ≥ n · δ/(2+δ) low-leverage vertices
(by Markov). For δ = 1: ≥ n/3 vertices. Sufficient for εn/3 steps.

**Key bound:** If each selected vertex has ℓ ≤ 2 + δ:
    dbar ≤ (2+δ)t / (ε · r_t) = (2+δ) / (ε · (m_0/t - 1))
At t = εm_0/3: dbar ≤ (2+δ) / (3 - ε) < 1 for δ < 1 - ε.

### Option D: Direct construction (bypass greedy)

For K_{t,r}: the right part is an independent set of size r ≥ εn/3 with
all edges light. Directly use the right part as the ε-light independent set.
No greedy needed.

**This is the correct solution for the PROBLEM** — the barrier greedy is a
proof technique, not the final answer. If the greedy fails but the problem
is trivially solvable, we need a different proof technique.

---

## 5. Impact on Problem 6

### What's salvageable

1. **Phase 1 closure (proved):** Unaffected. Phase 1 handles 64% of
   non-bipartite instances.

2. **K_k formula (proved):** Unaffected. K_k has uniform leverage.

3. **Ratio certificate (proved):** The algebraic identity min ≤ dbar/gbar
   is unconditional.

4. **Double-counting (proved):** The bound dbar ≤ 2Lt/(m_0 ε r_t) is
   unconditional when the proportionality assumption holds.

### What needs modification

1. **H2 regularization:** The D0 = 12 bound is too generous. Need either
   D0 = 2 (tight) or a different regularization strategy.

2. **GPL-H theorem statement:** Needs an additional hypothesis excluding
   the bipartite hub structure, OR needs to be proved under tighter H2.

3. **The greedy strategy:** May need to be leverage-aware (Option C) or
   replaced by a direct construction (Option D).

### The path forward

The cleanest fix is likely **Option C + tighter H2**:

1. Tighten H2 to ℓ_v ≤ C/ε for C = 2 + δ (small δ).
2. Modify the greedy to only select from low-leverage vertices.
3. Prove dbar < 1 from the leverage bound: dbar ≤ C/(3-ε) < 1 for C < 3.
4. Handle Case 2b with the modified greedy, Case 1/2a unchanged.

This makes GPL-H provable by construction (the leverage bound on selected
vertices directly controls dbar), at the cost of a more constrained greedy.

---

## 6. Files

- `scripts/verify-p6-bipartite-stress.py` — Full bipartite stress test (97 violations)
- `scripts/verify-p6-bipartite-h2-check.py` — H2 check for bipartite graphs
- `scripts/verify-p6-g1-single-neighbor.py` — Single-neighbor analysis (shows Phase 2 entry proof)
- `scripts/verify-p6-g1-heavy-structure.py` — Heavy-edge structure diagnostic
