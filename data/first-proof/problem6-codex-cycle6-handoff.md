# Problem 6 Cycle 6 Codex Handoff: Bridge the Leverage-Degree Conditional Proof

Date: 2026-02-13
Agent: Claude -> Codex
Type: Proof bridging + numerical verification + construction variant

## Background

Cycle 5b established a **conditional proof** of dbar0 < 1:

> If S is the prefix of size t in I_0 sorted by nondecreasing leverage
> degree ell_v, and t <= floor(eps*m/3), then
>
> dbar0 <= 2t(m-1) / (m*eps*(m-t)) <= 2(m-1) / (3(m-t)) < 1.

This uses induced Foster (sum ell_v <= 2(m-1)) and partial averages.

The **remaining bridge**: connect this to the barrier greedy's trajectory.
The barrier greedy selects vertices by min tr(Y_t(v)) (barrier contribution),
not by min ell_v (leverage degree). We need to show that the greedy's S_t
also satisfies dbar0_t < 1.

## Task 1: Numerical Comparison — Greedy vs Leverage-Degree Ordering

For each graph/eps in the existing test suite (C5 + C5b families), compare:

1. **Barrier greedy trajectory**: the standard construction (min tr(Y_t(v)))
2. **Leverage-degree prefix**: S = first t vertices sorted by ell_v
3. **Modified greedy**: among barrier-feasible v (||Y_t(v)|| < 1), pick min ell_v

For each, at every step t, compute:
- sum_{u in S_t} ell_u (total leverage degree of selected vertices)
- dbar0_t = tr(F_t) / (r_t * eps)
- tau_internal_t (S-internal leverage)
- The partial-averages bound: 2t(m-1)/m

**Key questions:**
1. Does the barrier greedy's sum ell_u stay below the partial-averages bound
   2t(m-1)/m? If yes, the conditional proof applies directly.
2. Does the modified greedy (barrier + leverage tiebreak) find T feasible
   vertices? If yes, construction (b) works.
3. How do the three trajectories compare in dbar0?

**Output**: JSON with per-step comparison data for all three orderings.

## Task 2: Order-Comparison Lemma (Bridge Option A)

Try to prove: the barrier greedy's S_t has
sum_{u in S_t} ell_u <= 2t(m-1)/m (the partial-averages bound).

**Approach 1 — Indirect via barrier feasibility:**
The barrier greedy selects v_t with tr(Y_t(v_t)) <= dbar_t.
Since tr(Y_t(v)) = tr(B_t C_t(v)) and B_t >= (1/eps)I:
- tr(Y_t(v)) >= ell_v^{S_t} / eps (where ell_v^{S_t} = cross-leverage to S_t)
- So the selected v_t has ell_{v_t}^{S_t} <= eps * dbar_t

But ell_v (full leverage degree) includes R-neighbors too. The cross-leverage
ell_v^{S_t} can be much smaller than ell_v for vertices with many R-neighbors.

**Question**: Is there a bound relating ell_v to ell_v^{S_t} + something
controlled by the greedy?

**Approach 2 — Amortized accounting:**
Track the running sum sum_{u in S_t} ell_u as t increases. When v is added:
delta = ell_v (the new vertex's full leverage degree).

We need: sum_{k=1}^t ell_{v_k} <= 2t(m-1)/m.

The average ell_v over all of I_0 is 2(m-1)/m. If the greedy selects
vertices with below-average ell_v, the bound holds. Test numerically
whether ell_{v_k} <= 2(m-1)/m at each step.

**Approach 3 — Stochastic domination:**
Can the barrier greedy's leverage-degree sequence be stochastically
dominated by the sorted sequence? If the greedy's sum is always <=
the sorted prefix's sum, the conditional proof transfers.

**Output**: Proof sketch or counterexample for each approach.

## Task 3: Modified Construction (Bridge Option B)

Implement and test the **leverage-degree barrier greedy**:

```
Sort I_0 by ell_v in nondecreasing order.
S = empty, M = 0.
For i = 1, ..., m:
    If ||Y(v_i)|| < 1 and |S| < T:
        Add v_i to S, update M.
    Else:
        Skip v_i.
Return S.
```

This tries vertices in leverage-degree order, adding only those that are
barrier-feasible. Since we process in leverage-degree order, the selected
S is a SUBSET of the first k vertices (for some k), and the Codex
conditional proof's partial-averages bound applies.

**Key questions:**
1. Does this modified greedy always find T = floor(eps*m/3) vertices?
   (It should, if the induction works — but verify numerically.)
2. What is the max ||M_S|| at the end? (Should be < eps by construction.)
3. What is max dbar0 across all steps? (Should be < 1 by conditional proof.)
4. How does it compare to the standard barrier greedy in terms of the
   final S size and quality?

**If this works numerically for all test graphs**, the proof closes:
- dbar0 < 1 at each step (conditional proof, since S is low-ell prefix subset)
- alpha < 1 (vertex-induced coordinate argument, proved C5)
- Assembly: dbar = dbar0 + (alpha*dbar0)*x/(1-x) < 1 (product bound, proved C5)
- Therefore: feasible v exists at each step
- |S| = T = eps*m/3 >= eps^2*n/9

The formal proof would need to verify that the induction closes: at step t,
dbar_t < 1 guarantees a barrier-feasible v exists among the remaining
low-ell candidates. This is the circular argument that needs unwinding.

**Output**: JSON with per-step data for the modified greedy, and assessment
of whether the construction works universally.

## Task 4: Direct Induction (Bridge Option C)

Try to prove dbar0_{t+1} < 1 given dbar0_t < 1, by tracking the change
when vertex v is added to S.

When v moves from R to S:
- tr(F_{t+1}) = tr(F_t) + ell_v^{R_{t+1}} - ell_v^{S_t}
  (new cross-edges from v to remaining R, minus old cross-edges from v to S)
- r_{t+1} = r_t - 1

So: dbar0_{t+1} = [tr(F_t) + ell_v^{R_{t+1}} - ell_v^{S_t}] / ((r_t-1)*eps)

For this to be < 1: tr(F_t) + ell_v^{R_{t+1}} - ell_v^{S_t} < (r_t-1)*eps.

Since dbar0_t < 1: tr(F_t) < r_t * eps.

So need: ell_v^{R_{t+1}} - ell_v^{S_t} < (r_t-1)*eps - tr(F_t)
        = (r_t-1)*eps - tr(F_t)
        <= (r_t-1)*eps - (something)

This is a LOCAL condition on the vertex added. The barrier greedy selects
v with small tr(Y_t(v)), which implies small ell_v^{S_t}. But ell_v^{R_{t+1}}
could be large.

**Key numerical question**: For the barrier greedy's trajectory, compute
ell_v^{R_{t+1}} - ell_v^{S_t} at each step. Is it always small enough?
What is the worst case?

**Output**: Per-step delta data, and assessment of whether the direct
induction closes.

## Task 5: Strict-Light Threshold Fix

The knife-edge dbar0 = 1 at K_10, eps=0.2 occurs because tau_e = eps
exactly. Fix by using **strict** light threshold:

- Heavy: tau_e >= eps (weak inequality)
- Light: tau_e < eps (strict)

**Verify:**
1. Turan bound still holds: |{heavy}| <= (n-1)/eps (since each heavy edge
   contributes >= eps to the sum n-1). CHECK.
2. The conditional proof works with strict inequality: all I_0 edges have
   tau_e < eps (strictly), so ell_v < deg(v)*eps, and partial averages
   give strict dbar0 < 2(m-1)/(3(m-t)) < 1.
3. Re-run the exhaustive small-n scan with strict threshold. Confirm no
   dbar0 >= 1 cases remain.
4. Check: does K_10 with eps=0.2 still produce I_0 of reasonable size?
   (With strict threshold: tau_e = 0.2 = eps makes edges heavy, so the
   heavy graph is complete, I_0 = single vertex, T = 0. This is fine —
   the theorem gives |S| >= eps^2*n/9 = 0.04*10/9 ~ 0, vacuously true.)

**Output**: Updated exhaustive scan with strict threshold, confirmation
that the knife-edge vanishes.

## Priority

Task 1 (numerical comparison) >= Task 3 (modified construction) >>
Task 2 (order-comparison lemma) >= Task 4 (direct induction) >=
Task 5 (strict threshold fix).

Task 1 gives the empirical landscape. Task 3 tests the cleanest proof
route. Tasks 2 and 4 are proof attempts. Task 5 is a boundary cleanup.

## If the Modified Greedy Works (Task 3)

This would close GPL-H and hence Problem 6. The complete proof chain:

1. Turan: I_0 >= eps*n/3, all internal edges strictly light (tau_e < eps)
2. Sort I_0 by leverage degree ell_v
3. Leverage-degree barrier greedy: process in ell_v order, add if barrier-feasible
4. At each step t:
   (a) dbar0_t < 1 (conditional proof via partial averages)
   (b) alpha_t < 1 (vertex-induced coordinate argument)
   (c) dbar_t = dbar0_t + (alpha_t*dbar0_t)*x_t/(1-x_t) < 1 (assembly + product bound)
   (d) Therefore: exists v in remaining candidates with ||Y_t(v)|| < 1
5. Greedy runs for T = floor(eps*m/3) steps: |S| >= eps^2*n/9

The induction requires: at step t, dbar_t < 1 guarantees a
barrier-feasible v among the not-yet-processed leverage-degree candidates.
This is the step that needs formal verification.

## Context Files

- Conditional proof: `data/first-proof/problem6-codex-cycle5b-verification.md`
- C5b results: `data/first-proof/problem6-codex-cycle5b-results.json`
- C5 results: `data/first-proof/problem6-codex-cycle5-results.json`
- Solution: `data/first-proof/problem6-solution.md`
- Wiring: `data/first-proof/problem6-wiring.json`
- Greedy script: `scripts/verify-p6-cycle5-codex.py` (or latest)
- Alpha script: `scripts/compute-alpha-rho.py`

## Output

- `data/first-proof/problem6-codex-cycle6-results.json`
- `data/first-proof/problem6-codex-cycle6-verification.md`
