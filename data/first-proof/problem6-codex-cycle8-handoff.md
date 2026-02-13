# Problem 6 Cycle 8 Codex Handoff: Close the Vertex-Level Feasibility Gap

Date: 2026-02-13
Agent: Claude -> Codex
Type: Numerical deep-dive + direct bound attempts

## Background

Cycle 7 **falsified BMI** (the barrier maintenance invariant dbar_t < 1).
12 base-suite steps have dbar >= 1, worst 1.739 (Reg_100_d10, eps=0.5,
t=16). K_n extremality also falsified (max ratio 2.28). The pigeonhole
route (dbar < 1 → exists feasible v) is dead.

**But the construction still works.** In all 116 base + 32 adversarial
runs, the modified greedy reaches the horizon. At every step, there
exists at least one feasible vertex in R_t with ||Y_t(v)|| < 1 — even
when the average barrier degree exceeds 1.

**C6/C7 threshold discrepancy (critical):** C6 used `heavy_ge=False`
(heavy if tau > eps), giving 0 skips. C7 used `heavy_ge=True` (heavy
if tau >= eps), giving **5 runs with skips** (max 3 skips per run,
infeasible normY up to 6.68). The strict threshold (`heavy_ge=True`)
is the correct one — it fixes the K_10 knife-edge. With the strict
threshold, the No-Skip conjecture is **weakened**: the trajectory is
NOT always the pure prefix.

**The actual invariant needed:**

> At each step t of the modified greedy (strict threshold), there
> **exists** v in R_t with ||Y_t(v)|| < 1.

This is weaker than No-Skip (which requires the NEXT vertex to be
feasible) but still can't be proved via pigeonhole (since dbar > 1).

**What's proved:**
- Induced Foster: sum tau_e over E(I_0) <= m-1 (Rayleigh monotonicity)
- Conditional dbar0 < 1 for the pure prefix: dbar0 < 2/(3-eps) < 1
- Alpha < 1 for vertex-induced partitions (not in critical path)
- K_n case: proved exactly (dbar = 5/6 < 1 at horizon)
- 0 failures to complete in 148 runs (construction always works)

**What's falsified (C7):**
- BMI (dbar < 1): FALSE, worst 1.739
- K_n extremality: FALSE, max ratio 2.28
- No-Skip (strict threshold): FALSE, 5/116 runs have skips

**Key C7 data:**
- Max candidate_normY (selected vertex): 0.954 (always < 1)
- Steps with candidate_normY > 0.9: 5/731 (0.7%)
- Eigenvalue structure at worst case: bimodal (87 near-zero + 12 high),
  single lambda_max = 0.493 vs eps = 0.500 drives the blow-up
- Correct formula: dbar = (1/r) sum_j f_j/(eps-lambda_j), NOT (pi_j-lambda_j)
- BSS resolvent identity exact: Phi_{t+1}-Phi_t = tr(B_{t+1} C_t(v) B_t)

## Task 1: Full R-Scan (HIGHEST PRIORITY)

**Goal:** At every step of the modified greedy, compute ||Y_t(v)|| for
ALL vertices v in R_t (not just the selected one). This gives the
complete distribution of barrier contributions across R.

For each step, record:
1. For every v in R_t: normY(v), traceY(v), deg_S(v) = |{u in S_t : u~v}|,
   ell_v^{I_0}, ell_v^{S_t} (cross-leverage to S)
2. The sorted vector of normY values across R
3. The minimum normY across R (= the "best available" vertex)
4. The number of feasible vertices (normY < 1) in R
5. The correlation between normY(v) and ell_v^{I_0}
6. The correlation between normY(v) and deg_S(v)

**Key questions:**
1. Is min_{v in R} normY(v) always < 1? (This IS the gap.)
2. How many feasible vertices are in R at each step? (If many, the
   argument is robust. If few, it's fragile.)
3. Does normY(v) correlate with ell_v^{I_0}? If so, leverage ordering
   helps. If not, the ordering is incidental.
4. Does normY(v) correlate better with deg_S(v) (cross-degree to S)?
   This would suggest a cross-degree argument.
5. For regular graphs (all ell equal), what determines which vertices
   have low normY?

**Run on:** Full C7 base suite + adversarial. Focus especially on the
5 runs with skips and the 12 steps with dbar >= 1.

**Output:** Per-step R-scan data (the full normY vector for each step
would be large; summarize with quantiles and correlations).

## Task 2: Threshold Reconciliation

**Goal:** Run the modified greedy with BOTH thresholds on the same
suite and compare results.

For each graph and eps:
1. Run with `heavy_ge=False` (C6 threshold: heavy if tau > eps)
2. Run with `heavy_ge=True` (C7 threshold: heavy if tau >= eps)
3. Compare: |I_0|, horizon T, skip count, worst normY, worst dbar

**Key questions:**
1. How many graphs have different I_0 between the two thresholds?
2. For graphs with different I_0: does the strict threshold always
   give smaller I_0? By how much?
3. Do skips occur ONLY with the strict threshold, or also with the
   non-strict one on larger graphs?
4. Is there a "safe" threshold (e.g., heavy if tau > eps - delta for
   small delta) that eliminates both knife-edges and skips?

**Output:** Comparison table across all runs.

## Task 3: Eigenspace Overlap Analysis

**Goal:** For each step, decompose C_t(v) into components along B_t's
eigenspaces and relate to feasibility.

Define the "high-amplification eigenspace" as the span of eigenvectors
of M_t with lambda_j > eps/2 (i.e., where 1/(eps-lambda_j) > 2/eps).

For each v in R_t, compute:
1. P_high C_t(v) P_high (projection of C_t(v) onto high-amplification space)
2. tr(P_high C_t(v) P_high) / tr(C_t(v)) = "high overlap fraction"
3. The contribution of the high-amplification eigenspace to normY(v)

**Key questions:**
1. Do feasible vertices (normY < 1) have small high-overlap fraction?
2. Do infeasible vertices (normY > 1, i.e., the skipped ones) have
   large high-overlap fraction?
3. Is the high-overlap fraction correlated with ell_v^{I_0} or deg_S(v)?
4. Can we bound the high-overlap fraction for low-leverage vertices?

**Computation note:** This requires diagonalizing M_t (already done in
C7 for eigenvalue data) and projecting C_t(v) = sum z_e z_e^T onto the
eigenvectors. For v's edges e = {v, u} with u in S_t:

    high_overlap(v) = sum_e sum_{j: lambda_j > eps/2} (z_e^T u_j)^2

This can be computed efficiently as ||P_high Z_v||_F^2 where Z_v stacks
the z_e vectors for v's cross-edges.

**Run on:** The 5 skip runs and the 12 dbar >= 1 steps. Include a few
"easy" steps (low dbar) for comparison.

**Output:** Per-vertex overlap data for selected steps.

## Task 4: Cross-Degree Bound Attempt

**Goal:** Test whether normY(v) can be bounded by a function of
deg_S(v) and the barrier state.

**Bound idea 1 — rank bound:**
C_t(v) = sum_{e: v~S_t} z_e z_e^T has rank <= deg_S(v).
Therefore Y_t(v) = B_t^{1/2} C_t(v) B_t^{1/2} has rank <= deg_S(v).
For a rank-k PSD matrix: ||Y|| <= tr(Y)/1 (trivial) but also
||Y|| = max eigenvalue, which can be bounded by other means for low-rank.

**Bound idea 2 — sum of rank-1 terms:**
||Y_t(v)|| = ||sum_e z_e^T B_t z_e * w_e w_e^T + cross-terms||
where w_e = B_t^{1/2} z_e / ||B_t^{1/2} z_e||.
If the w_e are nearly orthogonal: ||Y_t(v)|| ≈ max_e (z_e^T B_t z_e).
If they're nearly parallel: ||Y_t(v)|| ≈ sum_e (z_e^T B_t z_e).

**Bound idea 3 — max single edge:**
||Y_t(v)|| <= deg_S(v) * max_{e: v~S_t} (z_e^T B_t z_e).
This is crude but might close if deg_S(v) is small enough.

**Computation:**
For each step and each v in R_t:
1. Compute deg_S(v)
2. Compute max_{e: v~S_t} (z_e^T B_t z_e)
3. Compute the product deg_S(v) * max_e(z_e^T B_t z_e)
4. Compare to normY(v)

**Key question:** Is there a function f(deg_S, barrier_state) < 1
that holds for all feasible vertices? If f = deg_S * max_e_barrier_cost,
what is the tightest value of this product?

**Output:** Correlation data and tightest bounds.

## Task 5: Partial Averages with Skips

**Goal:** Verify that the dbar0 < 1 proof (Lemma 3) still holds when
the trajectory has skips.

With strict threshold, S_t may not be the pure prefix — it can skip
some vertices. If S_t consists of t vertices drawn from positions
{1, ..., t+s} in the sorted order (s skips), then:

    sum_{v in S_t} ell_v^{I_0} <= sum_{i=1}^{t+s} ell_{v_i}^{I_0}
                                <= (t+s) * 2(m-1) / m

The dbar0 bound becomes: dbar0 <= 2(t+s)(m-1) / (m * r_t * eps).
With s small (max 3 observed), this is a minor correction.

**Computation:**
For each run with skips:
1. Record the actual positions in the sorted order of selected vertices
2. Compute the partial averages bound WITH skips
3. Verify dbar0 is still < 1 with the corrected bound
4. What is the maximum s (skips before horizon) across all runs?

**Key question:** Does the dbar0 < 1 proof survive with the observed
number of skips? If not, what is the maximum s that still gives dbar0 < 1?

**Output:** Per-run skip-adjusted dbar0 bounds.

## Task 6: Alternative Existence Arguments

**Goal:** Explore proof strategies for "exists feasible v in R_t" that
don't use pigeonhole on dbar.

**Approach A — Probabilistic:**
For a random vertex v ~ Uniform(R_t), compute Pr[normY(v) < 1].
If this probability is > 0, existence follows. Note: Pr[normY < 1] > 0
is COMPATIBLE with dbar > 1 (the average trace can exceed 1 while most
individual spectral norms are < 1, since ||Y|| <= tr(Y) is loose when
Y has spread eigenvalues).

**Approach B — Min over R via Foster:**
sum_{v in R_t} tr(Y_t(v)) = r_t * dbar_t. But for a single vertex:
tr(Y_t(v)) = sum_{e: v~S_t} z_e^T B_t z_e.
The minimum over R of this quantity is:
min_v tr(Y_t(v)) <= (1/r_t) sum_v tr(Y_t(v)) = dbar_t.
This is just pigeonhole (fails when dbar > 1). BUT: ||Y_t(v)|| can be
much less than tr(Y_t(v)) when Y_t(v) has rank > 1 and spread eigenvalues.
So even if min_v tr(Y_t(v)) > 1, it's possible that min_v ||Y_t(v)|| < 1.

**Approach C — Isolation lemma:**
If at each step, there exists v in R_t with deg_S(v) = 0 (no edges to S),
then Y_t(v) = 0 and ||Y_t(v)|| = 0. This is trivially feasible.
Check: at each step, is there always a vertex in R with 0 cross-edges?
If not, at what step does every R-vertex first have at least one edge to S?

**Approach D — Second moment on normY:**
Compute E[||Y_t(v)||^2] over R. If E[||Y||^2] < E[||Y||]^2 (impossible
for a scalar, but the dispersion gives information), then the distribution
of ||Y|| is concentrated. Combined with the fact that some vertex has
||Y|| < 1 empirically, concentration could give a formal bound.

**Computation:**
For each step, compute:
1. Fraction of R with normY < 1
2. Fraction of R with deg_S = 0
3. Quantiles of normY across R (10th, 25th, 50th, 75th, 90th percentiles)
4. E[normY], E[normY^2], Var[normY]
5. E[traceY], E[traceY^2], ratio E[normY]/E[traceY]

**Output:** Per-step distribution summaries.

## Priority

Task 1 (R-scan) >> Task 3 (eigenspace overlap) >= Task 6 (existence
arguments) > Task 4 (cross-degree bound) >= Task 2 (threshold) >=
Task 5 (partial averages with skips).

Task 1 is the foundation — it gives the COMPLETE picture of which
vertices are feasible and why. Task 3 targets the mechanism (eigenspace
separation). Task 6 explores formal closure routes. Tasks 2, 4, 5 are
supporting.

## Context Files

- Bridge B formalization: `data/first-proof/problem6-bridge-b-formalization.md`
- Solution: `data/first-proof/problem6-solution.md` (Sections 5l-5m)
- Wiring: `data/first-proof/problem6-wiring.json` (v9-bmi-falsified)
- C7 results: `data/first-proof/problem6-codex-cycle7-results.json`
- C7 verification: `data/first-proof/problem6-codex-cycle7-verification.md`
- C7 script: `scripts/verify-p6-cycle7-codex.py`
- C6 script: `scripts/verify-p6-cycle6-codex.py`
- C5 script: `scripts/verify-p6-cycle5-codex.py`

## Output

- `data/first-proof/problem6-codex-cycle8-results.json`
- `data/first-proof/problem6-codex-cycle8-verification.md`
