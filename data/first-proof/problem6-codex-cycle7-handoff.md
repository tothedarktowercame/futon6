# Problem 6 Cycle 7 Codex Handoff: Close the Barrier Maintenance Invariant

Date: 2026-02-13
Agent: Claude -> Codex
Type: Formal proof closure + numerical deep-dive

## Background

Cycle 6 identified Bridge B (modified leverage-order barrier greedy) as
the winning proof strategy. The construction sorts I_0 by I_0-internal
leverage degree ell_v^{I_0} and processes in nondecreasing order, adding
barrier-feasible vertices.

**What's now proved:**
- Induced Foster: sum_{e in E(I_0)} tau_e <= m-1 (Rayleigh monotonicity)
- Conditional dbar0 < 1: for the pure prefix, dbar0 <= 2(m-1)/(m(3-eps)) < 1
- Alpha < 1: for vertex-induced partitions (coordinate argument)
- Product bound: alpha*dbar0 <= ((t-1)-tau)/(r*eps) <= 1/(3-eps)
- Assembly: dbar = dbar0 + alpha*dbar0*x/(1-x)
- Bridge C (direct induction on dbar0): proved for prefix with large m
- No-Skip: 0 skips in 116 runs (modified greedy = pure prefix)
- Strict threshold: heavy := tau_e >= eps fixes knife-edge

**The single remaining gap: Barrier Maintenance Invariant (BMI)**

> dbar_t := (1/r_t) tr(B_t F_t) < 1 at each step t of Construction B.

This is STRONGER than dbar0_t < 1 (which IS proved). The amplification
by B_t = (eps*I - M_t)^{-1} can make dbar_t >> dbar0_t when ||M_t||
approaches eps. The crude assembly bound gives:

    dbar <= 2/(3-eps) + (1/(3-eps)) * x/(1-x)

which closes only for x < (1-eps)/(2-eps) — i.e., eps < 0.7 for the
combined bound. Yet empirically dbar < 1 ALWAYS holds (worst: 5/6 at K_n).

**The complementarity:** When x is large (M nearly saturated), F has
little mass in col(M) directions, so alpha is small. The product
alpha*x/(1-x) stays bounded. But the current bounds don't capture this
joint structure because they bound dbar0 and alpha*dbar0 separately,
each tight at K_n but at different configurations.

## Task 1: Eigenvalue-Level BMI Computation

The full barrier degree has an exact eigenvalue decomposition:

    dbar_t = (1/r_t) sum_j (pi_j - lambda_j) / (eps - lambda_j)

where:
- lambda_j are eigenvalues of M_t on im(L) (the barrier eigenvalues)
- pi_j = u_j^T Pi_{I_0} u_j (projections of the I_0-edge-sum onto M_t's eigenvectors)
- Pi_{I_0} = sum_{e in E(I_0)} X_e (the I_0 internal Laplacian in L^{+/2} coords)

The identity F_t + M_t + L_R = Pi_{I_0} gives pi_j = lambda_j + f_j + l_j
where f_j = u_j^T F_t u_j and l_j = u_j^T L_R u_j.

**Compute for each step of the modified greedy on the C6 test suite:**
1. The eigenvalues lambda_j of M_t
2. The projections pi_j
3. The per-eigenspace contributions (pi_j - lambda_j)/(eps - lambda_j)
4. The sum = r_t * dbar_t

**Key questions:**
1. Which eigenspaces dominate the sum? (Low lambda → small contribution;
   high lambda → potentially large contribution but pi_j - lambda_j
   should be small there.)
2. Is pi_j <= 1 for all j? (True since Pi_{I_0} <= Pi = I on im(L).)
3. What is the tightest (pi_j - lambda_j)/(eps - lambda_j) ratio?
4. Does the identity pi_j - lambda_j = f_j + l_j give a usable bound?
   (f_j + l_j = (Pi_{I_0} - M_t)_{jj} and (Pi_{I_0} - M_t) is PSD
   since Pi_{I_0} >= M_t.)

**Output:** JSON with per-step eigenvalue data for the modified greedy.

## Task 2: Direct BMI Proof Attempt via pi_j Bound

**Goal:** Prove sum_j (pi_j - lambda_j)/(eps - lambda_j) < r for the
pure prefix at each step t <= T = eps*m/3.

**Approach 1 — Uniform pi bound:**
Since Pi_{I_0} <= I on im(L): pi_j <= 1 for all j. Then:

    (pi_j - lambda_j)/(eps - lambda_j) <= (1 - lambda_j)/(eps - lambda_j)

Sum: sum_j (1-lambda_j)/(eps-lambda_j) = (n-1) + (1-eps) * sum 1/(eps-lambda_j).

This sum is (n-1) + (1-eps) * tr(B_t). For this < r = m-t:
we need (1-eps) * tr(B_t) < m-t-(n-1). Since m < n in general,
m-t-(n-1) < 0, so this approach FAILS (the pi_j = 1 bound is too loose
when m << n).

**Approach 2 — Use Pi_{I_0} structure:**
Pi_{I_0} = sum_{e in E(I_0)} X_e is NOT the full identity. It has
tr(Pi_{I_0}) = sum tau_e <= m-1 (induced Foster). And Pi_{I_0} <= I.

Key: the EFFECTIVE dimension of Pi_{I_0} is at most m-1 (at most m-1
nonzero eigenvalues, since the edges of a graph on m vertices span at
most m-1 dimensions). So pi_j = 0 for at least (n-1)-(m-1) = n-m
eigenspaces. On those eigenspaces: contribution = -lambda_j/(eps-lambda_j).

For the remaining m-1 eigenspaces: pi_j <= ||Pi_{I_0}|| <= 1.

**Test numerically:** What fraction of pi_j are zero (or near-zero)?
What is the effective rank of Pi_{I_0} compared to n-1?

**Approach 3 — Cauchy-Schwarz on the sum:**
Write (pi_j - lambda_j)/(eps-lambda_j) = (pi_j-lambda_j) * b_j where
b_j = 1/(eps-lambda_j). Then:

    sum = sum (pi_j-lambda_j) * b_j <= sqrt(sum (pi_j-lambda_j)^2) * sqrt(sum b_j^2)

by Cauchy-Schwarz. Bound each factor using Pi_{I_0} - M_t >= 0 and
the trace of B_t^2.

**Output:** Proof or counterexample for each approach. Per-step data
for the eigenspace structure.

## Task 3: K_n Extremality for the Combined Expression

**Goal:** Prove that K_n maximizes dbar_t among all graphs with the same
m and eps, at the same step t.

For K_n: all lambda_j are equal (= t/n on the t-1 dimensional subspace
of col(M), and 0 elsewhere). All pi_j are equal within each eigenspace.
The exact formula gives dbar = (t-1)/(n*eps-t) + (t+1)/(n*eps) -> 5/6.

For non-K_n graphs: eigenvalues are non-uniform. BR4 showed that the
individual amplification function f(lambda) = (1-lambda)/(eps-lambda) is
CONVEX, so non-uniform eigenvalues give HIGHER values of sum f(lambda_j).

BUT dbar doesn't use f(lambda_j) = (1-lambda_j)/(eps-lambda_j). It uses
(pi_j - lambda_j)/(eps - lambda_j), where pi_j DEPENDS on the eigenspace
structure. For non-K_n graphs, high-lambda eigenspaces may have SMALL
pi_j (because Pi_{I_0} has different eigenvectors from M_t).

**Key question:** Is the expression sum (pi_j - lambda_j)/(eps - lambda_j)
a concave or convex function of the eigenvalue distribution, JOINTLY with
the pi_j adjustments?

**Test numerically:**
1. For each graph in the C6 suite, compute dbar_t at the horizon.
2. Compare to K_m at the same m, eps, t.
3. Is dbar_G <= dbar_{K_m} at all tested cases?
4. What is the max ratio dbar_G / dbar_{K_m}?

If K_m is extremal (max dbar among all graphs with same parameters),
then dbar <= 5/6 < 1 universally at the horizon, and BMI is proved.

**Output:** Comparison data and assessment of K_n extremality.

## Task 4: BSS Potential for Vertex-Block Updates

**Goal:** Adapt the BSS barrier potential to handle vertex-block updates
and derive a bound on dbar_t.

Define Phi_t = tr(B_t) = sum 1/(eps - lambda_j(M_t)).

When vertex v is added (moving from R to S): M_{t+1} = M_t + C_t(v)
where C_t(v) = sum_{e: v-S_t} X_e.

**Step 1:** Compute the potential change Phi_{t+1} - Phi_t for the
modified greedy trajectory. Track at every step.

**Step 2:** In the BSS framework for rank-1 updates X_e:
Phi increases by tau_e * z_e^T B_{t+1} z_e / ||z_e||^2.
For a vertex-block update C = sum X_e (multiple edges):
Phi_{t+1} - Phi_t = tr(B_{t+1} C_t(v)).

**Step 3:** Can we bound sum_{t=0}^{T-1} tr(B_{t+1} C_t(v_t))?
This telescopes: Phi_T - Phi_0 = Phi_T - (n-1)/eps.

**Step 4:** Relate Phi_T to dbar_T. Since dbar_T = tr(B_T F_T)/r_T
and F_T = Pi_{I_0} - M_T - L_R:
dbar_T = [tr(B_T Pi_{I_0}) - tr(B_T M_T) - tr(B_T L_R)] / r_T.

With tr(B_T) = Phi_T and tr(B_T M_T) known from the eigendecomposition,
can we bound dbar_T in terms of Phi_T?

**Output:** Potential trajectory data and assessment of whether the BSS
potential approach can bound dbar_t.

## Task 5: Stress Test — Adversarial Graph Families

The current test suite covers 29 graph families. Design and test
adversarial cases that might maximize dbar:

1. **Near-bipartite with uneven leverage:** Bipartite K_{a,b} with
   a << b, where the small side has high leverage.
2. **Expander + pendant:** Take a d-regular expander and attach pendant
   edges. The pendants have tau_e close to 1.
3. **Weighted graphs:** Use weights to concentrate leverage on specific
   edges or vertices within I_0.
4. **Barbell variants:** Two cliques connected by few edges. The bridge
   edges have high leverage.
5. **Random graphs with planted dense subgraph:** ER(n, p1) with a
   planted clique of size sqrt(n) at higher density p2.

For each: compute the full eigenvalue data from Task 1 and dbar at every
step. Report the worst dbar and margin to 1.

**Output:** Extended test suite results with adversarial cases.

## Priority

Task 1 (eigenvalue computation) >> Task 3 (K_n extremality) >=
Task 2 (direct proof) >= Task 5 (stress test) >= Task 4 (BSS potential).

Task 1 gives the structural data needed by all other tasks. Task 3 is
the cleanest closure path (if K_n is extremal, BMI follows from the
K_n exact formula). Task 2 is the direct algebraic route. Task 5
validates the bounds. Task 4 is speculative.

## If K_n Extremality Holds (Task 3)

This would close Problem 6. The complete proof:

1. Turan: I_0 >= eps*n/3, all edges strictly light (tau_e < eps).
2. Sort I_0 by ell_v^{I_0} nondecreasing.
3. Modified barrier greedy: add v iff barrier-feasible, in sorted order.
4. At each step t <= T = eps*m/3:
   (a) dbar0_t < 1 (proved: induced Foster + partial averages)
   (b) alpha_t < 1 (proved: coordinate argument)
   (c) dbar_t < 1 (by K_n extremality: dbar_G <= dbar_{K_m} = 5/6)
   (d) Exists v with ||Y_t(v)|| < 1 (proved: pigeonhole + PSD trace)
5. No-Skip: v_{t+1} is feasible (follows from dbar_t < 1 + leverage
   ordering ensuring the low-leverage vertex is among the feasible).
6. |S| = T >= eps^2*n/9.

Step 4(c) requires the K_n extremality proof. The key structural
reason it should hold: K_n has UNIFORM eigenvalues in M_t, and the
identity pi_j = eigenvalue of Pi_{I_0} projected onto u_j gives the
tightest pi-lambda alignment for uniform structures.

## If K_n Extremality Fails

Then the proof needs a different route. The most promising alternatives:
- Direct BMI proof via eigenspace analysis (Task 2)
- BSS potential bound (Task 4)
- Probabilistic argument (random set has dbar < 1 w.h.p., then
  derandomize via the method of conditional expectations)

## Context Files

- Bridge B formalization: `data/first-proof/problem6-bridge-b-formalization.md`
- Solution: `data/first-proof/problem6-solution.md` (Sections 5k-5l, 6)
- Wiring: `data/first-proof/problem6-wiring.json` (v8-bridge-b)
- C6 results: `data/first-proof/problem6-codex-cycle6-results.json`
- C6 verification: `data/first-proof/problem6-codex-cycle6-verification.md`
- C5b results: `data/first-proof/problem6-codex-cycle5b-results.json`
- Greedy scripts: `scripts/verify-p6-cycle5-codex.py`, `scripts/verify-p6-cycle6-codex.py`
- Alpha/rho script: `scripts/compute-alpha-rho.py`

## Output

- `data/first-proof/problem6-codex-cycle7-results.json`
- `data/first-proof/problem6-codex-cycle7-verification.md`
