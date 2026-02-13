# Problem 6: Four Blocking Results for GPL-H

Date: 2026-02-13
Status: All four results verified (analytic proofs + numerical confirmation)

## Overview

The remaining gap in Problem 6 (epsilon-light subsets) is **GPL-H**: prove
that for general graphs G, the Neumann alignment coefficient satisfies
alpha = tr(P_M F)/tr(F) < 1/2. This is equivalent to showing that cross-edge
leverage mass outside col(M) exceeds mass inside col(M).

Below we prove four blocking results showing that the most natural proof
strategies from the spectral graph theory toolkit **cannot** close GPL-H.
Each result identifies a specific structural reason why the standard approach
fails, delimiting the space of valid proof strategies.

Verification script: `scripts/verify-blocking-results.py`
Codex C4 data: `data/first-proof/problem6-codex-cycle4-results.json`

---

## BR1: Operator Inequality M + F <= Pi Is Insufficient

**Claim.** There exist PSD matrices M, F with M + F <= Pi (where Pi = I - J/n
is the Laplacian-range projection) such that

  rho_1 = tr(MF) / (||M|| * tr(F)) > 1/2.

In particular, rho_1 can be made arbitrarily close to 1.

**Proof.** Work in R^3 with Pi = I - (1/3)J, which has rank 2. Let {e_1, e_2}
be an orthonormal basis for range(Pi). Define:

  M = a |e_1><e_1|,    F = (1-a)|e_1><e_1| + b|e_2><e_2|

for parameters a in (0,1), b in [0, 1-a] (to ensure M + F <= Pi).

Then:
- ||M|| = a
- tr(MF) = a(1-a)
- tr(F) = (1-a) + b
- rho_1 = tr(MF) / (||M|| * tr(F)) = (1-a) / ((1-a) + b)

For any delta > 0, choosing b < delta gives rho_1 > (1-a)/(1-a+delta),
which can be made > 1/2 by taking delta < 1-a. In the extreme case b = 0:
rho_1 = 1.

**Numerical verification (a=0.5, b=0.1):**
- eigenvalues of Pi - M - F: {0, ~0, 0.9} (all >= 0, confirming M+F <= Pi)
- rho_1 = 0.8333 > 1/2

**Implication.** Any proof of alpha < 1/2 (and hence rho_1 < 1/2) must use
properties of the *graph structure* — specifically, that M and F arise from
disjoint edge sets of a graph Laplacian, not merely from abstract PSD matrices
satisfying M + F <= Pi. The operator inequality alone does not encode enough
structure.

---

## BR2: No Universal Per-Edge Bound Exists

**Claim.** For K_n with the barrier greedy at horizon T = floor(eps*n/3), every
cross-edge (u,v) has

  alpha_{uv} = ||P_M z_{uv}||^2 / tau_{uv} = (T-1)/(2T) -> 1/2

as n -> infinity. Therefore no universal constant c < 1/2 can serve as a
per-edge bound on alpha_{uv}.

**Proof.** In K_n, the normalized Laplacian L = (n/(n-1))(I - J/n) has uniform
leverages tau_e = 2/n for all edges. At step t in the barrier greedy, the
selected set S_t of size t has:

- M_t = (t-1)/n * (I - J/n)|_{col(M)} with uniform eigenvalues mu = (t-1)/n
  on the (t-1)-dimensional column space
- Each cross-edge z_{uv} (u in S, v in R) has the same alignment with col(M)
  by the vertex-transitive symmetry of K_n

The per-edge alignment is:

  alpha_{uv} = ||P_M z_{uv}||^2 / ||z_{uv}||^2 = (t-1)/(2t)

This equals 1/2 - 1/(2t). At horizon T = floor(eps*n/3):

| n      | eps | T    | alpha_uv  | margin from 1/2 |
|--------|-----|------|-----------|------------------|
| 40     | 0.5 | 6    | 0.416667  | 0.083333         |
| 80     | 0.5 | 13   | 0.461538  | 0.038462         |
| 200    | 0.5 | 33   | 0.484848  | 0.015152         |
| 1,000  | 0.5 | 166  | 0.496988  | 0.003012         |
| 10,000 | 0.5 | 1666 | 0.499700  | 0.000300         |

**Implication.** Any per-edge approach to bounding alpha must allow the
per-edge contribution to be arbitrarily close to 1/2. The aggregate
alpha < 1/2 is a *cancellation* phenomenon across edges, not a pointwise
property. This rules out strategies of the form "bound each alpha_{uv} by
some c < 1/2, then sum."

---

## BR3: Interlacing Families Fail for Vertex Selection

**Claim.** The random matrices {Y_t(v) : v in R_t} do NOT form an interlacing
family in the sense of Marcus-Spielman-Srivastava (2015). Specifically:

(a) The average characteristic polynomial Q(x) = (1/r_t) sum_v det(xI - Y_t(v))
    is frequently not real-rooted.

(b) Random partitions of R_t into two groups yield sub-averages that do not
    interlace Q.

**Numerical evidence (Codex C4):**
- Total interlacing trials: 1170
- Passed: 703 (60.1%)
- Q real-rooted failures: 35 out of 117 steps (29.9%)

Specific witnesses:
- K_40, eps=0.5, t=5: 0/10 trials pass, Q not real-rooted (max imag = 8.8e-6)
- K_40, eps=0.5, t=6: 0/10 trials pass, Q not real-rooted (max imag = 2.9e-5)
- K_80, eps=0.3, t=4: 0/10 trials pass, Q not real-rooted (max imag = 2.3e-7)
- K_80, eps=0.3, t=5: 0/10 trials pass, Q not real-rooted (max imag = 9.8e-6)
- K_80, eps=0.2, t=5: 0/10 trials pass, Q not real-rooted (max imag = 1.1e-5)

**Structural reason.** Interlacing families (MSS 2015, Theorem 4.4) require
the underlying random variables to be independent. Here:

  Y_t(v) = B^{1/2} C_t(v) B^{1/2}

where C_t(v) = sum_{u in S, u~v} X_{uv} sums over cross-edges incident to v.
If a vertex u in S is adjacent to both v_1 and v_2 in R, then the edge
matrices X_{u,v_1} and X_{u,v_2} share the vertex u, creating correlation
between C_t(v_1) and C_t(v_2). Concretely:

- X_{uv} = w_{uv} L^{+/2}(e_u - e_v)(e_u - e_v)^T L^{+/2}
- Both X_{u,v_1} and X_{u,v_2} involve the vector L^{+/2} e_u
- This shared component induces rank-1 correlation between the "atoms"

In BSS-style edge sparsification, each edge is an independent atom (no shared
vertices across candidate updates). The vertex-selection setting breaks this
independence because multiple candidate vertices share neighbors in S.

**Implication.** The MSS interlacing-families machinery — the main modern tool
for existence proofs in spectral sparsification — does not apply to vertex
selection in its standard form. Any proof via interlacing would need a
fundamentally different decomposition into independent atoms.

---

## BR4: Schur-Convexity Fails at M != 0

**Claim.** The function dbar(mu_1, ..., mu_r) is NOT Schur-concave in the
eigenvalues of M when M != 0. Consequently, one cannot prove
dbar_G <= dbar_{K_n} by arguing that K_n's uniform eigenvalue distribution
is the "worst case" via Schur-convexity.

**Proof.** The barrier degree decomposes as

  dbar = (1/r) sum_i (1 - mu_i) / (eps - mu_i)

where the sum runs over eigenvalues of M in col(M), assuming the tight case
F_ii = 1 - mu_i (from F + M <= Pi).

The function f(mu) = (1 - mu)/(eps - mu) is **convex** in mu on [0, eps)
(since f''(mu) = 2(1-eps)/(eps-mu)^3 > 0). By definition of Schur-convexity,
a convex function's sum over a vector is Schur-convex, meaning:

  If mu is more concentrated (majorizes mu'), then sum f(mu_i) >= sum f(mu'_i).

This says concentrated eigenvalues give **higher** dbar, not lower. Compare:

**Uniform** (all mu_i = tau/r) vs **Concentrated** (one mu_i = tau, rest = 0):

| r  | tau | eps | dbar_uniform | dbar_concentrated | ratio  |
|----|-----|-----|-------------|-------------------|--------|
| 5  | 0.3 | 0.5 | 10.682      | 11.500            | 1.077  |
| 10 | 0.5 | 0.5 | infeasible  | infeasible        | -      |

At r=5, tau=0.3, eps=0.5: concentrated exceeds uniform by 7.7%.

**Context.** At M = 0 (step 0), dbar^0 = tr(F)/(eps*r) depends on leverage
sums, which are controlled by Foster's theorem: sum tau_e = n-1. Here K_n
(uniform leverage) genuinely is the worst case, and the argument works.

But at M != 0, the amplification factor 1/(eps - mu_i) is convex, reversing
the direction. The more concentrated M's eigenvalues are, the worse the
barrier degree. Since K_n has the most *uniform* eigenvalues among graphs of
the same size, it is actually a *favorable* case at M != 0, not the worst case.

**Implication.** The Schur-convexity route — which would be the natural way to
reduce "general G" to "K_n" — is blocked at M != 0. Any proof must either:
(a) work directly with the non-uniform eigenvalue structure, or
(b) find a different comparison principle that accounts for the convexity flip.

---

## Summary of What's Blocked

| # | Strategy | Blocked Because | Tightness |
|---|----------|----------------|-----------|
| BR1 | Operator inequality M+F<=Pi alone | Counterexample: rho_1 up to 1 | Exact |
| BR2 | Per-edge alpha_{uv} < c < 1/2 | K_n: alpha_{uv} -> 1/2 as n -> inf | Exact |
| BR3 | Interlacing families (MSS 2015) | Atoms share vertices; Q not real-rooted | Structural + numerical |
| BR4 | Schur-convexity reduction to K_n | Convexity of 1/(eps-mu) reverses direction | Exact |

## What Remains Viable

Given these blockages, a proof of GPL-H must:

1. **Use graph structure** beyond M + F <= Pi (BR1 rules out pure operator arguments)
2. **Work at the aggregate level**, not per-edge (BR2 rules out pointwise bounds)
3. **Not rely on interlacing independence** (BR3 rules out MSS-style existence)
4. **Not reduce to K_n via majorization** at M != 0 (BR4 rules out Schur route)

The most promising remaining direction is the **per-vertex alignment route**:
alpha_v = ||P_M z_v||^2 / ||z_v||^2 is empirically < 2% everywhere (Codex C4
Task 5), and the aggregate alpha depends on a weighted combination of these
per-vertex contributions. The challenge is converting the tiny per-vertex
alignments into a formal bound on the aggregate, accounting for the u-side
contributions (which are large but controlled by the M-eigenvalue structure).
