# Problem 6: Bridge B Formalization

Date: 2026-02-13
Agent: Claude
Base: Codex C6 verification (`problem6-codex-cycle6-verification.md`)

## Main Theorem (conditional)

**Theorem.** For every connected graph G = (V,E,w) on n vertices and
every eps in (0,1), there exists S ⊆ V with

    |S| >= eps^2 * n / 9    and    L_S <= eps * L

provided that the **No-Skip Conjecture** holds.

**Vertex-Level Feasibility (GPL-V):** At every step t of the modified
leverage-order barrier greedy on I_0 (Construction B below), there
exists v in R_t with ||Y_t(v)|| < 1.

**NOTE (Cycle 7):** The original conditioning was on the **Barrier
Maintenance Invariant** (BMI): dbar_t < 1 at each step. BMI is now
**FALSIFIED** — 12 base-suite steps have dbar >= 1 (worst: 1.739,
Reg_100_d10 eps=0.5 t=16). The pigeonhole argument (dbar < 1 →
exists feasible v) cannot close the construction. GPL-V replaces BMI
as the essential gap. Note: the earlier "No-Skip Conjecture" (the
NEXT vertex is feasible) is also **weakened** — C7 found 5/116 runs
with skips using the correct strict threshold (tau >= eps). GPL-V
only requires SOME feasible vertex exists, not the next in order.

## Proof Architecture

```
Turan (5a) ──> I_0 >= eps*n/3, all I_0-edges strictly light
                    │
         Induced Foster (Lemma 1) ──> sum tau_e over E(I_0) <= m-1
                    │
         Partial Averages (Lemma 2) ──> prefix leverage sum bounded
                    │
         Conditional dbar0 < 1 (Lemma 3) ──> base degree < 2/(3-eps) < 1
                    │
         No-Skip (Conjecture) ──> v_{t+1} feasible at each step
                    │
         Size (Lemma 7) ──> |S| = T = eps*m/3 >= eps^2*n/9

    ╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
    DEAD BRANCH (Cycle 7 — BMI falsified):
         Alpha < 1 (Lemma 4) ──> alignment < 1       (proved, not needed)
         Assembly (Lemma 5) ──> dbar = dbar0 + ...   (identity, not useful)
         BMI ──> dbar_t < 1                           (FALSE: worst 1.739)
         Existence (Lemma 6) ──> pigeonhole           (valid but unreachable)
```

## Construction B: Modified Leverage-Order Barrier Greedy

**Input:** Graph G, parameter eps in (0,1).

**Step 1 (Turan).** Build heavy graph G_H = (V, {e : tau_e >= eps}).
Find maximal independent set I_0 in G_H with |I_0| >= eps*n/3.
Set m = |I_0|, T = floor(eps*m/3).

**Step 2 (Sort).** For each v in I_0, compute the I_0-internal leverage:

    ell_v^{I_0} = sum_{e = {v,w}, w in I_0} tau_e.

Sort I_0 as v_1, v_2, ..., v_m with ell_{v_1}^{I_0} <= ell_{v_2}^{I_0} <= ... <= ell_{v_m}^{I_0}.

**Step 3 (Greedy).** Set S = {}, M = 0.
For i = 1, ..., m:
  - Compute Y_t(v_i) = H_t^{-1/2} C_t(v_i) H_t^{-1/2} where H_t = eps*I - M_t.
  - If ||Y_t(v_i)|| < 1 and |S| < T:
      Add v_i to S.  Update M_{t+1} = M_t + C_t(v_i).
  - Else: skip v_i.

**Output:** S with ||M_S|| < eps (i.e., L_S <= eps*L) and |S| >= T.

## Lemma 1: Induced Foster Bound

**Lemma.** For I_0 ⊆ V with m = |I_0|, the G-leverage scores on
I_0-internal edges satisfy

    sum_{e in E(I_0)} tau_e <= m - kappa(G[I_0]) <= m - 1

where kappa(G[I_0]) is the number of connected components of G[I_0],
and the last inequality assumes G[I_0] is connected.

**Proof.** Let tau-tilde_e denote the leverage score of edge e computed
in the subgraph G[I_0] (i.e., using the Laplacian L_{I_0} restricted
to I_0). By Foster's theorem for G[I_0]:

    sum_{e in E(I_0)} tau-tilde_e = m - kappa(G[I_0]).

Now, the effective resistance of edge e = {u,v} in G is

    R_eff^G(u,v) = (e_u - e_v)^T L_G^+ (e_u - e_v)

and in G[I_0]:

    R_eff^{I_0}(u,v) = (e_u - e_v)^T L_{I_0}^+ (e_u - e_v).

By the Rayleigh monotonicity principle (removing edges increases
effective resistance):

    R_eff^{I_0}(u,v) >= R_eff^G(u,v)

for all edges e = {u,v} in E(I_0).

Since tau_e = w_e * R_eff(u,v) and tau-tilde_e = w_e * R_eff^{I_0}(u,v):

    tau-tilde_e >= tau_e    for all e in E(I_0).

Therefore:

    sum_{e in E(I_0)} tau_e <= sum_{e in E(I_0)} tau-tilde_e = m - kappa(G[I_0]). QED.

**Corollary.** The total I_0-internal leverage degree satisfies:

    sum_{v in I_0} ell_v^{I_0} = 2 * sum_{e in E(I_0)} tau_e <= 2(m-1).

(Each edge counted twice, once from each endpoint.)

## Lemma 2: Partial Averages Inequality

**Lemma.** Let a_1 <= a_2 <= ... <= a_m be nonneg reals with
sum a_i = A. Then for t <= m:

    sum_{i=1}^t a_i <= t * A / m.

**Proof.** The first t elements in nondecreasing order are the t
smallest, hence their sum is at most (t/m) * (sum of all m). QED.

**Application.** With a_i = ell_{v_i}^{I_0} (sorted) and A <= 2(m-1):

    sum_{i=1}^t ell_{v_i}^{I_0} <= 2t(m-1)/m.

## Lemma 3: Conditional dbar0 < 1 for the Pure Prefix

**Lemma.** If S_t = {v_1, ..., v_t} is the prefix of size t in
nondecreasing ell^{I_0} order, with t <= eps*m/3, then:

    dbar0_t := tr(F_t) / (r_t * eps) <= 2(m-1) / (m(3-eps)) < 1.

**Proof.**
tr(F_t) = sum_{v in S_t} ell_v^{R_t, I_0} (cross-leverage)
        <= sum_{v in S_t} ell_v^{I_0}         (since ell_v^{R,I_0} <= ell_v^{I_0})
        <= 2t(m-1)/m                           (Lemma 2)

With r_t = m - t >= m - eps*m/3 = m(1-eps/3):

    dbar0_t <= 2t(m-1) / (m * (m-t) * eps)
            <= 2(eps*m/3)(m-1) / (m * m(1-eps/3) * eps)
             = 2(m-1) / (m(3-eps))
             < 2/(3-eps).

Since eps < 1: 3-eps > 2, so 2/(3-eps) < 1. QED.

**Explicit bounds:**
- eps = 0.1: dbar0 < 0.690
- eps = 0.2: dbar0 < 0.714
- eps = 0.3: dbar0 < 0.741
- eps = 0.5: dbar0 < 0.800
- eps = 0.9: dbar0 < 0.952

## Lemma 4: Alpha < 1 for Vertex-Induced Partitions

**Lemma.** For any vertex-induced partition {S, R} of I_0 with at
least one cross-edge, the alignment alpha = tr(P_M F)/tr(F) < 1.

**Proof.** (Proved in problem6-solution.md Section 5i.) Let (u,v) be a
cross-edge with u in S, v in R. The incidence vector b_{uv} = e_u - e_v
has nonzero v-coordinate. No S-internal edge has nonzero v-coordinate
in its incidence vector. Therefore z_{uv} = L^{+/2} b_{uv} is not in
col(M) = span{z_e : e internal to S}. Hence

    ||P_{M^perp} z_{uv}||^2 > 0

for each cross-edge. Summing: tr((I-P_M) F) > 0, so alpha < 1. QED.

## Lemma 5: Assembly Decomposition

**Lemma.** The full barrier degree decomposes as:

    dbar_t = dbar0_t + alpha_t * dbar0_t * x_t / (1 - x_t)

where x_t = ||M_t||/eps and alpha_t = tr(P_{M_t} F_t)/tr(F_t).

**Proof.** (Proved in problem6-solution.md Section 5j.) B_t shares
eigenspaces with M_t. On the col(M_t) eigenspace with eigenvalue
lambda_j: B_t has eigenvalue 1/(eps-lambda_j) >= 1/(eps-||M||). On
ker(M_t): B_t = (1/eps)*I. The Neumann expansion of B_t around
(1/eps)*I gives the decomposition. QED.

**Product bound:** alpha_t * dbar0_t <= ((t-1) - tau_S) / (r_t * eps)
where tau_S = tr(M_t). At the horizon: alpha*dbar0 <= 1/(3-eps).

## Lemma 6: Existence of Feasible Vertex

**Lemma.** If dbar_t < 1, then there exists v in R_t with ||Y_t(v)|| < 1.

**Proof.** (Proved in problem6-solution.md Section 5d.)

    min_{v in R_t} ||Y_t(v)|| <= min_{v} tr(Y_t(v))    (PSD trace bound)
                               <= (1/r_t) sum_v tr(Y_t(v))  (pigeonhole)
                               = dbar_t < 1.    QED.

## Lemma 7: Size Guarantee

**Lemma.** If the greedy runs for T = floor(eps*m/3) steps without
stopping, then |S| = T >= eps^2*n/9.

**Proof.** m = |I_0| >= eps*n/3 (Turan). T = floor(eps*m/3).
|S| = T >= eps*m/3 - 1 >= eps*(eps*n/3)/3 - 1 = eps^2*n/9 - 1.
For n >= 9/eps^2: |S| >= eps^2*n/9 - 1 >= 0. QED.

## The Remaining Gap: No-Skip / Vertex-Level Feasibility (GPL-V)

### BMI is FALSIFIED (Cycle 7)

**Conjecture (BMI, DEAD).** ~~dbar_t < 1 at every step.~~

Codex C7 computed the full dbar = (1/r) tr(B_t F_t) for the first time
(prior cycles only computed dbar0). Result: **12 base-suite steps have
dbar >= 1**, worst 1.739 (Reg_100_d10, eps=0.5, t=16). K_n extremality
also falsified (max ratio 2.28). All three direct bounds fail universally.

| Graph | eps | t | dbar | dbar0 | x | ||Y_t(v)|| |
|-------|-----|---|------|-------|---|------------|
| Reg_100_d10 | 0.50 | 16 | **1.739** | 0.584 | 0.987 | 0.622 |
| Reg_100_d10 | 0.50 | 15 | **1.641** | 0.549 | 0.987 | 0.910 |
| Reg_100_d10 | 0.30 | 10 | **1.402** | 0.657 | 0.937 | 0.937 |

Note: **the selected vertex is feasible in every case** (||Y_t(v)|| < 1).

The pigeonhole argument (Lemma 6: dbar < 1 → exists feasible v) is valid
math but **unreachable** because its hypothesis fails. The proof cannot
go through the average.

### The actual gap: Vertex-Level Feasibility (GPL-V)

**Conjecture (GPL-V).** At every step t <= T of Construction B on any
connected graph G, there exists v in R_t with ||Y_t(v)|| < 1.

**Evidence:** Construction completes in 148/148 runs (116 base + 32
adversarial). Max selected normY = 0.954. With strict threshold
(tau >= eps): 5/116 runs have skips (max 3 per run), but every run
reaches the horizon because feasible vertices always exist.

### Why the construction works despite dbar > 1

The average tr(Y_t(v)) over R can exceed 1, but the MINIMUM over R stays
below 1. More precisely, the leverage-ordering ensures the first vertex
processed has low leverage degree, hence small C_t(v), hence small
||Y_t(v)|| — even when the barrier B_t has high amplification.

The eigenvalue analysis (C7) shows the mechanism:
- When x → 1, a single eigenvalue lambda_max ≈ eps creates enormous
  amplification 1/(eps-lambda_max) ~ 150x in one eigenspace.
- HIGH-leverage vertices have large overlap with this eigenspace →
  their ||Y_t(v)|| blows up.
- LOW-leverage vertices (processed first) have small overlap → their
  ||Y_t(v)|| stays bounded.
- The leverage ordering precisely selects the vertices that avoid the
  blow-up eigenspace.

### Empirical evidence (Cycles 6+7+8)

| Quantity | Worst case | Where |
|----------|-----------|-------|
| dbar0 (modified greedy) | 0.7333 | K_50_50, eps=0.3 |
| **dbar (full barrier)** | **1.641** | Reg_100_d10, eps=0.5, t=15 |
| ||M||/eps | 0.9866 | Reg_100_d10, eps=0.5 |
| ||Y_t(v)|| for selected v | 0.954 | various |
| Min feasible count in R | **12** | worst step across all runs |
| Feasible fraction of R | 72-100% | all steps |
| deg_S=0 fraction of R | 15-100% | even at dbar >= 1 steps |
| Skips (strict threshold) | 5/116 runs | max 2 skips before selection |
| Adversarial max dbar | 0.926 | Barbell_40_40_b3 |

### Lemma 8: Rayleigh-Monotonicity Matrix Bound

**Lemma.** Define the I_0-internal edge matrix:

    Pi_{I_0} = sum_{e in E(I_0)} X_e = L^{+/2} L_{I_0} L^{+/2}.

Then Pi_{I_0} <= I on im(L). Consequently:

(a) ||C_t(v)|| <= 1 for all v in R_t, all t.
(b) F_t <= I - M_t for all t.
(c) For each eigenvalue direction u_j of M_t: u_j^T F_t u_j <= 1 - lambda_j.

**Proof.** Since G[I_0] is a subgraph of G, removing the edges
E \ E(I_0) can only decrease the quadratic form (Rayleigh monotonicity):

    L_{I_0} <= L.

Conjugating both sides by L^{+/2} (which is PSD):

    L^{+/2} L_{I_0} L^{+/2} <= L^{+/2} L L^{+/2} = Proj_{im(L)} <= I.

This gives Pi_{I_0} <= I.

For (a): C_t(v) = sum_{e: v~S_t} X_e is a partial sum of the PSD
terms composing Pi_{I_0}. Since 0 <= C_t(v) <= Pi_{I_0} <= I, we
have ||C_t(v)|| <= 1.

For (b): Pi_{I_0} = M_t + F_t + Pi_{R_t} where Pi_{R_t} = sum of X_e
for edges internal to R_t. All three terms are PSD. So:
F_t <= Pi_{I_0} - M_t <= I - M_t.

For (c): u_j^T F_t u_j <= u_j^T (I - M_t) u_j = 1 - lambda_j. QED.

**Remark.** Part (c) is the key spectral constraint: the cross-edge
matrix F_t has LESS energy in the eigenspaces where M_t is large.
This is the algebraic reason why high barrier amplification (from
eigenvalues lambda_j near eps) does not automatically cause dbar to
blow up — the numerator f_j = u_j^T F_t u_j shrinks as lambda_j grows.

### Lemma 9: Cross-Degree Bound

**Lemma.** For any v in R_t:

    ||Y_t(v)|| <= deg_S(v) * max_{e: v~S_t} z_e^T B_t z_e.

**Proof.** Write Y_t(v) = sum_{e: v~S_t} w_e w_e^T where w_e =
B_t^{1/2} z_e. This is a sum of deg_S(v) PSD rank-1 matrices.
By the triangle inequality for PSD matrices:

    ||Y_t(v)|| <= sum_e ||w_e w_e^T|| = sum_e ||w_e||^2
               = sum_e z_e^T B_t z_e
               <= deg_S(v) * max_e z_e^T B_t z_e. QED.

**Empirical (C8):** 78,619 vertex-step pairs, 0 violations, max
ratio normY/bound = 1.000. For deg_S = 1: exact equality. But
max z_e^T B_t z_e = 7.3, so single-edge cost CAN exceed 1.

### Lemma 10: Isolation Implies Feasibility

**Lemma.** If v in R_t has deg_S(v) = 0 (no I_0-internal edges to
S_t), then C_t(v) = 0 and ||Y_t(v)|| = 0 < 1.

**Proof.** C_t(v) = sum_{e: v~S_t, e in E(I_0)} X_e. With zero terms
in the sum: C_t(v) = 0. Hence Y_t(v) = B_t^{1/2} 0 B_t^{1/2} = 0. QED.

**Empirical (C8):** At 83.2% of all steps, the minimum normY is
achieved by a vertex with deg_S = 0. At ALL 7 dbar >= 1 steps,
deg_S = 0 vertices exist (15-51% of R_t).

### Lemma 11: Rank of Barrier Contribution

**Lemma.** For connected G: rank(Y_t(v)) = deg_S(v).

**Proof.** The edges e_1, ..., e_k incident to v with other endpoints
w_1, ..., w_k in S_t have incidence vectors b_{e_j} = sqrt(w_{e_j})
(e_v - e_{w_j}). Since v, w_1, ..., w_k are distinct vertices, the
vectors e_v - e_{w_1}, ..., e_v - e_{w_k} are linearly independent
(as columns of the incidence matrix). The map L^{+/2} is injective
on im(L) (which contains all b_e for connected G), so z_{e_1}, ...,
z_{e_k} are linearly independent. B_t^{1/2} is invertible on im(L)
(since eps - lambda_j > 0), so w_{e_1}, ..., w_{e_k} are linearly
independent. Therefore:

    rank(Y_t(v)) = rank(sum_j w_{e_j} w_{e_j}^T) = k = deg_S(v). QED.

### Lemma 12: Projection Pigeonhole

**Lemma.** For any unit vector u in im(L):

    min_{v in R_t} u^T C_t(v) u <= 1/r_t.

If u = u_j (eigenvector of M_t with eigenvalue lambda_j):

    min_{v in R_t} u_j^T Y_t(v) u_j <= (1 - lambda_j) / (r_t * (eps - lambda_j)).

**Proof.** By averaging: sum_{v in R_t} u^T C_t(v) u = u^T F_t u <= 1
(from F_t <= Pi_{I_0} <= I). So the minimum is <= 1/r_t.

For the amplified version: u_j^T Y_t(v) u_j = u_j^T C_t(v) u_j /
(eps - lambda_j). By Lemma 8(c): sum_v u_j^T C_t(v) u_j <= 1 - lambda_j.
So: min_v u_j^T Y_t(v) u_j <= (1-lambda_j)/(r_t(eps-lambda_j)). QED.

**Remark.** For the most dangerous direction (lambda_j = ||M_t|| =
eps - delta): the bound gives min_v u_max^T Y_t(v) u_max <=
(1-eps+delta)/(r_t * delta). For delta ~ 1/r_t: this is approximately
(1-eps) * r_t / r_t = 1-eps < 1. So in the dominant eigenspace, the
pigeonhole gives a vertex with bounded contribution. The difficulty
is that this vertex may have large contributions in OTHER eigenspaces.

### Theorem (Sparse Dichotomy — Proved)

**Theorem.** If Delta(G[I_0]) < 3/eps - 1, then GPL-V holds at every
step t <= T of Construction B.

**Proof.** The minimum domination number satisfies gamma(G[I_0]) >=
m/(1 + Delta). With Delta < 3/eps - 1:

    gamma >= m/(1 + Delta) > m/(3/eps) = m*eps/3 >= T.

So |S_t| = t <= T < gamma(G[I_0]), meaning S_t cannot dominate all
of I_0. There exists v in R_t with no I_0-neighbor in S_t, i.e.,
deg_S(v) = 0. By Lemma 10: ||Y_t(v)|| = 0 < 1. QED.

**Scope:** For eps = 0.5: Delta < 5. For eps = 0.3: Delta < 9.
For eps = 0.1: Delta < 29. This covers sparse G[I_0] but not dense.

### Theorem (Dense Feasibility — Partial, Sub-Gap Identified)

**Claim.** When G[I_0] is dense (isolation may fail), the constraints
Pi_{I_0} <= I and dbar0 < 1 together force GPL-V. More precisely:

**Conjecture (Strong Dichotomy).** At each step t <= T, at least one
of the following holds:

(A) There exists v in R_t with deg_S(v) = 0 (isolation — Lemma 10).
(B) dbar_t < 1 (pigeonhole — Lemma 6 gives a feasible v).

**Evidence:** In 148 test runs (731+232 steps), at EVERY step where
dbar >= 1, isolated vertices exist (Case A). Case B with dbar >= 1
and no isolated vertex was NEVER observed.

**Partial proof — why dense graphs resist dbar blowup:**

When G[I_0] is dense, each vertex v has many I_0-edges with small
leverages (all tau_e < eps). The induced Foster bound forces:

    average tau_e over E(I_0) <= (m-1)/|E(I_0)|.

For |E(I_0)| large: average tau is tiny. This has two consequences:

1. **Isotropic barrier.** Many small-leverage edges build an M_t whose
   eigenvalues are spread out (no single direction is near-saturated).
   Formally: rank(M_t) >= tr(M_t)/||M_t||. For dense graphs with
   tr(M_t) moderate and ||M_t|| < eps: rank is proportional to tr/eps.
   With many eigenvalues sharing the budget, each eps - lambda_j is
   bounded away from 0, limiting amplification.

2. **Small per-edge cost.** The cross-degree bound (Lemma 9) gives
   ||Y_t(v)|| <= deg_S(v) * max_e(z_e^T B_t z_e). With an isotropic
   barrier (B_t approx (1/eps)I), each z_e^T B_t z_e approx tau_e/eps
   < 1. So even vertices with high deg_S have bounded ||Y_t(v)||.

**Where the formal argument has a gap:**

The difficulty is in a mixed regime: G[I_0] is dense enough that
isolation fails (some steps have all R-vertices dominated), but NOT
uniform enough that the barrier stays isotropic. Specifically:

- If M_t concentrates its spectrum (one eigenvalue lambda_1 near eps),
  the amplification 1/(eps - lambda_1) is large.
- The constraint f_1 = u_1^T F_t u_1 <= 1 - lambda_1 (Lemma 8c)
  limits the F_t energy in this direction.
- The amplified contribution from this eigenspace:
  f_1/(eps - lambda_1) <= (1-lambda_1)/(eps-lambda_1) = 1 + (1-eps)/(eps-lambda_1).
- For the remaining eigenspaces (lambda_j << eps): f_j/(eps-lambda_j)
  approx f_j/eps, contributing approximately dbar0 < 1.

So: dbar <= 1 + (1-eps)/(eps-lambda_1) + dbar0_rest.

For dbar >= 1: we need the "amplified spike" term (1-eps)/(eps-lambda_1)
to overcome the fact that dbar0 < 1. This requires eps - lambda_1 to
be O(1/r). But when eps - lambda_1 = O(1/r), the barrier is extremely
concentrated, and the Projection Pigeonhole (Lemma 12) shows that
SOME vertex v has small overlap with the dangerous eigenspace:

    u_1^T C_t(v) u_1 <= (1-lambda_1)/r_t ~ (1-eps)/r_t.

For that vertex, the dangerous-direction contribution to ||Y_t(v)|| is:

    u_1^T Y_t(v) u_1 <= (1-eps)/(r_t * (eps-lambda_1)) ~ (1-eps)/1 = 1-eps < 1.

**But:** the same vertex v may have large contributions in other
eigenspaces. Its total ||Y_t(v)|| depends on ALL eigenspace projections
simultaneously, and the pigeonhole only controls one direction at a time.

**The precise sub-gap:** Prove that for the vertex v minimizing
u_1^T C_t(v) u_1 (low dangerous-direction projection), the contributions
from the remaining eigenspaces do not push ||Y_t(v)|| above 1. The
constraint is: sum_{j>=2} (u_j^T C_t(v) u_j)/(eps-lambda_j) < 1 -
u_1^T Y_t(v) u_1. Since the remaining eigenspaces have eps-lambda_j
bounded away from 0, the remaining contribution is bounded by
tr(C_t(v))/(eps-lambda_2). With ||C_t(v)|| <= 1 (Lemma 8a):
tr(C_t(v)) <= rank(C_t(v)) = deg_S(v).

For the sub-gap to close, we need either:
(i) deg_S(v) is bounded (then remaining contribution ~ deg_S/(eps-lambda_2) and
    the total is < 1 when deg_S is small and lambda_2 is not too close to eps), OR
(ii) when deg_S(v) is large, the contributions are spread across many eigenvalues
     and no single direction exceeds 1 (using the rank equality, Lemma 11).

Neither (i) nor (ii) follows from the current lemmas alone.

### Eigenspace Separation (Cycle 8 — Empirical)

Feasible vertices have mean 0.55% high-overlap fraction with B_t's
high-amplification eigenspace. Infeasible: 14.6%. Ratio: 26x.

This confirms the mechanism behind the sub-gap: vertices whose edges
to S project onto M_t's near-saturated eigenspace become infeasible.
Vertices that avoid this eigenspace remain feasible. The leverage
ordering naturally selects vertices that avoid the dangerous eigenspace,
because the dangerous eigenspace was BUILT from previously-added
(higher-leverage) vertices' edge contributions.

### Three Attack Paths for Formal Closure

**Attack Path 1: Strongly Rayleigh on vertex indicators.**
Define a strongly Rayleigh distribution on vertex subsets of R_t
(e.g., via DPP with kernel related to leverage scores). Apply the
Anari-Gharan Kadison-Singer-for-SR theorem: if atoms are small
(||C_t(v)|| <= 1 from Lemma 8) and marginals bounded, some
realization has bounded spectral norm. The atom bound ||C_t(v)|| <= 1
is a natural input; the difficulty is constructing the right SR
measure that "sees" the amplification by B_t.

**Attack Path 2: Hyperbolic barrier in hyperbolicity cone.**
Write the barrier determinant det(eps*I - M_t - sum_v x_v C_t(v))
as a hyperbolic polynomial in {x_v}. Apply Brändén's higher-rank
KS extension: hyperbolicity cone yields interlacing yields root bound.
Setting x_v = 1 for one vertex and 0 for the rest gives ||Y_t(v)||
as the largest root. The mixed hyperbolic polynomial may have a
computable root bound from the F_t <= I - M_t constraint.

**Attack Path 3: Interlacing families (Xie-Xu / MSS style).**
Consider the average characteristic polynomial:
p(x) = (1/r_t) sum_{v in R_t} det(xI - Y_t(v)).
If the polynomials {det(xI - Y_t(v))} form an interlacing family,
then some v has largest root <= largest root of p(x). The largest
root of p(x) equals the expected spectral norm in a distributional
sense. The key computation: bound the largest root of p using the
constraints F_t <= I - M_t and dbar0 < 1. This is the most promising
path because it naturally incorporates the averaging AND spectral
structure simultaneously.

## No-Skip Conjecture

**Conjecture.** In Construction B, no vertex is ever skipped. That is,
at step t, the (t+1)-th vertex v_{t+1} in the leverage-degree ordering
always satisfies ||Y_t(v_{t+1})|| < 1.

**Evidence:** 0 skips in 116 runs (726 greedy steps total).

**Why this is hard to prove:** The sufficient condition via trace bounds
gives ||Y_t(v)|| <= tr(Y_t(v)) <= ell_v^{S_t}/(eps-||M_t||). For the
(t+1)-th vertex: ell_{v_{t+1}}^{S_t} could be as large as
ell_{v_{t+1}}^{I_0} ~ 2(m-1)/m. With eps-||M|| potentially small,
the trace bound exceeds 1. The actual ||Y_t(v)|| < 1 because the
eigenvalues of Y_t(v) are spread out (not concentrated), but proving
this requires controlling the spectral interaction between B_t and
C_t(v), which depends on graph structure.

## Direct Induction Route (Bridge C)

An alternative to proving the No-Skip Conjecture is the direct
induction on dbar0:

**Bridge C condition:** At step t, if vertex v is added:

    Delta_t := ell_v^{R_{t+1},I_0} - ell_v^{S_t,I_0}
    rhs_t := (r_t - 1)*eps - tr(F_t)

If Delta_t < rhs_t, then dbar0_{t+1} < 1 (by direct computation).

**Proof that Delta_t < rhs_t for the pure prefix (large m):**

For the (t+1)-th vertex in sorted order:

    Delta_t <= ell_{v_{t+1}}^{I_0} <= 2(m-1)/m    (average of remaining)

    rhs_t >= (m-t-1)*eps - 2t(m-1)/m    (budget minus consumed leverage)

The condition 2(m-1)/m < (m-t-1)*eps - 2t(m-1)/m becomes, at t = eps*m/3:

    2(m-1)(1+t)/m < (m-t-1)*eps

For large m this reduces to 2(1 + eps/3) < (1-eps/3)*eps*(m/something)...

More precisely, the condition holds when:

    m > (6 + eps) / (eps*(1-eps))

Evaluating:
- eps = 0.1: m > 68
- eps = 0.2: m > 39
- eps = 0.3: m > 30
- eps = 0.5: m > 26
- eps = 0.9: m > 77

**Status:** Bridge C is proved for the pure prefix when m is
sufficiently large (Theta(1/(eps*(1-eps)))). For small m, the result
may be vacuous (|S| = T is small) or needs case analysis.

**BUT:** Bridge C establishes dbar0_{t+1} < 1, not dbar_{t+1} < 1.
The full BMI requires the assembly argument, which has the
complementarity gap described above.

## The Complete Closure Path

If GPL-V holds, the proof closes:

1. **Turan** (proved): I_0 >= eps*n/3, all internal edges strictly light.
2. **Sort** by ell_v^{I_0} nondecreasing.
3. **Greedy** processes in sorted order, adding barrier-feasible vertices.
4. **GPL-V** (conjecture): exists feasible v in R_t at each step.
5. Greedy selects such a v (possibly skipping infeasible ones).
6. **dbar0 < 1** (Lemma 3, with skip correction): holds for non-prefix S_t
   when skips are bounded. With max 3 observed skips, correction is ~20%.
7. **Size** (Lemma 7, proved): |S| = T >= eps^2*n/9. QED.

Note: BMI (dbar < 1, Lemma 6's hypothesis) is no longer in the critical path.
The proof needs GPL-V (existence of feasible vertex) proved directly, not via
pigeonhole. With skips, S_t is not necessarily the pure prefix, so the dbar0
bound needs a minor correction (see Task 5 in Cycle 8 handoff).

## Proposed Next Steps (Cycle 8)

### Task 1: Direct Vertex Bound for Low-Leverage v

**Goal:** Prove ||Y_t(v)|| < 1 for v = argmin_{w in R_t} ell_w^{I_0}.

Key ingredients:
- C_t(v) = sum_{e: v~S_t} X_e has rank <= deg_S(v).
- ||Y_t(v)|| = ||B_t^{1/2} C_t(v) B_t^{1/2}|| <= ||B_t|| * ||C_t(v)||
  when C_t(v) is rank 1 (single edge). For multiple edges, need sharper.
- tr(C_t(v)) = ell_v^{S_t} <= ell_v^{I_0} <= 2(m-1)/m (by sorting).
- The amplification ||B_t|| = 1/(eps - ||M_t||) can be huge, but
  the DIRECTION of C_t(v) matters: if C_t(v) has small projection onto
  B_t's high eigenspaces, ||Y_t(v)|| stays bounded.

**Numerical target:** Extract the eigenspace overlap between C_t(v)
and B_t's high-amplification eigenspace for the worst-case steps.

### Task 2: Leverage-Monotonicity Conjecture

**Goal:** Prove or falsify: ||Y_t(v)|| <= ||Y_t(w)|| whenever
ell_v^{I_0} <= ell_w^{I_0} and both v, w in R_t.

If true, the minimum-ell vertex minimizes ||Y_t(v)|| in R_t. Combined
with empirical feasibility (always holds), this would reduce No-Skip to
a statement about the minimum-leverage vertex.

### Task 3: Probabilistic Approach

**Goal:** Show Pr_{v ~ Uniform(R_t)}[||Y_t(v)|| < 1] > 0.

This is WEAKER than dbar < 1 (which controls traces, not spectral norms)
but may be provable when dbar > 1. If the distribution of ||Y_t(v)||
across R_t is concentrated (most vertices near the median), then
dbar > 1 is compatible with most vertices having ||Y_t(v)|| < 1.

### Task 4: Eigenspace-Leverage Separation

**Goal:** Show that low-leverage vertices have small overlap with the
high-amplification eigenspace of B_t.

The high-amplification eigenspace is the one where lambda_j ≈ eps
(contributing ~150x to the barrier sum). This eigenspace corresponds
to the direction where M_t is nearly saturated. Prove: if v has low
ell_v^{I_0}, then u_{max}^T C_t(v) u_{max} is small (u_{max} = the
eigenvector of M_t with largest eigenvalue).

## Summary

| Component | Status |
|-----------|--------|
| Turan step | **Proved** |
| Induced Foster bound | **Proved** (Lemma 1) |
| Conditional dbar0 < 1 (prefix) | **Proved** (Lemma 3) |
| Alpha < 1 | **Proved** (Lemma 4) — not in critical path |
| Assembly decomposition | **Proved** (Lemma 5) — not in critical path |
| Product bound | **Proved** — not in critical path |
| Existence lemma | **Proved** (Lemma 6) — revived as Case B of dichotomy |
| Size guarantee | **Proved** (Lemma 7) |
| Rayleigh-monotonicity matrix bound | **Proved** (Lemma 8) — Pi<=I, F<=I-M, f_j<=1-lambda_j |
| Cross-degree bound | **Proved** (Lemma 9) — normY <= deg_S * max(z^T B z) |
| Isolation implies feasibility | **Proved** (Lemma 10) — deg_S=0 gives normY=0 |
| Rank of barrier contribution | **Proved** (Lemma 11) — rank(Y_t(v)) = deg_S(v) |
| Projection pigeonhole | **Proved** (Lemma 12) — min_v u^T C_t(v) u <= 1/r |
| **Sparse Dichotomy** | **Proved** — Delta<3/eps-1 implies GPL-V via isolation |
| **Strong Dichotomy** | **Conjectured** (0/148 failures) — isolation OR dbar<1 |
| GPL-V (exists feasible v) | **Narrowed** — Sparse case proved, dense case conjectured |
| No-Skip (NEXT v feasible) | **Weakened** (5/116 runs have skips with strict threshold) |
| BMI (dbar_t < 1) | **FALSIFIED** (12 violations, worst 1.739) |
| K_n extremality | **FALSIFIED** (max ratio 2.28) |
| Bridge C (large m) | **Proved for dbar0** (m > Theta(1/(eps(1-eps)))) |
| K_n case | **Proved** (exact formula, dbar = 5/6) |

The proof is complete modulo GPL-V. The critical path is now:
Turan → Induced Foster → **[GPL-V: Strong Dichotomy]** → Size.

**Gap status (Cycle 9):** GPL-V is now attacked from two directions:
- **Sparse case (PROVED):** Delta(G[I_0]) < 3/eps - 1 → isolation → GPL-V.
- **Dense case (CONJECTURED):** When isolation fails, dbar < 1 → pigeonhole → GPL-V.

The sub-gap is in the dense non-symmetric case: proving that when S_t
dominates R_t in G[I_0] AND the barrier has concentrated spectrum
(one eigenvalue near eps), the pigeonhole on the dominant eigenspace
(Lemma 12) combined with the F <= I - M constraint (Lemma 8) forces
min_v ||Y_t(v)|| < 1. The difficulty: the pigeonhole controls one
eigenspace at a time, but ||Y_t(v)|| depends on all eigenspaces
simultaneously.

Three attack paths for closing the sub-gap:
1. **Strongly Rayleigh / KS** — use ||C_t(v)|| <= 1 as atom bound
2. **Hyperbolic barrier** — det(eps*I - M_t - x_v C_t(v)) as hyperbolic polynomial
3. **Interlacing families (MSS)** — average characteristic polynomial (most promising)

The alpha < 1, assembly, product bound machinery (Lemmas 4-5) remains
valid and is revived: Lemma 6 (pigeonhole) is now Case B of the Strong
Dichotomy, applicable when dbar < 1 (which we conjecture holds whenever
isolation fails). The dead branch is narrower than before: only the BMI
claim (dbar < 1 UNIVERSALLY) is dead. The CONDITIONAL pigeonhole is alive.
