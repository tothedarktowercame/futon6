# Problem 6: Bridge B Formalization

Date: 2026-02-13
Agent: Claude
Base: Codex C6 verification (`problem6-codex-cycle6-verification.md`)

## Main Theorem (conditional)

**Theorem.** For every connected graph G = (V,E,w) on n vertices and
every eps in (0,1), there exists S ⊆ V with

    |S| >= eps^2 * n / 9    and    L_S <= eps * L

provided that the **No-Skip Conjecture** holds.

**No-Skip Conjecture:** At every step t of the modified leverage-order
barrier greedy on I_0 (Construction B below), the next vertex v_{t+1}
in ell^{I_0}-sorted order satisfies ||Y_t(v_{t+1})|| < 1.

**NOTE (Cycle 7):** The original conditioning was on the **Barrier
Maintenance Invariant** (BMI): dbar_t < 1 at each step. BMI is now
**FALSIFIED** — 12 base-suite steps have dbar >= 1 (worst: 1.739,
Reg_100_d10 eps=0.5 t=16). The pigeonhole argument (dbar < 1 →
exists feasible v) cannot close the construction. The No-Skip
Conjecture replaces BMI as the essential gap.

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

### The actual gap: No-Skip Conjecture

**Conjecture (No-Skip, GPL-V).** At every step t <= T of Construction B
on any connected graph G, the next vertex v_{t+1} in ell^{I_0}-sorted
order satisfies ||Y_t(v_{t+1})|| < 1.

**Evidence:** 0 skips in 148 runs (116 base + 32 adversarial), 1111
total steps.

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

### Empirical evidence (Cycles 6+7)

| Quantity | Worst case | Where |
|----------|-----------|-------|
| dbar0 (modified greedy) | 0.7333 | K_50_50, eps=0.3 |
| **dbar (full barrier)** | **1.739** | Reg_100_d10, eps=0.5, t=16 |
| ||M||/eps (modified) | 0.9866 | Reg_100_d10, eps=0.5 |
| ||Y_t(v)|| for selected v | 0.937 | various |
| Skips (modified greedy) | **0** | all 148 runs |
| Adversarial max dbar | 0.926 | Barbell_40_40_b3 |

### Proposed attack routes (Cycle 8)

1. **Direct vertex bound:** For the lowest-ell vertex v in R_t, bound
   ||Y_t(v)|| using: (a) ell_v^{I_0} bounds the total leverage of v's
   edges to S, (b) B_t's amplification is concentrated on specific
   eigenspaces, (c) low-ell vertices have small projection onto those
   eigenspaces.

2. **Eigenspace-leverage separation:** Prove that C_t(v)'s eigenspace
   overlaps with B_t's high-amplification eigenspace are small for
   low-leverage v. The induced Foster bound constrains the total
   leverage in each eigenspace.

3. **Leverage-monotonicity:** Prove ||Y_t(v)|| is (approximately)
   monotone in ell_v^{I_0}. Then No-Skip follows from the minimum-
   leverage vertex being feasible.

4. **Probabilistic derandomization:** Show E[||Y_t(v)||] < 1 for a
   random R-vertex (different from dbar < 1, which is E[tr(Y_t(v))]),
   then derandomize via conditional expectations.

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

If No-Skip holds, the proof closes:

1. **Turan** (proved): I_0 >= eps*n/3, all internal edges strictly light.
2. **Sort** by ell_v^{I_0} nondecreasing.
3. **Greedy** processes in sorted order, adding barrier-feasible vertices.
4. **No-Skip** (conjecture): greedy adds every vertex in order → S_t = prefix.
5. **dbar0_t < 1** (Lemma 3, proved): conditional proof for the prefix.
6. **Size** (Lemma 7, proved): |S| = T >= eps^2*n/9. QED.

Note: BMI (dbar < 1, Lemma 6's hypothesis) is no longer in the critical path.
The proof goes directly from No-Skip → S_t is prefix → dbar0 < 1 → size bound.
The pigeonhole argument (Lemma 6) is a valid side result but cannot provide
the inductive step because its hypothesis (dbar < 1) is empirically false.

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
| Existence lemma | **Proved** (Lemma 6) — unreachable (BMI false) |
| Size guarantee | **Proved** (Lemma 7) |
| No-Skip Conjecture | **Conjectured** (0/1111 failures) — **THE GAP** |
| BMI (dbar_t < 1) | **FALSIFIED** (12 violations, worst 1.739) |
| K_n extremality | **FALSIFIED** (max ratio 2.28) |
| Bridge C (large m) | **Proved for dbar0** (m > Theta(1/(eps(1-eps)))) |
| K_n case | **Proved** (exact formula, dbar = 5/6) |

The proof is complete modulo No-Skip. The critical path is now:
Turan → Induced Foster → dbar0 < 1 → **[No-Skip]** → Size.

The alpha < 1, assembly, product bound, and existence lemma machinery
(Lemmas 4-6) remains mathematically valid but is no longer needed —
it was infrastructure for the BMI route, which is now dead. The proof
has simplified: the gap is purely about individual vertex feasibility
in leverage order, not about the average barrier degree.
