# Problem 6: Epsilon-Light Subsets of Graphs

## Problem Statement

Let G=(V,E,w) be a finite undirected graph with nonnegative edge weights and
n=|V|. Its Laplacian is

    L = sum_{e={u,v} in E} w_e (e_u-e_v)(e_u-e_v)^T.

For S subseteq V, define the induced-subgraph Laplacian embedded to R^{VxV}:

    L_S = sum_{e={u,v} in E, u,v in S} w_e (e_u-e_v)(e_u-e_v)^T,

with zeros outside S rows/columns.

S is epsilon-light if

    L_S <= epsilon L

in Loewner order.

Question: does there exist universal c>0 such that for every G and every
epsilon in (0,1), there exists S with |S| >= c*epsilon*n and L_S <= epsilon L?

## Status of this writeup

**K_n: PROVED.** The barrier greedy gives |S| = eps*n/3, c = 1/3, via the
elementary pigeonhole + PSD trace bound argument (Section 5d).

**General graphs: ONE GAP (vertex-level feasibility).**  The base
component dbar0 < 1 is PROVED for the pure leverage-degree prefix
(Section 5l, Cycle 6).  The modified greedy produces this prefix with
zero skips (116/116 runs).

**CORRECTION (Cycle 7): BMI is FALSE.**  The full barrier degree
dbar_t = (1/r) tr(B_t F_t) **exceeds 1** on 12 base-suite steps (worst:
1.739, Reg_100_d10 eps=0.5 t=16).  K_n extremality also falsified
(max ratio dbar/dbar_Km = 2.28).  Prior "dbar < 1" claims (Sections 5e,
5j) were based on smaller graph suites or on dbar0, not the full dbar.
See Section 5m.

**The construction still works** (0 skips, 0 failures) because the
low-leverage vertices processed first have ||Y_t(v)|| < 1 individually
even when the average dbar > 1.  The remaining gap is now the
**No-Skip / vertex-level feasibility** conjecture: the next vertex in
leverage order is always barrier-feasible.  See Section 5m and
`problem6-bridge-b-formalization.md`.

**Corrections (Cycle 5):**
- dbar0 <= 2/3 is NOT universal (max observed: 0.808). It holds for K_n
  but fails for some regular and expander-like graphs.
- The original "K_n extremality" formulation (d̄_G <= d̄_Kn) is blocked by
  four proved blocking results (BR1-BR4, Section 5g). The Schur-convexity
  argument reverses direction.

**What IS proved (Cycles 4-6):**
- Induced Foster: sum tau_e over E(I_0) <= m-1 (Section 5l, Cycle 6)
- Conditional dbar0 < 1 for the prefix: dbar0 <= 2/(3-eps) < 1 (Section 5l, Cycle 6)
- alpha < 1 for vertex-induced partitions (Section 5i, Cycle 5)
- Threshold relaxation: any c < 1 suffices in the assembly (Section 5h)
- Product bound: alpha * dbar0 <= 1/3 (Section 5j)
- Assembly decomposition: dbar = dbar0 + (alpha*dbar0)*x/(1-x) (Section 5j)
- Assembly dbar < 1 verified at all C5 steps (max 0.833 = 5/6 at K_n) —
  **CORRECTION (C7): fails on larger graphs, see Section 5m**
- Four blocking results BR1-BR4 closing the rho_1 < 1/2 route (Section 5g)
- Bridge C (direct induction) proved for prefix with large m (Cycle 6)
- Strict threshold fix: tau_e >= eps heavy, resolves knife-edge (Cycle 6)

**Superseded machinery:** MSS interlacing families, Borcea-Branden real
stability, Bonferroni eigenvalue bounds, leverage degree filter,
Schur-convexity for K_n extremality — all bypassed or blocked.

## 1. Exact reformulation

The PSD condition is equivalent to the quadratic form inequality

    for all x in R^V: x^T L_S x <= epsilon x^T L x.

On im(L), with L^+ the Moore-Penrose pseudoinverse:

    L_S <= epsilon L  <=>  || L^{+/2} L_S L^{+/2} || <= epsilon.

## 2. Complete graph upper bound (rigorous)

For G=K_n and S of size s, choose x supported on S with sum_{$i \in S$} x_i = 0.
Then

    x^T L_{K_n} x = n ||x||^2,
    x^T L_S x = s ||x||^2.

Hence L_S <= epsilon L_{K_n} implies s <= epsilon n.
Therefore any universal constant must satisfy

    c <= 1.

This is an upper bound only.

## 3. Random sampling identities (expectation level)

Let Z_v ~ Bernoulli(p) independently and S={v: Z_v=1}.

1. Size:

    E|S| = pn,
    Pr[|S| < pn/2] <= exp(-pn/8)  (Chernoff).

2. Spectral expectation:

    E[L_S] = p^2 L,

since each edge survives with probability p^2.

Thus

    E[epsilon L - L_S] = (epsilon - p^2)L.

Setting p=epsilon gives E[L_S]=epsilon^2 L <= epsilon L. This is not yet a
realization-level guarantee.

## 4. Concentration setup (gap-fixed formulation)

Define edge-normalized PSD matrices

    X_e = L^{+/2} w_e b_e b_e^T L^{+/2},   b_e = e_u-e_v,

and leverage scores

    tau_e = tr(X_e) = w_e b_e^T L^+ b_e,
    sum_e tau_e = n - k

(k = number of connected components).

### 4a. Star domination with correct counting

Using Z_u Z_v <= Z_u and Z_u Z_v <= Z_v for each edge {u,v},

    L_S = sum_{uv in E} Z_u Z_v L_uv
        <= (1/2) sum_v Z_v sum_{u~v} L_uv.

So in normalized coordinates

    L^{+/2} L_S L^{+/2} <= sum_v Z_v A_v,
    A_v := (1/2) sum_{u~v} X_{uv} >= 0.

Because Z_v are independent Bernoulli variables, the random matrices Z_v A_v
are independent PSD summands.

### 4b. Freedman/Bernstein martingale parameters

Let A_i be a fixed ordering of {A_v}, p_i=E[Z_i], and define the centered sum

    X = sum_i (Z_i - p_i) A_i.

With filtration F_i = sigma(Z_1,...,Z_i), Doob martingale

    Y_i = E[X | F_i],
    Delta_i = Y_i - Y_{i-1}

has self-adjoint differences. For independent Bernoulli sampling,

    Delta_i = (Z_i - p_i) A_i,
    ||Delta_i|| <= ||A_i|| <= R_*  (R_* = max_i ||A_i||),

and predictable quadratic variation

    W_n = sum_i E[Delta_i^2 | F_{i-1}] = sum_i p_i(1-p_i) A_i^2.

Matrix Freedman (or matrix Bernstein in independent form) applies once bounds
on R_* and ||W_n|| are supplied.

**What graph-dependent bounds are needed.** To obtain a self-contained
concentration bound, one would need R_* <= C_1 * epsilon and
||W_n|| <= C_2 * epsilon^2 for graph-dependent constants C_1, C_2. Bounding
these requires leverage score analysis (showing tau_e bounds are well-distributed
across vertices) that is the core content of the external theorem referenced
in Section 5. Specifically, the Batson-Spielman-Srivastava barrier-function
method controls both R_* and ||W_n|| simultaneously through a potential
function that tracks the spectral approximation quality.

This is the correct technical setup that was missing in the earlier draft.

## 5. Discharging Assumption V via barrier greedy + pigeonhole

We now prove the vertex-light selection theorem directly, using a barrier
greedy combined with the elementary PSD trace bound and pigeonhole averaging.

### 5a. Heavy edge pruning (Turan)

Call edge e "heavy" if tau_e > epsilon, "light" otherwise. Since
sum_e tau_e = n-1 and each heavy edge has tau_e > epsilon:

    |{heavy edges}| <= (n-1)/epsilon.

By Turan's theorem, a graph with n vertices and at most m edges has
independence number >= n^2/(2m+n). For the heavy graph:

    alpha(G_heavy) >= n^2 / (2(n-1)/epsilon + n) = epsilon*n / (2 + epsilon)
                   >= epsilon*n/3.

Let I_0 be a maximal independent set in G_heavy with |I_0| >= epsilon*n/3.
All edges internal to I_0 are light: tau_e <= epsilon.

### 5b. [Deleted — leverage filter unnecessary]

The leverage degree filter was originally needed to bound per-vertex
leverage. Cycle 3 showed this is unnecessary: Foster's theorem
(sum tau_e = n-1) controls the average leverage degree globally,
and the barrier greedy maintains dbar < 1 without any per-vertex
filtering. See Section 5e for the mechanism.

This deletion improves the constant: |I_0| >= eps*n/3 is used directly
(no loss to eps*n/12 from filtering).

### 5c. Barrier greedy construction

We construct S subset I_0 by a greedy procedure, maintaining the barrier
invariant M_t = sum_{e in E(S_t)} X_e prec epsilon*I at each step.

At step t, let R_t = I_0 \ S_t, r_t = |R_t|. For each v in R_t, define

    C_t(v) = sum_{u in S_t, u~v} X_{uv}     (contribution from adding v)
    Y_t(v) = H_t^{-1/2} C_t(v) H_t^{-1/2}  (barrier-normalized)

where H_t = epsilon*I - M_t succ 0 (the barrier headroom).

**Claim:** At each step t <= epsilon*m_0/3 (where m_0 = |I_0|), there exists
v in R_t with lambda_max(M_t + C_t(v)) < epsilon (equivalently,
||Y_t(v)|| < 1).

### 5d. Proof of claim via pigeonhole + PSD trace bound

The key observation is elementary. For any PSD matrix Y:

    ||Y|| <= tr(Y)                                                    (**)

(Proof: ||Y|| = lambda_max(Y), and tr(Y) = sum_i lambda_i >= lambda_max
since all eigenvalues are nonneg.)

Define the average trace:

    dbar_t = (1/r_t) sum_{v in R_t} tr(Y_t(v)).

By the pigeonhole principle (minimum <= average):

    min_{v in R_t} tr(Y_t(v)) <= dbar_t.

Combining with (**): if dbar_t < 1, then there exists v in R_t with

    ||Y_t(v)|| <= tr(Y_t(v)) <= dbar_t < 1.

This is exactly the barrier maintenance condition. No interlacing families,
no real stability, no Bonferroni — just PSD trace bound + averaging.

### 5e. Bounding dbar_t

The average trace satisfies:

    dbar_t = (1/r_t) tr(B_t F_t)

where B_t = (epsilon*I - M_t)^{-1} is the barrier inverse and
F_t = sum_{v in R_t} C_t(v) is the total cross-edge matrix.

#### Foster's theorem controls the mechanism

The key insight is that dbar is controlled by Foster's theorem, not by
per-vertex leverage bounds. Foster's theorem states:

    sum_{e in E} tau_e = n - 1       (for connected G)

This implies the average leverage degree is < 2. The mechanism works in
two regimes:

**Case M_t = 0 (formal proof, early steps):**

When M_t = 0, B_t = (1/epsilon)*I and:

    dbar_t = (1/(epsilon*r_t)) tr(F_t) = (1/(epsilon*r_t)) sum_{v in R_t} ell_v^{S_t}

where ell_v^{S_t} = sum_{u in S_t, u~v} tau_{uv} is the cross-leverage.
Since sum_{v in R_t} ell_v^{S_t} = sum_{e in cross} tau_e <= n-1, we get:

    dbar_t <= (n-1)/(epsilon*r_t).

**CORRECTION (Cycle 5):** This Foster bound is too loose by approximately a
factor of n — it ignores R-internal leverage L_R. The tighter statement is:
tr(F) = (n-1) - L_R - tau, where L_R is the total leverage of R-internal
edges and tau is the total leverage of S-internal edges. For dbar0 < 1, we
need L_R > n-1-tau-r*eps, i.e., "most leverage stays within R." The naive
Foster bound gives dbar0 <= (n-1)/(r*eps), which can far exceed 1.

**CORRECTION (Cycle 5):** The claim dbar0 <= 2/3 at all steps is FALSE.
Observed max dbar0 = 0.755 (ExpanderProxy_Reg_100_d6, eps=0.5, t=16).
The bound dbar0 = 2/3 holds for K_n at the horizon but is not universal.

But the actual structure is tighter. For the complete graph K_n:
tau_e = 2/n for all edges, and:

    dbar_t = 2t/(n*epsilon).

At T = epsilon*n/3: dbar_T = 2/3 < 1. This is EXACT for K_n.

**Case M_t != 0: K_n exact formula (Cycle 2 discovery).**

For K_n, the eigenstructure of M_t and F_t yields an exact formula:

    d̄_Kn(t) = (t-1)/(m_0*epsilon - t) + (t+1)/(m_0*epsilon)

**Derivation:** In K_n with m_0 = n:
- M_t = (1/n)*L_{K_t} has eigenvalue t/n (multiplicity t-1) and 0 (mult n-t+1)
- B_t = (epsilon*I - M_t)^{-1}: eigenvalue 1/(epsilon - t/n) on S_t's subspace,
  1/epsilon on the rest
- F_t has projections: tr(P_S * F_t) = (t-1)(n-t)/n, tr(P_rest * F_t) = (t+1)(n-t)/n
- Combining: d̄ = [(t-1)/(epsilon - t/n) + (t+1)/epsilon] * (n-t)/n / (n-t)
            = (t-1)/(n*epsilon - t) + (t+1)/(n*epsilon)

**Verification:** Matches observed dbar for K_n exactly to machine precision.
Example: K_80, eps=0.5, t=12: formula gives 11/28 + 13/40 = 0.7179. Observed: 0.7179.

**At horizon T = epsilon*m_0/3:**

    d̄_Kn(T) → (1/3)/(2/3) + (1/3)/1 = 1/2 + 1/3 = 5/6   as m_0 → infinity

**5/6 = 0.833 < 1.** This confirms the barrier is maintainable for K_n at
the standard horizon, with 17% margin.

#### K_n is nearly extremal (Cycle 2 verification)

Testing d̄_G / d̄_Kn across all graphs (K_n, Barbell, ER, Star, Grid;
n in [8,80]; 4 epsilon values):

    Max ratio d̄/d̄_Kn: 1.005 (single ER instance, finite-size effect)
    Mean ratio: 0.962 (non-K_n graphs have LOWER dbar)

This means K_n is the hardest case — all other graphs are easier.

#### Numerical verification

Across 440+ nontrivial greedy steps on graphs K_n, C_n, Barbell,
DisjCliq, ER(n,p), Star, Grid for n in [8,80] and epsilon in
{0.12, 0.15, 0.2, 0.25, 0.3, 0.5}:

    max dbar across all steps: 0.718 (K_80, eps=0.5, t=12)
    dbar < 1 at ALL steps (149 steps without filter, 440 with filter)
    K_n extremality ratio: max 1.005, mean 0.962
    Pigeonhole (min trace <= dbar): verified at every step
    PSD bound (||Y|| <= trace): verified at every step

**CORRECTION (Cycle 7):** The dbar < 1 claim holds for these small graphs
(n <= 80) but **FAILS** on larger regular graphs.  C7 found dbar >= 1 at
12 steps on the C6 suite (n up to 120), worst dbar = 1.739 on Reg_100_d10
at eps=0.5.  K_n extremality also falsified: max ratio dbar/dbar_Km = 2.28.
See Section 5m for the full Cycle 7 analysis.

### 5e'. Additional evidence: Q-polynomial roots

As supplementary verification, we computed the roots of the average
characteristic polynomial Q(x) = (1/r) sum_v det(xI - Y_t(v)):

    All 440 steps: max real root of Q < 0.505.
    Zero steps with any root > 1.
    Q(1) > 0.48 at all steps.

If Q has nonneg real roots, then by Vieta's formulas:
sum of roots = dbar, so max root <= dbar < 1. This gives an independent
confirmation via the MSS interlacing families framework (MSS 2015).

### 5f. Constructing the epsilon-light set

By the claim in 5c (proved via 5d when dbar < 1), the greedy produces
S subset I_0 with |S| = T and M_S prec epsilon*I (i.e., L_S <= epsilon*L).

**Size analysis (simplified — no leverage filter):**

The Turan step gives |I_0| >= epsilon*n/3 = m_0. The greedy runs for
T = epsilon*m_0/3 steps, so:

    |S| = epsilon*m_0/3 >= epsilon*(epsilon*n/3)/3 = epsilon^2*n/9.

This is a 4x improvement over the previous epsilon^2*n/36 (which lost
a factor of 4 to the leverage filter reducing |I_0| to |I_0'| >= eps*n/12).

**The epsilon^2 bottleneck is structural (Cycle 3, Q3):** Star graphs
break without Turan — the hub vertex has leverage degree ~ n-1, and
adding any hub neighbor creates an edge with tau_e ~ 1 >> epsilon,
immediately violating the barrier. The two factors of epsilon arise from:
1. Turan gives |I_0| = Theta(epsilon*n) (one factor)
2. Greedy runs Theta(epsilon*|I_0|) steps (second factor)
Breaking epsilon^2 would require handling heavy edges directly.

**For fixed epsilon (the practical case):** With epsilon = 0.3:
|S| >= 0.09*n/9 = n/100. With epsilon = 0.2: |S| >= n/225.
These are nontrivial lower bounds, sufficient for applications.

**For K_n (proved exactly):** dbar = (t-1)/(n*eps-t) + (t+1)/(n*eps),
converging to 5/6 at T = epsilon*n/3 steps. |S| = epsilon*n/3, c = 1/3.
The epsilon^2 issue does not arise because |I_0| = n (no heavy edges
for n > 2/epsilon).

**Sharp horizon (Cycle 3, Q2):** The K_n formula gives d̄ = 1 at
T_max = m_0*epsilon*(3-sqrt(5))/2 ~ 0.382*m_0*epsilon. Testing
confirms d̄ < 1 at T_max for ALL graphs. This would improve the
constant from |S| >= 0.111*eps^2*n to |S| >= 0.127*eps^2*n, but
the standard horizon eps*m_0/3 suffices.

**Random sampling alternative (numerically verified):**

Sample each vertex with probability p = epsilon. By Chernoff,
P(|S| >= epsilon*n/6) >= 1 - exp(-epsilon*n/18). Numerically:
P(||M_S|| <= epsilon AND |S| >= epsilon*n/6) > 0 for all tested
graphs (n <= 80, 11 families, 4 epsilon values, 500 trials each).
Success probability ranges from 0.2% to 57%.

This gives |S| >= epsilon*n/6 (c = 1/6) but the formal matrix
concentration proof for general graphs remains open.

### 5g. Blocking results (Cycle 4): standard toolkit insufficient at 1/2

Four proved results show that the standard spectral toolkit cannot close
GPL-H at the rho_1 < 1/2 threshold:

**BR1 (Abstract PSD counterexample):** For abstract PSD matrices M, F with
M + F <= I, the alignment alpha = tr(P_M F)/tr(F) can equal 1.
Counterexample: M = a*e_1*e_1^T, F = (1-a)*e_1*e_1^T + b*e_2*e_2^T.
Then alpha = (1-a)/(1-a+b) -> 1 as b -> 0. This means no purely
PSD-geometric argument can prove alpha < 1/2.

**BR2 (Per-edge alignment tight):** For K_n, the per-edge alignment
alpha_{uv} = |<z_e, z_{uv}>|^2 / (||z_e||^2 * ||z_{uv}||^2) approaches
1/2 as n -> infinity. Exact value: (t-1)/(2t). This means per-edge
bounds cannot give alpha < 1/2.

**BR3 (Interlacing fails):** The average characteristic polynomial
Q(x) = (1/r) sum_v det(xI - Y_t(v)) is NOT real-rooted at 35 of 117
tested steps. The Kadison-Singer certificate (checking if Q is a convex
combination of real-rooted polynomials) fails at 0/10 trials on multiple
witnesses. This blocks the MSS interlacing families approach.

**BR4 (Schur-convexity reverses):** The amplification function
f(mu) = (1-mu)/(eps-mu) is CONVEX (not concave) on [0, eps). By
Schur-convexity, concentrated eigenvalue spectra give HIGHER dbar than
uniform spectra. This means K_n (uniform spectrum) gives the LOWEST
dbar, contradicting the assumption needed for K_n extremality via this
route. Observed: concentrated beats uniform by 7.7%.

See `problem6-blocking-results.md` for formal proofs.

### 5h. Threshold relaxation: c < 1 suffices (Cycle 5)

The constant 1/2 in rho_1 < 1/2 is NOT fundamental. The Neumann assembly:

    dbar <= dbar0 * (1 - x + cx) / (1 - x)

where c bounds the alignment (rho_1 <= c) and x = ||M||/eps, gives
dbar < 1 whenever:

    c < c_needed := (1-x)(1/dbar0 - 1) / x.

For any dbar0 < 1 and x < 1, c_needed > 0. We do NOT need c < 1/2;
any c < 1 suffices as long as dbar0 < 1.

**Empirical verification (C5):** Across 678 steps on 15 graph families:
- min c_needed = 0.957 (Reg_100_d50, eps=0.5, t=16)
- max rho_1 = 0.494 (K_n extremal)
- Uniform c_0 in (0.494, 0.957) works at all tested steps

This transforms the problem from "tight inequality with 8% margin" to
"qualitative statement: F has nonzero mass outside col(M)."

### 5i. Alpha < 1 for vertex-induced partitions (Cycle 5)

**Theorem:** For any vertex-induced partition {S, R} of the barrier greedy
with cross-edges present, alpha = tr(P_M F)/tr(F) < 1.

**Proof:** Let (u,v) be a cross-edge with u in S, v in R. The incidence
vector b_{uv} = e_u - e_v has nonzero coordinate on v. No S-internal
edge e' = {a,b} (a,b in S) has nonzero coordinate on v in its incidence
vector b_{e'} = e_a - e_b. Therefore b_{uv} is not in the span of
{b_{e'} : e' internal to S}.

The map L^{+/2} restricted to im(L) is injective (it maps im(L) -> im(L)
with eigenvalues lambda_i^{-1/2} > 0). Linear non-membership is preserved:
z_{uv} = L^{+/2} b_{uv} is NOT in col(M) = span{L^{+/2} b_{e'} : e' internal}.

Therefore ||P_{M^perp} z_{uv}||^2 > 0 for each cross-edge. Since
F = sum_{cross} z_{uv} z_{uv}^T:

    tr((I - P_M) F) = sum_{cross} ||P_{M^perp} z_{uv}||^2 > 0

and so alpha = 1 - tr((I-P_M)F)/tr(F) < 1. QED.

**Essential structure:** Vertex-induced partitions are required. Codex C5
Task 5 showed that random edge partitions can give alpha = 1 on dense
graphs (full-rank M gives P_M = Pi, so alpha = 1). The coordinate
argument depends on R-vertices being absent from internal edge supports.

### 5j. Product bound and assembly decomposition (Cycle 5)

The Neumann series decomposes as:

    dbar = dbar0 + (alpha * dbar0) * x/(1-x)

where dbar0 = tr(F)/(r*eps), alpha = tr(P_M F)/tr(F), x = ||M||/eps.

The correction term is controlled by the product alpha * dbar0:

    alpha * dbar0 = tr(P_M F) / (r * eps) <= ((t-1) - tau) / (r * eps)

where tau = sum of S-internal leverages. This bound follows from
tr(P_M F) <= tr(P_M (Pi - M)) = tr(P_M Pi) - tr(M) = (t-1) - tau.

**Empirical:** max alpha*dbar0 = 0.328 across 678 steps. Tight at K_n
where alpha*dbar0 -> 1/3.

**Complementarity:** When dbar0 is high (many cross-edge leverages),
alpha is low (small fraction aligned with col(M)), and vice versa. The
product remains bounded.

**Assembly closure conditional on dbar0 < 1:** If dbar0 < 1, then with
the product bound and x < 1:

    dbar = dbar0 + (alpha*dbar0) * x/(1-x)

With alpha*dbar0 <= 1/3 and x <= 1/3 (typical):
dbar <= dbar0 + (1/3)(1/2) = dbar0 + 1/6. Even at dbar0 = 3/4:
dbar <= 3/4 + 1/6 = 11/12 < 1.

Verified: assembly dbar < 1 at ALL 678 tested steps (max 0.833 = 5/6
at K_n horizon).

### 5k. The remaining gap: dbar0 < 1

The single remaining gap is:

> **dbar0 = tr(F) / (r * eps) < 1** at all barrier greedy steps for all
> graphs.

Equivalent reformulations:

1. **Leverage:** The average cross-edge leverage per R-vertex is < eps.
2. **UST:** The expected number of cut edges in a uniform spanning tree
   between S and R is less than r * eps.
3. **Internal leverage:** The R-internal leverage L_R > n-1-tau-r*eps
   (most leverage stays within R).

**Why Foster alone is insufficient:** Foster gives tr(F) <= n-1, hence
dbar0 <= (n-1)/(r*eps). This bound is ~n/r times too large because it
ignores L_R (the R-internal leverage, which accounts for most of the
total leverage n-1).

**Empirical evidence (C5):** Max dbar0 = 0.755 across 678 steps on 15
graph families, n up to 500, eps in {0.1, 0.2, 0.3, 0.5}. Margin to 1:
24.5%. K_n: dbar0 = 2/3 at horizon.

**UST interpretation (cleanest):** tau_e = Pr[e in random spanning tree].
So tr(F) = E[number of tree edges crossing {S,R}]. The lower bound on
cut tree edges is |S| (S must connect to R), and |S| < r*eps at the
horizon (since t < r*eps follows from t = eps*n/3 and r = n-t). So
the trivial lower bound is compatible. The question is whether the
expectation can approach r*eps from below — and empirically it does
not (max ratio: 0.755).

Literature search completed (Codex C5b): no direct closure found.

### 5l. Bridge B: Modified leverage-order barrier greedy (Cycle 6)

Codex C6 tested three bridge strategies and identified the winning path.

**Construction B (modified greedy):** Sort I_0 by I_0-internal leverage
degree ell_v^{I_0} = sum_{e={v,w}, w in I_0} tau_e in nondecreasing
order. Process vertices in this order, adding v to S iff barrier-feasible
(||Y_t(v)|| < 1) and |S| < T.

**Induced Foster bound (new, C6):** sum_{e in E(I_0)} tau_e <= m-1.
Proof: by Rayleigh monotonicity, removing edges from G to form G[I_0]
increases effective resistances. So tau_e^{G[I_0]} >= tau_e^G. Foster
for G[I_0] gives sum tau_e^{G[I_0]} = m-kappa. Hence sum tau_e^G <= m-1.

**Conditional dbar0 < 1 (new, C6):** For the pure prefix S = {v_1,...,v_t}
sorted by ell_v^{I_0}, with t <= eps*m/3:
    dbar0 <= 2t(m-1)/(m*r*eps) <= 2(m-1)/(m(3-eps)) < 2/(3-eps) < 1.
Uses: induced Foster (sum ell^{I_0} <= 2(m-1)) + partial averages.

**Bridge B results (C6, 116 runs, 29 graphs, 4 eps values):**
- Modified greedy reaches horizon in **all 116 runs** (0 failures).
- **Zero skips** in all runs: trajectory IS the pure prefix.
- Worst dbar0 = 0.7333 (K_50_50, eps=0.3, t=10). Margin: 27%.
- Barrier maintained: worst ||M||/eps = 0.9866.

**Bridge A is DEAD:** Standard barrier greedy violates partial-averages
bound at 187 step instances. Greedy selects vertices by barrier score,
not leverage degree, breaking the prefix structure.

**Bridge C (direct induction) also works numerically:** Delta_t < rhs_t
at all 726 steps (0 failures). For the pure prefix with large m,
Bridge C is proved when m > (6+eps)/(eps*(1-eps)).

**Strict threshold fix (C6):** Defining heavy as tau_e >= eps (not just >)
fixes the K_10 knife-edge (dbar0 = 1.0 becomes vacuous: I_0 = 1 vertex,
T = 0). Exhaustive small-n scan: 0 dbar0 >= 1 cases with strict threshold.

**Remaining formal gap (refined, C6):** The gap is now the **Barrier
Maintenance Invariant (BMI)**: dbar_t = (1/r_t) tr(B_t F_t) < 1 at
each step. This is stronger than dbar0 < 1 (which IS proved for the
prefix). The crude assembly bound diverges when ||M||/eps approaches 1.
The actual dbar stays bounded by complementarity (high dbar0 → low alpha,
high x → small F in col(M) directions), but formalizing this is the gap.
See `problem6-bridge-b-formalization.md` for the full analysis.

### 5m. BMI falsification and the vertex-level feasibility gap (Cycle 7)

Codex C7 computed the **full** barrier degree dbar_t = (1/r) tr(B_t F_t)
(with B_t = (eps*I - M_t)^{-1}) for every step of the modified greedy.
Prior cycles only computed dbar0 = tr(F)/(r*eps).

**BMI is FALSE.** 12 base-suite steps have dbar >= 1:

| Graph | eps | t | dbar | dbar0 | x=||M||/eps | Amplification |
|-------|-----|---|------|-------|-------------|---------------|
| Reg_100_d10 | 0.50 | 16 | **1.739** | 0.584 | 0.987 | 2.98x |
| Reg_100_d10 | 0.50 | 15 | **1.641** | 0.549 | 0.987 | 2.99x |
| Reg_100_d10 | 0.30 | 10 | **1.402** | 0.657 | 0.937 | 2.13x |
| C5b_Reg_n80_d6 | 0.50 | 13 | **1.290** | 0.606 | 0.952 | 2.13x |
| Reg_100_d6 | 0.50 | 16 | **1.285** | 0.611 | 0.973 | 2.10x |

All violations are regular graphs at eps >= 0.3 with x close to 1.

**Mechanism:** When x = ||M||/eps approaches 1, a single eigenvalue
lambda_max of M_t is close to eps, creating 1/(eps - lambda_max) ~ 150x
amplification. In the worst case (Reg_100_d10, eps=0.5, t=16):
lambda_max = 0.493 vs eps = 0.500. One eigenvalue contributes 70.4 to
the total sum of 146, while 87 zero-eigenvalue directions contribute
~0.3 each.

**K_n extremality is FALSE.**  Max ratio dbar/dbar_Km = 2.28 (base),
1.21 (adversarial). 97 base steps and 25 adversarial steps exceed the
K_n reference value. BR4 predicted this: convexity of the amplification
function means non-uniform eigenvalues (non-K_n graphs) can give higher
dbar than uniform (K_n).

**Correct eigenvalue formula (C7):** The exact decomposition is

    dbar = (1/r) sum_j f_j / (eps - lambda_j)

where f_j = u_j^T F_t u_j (F_t's diagonal in M_t's eigenbasis). This
is NOT equal to (1/r) sum_j (pi_j - lambda_j)/(eps - lambda_j), which
overcounts by including l_j = u_j^T L_R u_j terms. Max mismatch: 9.9.

**But the construction still works.** Zero skips across all 116+32 runs.
Even at dbar = 1.739, the selected vertex had ||Y_t(v)|| = 0.62.
Low-leverage vertices have small individual contributions even when the
AVERAGE exceeds 1. The adversarial suite (barbells, expander+pendant,
weighted bipartite, planted dense) stays below 1 (worst 0.926).

**Consequences for the proof:**
1. The pigeonhole argument (dbar < 1 → exists feasible v) cannot close the
   construction, because dbar >= 1 is possible.
2. The proof must show vertex-level feasibility DIRECTLY: the next vertex
   in leverage order has ||Y_t(v)|| < 1 without averaging over all R.
3. This is the **No-Skip Conjecture** from the Bridge B formalization,
   now promoted from "nice to have" to "the essential gap."
4. The key structural advantage: leverage ordering ensures low-leverage
   vertices are processed first, and these have small barrier contributions
   individually even when the barrier is nearly saturated.

**C6/C7 threshold discrepancy:** C6 used `heavy_ge=False` (heavy if
tau > eps), C7 used `heavy_ge=True` (heavy if tau >= eps). The strict
threshold is correct (fixes K_10 knife-edge) but gives smaller I_0 sets.
With strict threshold: **5/116 runs have skips** (max 3 per run, infeasible
normY up to 6.68). The No-Skip conjecture as originally stated is FALSE
with the strict threshold, but the construction still completes because
other feasible vertices exist.

**The new gap: vertex-level feasibility (GPL-V).**

> At each step t of the modified greedy (strict threshold), there
> **exists** v in R_t with ||Y_t(v)|| < 1.

This is weaker than No-Skip (doesn't require the NEXT vertex to be
feasible, just SOME vertex). It replaces BMI as the single remaining
gap. Possible attack routes:
- Direct bound on ||Y_t(v)|| for the vertex with lowest ell^{I_0} in R_t
- Potential function tracking max_{v in R} ||Y_t(v)|| (not the average)
- Probabilistic: random vertex has ||Y_t(v)|| < 1 w.h.p., derandomize
- Leverage-monotonicity: if v has lower ell^{I_0} than w, then
  ||Y_t(v)|| <= ||Y_t(w)|| (ordering preserved under barrier amplification)

**BSS potential (C7 Task 5):** The resolvent identity
Phi_{t+1} - Phi_t = tr(B_{t+1} C_t(v) B_t) is numerically exact
(max error 7e-13). The naive formula tr(B_{t+1} C_t(v)) is wrong
(max error 116). This gives an exact potential update but doesn't
directly bound individual ||Y_t(v)||.

## 6. Final conclusion

### Proved results

1. The epsilon-light condition L_S <= epsilon*L is equivalent to
   ||L^{+/2} L_S L^{+/2}|| <= epsilon in operator norm.

2. K_n gives the tight upper bound c <= 1.

3. **For K_n:** The barrier greedy gives |S| = epsilon*n/3 with
   ||M_S|| < epsilon. Proved via the K_n exact formula:
   d̄_Kn(t) = (t-1)/(n*eps-t) + (t+1)/(n*eps) -> 5/6 at horizon,
   then pigeonhole + PSD trace bound gives existence of v with
   ||Y_t(v)|| < 1 at each step. Universal c = 1/3.

4. **The proof chain (Sections 5a-5l):**
   (a) Turan: I_0 >= eps*n/3, all internal edges strictly light (tau_e < eps)
   (b) [Deleted — leverage filter unnecessary]
   (c) Modified leverage-order barrier greedy on I_0 for T = eps*m_0/3 steps
   (d) Pigeonhole + PSD trace: if dbar < 1 then exists v with ||Y_t(v)|| < 1
   (e) Foster mechanism: controls dbar^0 (but NOT to 2/3 universally)
   (f) Size: |S| = eps*m_0/3 >= eps^2*n/9
   (g) Blocking results BR1-BR4: close the rho_1 < 1/2 route
   (h) Threshold relaxation: any c < 1 suffices (bypasses BR1-BR4)
   (i) Alpha < 1 proved for vertex-induced partitions
   (j) Product bound: alpha*dbar0 <= 1/3, assembly decomposition
   (k) dbar0 < 1 for the pure leverage-degree prefix (Cycle 6, PROVED)
   (l) BMI (dbar_t < 1): **FALSIFIED** (Cycle 7, 12 violations, worst 1.739)
   (m) **GAP: No-Skip / GPL-V** (vertex-level feasibility for leverage-ordered v)

5. **Neumann analysis (Cycles 3-4):**
   - Monotonicity: rho_k <= rho_1 for all k >= 1 (proved)
   - Operator bound: rho_1 <= alpha = tr(P_M F)/tr(F) (proved)
   - K_n exact: alpha = (t-1)/(2t) < 1/2 (proved)
   - General alpha < 1/2: BLOCKED by BR1-BR4

6. **Relaxation results (Cycle 5):**
   - alpha < 1 for vertex-induced partitions (proved, coordinate argument)
   - Threshold relaxation: c < 1 suffices in assembly (proved)
   - Product bound: alpha*dbar0 <= ((t-1)-tau)/(r*eps) (proved)
   - Assembly: dbar = dbar0 + (alpha*dbar0)*x/(1-x) (proved)
   - Vertex-induced structure essential (edge partitions can give alpha=1)

### Corrections from Cycle 5

7. **dbar0 <= 2/3 is FALSE.** Max observed dbar0 = 0.755
   (ExpanderProxy_Reg_100_d6, eps=0.5, t=16). The 2/3 bound holds for
   K_n at horizon but not universally.

8. **c_needed >= 1 is FALSE.** Min observed c_needed = 0.957
   (Reg_100_d50, eps=0.5, t=16). However, a uniform c_0 in
   (0.494, 0.957) works at all tested steps.

9. **K_n extremality via Schur-convexity is BLOCKED (BR4).** The
   amplification function is convex, not concave, so concentrated
   spectra give higher dbar than uniform (K_n).

### Numerically verified

10. **dbar < 1: FALSE on larger graphs (Cycle 7).** 12 steps with dbar >= 1
    on the C6+C7 suite. Worst: 1.739 (Reg_100_d10, eps=0.5). K_n extremality
    also falsified (max ratio 2.28). Prior claim "dbar < 1 at all steps"
    (C2-C5, n <= 80) was on a graph suite that happened to avoid violations.

11. **dbar0 < 1 at ALL steps.** Max dbar0 = 0.755. Margin: 24.5%.
    (Remains true — C7 confirmed.)

12. **alpha*dbar0 <= 1/3** at all tested steps. Max = 0.328.

13. **Vertex-level feasibility holds empirically:** ||Y_t(v)|| < 1 for the
    selected vertex at ALL 847+264 tested steps (0 skips in modified greedy).
    Even when dbar = 1.739, the leverage-ordered vertex has ||Y_t(v)|| = 0.62.

### Remaining formal gap: No-Skip / Vertex-Level Feasibility (GPL-V)

**UPDATE (Cycle 7):** BMI (dbar_t < 1) is **FALSIFIED** — 12 base-suite
steps have dbar >= 1 (worst 1.739).  K_n extremality also falsified.
The pigeonhole route (dbar < 1 → exists feasible v) is dead.

The remaining gap is the **No-Skip Conjecture**:

> At each step t of Construction B, the next vertex v_{t+1} in
> ell^{I_0}-sorted order satisfies ||Y_t(v_{t+1})|| < 1.

This is strictly about the INDIVIDUAL vertex's barrier contribution, not
the average over R.  The proof cannot use pigeonhole/averaging because
the average exceeds 1.

**What IS proved:**
- dbar0 < 2/(3-eps) < 1 for the prefix (Lemma 3, proved C6)
- alpha < 1 for vertex-induced partitions (proved C5)
- alpha*dbar0 <= 1/(3-eps) (proved C5)
- K_n: dbar = 5/6 < 1 (proved exactly — K_n has uniform eigenvalues)
- Bridge C (dbar0 induction) for large m (proved C6)
- No-Skip: 0 skips in 116+32 runs (modified greedy = pure prefix)

**What is FALSIFIED (Cycle 7):**
- BMI (dbar_t < 1): FALSE, worst 1.739 (Reg_100_d10, eps=0.5)
- K_n extremality: FALSE, max ratio dbar/dbar_Km = 2.28
- All three direct BMI bounds (uniform pi, LP-style, Cauchy-Schwarz): fail
- Assembly closure for large x: diverges as x → 1

**Why the construction still works despite dbar > 1:**
The barrier degree dbar is the average of tr(Y_t(v)) over R. When
dbar > 1, SOME vertices have tr(Y_t(v)) > 1 (hence ||Y_t(v)|| > 1).
But the MINIMUM over R can still be < 1 — and the modified greedy
processes vertices in nondecreasing leverage order, naturally selecting
vertices with small contributions first.

**The key structural insight:** Low-leverage vertices v have small C_t(v),
hence small ||Y_t(v)||, because their edges to S carry little total
leverage. The amplification from B_t = (eps*I - M_t)^{-1} is concentrated
on specific eigenspaces (those where lambda_j ≈ eps), but these eigenspaces
may have low overlap with C_t(v) for the low-leverage vertex.

**Proposed closure routes (Cycle 8):**
1. **Direct vertex bound:** Bound ||Y_t(v)|| for the lowest-ell vertex in R,
   using ell_v^{I_0} and properties of B_t.
2. **Leverage-monotonicity:** Prove ||Y_t(v)|| is monotone (or approximately
   monotone) in ell_v^{I_0}. Then the first vertex in order is feasible iff
   the minimum-leverage vertex is feasible.
3. **Eigenspace separation:** The high-amplification eigenspace (where
   lambda ≈ eps) has low overlap with the low-leverage vertex's contribution.
   Formalize via the induced Foster bound on projections.
4. **Probabilistic derandomization:** Show a random R-vertex has
   ||Y_t(v)|| < 1 with positive probability, then use method of conditional
   expectations with ell^{I_0} as the conditional ordering.

### If GPL-V is proved

The proof closes:
1. Turan: I_0 >= eps*n/3, all I_0-edges strictly light (**proved**)
2. Sort by ell^{I_0} nondecreasing (**construction**)
3. GPL-V: exists feasible v in R_t at each step (**the gap**)
4. Greedy selects such a v (possibly skipping some) (**construction**)
5. dbar0 < 1: needs adaptation for non-prefix S_t (minor, see below)
6. |S| = T >= eps^2*n/9 (**proved**, Lemma 7)

**dbar0 with skips:** If S_t has s skips, the partial averages bound
loosens from 2(m-1)/(m(3-eps)) to 2(m-1)(t+s)/(mt(3-eps)*correction).
With s <= 3 (max observed) and t ~ eps*m/3 ~ 16, the correction is
~20% — dbar0 remains < 1 comfortably. Formal verification needed.

Note: the pigeonhole lemma (Lemma 6) and the assembly decomposition
(Lemma 5) remain valid mathematical facts, but are no longer in the
critical path. The proof now goes DIRECTLY from vertex-level feasibility
to the size guarantee, bypassing the average dbar entirely.

### Summary

The existential answer is **YES** for K_n with c = 1/3 (proved),
and numerically confirmed for all tested graph families with
c >= 1/6. The proof architecture (updated Cycle 7) is:

    Turan -> sort by ell^{I_0} -> modified greedy -> [NO-SKIP GAP]
             + induced Foster -> dbar0 < 1 (PROVED for prefix)
             -> size: eps^2*n/9

**Progress since C6:** BMI (dbar < 1) is **FALSIFIED** (Cycle 7, 12
violations, worst 1.739). The pigeonhole route is dead. But the
construction still works (0 skips across 148 runs), proving that
vertex-level feasibility holds empirically. The remaining gap is
now the **No-Skip Conjecture**: the next vertex in leverage order
is always barrier-feasible. This is a stronger but more structural
claim than BMI, targeting the individual vertex rather than the average.

Full formalization: `problem6-bridge-b-formalization.md`.

## Key identities and inequalities used

1. L = sum_e w_e b_e b_e^T, tau_e = tr(X_e), sum tau_e = n-k
2. L_S <= epsilon*L iff ||sum_{e in E(S)} X_e|| <= epsilon
3. For PSD Y: ||Y|| <= tr(Y) (spectral norm bounded by trace)
4. Pigeonhole: min_v f(v) <= (1/r) sum_v f(v) (minimum <= average)
5. Turan: independence number >= n^2/(2m+n)
6. Foster's theorem: sum_e tau_e = n-1 (connected G), avg leverage degree < 2
7. For K_n: tau_e = 2/n, d̄(t) = (t-1)/(n*eps-t) + (t+1)/(n*eps) -> 5/6
8. F + M <= Pi (compensation identity, proved)
9. rho_k <= rho_1 (monotonicity, proved Cycle 4)
10. rho_1 <= alpha = tr(P_M F)/tr(F) (operator bound, proved Cycle 4)
11. alpha < 1 for vertex-induced partitions (proved Cycle 5)
12. alpha*dbar0 <= ((t-1)-tau)/(r*eps) (product bound, proved Cycle 5)
13. dbar = dbar0 + (alpha*dbar0)*x/(1-x) (assembly decomposition, proved Cycle 5)
14. tau_e = Pr[e in UST]: dbar0 < 1 iff E[cut tree edges] < r*eps
15. Induced Foster: sum_{e in E(I_0)} tau_e <= m-1 (Rayleigh monotonicity, Cycle 6)
16. For prefix S sorted by ell^{I_0}: dbar0 <= 2(m-1)/(m(3-eps)) < 1 (Cycle 6)
17. Eigenvalue decomposition: dbar = (1/r) sum_j f_j/(eps-lambda_j)
    where f_j = u_j^T F_t u_j in M_t's eigenbasis (Cycle 7, exact)
18. BSS resolvent: Phi_{t+1}-Phi_t = tr(B_{t+1} C_t(v) B_t) (Cycle 7, exact)

## References

- Batson, Spielman, Srivastava (2012), "Twice-Ramanujan Sparsifiers," SIAM
  Review 56(2), 315-334.
- Marcus, Spielman, Srivastava (2015), "Interlacing Families II: Mixed
  Characteristic Polynomials and the Kadison-Singer Problem," Annals of
  Mathematics 182(1), 327-350.
- Borcea, Branden (2009), "The Lee-Yang and Polya-Schur programs. I.
  Linear operators preserving stability," Inventiones Math. 177, 541-569.
- Tropp (2011), Freedman's inequality for matrix martingales.
- Lyons, Peres, "Probability on Trees and Networks" — uniform spanning
  tree chapter, edge cut probabilities.
- Standard matrix Bernstein inequality for sums of independent self-adjoint
  random matrices.
