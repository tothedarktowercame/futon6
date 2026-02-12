# Problem 6: Epsilon-Light Vertex Subsets of Graphs

## Problem Statement

For a weighted graph G = (V, E, w) with n vertices and Laplacian L, a vertex
subset S is **epsilon-light** if L_{G[S]} <= epsilon L (Loewner order), where
L_{G[S]} is the induced-subgraph Laplacian zero-padded to R^{n x n}.

**Question.** Does there exist a universal constant c_0 > 0 such that for every
graph G and every epsilon in (0, 1), there exists S with |S| >= c_0 epsilon n
and L_{G[S]} <= epsilon L?

## Status

**Conditional.** Cases 1 and 2a proved unconditionally with c_0 = 1/3. Case 2b
(the sole remaining gap) reduced to a single operator-valued averaging lemma
(GPL-H). Strong numerical evidence supports GPL-H; no counterexample found.

## Unconditional Results

### Leverage Threshold Lemma

Any epsilon-light set S must be independent in the **heavy-edge subgraph**
G_H = (V, {e : tau_e > epsilon}), where tau_e = w_e R_eff(u,v) is the
leverage score of edge e.

*Proof.* Let M = L^{+/2} L_{G[S]} L^{+/2} and X_e = w_e L^{+/2} b_e b_e^T
L^{+/2}. If both endpoints of e lie in S, then M >= X_e (PSD sum dominance),
so ||M|| >= ||X_e|| = tau_e. If tau_e > epsilon, the condition ||M|| <= epsilon
is violated. QED

### Turan Bound

|H| <= n/epsilon edges (since sum_e tau_e = n - k <= n). By Turan's theorem,
alpha(G_H) >= epsilon n / (2 + epsilon) >= epsilon n / 3 for epsilon <= 1.

So there exists I, independent in G_H, with |I| >= epsilon n / 3, where all
edges internal to I satisfy tau_f <= epsilon.

### Case Resolution

**Case 1.** I is independent in G itself. Then L_{G[I]} = 0 <= epsilon L.
Take S = I, c_0 = 1/3.

**Case 2a.** I has internal edges, but ||sum_{f in I} X_f|| <= epsilon. Then
I itself is epsilon-light. Take S = I, c_0 = 1/3.

**Case 2b (OPEN).** ||sum_{f in I} X_f|| > epsilon. Need to thin I to
S subset I with |S| >= c_0 epsilon n and ||sum_{f in S} X_f|| <= epsilon.

### Star Domination Critique

The prior approach (replace Z_u Z_v with (Z_u + Z_v)/2, apply matrix
Bernstein) converts the quadratic p^2 sampling dependence to linear p,
destroying concentration headroom. For the star graph, this gives
||A_v|| = 1/2 for all vertices — making concentration impossible despite
the problem being trivially solvable (take all leaves).

### Technique Exhaustion

Six subsample-and-concentrate techniques all hit the same
quadratic-vs-linear scaling wall, producing c_0 = f(epsilon), never a
universal constant. The trace-only ceiling is proved: any certification
via tr(M) alone gives sublinear set size.

## The Remaining Gap: GPL-H

Case 2b reduces to a single conjecture. After deterministic core extraction
(Markov: regularize I to I_0 with |I_0| >= |I|/2 and leverage-degree
ell_v <= 12/epsilon), define the barrier greedy with potential
Phi_t = tr((epsilon I - M_t)^{-1}).

**Conjecture (GPL-H).** There exist universal c_step > 0 and theta in (0,1)
such that for every Case-2b state at time t <= c_step epsilon n with
M_t < epsilon I, there exists v in R_t with score_t(v) <= theta, where
score_t(v) = ||B_t^{1/2} C_t(v) B_t^{1/2}|| and B_t = (epsilon I - M_t)^{-1}.

**Reduction (proved).** GPL-H implies universal c_0 = gamma/6 via inductive
barrier maintenance: score < 1 guarantees M_{t+1} < epsilon I.

**Evidence.** 313 baseline + 2040 randomized greedy trajectories across
Erdos-Renyi, complete, barbell, random regular graphs (n <= 64). Worst
observed max_t min_v score_t(v) = 0.667. Exhaustive check at n <= 14
(13 Case-2b instances): worst score 0.476.

## References

- D. Spielman, N. Srivastava, "Graph sparsification by effective
  resistances," STOC 2008.
- A. Marcus, D. Spielman, N. Srivastava, "Interlacing families II:
  mixed characteristic polynomials and the Kadison-Singer problem,"
  Annals of Mathematics 182 (2015), 327-350.
- J. Batson, D. Spielman, N. Srivastava, "Twice-Ramanujan sparsifiers,"
  STOC 2009.
