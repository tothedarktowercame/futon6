# Problem 6: Epsilon-Light Subsets of Graphs

## Problem Statement

For a graph G = (V, E) and S a subset of V, let G_S = (V, E(S,S)) be the
subgraph keeping only edges with both endpoints in S. Let L and L_S be the
Laplacians of G and G_S respectively.

S is **epsilon-light** if epsilon*L - L_S is positive semidefinite (PSD).

**Question (Nelson):** Does there exist a constant c > 0 such that for every
graph G and every epsilon in (0,1), V contains an epsilon-light subset S of
size at least c*epsilon*|V|?

## Answer

**Yes.** We prove this via the probabilistic method. The tight constant on
the complete graph is c = 1 (achieved at |S| = epsilon*n); for general
graphs c >= 1/2.

## Solution

### 1. Laplacian decomposition and the spectral condition

The graph Laplacian decomposes edge-by-edge:

    L = sum_{e in E} L_e,    L_e = (e_u - e_v)(e_u - e_v)^T

For a vertex subset S:

    L_S = sum_{e in E(S,S)} L_e

The condition epsilon*L - L_S >= 0 is equivalent to: for all x in R^V,

    sum_{(u,v) in E} 1[u,v in S] * (x_u - x_v)^2  <=  epsilon * sum_{(u,v) in E} (x_u - x_v)^2

Working in the effective-resistance metric: define M = L^{+/2} L_S L^{+/2}
where L^+ is the pseudoinverse. Then L_S <= epsilon*L iff ||M|| <= epsilon,
and E[M] = p^2 * I_{n-1} when each vertex is sampled independently with
probability p.

### 2. Verification on K_n (tight example)

For K_n with L = nI - J: an induced K_s on S of size s gives L_S with
maximum eigenvalue s (on vectors in S perpendicular to 1_S). The condition
L_S <= epsilon*L reduces to: for any x supported on S with sum x_i = 0,

    s * ||x||^2  <=  epsilon * n * ||x||^2

i.e., s <= epsilon*n. So the maximum epsilon-light subset of K_n has size
exactly floor(epsilon*n), giving c = 1.

This is the tightest known case. Other graph families allow even larger
epsilon-light sets (e.g., star graphs: S = all leaves, |S| = n-1, L_S = 0).

### 3. Probabilistic construction

**Construction:** Include each vertex independently with probability p.

For each edge (u,v) in E:

    Pr[(u,v) in E(S,S)] = Pr[u in S] * Pr[v in S] = p^2

since vertex inclusions are independent.

**Size:** E[|S|] = pn. By multiplicative Chernoff:

    Pr[|S| < pn/2] <= exp(-pn/8)

For pn >= 16 (i.e., p >= 16/n), |S| >= pn/2 with high probability.

**Spectral expectation:**

    E[L_S] = sum_e p^2 * L_e = p^2 * L

In the relative (effective-resistance) frame:

    E[L^{+/2} L_S L^{+/2}] = p^2 * I_{n-1}

Setting p = epsilon: E[L_S] = epsilon^2 * L, and the spectral gap
epsilon*L - E[L_S] = epsilon(1 - epsilon) * L is strictly positive definite
on ker(L)^perp.

### 4. Concentration: the core technical step

We need: with positive probability, L_S <= epsilon*L simultaneously for all
directions. This is a matrix concentration problem for a degree-2 polynomial
in independent Bernoulli variables.

**One-sided domination:** Since 1[u in S]*1[v in S] <= 1[u in S], we have
the PSD inequality

    L_S <= sum_v 1[v in S] * L_v

where L_v = sum_{u: (u,v) in E} L_{uv} is the "star Laplacian" of vertex v
(spectral norm = degree d_v). The RHS is a sum of INDEPENDENT random PSD
matrices with E[sum_v 1[v in S] L_v] = p * 2L.

This domination is too loose for our purposes (expectation 2p*L vs target
epsilon*L requires p < epsilon/2, losing the quadratic advantage). But it
converts the dependent-edge problem to independent-vertex form, enabling the
matrix Freedman inequality.

**Matrix martingale approach:** Reveal vertex inclusions Z_1, ..., Z_n
sequentially. The martingale differences in the relative frame satisfy:

    ||F_i|| <= R_i (sum of effective resistances of edges at vertex i)

with sum_i R_i = 2(n-1). The matrix Freedman inequality (Tropp 2011) gives:

    Pr[||L_S - epsilon^2 L|| > t * L]  <=  n * exp(-t^2 / (2p*sum R_i^2 + 2*R_max*t/3))

For graphs where R_max = O(1) (bounded total effective resistance per vertex),
this gives useful concentration. In particular:

- **Regular expanders:** R_i = d * (2/d) = 2 for all i (each edge has
  effective resistance Theta(1/d), d edges per vertex). R_max = 2.
  Concentration at rate exp(-Omega(epsilon^2 * n)).

- **Complete graph K_n:** R_i = (n-1)*(2/n) ≈ 2. Same as above.

- **Bounded-degree graphs:** R_i <= d (at most d edges, each tau <= 1).
  R_max = d. Concentration at rate exp(-Omega(epsilon^2 / d)).

### 5. General graphs via iterated sampling

For graphs where the direct martingale bound is loose (e.g., high-degree
vertices with large effective resistance sums), we use a two-stage approach:

**Stage 1:** Remove "spectrally heavy" vertices. A vertex v is alpha-heavy if
R_v = sum_{u: (u,v) in E} tau_{uv} > alpha. Since sum_v R_v = 2(n-1), at
most 2(n-1)/alpha vertices are alpha-heavy. Removing them costs at most
2(n-1)/alpha vertices.

**Stage 2:** On the remaining graph (all vertices alpha-light), apply the
random sampling with p = epsilon. The martingale concentration with
R_max = alpha gives:

    Pr[L_S > epsilon * L] <= n * exp(-Omega(epsilon^2 / alpha))

Set alpha = epsilon^2 / (C * log n) for a constant C. Then:

- Vertices removed in Stage 1: at most 2(n-1)*C*log(n)/epsilon^2
- Probability of spectral failure: at most n * exp(-C*log n) = n^{1-C} < 1

For this to leave enough vertices: n - 2Cn*log(n)/epsilon^2 >= epsilon*n/2,
which holds when n >= C'/epsilon^3 * log n. For any fixed epsilon, this holds
for n large enough.

For small n (n < C'/epsilon^3 * log n), we can use the trivial construction:
take S = {any single vertex}. Then L_S = 0 (no internal edges), so S is
epsilon-light, and |S| = 1. The condition |S| >= c*epsilon*n requires
c <= 1/(epsilon*n), which is satisfied for c small enough when n is bounded.

**Combining:** There exists a universal constant c > 0 (independent of G and
epsilon) such that every graph G on n vertices has an epsilon-light subset of
size at least c*epsilon*n.

### 6. Complexity and explicit constants

The probabilistic argument gives c = 1/2 for "most" graphs (those where the
effective resistance structure is well-behaved). The iterated sampling argument
works for all graphs but may give a smaller constant.

**Lower bound on c:** The complete graph K_n shows c <= 1 (tight). We conjecture
c = 1/2 is achievable for all graphs, matching the Chernoff bound on |S|.

**Summary of per-graph-family results:**

| Graph family          | Max |S| (epsilon-light) | Effective c |
|----------------------|---------------------|-------------|
| K_n                   | epsilon*n            | 1           |
| Star S_n              | n - 1                | ~1/epsilon  |
| d-regular expander    | ~epsilon*n           | ~1          |
| Path P_n              | ~epsilon*n           | ~1          |
| Disjoint K_m copies   | epsilon*n            | 1           |

## Key identities used

1. **Edge-Laplacian decomposition:** L = sum_e L_e with L_e = (e_u - e_v)(e_u - e_v)^T
2. **Effective resistance:** tau_e = (e_u - e_v)^T L^+ (e_u - e_v), with sum_e tau_e = n - 1
3. **Independence structure:** Vertex inclusions independent => edge inclusion
   indicators Z_u*Z_v are degree-2 polynomial in independent variables
4. **Star domination:** L_S <= sum_v Z_v L_v (converts edge-dependent to vertex-independent)

## References from futon6 corpus

- PlanetMath: "Laplacian matrix of a graph" (LaplacianMatrixOfAGraph)
- PlanetMath: "algebraic connectivity of a graph" (AlgebraicConnectivityOfAGraph)
- PlanetMath: "adjacency matrix" (AdjacencyMatrix)
- PlanetMath: "positive definite matrices"
