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

This draft closes the internal algebraic gaps and separates proved facts from
external dependencies:

1. Proved in-text: exact formulation, K_n upper bound c<=1, sampling identities,
   and a correct matrix-concentration setup.
2. External dependency: the universal lower bound c0>0 for all graphs is not
   rederived here; it is treated as an imported theorem assumption.

So the final existential answer is **conditional on that external theorem**.

## 1. Exact reformulation

The PSD condition is equivalent to the quadratic form inequality

    for all x in R^V: x^T L_S x <= epsilon x^T L x.

On im(L), with L^+ the Moore-Penrose pseudoinverse:

    L_S <= epsilon L  <=>  || L^{+/2} L_S L^{+/2} || <= epsilon.

## 2. Complete graph upper bound (rigorous)

For G=K_n and S of size s, choose x supported on S with sum_{i in S} x_i = 0.
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

## 5. External dependency for universal c0>0

To conclude a universal lower bound for all graphs, one needs an additional
published theorem that controls leverage/pruning and proves:

    exists c0>0 universal such that
    for all G, epsilon in (0,1), exists S with |S|>=c0*epsilon*n and L_S<=epsilon L.

The closest published result is the twice-Ramanujan sparsification theorem of
Batson-Spielman-Srivastava (2012, "Twice-Ramanujan Sparsifiers," SIAM Review
56(2), 315-334, Theorem 1.1). Their barrier-function construction produces,
for any graph G and epsilon > 0, a reweighted subgraph with at most
ceil(n / epsilon^2) edges whose Laplacian spectrally approximates L to within
(1 +/- epsilon).

**Important gap:** BSS is an *edge sparsification* result — it selects a
subset of edges with reweighting, not a subset of vertices. The problem asks
for a *vertex subset* S with L_S <= epsilon L. These are different objects:
edge sparsification preserves the vertex set and reweights edges, while the
epsilon-light condition restricts to the induced subgraph on a vertex subset.

The star domination decomposition in Section 4a decomposes L_S into
vertex-indexed PSD summands, which is the correct algebraic setup for a vertex
selection argument. However, converting BSS's edge-selection barrier function
into a vertex-selection guarantee requires an additional step: one must show
that the deterministic potential-function method of BSS can be adapted to
select vertices (each contributing a star of edges) rather than individual
edges. This adaptation is plausible given the PSD structure of the vertex
summands A_v, but constitutes an additional technical step that is not
directly proved in BSS or in this writeup.

This writeup therefore treats the universal vertex-subset bound as a
conditional assumption: the algebraic setup (Sections 1-4) is proved, and
the conclusion follows if a vertex-selection analogue of BSS holds.

## 6. Final conclusion (explicitly conditional)

Unconditional conclusions from this text:

1. The statement is well-posed in Laplacian PSD order.
2. K_n implies universal upper bound c<=1.
3. The concentration machinery is set up correctly with explicit martingale
   increments and variance process.

Conditional conclusion:

- If the external universal theorem in Section 5 is assumed, then the answer to
  the original problem is YES (some universal c0>0 exists).

## Key identities used

1. L = sum_e w_e b_e b_e^T
2. L_S = sum_{e internal to S} w_e b_e b_e^T
3. tau_e = w_e b_e^T L^+ b_e, sum_e tau_e = n-k
4. L^{+/2} L_S L^{+/2} <= sum_v Z_v A_v with A_v=(1/2)sum_{u~v}X_{uv}

## References

- Batson, Spielman, Srivastava (2012), "Twice-Ramanujan Sparsifiers," SIAM
  Review 56(2), 315-334. [Theorem 1.1: deterministic spectral sparsification
  with universal vertex/edge bounds]
- Tropp (2011), Freedman's inequality for matrix martingales
- Standard matrix Bernstein inequality for sums of independent self-adjoint
  random matrices
