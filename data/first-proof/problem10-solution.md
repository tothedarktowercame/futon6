# Problem 10: RKHS-Constrained Tensor CP via Preconditioned Conjugate Gradient

## Problem Statement

Given the mode-k subproblem of RKHS-constrained CP decomposition with
missing data, the system to solve is:

    [(Z x K)^T SS^T (Z x K) + l(I_r x K)] vec(W) = (I_r x K) vec(B)

where:
- W in R^{n x r} is the unknown factor (A_k = KW)
- K in R^{n x n} is the PSD RKHS kernel matrix
- Z in R^{M x r} is the Khatri-Rao product of all other factors
- S in R^{N x q} is the selection matrix (q observed entries out of N = nM)
- B = TZ in R^{n x r} is the MTTKRP
- l > 0 is the regularization parameter
- n, r < q << N

Explain how PCG solves this without O(N) computation.

## Solution

### 1. Why direct methods fail

The system matrix A = (Z x K)^T SS^T (Z x K) + l(I_r x K) is nr x nr.
A direct solve costs O(n^3 r^3). But forming A explicitly requires
materializing (Z x K) in R^{N x nr}, which costs O(Nnr) -- proportional
to N. Since N = nM = n prod_i n_i can be enormous while only q entries
are observed, this is infeasible.

### 2. Implicit matrix-vector product (the key insight)

CG only requires the action v -> Av, never the matrix A itself. We compute
this in O(n^2 r + qr), independent of N.

Given v in R^{nr}, reshape as V in R^{n x r}.

**Step 2a: Forward map at observed entries only.**

By the Kronecker identity (A x B)vec(X) = vec(BXA^T):

    (Z x K) vec(V) = vec(KVZ^T)

The full result lives in R^N, but SS^T selects only q entries. Each
observed entry l at position (i_l, j_l) in the n x M unfolding satisfies:

    u_l = k_{i_l}^T V z_{j_l}

where k_i is row i of K and z_j is row j of Z. Grouping by row index i:
- Compute k_i^T V once per unique row: O(nr) each, O(n^2 r) total
- Dot with z_j per entry: O(r) each, O(qr) total

Cost: O(n^2 r + qr).

**Step 2b: Adjoint map from sparse result.**

The sparse vector w in R^N (q nonzeros) maps back via (Z x K)^T = (Z^T x K):

    (Z^T x K) w = vec(K W' Z)

where W' in R^{n x M} has the q nonzero entries of w. Since W' is sparse:
- W'Z touches only q nonzero entries: O(qr)
- K(W'Z) is n x n times n x r: O(n^2 r)

Cost: O(qr + n^2 r).

**Step 2c: Regularization term.**

    l(I_r x K) vec(V) = l vec(KV)

Cost: O(n^2 r).

**Total per matvec: O(n^2 r + qr).  No dependence on N.**

### 3. Right-hand side

    b = (I_r x K) vec(B)  where  B = TZ

T in R^{n x M} is the sparse mode-k unfolding (q nonzeros), so:
- TZ: O(qr) via sparse-dense multiply
- KB: O(n^2 r)

Cost: O(qr + n^2 r).

### 4. Preconditioner

**Choice:** P = (H x K) where H = Z^T Z + lI_r.

This approximates A by replacing SS^T with I (pretending full observation).
It captures both the kernel structure (K) and inter-factor coupling (Z^T Z).

**Why this structure?** The Khatri-Rao Hadamard property gives:

    Z^T Z = (A_1^T A_1) * (A_2^T A_2) * ... * (A_d^T A_d)

(elementwise/Hadamard product, excluding mode k). Each A_i^T A_i is r x r
and costs O(n_i r^2), so Z^T Z costs O(sum_i n_i r^2) -- vastly cheaper
than the naive O(Mr^2).

**Preconditioner solve** P y = z uses the Kronecker inverse:

    P^{-1} = H^{-1} x K^{-1}

Precompute once:
- Cholesky of K: O(n^3)
- Cholesky of H (r x r): O(r^3)

Each preconditioner application:
- Reshape z to Z' in R^{n x r}
- Solve K Y H^T = Z' via two triangular solves: O(n^2 r + nr^2)

Cost per solve: O(n^2 r).

### 5. Convergence

CG on the preconditioned system P^{-1}A converges in t iterations where:

    t = O(sqrt(kappa) log(1/eps))

and kappa = cond(P^{-1}A). Since P approximates A well when q/N is not
too small (the preconditioner assumes full observation), and the rank-r
structure constrains the spectrum, practical convergence is:

    t = O(r)  to  O(r sqrt(n/q) log(1/eps))

in typical tensor completion regimes.

### 6. Complexity summary

**Setup (once per ALS outer iteration):**

| Operation               | Cost                    |
|------------------------|-------------------------|
| Cholesky of K          | O(n^3)                  |
| Z^T Z via Hadamard     | O(sum_i n_i r^2)        |
| Cholesky of H          | O(r^3)                  |
| RHS: TZ + KB           | O(qr + n^2 r)           |

**Per CG iteration:**

| Operation               | Cost                    |
|------------------------|-------------------------|
| Matvec with A          | O(n^2 r + qr)           |
| Preconditioner solve   | O(n^2 r)                |

**Total per mode-k subproblem:**

    O(n^3 + t(n^2 r + qr))

where t = number of CG iterations (typically O(r)).

**Compare with direct solve:** O(n^3 r^3 + Nnr).

The PCG approach replaces the N-dependent term with q-dependent terms,
and the n^3 r^3 cubic-in-r term with n^3 + n^2 r^2 (via t ~ r iterations).
Since n, r < q << N, this achieves the required complexity reduction.

### 7. Algorithm

```
SETUP:
  L_K = cholesky(K)                          # O(n^3)
  H = hadamard_product(A_i^T A_i for i != k) + l * I_r  # O(sum n_i r^2)
  L_H = cholesky(H)                          # O(r^3)
  B = sparse_mttkrp(T, Z)                    # O(qr)
  b = vec(K @ B)                             # O(n^2 r)

PCG ITERATION (solve Ax = b):
  x_0 = 0
  r_0 = b
  z_0 = precond_solve(L_K, L_H, r_0)        # O(n^2 r)
  p_0 = z_0
  for i = 0, 1, 2, ...:
    w = matvec(p_i)                          # O(n^2 r + qr)
    alpha = (r_i^T z_i) / (p_i^T w)
    x_{i+1} = x_i + alpha * p_i
    r_{i+1} = r_i - alpha * w
    if ||r_{i+1}|| < eps * ||b||: break
    z_{i+1} = precond_solve(L_K, L_H, r_{i+1})  # O(n^2 r)
    beta = (r_{i+1}^T z_{i+1}) / (r_i^T z_i)
    p_{i+1} = z_{i+1} + beta * p_i

  W = reshape(x, n, r)

MATVEC(v):
  V = reshape(v, n, r)
  # Forward: evaluate at observed entries
  for each observed entry (i_l, j_l):
    u_l = k_{i_l}^T V z_{j_l}
  # Adjoint: accumulate from observed entries
  W' = sparse_matrix(n, M, entries u_l at positions (i_l, j_l))
  result = vec(K @ (W' @ Z)) + l * vec(K @ V)
  return result

PRECOND_SOLVE(L_K, L_H, z):
  Z' = reshape(z, n, r)
  solve L_K Y L_H^T = Z' by triangular substitution
  return vec(Y)
```

## Key References from futon6 corpus

- PlanetMath: "conjugate gradient algorithm", "method of conjugate gradients"
- PlanetMath: "Kronecker product", "positive definite matrices"
- PlanetMath: "properties of tensor product"
- physics.SE #27466: iterative solvers for large systems in physics
- physics.SE #27556: preconditioning for elliptic PDEs (Farago-Karatson)
