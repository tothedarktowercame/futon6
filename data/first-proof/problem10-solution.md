# Problem 10: RKHS-Constrained Tensor CP via Preconditioned Conjugate Gradient

## Problem Statement

Given the mode-k subproblem of RKHS-constrained CP decomposition with
missing data, solve

    [(Z x K)^T D (Z x K) + lambda (I_r x K)] vec(W) = (I_r x K) vec(B),

where D = S S^T is the observation projector (q observed entries out of
N = nM), B = T Z, and n, r << q << N.

## Assumptions Used (explicit)

1. lambda > 0.
2. For standard PCG and Cholesky-based preconditioning, use a PD kernel
   K_tau = K + tau I_n with tau > 0 (or assume K is already PD).
3. S is a selection operator, so D is diagonal/projector and sparse by index list.

Then the solved system is

    A_tau x = b_tau,
    A_tau = (Z x K_tau)^T D (Z x K_tau) + lambda (I_r x K_tau),
    x = vec(W),
    b_tau = (I_r x K_tau) vec(B).

Under these assumptions A_tau is SPD, so PCG applies.

## Solution

### 1. Why naive direct methods fail

A_tau is an (nr) x (nr) system. Dense direct factorization costs
O((nr)^3) = O(n^3 r^3).

A naive explicit route also materializes Phi = Z x K_tau in R^{N x nr},
which costs O(N n r) memory/work before factorization. This is the
N-dependent bottleneck we avoid with matrix-free PCG.

### 2. Implicit matrix-vector product in O(n^2 r + q r)

CG needs only y = A_tau x, not A_tau explicitly.

Given x = vec(V), V in R^{n x r}:

1. U = K_tau V. Cost O(n^2 r).
2. Forward sampled action (only observed entries):

       (Z x K_tau) vec(V) = vec(K_tau V Z^T).

   For each observed coordinate (i_l, j_l),

       u_l = <U[i_l, :], Z[j_l, :]>.

   Total O(q r).
3. Form sparse W' in R^{n x M} from u_l. Let s = nnz(W') <= q.
4. Adjoint sampled action:

       (Z^T x K_tau) vec(W') = vec(K_tau W' Z).

   Compute W' Z in O(s r) <= O(q r), then left-multiply by K_tau in O(n^2 r).
5. Add regularization term lambda vec(K_tau V), cost O(n^2 r).

Total per matvec:

    O(n^2 r + q r),

with no O(N) term.

### 3. Right-hand side

B = T Z with T sparse (q nonzeros):

1. T Z: O(q r)
2. K_tau B: O(n^2 r)

So b_tau = (I_r x K_tau) vec(B) is formed in O(q r + n^2 r).

### 4. Preconditioner that matches the corrected algebra

Use D = S S^T and whiten by K_tau^{-1/2}:

    x = (I_r x K_tau^{-1/2}) y.

Then

    Ahat = (I_r x K_tau^{-1/2}) A_tau (I_r x K_tau^{-1/2})
         = (Z x K_tau^{1/2})^T D (Z x K_tau^{1/2}) + lambda I.

If sampling is roughly uniform, D ~ c I with c = q/N. Then

    Ahat ~ c (Z^T Z x K_tau) + lambda I.

Choose Kron preconditioner in whitened coordinates:

    Phat = (c Z^T Z + lambda I_r) x I_n.

Mapping back gives

    P = (c Z^T Z + lambda I_r) x K_tau = H x K_tau,
    H = c Z^T Z + lambda I_r.

This is the missing justification for using H x K_tau (instead of claiming
it is the exact D = I system).

Khatri-Rao identity still gives efficient Gram formation:

    Z^T Z = Hadamard_i (A_i^T A_i),

cost O(sum_i n_i r^2).

Preconditioner apply:

    P^{-1} = H^{-1} x K_tau^{-1},

implemented by solving K_tau Y H^T = Z' after reshape.
Per application cost is O(n^2 r + n r^2) (often simplified to O(n^2 r)
when n >> r).

### 5. Convergence (tightened)

For SPD A_tau and SPD P, standard PCG gives

    ||e_t||_{A_tau} <= 2 ((sqrt(kappa)-1)/(sqrt(kappa)+1))^t ||e_0||_{A_tau},

with kappa = cond(P^{-1/2} A_tau P^{-1/2}), so

    t = O(sqrt(kappa) log(1/eps)).

To claim "fast" convergence, add a spectral-equivalence hypothesis, e.g.

    (1-delta) P <= A_tau <= (1+delta) P, 0 < delta < 1,

which implies

    kappa(P^{-1} A_tau) <= (1+delta)/(1-delta).

Hence t is logarithmic in 1/eps with a modest sqrt(kappa) factor when
delta is bounded away from 1. (No unsupported closed-form t = O(r sqrt(n/q))
claim is needed.)

### 6. Complexity summary

Setup per ALS outer step:

1. Cholesky(K_tau): O(n^3)
2. Z^T Z via Hadamard Grams: O(sum_i n_i r^2)
3. Cholesky(H): O(r^3)
4. RHS: O(q r + n^2 r)

Per PCG iteration:

1. Matvec: O(n^2 r + q r)
2. Preconditioner apply: O(n^2 r + n r^2)

Total:

    O(n^3 + r^3 + sum_i n_i r^2 + q r + n^2 r
      + t (n^2 r + q r + n r^2)).

In the common regime n >= r, this simplifies to

    O(n^3 + t (n^2 r + q r)),

with dependence on q (observed entries) rather than N (all entries).

### 7. Algorithm

```text
SETUP:
  K_tau = K + tau * I_n                    # tau > 0 if K is only PSD
  L_K = cholesky(K_tau)                    # O(n^3)
  G = hadamard_product(A_i^T A_i for i != k)  # O(sum_i n_i r^2)
  c = q / N
  H = c * G + lambda * I_r
  L_H = cholesky(H)                        # O(r^3)
  B = sparse_mttkrp(T, Z)                  # O(qr)
  b = vec(K_tau @ B)                       # O(n^2 r)

PCG(A_tau x = b, preconditioner P = H x K_tau):
  x0 = 0
  r0 = b
  z0 = precond_solve(L_K, L_H, r0)         # O(n^2 r + n r^2)
  p0 = z0
  repeat until convergence:
    w = matvec_A_tau(p)                    # O(n^2 r + q r)
    alpha = (r^T z) / (p^T w)
    x = x + alpha * p
    r_new = r - alpha * w
    if ||r_new|| <= eps * ||b||: break
    z_new = precond_solve(L_K, L_H, r_new)
    beta = (r_new^T z_new) / (r^T z)
    p = z_new + beta * p
    r, z = r_new, z_new
  W = reshape(x, n, r)

matvec_A_tau(v):
  V = reshape(v, n, r)
  U = K_tau @ V
  for each observed (i_l, j_l):
    u_l = dot(U[i_l, :], Z[j_l, :])
  Wprime = sparse(n, M, entries u_l)
  Y = K_tau @ (Wprime @ Z) + lambda * (K_tau @ V)
  return vec(Y)

precond_solve(L_K, L_H, z):
  Zp = reshape(z, n, r)
  solve K_tau Y H^T = Zp using triangular solves with L_K, L_H
  return vec(Y)
```

## Key References from futon6 corpus

- PlanetMath: conjugate gradient algorithm; method of conjugate gradients
- PlanetMath: Kronecker product; positive definite matrices
- PlanetMath: properties of tensor product
- physics.SE #27466: iterative solvers for large systems in physics
- physics.SE #27556: preconditioning for elliptic PDEs
