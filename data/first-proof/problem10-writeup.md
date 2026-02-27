# Problem 10: PCG for RKHS-Constrained Tensor CP Decomposition

## Problem Statement

Given the mode-k subproblem of RKHS-constrained CP decomposition with
missing data:

    [(Z x K_tau)^T S S^T (Z x K_tau) + lambda (I_r x K_tau)] vec(W)
        = (I_r x K_tau) vec(B),

where K_tau = K + tau I (PD kernel), S is a sparse selection operator
(q observed entries out of N = nM total), B = TZ, and n, r << q << N.

Explain how preconditioned conjugate gradient solves this without O(N)
computation per iteration.

## Solution

### 1. Implicit matrix-vector product

Write A = (Z x K_tau)^T SS^T (Z x K_tau) + lambda(I_r x K_tau). CG requires
only y = Ax, not A explicitly. Given x = vec(V), V in R^{n x r}:

(i) Compute U = K_tau V in O(n^2 r).
(ii) Evaluate at observed entries: for each (i_l, j_l), compute
     u_l = <U(i_l,:), Z(j_l,:)> using the Kronecker identity
     (Z x K_tau)vec(V) = vec(K_tau V Z^T). Total: O(qr).
(iii) Form sparse W' in R^{n x M} from {u_l}, compute W'Z in O(qr),
      then K_tau(W'Z) in O(n^2 r).
(iv) Add regularization lambda K_tau V in O(n^2 r).

**Total per matvec: O(n^2 r + qr), with no dependence on N.**

### 2. Right-hand side

b = (I_r x K_tau) vec(TZ). Since T has q nonzeros, TZ costs O(qr) and
K_tau(TZ) costs O(n^2 r). Total: O(qr + n^2 r).

### 3. Preconditioner

#### Original preconditioner (gap identified)

Whitening by K_tau^{-1/2} and replacing SS^T by (q/N)I gives the
Kronecker preconditioner P_old = H x K_tau with H = c Z^T Z + lambda I_r.

**Problem:** The system matrix has K_tau^2 in the signal term but P_old
has only K_tau. Even when D = cI exactly, the mismatch is
A - P = c Z^T Z x K_tau(K_tau - I). Numerical experiments (P10-C001)
confirm delta in [5.2, 22.7] — spectral equivalence fails.

#### Improved preconditioner (gap resolved)

Use K_tau^2 in the signal term to match the system:

    P_new = c (G x K_tau^2) + lambda (I_r x K_tau),

where G = (1/q) Z^T Z. This exactly matches A when D = cI (delta = 0
at machine precision). Efficiently invertible via eigendecomposition of
K_tau: block-diagonalizes to n blocks of r x r at cost O(n^2 r + nr^2).

Numerical evidence (P10-C002): delta < 1 for 18/22 configs under uniform
sampling (mean delta = 0.89), a 12.4x improvement over the original.

### 4. Convergence

Standard PCG: ||e_t||_A <= 2((sqrt(kappa)-1)/(sqrt(kappa)+1))^t ||e_0||_A,
with kappa = cond(P^{-1/2} A P^{-1/2}). If

    (1-delta)P <= A <= (1+delta)P  for delta in (0,1),

then kappa <= (1+delta)/(1-delta) and t = O(sqrt(kappa) log(1/eps)).

With P_new and uniform sampling (q >= C n log n), the exact-match property
at D = cI plus matrix concentration (Tropp 2011) gives
delta = O(sqrt(n log n / q)) < 1, so kappa = O(1) and PCG converges in
O(log(1/eps)) iterations.

Caveat: adversarial row-concentrated sampling gives delta in [1.4, 2.5]
even with P_new. The O(log(1/eps)) claim requires sampling regularity.

Necessity note:
- Counterexamples in `data/first-proof/problem10-necessity-counterexamples.md`.
- Gap analysis in `scripts/verify-p10-convergence-gap.py`.
- Resolution in `scripts/verify-p10-improved-preconditioner.py`.

### 5. Total complexity

Setup: O(n^3 + r^3 + sum_i n_i r^2 + qr + n^2 r).

Per PCG iteration: O(n^2 r + qr + nr^2).

Total: O(n^3 + t(n^2 r + qr)) for n >= r, compared to O(n^3 r^3 + Nnr) for
direct methods. The explicit dependence on N is eliminated. QED

## References

- J. Tropp, "Freedman's inequality for matrix martingales," Adv. Math.
  230(3) (2012), 761-779. Theorem 1.6.
- A. Rudi, D. Calandriello, L. Rosasco, "FALKON: An optimal large scale
  kernel method," NeurIPS 2017 (Nystrom alternative for large n).
