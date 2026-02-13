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
4. Necessity checks are explicit: dropping (2) can break SPD; dropping
   sampling regularity can preserve SPD but destroy fast conditioning. See
   `data/first-proof/problem10-necessity-counterexamples.md`.

Then the solved system is

    A_tau x = b_tau,
    A_tau = (Z x K_tau)^T D (Z x K_tau) + lambda (I_r x K_tau),
    x = vec(W),
    b_tau = (I_r x K_tau) vec(B).

Under these assumptions A_tau is SPD, so PCG applies.

## Solution

### 1. Why naive direct methods fail

A_tau is an (nr) x (nr) system. Dense direct factorization costs
$O((nr)^3)$ = $O(n^3 r^3)$.

A naive explicit route also materializes Phi = Z x K_tau in R^{N x nr},
which costs $O(N n r)$ memory/work before factorization. This is the
$N$-dependent bottleneck we avoid with matrix-free PCG.

### 2. Implicit matrix-vector product in $O(n^2 r + q r)$

CG needs only y = A_tau x, not A_tau explicitly.

Given x = vec(V), $V \in R$^{n x r}:

1. U = K_tau V. Cost $O(n^2 r)$.
2. Forward sampled action (only observed entries):

       (Z x K_tau) vec(V) = vec(K_tau V Z^T).

   For each observed coordinate (i_l, j_l),

       u_l = <U[i_l, :], Z[j_l, :]>.

   Total $O(q r)$.
3. Form sparse W' in R^{n x M} from u_l. Let s = nnz(W') <= q.
4. Adjoint sampled action:

       (Z^T x K_tau) vec(W') = vec(K_tau W' Z).

   Compute W' Z in $O(s r)$ <= $O(q r)$, then left-multiply by K_tau in $O(n^2 r)$.
5. Add regularization term lambda vec(K_tau V), cost $O(n^2 r)$.

Total per matvec:

    O(n^2 r + q r),

with no $O(N)$ term.

### 3. Right-hand side

B = T Z with T sparse (q nonzeros):

1. T Z: $O(q r)$
2. K_tau B: $O(n^2 r)$

So b_tau = (I_r x K_tau) vec(B) is formed in $O(q r + n^2 r)$.

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

cost $O(sum_i n_i r^2)$.

Preconditioner apply:

    P^{-1} = H^{-1} x K_tau^{-1},

implemented by solving K_tau Y H^T = Z' after reshape.
Per application cost is $O(n^2 r + n r^2)$ (often simplified to $O(n^2 r)$
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
delta is bounded away from 1. (No unsupported closed-form t = $O(r sqrt(n/q))$
claim is needed.)

**Sufficient conditions for bounded delta.** The spectral equivalence
(1-delta)P <= A_tau <= (1+delta)P holds with delta bounded away from 1 when
the sampling pattern satisfies a restricted isometry-type condition:
the restricted isometry holds for the column space of Z ⊗ K_tau^{1/2},
i.e. (Z ⊗ K_tau^{1/2})^T (D - cI) (Z ⊗ K_tau^{1/2}) is small in
operator norm relative to lambda. Under standard leverage/coherence
assumptions and sufficient sampling scaling with model dimension,
concentration yields delta bounded away from 1 with high probability. Under this regime, kappa = $O(1)$ and PCG converges in
$O(log(1/eps))$ iterations.

### 5a. Necessity checks (counterexamples)

Two explicit toy counterexamples are recorded in:

- `data/first-proof/problem10-necessity-counterexamples.md`

Summary:

1. If `K_tau` is not PD (e.g., `tau = 0` with singular `K`), `A_tau` can lose
   SPD, so the standard PCG guarantee does not apply.
2. If sampling regularity fails, `A_tau` may remain SPD but
   `kappa(P^{-1}A_tau)` can become large, invalidating the fast-convergence
   interpretation.

### 6. Complexity summary

Setup per ALS outer step:

1. Cholesky(K_tau): $O(n^3)$
2. Z^T Z via Hadamard Grams: $O(sum_i n_i r^2)$
3. Cholesky(H): $O(r^3)$
4. RHS: $O(q r + n^2 r)$

Per PCG iteration:

1. Matvec: $O(n^2 r + q r)$
2. Preconditioner apply: $O(n^2 r + n r^2)$

Total:

    O(n^3 + r^3 + sum_i n_i r^2 + q r + n^2 r
      + t (n^2 r + q r + n r^2)).

In the common regime n >= r, this simplifies to

    O(n^3 + t (n^2 r + q r)),

with dependence on q (observed entries) rather than N (all entries).

**Regime caveat.** When n is large enough that the $O(n^3)$ Cholesky setup
dominates (i.e., n^3 > t(n^2 r + q r)), the per-ALS-step cost is effectively
$O(n^3)$. In this regime, low-rank kernel approximations (e.g., Nystrom
approximation with rank p << n, reducing the kernel factorization to $O(n p^2)$)
or iterative inner solves (conjugate gradient on K_tau y = z, cost $O(n^2)$
per inner iteration) can replace the exact Cholesky, reducing the setup to
$O(n p^2 + t(n p r + q r))$. This is a well-known practical optimization
(see Rudi-Calandriello-Rosasco 2017) and is compatible with the PCG framework
as presented.

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

## 8. Gap Ledger (Requirement Compliant)

Status labels follow `proved | partial | open | false | numerically verified`.

| ID | Item | Status | Why | Evidence artifact |
|---|---|---|---|---|
| P10-G1 | Node-level external verifier run integrity | proved | Supported-model rerun completed with parseable outputs for all nodes (`15/15`; `8 verified`, `7 plausible`, `0 gap`, `0 error`). | `data/first-proof/problem10-codex-results.jsonl` |
| P10-G2 | Convergence-rate strength under sampling assumptions | partial | Necessity counterexamples now show why assumptions matter; sufficiency bounds are still conditional and not fully tightened. | Section 5; `data/first-proof/problem10-necessity-counterexamples.md` |
| P10-G3 | Explicit cycle record and named-gap discipline | proved | This section and Section 9 provide named gaps and cycle metadata. | This file (Sections 8-9) |

Interpretation:
- The mathematical writeup is **conditionally closed** under stated assumptions.
- The process-integrity blocker (`P10-G1`) is resolved; remaining work is substantive convergence-strength evidence (`P10-G2`).

## 9. Cycle Record (2026-02-13 Remediation)

```text
cycle_id: P10-remediation-2026-02-13
problem_id: P10
blocker_id: P10-G1
hypothesis: Rerunning node-level verifier with a supported model restores valid machine-readable verification artifacts.
stop_conditions: either (a) results regenerate with parseable JSON outputs, or (b) runtime/tooling failure is explicitly recorded with reproducible stderr evidence.
execution_artifact_paths:
  - data/first-proof/problem10-codex-prompts.jsonl
  - data/first-proof/problem10-codex-results.jsonl
validation_artifact_paths:
  - data/first-proof/problem10-codex-results.jsonl
result_status: completed
status_change: P10-G1 moved from false to proved via supported-model rerun with fully parseable outputs.
validation_summary: 15/15 parseable; 8 verified; 7 plausible; 0 gap; 0 error.
failure_point: none observed in this remediation cycle; unresolved risk remains convergence-strength assumptions (P10-G2).
next_blocker: P10-G2
commit_hash: pending
```
