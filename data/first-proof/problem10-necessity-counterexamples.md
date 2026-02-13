# Problem 10: Necessity Counterexamples (Plausibility Checks)

This note records small explicit counterexamples showing why the main
assumptions in Problem 10 are necessary.

These are not claimed as exhaustive impossibility theorems; they are
constructive sanity checks demonstrating failure modes when assumptions are
removed.

## CE-1: Dropping `K_tau` PD can destroy SPD of `A_tau`

Assumption being tested:
- `tau > 0` (or `K` already PD), so `K_tau = K + tau I` is PD.

Toy setup:
- `n = 2`, `r = 1`, `M = 1`.
- `Z = [1]`.
- `D = I_2`.
- `K = diag(1,0)` (PSD but singular), and set `tau = 0`.
- `lambda = 1`.

Then

`A_tau = (Z x K)^T D (Z x K) + lambda (I_r x K) = K^2 + K = diag(2,0)`.

So `A_tau` is singular, not SPD. The linear system is not uniquely solvable in
the SPD sense, and standard PCG assumptions are violated.

Conclusion:
- The `K_tau` positive-definiteness assumption is necessary for the SPD claim.

## CE-2: Dropping sampling regularity can make preconditioning ineffective

Assumption being tested:
- Sampling/coherence regularity yielding
  `(1-delta)P <= A_tau <= (1+delta)P` with `delta < 1` bounded away from 1.

Toy setup (keep SPD, break fast-rate conditions):
- `n = 1`, `r = 2`, `M = 2`, so `N = 2`.
- `K_tau = [1]` (PD), `lambda = 10^{-3}`.
- `Z = I_2`.
- One observed entry only: `D = diag(1,0)`, so `q = 1`, `c = q/N = 1/2`.

Compute:

`A_tau = Z^T D Z + lambda I_2 = diag(1+lambda, lambda)`.

`P = c Z^T Z + lambda I_2 = (1/2 + lambda) I_2`.

Hence the eigenvalues of `P^{-1} A_tau` are

- `mu_1 = (1+lambda)/(1/2+lambda)`,
- `mu_2 = lambda/(1/2+lambda)`.

With `lambda = 10^{-3}`:
- `mu_1 ~= 1.998`,
- `mu_2 ~= 0.001996`,
- `kappa(P^{-1}A_tau) ~= 1001`.

So the system is SPD, but conditioning is poor and the "fast PCG" regime is
not justified. This is exactly what the `delta`-control assumption is intended
to prevent.

Conclusion:
- Sampling regularity is necessary for the bounded-condition-number claim.

## Interpretation for P10 status

- These counterexamples support why Problem 10 remains
  "closed under stated assumptions" rather than unqualifiedly solved.
- They justify retaining `P10-G2` as `partial` until explicit sufficient
  sampling/coherence bounds are pinned down.
