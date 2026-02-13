# Problem 4 CT Analysis (Task 3)

Date: 2026-02-13

## 1. Monoidal Structure

Let `RR_n` be the set of real-rooted monic degree-`n` polynomials (mod translation when centered).
Finite free additive convolution gives a commutative monoid:

`(RR_n, ⊞_n, He_0)`

with semigroup action by Hermite flow:

`H_t(p) := p ⊞_n He_t`, and `H_s(H_t(p)) = H_{s+t}(p)`.

## 2. Lax Monoidal Functionals

Define:

- `F_n(p) = 1/Φ_n(p)` (Stam target)
- `R_n(p) = 1/r_n(p)` where `r_n = Ψ_n/Φ_n` (new lead)

Empirically:

- `R_n(p ⊞_n q) >= R_n(p) + R_n(q)` (Stam-for-r)
- `F_n(p ⊞_n q) >= F_n(p) + F_n(q)` (Stam)

So both look like lax monoidal functors into `(R_+, +)`.

## 3. Natural-Transformation Lens

If there were a natural comparison `eta_n : R_n => F_n` compatible with `⊞_n`, then Stam-for-r could lift to Stam.

A practical algebraic form would be:

`F_n(p) >= C_n(p) * R_n(p)` with multiplicative/subadditive control of `C_n` under `⊞_n`.

This is the concrete transformation search problem: identify a correction factor `C_n` that is stable enough to preserve lax monoidality.

## 4. What Task 2 Says About Projection Morphisms

A projection-style score decomposition (finite Blachman route) would define a morphism
from `(S(p), S(q))` to `S(p ⊞ q)` with controlled Hilbert norm behavior.

Task 2 results reject simple candidates:

- global affine maps have low explanatory power (`R^2 ~ 0.01-0.02`)
- projection-like diagonal models have very large residuals
- per-sample scalar blend is also poor (`rel_err_mean` grows with `n`)

So any valid projection morphism must be nonlinear and root-configuration dependent.

## 5. Wiring-Diagram Formulation

Use two commuting diagrams:

1. Polynomial semigroup:
   `(p, q) --⊞--> c --H_t--> c_t`
   equals
   `(p, q) --(H_t,H_t)--> (p_t, q_t) --⊞--> c_{2t}`.

2. Functional side:
   `RR_n --(F_n, R_n, d/dt)--> R_+`.

The proof problem becomes: find a natural inequality transformer on the functional diagram
that respects the semigroup factor-2 time relation.

## 6. Concrete Algebraic Consequences to Test

1. `K_n` bridge search:
   test whether `F_n/R_n` admits uniform lower bounds on normalized cones (`a2=-1`).
2. Monotone correction:
   test whether `C_n(p) := F_n(p)/R_n(p)` is nondecreasing under `H_t`.
3. Two-object inequality:
   test whether `C_n(p ⊞ q)` is controlled by `C_n(p), C_n(q)` via min/harmonic mean.
4. Nonlinear projection ansatz:
   parameterize `S(p ⊞ q)` by low-rank nonlinear features of `(S(p), S(q), gaps(p), gaps(q))` and test Cauchy-Schwarz closure numerically.

## 7. Assessment

- CT is useful as an organizing language for proof obligations.
- It does not by itself close the estimate.
- Immediate value is in constraining the search to transformation laws compatible with
  monoidality and semigroup naturality.
