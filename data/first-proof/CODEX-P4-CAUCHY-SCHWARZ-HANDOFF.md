# Codex Handoff: Problem 4 — Cauchy-Schwarz Proof via Matching Average

**Date:** 2026-02-13
**From:** Claude (explore cycle)
**Priority:** HIGH — most promising direct proof route for Stam n≥4
**Prerequisite:** `CODEX-P4-STAM-FOR-R-HANDOFF.md` for numerical evidence

---

## Executive Summary

We discovered that the MSS convolution is literally the **expected characteristic
polynomial over random matchings**:

```
(p ⊞_n q)(x) = (1/n!) Σ_σ∈S_n  Π_k (x - (γ_k + δ_{σ(k)}))
```

Confirmed to machine precision for n=3,4. This opens a Cauchy-Schwarz proof path
for the Stam inequality, analogous to the classical Blachman (1965) proof.

**Key structural insight:** Stam is TIGHT (equality) for equi-spaced same-shape
polynomials. The mechanism is Pythagorean variance addition:
- For p = {-d_p, 0, d_p}, q = {-d_q, 0, d_q}: c = {-√(d²_p+d²_q), 0, √(d²_p+d²_q)}
- 1/Φ ∝ d², so 1/Φ(c) = 1/Φ(p) + 1/Φ(q) exactly
- General polynomials have SURPLUS from shape regularization

**Hypothesis:** Stam decomposes as:
```
1/Φ(p⊞q) = [variance addition term] + [shape regularization surplus ≥ 0]
```
The Cauchy-Schwarz inequality on the matching average should give exactly this.

---

## The Matching Model

### Confirmed Identity

```
(p ⊞_n q)(x) = E_σ[det(xI - (diag(γ) + P_σ diag(δ) P_σ^T))]
             = (1/n!) Σ_{σ∈S_n} Π_{k=1}^n (x - (γ_k + δ_{σ(k)}))
```

where P_σ is the permutation matrix. Script: `explore-p4-score-decomposition.py` Part 5.

### Consequence for Φ

For each matching σ, define the matched polynomial m_σ(x) = Π(x - r_k^σ) where
r_k^σ = γ_k + δ_{σ(k)}.

The MSS convolution c(x) = (1/n!) Σ_σ m_σ(x) is the AVERAGE polynomial.

Φ(c) is the Fisher information of this average. By the matching structure:
- Some matchings give coincident roots (r_k^σ = r_j^σ), with Φ = ∞
- But the average polynomial has well-separated roots
- Numerically: Φ(c) << E[Φ(matching)] (the averaging helps enormously)

### Score Identity

S_k = p''(γ_k) / (2·p'(γ_k))

This connects the score field to the polynomial. For the MSS average:
- c(x) = E_σ[m_σ(x)]
- c'(x) = E_σ[m_σ'(x)]
- c''(x) = E_σ[m_σ''(x)]

So at the roots ζ_k of c(x):
S_k(c) = c''(ζ_k) / (2·c'(ζ_k)) = E_σ[m_σ''(ζ_k)] / (2·E_σ[m_σ'(ζ_k)])

This is a RATIO of expectations, not an expectation of ratios. By Cauchy-Schwarz,
these are different, and the difference controls the Stam surplus.

---

## Task A: Prove the Pythagorean Property for Equi-Spaced Polynomials

### What to prove

For centered equi-spaced degree-n polynomials (roots at arithmetic progression),
the MSS convolution satisfies Stam with EQUALITY:

1/Φ(p ⊞_n q) = 1/Φ(p) + 1/Φ(q)

when p and q have the same gap ratios (same "shape", different scales).

### Approach

1. For equi-spaced roots γ_k = (k - (n+1)/2) · d, compute Φ(p) as a function
   of d and n. Should get Φ = c_n / d² for some constant c_n.
2. Show that the MSS convolution of two such polynomials with gaps d_p and d_q
   gives an equi-spaced polynomial with gap d_c = √(d_p² + d_q²).
3. Verify Φ(c) = c_n / (d_p² + d_q²) = c_n·d_p²·d_q² / (d_p²+d_q²)·(d_p²·d_q²)...
   wait, just check that 1/Φ(c) = 1/Φ(p) + 1/Φ(q).
4. Compute c_n explicitly for n = 3,4,5,6.

This should be a straightforward sympy computation. The key formula to verify:

```python
# For equi-spaced roots {a, a+d, a+2d, ..., a+(n-1)d}:
# Φ = ? (express in terms of n and d)
# After centering (a = -(n-1)d/2), Φ should be c_n/d² for some c_n.
```

### Deliverable

Sympy script proving the Pythagorean property for n=3,4,5,6. Include explicit
computation of c_n and verification that MSS of equi-spaced = equi-spaced.

---

## Task B: Cauchy-Schwarz on the Matching Average

### The Core Proof Attempt

**Claim (to verify):** The Stam inequality follows from applying Cauchy-Schwarz
to the matching-average representation of the score.

At roots ζ_k of c = p ⊞ q:

```
S_k(c) = c''(ζ_k) / (2·c'(ζ_k))
       = E_σ[m_σ''(ζ_k)] / (2·E_σ[m_σ'(ζ_k)])
```

Now, Φ(c) = Σ_k S_k(c)² = Σ_k [E_σ[m_σ''(ζ_k)] / (2·E_σ[m_σ'(ζ_k)])]²

By Cauchy-Schwarz (or Jensen on the convex function x²):

```
[E[f]/E[g]]² ≤ E[f²/g²] · E[g²]/E[g]²    (?)
```

This isn't standard C-S. The right formulation might be:

**Attempt 1:** Write S_k(c) = Σ_σ w_σ · S_k^σ(ζ) where w_σ = m_σ'(ζ_k) / c'(ζ_k).
Then S_k(c) is a WEIGHTED average of "score-like" quantities. Apply Cauchy-Schwarz:
Φ(c) = ||Σ w_σ v_σ||² ≤ (Σ w_σ ||v_σ||²) if weights are appropriate.

**Attempt 2:** Use the Cauchy-Schwarz inequality on sums directly:
(Σ a_i b_i)² ≤ (Σ a_i²)(Σ b_i²) applied to the numerator/denominator.

**Attempt 3:** The classical Blachman proof uses:
1/J(X+Y) = min_a [a²/J(X) + (1-a)²/J(Y)]  (variational characterization)
Find the finite analog.

### What to do

1. For n=3, write out the matching average EXPLICITLY (6 matchings)
2. Express S_k(c) as a function of the 6 matching scores
3. Try each Cauchy-Schwarz formulation above
4. Check: does the bound give exactly Stam? Or something weaker?
5. If it works for n=3, check n=4 (24 matchings)

### Key obstacle to watch for

The matching average is over COEFFICIENTS (characteristic polynomials), not over
ROOTS. The roots of the average ≠ average of roots. So S(c) at the roots of c
is NOT the average of S^σ at those same points. The nonlinearity of the
root-finding step is the main challenge.

### Deliverable

Python + sympy script for n=3 that:
- Writes out the explicit Cauchy-Schwarz argument
- Identifies whether it gives Stam exactly or with slack
- Reports what goes wrong if it fails

---

## Task C: Category Theory — Coend Structure of MSS Convolution

### The Categorical View

The MSS convolution is a **coend** (categorical averaging):

```
p ⊞_n q = ∫^{σ∈S_n} m_σ = coend of the functor S_n → Poly_n
```

where the functor sends σ to the matched polynomial Π(x - (γ_k + δ_{σ(k)})).

The Stam inequality says: 1/Φ is a **lax monoidal functor** from (RR_n, ⊞_n) to (ℝ₊, +).

### Relevant CT Framework

From futon5/CT DSL:
- **Monoidal category** (RR_n, ⊞_n, He_n) with He_n as unit
- **Lax monoidal functor** 1/Φ: preserves monoid structure up to inequality
- **Natural transformation** from 1/r to 1/Φ (both are lax monoidal — can we
  factor one through the other?)

The coend representation suggests:
- **Enriched Cauchy-Schwarz:** In enriched category theory, there are
  "weighted colimit" inequalities. The Stam inequality might be an instance.
- **Day convolution:** The MSS convolution resembles Day convolution in the
  functor category. Stam as a property of Day convolution?

### What to investigate

1. Formalize (RR_n, ⊞_n) as a symmetric monoidal category
2. Express 1/Φ and 1/r as lax monoidal functors
3. Check if there's a natural transformation 1/r → 1/Φ or vice versa
4. Look for "enriched Cauchy-Schwarz" or "weighted colimit" theorems
   that could give Stam from the coend structure
5. Concretely: does the CT viewpoint suggest new ALGEBRAIC inequalities to test?

### Practical note

Don't build infrastructure. Use CT as a LENS to find algebraic proof strategies.
The value is in new inequalities/identities to test, not in abstract machinery.

### Deliverable

Markdown document describing:
- The categorical formalization
- Any new algebraic consequences (inequalities to test)
- Whether the coend perspective suggests a proof mechanism

---

## Task D: Explicit n=3 Stam via Matching Average

### What to prove

For n=3 with roots p = {γ₁, γ₂, γ₃} and q = {δ₁, δ₂, δ₃}:

```
1/Φ(p ⊞₃ q) ≥ 1/Φ(p) + 1/Φ(q)
```

where p ⊞₃ q = (1/6) Σ_{σ∈S₃} Π_k(x - (γ_k + δ_{σ(k)})).

### Approach

1. Center both polynomials (γ₁+γ₂+γ₃ = 0, δ₁+δ₂+δ₃ = 0)
2. Parameterize by gaps: d₁₂ = γ₂-γ₁, d₂₃ = γ₃-γ₂ for p; similarly for q
3. Express MSS convolution coefficients explicitly
4. Find roots of p⊞q symbolically (cubic formula or Cardano)
5. Express Φ(p⊞q) as rational function of the 4 gap parameters
6. Prove 1/Φ(p⊞q) ≥ 1/Φ(p) + 1/Φ(q) as polynomial inequality

### Simplification for n=3

The MSS convolution coefficients for n=3:
```
c₁ = e₁(p) + e₁(q)                    [sum of roots, = 0 if centered]
c₂ = e₂(p) + (2/3)e₁(p)e₁(q) + e₂(q) [= e₂(p) + e₂(q) if centered]
c₃ = e₃(p) + (1/3)e₂(p)e₁(q) + (1/3)e₁(p)e₂(q) + e₃(q) [= e₃(p) + e₃(q)]
```

So for centered polynomials: e_k(p⊞q) = e_k(p) + e_k(q) for k=1,2,3.
This is REMARKABLE — centered MSS convolution = coefficient-wise addition!

This means: for centered p and q, the MSS convolution simply ADDS the elementary
symmetric functions. This is the finite analog of "convolution adds cumulants."

### Consequence

If e_k(c) = e_k(p) + e_k(q), then proving Stam reduces to:

```
Φ(roots(e₂_p + e₂_q, e₃_p + e₃_q)) ≤ harm(Φ(roots(e₂_p, e₃_p)), Φ(roots(e₂_q, e₃_q)))
```

where harm is harmonic mean and roots(e₂, e₃) denotes the roots of x³ + e₂x - e₃
(centered monic cubic).

For centered cubics, the roots are determined by (e₂, e₃) alone. So Stam for n=3
is a 4-variable polynomial inequality (e₂_p, e₃_p, e₂_q, e₃_q) with constraint
that both cubics are real-rooted (discriminant ≥ 0).

### Deliverable

Sympy script that:
- Verifies e_k(p⊞q) = e_k(p) + e_k(q) for centered polynomials
- Expresses Φ as function of (e₂, e₃) for centered cubics
- Sets up the Stam inequality as a constrained polynomial optimization
- Attempts SOS or Sturm-chain proof

---

## Priority

**D > B > A > C**

Task D (explicit n=3 via coefficient addition) is the most concrete: for centered
polynomials, MSS just adds coefficients, reducing Stam to a 4-variable inequality.
This should be tractable by SOS.

Task B (Cauchy-Schwarz on matching average) is the path to general n.

Task A (Pythagorean property) establishes the equality case.

Task C (CT) is for structural insight.

---

## Scripts Reference

| Script | Role |
|--------|------|
| `explore-p4-score-decomposition.py` | Discoveries: matching model, Pythagorean, score identity |
| `prove-p4-r-submultiplicative.py` | "Stam for r" verification |
| `prove-p4-stam-from-r.py` | Φ, Ψ, W inequality landscape |
| `prove-p4-hprime-leq-zero.py` | Hermite structure, r non-increasing |
