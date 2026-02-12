# Problem 4 (n>=4): Proof Strategies

**Date:** 2026-02-12
**Status:** Three strategies outlined. Strategy A assigned to Codex.

---

## Background

The inequality 1/Phi_n(p ⊞_n q) >= 1/Phi_n(p) + 1/Phi_n(q) is the
**finite-dimensional analog of the free Stam inequality**, proved by
Voiculescu (1998) for measures under free convolution:

    1/Φ*(μ ⊞ ν) >= 1/Φ*(μ) + 1/Φ*(ν)

with equality iff μ, ν are semicircular (the free Gaussian).

We have proved the finite case for n=2 (equality) and n=3 (Cauchy-Schwarz).
For n>=4, we have 35K+ numerical tests with 0 violations. The question is
whether a proof exists.

---

## Strategy A: Dyson Brownian Motion (Finitize Voiculescu's Proof)

**Probability of success: ~35%**
**Assigned to: Codex**

### The idea

Voiculescu's proof of the free Stam inequality uses **free stochastic
calculus**: the free Brownian motion X_t = X_0 + S_t (where S_t is a
free semicircular process) satisfies d/dt[1/Φ*(X_t)] = something >= 0
at t=0, which gives the inequality via a convexity/monotonicity argument.

The finite-n analog of free Brownian motion is **Dyson Brownian motion**:
the eigenvalue process of A + √t · GUE, where the eigenvalues evolve as

    dλ_i = dB_i/√n + (1/n) Σ_{j≠i} dt/(λ_i - λ_j)

The key steps to finitize:

1. **Define f(t) = E[1/Phi_n(eigenvalues of A + √t · GUE)]**.
   At t=0, f(0) = 1/Phi_n(A) (deterministic).
   As t → ∞, f(t) → ∞ (GUE dominates, roots spread).

2. **Compute df/dt using Ito's formula on eigenvalues.**
   The Dyson process has known drift and diffusion. Phi_n is a smooth
   function of eigenvalues (away from collisions). The Ito calculation
   should give df/dt in terms of derivatives of 1/Phi_n.

3. **Show that at t=0 with initial condition A+QBQ* (Q Haar unitary),
   the expected 1/Phi_n satisfies the superadditivity bound.**

### What Codex should investigate

- Ito calculus for Phi_n along Dyson Brownian motion
- Whether E[1/Phi_n(A + √t · H)] is convex or concave in t
- The analog of Voiculescu's free entropy power inequality at finite n
- Whether the Dyson process preserves the structure needed for the bound

### Key references

- Voiculescu (1998), Invent. Math. 132 (free Stam via free stochastic calculus)
- Anderson, Guionnet, Zeitouni (2010), "An Introduction to Random Matrices"
  (Dyson Brownian motion, Ch. 4)
- Biane (1997), "Free Brownian motion, free stochastic calculus and
  random matrices" (bridges free and matrix-level)

### Risks

- The Ito calculation may produce terms that don't have a definite sign
- The expectation E[1/Phi_n] may not behave as nicely as 1/E[Phi_n]
  (Jensen's inequality goes the wrong way for concave functions)
- Finite-n corrections to the free stochastic calculus may be hard to control

---

## Strategy B: Induction via Differentiation

**Probability of success: ~25%**

### The idea

Marcus (2021) shows that differentiation interacts with ⊞_n: bounds on
roots of p ⊞_n q at degree n can be obtained by reducing to degree n-1
via differentiation. Rolle's theorem gives interlacing between roots of p
and roots of p'.

If we can show:

    1/Phi_n(p) relates to 1/Phi_{n-1}(p'/n)  (*)

then the n>=4 inequality follows by induction from the proved n=3 base case,
provided the ⊞_n → ⊞_{n-1} compatibility under differentiation is strong
enough.

### What needs to happen

1. **Establish (*) quantitatively.** Phi_n involves all root gaps; Phi_{n-1}
   of the derivative involves gaps between the n-1 critical points. These
   are related by interlacing but not identical. Need: either an inequality
   1/Phi_n >= c · 1/Phi_{n-1}(p'/n) or an exact identity.

2. **Show ⊞_n commutes with differentiation.** Marcus's result is that
   (p ⊞_n q)' relates to p' ⊞_{n-1} q' up to normalization. Need the
   precise statement and whether it's exact or approximate.

3. **Chain the induction.** If the above two pieces work, then:
   1/Phi_n(p ⊞_n q) >= c · 1/Phi_{n-1}((p ⊞_n q)'/n)
                       = c · 1/Phi_{n-1}(p'/n ⊞_{n-1} q'/n)  [Step 2]
                       >= c · [1/Phi_{n-1}(p'/n) + 1/Phi_{n-1}(q'/n)]  [induction]
                       >= 1/Phi_n(p) + 1/Phi_n(q)  [reverse of Step 1]

### Key references

- Marcus (2021), arXiv:2108.07054, Section 5.2
- Marcus, Spielman, Srivastava (2018), arXiv:1811.06382 (barrier method,
  differential operators on finite free convolutions)
- arXiv:2505.01705v2 (infinitesimal distributions, Proposition 5.4)

### Risks

- The relationship (*) may go the wrong direction (inequality flips)
- The constant c may depend on n, breaking the induction
- Phi_n → Phi_{n-1} under differentiation may not have a clean relationship
  (the interlacing of roots and critical points is well-understood for
  individual polynomials but less so for the energy functional)

---

## Strategy C: Direct Algebraic (Sum-of-Squares Decomposition)

**Probability of success: ~15%**

### The idea

Express the surplus

    S = 1/Phi_n(p ⊞_n q) - 1/Phi_n(p) - 1/Phi_n(q)

as a manifestly non-negative expression (sum of squares, or a sum of
terms each shown positive by AM-GM / Cauchy-Schwarz). This is how the
n=3 proof works: the surplus factors as (3/2) times a Titu's-lemma
expression that is >= 0.

### What would be needed

1. **Write 1/Phi_n in terms of coefficients.** For n=3, we found
   1/Phi_3 = -2a_2/9 - 3a_3^2/(2a_2^2) via the identity Phi_3·disc = 18a_2^2.
   For n>=4, no such clean formula exists (Phi_4·disc depends on a_3, a_4).
   Would need a different decomposition.

2. **Use the MSS coefficient formula** c_k = Σ w(n,i,j) a_i b_j to
   express S in terms of a_i, b_j. Then find a non-negative decomposition
   of S.

3. **Possibly use computer algebra** (SOS relaxation, Positivstellensatz)
   to find the decomposition symbolically for specific small n (n=4, n=5),
   then generalize.

### Key references

- MO 287724 + answer 287799 (finite free bilinearity/induction mechanics)
- Arizmendi, Perales (2018) (finite free cumulants)

### Risks

- The n=3 proof worked because Phi_3·disc was constant. At n>=4, there's
  no analogous simplification. The expressions become extremely complex.
- SOS decompositions for multivariate rational functions are computationally
  hard (SDP relaxation) and may not exist in the polynomial ring.
- Even if found for n=4, generalizing to all n requires structural insight
  that the computation alone doesn't provide.

---

## Recommendation

**Start with Strategy A** (Dyson Brownian Motion). Reasons:

1. It's directly finitizing a known proof (Voiculescu 1998), so the proof
   template exists.
2. Dyson Brownian motion and Ito calculus on eigenvalues is well-developed
   machinery (AGZ 2010, Ch. 4).
3. If the calculation works, it gives the result for ALL n simultaneously,
   not just n=4.
4. The Ito approach naturally produces monotonicity/convexity results,
   which is what we need.

Strategy B is the fallback — it builds on Marcus's existing work and our
proved n=3 base case, but requires establishing a new Phi_n/Phi_{n-1}
relationship that may not be clean.

Strategy C is the least promising for n>=4 but might work for n=4 alone
if we need a specific case.
