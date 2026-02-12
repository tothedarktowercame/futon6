# Problem 4 (n>=4): Proof Strategies

**Date:** 2026-02-12
**Updated:** 2026-02-12 (after Codex Strategy A results, commit cd8396f)
**Status:** Strategies A and B explored computationally. Both naive forms
blocked. Voiculescu's actual proof mechanism identified — reframes approach.

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

## Key discovery: Voiculescu's actual proof mechanism

**Codex Step 1 (plausible) revealed that Voiculescu does NOT use Dyson/heat-
flow monotonicity.** The free Stam inequality is proved via **conjugate
variables and L²-projection**:

1. Define conjugate variable J(X) with Φ*(X) = ‖J(X)‖²
2. Write J(X+Y) = E_{W*(X+Y)}[αJ(X:Y) + (1-α)J(Y:X)]
3. L²-contractivity of conditional expectation:
   Φ*(X+Y) = ‖J(X+Y)‖² ≤ ‖αJ(X:Y) + (1-α)J(Y:X)‖²
4. Freeness simplifies: J(X:Y) = J(X), J(Y:X) = J(Y), mixed ⟨·,·⟩ = 0
5. Expand: Φ*(X+Y) ≤ α²Φ*(X) + (1-α)²Φ*(Y)
6. Optimize α = Φ*(Y)/(Φ*(X)+Φ*(Y)) → Stam inequality

**This completely reframes the finite-n proof strategy.** The finite analog
should look for:
- A **finite conjugate variable** (root-force field S_i = Σ_{j≠i} 1/(λ_i-λ_j)
  is the natural candidate — it's already the "score function" of the log-gas)
- A **finite projection/conditional-expectation** surrogate for the ⊞_n operation
- **Finite orthogonality** or controlled cross-term bound

Free Brownian motion / Dyson BM enter elsewhere in Voiculescu's paper (de Bruijn
relations, entropy results) but are NOT the core mechanism for Stam itself.

---

## Strategy A: Finite Score Projection (Finitize Voiculescu's ACTUAL Proof)

**Probability of success: ~30%** (revised down from 35% — mechanism is
clearer but finitization obstacles are real)

### The idea (revised)

~~Dyson BM monotonicity~~ → **Conjugate-variable projection.**

The finite analog of Voiculescu's proof should:

1. **Define the finite score.** For p with roots λ_1 < ... < λ_n, the
   root-force field S_i(λ) = Σ_{j≠i} 1/(λ_i - λ_j) is the finite analog
   of J(X). Note: Φ_n(p) = Σ_i S_i² = ‖S‖².

2. **Relate the score of p ⊞_n q to projections of individual scores.**
   Since p ⊞_n q = E_U[det(xI - A - UBU*)], the roots of the convolution
   are "averaged" from the orbital integral. Need a projection identity:
   S(p ⊞_n q) relates to projections of S(random eigenvalues of A+UBU*).

3. **Prove a contraction inequality.** The L²-norm of the projected score
   should be bounded by a weighted combination of Φ_n(p) and Φ_n(q),
   giving the harmonic-mean form.

### What's been established

- Codex Step 2: Full Itô calculus for Φ_n under Dyson BM. Drift L(1/Φ_n)
  is NOT sign-definite (counterexample: n=3, λ=(-1,0,1), L(1/Φ_n) = -1/27).
  **Confirms Dyson monotonicity route is dead.**
- Codex Step 3: A+UBU* is NOT a stopped Dyson process. HCIZ gives
  E_U[f(eigs(A+UBU*))], but no Jensen direction without convexity.
- Codex Step 5: No closed SDE/PDE for roots of the averaged polynomial.
  The obstruction: Φ_n acts on roots of E_U[det(...)], not on sample
  eigenvalues before averaging.

### Open blockers (from Codex Step 6)

| Blocker | What's needed | Likely true? |
|---------|--------------|-------------|
| A: Finite de Bruijn | Entropy H_n with d/dt H_n(p_t) = -Φ_n(p_t) along finite heat flow | medium-high |
| B: Finite score/projection | Inequality playing role of free-score projection for ⊞_n | medium |
| C: Monotonicity step | d/dt[1/Φ_n(r_t) - 1/Φ_n(p_t) - 1/Φ_n(q_t)] ≥ 0 | medium (if A+B) |

### Key references

- Voiculescu (1998), Invent. Math. 132 — the actual proof mechanism
- Anderson, Guionnet, Zeitouni (2010), Ch. 4 — Dyson BM / Itô framework
- Biane (1997) — free/matrix Brownian bridge
- Marcus, Spielman, Srivastava (2015), arXiv:1504.00350 — ⊞_n definition

### Risks

- The finite score field S_i lives on eigenvalues, but ⊞_n operates on
  polynomial coefficients. Bridging these two representations is the core
  difficulty.
- No known finite analog of W*(X+Y)-conditional expectation. The projection
  would need to be constructed.
- The "freeness implies orthogonality" step (Step 4 above) has no obvious
  finite-n counterpart — Haar-unitary averaging is not the same as free
  independence.

---

## Strategy B: Induction via Differentiation

**Probability of success: ~15%** (revised down from 25% — naive induction
blocked by wrong-direction inequality)

### The idea

Marcus (2021) shows that differentiation interacts with ⊞_n. We proved:

    (p ⊞_n q)'/n = (p'/n) ⊞_{n-1} (q'/n)    [EXACT, algebraic proof]

Verified numerically for n=3..7, error < 10^{-13}.

If we can show 1/Phi_n(p) relates to 1/Phi_{n-1}(p'/n), the n≥4 inequality
follows by induction from the proved n=3 base case.

### What's been established

**Differentiation commutativity: PROVED** (commit 827f4c6)
- Algebraic proof via MSS coefficient weights
- Numerical verification for n=3..7

**Naive induction chain: BLOCKED** (commit 827f4c6)
- Step A: 1/Phi_n(p) ≥ c · 1/Phi_{n-1}(p'/n) — **FAILS 100%**
  (inequality goes wrong direction: 1/Phi_n < 1/Phi_{n-1}(p'/n) always)
- Step B: Apply induction hypothesis at degree n-1 — **holds** (100%)
- Step C: Reverse Step A for the convolution — **holds** (100%)

The induction chain breaks at the first link. The root-force field
concentrates MORE at the derivative's critical points than at the original
roots, so Phi_{n-1}(p'/n) < Phi_n(p) always.

### What might still work

- A **different functional** (not 1/Phi_n) that does chain correctly
  through differentiation. Codex Step 6 suggests investigating
  discriminant-based entropy candidates.
- An **inequality in the other direction**: if we can prove
  1/Phi_n(p ⊞_n q) ≥ 1/Phi_{n-1}((p ⊞_n q)'/n), and then use the
  exact commutativity + induction, the chain might close differently.
- A **double induction** combining differentiation with another structural
  reduction.

### Key references

- Marcus (2021), arXiv:2108.07054, Section 5.2
- Marcus, Spielman, Srivastava (2018), arXiv:1811.06382
- arXiv:2505.01705v2 (infinitesimal distributions, Proposition 5.4)

### Risks

- The wrong-direction inequality at Step A is structural, not numerical
  noise. The critical points of p are more closely spaced than the roots
  (by interlacing), which increases the Coulomb energy.
- Finding a replacement functional that chains correctly AND satisfies the
  target inequality is a design problem, not just a verification problem.

---

## Strategy C: Direct Algebraic (Sum-of-Squares Decomposition)

**Probability of success: ~20% for n=4; ~5% for all n** (revised up for
n=4 specifically based on Codex Step 6 assessment)

### The idea

Express the surplus Δ_n(p,q) = 1/Phi_n(p ⊞_n q) - 1/Phi_n(p) - 1/Phi_n(q)
as a manifestly non-negative expression. For n=3, the surplus factors as
(3/2) times a Titu's-lemma expression.

### What would be needed

1. For **n=4 specifically**: normalize by translation + scaling (WLOG a_1=b_1=0,
   leading coefficient 1), clear denominators using discriminants, seek an SOS
   or Lorentzian polynomial certificate via CAD/SDP.

2. For **all n**: structural insight from the n=4 computation that
   generalizes. The MSS weights w(n,i,j) = (n-i)!(n-j)!/(n!(n-k)!) have
   combinatorial structure that might yield a pattern.

### Key references

- MO 287724 + answer 287799 (finite free bilinearity/induction mechanics)
- Arizmendi, Perales (2018) (finite free cumulants)

### Risks

- Expressions become extremely complex for n≥4
- SOS decompositions for multivariate rational functions are computationally
  hard (SDP relaxation) and may not exist in the polynomial ring
- Even if found for n=4, generalizing requires structural insight

---

## Strategy D: Conditional Theorem + Explicit Lemmas (NEW)

**Probability of success: ~70% for conditional result**

### The idea (from Codex Step 6 synthesis)

Instead of a direct proof, prove a **conditional finite Stam theorem**:
"If Lemma A and Lemma B hold, then the finite Stam inequality holds for
all n." Then attack the lemmas separately.

### Structure

**Theorem (conditional):** Let p, q be monic real-rooted degree-n
polynomials. Suppose:
- (Lemma A) There exists a finite entropy functional H_n and a finite
  heat flow p_t such that d/dt H_n(p_t) = -Φ_n(p_t).
- (Lemma B) The finite score satisfies a projection inequality under ⊞_n:
  ‖S(p ⊞_n q)‖² ≤ (Φ_n(p)·Φ_n(q))/(Φ_n(p)+Φ_n(q)).

Then 1/Phi_n(p ⊞_n q) >= 1/Phi_n(p) + 1/Phi_n(q).

### What needs to happen

1. **Formalize the conditional theorem** with precise lemma statements.
2. **Prove the implication** (Lemma A + Lemma B ⟹ Stam) rigorously.
3. **Attack Lemma A**: Investigate Calogero-Moser root dynamics,
   discriminant-based entropy candidates, carré-du-champ calculations.
4. **Attack Lemma B**: Use HCIZ/Weingarten identities, matrix-model
   representation, orthogonal projection in polynomial coefficient
   coordinates.

### Fallback positions

- Conditional theorem published with explicit lemma dependencies
- n=4 exact proof (from Strategy C) as a separate concrete result
- Asymptotic liminf theorem recovering Voiculescu as n→∞
- Restricted-class proofs (one factor near-semicircular, or perturbative
  neighborhoods of equal-spacing configurations)

### Key references

- All references from Strategies A-C, plus:
- Codex results: `data/first-proof/problem4-stam-codex-results.jsonl`

### Risks

- The conditional theorem might be vacuously true (lemmas might be false)
- The lemma decomposition might not be sharp enough to be provable
  independently
- Publication value of a conditional result depends on the lemmas being
  "natural" and independently interesting

---

## Recommendation (updated)

**Two-track approach** (from Codex Step 6):

**Track A (conceptual):** Pursue Strategy D — the conditional theorem.
This is the most likely to produce a publishable result. The key step is
formalizing the finite analogs of Voiculescu's conjugate-variable projection
mechanism (Step 1 of Codex results). The conditional theorem isolates the
hard parts (Lemmas A, B) while establishing that the inequality WOULD
follow from natural structural properties.

**Track B (computational):** Pursue Strategy C for n=4 specifically. An
exact SOS/CAD certificate for Δ_4 ≥ 0 would be a concrete theorem and
would likely reveal the structural pattern needed for the general case.
After normalization (translate, scale), n=4 has only 2 free parameters
per polynomial — feasible for computer algebra.

**Strategy B (induction)** is deprioritized: the naive chain is blocked
(Step A fails 100%), and no replacement functional has been identified.
The exact differentiation commutativity identity is valuable but needs a
different framing to be useful.

**Codex next steps:** Rerun Step 4 (Jensen gap, timed out) with longer
timeout. Also consider a new step targeting the finite score projection
specifically — what is the finite analog of J(X:Y) = J(X) under freeness?

## Evidence trail

| Source | Commit | Key finding |
|--------|--------|-------------|
| Stress test | b6b5585 | 0 violations in 35K+ trials |
| Free Stam discovery | 9efda5b | Voiculescu 1998 = our inequality at n→∞ |
| Strategy B exploration | 827f4c6 | Differentiation commutes exactly; naive induction blocked |
| Strategy A exploration | 9e3dbb6 | Indefinite Hessian + superadditivity; convexity route dead |
| Codex Strategy A | cd8396f | Voiculescu uses projection not heat flow; four blockers identified |
