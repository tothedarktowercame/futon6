# Codex Handoff: Problem 4 — "Stam for r" and Algebraic Proof Paths

**Date:** 2026-02-13
**From:** Claude (monograph author)
**Priority:** HIGH — strongest new lead for closing Stam for n≥4
**Prerequisite:** Read `CODEX-P4-DEEPDIVE-HANDOFF.md` for context on what's proved/killed

---

## Executive Summary

We discovered three universal numerical inequalities for the MSS convolution ⊞_n,
each holding at 0 violations across 60,000+ adversarial tests for n=3..10:

| Inequality | Meaning | Tests |
|------------|---------|-------|
| r(p⊞q) ≤ min(r(p), r(q)) | r non-increasing | 0/40,000 |
| r(p⊞q) ≤ r(p)·r(q)/(r(p)+r(q)) | "Stam for r" (harmonic mean) | 0/18,000 |
| Ψ(p⊞q) ≤ min(Ψ(p), Ψ(q)) | Ψ non-increasing | 0/18,000 |

Where:
- S_k = Σ_{j≠k} 1/(γ_k - γ_j) is the **score field** (Coulomb velocity)
- Φ = Σ_k S_k² is the **Fisher information analog**
- T_{kj} = (S_k - S_j)/(γ_k - γ_j) is the **discrete score derivative**
- Ψ = Σ_{k<j} T_{kj}² and r = Ψ/Φ is the **relative SOS decay rate**
- De Bruijn identity: dΦ/dt = -2Ψ along Hermite flow

The actual Stam inequality 1/Φ(p⊞q) ≥ 1/Φ(p) + 1/Φ(q) also holds at 0/30,000.

**The challenge:** We tried to prove Stam via semigroup monotonicity
(h(t) = log-surplus along the flow), but this is blocked by a factor-of-2
gap from the semigroup identity p_t ⊞ q_t = (p⊞q)_{2t}. Specifically,
h'(t) = 4r_c(2t) - 2·wavg, and "Stam for r" only gives 4r_c ≤ 4·harm(r_p,r_q),
while we need 4r_c ≤ 2·wavg.

**This handoff asks Codex to:**
1. Prove "Stam for r" algebraically (start with n=3, then n=4)
2. Investigate score decomposition under ⊞_n (Cauchy-Schwarz route)
3. Test whether category theory (monoidal functors) can organize the proof

---

## DEAD ENDS — DO NOT RE-EXPLORE

Everything in `CODEX-P4-DEEPDIVE-HANDOFF.md` plus:

5. **h'(t) ≤ 0 universally:** FAILS for ALL n, including n≥6 (earlier 0% at n=6
   was a sampling artifact from limited scale range). With adversarial configs
   (scale ratio 10^{-2} to 10^2), worst 2r_c/wavg > 1.6 even at n=6.
   Script: `prove-p4-hprime-leq-zero.py`.

6. **Stronger r bound (harm/2):** r(p⊞q) ≤ harm(r_p,r_q)/2 FAILS at 50-70%.
   The factor-of-2 cannot be obtained from a tighter r inequality.
   Script: `prove-p4-stam-from-r.py` TEST 5.

7. **Stam for W = Ψ/Φ²:** Badly fails. W is not well-behaved under ⊞.
   Script: `prove-p4-stam-from-r.py` TEST 3.

---

## Task 1: Symbolic Proof of "Stam for r" for n=3

### What to prove

For degree-3 polynomials with roots γ₁ < γ₂ < γ₃:

r(p) = Ψ(p)/Φ(p) where Φ = Σ S_k², Ψ = Σ_{k<j} T_{kj}²

Prove: r(p ⊞₃ q) ≤ r(p)·r(q)/(r(p)+r(q))

### What we already know (n=3 symbolic)

For roots {0, d, d+e}, the shape function d²·r is:

```
d²·r = (2t⁶ + 6t⁵ + 3t⁴ - 4t³ + 3t² + 6t + 2) / (t²·(t⁴ + 3t³ + 4t² + 3t + 1))
```

where t = e/d is the gap ratio. This has:
- Minimum 1.302 at t ≈ 1.363
- Value 3/2 at t = 1 (equispaced)
- Limit 2 as t → ∞
- Limit ∞ as t → 0 (two roots collide)

### Approach

1. Express the MSS convolution for n=3 explicitly in terms of the elementary
   symmetric polynomials of the two root sets
2. Compute r(p⊞₃q) symbolically (will be a rational function of 6 variables:
   γ₁,γ₂,γ₃ for p and δ₁,δ₂,δ₃ for q)
3. By translation/scale invariance, reduce to 4 effective variables
4. Show the inequality via SOS or Sturm chain or other certified method

### Key reference code

```python
# MSS convolution for n=3, coefficient k:
# c_k = Σ_{i+j=k} w_{ij} · a_i · b_j
# w_{ij} = C(n-i,n-k) * C(n-j,n-k) / C(n,n-k)  [binomial]

# For n=3:
# c_1 = (a_1 + b_1) / 1       [= -(sum of roots of p + sum of roots of q)/3... etc]
# c_2 = (a_2 + a_1·b_1·(2/3) + b_2) / 1
# c_3 = (a_3 + a_2·b_1·(1/3) + a_1·b_2·(1/3) + b_3) / 1
```

Verify these weight formulas against the code in `prove-p4-r-submultiplicative.py`.

### Deliverable

A sympy script that either:
- Proves r(p⊞₃q) ≤ harm(r_p, r_q) via certified SOS decomposition, OR
- Identifies the exact algebraic obstruction if SOS doesn't close

---

## Task 2: Score Decomposition under ⊞_n (Cauchy-Schwarz Route)

### Motivation

The classical Stam inequality proof (Blachman 1965) for continuous X+Y uses:

1. **Score projection:** s_{X+Y}(z) = E[s_X(X) | X+Y=z]
2. **Cauchy-Schwarz:** J(X+Y) = E[s_{X+Y}²] ≤ ... (with appropriate weights)
3. This gives 1/J(X+Y) ≥ 1/J(X) + 1/J(Y) directly

For the MSS convolution, we need a finite analog. The question: is there a
"score projection" formula expressing S_k(p⊞q) in terms of S_k(p) and S_k(q)?

### What to investigate

1. **Numerical decomposition:** For random p, q at n=3,4,5, compute:
   - S(p⊞q) — the score field of the convolution
   - Is there a matrix M such that S(p⊞q) = M·S(p) + (I-M)·S(q)?
   - Is there a nonlinear relationship S(p⊞q) = f(S(p), S(q), roots)?

2. **Cauchy matrix structure:** The score is S_k = Σ_j C_{kj} where C is the
   Cauchy matrix 1/(γ_k - γ_j). The MSS convolution mixes the roots.
   Is there a formula relating the Cauchy matrices of p, q, and p⊞q?

3. **If a decomposition exists:** Apply Cauchy-Schwarz to get
   Φ(p⊞q) = Σ S_k(p⊞q)² ≤ ... → Stam.
   The key: we need the decomposition to give the RIGHT constant (factor of 2).

### What we already know

- Resolvent score decomposition (S = S^A + S^B via partial fractions)
  was KILLED — no orthogonality, inner product has random sign.
  (See `verify-p4-new-approaches.py`, Approach F.)
- But that was a DIFFERENT decomposition. The Cauchy-Schwarz route
  uses a CONDITIONAL EXPECTATION / PROJECTION, not a direct sum.

### Key question

The MSS convolution is the EXPECTED characteristic polynomial:
(p ⊞_n q)(x) = E_σ[Π_k (x - γ_k + δ_{σ(k)}) / ... ]  (over random matchings σ)

Does this expectation structure give a natural conditional-expectation
decomposition of the score?

### Deliverable

A Python script that:
- Computes the score decomposition numerically for n=3,4,5
- Reports whether a linear/affine decomposition exists
- If yes: verifies Cauchy-Schwarz gives Stam
- If no: characterizes the nonlinearity and whether it helps or hurts

---

## Task 3: Category Theory Organization

### Motivation

We haven't used the CT machinery from futon5 yet. The algebraic structures
here have a natural categorical interpretation:

1. **(RR_n, ⊞_n)** — real-rooted degree-n polynomials form a commutative monoid
2. **Φ_n: RR_n → ℝ₊** — Fisher information is a "norm" on this monoid
3. **r_n: RR_n → ℝ₊** — relative decay rate is another "norm"
4. **Hermite flow** — a semigroup action He_t: RR_n → RR_n

Stam says: 1/Φ is a **lax monoidal functor** from (RR_n, ⊞_n) to (ℝ₊, +).
"Stam for r" says: 1/r is also lax monoidal.

### What to investigate

1. **Monoidal functor tower:** We have a chain of lax monoidal functors:
   ```
   (RR_n, ⊞_n) --1/r--> (ℝ₊, +)
   (RR_n, ⊞_n) --1/Φ--> (ℝ₊, +)   [Stam, to prove]
   ```
   Is there a natural transformation 1/r → 1/Φ that preserves lax monoidality?
   (This would lift "Stam for r" to "Stam for Φ".)

2. **Wiring diagram formulation:** In futon5's CT DSL, can we express:
   - The MSS convolution as a wiring diagram composition
   - The Stam inequality as a diagram inequality
   - The de Bruijn identity as a naturality condition

3. **Enriched category:** (RR_n, ⊞_n) with the partial order p ≤ q iff
   "Φ(p) ≥ Φ(q)" (ordering by information content) — is this an enriched
   monoidal category? Does the enrichment structure help?

4. **Kan extension:** The Hermite flow He_t is a colimit construction
   (averaging over random matchings). Can Stam be expressed as a
   Kan extension property?

### Practical constraint

Don't build new CT infrastructure. Use the ideas to ORGANIZE existing
algebraic facts and suggest proof structures. The question is: does the
categorical viewpoint reveal a proof strategy that pure algebra misses?

### Deliverable

A markdown document (`problem4-ct-analysis.md`) describing:
- Which categorical structures apply
- Whether natural transformation / Kan extension ideas suggest a proof
- Concrete algebraic consequences (new inequalities to test numerically)

---

## Task 4: Direct Algebraic Proof for n=4 via SOS

### Motivation

n=3 is proved. n=4 is the first open case. For n=4:
- MSS convolution has 4 coefficients (e₁, e₂, e₃, e₄)
- The convolution formula is explicit (4×4 weight matrix)
- Φ, Ψ, r are rational functions of 4 roots

An SOS (sum-of-squares) proof of Stam for n=4 would:
1. Be a complete proof for a specific case
2. Potentially reveal the algebraic structure for general n
3. Give confidence that general-n is provable

### What to do

1. Set up the MSS convolution for n=4 in sympy
2. Express Stam as a polynomial inequality in 8 variables (4 roots of p, 4 of q)
3. Use symmetry reduction:
   - Center both polynomials (e₁ = 0): reduces to 6 variables
   - Use scale invariance: reduces to 5 variables
4. Attempt SOS decomposition using `sympy` or interface with `dsos`/`sdsos`
5. If full SOS is too expensive, try:
   - Fixed gap ratios (1-parameter families)
   - Near-equispaced perturbation analysis
   - AM-GM / Schur convexity arguments on the reduced expression

### Key formulas

```python
# For n=4, MSS convolution weights w_{ij} = C(4-i,4-k)·C(4-j,4-k)/C(4,4-k):
# k=1: w_{10}=w_{01}=1, w_{ij}=0 otherwise  → c_1 = a_1 + b_1
# k=2: w_{20}=w_{02}=1, w_{11}=2/3           → c_2 = a_2 + (2/3)a_1·b_1 + b_2
# k=3: w_{30}=w_{03}=1, w_{21}=w_{12}=1/2    → c_3 = a_3 + (1/2)a_2·b_1 + (1/2)a_1·b_2 + b_3
# k=4: w_{40}=w_{04}=1, w_{31}=w_{13}=1/3,
#       w_{22}=1/6                             → c_4 = a_4 + (1/3)a_3·b_1 + (1/6)a_2·b_2 + (1/3)a_1·b_3 + b_4
```

Verify against the `mss_convolve` function in the scripts.

### Deliverable

A sympy script that either:
- Produces an SOS certificate for Stam at n=4, OR
- Identifies the minimal reduced polynomial and its properties (degree, terms, sign structure)

---

## Verification Scripts

All scripts are in `scripts/`:

| Script | What it does |
|--------|-------------|
| `prove-p4-r-submultiplicative.py` | Large-scale r ≤ min test, Stam-for-r, symbolic n=3, flow test |
| `prove-p4-stam-from-r.py` | Tests W, g'(t), Φ relationships, combined inequalities |
| `prove-p4-hprime-leq-zero.py` | Hermite structure, r distribution, extremal analysis |
| `verify-p4-hprime-analysis.py` | h'(t) explicit formula derivation and verification |
| `verify-p4-approach-e-refined.py` | Semigroup identity approach, h(t) monotonicity |
| `verify-p4-new-approaches.py` | Tests 6 approaches (E-K), killed 4 |

## Priority

**Task 1 > Task 2 > Task 4 > Task 3**

Task 1 (symbolic n=3 Stam-for-r) is the most concrete and achievable.
Task 2 (score decomposition) is the most likely to yield a general-n proof.
Task 4 (n=4 SOS) is brute-force but definitive for one case.
Task 3 (CT) is organizational and may suggest new angles.

Tasks 1 and 4 are independent. Task 2 informs Task 3.
