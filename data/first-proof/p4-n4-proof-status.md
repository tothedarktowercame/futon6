# P4 n=4 Proof Status: Finite Free Stam Inequality

**Date:** 2026-02-12
**Claim:** For monic real-rooted degree-4 polynomials p, q:
  1/Φ₄(p ⊞₄ q) ≥ 1/Φ₄(p) + 1/Φ₄(q)
with equality iff p = q = x⁴ - x² + 1/12 (degree-4 semicircular).

---

## What Is Proved

### 1. Key Identity (Theorem)

**Φ₄(p) · disc(p) = -4 · (a₂² + 12a₄) · (2a₂³ - 8a₂a₄ + 9a₃²)**

- Verified symbolically via SymPy (root variables → coefficient comparison)
- Upgraded from numerical (200+ tests, error < 3e-14) to theorem
- Script: `scripts/verify-p4-n4-algebraic.py`, Stage 1
- Commit: c609e47

### 2. Symmetric Subfamily (a₃ = b₃ = 0) — PROVED

For centered polynomials with a₂ = b₂ = -1 and a₃ = b₃ = 0:
- Change of variables: w = a₄ + b₄, r = a₄·b₄
- Surplus = F(w,r)/(positive denominator)
- F(w,r) = (polynomial with coefficient g(w) on r, which is negative on [0, 1/2])
- Therefore F is decreasing in r, minimized at r = w²/4 (i.e., a₄ = b₄)
- F(w, w²/4) = 3w²(w+1)(3w+1) ≥ 0 on domain w ∈ [0, 1/2]
- Equality iff w = 0, i.e., a₄ = b₄ = 0 → p = q = x⁴ - x² + 1/12

Source: `data/first-proof/deep-dive-strategy-c.md`, Section 4.2

### 3. Unique Critical Point in Symmetric Domain (Theorem)

The symmetric surplus S(a₄, b₄) = surplus|_{a₃=b₃=0} has exactly **23 critical
points** (∇S = 0), of which:
- **1 is in the real-rooted domain**: a₄ = b₄ = 1/12, surplus = 0 (the equality point)
- 8 are outside the domain (real but violating constraints)
- 14 are complex

Script: `scripts/verify-p4-n4-sos-reduced.py`, Approach 2

### 4. Full 4D Hessian at Equality Point — POSITIVE DEFINITE

At (a₃, b₃, a₄, b₄) = (0, 0, 1/12, 1/12):

```
H = [[27/16,     0,  15/16,     0],
     [    0,     7,      0,    -1],
     [15/16,     0,  27/16,     0],
     [    0,    -1,      0,     7]]
```

Eigenvalues: **3/4, 21/8, 6, 8** — all strictly positive.

Block structure:
- (a₃, b₃) block: eigenvalues 21/8, 3/4
- (a₄, b₄) block: eigenvalues 8, 6

This proves the equality point is a **strict local minimum** of the surplus.

Script: `scripts/verify-p4-n4-global-min.py`, Step 1

### 5. Surplus Numerator (Computed)

The surplus, after clearing denominators, is a polynomial N(a₃, a₄, b₃, b₄):
- **233 terms**, total degree 10
- Max degree 6 in a₃ or b₃, max degree 5 in a₄ or b₄
- Even in (a₃, b₃) under simultaneous sign flip
- Symmetric under (a₃, a₄) ↔ (b₃, b₄)
- **NOT globally SOS** (needs domain constraints for positivity)

Script: `scripts/verify-p4-n4-algebraic.py`, Stage 2

---

## What Is NOT Yet Proved (But Numerically Verified)

### General Case (a₃, b₃ ≠ 0)

The surplus is non-negative for ALL (a₃, a₄, b₃, b₄) in the real-rooted domain:
- **5000 local optimizations**: 0 violations, minimum at equality point
- **500,000 Monte Carlo trials**: 0 violations
- **100,000 boundary trials**: 0 violations
- Differential evolution global optimizer: converges to equality point

Scripts: `scripts/verify-p4-n4-global-min.py`, `scripts/verify-p4-n4-global-min2.py`

---

## Failed Approaches

### Perturbation from Symmetric Case — FAILED

**Idea:** If the Hessian in (a₃, b₃) at a₃=b₃=0 is PSD for all (a₄, b₄),
then the symmetric case is the minimum and we're done.

**Result:** Hessian is NOT PSD everywhere:
- H₁₁ < 0 for 24% of sampled (a₄, b₄)
- det(H) < 0 for 40% of sampled (a₄, b₄)

The surplus is NOT always minimized at a₃=b₃=0 when (a₄, b₄) are fixed.
The proof must handle all four variables jointly.

Script: `scripts/verify-p4-n4-perturbation.py`

### 2D Positivstellensatz at Fixed (a₄, b₄) — PARTIAL

**Idea:** Fix (a₄, b₄) and find a Positivstellensatz certificate for the
surplus polynomial in (a₃, b₃) on the 2D domain.

**Result:** Certificates found at 6/8 test points:
- Works well at interior points (1/12, 1/8, 1/6, 1/24 values)
- **Fails** near domain boundary (a₄ ≈ 1/5 or larger)
- Needs richer multiplier sets (e.g., cross-terms disc·(-f₂)) near boundary

Script: `scripts/verify-p4-n4-sos-reduced.py`, Approach 1

### 4-Variable SDP — OOM

The full 4-variable Positivstellensatz with Putinar multipliers:
  N = σ₀ + σ₁·disc_p + σ₂·disc_q + σ₃·(-f₂_p) + σ₄·(-f₂_q) + ...

Requires 126×126 Gram matrix (8001 parameters) + multiplier terms.
CVXPY runs out of memory building the constraint expressions.

Could be fixed with:
1. Symmetry reduction (even powers only → ~50×50 + 16×16)
2. Direct SDP formulation bypassing CVXPY expression tree
3. External solver (MOSEK, SageMath SOS)

Script: `scripts/verify-p4-n4-sos.py`

---

## Proof Structure (What Would Complete the Proof)

### Path A: Unique Global Minimum

1. ✅ Equality point is a strict local minimum (4D Hessian PD)
2. ✅ In the symmetric subfamily, it's the unique critical point in the domain
3. 🔲 Show it's the unique global minimum over all 4 variables
   - Find all critical points of the 4-variable surplus (polynomial system)
   - Show surplus > 0 on the boundary (disc_p = 0 or disc_q = 0)

### Path B: Computational Certificate

1. 🔲 Get the 4-variable SDP working (with symmetry reduction or external solver)
2. A Positivstellensatz certificate would constitute a computer-assisted proof

### Path C: Domain Decomposition

1. ✅ 2D certificates at interior (a₄, b₄) points
2. 🔲 Fix the near-boundary failures (richer multiplier sets)
3. 🔲 Finite covering of the (a₄, b₄) domain with certificates at each point

---

## Key Numbers

| Quantity | Value |
|----------|-------|
| Surplus numerator terms | 233 |
| Total degree | 10 |
| Variables | 4 (a₃, a₄, b₃, b₄) |
| 4D Hessian eigenvalues | 3/4, 21/8, 6, 8 |
| Symmetric critical points (total) | 23 |
| Symmetric critical points (in domain) | 1 |
| Numerical trials (no violations) | > 500,000 |
| 2D certificates found / attempted | 6 / 8 |
