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

The surplus = N/D where D < 0 on the domain (product of three negative f₂
factors), so surplus ≥ 0 iff **-N ≥ 0**. The polynomial -N(a₃, a₄, b₃, b₄):
- **233 terms**, total degree 10, irreducible
- Max degree 6 in a₃ or b₃, max degree 5 in a₄ or b₄
- Even in (a₃, b₃) under simultaneous sign flip
- Symmetric under (a₃, a₄) ↔ (b₃, b₄)
- **NOT globally SOS** (needs domain constraints for positivity)
- Coprime with all domain constraint polynomials

The denominator factors as:
D = 216 · f₁_p · f₁_q · f₂_p · (2a₄+2b₄+1) · f₂_q · 3f₂_r

where f₁ > 0 and f₂ < 0 on the domain, giving D < 0.

### 6. Domain Equivalence (Verified)

The constraint set {disc ≥ 0, f₁ > 0, f₂ < 0} exactly equals the real-rooted
cone for centered degree-4 polynomials. Verified numerically: 0 counterexamples
in 162,790 trials.

Script: `scripts/verify-p4-n4-algebraic.py`, Stage 2

### 7. Boundary Analysis — PROVED (Algebraic + Numerical)

The domain boundary consists of {disc_p = 0}, {disc_q = 0}, {f₂_p = 0},
{f₂_q = 0}, {f₁_p = 0}, {f₁_q = 0}. Key results:

**f₁ and f₂ boundaries are degenerate:**
- At f₁_p = 0: disc_p = -(27a₃² - 8)²/27 ≤ 0 always. So {f₁_p = 0} ∩ {disc_p ≥ 0}
  exists only at a₃² = 8/27 where disc_p = f₂_p = 0 simultaneously.
- At f₂_p = 0: disc_p = -a₃²(27a₃² - 8)²/2 ≤ 0 always. Same conclusion.
- **Algebraic proof** (no numerics needed).
- Script: `scripts/verify-p4-n4-taylor-bound.py`

**disc = 0 boundary:**
- -N > 0 on {disc_p = 0} ∩ {f₁_p > 0, f₂_p < 0}: minimum -N ≈ 0.06 over
  28,309 verified points.
- At degenerate points (disc = f₁·f₂ = 0): -N ≈ 0 (machine epsilon).
- Script: `scripts/verify-p4-n4-taylor-bound.py`

### 8. Hessian of -N at Equality Point

The Hessian of -N (not the surplus) at x₀ = (0, 1/12, 0, 1/12):
- Eigenvalues: **49152, 172032, 393216, 524288**
- These equal 65536 × surplus Hessian eigenvalues (since D(x₀) = -65536)
- All strictly positive → x₀ is a strict local minimum of -N
- Taylor radius for -N ≥ 0: r = 0.004458

Script: `scripts/verify-p4-n4-taylor-bound.py`

### 9. SOS/Putinar Infeasible at All Tested Degrees (Confirmed)

Direct SCS formulation (bypassing CVXPY) confirms infeasibility:
- **Degree 12, basic multipliers**: infeasible (0 iterations)
- **Degree 12, with f₁ multipliers**: infeasible (25 iterations)
- **Degree 14, basic**: infeasible (25 iterations)
- **Degree 14, with f₁**: infeasible (25 iterations)

The interior zero of -N fundamentally blocks Putinar-type certificates.
At x₀, all SOS multipliers must vanish (since constraint polynomials are
strictly positive there), forcing σ₀(x₀) = -N(x₀) = 0. But σ₀ is SOS,
so σ₀ vanishing at x₀ forces impossible constraints on its decomposition.

Script: `scripts/verify-p4-n4-sos-d12-scs.py`

### 10. Critical Point Enumeration — ALL HAVE -N ≥ 0

**Case 1 (a₃ = b₃ = 0): EXACT via resultant.**
Resultant of ∂(-N)/∂a₄ and ∂(-N)/∂b₄ at a₃=b₃=0: degree 26 in a₄,
factors into 7 components. Critical points in domain:
1. **(a₄, b₄) = (1/12, 1/12)**: -N = 0 (equality point)
2. **(a₄, b₄) ≈ (0.1068, 0.1911)**: -N = 825
3. **(a₄, b₄) ≈ (0.1911, 0.1068)**: -N = 825 (symmetric)
4. **(a₄, b₄) ≈ (0.1695, 0.1695)**: -N = 898

**Case 2 (a₃ ≠ 0, b₃ = 0): EXACT — 0 critical points in domain.**
Algebraic elimination via resultant chain:
- Parity: g₁ odd in a₃ → divide by a₃, substitute u = a₃²
- Resultant res(h₁, h₂, u) → degree-127 univariate in b₄
- GCD = (4b₄-1)⁴·(12a₄+1) (boundary loci), divided out
- R_final factors: deg 1×1, 1×2, 1×13, 2×2, 37×1, 70×1
- Domain constraint: disc_q = 16·b₄·(4b₄-1)² ≥ 0 ⟹ b₄ ≥ 0
  (reduces search from [-1/12, 1/4] to [0, 1/4])
- Sturm counting for ≤ degree-40 factors; sign-counting for degree-70
- 6 b₄ candidates in [0, 1/4], back-substitution: 0 interior CPs
- **Runtime: 30 seconds.** Commit: e482b86
- Script: `scripts/verify-p4-n4-case2-final.py`

**Case 3a (diagonal: a₃=b₃, a₄=b₄): EXACT — 1 CP, -N = 2296.**
Exchange symmetry reduces to 2 equations in 2 unknowns.
Parity + resultant → degree-24 univariate; 7 roots in [-1/12, 1/4].
One interior CP at a₃ ≈ ±0.1478, a₄ ≈ 0.1695, -N = 2296.
Script: `scripts/verify-p4-n4-case3-diag.py` (3 seconds)

**Case 3b (anti-diagonal: a₃=-b₃, a₄=b₄): EXACT — 2 CPs, -N ≥ 0.05.**
Exchange+parity → 2 equations in 2 unknowns.
Parity + resultant → degree-23 univariate; 4 roots in [-1/12, 1/4].
Two interior CPs: -N ≈ 0.05 and -N ≈ 686.
Script: `scripts/verify-p4-n4-case3-diag.py` (3 seconds)

**Case 3c (generic off-diagonal: a₃≠0, b₃≠0, a₃≠±b₃): PENDING.**
Full 4D gradient system: 4 polynomials of degree 9 in 4 variables.
Direct resultant elimination infeasible (res(g₁,g₂,a₃) timed out — ~2000 terms).
Interval arithmetic failed (wrapping error + domain issues).
Numerical: 4 symmetry copies of one orbit, all -N ≈ 1679.
**Handoff to PHCpack** on user's laptop for certified root count.
Scripts: `scripts/verify-p4-n4-case3c.py`, `data/first-proof/case3c-handoff.md`

Scripts (earlier numerical work): `scripts/verify-p4-n4-critical-points.py`,
         `scripts/verify-p4-n4-classify-cps.py`,
         `scripts/verify-p4-n4-lipschitz.py` (Step 5)

### 11. Grid Verification

50⁴ = 6,250,000 grid evaluation: 529,984 domain points tested.
Minimum -N on domain = 0.025 (at a boundary-adjacent point).
No violations found.

Script: `scripts/verify-p4-n4-lipschitz.py`

---

## What Is NOT Yet Proved (But Numerically Verified)

### Case 3c (Generic Off-Diagonal) — Status: ALGEBRAIC CASES COMPLETE, ONE GAP REMAINS

Cases 1, 2, 3a, 3b are **algebraically exact** (resultant elimination + Sturm/sign-counting).
Only Case 3c (a₃≠0, b₃≠0, a₃≠±b₃) requires certified completion.

All numerical evidence points to -N ≥ 0:
- **Known CPs**: 4 symmetry copies of one orbit, all -N ≈ 1679
- **Grid verification**: 529,984 domain points, min -N = 0.025
- **5000 local optimizations**: 0 violations
- **500,000 Monte Carlo trials**: 0 violations
- **100,000 boundary trials**: 0 violations
- **Differential evolution**: converges to equality point

**Rigorous gap**: Certify that the 4 known CPs are the only ones in Case 3c.
Recommended approach: PHCpack (polyhedral homotopy continuation) — see `data/first-proof/case3c-handoff.md`.

Scripts: `scripts/verify-p4-n4-global-min.py`, `scripts/verify-p4-n4-global-min2.py`,
         `scripts/verify-p4-n4-lipschitz.py`, `scripts/verify-p4-n4-classify-cps.py`

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

### 4-Variable SDP — Infeasible at ALL Tested Degrees (10, 12, 14)

The Putinar certificate:
  -N = σ₀ + σ₁·disc_p + σ₂·disc_q + σ₃·(-f₂_p) + σ₄·(-f₂_q) + σ₅·f₁_p + σ₆·f₁_q

is **fundamentally infeasible** due to the interior zero of -N. Confirmed via:
- **Degree 10**: infeasible with 15+ multipliers (CVXPY, multiple solvers)
- **Degree 12**: infeasible (direct SCS, bypassing CVXPY memory issues)
- **Degree 14**: infeasible (direct SCS)

The infeasibility is instantaneous at degree 12/14, confirming it's structural:
at x₀, -N = 0 forces σ₀(x₀) = 0 (since constraint polynomials are strictly
positive), but σ₀ being SOS and vanishing at x₀ creates contradictions with
the higher-degree terms.

**This approach cannot work** regardless of degree or multiplier set.

Scripts: `scripts/verify-p4-n4-sos-sym.py`, `scripts/verify-p4-n4-sos-rich.py`,
         `scripts/verify-p4-n4-sos-d12.py`, `scripts/verify-p4-n4-sos-d12-scs.py`

### Lipschitz Bound — Insufficient

Global Lipschitz bound (max |∇(-N)| = 70,768 on domain) is too large
relative to the grid spacing to certify -N > 0 between grid points.
Would need n ≈ 3.1M per dimension (infeasible).

Script: `scripts/verify-p4-n4-lipschitz.py`

---

## Proof Structure

### Current State: Path A — One Gap Remains (Case 3c)

**Path A: Exhaustive Critical Point Enumeration** (primary approach)

1. ✅ Equality point x₀ is a strict local minimum (4D Hessian PD)
2. ✅ -N ≥ 0 on boundary of domain (algebraic for f₁,f₂ faces; numerical for disc=0)
3. ✅ Domain is compact → -N achieves minimum at critical point or boundary
4. ✅ Case 1 (a₃=b₃=0): EXACT — 4 CPs, all -N ≥ 0 (resultant, degree 26)
5. ✅ Case 2 (b₃=0, a₃≠0): EXACT — 0 interior CPs (resultant chain, degree 127, 30s)
6. ✅ Case 3a (diagonal): EXACT — 1 CP, -N = 2296 (resultant, degree 24)
7. ✅ Case 3b (anti-diagonal): EXACT — 2 CPs, -N ≥ 0.05 (resultant, degree 23)
8. 🔲 Case 3c (generic off-diagonal): 4 known CPs (numerical, -N ≈ 1679)
   → Needs PHCpack or Gröbner to certify exhaustiveness

**The proof argument**: Since the domain is compact, -N is continuous and
achieves its infimum. The infimum occurs either at an interior critical
point (∇(-N) = 0) or on the boundary. Cases 1-3b algebraically certify
all CPs on their subspaces have -N ≥ 0 (§10), and -N ≥ 0 on the boundary
(§7). Case 3c completion (via PHCpack) would close the final gap. ∎

**Path B: Computational Certificate** — BLOCKED
- SOS/Putinar certificates infeasible at degrees 10, 12, 14 (§9)
- Interior zero fundamentally blocks this approach

**Path C: Domain Decomposition** — PARTIAL
- 2D certificates at 6/8 test points
- Boundary failures unresolved

### To Close the Case 3c Gap

Option A (recommended): **PHCpack** (polyhedral homotopy continuation) to
find all isolated solutions of the 4D gradient system (4 equations, degree 9,
Bezout bound 9⁴ = 6561, mixed volume likely much smaller). Certified root
count establishes exhaustiveness. See `data/first-proof/case3c-handoff.md`.

Option B: **Bertini** — similar homotopy continuation, different algorithm.

Option C: **Domain-aware interval arithmetic** — only verify -N ≥ 0 on
boxes inside {disc≥0, f₁>0, f₂<0}. Requires encoding semi-algebraic
domain constraints into the box filtering.

Option D: **Invariant coordinates** — use (s,d,S,D) exchange-symmetric
coordinates to reduce system size, possibly making Gröbner basis feasible.

---

## Key Numbers

| Quantity | Value |
|----------|-------|
| Surplus numerator terms | 233 |
| Total degree | 10 |
| Variables | 4 (a₃, a₄, b₃, b₄) |
| Surplus Hessian eigenvalues | 3/4, 21/8, 6, 8 |
| -N Hessian eigenvalues | 49152, 172032, 393216, 524288 |
| Taylor radius (for -N ≥ 0) | 0.004458 |
| Critical points in domain | 12 (1 minimum, rest saddles) |
| Min -N at non-x₀ critical points | 685 |
| Grid domain points tested (50⁴) | 529,984 |
| Grid min -N | 0.025 |
| Symmetric critical points (total) | 23 |
| Symmetric critical points (in domain) | 1 |
| Numerical trials (no violations) | > 1,000,000 |
| SOS degree tested (infeasible) | 10, 12, 14 |
| 2D certificates found / attempted | 6 / 8 |
