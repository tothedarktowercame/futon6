# Claude Handoff: Prove Stam Inequality for Symmetric Quartics (n=4, e₃=0)

**Date:** 2026-02-13
**From:** Claude (explore/discover cycle)
**Task type:** Symbolic proof — algebraic reasoning, not heavy computation
**Context:** This is Problem 4 of a math monograph. n=3 is proved. n=4 is open.

---

## The Goal

Prove that for all s, t > 0 and 0 < 4a < s², 0 < 4b < t²:

```
f(S, A) ≥ f(s, a) + f(t, b)
```

where f(σ, e₄) = 2e₄(σ² − 4e₄) / [σ(σ² + 12e₄)], S = s+t, A = a+b+st/6.

This is the **finite Stam inequality** for symmetric quartics (degree-4
polynomials with roots ±α, ±β). The function f = 1/Φ₄ is the reciprocal
Fisher information.

## What's Already Proved

- **n=2:** Equality (trivial — 1/Φ₂ is linear).
- **n=3:** Proved via Titu's lemma. The surplus = (3/2)[u²/s² + v²/t² − (u+v)²/(s+t)²],
  which is non-negative by Cauchy-Schwarz. SOS certificate:
  N = (s²v − t²u)² + st(sv+tu)² + st(sv−tu)².

## What You Have

The surplus = f(S, A) − f(s, a) − f(t, b) equals N/D where:

**Denominator** (positive on the feasible region):
```
D = 9·s·t·(12a + s²)·(12b + t²)·(s + t)·(12a + 12b + s² + 4st + t²)
```

**Numerator** (30 terms, degree 10 — need to show N ≥ 0):
```
N = 10368·a³bt² + 864·a³t⁴
  + 10368·a²b²s² + 10368·a²b²t²
  + 864·a²bs²t² + 3456·a²bst³ + 1728·a²bt⁴
  + 288·a²s²t⁴ + 576·a²st⁵ + 72·a²t⁶
  + 10368·ab³s² + 1728·ab²s⁴ + 3456·ab²s³t + 864·ab²s²t²
  − 936·ab·s⁴t² − 1728·ab·s³t³ − 936·ab·s²t⁴
  − 42·a·s⁴t⁴ − 24·a·s³t⁵ + 18·a·s²t⁶
  + 864·b³s⁴ + 72·b²s⁶ + 576·b²s⁵t + 288·b²s⁴t²
  + 18·b·s⁶t² − 24·b·s⁵t³ − 42·b·s⁴t⁴
  + 3·s⁶t⁴ + 4·s⁵t⁵ + 3·s⁴t⁶
```

**Feasibility constraints:** s, t > 0; 4a < s²; 4b < t².
(The convolution is automatically real-rooted — no extra constraint needed.)

## Structural Observations

1. **Swap symmetry:** N is symmetric under (s,a) ↔ (t,b). Verify by inspection.

2. **Self-convolution factors completely:** At s=t, a=b:
   ```
   N = 2s²·(s² − 12a)²·(12a + s²)·(12a + 5s²)
   ```
   All factors manifestly ≥ 0 on the feasible region. (s² − 12a can be negative
   when a > s²/12, but it's squared.)

3. **Pure s,t terms** (a=b=0): N = 3s⁶t⁴ + 4s⁵t⁵ + 3s⁴t⁶ = s⁴t⁴(3s² + 4st + 3t²) ≥ 0.

4. **Negative terms** all involve ab or single a,b multiplied by high powers of st.
   The negative terms are: −936ab(s⁴t²+s²t⁴) − 1728ab·s³t³ − 42(a+b)s⁴t⁴
   − 24(as³t⁵ + bs⁵t³) + 18(as²t⁶ + bs⁶t²).
   Wait — the last terms (+18) are positive. The truly negative terms are:
   −936ab·s²t²(s²+t²) − 1728ab·s³t³ − 42s⁴t⁴(a+b) − 24st(as²t⁴+bs⁴t²).

5. **Scaling:** Under (s,t,a,b) → (λs, λt, λ²a, λ²b), N → λ¹⁰N. So N is
   homogeneous of degree 10 if we assign deg(s)=deg(t)=1, deg(a)=deg(b)=2.

## Suggested Proof Approaches

### Approach A: Substitute a = s²α/4, b = t²β/4

With 0 < α, β < 1 (parametrizing the feasibility region), N becomes a polynomial
in (s, t, α, β) with s, t > 0. Factor out s⁴t⁴ (or similar) and reduce to a
polynomial in the ratio r = s/t and (α, β). By swap symmetry, WLOG r ≥ 1.

### Approach B: Schur-like / Muirhead

Group terms by their (a,b)-degree and show each group is non-negative using
AM-GM or Schur's inequality. The self-convolution factorization hints at
Schur-convexity structure.

### Approach C: Complete the square

The negative ab terms look like they come from expanding a square minus something.
Try writing N = Q² + (positive terms) − R, then show R ≤ Q² on the feasible region.

### Approach D: Partial fractions on the surplus

Instead of working with the numerator directly, decompose f(σ, e₄) into simpler
pieces. Note:
```
f(σ, e₄) = 2e₄(σ² − 4e₄)/[σ(σ² + 12e₄)]
          = (2/σ)·[σ²e₄ − 4e₄²]/(σ² + 12e₄)
          = (2/σ)·[e₄ − 16e₄²/(σ² + 12e₄)]
```
Maybe the surplus decomposes into terms that each satisfy a Titu/C-S bound.

### Approach E: Reduce to 2 variables

By homogeneity, set s+t = 1 (or s = 1). Then by swap symmetry, parametrize
by λ = s/(s+t) ∈ [1/2, 1]. This reduces to 3 variables (λ, a, b), or
even 2 after further substitution.

## What Success Looks Like

A human-readable algebraic proof that N ≥ 0 on the feasible region. Ideally:
- An explicit SOS decomposition (like the n=3 case), OR
- A chain of AM-GM / Titu / Schur inequalities, OR
- A reduction to a known inequality

The proof should fit in ≤ 2 pages of the monograph.

## Verification

Test any claimed identity with the script `scripts/explore-p4-n4-symmetric-quartic.py`.
To verify numerically:

```python
import numpy as np
# Pick random s,t > 0, 0 < 4a < s², 0 < 4b < t²
s, t = 2.0, 3.0
a, b = 0.5, 1.0  # 4a=2 < 4=s², 4b=4 < 9=t²
S, A = s+t, a+b+s*t/6
f = lambda sig, e4: 2*e4*(sig**2 - 4*e4)/(sig*(sig**2 + 12*e4))
surplus = f(S, A) - f(s, a) - f(t, b)
print(f"surplus = {surplus}")  # should be ≥ 0
```

## Files

- `scripts/explore-p4-n4-symmetric-quartic.py` — derivation and verification
- `scripts/explore-p4-n4-inv-phi.py` — Φ₄·disc identity discovery
- `scripts/verify-p4-n3-cauchy-sos.py` — the n=3 proof (model to follow)
- `data/first-proof/problem4-solution.md` — proof document to update if successful
