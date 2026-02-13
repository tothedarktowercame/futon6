# Claude Handoff: Prove Full n=4 Stam Inequality (6-variable case)

**Date:** 2026-02-13
**From:** Claude (explore/discover cycle)
**Task type:** Algebraic proof — extend symmetric case to full centered quartics
**Context:** Symmetric case (e₃=0) proved. Full case structured but open.

---

## The Goal

Prove that for centered monic real-rooted quartics p, q:

```
1/Φ₄(p ⊞₄ q) ≥ 1/Φ₄(p) + 1/Φ₄(q)
```

with equality iff p = q = x⁴ - x² + 1/12 (degree-4 semicircular).

## What's Already Proved

### n=2, n=3: Done.

### n=4, e₃=0 (symmetric quartics): PROVED

Two independent proofs:
1. **30-term certificate** (`prove-p4-n4-symmetric-30term.py`): Normalize to
   r=t/s, x=4a/s², y=4b/t². Then N = s¹⁰r⁴G(r,x,y) where G = Ar² + Br + C
   with A,B,C ≥ 0 via discriminant bounds on [-1,2].
2. **Codex verification** (`prove-p4-n4-via-phi4disc.py`): Independent symbolic
   check using disc/P formulation.

### n=4 critical point enumeration: 99% done

The "other Claude" proved Cases 1, 2, 3a, 3b algebraically (resultant + Sturm).
Case 3c (generic off-diagonal) has 4 CPs via PHCpack, all with -N ≈ 1679 > 0.
**Gap**: 5242/6561 homotopy paths failed (separate Codex task for certification).

## The Full 6-Variable Problem

### Formulation

Variables: (s, t, u, v, a, b) where s=-e₂(p), t=-e₂(q), u=e₃(p), v=e₃(q),
a=e₄(p), b=e₄(q).

Convolution: S=s+t, U=u+v, A=a+b+st/6.

```
1/Φ₄(σ, ε, α) = disc(σ, ε, α) / [4·(σ²+12α)·(2σ³-8σα-9ε²)]
```

where disc is the discriminant of x⁴ - σx² - εx + α.

The surplus = 1/Φ₄(S,U,A) - 1/Φ₄(s,u,a) - 1/Φ₄(t,v,b).

### Normalized form

Set r=t/s, x=4a/s², y=4b/t², p=u/s^(3/2), q=v/s^(3/2). Then:

```
surplus_numerator = s¹⁶ · K(r, x, y, p, q)
```

K has 659 terms, degree 18 in (r,x,y,p,q).

### Block decomposition (KEY STRUCTURAL FINDING)

K decomposes by total (p,q)-degree:

```
K = K0(r,x,y) + K2(r,x,y,p,q) + K4(...) + K6(...) + K8(...)
```

| Block | Terms | r-degree | Sign on feasible |
|-------|-------|----------|-----------------|
| K0 | 102 | 10 | **always ≥ 0** (proved) |
| K2 | 200 | 10 | indefinite (26.6% neg) |
| K4 | 201 | 10 | indefinite (39.9% neg) |
| K6 | 114 | 7 | **almost always ≤ 0** (98.8% neg) |
| K8 | 42 | 4 | **always ≥ 0** (0% neg in 50k samples) |

### The critical coupled inequality

**K0 + K2 + K4 ≥ 0** on all 50,000 feasible samples (min ≈ 1.4e-7).

This is 503 terms in 5 variables. Proving this is the main challenge.

After that: K = (K0+K2+K4) + K6 + K8. Since K0+K2+K4 ≥ 0 and K8 ≥ 0,
the remaining question is whether K6 + K8 + (surplus from K0+K2+K4) ≥ 0.
Empirically, the full K ≥ 0 always.

## Why Direct Approaches Fail

1. **SOS/Putinar**: Fundamentally infeasible due to interior zero at the
   equality point. Confirmed at degrees 10, 12, 14.
2. **Blockwise positivity**: K2 is indefinite — can't prove each block ≥ 0.
3. **Quadratic-in-r**: The Codex formulation gives degree 10 in r (too high).
   The symmetric case worked because a simpler common denominator gave degree 2.
4. **Perturbation from symmetric**: Hessian in (a₃,b₃) NOT PSD for 24% of (a₄,b₄).

## Suggested Proof Approaches

### Approach A: Better common denominator

The symmetric case used 1/Φ₄ = 2a(s²-4a)/[s(s²+12a)], where the (s²-4a)
factor cancelled from disc/f₂. For the full case, try:

```
1/Φ₄(s,u,a) = disc/(4·f₁·f₂)
```

where f₂ = 2s³-8sa-9u². The surplus common denominator is 4·f₁p·f₂p·f₁q·f₂q·f₁c·f₂c.
But disc and f₂ might share common factors after expanding the surplus numerator.
Look for cancellations that reduce r-degree.

**Concrete test**: Compute the surplus numerator with the "natural" common denominator
(product of f₁·f₂ for p, q, conv) and check if it factors or has lower r-degree
than the 659-term Codex version.

### Approach B: SOS in (p,q) with parametric r,x,y

K is degree 8 in (p,q) and even (only even powers). Write K as:

```
K = v^T M(r,x,y) v
```

where v = [1, p², pq, q², p⁴, p³q, p²q², pq³, q⁴]^T and M is a 9×9 matrix
of polynomials in (r,x,y). Show M is PSD for all feasible (r,x,y).

This is a "parametric SOS" problem. The matrix M has 9×9 = 81 entries, each
a polynomial in 3 variables. PSD-ness can be checked via:
- Sylvester criterion (principal minors ≥ 0)
- Direct eigenvalue analysis at sampled points + interval arithmetic

### Approach C: Cauchy-Schwarz generalization

For n=3, the surplus had the Titu form:
```
surplus = (3/2)[u²/s² + v²/t² - (u+v)²/(s+t)²]
```

For n=4, decompose 1/Φ₄ as:
```
1/Φ₄(s,u,a) = f(s,a) + u²·g(s,a) + u⁴·h(s,a)
```

(since 1/Φ₄ is rational in u² by the factored form). Then the surplus
decomposes into:
- f-surplus (the symmetric case, proved)
- g-surplus: involves u²g(s,a) + v²g(t,b) - (u+v)²g(S,A)
- h-surplus: similar with u⁴, v⁴

The g-surplus might have a Titu/C-S structure. Check!

### Approach D: Numerical algebraic geometry (complement Path 1)

Use the critical point approach with better tools:
- Certify Case 3c via alpha theory (Codex handoff already written)
- Or use the invariant coordinate reduction to make Gröbner basis feasible

### Approach E: Domain-constrained polynomial optimization

Since the feasibility constraints bound p² ≤ 2(1-x)/9 and q² ≤ 2r³(1-y)/9,
substitute p = √(2(1-x)/9)·sinθ, q = √(2r³(1-y)/9)·sinφ (or rational
parametrizations of the ellipse). This converts the problem to showing K ≥ 0
on a rectangle in (θ,φ) for each (r,x,y), which might be easier.

## Key Files

| File | What |
|------|------|
| `scripts/explore-p4-n4-block-coupling.py` | Block coupling discovery (this session) |
| `scripts/explore-p4-n4-r-structure.py` | r-degree analysis |
| `scripts/analyze-p4-n4-full-normalized.py` | Full normalized surplus + K2 indefiniteness |
| `scripts/prove-p4-n4-symmetric-30term.py` | Symmetric case proof |
| `scripts/prove-p4-n4-via-phi4disc.py` | Codex verification of symmetric + identities |
| `data/proof-state/P4.edn` | Full proof DAG |
| `data/first-proof/CODEX-P4-CASE3C-CERTIFICATION.md` | Path 1 handoff |

## Verification

```python
import numpy as np
# Pick random feasible point
s, t = 2.0, 3.0
u, v = 0.1, -0.2
a, b = 0.5, 1.0  # need disc(p)>=0, disc(q)>=0, disc(conv)>=0
S, U, A = s+t, u+v, a+b+s*t/6
f = lambda sig, eps, alp: (256*alp**3 - 128*sig**2*alp**2 + 144*sig*eps**2*alp
    + 16*sig**4*alp - 27*eps**4 - 4*sig**3*eps**2) / (
    4*(sig**2+12*alp)*(2*sig**3-8*sig*alp-9*eps**2))
surplus = f(S, U, A) - f(s, u, a) - f(t, v, b)
print(f"surplus = {surplus}")  # should be >= 0
```

## What Success Looks Like

An algebraic proof that K(r,x,y,p,q) ≥ 0 on the feasible domain:
- r > 0, x ∈ (0,1), y ∈ (0,1)
- p² < 2(1-x)/9, q² < 2r³(1-y)/9
- convolution constraint g3 > 0

Ideally: a decomposition K = Σ fᵢ · gᵢ² (or similar) that's checkable by
computer algebra. The proof should fit in ≤ 3 pages of the monograph.
