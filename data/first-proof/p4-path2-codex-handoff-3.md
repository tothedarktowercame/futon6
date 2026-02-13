# P4 Path 2 Cycle 3: Codex Handoff — Same-Sign K_red ≥ 0

## What Changed Since Cycle 2

Cycle 2 proved:
- **Claim 1**: C1 ≥ 0 (via M_y - 4L factorization)
- **Claim 2**: C0(p,-p) ≥ 0 (concavity + endpoint signs at q = -p)

**Gap identified**: Claim 2 proves K_red ≥ 0 at the q = -p boundary (opposite
sign, "easy" case where pq·B > 0). But for **same-sign** p,q:
- pq > 0 and B ≤ 0, so pq·B < 0 — this HURTS
- C0 is negative for ~49% of same-sign feasible samples
- C1·(p+q)² must compensate, and it does (0/200k violations)

The Cycle 2 conclusion "this yields K_red ≥ 0" was premature. The missing
piece is **Claim 3**: K_red(p,q) ≥ 0 for p,q ≥ 0 on the feasible domain.

## What Codex Should Prove

### Claim 3: K_red(p,q) ≥ 0 for p,q ≥ 0

Since K_red = A(P,Q) + pq·B(P,Q) with B ≤ 0, the minimum over sign choices
is at same-sign (p,q ≥ 0), giving K_red = A(P,Q) + √(PQ)·B(P,Q).

This must be ≥ 0 for all (P,Q) ∈ [0, Pmax] × [0, Qmax] where
Pmax = 2(1-x)/9, Qmax = 2r³(1-y)/9.

There are three sub-parts:

#### 3a. Edge p=0 (or equivalently q=0)

K_red(0,q) = a00 + a01·Q + a02·Q²    (quadratic in Q = q²)

a02 = 72(1-x)·Mx ≥ 0, a00 ≥ 0 (both proved in Cycle 2). Need:
a00 + a01·Q + a02·Q² ≥ 0 for Q ∈ [0, Qmax].

Since a02 ≥ 0 and a00 ≥ 0, this is a convex quadratic with non-negative
endpoints. Two proof routes:
- If a01 ≥ 0: all three terms ≥ 0, done.
- If a01 < 0: the minimum is at Q* = -a01/(2a02). Either Q* > Qmax (min at
  endpoint Qmax, check sign), or discriminant a01² - 4a02·a00 ≤ 0.

By symmetry: K_red(p,0) = a00 + a10·P + a20·P² follows the same analysis.

#### 3b. Diagonal p = q (the hardest 1D slice)

Define g(T) = K_red(√T, √T) for T = p² ∈ [0, tmax], tmax = min(Pmax, Qmax).

```
g(T) = a00 + (a10+a01+b00)·T + (delta1+4a20+4a02)·T² + 4a12·T³
```

This is a **cubic** in T (Claim 2's f(T) was quadratic in the q=-p case).

**What's already verified**:
- g(T) == K_red(√T,√T): **True** (symbolic identity check)
- g(0) = a00 ≥ 0 (same as Claim 2)
- g(Pmax) and g(Qmax) have exact factorizations (see below)
- Endpoint sign rule works: g(tmax) ≥ 0 (proved algebraically below)
- Interior minimum ≥ 0: 0/300k violations (min value ~1.76e-10)

**Exact endpoint factorizations** (verified symbolically):

```
g(Pmax) = 32*(x-1)*(3x+1)²*(3y+1)*D*pos_core*new_P / 27

g(Qmax) = -32*r⁴*(3x+1)*(y-1)*(3y+1)²*D*pos_core*new_Q / 27
```

where:
```
D = r³(1-y) - (1-x)         (same as Claim 2)
pos_core = 3r²y+r²+4r+3x+1  (same as Claim 2)
new_P = (r+1)*W + 12*(1-x)
new_Q = (r+1)*W + 12*r³*(1-y)
W = 3r²y - 3r² - 4r + 3x - 3  (< 0 on feasible)
```

**Compare with Claim 2**: fP had factor (r+1)·W; g(Pmax) has new_P = (r+1)·W + 12(1-x).

**Endpoint sign analysis**:
- When D ≥ 0 (active endpoint is Pmax): need new_P ≤ 0.
  Substitute (1-x) ≤ r³(1-y) [from D ≥ 0]:
  new_P ≤ (r+1)·W + 12r³(1-y) = -3r²(1-y)(r-1)² - 4r(r+1) ≤ 0.  ✓

- When D ≤ 0 (active endpoint is Qmax): need new_Q ≤ 0.
  Substitute (1-y) ≤ (1-x)/r³ [from D ≤ 0]:
  new_Q ≤ (r+1)·W + 12(1-x) = ... = -3(1-x)(r-1)²/r - 4r(r+1) ≤ 0.  ✓

So g(tmax) ≥ 0 at the active endpoint. ✓

**What remains for 3b**: Prove g(T) ≥ 0 on the INTERIOR of [0, tmax].

The cubic g has g3 = 4a12 = -5184rL < 0 (when L > 0). Numerically:
- g'(0) < 0 for 13.1% of samples (so "initially increasing" argument fails)
- g''(0) > 0 for 86.7% (so "concavity" argument fails)
- Interior min exists in (0,tmax) for only 1.9% of samples
- That interior min is ALWAYS ≥ 0 (0/300k violations)

Suggested approaches for the interior:
1. **Discriminant of the cubic**: Δ = 18g3g2g1g0 - 4g2³g0 + g2²g1² - 4g3g1³ - 27g3²g0².
   If Δ ≤ 0: one real root, g > 0 on [0, T_root) ⊃ [0, tmax].
   But Δ > 0 occurs ~1.9% of the time.

2. **Critical point value**: When g has a local min at T* = (-g2 - √(g2²-3g3g1))/(3g3),
   compute g(T*) and show it factors through positive terms.

3. **Polynomial bound**: Show g(T) ≥ h(T) for some simpler h that's ≥ 0.

4. **Direct SOS/Sturm**: Use Sturm's theorem to count roots in [0, tmax].

#### 3c. Off-diagonal closure

After proving 3a and 3b, need to show the global minimum of K_red(p,q) for
p,q ≥ 0 on the feasible box is at the boundary (edges or diagonal), not at
an interior point. This might follow from concavity/convexity structure, or
might require a separate argument.

Alternatively, a unified approach: prove K_red(α,β) ≥ 0 for α,β ≥ 0 directly,
without reducing to 1D slices. K_red(α,β) is a degree-6 polynomial in (α,β):

```
K_red(α,β) = a00 + b00·αβ + a10·α² + a01·β²
             + 2a20·α³β + 2a02·αβ³
             + a20·α⁴ + a11·α²β² + a02·β⁴
             + a12·α²β²(α+β)²
```

## Scripts to Reference

- `scripts/explore-p4-path2-codex-handoff2.py` — Cycle 2: Claims 1+2 proofs
- `scripts/explore-p4-path2-gap-analysis.py` — gap confirmation + g(Pmax)/g(Qmax)
- `scripts/explore-p4-path2-gap-analysis2.py` — cubic interior analysis

## Key Identities (use `build_exact_k_red()` from Cycle 2 script)

```python
# g(T) coefficients:
g0 = a00  # ≥ 0 always
g1 = a10 + a01 + b00
g2 = delta1 + 4*a20 + 4*a02  # = a11 + 3*(a20+a02)
g3 = 4*a12  # = -5184*r*L

# Endpoint targets (VERIFIED):
gP_target = 32*(x-1)*(3*x+1)**2*(3*y+1)*D*pos_core*new_P/27
gQ_target = -32*r**4*(3*x+1)*(y-1)*(3*y+1)**2*D*pos_core*new_Q/27
# where new_P = (r+1)*W + 12*(1-x), new_Q = (r+1)*W + 12*r**3*(1-y)

# Upper bounds when active (VERIFIED):
# D >= 0: new_P <= r*(3r(y-1)(r-1)^2 - 4(r+1)) <= 0
# D <= 0: new_Q <= (-3(1-x)(r-1)^2/r - 4r(r+1)) <= 0
```

## Numeric Evidence

| Test | Samples | Violations | Min value |
|------|---------|------------|-----------|
| K_red same-sign | 200k | 0 | 8.6e-20 |
| g(T) diagonal | 200k | 0 | 3.8e-23 |
| g interior min | 300k | 0 | 1.8e-10 |
| A(P,0) edge | (implicit) | 0 | — |

## Proof Ledger Update

On success of Claim 3, the full proof chain is:
1. C1 ≥ 0 (Claim 1, Cycle 2) — proved
2. C0(p,-p) ≥ 0 (Claim 2, Cycle 2) — proved
3. K_red(p,q) ≥ 0 for same-sign p,q (Claim 3, Cycle 3) — **target**

Together: K_red ≥ 0 for ALL feasible (p,q), completing the T2+R surplus proof.
