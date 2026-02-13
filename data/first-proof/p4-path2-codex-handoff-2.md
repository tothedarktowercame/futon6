# P4 Path 2 Cycle 2: Codex Handoff — C1·u² + C0 Decomposition

## What Changed Since Cycle 1

Codex Cycle 1 found the A+pq·B decomposition and the coefficient identities
B₀₁=2A₀₂, B₁₀=2A₂₀, B₁₁=2A₁₂=2A₂₁. Claude used these identities to discover:

**K_red = C1·(p+q)² + C0**

where C1 ≥ 0 on ALL 500k feasible samples (not just L≤0). This reduces
the proof to two clean algebraic claims.

## The Decomposition (VERIFIED ALGEBRAICALLY)

Using the exact K_red from Codex's `build_exact_k_red()` in
`scripts/explore-p4-path2-codex.py`:

```python
# A(P,Q) + pq·B(P,Q) decomposition (Codex Cycle 1)
# With P=p², Q=q². Coefficient identities:
#   B01 = 2*A02, B10 = 2*A20, B11 = 2*A12, A21 = A12

# Since A21 = A12, the (p+q)² factorization works:
# K_red = C1·(p+q)² + C0  where:

C1 = a20*p**2 + a02*q**2 + a12*p**2*q**2
#   = a20*P + a02*Q + a12*P*Q   (in P,Q variables)

C0 = a00 + b00*p*q + a10*p**2 + a01*q**2 + delta1*p**2*q**2
#   delta1 = a11 - a20 - a02
```

This was VERIFIED: `expand(C1*(p+q)**2 + C0 - K_red) == 0`.

## Exact Factored Coefficients

All coefficients are polynomials in (r,x,y) only:

```
a20 = -72*r⁴*(y-1)*(81r²xy³ + 81r²xy² + 27r²xy + 3r²x + 27r²y³ + 27r²y²
      + 9r²y + r² + 108rxy² - 36rxy + 24rx + 72ry² + 12ry + 12r + 81x²y²
      + 54x²y + 45x² + 54xy² - 72xy - 6x + 9y² + 18y + 9)
    = 72*r⁴*(1-y)*M_y    where M_y > 0 on feasible

a02 = -72*(x-1)*(81r²x²y² + 54r²x²y + 9r²x² + 54r²xy² - 72r²xy + 18r²x
      + 45r²y² - 6r²y + 9r² + 108rx²y + 72rx² - 36rxy + 12rx + 24ry + 12r
      + 81x³y + 27x³ + 81x²y + 27x² + 27xy + 9x + 3y + 1)
    = 72*(1-x)*M_x    where M_x > 0 on feasible

a12 = a21 = -1296*r*L    (PROVED: same L as d6 factorization)

delta1 = 24*W*(positive 61-term polynomial)    where W < 0 always on feasible
       → delta1 < 0 ALWAYS

a00 = -32*r⁴*(x-1)*(y-1)*W*(big positive polynomial)/27
    → a00 ≥ 0 ALWAYS (product of negatives gives positive: (x-1)<0, (y-1)<0, W<0)

b00 = -32*r³*(x-1)*(y-1)*(positive polynomial)
    → b00 ≤ 0 ALWAYS (≤0 since (x-1)(y-1)>0 on feasible, times negative)
```

Where:
```
L = 9x² - 27xy(1+r) + 3x(r-1) + 9ry² - 3ry + 2r + 3y + 2
W = 3r²(y-1) + 3(x-1) - 4r    (< 0 always on feasible interior)
```

## Numerical Evidence (500k feasible samples)

| Quantity | Sign on feasible | Notes |
|----------|-----------------|-------|
| a20 | ≥ 0 always | 0/500k negative |
| a02 | ≥ 0 always | 0/500k negative |
| a12 | < 0 when L>0 | = -1296rL |
| **C1** | **≥ 0 always** | **0/500k negative** |
| a00 | ≥ 0 always | 0/500k negative |
| b00 | ≤ 0 always | 0/500k positive |
| a10 | indefinite | <0 for 14% of samples |
| a01 | indefinite | <0 for 14% of samples |
| delta1 | < 0 always | = 24·W·(positive) |
| C0 | indefinite | <0 for 17% of samples |
| K_red = C1u²+C0 | **≥ 0 always** | 0/500k negative |

## What Codex Should Prove

### Claim 1: C1 ≥ 0 on feasible domain

```
C1 = 72r⁴(1-y)M_y · p² + 72(1-x)M_x · q² - 1296rL · p²q²
```

When L ≤ 0: all three terms ≥ 0, done.

When L > 0: need `a20·p² + a02·q² ≥ 1296rL·p²q²`.
Dividing by p²q²: `a20/q² + a02/p² ≥ 1296rL`.
By AM-GM or direct: use `q² ≤ 2r³(1-y)/9` so `a20/q² ≥ a20·9/(2r³(1-y)) = 72r⁴(1-y)M_y·9/(2r³(1-y)) = 324rM_y`.
Need: `324M_y ≥ 1296L`, i.e., `M_y ≥ 4L`.

**Suggested approach**: Show M_y - 4L is a sum of squares or has all non-negative
coefficients when expressed in the right basis. M_y has ~25 terms, L has 10 terms.

### Claim 2: C0(p,-p) ≥ 0 for feasible p (the u=0 boundary)

At p = -q: u = p+q = 0, so K_red = C0(p,-p).

```
C0(p,-p) = a00 + (a10 + a01 + |b00|)·p² + delta1·p⁴
```

(Note: pq = -p² at q=-p, and b00 ≤ 0 so -b00·p² = |b00|p².)

This is a quadratic in t = p² with negative leading coefficient delta1:
```
f(t) = delta1·t² + (a10+a01+|b00|)·t + a00
```

Need f(t) ≥ 0 for t ∈ [0, t_max] where t_max = min(2(1-x)/9, 2r³(1-y)/9).

**Suggested approach**: Show the discriminant D = (a10+a01+|b00|)² - 4|delta1|·a00 ≤ 0,
which would mean f has no real roots and is non-negative everywhere in [0, ∞).
OR show f(t_max) ≥ 0 using the domain bound.

### Claim 3 (if needed): General C0 < 0 compensated by C1·u²

If Claims 1 and 2 don't suffice for the general case, show:
For any feasible (r,x,y,p,q), when C0(r,x,y,p,q) < 0:
```
C1(r,x,y,p,q) · (p+q)² ≥ |C0(r,x,y,p,q)|
```

This might follow from Claim 1 + Claim 2 + a monotonicity argument
(K_red is "most stressed" at the u=0 boundary).

## Scripts to Reference

- `scripts/explore-p4-n4-K-factored-uq.py` — the C1·u²+C0 decomposition (THIS SESSION)
- `scripts/explore-p4-path2-codex.py` — Codex Cycle 1: A+pq·B decomposition
- `scripts/explore-p4-n4-R-surplus-alg.py` — delta/W factorization, L⟹3x+3y≥2

## Proof Ledger Update Needed

On success, update `data/proof-state/P4.edn`:
- L-n4-T2R-surplus: status → :proved
- Add new items for C1≥0 and C0-boundary claims
- This closes L-n4-algebraic-cert → closes L-n4 (with Path 1 case3c)
