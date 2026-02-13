# P4 Path 2: Codex Handoff — Prove K_T2R/r² ≥ 0

## Problem Statement

Prove the following polynomial inequality on the feasible domain:

```
K_T2R_num(r,x,y,p,q) ≥ 0
```

where `K_T2R_num = 2·L·f2p·f2q·f2c + r·(1+3x)(1+3y)·f1c·R_surplus_num`

This is the **unified numerator** of the T2+R surplus in the 3-piece Cauchy-Schwarz
decomposition of the finite Stam inequality for n=4.

## Definitions

```python
r, x, y, p, q = symbols('r x y p q')

# f1, f2 components for polynomials p, q, and combined c = p ⊞ q
f1p = 1 + 3*x
f2p = 2*(1-x) - 9*p**2
f1q = r**2*(1 + 3*y)
f2q = 2*r**3*(1-y) - 9*q**2

Sv = 1 + r
Av12 = 3*x + 3*y*r**2 + 2*r
f1c = Sv**2 + Av12                              # = 1+4r+r²+3x+3yr²
f2c = 2*Sv**3 - 2*Sv*Av12/3 - 9*(p+q)**2

# Cauchy-Schwarz coefficients
C_p = x - 1                  # always < 0 on feasible
C_q = r**2*(y - 1)           # always < 0 on feasible
C_c = Av12/3 - Sv**2         # combined

# The key polynomial L (controls T2 sign and d6 sign)
L = 9*x**2 - 27*x*y*(1+r) + 3*x*(r-1) + 9*r*y**2 - 3*r*y + 2*r + 3*y + 2

# R_surplus_num (92 terms, degree 4 in p,q):
R_surplus_num = expand(C_c*f1c*f2p*f2q - C_p*f1p*f2c*f2q - C_q*f1q*f2c*f2p)

# K_T2R unified numerator:
K_T2R_num = expand(2*L*f2p*f2q*f2c + r*(1+3*x)*(1+3*y)*f1c*R_surplus_num)
```

### Where this comes from

```
T2_surplus = 2L / [9r·(1+3x)·(1+3y)·f1c]     # T2_surplus_num = 4rL (PROVED)
R_surplus  = R_surplus_num / [9·f2p·f2q·f2c]   # (92 terms)

T2+R = K_T2R_num / [9r·(1+3x)(1+3y)·f1c·f2p·f2q·f2c]
```

Denominator is positive on the feasible domain. So T2+R ≥ 0 ⟺ K_T2R_num ≥ 0.

## Feasible Domain

```
r > 0                          # ratio of leading coeff sums
x ∈ (0, 1)                    # normalized discriminant parameter for p
y ∈ (0, 1)                    # normalized discriminant parameter for q
f2p > 0:  9p² < 2(1-x)        # real-rootedness of p
f2q > 0:  9q² < 2r³(1-y)      # real-rootedness of q
f2c > 0:  9(p+q)² < 2(1+r)³ - 2(1+r)(3x+3yr²+2r)/3   # real-rootedness of p⊞q
```

Equality at: r=1, x=y=1/3, p=q=0 (here K_T2R_num = 0).

## Structural Discoveries (USE THESE)

### 1. K_T2R_num is divisible by r² (probably)

Check this. If so, work with K_red = K_T2R_num / r².

### 2. d6 factorization (PROVED)

The degree-6 part in (p,q) of K_T2R/r² factors completely:

```
d6 = -1296·r·L·p²q²(p+q)²
```

When L ≤ 0: d6 ≥ 0. When L > 0: d6 < 0.

### 3. The polynomial W = 3r²(y-1) + 3(x-1) - 4r

This is **always negative** on the feasible interior (since x<1, y<1, r>0).

### 4. δ factorization (PROVED)

The "quartic correction" δ = c22 - c40 - c04 (where cij are coefficients of
p^i·q^j in R_surplus_num) factors as:

```
δ = 27·W·(3r²y + r² + 4r + 3x + 1)
```

Always negative (W < 0, second factor > 0).

### 5. L ≤ 0 implies 3x+3y ≥ 2 (PROVED)

On the line x+y = 2/3:
```
L|_{y=2/3-x} = 4(r+1)(3x-1)²  ≥ 0
```
with discriminant = 0 (perfect square!). So L can only be ≤ 0 when x+y > 2/3.

### 6. T2 and R surpluses are mutually protective (NUMERICALLY VERIFIED)

- T2_surplus < 0 and R_surplus < 0 are **mutually exclusive** (0/200000)
- When T2 < 0: R/|T2| ≥ 1.09
- When R < 0: T2/|R| ≥ 1.09

### 7. Quartic form decomposition

The degree-4 part of R_surplus_num satisfies c31/c40 = c13/c04 = 2, giving:

```
Q4 = (c40·p² + c04·q²)(p+q)² + δ·p²q²
```

where c40 = 81r⁴(1-y)(3y+1) ≥ 0, c04 = 81(1-x)(3x+1) ≥ 0, δ < 0.

### 8. Numerical evidence

- K_T2R/r² ≥ 0: 200k/200k feasible samples, min ≈ 6.74e-05
- R_surplus_num ≥ 0 when L ≤ 0: 276k/276k samples
- R_surplus_num < 0 when L > 0: 8193/87680 (T2 compensates)

## What Codex Should Do

### Task 1: Compute K_T2R_num and verify structure

1. Compute K_T2R_num from the formula above
2. Check r² divisibility → compute K_red = K_T2R_num/r²
3. Count terms, verify degree 6 in (p,q)
4. Verify the d6 factorization: degree-6 part = -1296rL·p²q²(p+q)²

### Task 2: Analyze in (u,v) = (p+q, p-q) coordinates

Substitute p = (u+v)/2, q = (u-v)/2 into K_red.
Factor each u^i·v^j coefficient in (r,x,y).
Check which coefficients are divisible by W and/or L.

### Task 3: Try to write K_red as a sum of non-negative terms

Ideas:
- K_red = (positive part in u²) + (positive part in v²) + (cross terms)
- Use the quartic decomposition Q4 = (c40p²+c04q²)(p+q)² + δp²q²
- Exploit W < 0 everywhere and L sign cases
- Try completing the square in u = p+q first, then handle v = p-q

### Task 4: Domain-constrained SOS

If K_red is not globally SOS, try:
```
K_red = σ₀ + σ₁·(Pmax - p²) + σ₂·(Qmax - q²) + σ₃·(Umax - (p+q)²)
```
where σᵢ are SOS in (p,q), Pmax = 2(1-x)/9, Qmax = 2r³(1-y)/9.

### Task 5: If needed, prove the two cases separately

**Case L ≤ 0**: d6 ≥ 0, so K_red ≥ 0 iff d0+d2+d4 ≥ 0.
d0+d2+d4 has degree 4 in (p,q) and is non-negative on feasible (100k/100k).

**Case L > 0**: d6 < 0, T2_surplus > 0 compensates R_surplus < 0.

## Scripts to Reference

- `scripts/explore-p4-n4-R-surplus-proof.py` — R_surplus algebraic structure (92 terms)
- `scripts/explore-p4-n4-R-surplus-alg.py` — δ factorization, L⟹3x+3y≥2 proof, (u,v) analysis
- `scripts/explore-p4-n4-R-surplus-gram2.py` — Gram SOS test (FAILED: 0/50 PSD)
- `scripts/explore-p4-n4-T2R-partition.py` — T2/R mutual exclusion, L=T2_surplus/4r
- `scripts/explore-p4-n4-K-T2R-fast.py` — even-odd decomposition, d6 factoring
- `scripts/explore-p4-n4-cauchy-deep.py` — full 3-piece decomposition structure

## Failed Routes (Don't Repeat These)

1. **Global SOS for R_surplus_num**: Gram matrix 0/50 PSD, eigenvalues ~ -400k
2. **d0+d2+d4 globally non-negative**: FALSE — d4 quartic is indefinite (negative at 69%)
3. **Hilbert 1888 for d0+d2+d4**: Inapplicable since it's not globally non-negative
4. **R_surplus_num divisible by L**: FALSE — L does not divide R_surplus_num
5. **Quartic form = perfect square (p²+pq+q²)²**: FALSE — c22-3c40 ≠ 0

## Proof Ledger State

See `data/proof-state/P4.edn`:
- L-n4-titu-split: PROVED (T1_surplus ≤ 0)
- L-n4-d6-factored: PROVED (d6 = -1296rL·p²q²(p+q)²)
- L-n4-T2R-partition: numerically-verified (T2<0, R<0 mutually exclusive)
- L-n4-T2R-surplus: numerically-verified (200k/200k, needs algebraic proof)
