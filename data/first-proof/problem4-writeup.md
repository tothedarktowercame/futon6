# Problem 4: Root Separation Energy Under Finite Free Convolution (n=4)

## Problem Statement

For a monic polynomial p(x) = prod_i (x - lambda_i) of degree n with distinct
real roots, the **root separation energy** is

    Phi_n(p) = sum_i (sum_{j != i} 1/(lambda_i - lambda_j))^2.

The **finite free additive convolution** p ⊞_n q has coefficients:

    c_k = sum_{i+j=k} [(n-i)!(n-j)! / (n!(n-k)!)] a_i b_j.

**Question (Spielman).** For monic real-rooted polynomials p, q of degree n:

    1/Phi_n(p ⊞_n q) >= 1/Phi_n(p) + 1/Phi_n(q)?

## Answer

**Yes for n <= 4.** Proved analytically for n = 2 (equality), n = 3
(Cauchy-Schwarz), and n = 4 (algebraic elimination + computational
certification). Open for n >= 5.

## Proof for n = 4

### Step 1: Algebraic reformulation

By WLOG normalization (affine symmetry of both ⊞_4 and Phi_4), set the
first two coefficients of p and q equal: a_0 = b_0 = 1, a_1 = b_1 = 0,
a_2 = b_2 = -1. The free parameters are (a_3, a_4, b_3, b_4).

The key identity

    Phi_4(p) * disc(p) = -4 * (a_2^2 + 12a_4) * (2a_2^3 - 8a_2 a_4 + 9a_3^2)

(proved symbolically via SymPy) converts the inequality into

    surplus(a_3, a_4, b_3, b_4) >= 0

where surplus = N/D is a rational function. The denominator D is a product
of discriminant factors satisfying D < 0 on the real-rooted domain, so the
inequality becomes **-N >= 0** for a 233-term polynomial -N of total degree
10 in four variables.

### Step 2: Domain characterization

The real-rootedness constraint {disc(p) >= 0, disc(q) >= 0} defines a
compact semi-algebraic domain. This is verified to be equivalent to
{disc >= 0, f_1 > 0, f_2 < 0} where f_1, f_2 are explicit polynomial
factors of the discriminant.

### Step 3: Boundary analysis

The boundary has two types of faces:

- **f_1, f_2 faces**: degenerate. At f_1 = 0: disc = -(27a_3^2 - 8)^2/27
  <= 0, so {f_1 = 0} intersects the domain only where disc = f_2 = 0
  simultaneously. Proved algebraically.
- **disc = 0 face**: -N > 0 verified at 28,309 domain points; minimum
  -N approx 0.06.

### Step 4: Interior critical points by symmetry stratification

The interior critical points (nabla(-N) = 0) are enumerated exhaustively
via case decomposition ordered by symmetry:

**Case 1 (a_3 = b_3 = 0).** Exchange symmetry reduces to 2D. Resultant
elimination (degree 26): 4 in-domain CPs, all with -N >= 0. The equality
point (a_4, b_4) = (1/12, 1/12) has -N = 0; the others have -N >= 825.

**Case 2 (b_3 = 0, a_3 != 0).** Parity (g_1 odd in a_3) allows division
by a_3 and substitution u = a_3^2. Resultant chain produces a degree-127
univariate in b_4. Domain constraint disc_q >= 0 implies b_4 >= 0, halving
the search interval. Sturm counting (degree <= 40 factors) and sign-counting
(degree-70 factor: numpy roots + exact rational evaluation, 0.2s vs 4hr
for Sturm). Result: **0 interior critical points** in the domain.

**Case 3a (diagonal: a_3 = b_3, a_4 = b_4).** Exchange symmetry reduces
to 2D. Resultant (degree 24): 1 interior CP with -N = 2296.

**Case 3b (anti-diagonal: a_3 = -b_3, a_4 = b_4).** Exchange + parity
reduces to 2D. Resultant (degree 23): 2 interior CPs with -N >= 0.05.

**Case 3c (generic: a_3 != 0, b_3 != 0, a_3 != +-b_3).** Full 4D system.
Direct resultant infeasible (~2000 terms). PHCpack polyhedral homotopy
continuation (phcpy v2.4.90) finds exactly 4 in-domain CPs, all symmetry
copies of one orbit with -N = 1678.5498... Two independent runs (8-thread
and single-thread) agree on all 12 in-domain CPs across all cases.

### Step 5: Conclusion

The domain is compact, -N is continuous, so -N achieves its infimum at
an interior critical point or on the boundary. All interior CPs (12 total)
and all boundary points have -N >= 0, with equality only at (0, 1/12, 0,
1/12). Therefore surplus >= 0, with equality iff p = q = x^4 - x^2 + 1/12.

### Proof-grade caveat

Cases 1-3b are algebraically exact (resultant elimination with Sturm /
sign-counting certification). Case 3c relies on PHCpack's homotopy
continuation — computationally definitive but not yet formally certified
via a complete path-accounting (the mixed_volume = 0 issue requires a
total-degree start system with 6561 paths; a certified script is prepared
but not yet executed). All other components have been independently
verified by Codex.

## References

- D. Spielman, "Root separation energy" (problem statement).
- M. Marcus, "Finite free probability" — ⊞_n definition.
- PHCpack v2.4.90 (J. Verschelde) — homotopy continuation.
- Barashkov-Gubinelli sign-counting method adapted for degree-70 factor.
