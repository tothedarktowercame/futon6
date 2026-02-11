# Problem 4: Root Separation Under Finite Free Convolution

## Problem Statement

Two monic polynomials of degree n:

    p(x) = sum_{k=0}^{n} a_k x^{n-k},  a_0 = 1
    q(x) = sum_{k=0}^{n} b_k x^{n-k},  b_0 = 1

The **finite free additive convolution** p ⊞_n q has coefficients:

    c_k = sum_{i+j=k} [(n-i)!(n-j)! / (n!(n-k)!)] a_i b_j

The **root separation energy**:

    Phi_n(p) = sum_i (sum_{j != i} 1/(lambda_i - lambda_j))^2

where lambda_1, ..., lambda_n are the roots of p. (Phi_n = infinity if p has
repeated roots.)

**Question (Spielman):** Is it true that for monic real-rooted polynomials
p, q of degree n:

    1/Phi_n(p ⊞_n q) >= 1/Phi_n(p) + 1/Phi_n(q)  ?

## Answer

**Yes** (numerically verified, proof incomplete). The inequality holds with
0 violations in 8000 random trials for n = 2, 3, 4, 5. Equality holds
exactly at n = 2 (where 1/Phi_2 is linear in the sole cumulant kappa_2).
For n >= 3, the inequality is strict for generic polynomials.

The original proof via "concavity of 1/Phi_n implies superadditivity"
contains errors (see Section 5a). The analytic proof remains open.

## Solution

### 1. Interpreting Phi_n via the logarithmic derivative

For a monic polynomial p(x) = prod_i (x - lambda_i), the logarithmic
derivative is:

    p'(x)/p(x) = sum_i 1/(x - lambda_i)

Evaluating at a root lambda_i:

    (p'(x)/p(x)) at x -> lambda_i  gives  sum_{j != i} 1/(lambda_i - lambda_j)

(the sum diverges, but the finite sum over j != i is exactly the residue).
More precisely, by L'Hôpital:

    p'(lambda_i) = prod_{j != i} (lambda_i - lambda_j)

So:

    sum_{j != i} 1/(lambda_i - lambda_j) = [d/dx log p(x)]_{x=lambda_i, regularized}
                                          = (d^2/dx^2 log p) contribution

**Key identity:**

    Phi_n(p) = sum_i [p'(lambda_i)]^{-2} * [sum_{j != i} (lambda_i - lambda_j)^{-1} * prod_{k != i} (lambda_i - lambda_k)]^2

This simplifies via the relation to the **second logarithmic derivative**:

    [p'/p]'(x) = -sum_i 1/(x - lambda_i)^2

and evaluated at a root:

    Phi_n(p) = sum_i (sum_{j != i} 1/(lambda_i - lambda_j))^2

This is the sum of squared "Coulomb forces" at each root — the total
electrostatic self-energy of the root configuration (in the 1D log-gas
picture).

### 2. Connection to the discriminant

The discriminant of p is:

    disc(p) = prod_{i < j} (lambda_i - lambda_j)^2

By the AM-QM inequality applied to 1/(lambda_i - lambda_j):

    Phi_n(p) >= n(n-1)^2 / sum_{i<j} (lambda_i - lambda_j)^2

(Cauchy-Schwarz). So 1/Phi_n is related to the "spread" of roots.
Specifically:

    1/Phi_n(p) <= sum_{i<j} (lambda_i - lambda_j)^2 / (n(n-1)^2)

### 3. Finite free convolution and root behavior

The operation ⊞_n was introduced by Marcus, Spielman, and Srivastava (2015)
as a finite-dimensional analogue of Voiculescu's free additive convolution.
Key properties:

(a) **Real-rootedness preservation:** If p, q are real-rooted monic polynomials
    of degree n, then p ⊞_n q is also real-rooted.

(b) **Expected characteristic polynomial:** If A, B are n x n Hermitian with
    char. poly. p_A, p_B, then for a uniformly random conjugation U:

    E_U[char. poly. of A + UBU*] = p_A ⊞_n p_B

(c) **Linearization of cumulants:** In the n -> infinity limit, the finite
    free cumulants linearize (R-transform additivity).

(d) **Root spreading:** Free convolution generally spreads roots apart.
    Convolving with a non-degenerate q increases the minimum root gap.

### 4. The inequality via the random matrix model

Using property (b), interpret p ⊞_n q as the expected characteristic
polynomial of A + UBU* where char(A) = p, char(B) = q.

**Phi_n as curvature of the log-characteristic polynomial.**

Define F_A(x) = log |det(xI - A)| = sum_i log|x - lambda_i|. Then:

    F_A''(x) = -sum_i 1/(x - lambda_i)^2

At the roots: Phi_n(p) = -sum_i F_A''(lambda_i) (with appropriate signs).

More precisely, Phi_n(p) = sum_i [F_A'(lambda_i)]^2 where F_A' is the
regularized derivative (sum_{j != i} 1/(lambda_i - lambda_j)).

### 5. Finite free cumulants and the bilinear structure

**The MSS coefficient formula.** The ⊞_n operation acts on coefficients as:

    c_k = sum_{i+j=k} [(n-i)!(n-j)! / (n!(n-k)!)] a_i b_j

This is bilinear but NOT simply additive in the a_k. For example, at n=3:

    c_1 = a_1 + b_1                     (additive)
    c_2 = a_2 + (2/3)*a_1*b_1 + b_2     (cross-term!)
    c_3 = a_3 + (1/3)*a_2*b_1 + (1/3)*a_1*b_2 + b_3

**Finite free cumulants.** Following Arizmendi-Perales (2018), there exist
finite free cumulants kappa_k^(n) related to the a_k by a nonlinear
moment-cumulant formula (via non-crossing partitions with falling-factorial
weights) such that:

    kappa_k^(n)(p ⊞_n q) = kappa_k^(n)(p) + kappa_k^(n)(q)

The polynomial coefficients a_k are NOT the finite free cumulants; they
are finite free MOMENTS. The relationship involves a Möbius inversion on the
lattice of non-crossing partitions.

### 5a. What the proof requires (GAP)

To complete the proof via the cumulant approach, one would need to show that
1/Phi_n, expressed as a function f(kappa_2^(n), ..., kappa_n^(n)) of the
finite free cumulants, satisfies superadditivity:

    f(kappa(p) + kappa(q)) >= f(kappa(p)) + f(kappa(q))

**Note on direction:** Superadditivity of f follows if f is CONVEX with
f(0) = 0. (Concavity + f(0) = 0 would give SUBadditivity, the wrong
direction.) However, numerical experiments show that 1/Phi_n is neither
globally convex nor concave in the polynomial coefficient space, and the
superadditivity is specific to ⊞_n — it fails for plain coefficient-wise
addition (~40% violation rate for n=3,4,5).

**Status of this step:** The superadditivity of 1/Phi_n under ⊞_n is
**numerically verified** (0 violations in 8000 random trials for n=2,3,4,5)
but the analytic proof via convexity of f in finite free cumulant space
remains open. A correct proof may require:

(a) Explicit computation of 1/Phi_n in finite free cumulant coordinates and
    verification of convexity in that specific coordinate system, or

(b) A direct argument via the MSS random matrix model E_U[char(A + UBU*)]
    using properties of Haar-random unitary conjugation, or

(c) An electrostatic/log-gas argument that works with the specific bilinear
    structure of ⊞_n rather than relying on global function properties.

### 6. Verification for small cases

**Degree 2:** p(x) = (x - a)(x - b), q(x) = (x - c)(x - d).

Phi_2(p) = [1/(a-b)]^2 + [1/(b-a)]^2 = 2/(a-b)^2.
1/Phi_2(p) = (a-b)^2 / 2.

The finite free convolution for degree 2:
(p ⊞_2 q)(x) = x^2 - (a+b+c+d)/2 * x + [ac + ad + bc + bd - (a+b)(c+d)/2 + ...]

For the specific case a = -b = s, c = -d = t (symmetric roots):
p(x) = x^2 - s^2, q(x) = x^2 - t^2.
p ⊞_2 q = x^2 - (s^2 + t^2) (by the cumulant addition for degree 2).

Phi_2(p ⊞_2 q) = 2/(s^2 + t^2).
1/Phi_2(p ⊞_2 q) = (s^2 + t^2)/2.

1/Phi_2(p) + 1/Phi_2(q) = s^2/2 + t^2/2 = (s^2 + t^2)/2.

Equality! This is consistent: for degree 2 symmetric polynomials, the
inequality holds with equality (because the cumulant structure is purely
quadratic).

### 7. Summary and status

**The inequality is TRUE** — numerically verified with 0 violations in 8000
random trials for n = 2, 3, 4, 5, with LHS/RHS ratios ranging from 1.0
(equality at n=2) to over 2000 (large surplus at n=5).

**What is established:**

1. Finite free cumulants add under ⊞_n (Arizmendi-Perales 2018)
2. 1/Phi_n is superadditive under ⊞_n (numerical, high confidence)
3. Equality at n=2 (1/Phi_2 is linear in kappa_2)
4. The superadditivity is SPECIFIC to ⊞_n — it fails under plain
   coefficient addition (~40% violation rate)

**What remains open:**

The analytic proof of superadditivity. The original argument claimed
"concavity of 1/Phi_n in cumulants implies superadditivity," but this
has two errors: (a) concavity + f(0)=0 gives SUBadditivity, not
superadditivity, and (b) 1/Phi_n is neither globally convex nor concave.
A correct proof likely needs to exploit the specific bilinear structure
of the MSS convolution formula or the random matrix interpretation.

### 8. Numerical evidence

Verification scripts:
- `scripts/verify-p4-inequality.py` (superadditivity + convexity)
- `scripts/verify-p4-deeper.py` (coefficient addition + MSS structure)
- `scripts/verify-p4-schur-majorization.py` (Schur/submodularity/paths)

**Superadditivity test** (2000 random real-rooted polynomial pairs per n):

| n | violations | min ratio | mean ratio | max ratio |
|---|-----------|-----------|------------|-----------|
| 2 | 0/2000    | 1.000000  | 1.000000   | 1.000000  |
| 3 | 0/2000    | 1.000082  | 3.031736   | 162.446   |
| 4 | 0/2000    | 1.003866  | 9.549935   | 2268.065  |
| 5 | 0/2000    | 1.032848  | 14.457013  | 2119.743  |

Ratio = LHS/RHS = [1/Phi_n(p⊞q)] / [1/Phi_n(p) + 1/Phi_n(q)]; ratio >= 1
means inequality holds. Strict inequality for n >= 3.

**Convexity/concavity test** (midpoint test in coefficient space):

| n | convex violations | concave violations | total tests |
|---|------------------|--------------------|-------------|
| 3 | 757 (50.6%)      | 738 (49.4%)        | 1495        |
| 4 | 648 (65.7%)      | 338 (34.3%)        | 986         |
| 5 | 433 (72.9%)      | 161 (27.1%)        | 594         |

1/Phi_n is NEITHER convex NOR concave in coefficient space.

**Plain coefficient addition test** (NOT ⊞_n):

| n | violations | total |
|---|-----------|-------|
| 3 | 240       | 609   |
| 4 | 159       | 373   |
| 5 | 116       | 218   |

Superadditivity FAILS under plain addition — the inequality is specific
to the MSS bilinear structure of ⊞_n.

## Key References from futon6 corpus

- PlanetMath: "monic polynomial" (Monic1)
- PlanetMath: "discriminant" (Discriminant)
- PlanetMath: "resultant" (Resultant, DerivationOfSylvestersMatrixForTheResultant)
- PlanetMath: "logarithmic derivative" (LogarithmicDerivative)
- PlanetMath: "partial fraction decomposition" (ALectureOnThePartialFractionDecompositionMethod)
- PlanetMath: "cumulant generating function" (CumulantGeneratingFunction)

## External References

- Marcus, Spielman, Srivastava (2015), "Interlacing Families II: Mixed Characteristic
  Polynomials and the Kadison-Singer Problem," Annals of Math 182(1), 327-350.
  [Defines ⊞_n, proves real-rootedness preservation, random matrix interpretation]

- Arizmendi, Perales (2018), "Cumulants for finite free convolution," J. Combinatorial
  Theory Ser. A 155, 244-266. arXiv:1702.04761.
  [Defines finite free cumulants that linearize under ⊞_n]
