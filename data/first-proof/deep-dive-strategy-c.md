# Deep Dive: Strategy C -- Direct Algebraic / SOS for n=4

**Date:** 2026-02-12
**Status:** Symmetric subfamily (a3=b3=0) PROVED. General case structured with clear path.

---

## 1. Executive Summary

This report documents a systematic computational and theoretical exploration
of Strategy C for proving the finite free Stam inequality at n=4. The key
results are:

1. **Key identity discovered:** `Phi_4(p) * disc(p) = -4 * (a2^2 + 12*a4) * (2*a2^3 - 8*a2*a4 + 9*a3^2)` -- the n=4 analog of the n=3 identity `Phi_3 * disc = 18 * a2^2`.

2. **Symmetric subfamily PROVED:** For centered quartics with a3=b3=0 (symmetric quartics p(x)=x^4+a2x^2+a4), the superadditivity 1/Phi_4(p boxplus q) >= 1/Phi_4(p) + 1/Phi_4(q) holds, with equality iff both are proportional to x^4-x^2+1/12.

3. **Equality characterizer identified:** The polynomial x^4-x^2+1/12 plays the role of the "degree-4 semicircular distribution" (roots approximately +/-0.303, +/-0.953).

4. **General case structured:** After centering and scaling, the surplus depends on 4 free parameters. The polynomial numerator is NOT SOS, requiring a Positivstellensatz-style certificate. Feasible with SDP tools.

5. **SOS tool landscape mapped:** Python tools (posipoly, CVXPY+MOSEK), Julia tools (SumOfSquares.jl, TSSOS), and CAD implementations available.

---

## 2. Setup: The n=4 Normalized Problem

### 2.1 Centering

Since Phi_n depends only on root differences (translation-invariant) and the MSS
convolution boxplus_n commutes with translation, we may assume WLOG a_1 = b_1 = 0
(centered polynomials):

    p(x) = x^4 + a_2 x^2 + a_3 x + a_4
    q(x) = x^4 + b_2 x^2 + b_3 x + b_4

### 2.2 MSS Convolution for Centered Quartics

The finite free additive convolution p boxplus_4 q has coefficients c_k = sum_{i+j=k} w(4,i,j) a_i b_j where w(n,i,j) = (n-i)!(n-j)!/(n!(n-k)!). For centered degree-4 polynomials:

    c_0 = 1                        (monic)
    c_1 = 0                        (centered)
    c_2 = a_2 + b_2                (purely additive)
    c_3 = a_3 + b_3                (purely additive)
    c_4 = a_4 + (1/6)*a_2*b_2 + b_4   (cross-term!)

The bilinear cross-term (1/6)*a_2*b_2 in c_4 is essential: plain coefficient
addition (without this term) violates the superadditivity ~29% of the time.

### 2.3 Scaling Reduction

Under x -> t*x, the coefficients scale as a_k -> a_k/t^k and Phi_n scales
as Phi_n -> Phi_n/t^2 (since each 1/(l_i - l_j) scales as 1/t). So 1/Phi_n
scales as t^2. Setting t = sqrt(-a_2) normalizes a_2 = -1 for each polynomial
independently.

**After centering and scaling:** a_2 = b_2 = -1. The surplus

    Delta_4 = 1/Phi_4(p boxplus q) - 1/Phi_4(p) - 1/Phi_4(q)

depends on 4 free parameters: (a_3, a_4, b_3, b_4).

The convolution with a_2 = b_2 = -1 gives: c_2 = -2, c_3 = a_3 + b_3,
c_4 = a_4 + 1/6 + b_4.

For real-rootedness with a_2 = -1: need disc > 0 and appropriate sign conditions.
For the symmetric case (a_3 = 0): need 0 < a_4 < 1/4.

---

## 3. The Key Identity: Phi_4 * disc

### 3.1 Discovery

**Theorem (computational).** For a centered monic quartic p(x) = x^4 + a_2 x^2 + a_3 x + a_4 with distinct real roots:

    Phi_4(p) * disc(p) = -4 * (a_2^2 + 12*a_4) * (2*a_2^3 - 8*a_2*a_4 + 9*a_3^2)

**Verification:** Tested against root-based computation for 200+ random quartics. Maximum relative error: 2.78e-14. Also verified symbolically for the symmetric subfamily (a_3 = 0).

### 3.2 Comparison with n=3

| n | Identity | Factored form |
|---|---------|---------------|
| 3 | Phi_3 * disc = 18 * a_2^2 | Monomial in a_2 |
| 4 | Phi_4 * disc = -4*(a_2^2+12*a_4)*(2*a_2^3-8*a_2*a_4+9*a_3^2) | Product of two polynomials |

The n=4 identity is more complex: two factors instead of a monomial. Both factors have definite signs on the real-rooted cone:

- Factor f_1 = a_2^2 + 12*a_4 > 0 always (0% negative in 9665 random tests)
- Factor f_2 = 2*a_2^3 - 8*a_2*a_4 + 9*a_3^2 < 0 always (100% negative in 9665 tests)

Since Phi_4 > 0 and disc > 0, the product must be positive, confirming -4*f_1*f_2 > 0, i.e., f_1*f_2 < 0 (consistent since f_1 > 0, f_2 < 0).

### 3.3 Resulting Formula for 1/Phi_4

    1/Phi_4 = -disc / (4 * f_1 * f_2) = disc / (4 * f_1 * |f_2|)

This is a **rational function** of the coefficients (degree 6 numerator / degree 5 denominator in weighted degree), unlike the n=3 case where 1/Phi_3 was a polynomial.

### 3.4 Symmetric Subfamily (a_3 = 0)

For x^4 + a_2 x^2 + a_4 with roots +/-alpha, +/-beta:

    Phi_4 = (alpha^2+beta^2)*(alpha^4+14*alpha^2*beta^2+beta^4) / (2*alpha^2*beta^2*(alpha-beta)^2*(alpha+beta)^2)

In coefficient form:

    Phi_4 = (-a_2)*(a_2^2+12*a_4) / (2*a_4*(a_2^2-4*a_4))
    disc = 16*a_4*(a_2^2-4*a_4)^2

Both verified symbolically and numerically (match to machine precision).

---

## 4. Proof for the Symmetric Subfamily (a_3 = b_3 = 0)

### 4.1 Statement

**Theorem.** Let p(x) = x^4 + a_2 x^2 + a_4 and q(x) = x^4 + b_2 x^2 + b_4 be centered monic quartics with distinct real roots. Then

    1/Phi_4(p boxplus_4 q) >= 1/Phi_4(p) + 1/Phi_4(q)

with equality if and only if both p and q are proportional (under the scaling x -> tx) to x^4 - x^2 + 1/12.

### 4.2 Proof

**Step 1: Normalize.** Scale p and q independently to a_2 = b_2 = -1. Define s = a_4, t = b_4 with 0 < s, t < 1/4. The convolution has c_2 = -2, c_4 = s + 1/6 + t.

**Step 2: Compute the surplus.** Using the formula 1/Phi_4 = 2*a_4*(a_2^2-4*a_4) / ((-a_2)*(a_2^2+12*a_4)) for symmetric quartics:

    1/Phi(p) = 2s(1-4s) / (1+12s)
    1/Phi(q) = 2t(1-4t) / (1+12t)
    1/Phi(conv) = (6s+6t+1)(5-6s-6t) / (54+108s+108t)

The surplus reduces to:

    Delta_4 = N(s,t) / [54 * (12s+1) * (12t+1) * (2s+2t+1)]

where the denominator is manifestly positive (all factors positive for s,t > 0), and

    N(s,t) = 5184*s^3*t + 432*s^3 + 10368*s^2*t^2 + 3024*s^2*t + 468*s^2
           + 5184*s*t^3 + 3024*s*t^2 - 1800*s*t - 24*s + 432*t^3 + 468*t^2 - 24*t + 5

N is symmetric in (s,t). Its only interior critical point in (0,1/4)^2 is at (s,t) = (1/12, 1/12), where N = 0.

**Step 3: Change coordinates.** Let u = s - 1/12, v = t - 1/12, and define w = u + v, r = uv. Then:

    N = 144 * F(w,r)

where

    F(w,r) = 6w^3 + 7w^2 + r * g(w),    g(w) = 36w^2 + 24w - 16

**Step 4: Monotonicity in r.** The coefficient of r is g(w) = 4(9w^2 + 6w - 4), which has roots w = (-1 +/- sqrt(5))/3. Since (-1+sqrt(5))/3 ~ 0.412 and the domain has w <= s + t - 1/6 < 1/4 + 1/4 - 1/6 = 1/3, we have g(w) < 0 throughout the domain w in [-1/6, 1/3].

Therefore F is **strictly decreasing in r** on the feasible set.

**Step 5: Minimize over r.** For fixed w, the constraint u,v >= -1/12 with u+v=w and uv=r gives r <= w^2/4 (maximum at u = v = w/2). Since F is decreasing in r:

    F(w,r) >= F(w, w^2/4) = 3w^2(w+1)(3w+1)

**Step 6: Non-negativity of the minimum.** On w in [-1/6, 1/3]:
- w^2 >= 0 (= 0 iff w = 0)
- w + 1 >= 5/6 > 0
- 3w + 1 >= 1/2 > 0

Therefore F(w, w^2/4) >= 0 with equality iff w = 0.

**Step 7: Conclusion.** F(w,r) >= 0 on the entire feasible domain, hence N(s,t) >= 0 and Delta_4 >= 0. Equality holds iff w = 0 and r = 0, i.e., s = t = 1/12, corresponding to both p and q being x^4 - x^2 + 1/12 (up to scaling). QED.

### 4.3 Diagonal Check

On the diagonal s = t, the surplus factors as:

    Delta_4|_{s=t} = (12s-1)^2 * (12s+5) / [54 * (4s+1) * (12s+1)]

which is manifestly non-negative with a double zero at s = 1/12.

### 4.4 Boundary Check

At s = 0 (boundary of domain):

    N(0,t) = 432*t^3 + 468*t^2 - 24*t + 5

This cubic has its only real root at t ~ -1.224, well below 0. So N(0,t) > 0 for all t > 0.

---

## 5. The General Case: Structure and Obstacles

### 5.1 Numerical Evidence

Root-based surplus computation (requiring all three polynomials p, q, p boxplus q to have distinct real roots):

| Test set | Trials | Violations | Min surplus |
|----------|--------|------------|-------------|
| a_2=b_2=-1, random (a3,a4,b3,b4) | 10,659 | 0 | 0.000229 |
| Random a_2, b_2 (root-based) | 46,754 | 0 | 0.000028 |
| Optimization (Nelder-Mead) | converged | 0 | 0.0 (at semicircular) |

The global minimum of Delta_4 is **exactly 0**, achieved at (a_3, a_4, b_3, b_4) = (0, 1/12, 0, 1/12) -- confirming the symmetric case equality characterizer.

### 5.2 Why the Symmetric Proof Does Not Immediately Generalize

For a_3, b_3 != 0:

1. **1/Phi_4 is a rational function**, not polynomial. The surplus Delta_4 is a rational function of 4 variables. After clearing denominators, we get a polynomial inequality on a semialgebraic set (the real-rooted cone).

2. **The surplus numerator is NOT a sum of squares** (as a polynomial in (s,t) even in the symmetric case). Proof: the SOS Gram matrix requires Q[3,3] = 0 (from the s^4 coefficient being 0) but also Q[1,3] = 216 != 0, which contradicts PSD. The positivity fundamentally depends on the domain constraint s, t > 0.

3. **The (w,r) monotonicity argument** exploits the linear dependence of N on r (the product uv). In the general case with a_3, b_3 != 0, the surplus has a more complex dependence structure.

### 5.3 What a General Proof Needs

After clearing denominators, the inequality becomes:

    P(a_3, a_4, b_3, b_4) >= 0   subject to   g_i(a_3, a_4, b_3, b_4) >= 0

where P is a polynomial in the coefficients and the constraints g_i encode:
- disc(p) > 0 (p has 4 distinct real roots)
- disc(q) > 0
- disc(p boxplus q) > 0
- Various factor signs

**Approach 1: Positivstellensatz.** By Stengle/Schmudgen, if P >= 0 on the semialgebraic set defined by {g_i >= 0}, there exists a certificate:

    P = sigma_0 + sum_i sigma_i * g_i + sum_{i<j} sigma_{ij} * g_i * g_j + ...

where sigma_k are SOS polynomials. This certificate can be found computationally via SDP.

**Approach 2: Direct algebraic manipulation.** Extend the (w,r) trick:
- Define symmetric/antisymmetric combinations of the a_3, b_3 variables
- Look for monotonicity in auxiliary variables
- Factor the numerator using the symmetry Delta_4(a_3,a_4,b_3,b_4) = Delta_4(-a_3,a_4,-b_3,b_4) (reflection symmetry)

**Approach 3: Perturbation from symmetric case.** The symmetric case Delta_4|_{a3=b3=0} >= 0 is proved. Show that introducing small a_3, b_3 only increases the surplus (or at least keeps it non-negative) by analyzing the Hessian with respect to (a_3, b_3) at the minimum.

---

## 6. SOS Tools and Computational Feasibility

### 6.1 Available Tools

| Tool | Language | Backend | Notes |
|------|----------|---------|-------|
| [posipoly](https://github.com/pettni/posipoly) | Python | MOSEK | Supports SOS and SDSOS |
| CVXPY + MOSEK | Python | MOSEK | General SDP; can encode SOS via Gram matrix |
| [SumOfSquares.jl](https://sums-of-squares.github.io/sos/) | Julia | Multiple | Full Positivstellensatz support |
| SOSTOOLS | MATLAB | SeDuMi/MOSEK | Mature, well-documented |
| [CAD (pure Python)](https://github.com/mmaaz-git/cad) | Python | SymPy | Cylindrical algebraic decomposition |
| ncpol2sdpa | Python | MOSEK | Noncommutative polynomial SDP |

### 6.2 Feasibility Assessment

For the symmetric case (2 variables, degree 4 polynomial): trivially within SDP solver capabilities, but we proved it directly.

For the general case (4 variables, estimated degree 8-12 polynomial after clearing denominators):
- **SOS/SDP:** The Gram matrix has dimension C(4+d/2, d/2) where d is the degree. For d=8 in 4 variables: monomial basis of degree 4 has C(8,4) = 70 terms, giving a 70x70 PSD matrix. **Feasible** with modern solvers.
- **DSOS/SDSOS:** LP/SOCP relaxations of SOS. Faster but may not find a certificate if one exists.
- **CAD:** Exact but doubly exponential in number of variables. For 4 variables: likely impractical.

**Recommended approach:** Use SumOfSquares.jl or CVXPY+MOSEK to search for a Positivstellensatz certificate. The domain constraints (discriminant positivity, factor signs) should be included as multipliers.

### 6.3 Problem Size Estimate

After normalizing a_2 = b_2 = -1:
- Surplus numerator: polynomial in (a_3, a_4, b_3, b_4) of weighted degree ~ 10-12
- Domain constraints: disc(p) = polynomial of degree 4 in (a_3, a_4)
- Full certificate: 4-variable SOS in degree 6 basis (monomial vector of size ~84)
- SDP matrix: 84 x 84 -> 3528 free variables
- **This is well within the capability of MOSEK** (which handles matrices up to ~1000x1000)

---

## 7. Lorentzian Polynomial Connection

### 7.1 Background

Branden and Huh (2020, Annals of Mathematics 192(3)) introduced Lorentzian polynomials: homogeneous polynomials whose Hessian has exactly one positive eigenvalue on the positive orthant. Key properties:
- Contains homogeneous stable polynomials
- Preserved by a large class of linear operators
- Connected to matroid theory and negative dependence

### 7.2 Relevance to Our Problem

**Direct applicability: LIMITED.** Our surplus Delta_4 is a **rational function**, not a polynomial. Lorentzian polynomial theory concerns polynomial positivity/negativity, not rational function positivity.

However, several indirect connections exist:

1. **MSS convolution preserves real-rootedness**, which is a Lorentzian-adjacent property. The class of real-rooted polynomials (viewed as a cone in coefficient space) has structural similarities to the Lorentzian cone.

2. **Operator preservation:** Branden-Huh show that operators preserving stable polynomials and non-negative coefficients also preserve Lorentzian polynomials. The MSS convolution operator might fall into this class (or a variant).

3. **Negative dependence:** Lorentzian polynomials satisfy log-concavity and ultra-log-concavity properties. Our Phi_4*disc factorization involves factors related to these properties.

4. **After clearing denominators:** the polynomial inequality P >= 0 on the real-rooted cone MIGHT have a certificate expressible in terms of Lorentzian polynomial theory, if P can be decomposed into products of Lorentzian factors.

### 7.3 Assessment

Using Lorentzian polynomial theory for this problem would require:
- Reformulating the rational surplus as a polynomial inequality
- Showing the resulting polynomial is in the cone of "Lorentzian-certifiable" non-negative polynomials
- This is a **research-level question**, not a straightforward application

**Verdict: not the most efficient path for n=4, but potentially important for the general-n theory.**

---

## 8. Code Snippets and Reproducibility

### 8.1 Computing Phi_4 from Roots

```python
def phi4_from_roots(roots):
    """Compute Phi_4 from root array."""
    n = len(roots)
    total = 0.0
    for i in range(n):
        s = sum(1.0/(roots[i] - roots[j]) for j in range(n) if j != i)
        total += s**2
    return total
```

### 8.2 MSS Convolution for Centered Quartics

```python
def mss_convolve_centered_4(a2, a3, a4, b2, b3, b4):
    c2 = a2 + b2
    c3 = a3 + b3
    c4 = a4 + (1.0/6.0)*a2*b2 + b4
    return c2, c3, c4
```

### 8.3 Verifying the Key Identity

```python
def verify_phi4_disc_identity(a2, a3, a4):
    """Check Phi_4 * disc = -4*(a2^2+12*a4)*(2*a2^3-8*a2*a4+9*a3^2)"""
    roots = np.roots([1, 0, a2, a3, a4])
    roots = sorted(r.real for r in roots)
    phi = phi4_from_roots(roots)
    disc = 256*a4**3 - 128*a2**2*a4**2 + 144*a2*a3**2*a4 \
         - 27*a3**4 + 16*a2**4*a4 - 4*a2**3*a3**2
    formula = -4*(a2**2+12*a4)*(2*a2**3-8*a2*a4+9*a3**2)
    return abs(phi*disc - formula) / max(abs(phi*disc), 1e-15)
```

### 8.4 Symmetric Case Surplus

```python
def surplus_symmetric(s, t):
    """Surplus for symmetric quartics with a2=b2=-1, a3=b3=0."""
    inv_p = 2*s*(1-4*s) / (1+12*s)
    inv_q = 2*t*(1-4*t) / (1+12*t)
    c4 = s + 1.0/6 + t
    inv_c = -(6*s+6*t-5)*(6*s+6*t+1) / (108*s+108*t+54)
    return inv_c - inv_p - inv_q
```

---

## 9. The Equality Characterizer: The Degree-4 Semicircular Polynomial

The polynomial

    p_sc(x) = x^4 - x^2 + 1/12

achieves equality in the Stam inequality for the symmetric subfamily. Its roots are:

    +/- sqrt((1 + sqrt(6)/3) / 2) ~ +/- 0.9530
    +/- sqrt((1 - sqrt(6)/3) / 2) ~ +/- 0.3029

Root gaps: approximately 0.650, 0.606, 0.650 (nearly uniform spacing).

Key properties:
- Phi_4 = 18
- 1/Phi_4 = 1/18 ~ 0.0556
- disc = 16*(1/12)*(1-4/12)^2 = 16/12 * 4/9 = 64/108 ~ 0.5926

This polynomial is the finite analog of the semicircular (Wigner) distribution, which achieves equality in Voiculescu's free Stam inequality. The connection:
- Free Stam equality: mu is semicircular
- n=3 finite Stam equality: p is proportional to x^3 - x (the degree-3 "semicircular", equally spaced roots -1, 0, 1)
- n=4 finite Stam equality: p is proportional to x^4 - x^2 + 1/12

---

## 10. Viability Assessment and Next Steps

### 10.1 What Has Been Achieved

| Result | Status | Method |
|--------|--------|--------|
| Key identity Phi_4*disc factorization | VERIFIED | Numerical (rel err < 3e-14) |
| Symmetric case (a3=b3=0) proof | COMPLETE | Monotonicity in (w,r) coordinates |
| Equality characterizer identified | COMPLETE | Optimization + explicit verification |
| General case numerical verification | COMPLETE | 46K+ root-based tests, 0 violations |
| N is not SOS (global) | PROVED | Gram matrix obstruction |
| SOS tool feasibility | ASSESSED | Problem size ~ 84x84 Gram matrix |

### 10.2 Probability of Success

- **Completing general n=4 proof via SDP/Positivstellensatz:** 55-65%
  - The problem size (4 variables, degree ~10) is well within SDP solver capabilities
  - The main risk is that the Positivstellensatz multiplier degree might be too high
  - The symmetric case proof provides a template for the certificate structure

- **Finding a human-readable proof for general n=4:** 25-35%
  - The (w,r) monotonicity trick may extend if we can identify the right auxiliary variables
  - The a_3-dependence might factor in a way that reduces to the symmetric case

- **Generalizing to all n:** 5-10%
  - The Phi_n*disc identity would need to be discovered for each n
  - The equality characterizer (degree-n semicircular) can likely be identified
  - A pattern might emerge from n=3,4 that suggests the general structure

### 10.3 Immediate Next Steps

1. **Symbolic verification of the key identity** -- prove Phi_4*disc = -4*f1*f2 algebraically (via resultant computation or direct SymPy simplification). This upgrades the identity from "numerically verified" to "theorem."

2. **Compute the general surplus numerator** -- clear denominators in Delta_4(a3, a4, b3, b4) to obtain a polynomial P. Determine its degree and monomial structure.

3. **Run SDP solver** -- use SumOfSquares.jl or CVXPY+MOSEK to search for a Positivstellensatz certificate for P >= 0 on the real-rooted domain.

4. **Investigate (w,r)-style reduction for general case** -- define w1 = a3+b3, w2 = a3-b3, w3 = a4+b4-1/6, w4 = a4-b4 and look for monotonicity/factoring in these coordinates.

5. **Compute Phi_5*disc** -- check if the factored identity generalizes. Conjecture: Phi_n*disc = (-1)^? * c_n * product of polynomial factors in the centered coefficients.

### 10.4 Risk Mitigation

Even if the full general proof for n=4 is not completed:

- The **symmetric case proof** is a publishable result on its own
- The **key identity** Phi_4*disc = -4*f1*f2 is independently interesting
- The **equality characterizer** x^4-x^2+1/12 establishes the degree-4 semicircular polynomial
- Combined with the existing n=2 (equality) and n=3 (Titu's lemma) proofs, this significantly advances the conjecture

---

## 11. References

### Primary Sources

- Marcus, Spielman, Srivastava (2015). [Finite free convolutions of polynomials](https://arxiv.org/abs/1504.00350). Probability Theory and Related Fields.
- Voiculescu (1998). The analogues of entropy and of Fisher's information measure in free probability theory, V. Inventiones Mathematicae 132.
- Arizmendi, Perales (2018). Cumulants for finite free convolution. J. Combinatorial Theory Ser. A 155, 244-266.
- Branden, Huh (2020). [Lorentzian polynomials](https://annals.math.princeton.edu/2020/192-3/p04). Annals of Mathematics 192(3).

### SOS/Optimization Tools

- [posipoly](https://github.com/pettni/posipoly) -- Python SOS/SDSOS optimization
- [SumOfSquares.jl](https://sums-of-squares.github.io/sos/) -- Julia SOS programming
- [DSOS/SDSOS paper](https://epubs.siam.org/doi/10.1137/18M118935X) -- LP/SOCP alternatives to SOS
- [CAD in Python](https://github.com/mmaaz-git/cad) -- Cylindrical algebraic decomposition (SymPy)
- Kunisky (2022). [Lecture Notes on Sum-of-Squares Optimization](http://www.kunisky.com/static/teaching/2022spring-sos/sos-notes.pdf)

### Positivity Certificates

- Stengle (1974). A Nullstellensatz and a Positivstellensatz in semialgebraic geometry.
- Schmudgen (1991). The K-moment problem for compact semi-algebraic sets.
- Putinar (1993). Positive polynomials on compact semi-algebraic sets.
- [Effective Positivstellensatz](https://arxiv.org/abs/2410.04845) -- recent work on rational certificates

### Background

- Anderson, Guionnet, Zeitouni (2010). An Introduction to Random Matrices, Ch. 4.
- [Zucker: Chebyshev via SOS](https://www.philipzucker.com/deriving-the-chebyshev-polynomials-using-sum-of-squares-optimization-with-sympy-and-cvxpy/) -- practical Python SOS tutorial
- [Matt Baker's blog on Lorentzian polynomials](https://mattbaker.blog/2019/08/30/lorentzian-polynomials/) -- accessible introduction

---

## Appendix A: Detailed Computation Log

### A.1 Phi_4 for Symmetric Quartics

For p(x) = x^4 + a_2 x^2 + a_4 with roots +/-alpha, +/-beta:

    S_1(alpha) = (5*alpha^2 - beta^2) / (2*alpha*(alpha^2 - beta^2))
    S_2(-alpha) = -(5*alpha^2 - beta^2) / (2*alpha*(alpha^2 - beta^2))
    S_3(beta) = (alpha^2 - 5*beta^2) / (2*beta*(alpha^2 - beta^2))
    S_4(-beta) = -(alpha^2 - 5*beta^2) / (2*beta*(alpha^2 - beta^2))

    Phi_4 = (alpha^2+beta^2)*(alpha^4+14*alpha^2*beta^2+beta^4) / (2*alpha^2*beta^2*(alpha-beta)^2*(alpha+beta)^2)

In coefficient coordinates:
- alpha^2 + beta^2 = -a_2
- alpha^2 * beta^2 = a_4
- (alpha^2 - beta^2)^2 = a_2^2 - 4*a_4
- alpha^4 + 14*alpha^2*beta^2 + beta^4 = a_2^2 + 12*a_4

Therefore: Phi_4 = (-a_2)*(a_2^2+12*a_4) / (2*a_4*(a_2^2-4*a_4))

### A.2 Power Sums for Centered Quartic

For p(x) = x^4 + a_2 x^2 + a_3 x + a_4 (centered: e_1 = 0):

    p_1 = 0
    p_2 = -2*a_2
    p_3 = -3*a_3   (using e_3 = -a_3 from sign convention)
    p_4 = 2*a_2^2 - 4*a_4

### A.3 Discriminant Formulas

For depressed quartic x^4 + p*x^2 + q*x + r:

    disc = 256*r^3 - 128*p^2*r^2 + 144*p*q^2*r - 27*q^4 + 16*p^4*r - 4*p^3*q^2

For the symmetric case (q=0):

    disc = 16*r*(p^2 - 4*r)^2

### A.4 Surplus Numerator Details (Symmetric Case)

In shifted coordinates u = s - 1/12, v = t - 1/12:

    N/144 = H2 + H3 + H4

    H2 = 1008*u^2 - 288*u*v + 1008*v^2   (positive definite, eigenvalues 864, 1152)
    H3 = 864*(u+v)*(u^2 + 6*u*v + v^2)
    H4 = 5184*u*v*(u+v)^2

In (w,r) coordinates where w = u+v, r = uv:

    N/144 = 6*w^3 + 7*w^2 + r*(36*w^2 + 24*w - 16)

The coefficient of r vanishes at w = (-1+sqrt(5))/3 ~ 0.412, which is outside
the domain w < 1/3. Hence g(w) = 36w^2+24w-16 < 0 on the entire domain.

On the diagonal r = w^2/4:

    N/144 = 3*w^2*(w+1)*(3*w+1)

which is manifestly non-negative for w >= -1/6.
