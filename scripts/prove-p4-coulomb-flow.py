#!/usr/bin/env python3
"""Rigorous symbolic proof of the Finite Coulomb Flow Theorem.

THEOREM: Let p_t = p ⊞_n He_t (MSS convolution with Hermite heat kernel).
Then the roots γ_k(t) satisfy dγ_k/dt = S_k(γ) = Σ_{j≠k} 1/(γ_k - γ_j).

PROOF STRATEGY:
  Step 1: Show He_t satisfies d/dt He_t = -(1/2) He_t'' (backward heat equation)
  Step 2: Show this lifts to p_t via an MSS weight identity:
          d/dt p_t = -(1/2) p_t''  (backward heat equation for the convolution)
  Step 3: At root γ_k, implicit differentiation gives:
          γ_k' = -(dp_t/dt)(γ_k)/p_t'(γ_k) = (1/2)p_t''(γ_k)/p_t'(γ_k)
  Step 4: Standard identity: p_t''(γ_k) = 2 p_t'(γ_k) · S_k(γ), so γ_k' = S_k(γ).

The key algebraic identity (Step 2) is:
  w(n,i,j)·(n-j+2)(n-j+1) = (n-i-j+2)(n-i-j+1)·w(n,i,j-2)
where w(n,i,j) = (n-i)!(n-j)! / (n!(n-i-j)!).

This script verifies all steps symbolically and numerically.
"""

import sympy as sp
from sympy import factorial, symbols, simplify, Rational, Poly, diff
from sympy import Matrix, sqrt, Function
import numpy as np
from math import factorial as fact
import sys

sys.stdout.reconfigure(line_buffering=True)


# ══════════════════════════════════════════════════════════════════════
# STEP 1: He_t satisfies the backward heat equation d/dt He_t = -(1/2)He_t''
# ══════════════════════════════════════════════════════════════════════

def verify_hermite_backward_heat():
    """Verify d/dt He_t(x) = -(1/2) He_t''(x) symbolically."""
    print("=" * 70)
    print("STEP 1: Hermite backward heat equation")
    print("=" * 70)

    x, t = sp.symbols('x t', positive=True)

    for n in range(2, 8):
        # Build He_t(x) = t^{n/2} He_n(x/√t)
        # Probabilist's Hermite: He_n(y) = Σ (-1)^m n!/(m! 2^m (n-2m)!) y^{n-2m}
        He_t = sp.Integer(0)
        for m in range(n // 2 + 1):
            coeff = sp.Rational((-1)**m * fact(n), fact(m) * 2**m * fact(n - 2*m))
            He_t += coeff * t**m * x**(n - 2*m)

        # Compute d/dt and -(1/2) d²/dx²
        dHe_dt = sp.diff(He_t, t)
        neg_half_Hepp = sp.Rational(-1, 2) * sp.diff(He_t, x, 2)

        diff_check = sp.expand(dHe_dt - neg_half_Hepp)

        status = "✓" if diff_check == 0 else "✗"
        print(f"  n={n}: d/dt He_t = -(1/2) He_t''  {status}")
        if diff_check != 0:
            print(f"    RESIDUAL: {diff_check}")

    print()
    return True


# ══════════════════════════════════════════════════════════════════════
# STEP 2: MSS weight identity
# ══════════════════════════════════════════════════════════════════════

def verify_mss_weight_identity():
    """Verify: w(n,i,j)(n-j+2)(n-j+1) = (n-i-j+2)(n-i-j+1) w(n,i,j-2).

    This is the key algebraic identity that lifts the Hermite backward
    heat equation to the MSS convolution.
    """
    print("=" * 70)
    print("STEP 2: MSS weight identity")
    print("=" * 70)

    def w(n, i, j):
        """MSS weight: (n-i)!(n-j)! / (n!(n-i-j)!)"""
        if i < 0 or j < 0 or i + j > n:
            return sp.Integer(0)
        return sp.Rational(fact(n-i) * fact(n-j), fact(n) * fact(n-i-j))

    all_pass = True
    for n in range(2, 10):
        for i in range(n + 1):
            for j in range(2, n + 1):  # j >= 2 for j-2 >= 0
                if i + j > n:
                    continue
                lhs = w(n, i, j) * (n - j + 2) * (n - j + 1)
                rhs = (n - i - j + 2) * (n - i - j + 1) * w(n, i, j - 2)
                if lhs != rhs:
                    print(f"  FAIL: n={n}, i={i}, j={j}: {lhs} ≠ {rhs}")
                    all_pass = False

    if all_pass:
        print("  All (n,i,j) with n=2..9: w(n,i,j)(n-j+2)(n-j+1) = (n-k+2)(n-k+1)w(n,i,j-2)  ✓")
    print()

    # Symbolic proof
    print("  SYMBOLIC PROOF:")
    print("    w(n,i,j) = (n-i)!(n-j)! / (n!(n-i-j)!)")
    print()
    print("    LHS = w(n,i,j) · (n-j+2)(n-j+1)")
    print("        = (n-i)!(n-j)! · (n-j+2)(n-j+1) / (n!(n-i-j)!)")
    print("        = (n-i)!(n-j+2)! / (n!(n-i-j)!)")
    print()
    print("    RHS = (n-i-j+2)(n-i-j+1) · w(n,i,j-2)")
    print("        = (n-i-j+2)(n-i-j+1) · (n-i)!(n-j+2)! / (n!(n-i-j+2)!)")
    print("        = (n-i)!(n-j+2)! · (n-i-j+2)(n-i-j+1) / (n!(n-i-j+2)!)")
    print("        = (n-i)!(n-j+2)! / (n!(n-i-j)!)")
    print("    since (n-i-j+2)! / ((n-i-j+2)(n-i-j+1)) = (n-i-j)!.")
    print()
    print("    LHS = RHS.  ∎")
    print()

    return all_pass


# ══════════════════════════════════════════════════════════════════════
# STEP 2b: Lift to the full convolution p_t = p ⊞_n He_t
# ══════════════════════════════════════════════════════════════════════

def verify_convolution_backward_heat():
    """Verify d/dt(p ⊞_n He_t) = -(1/2)(p ⊞_n He_t)'' symbolically for small n."""
    print("=" * 70)
    print("STEP 2b: Backward heat equation for p_t = p ⊞_n He_t")
    print("=" * 70)

    x, t = sp.symbols('x t')

    for n in range(2, 7):
        # Symbolic coefficients of p
        a = [sp.Integer(1)] + [sp.Symbol(f'a{k}') for k in range(1, n + 1)]

        # Hermite coefficients h_j(t)
        def h(j):
            if j == 0:
                return sp.Integer(1)
            if j % 2 == 1:
                return sp.Integer(0)
            m = j // 2
            return sp.Rational((-1)**m * fact(n), fact(m) * 2**m * fact(n - 2*m)) * t**m

        # MSS weight
        def w(i, j):
            if i < 0 or j < 0 or i + j > n:
                return sp.Integer(0)
            return sp.Rational(fact(n-i) * fact(n-j), fact(n) * fact(n-i-j))

        # Build p_t(x) = x^n + Σ c_k x^{n-k}
        p_t = x**n
        for k in range(1, n + 1):
            c_k = sp.Integer(0)
            for i in range(k + 1):
                j = k - i
                if j > n:
                    continue
                c_k += w(i, j) * a[i] * h(j)
            p_t += c_k * x**(n - k)

        p_t = sp.expand(p_t)

        # Compute d/dt p_t and -(1/2) d²/dx² p_t
        dp_dt = sp.expand(sp.diff(p_t, t))
        neg_half_ppp = sp.expand(sp.Rational(-1, 2) * sp.diff(p_t, x, 2))

        diff_check = sp.expand(dp_dt - neg_half_ppp)
        status = "✓" if diff_check == 0 else "✗"
        print(f"  n={n}: d/dt p_t = -(1/2) p_t''  {status}")

        if diff_check != 0:
            print(f"    RESIDUAL: {diff_check}")

    print()
    return True


# ══════════════════════════════════════════════════════════════════════
# STEP 3 & 4: Root derivative identity
# ══════════════════════════════════════════════════════════════════════

def verify_root_derivative():
    """Verify that p''(γ_k) = 2 p'(γ_k) S_k(γ) for small n with random roots."""
    print("=" * 70)
    print("STEP 3-4: p''(γ_k) = 2 p'(γ_k) S_k(γ)")
    print("=" * 70)

    np.random.seed(42)

    for n in [3, 4, 5, 6, 7]:
        errors = []
        for _ in range(100):
            roots = np.sort(np.random.randn(n) * 2)
            # Ensure roots are distinct
            while np.min(np.diff(roots)) < 0.1:
                roots = np.sort(np.random.randn(n) * 2)

            poly = np.poly(roots)  # monic, descending

            for k in range(n):
                gk = roots[k]

                # p'(γ_k) = product of (γ_k - γ_j) for j ≠ k
                pprime_val = np.polyval(np.polyder(poly), gk)

                # p''(γ_k)
                ppp_val = np.polyval(np.polyder(poly, 2), gk)

                # S_k = Σ_{j≠k} 1/(γ_k - γ_j)
                S_k = sum(1.0 / (gk - roots[j]) for j in range(n) if j != k)

                # Check: p''(γ_k) = 2 p'(γ_k) S_k
                lhs = ppp_val
                rhs = 2 * pprime_val * S_k
                rel_err = abs(lhs - rhs) / (abs(rhs) + 1e-15)
                errors.append(rel_err)

        max_err = max(errors)
        status = "✓" if max_err < 1e-8 else "✗"
        print(f"  n={n}: max relative error = {max_err:.2e}  {status}")

    print()
    print("  PROOF: This is a standard identity.")
    print("  p(x) = Π_j (x - γ_j)")
    print("  p'(x) = Σ_i Π_{j≠i} (x - γ_j)")
    print("  p''(x) = Σ_{i≠j} Π_{l≠i,j} (x - γ_l)")
    print("  At x = γ_k:")
    print("    p'(γ_k) = Π_{j≠k} (γ_k - γ_j)")
    print("    p''(γ_k) = 2 Σ_{j≠k} Π_{l≠k,j} (γ_k - γ_l)")
    print("             = 2 Π_{l≠k}(γ_k - γ_l) · Σ_{j≠k} 1/(γ_k - γ_j)")
    print("             = 2 p'(γ_k) · S_k(γ).  ∎")
    print()


# ══════════════════════════════════════════════════════════════════════
# FULL THEOREM: Numerical verification of backward heat equation → Coulomb flow
# ══════════════════════════════════════════════════════════════════════

def verify_full_theorem_numerically():
    """End-to-end numerical verification of the full proof chain."""
    print("=" * 70)
    print("FULL THEOREM: Numerical end-to-end verification")
    print("=" * 70)

    np.random.seed(2026)

    def mss_convolve(a_coeffs, b_coeffs, n):
        c = np.zeros(n)
        for k in range(1, n + 1):
            s = 0.0
            for i in range(k + 1):
                j = k - i
                if i > n or j > n:
                    continue
                ai = 1.0 if i == 0 else a_coeffs[i - 1]
                bj = 1.0 if j == 0 else b_coeffs[j - 1]
                w = (fact(n - i) * fact(n - j)) / (fact(n) * fact(n - k))
                s += w * ai * bj
            c[k - 1] = s
        return c

    def hermite_roots(n, t):
        from numpy.polynomial.hermite_e import hermeroots
        base = sorted(hermeroots([0]*n + [1]))
        return np.array(base) * np.sqrt(t)

    for n in [3, 4, 5, 6]:
        print(f"\n  n={n}:")
        dt = 0.0001
        max_errs = []

        for trial in range(20):
            p_roots = np.sort(np.random.randn(n) * 2)
            a_coeffs = np.poly(p_roots)[1:]
            t = 1.0

            # Convolutions at t and t±dt
            h_t = hermite_roots(n, t)
            h_p = hermite_roots(n, t + dt)
            h_m = hermite_roots(n, t - dt)

            c_t = mss_convolve(a_coeffs, np.poly(h_t)[1:], n)
            c_p = mss_convolve(a_coeffs, np.poly(h_p)[1:], n)
            c_m = mss_convolve(a_coeffs, np.poly(h_m)[1:], n)

            poly_t = np.concatenate([[1.0], c_t])
            poly_p = np.concatenate([[1.0], c_p])
            poly_m = np.concatenate([[1.0], c_m])

            # Check real-rootedness
            r_t = np.roots(poly_t)
            if np.max(np.abs(r_t.imag)) > 1e-6:
                continue
            roots_t = np.sort(r_t.real)

            # Method 1: Numerical dp_t/dt
            dc_dt_numerical = (c_p - c_m) / (2 * dt)

            # Method 2: -(1/2) p_t'' coefficients
            # p_t(x) = x^n + c_1 x^{n-1} + ... + c_n
            # p_t''(x) = n(n-1)x^{n-2} + c_1(n-1)(n-2)x^{n-3} + ...
            # Coefficient of x^{n-k} in p_t'': (n-k+2)(n-k+1) c_{k-2}
            neg_half_ppp_coeffs = np.zeros(n)
            for k in range(1, n + 1):
                if k >= 3:
                    ck_minus_2 = c_t[k - 3]  # c_{k-2} (0-indexed: c_t[k-3])
                elif k == 2:
                    ck_minus_2 = 1.0  # c_0 = 1 (leading coefficient)
                else:
                    ck_minus_2 = 0.0  # c_{-1} = 0
                neg_half_ppp_coeffs[k - 1] = -0.5 * (n - k + 2) * (n - k + 1) * ck_minus_2

            # Compare
            err = np.max(np.abs(dc_dt_numerical - neg_half_ppp_coeffs))
            max_errs.append(err)

            # Also verify Coulomb flow at roots
            r_p = np.sort(np.roots(poly_p).real)
            r_m = np.sort(np.roots(poly_m).real)
            velocity = (r_p - r_m) / (2 * dt)
            S = np.array([sum(1.0 / (roots_t[k] - roots_t[j])
                             for j in range(n) if j != k) for k in range(n)])
            coulomb_err = np.max(np.abs(velocity - S) / (np.abs(S) + 1e-15))
            max_errs.append(coulomb_err)

        if max_errs:
            print(f"    backward heat eq max coeff error: {max(max_errs[0::2]):.2e}")
            print(f"    Coulomb flow max relative error:  {max(max_errs[1::2]):.2e}")


# ══════════════════════════════════════════════════════════════════════
# WRITE PROOF
# ══════════════════════════════════════════════════════════════════════

def print_proof():
    print()
    print("=" * 70)
    print("COMPLETE PROOF OF THE FINITE COULOMB FLOW THEOREM")
    print("=" * 70)
    print("""
THEOREM (Finite Coulomb Flow). Let p be a monic real-rooted degree-n
polynomial. Let He_t(x) = t^{n/2} He_n(x/√t) be the scaled probabilist's
Hermite polynomial, and p_t = p ⊞_n He_t. Let γ_1(t) < ... < γ_n(t) be
the roots of p_t. Then:

    dγ_k/dt = S_k(γ) := Σ_{j≠k} 1/(γ_k - γ_j)

PROOF. The proof has three steps.

━━━ Step 1: Backward heat equation for He_t ━━━

LEMMA 1. d/dt He_t(x) = -(1/2) He_t''(x).

Proof. He_t(x) = Σ_{m=0}^{⌊n/2⌋} C_m t^m x^{n-2m} where
C_m = (-1)^m n!/(m! 2^m (n-2m)!). Then:

  dHe_t/dt = Σ_{m≥1} m C_m t^{m-1} x^{n-2m}

  He_t''   = Σ_{m≥0} C_m t^m (n-2m)(n-2m-1) x^{n-2m-2}
           = Σ_{m≥1} C_{m-1} t^{m-1} (n-2m+2)(n-2m+1) x^{n-2m}

So we need: m C_m = -(1/2)(n-2m+2)(n-2m+1) C_{m-1}.

Compute: C_m/C_{m-1} = (-1)·(n-2m+2)!/(n-2m)! · (m-1)! 2^{m-1}/(m! 2^m)
                      = (-1)·(n-2m+2)(n-2m+1)/(2m).

So m C_m = -(1/2)(n-2m+2)(n-2m+1) C_{m-1}.  ∎

━━━ Step 2: Backward heat equation for p_t ━━━

LEMMA 2 (Key Lemma). d/dt p_t(x) = -(1/2) p_t''(x).

That is, p_t = p ⊞_n He_t satisfies the backward heat equation.

Proof. Write p_t(x) = x^n + Σ_{k=1}^n c_k(t) x^{n-k} where

  c_k(t) = Σ_{i+j=k} w(n,i,j) a_i h_j(t),
  w(n,i,j) = (n-i)!(n-j)! / (n!(n-i-j)!),

a_i = coefficients of p, h_j(t) = coefficients of He_t.

The coefficient of x^{n-k} in dp_t/dt is:

  dc_k/dt = Σ_{i+j=k} w(n,i,j) a_i (dh_j/dt)

By Lemma 1, dh_j/dt is the coefficient of x^{n-j} in -(1/2)He_t'', which
equals -(1/2)(n-j+2)(n-j+1) h_{j-2}(t). So:

  dc_k/dt = -(1/2) Σ_{i+j=k} w(n,i,j)(n-j+2)(n-j+1) a_i h_{j-2}(t)   ...(I)

The coefficient of x^{n-k} in -(1/2)p_t'' is:

  -(1/2)(n-k+2)(n-k+1) c_{k-2}(t)
  = -(1/2)(n-k+2)(n-k+1) Σ_{i+j'=k-2} w(n,i,j') a_i h_{j'}(t)

Setting j = j'+2 (so i+j = k):

  = -(1/2)(n-k+2)(n-k+1) Σ_{i+j=k} w(n,i,j-2) a_i h_{j-2}(t)   ...(II)

For (I) = (II), we need the MSS WEIGHT IDENTITY:

  w(n,i,j) · (n-j+2)(n-j+1) = (n-i-j+2)(n-i-j+1) · w(n,i,j-2)

with k = i+j. Expanding both sides:

  LHS = (n-i)!(n-j)!·(n-j+2)(n-j+1) / (n!(n-i-j)!)
      = (n-i)!(n-j+2)! / (n!(n-i-j)!)

  RHS = (n-i-j+2)(n-i-j+1) · (n-i)!(n-j+2)! / (n!(n-i-j+2)!)
      = (n-i)!(n-j+2)! / (n!(n-i-j)!)

since (n-i-j+2)! / ((n-i-j+2)(n-i-j+1)) = (n-i-j)!.

LHS = RHS.  ∎

━━━ Step 3: From backward heat equation to Coulomb flow ━━━

At a root γ_k(t) of p_t, implicit differentiation of p_t(γ_k(t), t) = 0:

  (∂p_t/∂t)(γ_k) + p_t'(γ_k) · γ_k' = 0
  γ_k' = -(∂p_t/∂t)(γ_k) / p_t'(γ_k)
       = (1/2) p_t''(γ_k) / p_t'(γ_k)     [by Lemma 2]

Now use the STANDARD ROOT IDENTITY: for p(x) = Π_j(x-γ_j),

  p''(γ_k) = 2 p'(γ_k) · Σ_{j≠k} 1/(γ_k - γ_j)

(Proof: p'(x) = Σ_i Π_{j≠i}(x-γ_j). Differentiating again and evaluating
at γ_k, only terms with i=k or one factor (x-γ_k) in the product survive,
giving 2 Σ_{j≠k} Π_{l≠k,j}(γ_k-γ_l) = 2 p'(γ_k) S_k.)

Therefore:
  γ_k' = (1/2) · 2 p_t'(γ_k) S_k(γ) / p_t'(γ_k) = S_k(γ).  ∎

COROLLARY (Finite De Bruijn Identity). d/dt H'_n(p_t) = Φ_n(p_t),
where H'_n = Σ_{i<j} log|γ_i - γ_j| and Φ_n = Σ_k S_k².

(Proof in problem4-debruijn-discovery.md: chain rule + rearrangement.)
""")


def main():
    # Symbolic verifications
    verify_hermite_backward_heat()
    ok = verify_mss_weight_identity()
    verify_convolution_backward_heat()
    verify_root_derivative()

    # Numerical end-to-end
    verify_full_theorem_numerically()

    # Print the proof
    print_proof()


if __name__ == '__main__':
    main()
