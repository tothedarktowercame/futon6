#!/usr/bin/env python3
"""Verify Problem 4 inequality numerically and test convexity/concavity of 1/Phi_n.

Tests:
1. Does 1/Phi_n(p ⊞ q) >= 1/Phi_n(p) + 1/Phi_n(q) hold for random polynomials?
2. Is 1/Phi_n convex or concave in the finite free cumulants?
"""
import numpy as np
from itertools import product as iprod


def phi_n(roots):
    """Compute Phi_n = sum_i (sum_{j!=i} 1/(lambda_i - lambda_j))^2."""
    n = len(roots)
    total = 0.0
    for i in range(n):
        force = 0.0
        for j in range(n):
            if j != i:
                diff = roots[i] - roots[j]
                if abs(diff) < 1e-12:
                    return float('inf')
                force += 1.0 / diff
        total += force ** 2
    return total


def finite_free_conv(p_coeffs, q_coeffs):
    """Compute p ⊞_n q using the coefficient formula.

    p_coeffs = [a_0, a_1, ..., a_n] where p(x) = sum a_k x^{n-k}, a_0 = 1.
    """
    n = len(p_coeffs) - 1
    c = np.zeros(n + 1)
    from math import factorial
    for k in range(n + 1):
        for i in range(k + 1):
            j = k - i
            if i <= n and j <= n:
                coeff = (factorial(n - i) * factorial(n - j)) / (factorial(n) * factorial(n - k))
                c[k] += coeff * p_coeffs[i] * q_coeffs[j]
    return c


def roots_to_coeffs(roots):
    """Convert roots to monic polynomial coefficients [a_0=1, a_1, ..., a_n].

    p(x) = prod(x - r_i) = x^n - e_1 x^{n-1} + e_2 x^{n-2} - ...
    So a_k = (-1)^k * e_k(roots).
    """
    n = len(roots)
    # Use numpy to get coefficients, then convert to our convention
    poly = np.polynomial.polynomial.polyfromroots(roots)  # ascending order
    poly = poly[::-1]  # descending order: x^n, x^{n-1}, ...
    poly = poly / poly[0]  # make monic
    # Now poly = [1, -e_1, e_2, -e_3, ...]
    return poly


def coeffs_to_roots(coeffs):
    """Convert [a_0=1, a_1, ..., a_n] to roots.

    p(x) = x^n + a_1 x^{n-1} + a_2 x^{n-2} + ...
    """
    # numpy.roots expects [leading, ..., constant]
    return np.sort(np.roots(coeffs)).real


def test_inequality(n_roots, n_trials=1000):
    """Test the superadditivity inequality for random real-rooted polynomials."""
    n = n_roots
    violations = 0
    ratios = []

    for _ in range(n_trials):
        # Generate random real roots
        roots_p = np.sort(np.random.randn(n) * 2)
        roots_q = np.sort(np.random.randn(n) * 2)

        # Ensure distinct roots (add small perturbation if needed)
        for i in range(1, n):
            if roots_p[i] - roots_p[i-1] < 0.01:
                roots_p[i] = roots_p[i-1] + 0.01
            if roots_q[i] - roots_q[i-1] < 0.01:
                roots_q[i] = roots_q[i-1] + 0.01

        coeffs_p = roots_to_coeffs(roots_p)
        coeffs_q = roots_to_coeffs(roots_q)

        coeffs_conv = finite_free_conv(coeffs_p, coeffs_q)
        roots_conv = coeffs_to_roots(coeffs_conv)

        # Check if convolution roots are real
        if not np.all(np.isreal(np.roots(coeffs_conv))):
            continue

        phi_p = phi_n(roots_p)
        phi_q = phi_n(roots_q)
        phi_conv = phi_n(roots_conv)

        if phi_p == float('inf') or phi_q == float('inf') or phi_conv == float('inf'):
            continue

        lhs = 1.0 / phi_conv
        rhs = 1.0 / phi_p + 1.0 / phi_q

        ratio = lhs / rhs if rhs > 0 else float('inf')
        ratios.append(ratio)

        if lhs < rhs - 1e-10:  # violation (with tolerance)
            violations += 1

    ratios = np.array(ratios)
    return violations, len(ratios), ratios


def test_convexity_1d(n_roots, n_trials=500):
    """Test whether 1/Phi_n is convex or concave along lines in cumulant space.

    For convexity: f(t*a + (1-t)*b) <= t*f(a) + (1-t)*f(b)
    For concavity: f(t*a + (1-t)*b) >= t*f(a) + (1-t)*f(b)
    """
    n = n_roots
    convex_violations = 0  # points where convexity fails
    concave_violations = 0  # points where concavity fails
    total = 0

    for _ in range(n_trials):
        roots_a = np.sort(np.random.randn(n) * 2)
        roots_b = np.sort(np.random.randn(n) * 2)

        for i in range(1, n):
            if roots_a[i] - roots_a[i-1] < 0.01:
                roots_a[i] = roots_a[i-1] + 0.01
            if roots_b[i] - roots_b[i-1] < 0.01:
                roots_b[i] = roots_b[i-1] + 0.01

        coeffs_a = roots_to_coeffs(roots_a)
        coeffs_b = roots_to_coeffs(roots_b)

        phi_a = phi_n(roots_a)
        phi_b = phi_n(roots_b)
        if phi_a == float('inf') or phi_b == float('inf'):
            continue

        for t in [0.25, 0.5, 0.75]:
            # Interpolate in coefficient space (which is linear in cumulants)
            coeffs_mid = t * coeffs_a + (1 - t) * coeffs_b
            roots_mid = coeffs_to_roots(coeffs_mid)

            if not np.all(np.isreal(np.roots(coeffs_mid))):
                continue

            phi_mid = phi_n(roots_mid)
            if phi_mid == float('inf'):
                continue

            f_mid = 1.0 / phi_mid
            f_interp = t * (1.0 / phi_a) + (1 - t) * (1.0 / phi_b)

            total += 1
            if f_mid > f_interp + 1e-10:  # violates convexity
                convex_violations += 1
            if f_mid < f_interp - 1e-10:  # violates concavity
                concave_violations += 1

    return convex_violations, concave_violations, total


if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 60)
    print("Problem 4: 1/Phi_n(p ⊞ q) >= 1/Phi_n(p) + 1/Phi_n(q)?")
    print("=" * 60)

    for n in [2, 3, 4, 5]:
        violations, total, ratios = test_inequality(n, n_trials=2000)
        if total > 0:
            print(f"\nn={n}: {violations}/{total} violations")
            print(f"  ratio = LHS/RHS: min={ratios.min():.6f}, "
                  f"mean={ratios.mean():.6f}, max={ratios.max():.6f}")
            print(f"  (ratio >= 1 means inequality holds)")

    print("\n" + "=" * 60)
    print("Convexity/concavity test of 1/Phi_n in coefficient space")
    print("=" * 60)

    for n in [3, 4, 5]:
        cv, ccv, total = test_convexity_1d(n, n_trials=1000)
        if total > 0:
            print(f"\nn={n}: {total} midpoint tests")
            print(f"  convexity violations: {cv} ({100*cv/total:.1f}%)")
            print(f"  concavity violations: {ccv} ({100*ccv/total:.1f}%)")
            if cv == 0:
                print(f"  => 1/Phi_{n} appears CONVEX")
            elif ccv == 0:
                print(f"  => 1/Phi_{n} appears CONCAVE")
            else:
                print(f"  => 1/Phi_{n} is NEITHER convex nor concave")
