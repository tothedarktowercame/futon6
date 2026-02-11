#!/usr/bin/env python3
"""Deeper investigation of Problem 4.

Tests:
1. Does 1/Phi_n superadditivity hold under plain coefficient addition?
2. What happens in cumulant space for the MSS convolution?
3. Degree 3 explicit computation to find structure.
"""
import numpy as np
import importlib, sys, os
sys.path.insert(0, os.path.dirname(__file__))
_mod = importlib.import_module("verify-p4-inequality")
phi_n = _mod.phi_n
finite_free_conv = _mod.finite_free_conv
roots_to_coeffs = _mod.roots_to_coeffs
coeffs_to_roots = _mod.coeffs_to_roots


def test_coeffwise_addition(n_roots, n_trials=1000):
    """Test if 1/Phi_n is superadditive under plain coefficient addition."""
    n = n_roots
    violations = 0
    total = 0

    for _ in range(n_trials):
        roots_p = np.sort(np.random.randn(n) * 2)
        roots_q = np.sort(np.random.randn(n) * 2)

        for i in range(1, n):
            if roots_p[i] - roots_p[i-1] < 0.01:
                roots_p[i] = roots_p[i-1] + 0.01
            if roots_q[i] - roots_q[i-1] < 0.01:
                roots_q[i] = roots_q[i-1] + 0.01

        coeffs_p = roots_to_coeffs(roots_p)
        coeffs_q = roots_to_coeffs(roots_q)

        # Plain coefficient-wise addition
        coeffs_sum = coeffs_p + coeffs_q
        coeffs_sum[0] = 1.0  # keep monic

        roots_sum = coeffs_to_roots(coeffs_sum)
        if not np.all(np.isreal(np.roots(coeffs_sum))):
            continue

        phi_p = phi_n(roots_p)
        phi_q = phi_n(roots_q)
        phi_sum = phi_n(roots_sum)

        if phi_p == float('inf') or phi_q == float('inf') or phi_sum == float('inf'):
            continue

        lhs = 1.0 / phi_sum
        rhs = 1.0 / phi_p + 1.0 / phi_q
        total += 1

        if lhs < rhs - 1e-10:
            violations += 1

    return violations, total


def test_convexity_cumulant_space(n_roots, n_trials=500):
    """Test convexity of 1/Phi_n in the finite free cumulant space.

    For n=2: cumulant kappa_2 = (lambda_1 - lambda_2)^2 / 4
    For n=3: need the full moment-cumulant relation.

    We test indirectly: if p and q have cumulants kappa(p), kappa(q),
    then p ⊞ q has cumulants kappa(p) + kappa(q) (if cumulant additivity holds).
    The midpoint test: take t*p ⊞ (1-t)*q vs t*(identity ⊞ p) + (1-t)*(identity ⊞ q).

    Actually, we test directly by constructing polynomials at specific cumulant points.
    """
    # For n=2: kappa_1 = -(lambda_1+lambda_2)/2, kappa_2 = (lambda_1-lambda_2)^2/4
    # 1/Phi_2 = (lambda_1-lambda_2)^2/2 = 2*kappa_2, which is LINEAR in kappa_2.

    # For n=3: test numerically
    n = n_roots
    convex_violations = 0
    concave_violations = 0
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

        # The ⊞_n midpoint: (p ⊞_n q) at t=0.5
        # We construct: midpoint as p_mid where kappa(p_mid) = 0.5*kappa(a) + 0.5*kappa(b)
        # Since ⊞_n adds cumulants (if that's true), this is the same as
        # p_mid ⊞_n p_mid where p_mid has half the cumulants.

        # Instead, test convexity along the free convolution semigroup:
        # Construct scaled versions: p_t has coefficients scaled by factor t
        # If a_k -> t^k * a_k preserves real-rootedness...

        # Actually, simplest test: use ⊞_n itself to test superadditivity at midpoints
        # Take the degenerate polynomial delta = x^n (all roots at 0)
        # Then p ⊞_n delta = p (identity for ⊞_n)

        # Direct approach: interpolate in the ⊞_n sense
        # mid = (p/2) ⊞_n (q/2) where p/2 means "half the cumulants of p"
        # But we can't easily compute "half cumulants" directly.

        # Fall back: test convexity by interpolating in coefficient space
        # (which is NOT cumulant space, but let's see what happens)
        for t in [0.5]:
            # ⊞-midpoint: convolve scaled versions
            # This doesn't directly test cumulant-space convexity.
            # Instead, let's test the SPECIFIC property we need:
            # 1/Phi_n(p ⊞ q) >= 1/Phi_n(p) + 1/Phi_n(q)
            # which is superadditivity, not convexity.
            pass

    # The real question: is the proof fixable?
    # Let me just test what coefficient-space convexity looks like for ⊞_n
    print("  (Cumulant space test skipped — would need full moment-cumulant inversion)")
    return 0, 0, 0


def degree3_explicit():
    """Explicit computation for degree 3 to understand structure."""
    print("\n--- Degree 3 explicit analysis ---")

    # Symmetric case: roots at -s, 0, s
    # p(x) = x^3 - s^2 x, so a_0=1, a_1=0, a_2=-s^2, a_3=0
    # Phi_3 = sum of squared Coulomb forces at each root:
    # At x=-s: force = 1/(-s-0) + 1/(-s-s) = -1/s - 1/(2s) = -3/(2s)
    # At x=0:  force = 1/(0-(-s)) + 1/(0-s) = 1/s - 1/s = 0
    # At x=s:  force = 1/(s-(-s)) + 1/(s-0) = 1/(2s) + 1/s = 3/(2s)
    # Phi_3 = (-3/(2s))^2 + 0^2 + (3/(2s))^2 = 2 * 9/(4s^2) = 9/(2s^2)
    # 1/Phi_3 = 2s^2/9

    for s in [1.0, 2.0]:
        roots = np.array([-s, 0, s])
        phi = phi_n(roots)
        print(f"  roots=[-{s}, 0, {s}]: Phi_3={phi:.6f}, 1/Phi_3={1/phi:.6f}, "
              f"expected 2s^2/9={2*s**2/9:.6f}")

    # Now test: p = x^3 - s^2 x (roots -s,0,s), q = x^3 - t^2 x (roots -t,0,t)
    # p ⊞_3 q = ?
    print()
    for s, t in [(1.0, 1.0), (1.0, 2.0), (2.0, 3.0)]:
        coeffs_p = np.array([1, 0, -s**2, 0])  # x^3 + 0*x^2 - s^2*x + 0
        coeffs_q = np.array([1, 0, -t**2, 0])  # x^3 + 0*x^2 - t^2*x + 0
        coeffs_conv = finite_free_conv(coeffs_p, coeffs_q)
        roots_conv = coeffs_to_roots(coeffs_conv)

        phi_p = phi_n(np.array([-s, 0, s]))
        phi_q = phi_n(np.array([-t, 0, t]))
        phi_conv = phi_n(roots_conv)

        lhs = 1/phi_conv
        rhs = 1/phi_p + 1/phi_q

        print(f"  s={s}, t={t}:")
        print(f"    p ⊞ q coeffs = {coeffs_conv}")
        print(f"    p ⊞ q roots = {roots_conv}")
        print(f"    1/Phi(p⊞q) = {lhs:.6f}")
        print(f"    1/Phi(p) + 1/Phi(q) = {rhs:.6f}")
        print(f"    ratio = {lhs/rhs:.6f}")


def test_mss_coefficient_structure():
    """Examine the MSS ⊞_n coefficient formula to understand what's really additive."""
    print("\n--- MSS coefficient formula analysis ---")
    print("For n=3, the ⊞_3 formula gives:")
    print("  c_0 = 1")
    print("  c_1 = a_1 + b_1")
    print("  c_2 = a_2 + (2/3)*a_1*b_1 + b_2")
    print("  c_3 = a_3 + (1/3)*a_2*b_1 + (1/3)*a_1*b_2 + b_3")
    print()

    from math import factorial
    n = 3
    print(f"Coefficient matrix for n={n}:")
    for k in range(n+1):
        terms = []
        for i in range(k+1):
            j = k - i
            if i <= n and j <= n:
                coeff = (factorial(n-i) * factorial(n-j)) / (factorial(n) * factorial(n-k))
                if abs(coeff) > 1e-15:
                    terms.append(f"  ({coeff:.4f}) * a_{i} * b_{j}")
        print(f"  c_{k} = ")
        for term in terms:
            print(f"    {term}")

    print()
    print("So c_1 = a_1 + b_1 (additive)")
    print("But c_2 = a_2 + b_2 + (2/3)*a_1*b_1 (NOT additive — cross-term)")
    print("This means a_k are NOT finite free cumulants.")
    print("The finite free cumulants kappa_k satisfy kappa_k(p⊞q) = kappa_k(p)+kappa_k(q)")
    print("and are related to a_k by a nonlinear transformation.")
    print()
    print("For the proof: the claim 'cumulants add under ⊞_n' needs the")
    print("CORRECT finite free cumulants, not just the polynomial coefficients.")
    print("The R-transform/cumulant theory of Arizmendi-Perales (2018) defines")
    print("these for finite free convolution.")


if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 60)
    print("Test 1: Superadditivity under PLAIN coefficient addition")
    print("=" * 60)
    for n in [3, 4, 5]:
        v, total = test_coeffwise_addition(n, n_trials=1000)
        print(f"  n={n}: {v}/{total} violations")

    print()
    print("=" * 60)
    print("Test 2: Degree 3 explicit structure")
    print("=" * 60)
    degree3_explicit()

    print()
    print("=" * 60)
    print("Test 3: MSS coefficient formula structure")
    print("=" * 60)
    test_mss_coefficient_structure()
