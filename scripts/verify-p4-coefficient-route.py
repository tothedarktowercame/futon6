#!/usr/bin/env python3
"""Problem 4: Direct coefficient route for the superadditivity proof.

For n=2: 1/Phi_2 = disc(p)/4 = (a_1^2 - 4a_2)/4, which is linear in
coefficients and preserved exactly by ⊞_2.

For n=3: compute Phi_3 symbolically, express in terms of coefficients,
apply ⊞_3 formula, and check the surplus.
"""
from sympy import (symbols, Rational, expand, simplify, factor,
                   collect, Poly, resultant, discriminant as sym_disc,
                   sqrt, together, cancel, Symbol, solve, Sum,
                   Function, series, oo, prod as sprod)
from sympy import Matrix
import sympy


def verify_n2():
    """Verify the n=2 case symbolically."""
    print("=" * 65)
    print("n=2: Symbolic verification")
    print("=" * 65)

    a1, a2, b1, b2 = symbols('a1 a2 b1 b2')

    # p(x) = x^2 + a1*x + a2, roots lambda_1, lambda_2
    # disc(p) = a1^2 - 4*a2
    # Phi_2 = 2/(lambda_1 - lambda_2)^2 = 2/disc(p)
    # 1/Phi_2 = disc(p)/2 = (a1^2 - 4*a2)/2

    inv_phi_p = (a1**2 - 4*a2) / 2
    inv_phi_q = (b1**2 - 4*b2) / 2

    # ⊞_2 formula: c_k = sum_{i+j=k} w(2,i,j) a_i b_j
    # w(n,i,j) = (n-i)!(n-j)! / (n!(n-k)!)
    # n=2: w(2,0,0) = 2!*2!/(2!*2!) = 1
    #       w(2,0,1) = 2!*1!/(2!*1!) = 1, w(2,1,0) = 1
    #       w(2,0,2) = 2!*0!/(2!*0!) = 1, w(2,1,1) = 1!*1!/(2!*0!) = 1/2, w(2,2,0) = 1
    c1 = a1 + b1  # a0=b0=1
    c2 = a2 + Rational(1, 2) * a1 * b1 + b2

    inv_phi_conv = (c1**2 - 4*c2) / 2
    inv_phi_conv_expanded = expand(inv_phi_conv)

    surplus = expand(inv_phi_conv_expanded - inv_phi_p - inv_phi_q)

    print(f"  1/Phi_2(p) = {inv_phi_p}")
    print(f"  1/Phi_2(q) = {inv_phi_q}")
    print(f"  c1 = {c1}")
    print(f"  c2 = {c2}")
    print(f"  1/Phi_2(p⊞q) = {inv_phi_conv_expanded}")
    print(f"  surplus = 1/Phi(conv) - 1/Phi(p) - 1/Phi(q) = {surplus}")
    print(f"  => EQUALITY (surplus = 0)")


def compute_phi3_symbolic():
    """Compute Phi_3 symbolically in terms of roots and coefficients."""
    print("\n" + "=" * 65)
    print("n=3: Symbolic computation of Phi_3")
    print("=" * 65)

    l1, l2, l3 = symbols('lambda1 lambda2 lambda3')

    # Forces at each root
    f1 = 1/(l1 - l2) + 1/(l1 - l3)
    f2 = 1/(l2 - l1) + 1/(l2 - l3)
    f3 = 1/(l3 - l1) + 1/(l3 - l2)

    # Phi_3 = sum of squared forces
    phi3 = f1**2 + f2**2 + f3**2

    # Simplify
    phi3_simplified = simplify(phi3)
    print(f"\n  Phi_3 = {phi3_simplified}")

    # Express using p'(lambda_i) identity
    # f_i = p''(lambda_i) / (2 p'(lambda_i))
    # p(x) = (x-l1)(x-l2)(x-l3) = x^3 - e1*x^2 + e2*x - e3
    e1 = l1 + l2 + l3
    e2 = l1*l2 + l1*l3 + l2*l3
    e3 = l1*l2*l3

    # Get Phi_3 in terms of e1, e2, e3 (or equivalently a1, a2, a3)
    # a1 = -e1, a2 = e2, a3 = -e3

    # Try to express Phi_3 as a rational function of (e1, e2, e3)
    # Numerator and denominator separately
    phi3_together = together(phi3)
    print(f"\n  Phi_3 (combined fraction) = {phi3_together}")

    # The denominator should be related to the discriminant
    # disc = (l1-l2)^2 (l1-l3)^2 (l2-l3)^2
    disc = (l1-l2)**2 * (l1-l3)**2 * (l2-l3)**2
    disc_expanded = expand(disc)

    # Let's compute numerator * disc / denominator to see what cancels
    # Phi_3 = N/D where D = (l1-l2)^2*(l1-l3)^2 * (l2-l1)^2*(l2-l3)^2 * ... no wait
    # Let me compute it differently

    # f_i = ((l_i - l_j) + (l_i - l_k)) / ((l_i-l_j)(l_i-l_k))  where {j,k} = {1,2,3}\{i}
    # f_i = (2*l_i - l_j - l_k) / ((l_i - l_j)(l_i - l_k))
    # Since l_i + l_j + l_k = e1, we have l_j + l_k = e1 - l_i
    # So f_i = (2*l_i - (e1 - l_i)) / ((l_i-l_j)(l_i-l_k)) = (3*l_i - e1) / p'(l_i)/1

    # Actually p'(l_i) = (l_i - l_j)(l_i - l_k) for degree 3
    # f_i = (3*l_i - e1) / p'(l_i) ... wait let me recheck

    # p(x) = x^3 - e1*x^2 + e2*x - e3
    # p'(x) = 3x^2 - 2*e1*x + e2
    # p'(l_i) = 3*l_i^2 - 2*e1*l_i + e2 = (l_i - l_j)(l_i - l_k) [standard identity for monic p]

    # Wait, p'(l_i) should equal prod_{j≠i}(l_i - l_j) since p is monic degree 3
    # p(x) = (x-l1)(x-l2)(x-l3)
    # p'(x) = (x-l2)(x-l3) + (x-l1)(x-l3) + (x-l1)(x-l2)
    # p'(l1) = (l1-l2)(l1-l3)  ✓

    # And p''(x) = 2(x-l3) + 2(x-l1) + 2(x-l2) - ... no
    # p''(x) = 2*(3x - (l1+l2+l3)) = 2*(3x - e1) = 6x - 2e1
    # So p''(l_i) = 6*l_i - 2*e1

    # f_i = p''(l_i)/(2*p'(l_i)) = (6*l_i - 2*e1) / (2*(l_i-l_j)(l_i-l_k))
    #      = (3*l_i - e1) / ((l_i-l_j)(l_i-l_k))

    # Phi_3 = sum_i f_i^2 = sum_i (3*l_i - e1)^2 / ((l_i-l_j)(l_i-l_k))^2
    #       = sum_i (3*l_i - e1)^2 / p'(l_i)^2

    # Numerator of Phi_3 (with common denominator disc):
    # disc = p'(l1)^2 * p'(l2)^2 * p'(l3)^2 ... no
    # disc = prod_{i<j}(l_i - l_j)^2 = (l1-l2)^2 (l1-l3)^2 (l2-l3)^2

    # p'(l1)^2 = (l1-l2)^2 (l1-l3)^2
    # p'(l2)^2 = (l2-l1)^2 (l2-l3)^2 = (l1-l2)^2 (l2-l3)^2
    # p'(l3)^2 = (l3-l1)^2 (l3-l2)^2 = (l1-l3)^2 (l2-l3)^2

    # So disc = p'(l1)^2 * (l2-l3)^2 = p'(l2)^2 * (l1-l3)^2 = p'(l3)^2 * (l1-l2)^2

    # Phi_3 = (3l1-e1)^2/p'(l1)^2 + (3l2-e1)^2/p'(l2)^2 + (3l3-e1)^2/p'(l3)^2
    # Common denom = p'(l1)^2 * p'(l2)^2 * p'(l3)^2 = disc^3 / ((l1-l2)(l1-l3)(l2-l3))^{something}
    # Hmm, this is getting complicated. Let me use a different approach.

    # Actually: disc = p'(l1)^2 * p'(l2)^2 * p'(l3)^2 / something?
    # No. disc(p) = prod_{i<j}(l_i-l_j)^2.
    # prod_i p'(l_i) = prod_i prod_{j≠i}(l_i-l_j) = product over all ordered pairs
    # = (-1)^{n(n-1)/2} * disc(p) ... for n=3, (-1)^3 * disc = -disc
    # Actually: prod_i p'(l_i) = prod_{i<j}(l_i-l_j) * prod_{i>j}(l_i-l_j)
    # = prod_{i<j}(l_i-l_j) * prod_{i<j}(l_j-l_i) * (-1)^{something}
    # = prod_{i<j}(l_i-l_j) * (-1)^{3} * prod_{i<j}(l_i-l_j) = -disc

    # Anyway, let me just compute numerically what Phi_3 looks like as a
    # function of the coefficients, and then check the surplus.

    # For the coefficient approach, let me parametrize.
    # WLOG set e1 = 0 (translate roots so mean = 0; doesn't affect Phi_n).
    # Then l1+l2+l3 = 0, so a1 = 0.
    # Remaining: a2 = e2 = l1*l2+l1*l3+l2*l3, a3 = -e3 = -l1*l2*l3.

    # With e1=0: f_i = 3*l_i / p'(l_i) = 3*l_i / (3*l_i^2 + e2)
    # Because p'(l_i) = 3*l_i^2 - 0 + e2 = 3*l_i^2 + e2

    # Phi_3 = sum_i (3*l_i)^2 / (3*l_i^2 + e2)^2 = 9 * sum_i l_i^2 / (3*l_i^2 + e2)^2

    # And disc(p) at e1=0: 4*e2^3 + 27*e3^2 ... actually
    # disc(x^3 + e2*x - e3) = -4*e2^3 - 27*e3^2
    # (standard formula for depressed cubic x^3 + px + q: disc = -4p^3 - 27q^2)

    print("\n  Setting e1 = 0 (WLOG by translation invariance of Phi_n):")
    print("  f_i = 3*l_i / (3*l_i^2 + e2)")
    print("  Phi_3 = 9 * sum_i l_i^2 / (3*l_i^2 + e2)^2")
    print("  where e2 = l1*l2 + l1*l3 + l2*l3 and l1+l2+l3 = 0")

    return phi3_simplified


def n3_coefficient_surplus():
    """Check the surplus for n=3 using the ⊞_3 formula numerically on a grid."""
    print("\n" + "=" * 65)
    print("n=3: Coefficient surplus under ⊞_3")
    print("=" * 65)

    import numpy as np
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    _mod = __import__("verify-p4-inequality")
    phi_n_func = _mod.phi_n
    finite_free_conv_func = _mod.finite_free_conv
    coeffs_to_roots_func = _mod.coeffs_to_roots

    def inv_phi(coeffs):
        roots = coeffs_to_roots_func(coeffs)
        if not np.all(np.isreal(np.roots(coeffs))):
            return float('nan')
        roots = np.sort(roots.real)
        for i in range(1, len(roots)):
            if roots[i] - roots[i-1] < 1e-6:
                return float('nan')
        phi = phi_n_func(roots)
        if phi == float('inf') or phi == 0:
            return float('nan')
        return 1.0 / phi

    # For n=3, WLOG set a1=b1=0 (center roots at 0)
    # p(x) = x^3 + a2*x + a3 with a2 < 0 for real roots
    # q(x) = x^3 + b2*x + b3 with b2 < 0 for real roots
    # Discriminant: -4*a2^3 - 27*a3^2 > 0 for 3 distinct real roots
    # => a2 < 0 and |a3| < 2*(-a2/3)^{3/2}

    print("\n  Testing centered cubics p(x) = x^3 + a2*x + a3:")
    print("  ⊞_3 formula: c1=0, c2=a2+b2, c3=a3+(1/3)a2b1+(1/3)a1b2+b3")
    print("  Since a1=b1=0: c2=a2+b2, c3=a3+b3")
    print("  (The cross-terms vanish when a1=b1=0!)")
    print()

    # When a1=b1=0, the ⊞_3 formula gives:
    # c_0 = 1
    # c_1 = a_1 + b_1 = 0
    # c_2 = a_2 + (2/3)*a_1*b_1 + b_2 = a_2 + b_2
    # c_3 = a_3 + (1/3)*a_2*b_1 + (1/3)*a_1*b_2 + b_3 = a_3 + b_3

    # So for centered cubics, ⊞_3 is just coefficient addition!
    # But we showed earlier that plain coefficient addition FAILS superadditivity ~40%.
    # Hmm, that contradicts... unless the centered case is special.

    # Wait, let me re-check. For centered cubics (a1=b1=0), ⊞_3 acts as:
    # c_2 = a_2 + b_2, c_3 = a_3 + b_3
    # So if superadditivity holds for ⊞_3 but not for plain coefficient addition,
    # the centered case should also sometimes fail... unless it's exactly the non-centered
    # cases that fail.

    # Let me test both centered and non-centered cases.

    np.random.seed(42)

    # Test centered (a1=b1=0)
    centered_violations = 0
    centered_total = 0
    for _ in range(2000):
        a2 = -np.random.uniform(0.5, 5)
        a3_max = 2 * (-a2/3)**(1.5)
        a3 = np.random.uniform(-0.9*a3_max, 0.9*a3_max)

        b2 = -np.random.uniform(0.5, 5)
        b3_max = 2 * (-b2/3)**(1.5)
        b3 = np.random.uniform(-0.9*b3_max, 0.9*b3_max)

        coeffs_p = np.array([1, 0, a2, a3])
        coeffs_q = np.array([1, 0, b2, b3])
        coeffs_conv = finite_free_conv_func(coeffs_p, coeffs_q)

        ip = inv_phi(coeffs_p)
        iq = inv_phi(coeffs_q)
        ic = inv_phi(coeffs_conv)

        if np.isnan(ip) or np.isnan(iq) or np.isnan(ic):
            continue

        centered_total += 1
        if ic < ip + iq - 1e-10:
            centered_violations += 1

    print(f"  Centered (a1=b1=0): {centered_violations}/{centered_total} violations")

    # Now also test: is plain addition the same as ⊞_3 for centered cubics?
    print("\n  Verifying: does ⊞_3 = plain addition when a1=b1=0?")
    for _ in range(5):
        a2 = -np.random.uniform(1, 3)
        a3_max = 2 * (-a2/3)**(1.5)
        a3 = np.random.uniform(-0.5*a3_max, 0.5*a3_max)
        b2 = -np.random.uniform(1, 3)
        b3_max = 2 * (-b2/3)**(1.5)
        b3 = np.random.uniform(-0.5*b3_max, 0.5*b3_max)

        coeffs_p = np.array([1, 0, a2, a3])
        coeffs_q = np.array([1, 0, b2, b3])
        coeffs_conv = finite_free_conv_func(coeffs_p, coeffs_q)
        coeffs_sum = coeffs_p + coeffs_q
        coeffs_sum[0] = 1

        print(f"    ⊞_3: {coeffs_conv}")
        print(f"    sum: {coeffs_sum}")
        print(f"    same? {np.allclose(coeffs_conv, coeffs_sum)}")

    # Test non-centered
    print()
    noncentered_violations = 0
    noncentered_total = 0
    for _ in range(2000):
        roots_p = np.sort(np.random.randn(3) * 2)
        roots_q = np.sort(np.random.randn(3) * 2)
        for i in range(1, 3):
            if roots_p[i] - roots_p[i-1] < 0.01:
                roots_p[i] = roots_p[i-1] + 0.01
            if roots_q[i] - roots_q[i-1] < 0.01:
                roots_q[i] = roots_q[i-1] + 0.01

        poly_p = np.polynomial.polynomial.polyfromroots(roots_p)[::-1]
        poly_p = poly_p / poly_p[0]
        poly_q = np.polynomial.polynomial.polyfromroots(roots_q)[::-1]
        poly_q = poly_q / poly_q[0]

        coeffs_conv = finite_free_conv_func(poly_p, poly_q)
        ip = inv_phi(poly_p)
        iq = inv_phi(poly_q)
        ic = inv_phi(coeffs_conv)

        if np.isnan(ip) or np.isnan(iq) or np.isnan(ic):
            continue

        noncentered_total += 1
        if ic < ip + iq - 1e-10:
            noncentered_violations += 1

    print(f"  Non-centered (general a1,b1): {noncentered_violations}/{noncentered_total} violations")

    # THE KEY QUESTION: when a1=b1=0 and ⊞_3 = plain addition,
    # does superadditivity still hold?
    # If YES: then the centered case has a proof purely from the
    # algebraic structure of Phi_3 as a function of (a2, a3).
    # The general case would then need to handle the a1 cross-terms.


def n3_discriminant_formula():
    """Explore Phi_3 in terms of the discriminant for centered cubics."""
    print("\n" + "=" * 65)
    print("n=3: Phi_3 vs discriminant for centered cubics")
    print("=" * 65)

    import numpy as np
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    _mod = __import__("verify-p4-inequality")
    phi_n_func = _mod.phi_n

    # For centered cubic x^3 + a2*x + a3:
    # disc = -4*a2^3 - 27*a3^2
    # Phi_3 = 9 * sum_i l_i^2 / (3*l_i^2 + a2)^2 where l1+l2+l3=0

    print("\n  For x^3 + a2*x + a3 (centered, a1=0):")
    print("  disc = -4*a2^3 - 27*a3^2")
    print()
    print(f"  {'a2':>6s} {'a3':>6s} {'disc':>10s} {'Phi_3':>10s} {'1/Phi_3':>10s} {'disc*1/Phi':>10s}")

    for a2, a3 in [(-3, 0), (-3, 1), (-3, 2), (-1, 0), (-1, 0.2),
                    (-4, 0), (-4, 2), (-4, 4), (-6, 0), (-6, 3)]:
        disc = -4*a2**3 - 27*a3**2
        if disc <= 0:
            continue
        coeffs = np.array([1.0, 0.0, float(a2), float(a3)])
        roots = np.sort(np.roots(coeffs)).real
        phi = phi_n_func(roots)
        if phi == float('inf') or phi == 0:
            continue
        inv_phi = 1.0 / phi
        print(f"  {a2:6.1f} {a3:6.1f} {disc:10.2f} {phi:10.4f} {inv_phi:10.6f} {disc*inv_phi:10.4f}")

    # Check if 1/Phi_3 = disc * g(a2, a3) for some simple g
    # Or if Phi_3 = h(a2, a3) / disc for some simple h
    print("\n  Checking: is Phi_3 * disc a simpler expression?")
    print(f"  {'a2':>6s} {'a3':>6s} {'Phi_3*disc':>12s} {'a2^2':>8s} {'ratio':>10s}")
    for a2 in [-1, -2, -3, -4, -5, -6]:
        for a3 in [0, 0.5, 1.0]:
            disc = -4*a2**3 - 27*a3**2
            if disc <= 0:
                continue
            coeffs = np.array([1.0, 0.0, float(a2), float(a3)])
            roots = np.sort(np.roots(coeffs)).real
            phi = phi_n_func(roots)
            if phi == float('inf') or phi == 0:
                continue
            print(f"  {a2:6.1f} {a3:6.1f} {phi*disc:12.4f} {a2**2:8.1f} "
                  f"{phi*disc/a2**2 if a2 != 0 else 'nan':>10.4f}")


if __name__ == "__main__":
    verify_n2()
    compute_phi3_symbolic()
    n3_coefficient_surplus()
    n3_discriminant_formula()
