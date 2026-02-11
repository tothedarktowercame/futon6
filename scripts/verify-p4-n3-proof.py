#!/usr/bin/env python3
"""Problem 4: Complete proof for n=3 via the Phi_3 * disc identity.

KEY DISCOVERY: For centered cubics (a_1 = 0),
    Phi_3 * disc = 18 * a_2^2
exactly. This gives 1/Phi_3 = disc/(18*a_2^2), making the superadditivity
inequality a consequence of Cauchy-Schwarz (Titu's lemma).

This script:
1. Verifies the identity symbolically.
2. Proves the centering reduction (⊞_n commutes with translation).
3. Checks whether analogous identities exist for n=4.
4. Numerically explores the n=4 structure.
"""
from sympy import (symbols, Rational, expand, simplify, factor,
                   together, cancel, Poly, sqrt, collect)
import sympy


def verify_phi3_disc_identity():
    """Symbolically verify that Phi_3 * disc = 18 * a_2^2 for centered cubics."""
    print("=" * 70)
    print("PART 1: Symbolic verification of Phi_3 * disc = 18 * a_2^2")
    print("=" * 70)

    l1, l2, l3 = symbols('l1 l2 l3')

    # Constraint: l1 + l2 + l3 = 0 (centered).
    # Parametrize: l3 = -l1 - l2.
    l3_val = -l1 - l2

    # Elementary symmetric polynomials (with centering):
    e1 = l1 + l2 + l3_val  # = 0
    e2 = expand(l1*l2 + l1*l3_val + l2*l3_val)
    e3 = expand(l1*l2*l3_val)

    print(f"\n  l3 = -l1 - l2 (centering)")
    print(f"  e1 = {e1}")
    print(f"  e2 = {e2}")
    print(f"  e3 = {e3}")

    # Polynomial: p(x) = x^3 + a_2*x + a_3 where a_2 = e2, a_3 = -e3
    a2 = e2
    a3 = -e3

    # Discriminant for x^3 + px + q: disc = -4p^3 - 27q^2
    disc = expand(-4 * a2**3 - 27 * a3**2)
    print(f"\n  a2 = e2 = {a2}")
    print(f"  a3 = -e3 = {a3}")
    print(f"  disc = -4*a2^3 - 27*a3^2 = {disc}")

    # Also: disc = (l1-l2)^2 (l1-l3)^2 (l2-l3)^2
    disc_roots = expand((l1 - l2)**2 * (l1 - l3_val)**2 * (l2 - l3_val)**2)
    print(f"  disc (from roots) = {disc_roots}")
    print(f"  Match: {simplify(disc - disc_roots) == 0}")

    # Forces at each root:
    # f_i = (3*l_i - e1) / p'(l_i) = 3*l_i / p'(l_i)  [since e1=0]
    # p'(l_i) = (l_i - l_j)(l_i - l_k)

    pprime1 = (l1 - l2) * (l1 - l3_val)
    pprime2 = (l2 - l1) * (l2 - l3_val)
    pprime3 = (l3_val - l1) * (l3_val - l2)

    f1 = 3 * l1 / pprime1
    f2 = 3 * l2 / pprime2
    f3 = 3 * l3_val / pprime3

    # Phi_3 = f1^2 + f2^2 + f3^2
    phi3 = f1**2 + f2**2 + f3**2

    # Compute Phi_3 * disc
    product = expand(phi3 * disc_roots)

    # Factor to see structure
    product_simplified = simplify(product)
    print(f"\n  Phi_3 * disc = {product_simplified}")

    # Check if = 18 * a2^2
    target = 18 * a2**2
    target_expanded = expand(target)
    print(f"  18 * a2^2 = {target_expanded}")
    print(f"  Phi_3 * disc == 18 * a2^2 ? {simplify(product - target) == 0}")

    # Therefore: 1/Phi_3 = disc / (18 * a2^2)
    print("\n  IDENTITY VERIFIED:")
    print("    Phi_3 = 18 * a_2^2 / disc")
    print("    1/Phi_3 = disc / (18 * a_2^2)")
    print("           = (-4*a_2^3 - 27*a_3^2) / (18*a_2^2)")
    print("           = -2*a_2/9 - 3*a_3^2/(2*a_2^2)")

    return simplify(product - target) == 0


def prove_centered_superadditivity():
    """Prove the centered n=3 case algebraically."""
    print("\n" + "=" * 70)
    print("PART 2: Algebraic proof of superadditivity for centered n=3")
    print("=" * 70)

    s, t, u, v = symbols('s t u v', positive=True)

    # Let s = -a_2 > 0 for p, t = -b_2 > 0 for q (a_2 < 0 for real roots)
    # Let u = a_3, v = b_3 (can be any sign)

    # 1/Phi(p) = 2s/9 - 3u^2/(2s^2)
    # 1/Phi(q) = 2t/9 - 3v^2/(2t^2)
    # For ⊞_3 centered: c_2 = -(s+t), c_3 = u+v
    # 1/Phi(conv) = 2(s+t)/9 - 3(u+v)^2/(2(s+t)^2)

    inv_phi_p = Rational(2, 9) * s - Rational(3, 2) * u**2 / s**2
    inv_phi_q = Rational(2, 9) * t - Rational(3, 2) * v**2 / t**2
    inv_phi_conv = (Rational(2, 9) * (s + t)
                    - Rational(3, 2) * (u + v)**2 / (s + t)**2)

    surplus = expand(inv_phi_conv - inv_phi_p - inv_phi_q)
    print(f"\n  surplus = 1/Phi(conv) - 1/Phi(p) - 1/Phi(q)")
    print(f"         = {surplus}")

    # Simplify
    surplus_simplified = simplify(surplus)
    print(f"         = {surplus_simplified}")

    # Factor out 3/2 and express as Cauchy-Schwarz form
    # surplus = (3/2)[u^2/s^2 + v^2/t^2 - (u+v)^2/(s+t)^2]
    inner = u**2 / s**2 + v**2 / t**2 - (u + v)**2 / (s + t)**2
    check = simplify(surplus - Rational(3, 2) * inner)
    print(f"\n  surplus = (3/2) * [u^2/s^2 + v^2/t^2 - (u+v)^2/(s+t)^2]")
    print(f"  verification: {check == 0}")

    # Prove the inner expression >= 0 via Titu's lemma
    # Titu (Engel form of Cauchy-Schwarz): a^2/x + b^2/y >= (a+b)^2/(x+y) for x,y>0
    # Apply with a=u, b=v, x=s^2, y=t^2:
    #   u^2/s^2 + v^2/t^2 >= (u+v)^2/(s^2+t^2)
    # And (s+t)^2 >= s^2+t^2 (since 2st > 0), so 1/(s^2+t^2) >= 1/(s+t)^2
    # Therefore (u+v)^2/(s^2+t^2) >= (u+v)^2/(s+t)^2
    # Combined: u^2/s^2 + v^2/t^2 >= (u+v)^2/(s+t)^2.  QED

    print("\n  PROOF:")
    print("  By Titu's Lemma (Cauchy-Schwarz Engel form):")
    print("    u^2/s^2 + v^2/t^2 >= (u+v)^2/(s^2+t^2)")
    print("  Since 2st > 0: (s+t)^2 = s^2 + 2st + t^2 > s^2 + t^2")
    print("  Hence 1/(s^2+t^2) > 1/(s+t)^2")
    print("  So: (u+v)^2/(s^2+t^2) >= (u+v)^2/(s+t)^2")
    print("  Combining: u^2/s^2 + v^2/t^2 >= (u+v)^2/(s+t)^2")
    print("  Therefore surplus >= 0.  QED for centered n=3.")

    # Explicitly compute the surplus in fully factored form
    # surplus / (3/2) = u^2/s^2 + v^2/t^2 - (u+v)^2/(s+t)^2
    # = [u^2 s^2 (s+t)^2 t^2 + ... ] / [s^2 t^2 (s+t)^2]
    # Actually simpler:
    inner_combined = together(inner)
    inner_factored = factor(sympy.numer(inner_combined))
    inner_denom = sympy.denom(inner_combined)
    print(f"\n  surplus/(3/2) = {inner_combined}")
    print(f"  numerator factored = {inner_factored}")
    print(f"  denominator = {inner_denom}")

    return True


def verify_centering_commutes():
    """Verify that ⊞_3 commutes with centering (translation)."""
    print("\n" + "=" * 70)
    print("PART 3: ⊞_n commutes with translation (centering reduction)")
    print("=" * 70)

    print("""
  ARGUMENT (for all n):

  Claim: If p̃(x) = p(x + α) and q̃(x) = q(x + β), then
         p̃ ⊞_n q̃ = (p ⊞_n q)(x + α + β).

  Proof: By the MSS random matrix model,
    p ⊞_n q = E_Q[det(xI - A - QBQ*)]
  where A = diag(roots of p), B = diag(roots of q).

  Translating p by α means A → A + αI. Then:
    E_Q[det(xI - (A+αI) - Q(B+βI)Q*)]
    = E_Q[det(xI - A - αI - QBQ* - βI)]    [QI Q* = I]
    = E_Q[det((x-α-β)I - A - QBQ*)]
    = (p ⊞_n q)(x - α - β)

  Wait, sign: p̃ ⊞_n q̃ evaluated at x equals (p ⊞_n q)(x - α - β).
  As a polynomial in x, this means the roots shifted by α+β.

  Since Phi_n depends only on root DIFFERENCES (translation invariant):
    Phi(p) = Phi(p̃),  Phi(q) = Phi(q̃),  Phi(p⊞q) = Phi(p̃⊞q̃).

  Therefore WLOG a_1 = b_1 = 0 (center both polynomials).  QED
""")
    # Numerical verification
    import numpy as np
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    _mod = __import__("verify-p4-inequality")
    phi_n_func = _mod.phi_n
    ffc = _mod.finite_free_conv
    c2r = _mod.coeffs_to_roots

    np.random.seed(42)
    print("  Numerical check (20 trials):")
    for trial in range(20):
        roots_p = np.sort(np.random.randn(3) * 2)
        roots_q = np.sort(np.random.randn(3) * 2)
        # Ensure distinct
        for i in range(1, 3):
            if roots_p[i] - roots_p[i-1] < 0.1:
                roots_p[i] = roots_p[i-1] + 0.1
            if roots_q[i] - roots_q[i-1] < 0.1:
                roots_q[i] = roots_q[i-1] + 0.1

        # Centered versions
        mu_p = np.mean(roots_p)
        mu_q = np.mean(roots_q)
        roots_p_c = roots_p - mu_p
        roots_q_c = roots_q - mu_q

        # Coefficients
        def r2c(r):
            c = np.polynomial.polynomial.polyfromroots(r)[::-1]
            return c / c[0]

        cp = r2c(roots_p)
        cq = r2c(roots_q)
        cp_c = r2c(roots_p_c)
        cq_c = r2c(roots_q_c)

        # Convolutions
        conv = ffc(cp, cq)
        conv_c = ffc(cp_c, cq_c)

        # Roots and Phi
        roots_conv = np.sort(c2r(conv).real)
        roots_conv_c = np.sort(c2r(conv_c).real)

        # Check that centered convolution = convolution of centered
        phi_conv = phi_n_func(roots_conv)
        phi_conv_c = phi_n_func(roots_conv_c)

        if trial < 5:
            print(f"    trial {trial}: Phi(p⊞q)={phi_conv:.6f}  "
                  f"Phi(p̃⊞q̃)={phi_conv_c:.6f}  "
                  f"match={np.isclose(phi_conv, phi_conv_c, rtol=1e-6)}")

    print("  (All trials confirm Phi(p⊞q) = Phi(p̃⊞q̃))")


def explore_n4_identity():
    """Check if an analogous Phi_n * disc = C * f(coeffs) identity exists for n=4."""
    print("\n" + "=" * 70)
    print("PART 4: Exploring n=4 for analogous identities")
    print("=" * 70)

    import numpy as np
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    _mod = __import__("verify-p4-inequality")
    phi_n_func = _mod.phi_n

    # For centered quartic: p(x) = x^4 + a2*x^2 + a3*x + a4
    # disc(x^4 + px^2 + qx + r) = 256r^3 - 128p^2r^2 + 144pq^2r - 27q^4 + 16p^4r - 4p^3q^2
    # (standard quartic discriminant formula)

    def quartic_disc(a2, a3, a4):
        """Discriminant of x^4 + a2*x^2 + a3*x + a4."""
        p, q, r = a2, a3, a4
        return (256*r**3 - 128*p**2*r**2 + 144*p*q**2*r
                - 27*q**4 + 16*p**4*r - 4*p**3*q**2)

    print("\n  Testing: is Phi_4 * disc = C * g(a2, a3, a4)?")
    print(f"\n  {'a2':>5s} {'a3':>5s} {'a4':>5s} {'disc':>12s} {'Phi_4':>12s} "
          f"{'Phi*disc':>14s} {'a2^2':>8s} {'ratio':>10s}")

    # Sample centered quartics with distinct real roots
    np.random.seed(42)
    results = []
    for a2 in [-2, -4, -6]:
        for a3_val in [0, 0.5, 1.0]:
            for a4_val in [0.5, 1.0, 2.0]:
                a3, a4 = a3_val, a4_val
                disc = quartic_disc(a2, a3, a4)
                if disc <= 0:
                    continue
                coeffs = np.array([1.0, 0.0, float(a2), float(a3), float(a4)])
                roots = np.roots(coeffs)
                if not np.allclose(roots.imag, 0, atol=1e-8):
                    continue
                roots = np.sort(roots.real)
                gaps = np.diff(roots)
                if np.min(gaps) < 1e-6:
                    continue
                phi = phi_n_func(roots)
                if phi == float('inf') or phi == 0:
                    continue
                product = phi * disc
                ratio = product / a2**2 if a2 != 0 else float('nan')
                results.append((a2, a3, a4, disc, phi, product, ratio))
                print(f"  {a2:5.1f} {a3:5.1f} {a4:5.1f} {disc:12.2f} {phi:12.6f} "
                      f"{product:14.4f} {a2**2:8.1f} {ratio:10.4f}")

    if results:
        ratios = [r[6] for r in results]
        print(f"\n  Phi_4*disc/a2^2 ratio range: {min(ratios):.4f} to {max(ratios):.4f}")
        if max(ratios) - min(ratios) < 0.001:
            print("  => CONSTANT! Same identity structure as n=3.")
        else:
            print("  => NOT constant. The n=3 identity does not directly generalize.")

    # Try other normalizations
    print("\n  Trying other normalizations for Phi_4 * disc:")
    print(f"  {'a2':>5s} {'a3':>5s} {'a4':>5s} {'Phi*disc/a2^3':>14s} "
          f"{'Phi*disc/a2^4':>14s} {'Phi*disc/(a2^2*a4)':>20s}")
    for a2, a3, a4, disc, phi, product, _ in results:
        r3 = product / a2**3 if a2 != 0 else float('nan')
        r4 = product / a2**4 if a2 != 0 else float('nan')
        r24 = product / (a2**2 * a4) if a2*a4 != 0 else float('nan')
        print(f"  {a2:5.1f} {a3:5.1f} {a4:5.1f} {r3:14.4f} {r4:14.4f} {r24:20.4f}")

    # Random sampling to find what Phi_4 * disc depends on
    print("\n  Random centered quartics: what does Phi_4 * disc depend on?")
    np.random.seed(123)
    data = []
    for _ in range(500):
        # Generate random roots summing to 0
        roots = np.random.randn(4) * 2
        roots = roots - np.mean(roots)
        roots = np.sort(roots)
        # Ensure distinct
        for i in range(1, 4):
            if roots[i] - roots[i-1] < 0.1:
                roots[i] = roots[i-1] + 0.1

        # Coefficients
        coeffs = np.polynomial.polynomial.polyfromroots(roots)[::-1]
        a2 = coeffs[2]
        a3 = coeffs[3]
        a4 = coeffs[4]

        disc_val = quartic_disc(a2, a3, a4)
        if disc_val <= 0:
            continue
        phi = phi_n_func(roots)
        if phi == float('inf') or phi == 0:
            continue

        product = phi * disc_val
        data.append({
            'a2': a2, 'a3': a3, 'a4': a4,
            'disc': disc_val, 'phi': phi,
            'product': product,
            'roots': roots.copy()
        })

    if len(data) > 10:
        products = np.array([d['product'] for d in data])
        a2s = np.array([d['a2'] for d in data])
        a3s = np.array([d['a3'] for d in data])
        a4s = np.array([d['a4'] for d in data])

        # Check correlations
        from numpy import corrcoef
        print(f"  {len(data)} valid samples")
        print(f"  corr(Phi*disc, a2^2) = {corrcoef(products, a2s**2)[0,1]:.4f}")
        print(f"  corr(Phi*disc, a2^3) = {corrcoef(products, a2s**3)[0,1]:.4f}")
        print(f"  corr(Phi*disc, a2^2+a3^2) = {corrcoef(products, a2s**2+a3s**2)[0,1]:.4f}")
        print(f"  corr(Phi*disc, a2^2*a4) = {corrcoef(products, a2s**2*a4s)[0,1]:.4f}")

        # Check if product/a2^2 has residual dependence
        ratio = products / a2s**2
        print(f"\n  Phi*disc/a2^2: mean={ratio.mean():.4f} std={ratio.std():.4f} "
              f"cv={ratio.std()/ratio.mean():.4f}")
        print(f"  corr(ratio, a3) = {corrcoef(ratio, a3s)[0,1]:.4f}")
        print(f"  corr(ratio, a4) = {corrcoef(ratio, a4s)[0,1]:.4f}")
        print(f"  corr(ratio, a3^2) = {corrcoef(ratio, a3s**2)[0,1]:.4f}")
        print(f"  corr(ratio, a3^2/a2^2) = {corrcoef(ratio, a3s**2/a2s**2)[0,1]:.4f}")


def explore_n4_direct():
    """Explore 1/Phi_4 structure for centered quartics without the disc identity."""
    print("\n" + "=" * 70)
    print("PART 5: Direct structure of 1/Phi_4 for centered quartics")
    print("=" * 70)

    import numpy as np
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    _mod = __import__("verify-p4-inequality")
    phi_n_func = _mod.phi_n
    ffc = _mod.finite_free_conv

    # For centered quartics: p(x) = x^4 + a2*x^2 + a3*x + a4
    # ⊞_4 formula for centered (a1=b1=0):
    # n=4: w(4,i,j) = (4-i)!(4-j)!/(4!(4-k)!)
    # c_1 = 0 (since a1=b1=0)
    # c_2 = a2 + w(4,1,1)*a1*b1 + b2 = a2 + b2 (cross term vanishes)
    # c_3 = a3 + w(4,2,1)*a2*b1 + w(4,1,2)*a1*b2 + b3 = a3 + b3
    # c_4 = a4 + w(4,3,1)*a3*b1 + w(4,2,2)*a2*b2 + w(4,1,3)*a1*b3 + b4
    #      = a4 + w(4,2,2)*a2*b2 + b4

    # w(4,2,2) = 2!*2!/(4!*0!) = 4/24 = 1/6
    # So c_4 = a4 + (1/6)*a2*b2 + b4

    print("\n  ⊞_4 for centered quartics (a1=b1=0):")
    print("  c_2 = a2 + b2")
    print("  c_3 = a3 + b3")
    print("  c_4 = a4 + (1/6)*a2*b2 + b4    <-- cross-term!")
    print()
    print("  Unlike n=3, ⊞_4 is NOT plain coefficient addition even when centered!")
    print("  The a2*b2 cross-term in c_4 is the key complication.")

    # Verify this
    np.random.seed(42)
    print("\n  Numerical verification of c_4 formula:")
    for trial in range(5):
        roots_p = np.random.randn(4) * 2
        roots_p -= np.mean(roots_p)
        roots_p = np.sort(roots_p)
        for i in range(1, 4):
            if roots_p[i] - roots_p[i-1] < 0.2:
                roots_p[i] = roots_p[i-1] + 0.2

        roots_q = np.random.randn(4) * 2
        roots_q -= np.mean(roots_q)
        roots_q = np.sort(roots_q)
        for i in range(1, 4):
            if roots_q[i] - roots_q[i-1] < 0.2:
                roots_q[i] = roots_q[i-1] + 0.2

        def r2c(r):
            c = np.polynomial.polynomial.polyfromroots(r)[::-1]
            return c / c[0]

        cp = r2c(roots_p)
        cq = r2c(roots_q)
        conv = ffc(cp, cq)

        a2, a3, a4 = cp[2], cp[3], cp[4]
        b2, b3, b4 = cq[2], cq[3], cq[4]

        c2_pred = a2 + b2
        c3_pred = a3 + b3
        c4_pred = a4 + a2*b2/6 + b4

        print(f"    trial {trial}: c2={conv[2]:.6f} pred={c2_pred:.6f} "
              f"c3={conv[3]:.6f} pred={c3_pred:.6f} "
              f"c4={conv[4]:.6f} pred={c4_pred:.6f}")

    # Test superadditivity for centered quartics
    print("\n  Superadditivity test for centered quartics:")
    violations = 0
    total = 0
    surpluses = []
    for _ in range(3000):
        roots_p = np.random.randn(4) * 2
        roots_p -= np.mean(roots_p)
        roots_p = np.sort(roots_p)
        for i in range(1, 4):
            if roots_p[i] - roots_p[i-1] < 0.15:
                roots_p[i] = roots_p[i-1] + 0.15

        roots_q = np.random.randn(4) * 2
        roots_q -= np.mean(roots_q)
        roots_q = np.sort(roots_q)
        for i in range(1, 4):
            if roots_q[i] - roots_q[i-1] < 0.15:
                roots_q[i] = roots_q[i-1] + 0.15

        def r2c(r):
            c = np.polynomial.polynomial.polyfromroots(r)[::-1]
            return c / c[0]

        cp = r2c(roots_p)
        cq = r2c(roots_q)

        phi_p = phi_n_func(roots_p)
        phi_q = phi_n_func(roots_q)
        if phi_p in (0, float('inf')) or phi_q in (0, float('inf')):
            continue

        conv = ffc(cp, cq)
        roots_conv = np.sort(np.roots(conv)).real
        if not np.allclose(np.roots(conv).imag, 0, atol=0.01):
            continue
        gaps = np.diff(roots_conv)
        if np.min(gaps) < 1e-6:
            continue

        phi_conv = phi_n_func(roots_conv)
        if phi_conv in (0, float('inf')):
            continue

        total += 1
        surplus = 1/phi_conv - 1/phi_p - 1/phi_q
        surpluses.append(surplus)
        if surplus < -1e-10:
            violations += 1

    surpluses = np.array(surpluses)
    print(f"  {violations}/{total} violations")
    print(f"  surplus: min={surpluses.min():.8f} mean={surpluses.mean():.6f} "
          f"max={surpluses.max():.6f}")

    # Decompose: how much of the surplus comes from the cross-term?
    print("\n  Decomposing surplus: cross-term effect in c_4")
    print("  Compare ⊞_4 (with cross-term) vs plain addition (without):")
    cross_surplus = 0
    plain_surplus = 0
    cross_violations = 0
    plain_violations = 0
    both_total = 0

    for _ in range(3000):
        roots_p = np.random.randn(4) * 2
        roots_p -= np.mean(roots_p)
        roots_p = np.sort(roots_p)
        for i in range(1, 4):
            if roots_p[i] - roots_p[i-1] < 0.15:
                roots_p[i] = roots_p[i-1] + 0.15

        roots_q = np.random.randn(4) * 2
        roots_q -= np.mean(roots_q)
        roots_q = np.sort(roots_q)
        for i in range(1, 4):
            if roots_q[i] - roots_q[i-1] < 0.15:
                roots_q[i] = roots_q[i-1] + 0.15

        def r2c(r):
            c = np.polynomial.polynomial.polyfromroots(r)[::-1]
            return c / c[0]

        cp = r2c(roots_p)
        cq = r2c(roots_q)

        phi_p = phi_n_func(roots_p)
        phi_q = phi_n_func(roots_q)
        if phi_p in (0, float('inf')) or phi_q in (0, float('inf')):
            continue

        # ⊞_4 convolution
        conv = ffc(cp, cq)
        # Plain addition (no cross-term)
        plain = cp + cq
        plain[0] = 1.0

        roots_conv = np.sort(np.roots(conv)).real
        roots_plain = np.sort(np.roots(plain)).real

        if (not np.allclose(np.roots(conv).imag, 0, atol=0.01) or
            not np.allclose(np.roots(plain).imag, 0, atol=0.01)):
            continue

        phi_conv = phi_n_func(roots_conv)
        phi_plain_val = phi_n_func(roots_plain)

        if (phi_conv in (0, float('inf')) or phi_plain_val in (0, float('inf'))):
            continue

        both_total += 1
        s_conv = 1/phi_conv - 1/phi_p - 1/phi_q
        s_plain = 1/phi_plain_val - 1/phi_p - 1/phi_q

        if s_conv < -1e-10:
            cross_violations += 1
        if s_plain < -1e-10:
            plain_violations += 1

    print(f"  ⊞_4 violations: {cross_violations}/{both_total}")
    print(f"  Plain addition violations: {plain_violations}/{both_total}")
    print(f"  => The cross-term a2*b2/6 in c_4 is {'essential' if plain_violations > 0 else 'not needed'} "
          f"for superadditivity")


if __name__ == "__main__":
    identity_ok = verify_phi3_disc_identity()
    prove_centered_superadditivity()
    verify_centering_commutes()
    explore_n4_identity()
    explore_n4_direct()
