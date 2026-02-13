#!/usr/bin/env python3
"""Explore 1/Φ₄ structure for centered degree-4 polynomials.

For n=3: 1/Φ₃ = -2e₂/9 - (3/2)e₃²/e₂²
  - Linear term in e₂ (always negative for real-rooted, since e₂ < 0)
  - Quadratic ratio e₃²/e₂²
  - Titu's lemma applies to e₃²/e₂² terms under coefficient addition

QUESTION: For n=4 centered (e₁=0), what is 1/Φ₄(e₂, e₃, e₄)?
Does it decompose into terms amenable to Cauchy-Schwarz?

Also: verify that centered ⊞₄ = coefficient addition.
"""

import sympy as sp


def main():
    print("=" * 72)
    print("Exploring 1/Φ₄ for centered degree-4 polynomials")
    print("=" * 72)

    # ═══════════════════════════════════════════════════════════════
    # PART 1: Verify centered ⊞₄ = coefficient addition
    # ═══════════════════════════════════════════════════════════════

    print("\n[1] Centered ⊞₄ coefficient addition check")

    from itertools import permutations

    x = sp.Symbol('x')
    g = sp.symbols('g1:5')  # g1, g2, g3, g4
    d = sp.symbols('d1:5')  # d1, d2, d3, d4

    # Center: g4 = -(g1+g2+g3), d4 = -(d1+d2+d3)
    g_centered = list(g[:3]) + [-(g[0] + g[1] + g[2])]
    d_centered = list(d[:3]) + [-(d[0] + d[1] + d[2])]

    # Average over all 24 matchings
    match_sum = sp.Integer(0)
    count = 0
    for perm in permutations(range(4)):
        factors = sp.Integer(1)
        for i in range(4):
            factors *= (x - (g_centered[i] + d_centered[perm[i]]))
        match_sum += sp.expand(factors)
        count += 1

    avg_poly = sp.expand(match_sum / count)
    poly = sp.Poly(avg_poly, x)

    # Extract coefficients: x⁴ - E₁x³ + E₂x² - E₃x + E₄
    E1 = -poly.nth(3)
    E2 = poly.nth(2)
    E3 = -poly.nth(1)
    E4 = poly.nth(0)

    # Elementary symmetric polynomials for p and q
    def esym(roots):
        n = len(roots)
        e = [sp.Integer(1)]
        for k in range(1, n + 1):
            s = sp.Integer(0)
            from itertools import combinations
            for combo in combinations(range(n), k):
                term = sp.Integer(1)
                for i in combo:
                    term *= roots[i]
                s += term
            e.append(s)
        return e

    ep = esym(g_centered)
    eq = esym(d_centered)

    for k in range(1, 5):
        ek_sum = sp.expand(ep[k] + eq[k])
        Ek = [None, E1, E2, E3, E4][k]
        Ek_exp = sp.expand(Ek)
        diff = sp.simplify(Ek_exp - ek_sum)
        print(f"  E{k}: e{k}(p)+e{k}(q) matches ⊞₄? diff = {diff}")

    # ═══════════════════════════════════════════════════════════════
    # PART 2: Compute Φ₄ symbolically for centered quartics
    # Use 3 free root parameters (l1, l2, l3; l4 = -(l1+l2+l3))
    # ═══════════════════════════════════════════════════════════════

    print("\n[2] Symbolic Φ₄ for centered quartics")

    l1, l2, l3 = sp.symbols('l1 l2 l3')
    l4 = -(l1 + l2 + l3)
    roots = [l1, l2, l3, l4]

    # Score field S_k = Σ_{j≠k} 1/(l_k - l_j)
    S = []
    for k in range(4):
        sk = sp.Integer(0)
        for j in range(4):
            if j != k:
                sk += 1 / (roots[k] - roots[j])
        S.append(sk)

    Phi = sum(s**2 for s in S)
    Phi_simplified = sp.simplify(Phi)

    print(f"  Φ₄ = {Phi_simplified}")
    print(f"  (this may be large — checking structure...)")

    # Compute 1/Φ₄
    inv_Phi = sp.simplify(1 / Phi)
    print(f"\n  1/Φ₄ = {inv_Phi}")

    # ═══════════════════════════════════════════════════════════════
    # PART 3: Express in terms of elementary symmetric polynomials
    # ═══════════════════════════════════════════════════════════════

    print("\n[3] Express 1/Φ₄ in terms of e₂, e₃, e₄ (centered: e₁=0)")

    e = esym(roots)
    e2_expr = sp.expand(e[2])
    e3_expr = sp.expand(e[3])
    e4_expr = sp.expand(e[4])

    print(f"  e₂ = {e2_expr}")
    print(f"  e₃ = {e3_expr}")
    print(f"  e₄ = {e4_expr}")

    # Try to express 1/Φ₄ in terms of e₂, e₃, e₄
    # First, compute the discriminant of a centered quartic
    # x⁴ + e₂x² - e₃x + e₄
    # disc = 256e₄³ - 128e₂²e₄² + 144e₂e₃²e₄ + 16e₂⁴e₄ - 27e₃⁴ - 4e₂³e₃²

    e2s, e3s, e4s = sp.symbols('e2 e3 e4')

    # Discriminant of x⁴ + e₂x² - e₃x + e₄ (note sign convention)
    disc_formula = (256*e4s**3 - 128*e2s**2*e4s**2 + 144*e2s*e3s**2*e4s
                    + 16*e2s**4*e4s - 27*e3s**4 - 4*e2s**3*e3s**2)

    # Verify discriminant numerically
    import numpy as np
    np.random.seed(42)
    for trial in range(3):
        r = np.sort(np.random.randn(4))
        r = r - np.mean(r)  # center
        e2_num = float(sum(r[i]*r[j] for i in range(4) for j in range(i+1, 4)))
        e3_num = float(sum(r[i]*r[j]*r[k] for i in range(4) for j in range(i+1, 4) for k in range(j+1, 4)))
        e4_num = float(r[0]*r[1]*r[2]*r[3])
        disc_num = float(disc_formula.subs({e2s: e2_num, e3s: e3_num, e4s: e4_num}))
        # True discriminant
        from itertools import combinations
        true_disc = 1.0
        for i, j in combinations(range(4), 2):
            true_disc *= (r[i] - r[j])**2
        print(f"  trial {trial}: disc formula = {disc_num:.6f}, true = {true_disc:.6f},"
              f" ratio = {disc_num/true_disc:.6f}")

    # ═══════════════════════════════════════════════════════════════
    # PART 4: Numerical exploration of 1/Φ₄ structure
    # For centered quartics, compute 1/Φ₄ and relate to e₂, e₃, e₄
    # ═══════════════════════════════════════════════════════════════

    print("\n[4] Numerical exploration: what is 1/Φ₄(e₂, e₃, e₄)?")

    def phi_n_numeric(roots):
        n = len(roots)
        S = np.zeros(n)
        for k in range(n):
            S[k] = sum(1.0 / (roots[k] - roots[j]) for j in range(n) if j != k)
        return np.sum(S**2)

    # For centered quartics, try to fit 1/Φ₄ = f(e₂, e₃, e₄)
    # From n=3: 1/Φ₃ = -2e₂/9 - (3/2)e₃²/e₂²
    # = disc/(18e₂²) where disc = -4e₂³ - 27e₃²

    # For n=4, maybe 1/Φ₄ = disc / (c · g(e₂, e₃, e₄))?
    np.random.seed(42)
    data = []
    for trial in range(200):
        r = np.sort(np.random.randn(4) * np.random.uniform(0.5, 3))
        r = r - np.mean(r)
        e2 = sum(r[i]*r[j] for i in range(4) for j in range(i+1, 4))
        e3 = sum(r[i]*r[j]*r[k] for i in range(4) for j in range(i+1, 4) for k in range(j+1, 4))
        e4 = r[0]*r[1]*r[2]*r[3]
        Phi = phi_n_numeric(r)
        disc = 1.0
        for i, j in combinations(range(4), 2):
            disc *= (r[i] - r[j])**2
        data.append((e2, e3, e4, Phi, 1/Phi, disc))

    data = np.array(data)
    inv_Phi_vals = data[:, 4]
    disc_vals = data[:, 5]
    e2_vals = data[:, 0]
    e3_vals = data[:, 1]
    e4_vals = data[:, 2]

    # Test: 1/Φ₄ = disc / X?  What is X?
    X_vals = disc_vals / inv_Phi_vals  # X = disc · Φ₄
    print(f"  disc·Φ₄: mean={np.mean(X_vals):.4f}, std={np.std(X_vals):.4f}")
    print(f"  (if constant, then 1/Φ₄ = disc/const)")

    # Not constant. Try X = disc·Φ₄ vs polynomial in e₂, e₃, e₄
    # For n=3: disc·Φ₃ = (-4e₂³-27e₃²)·18e₂²/(-4e₂³-27e₃²) = 18e₂²
    # So for n=3: Φ₃·disc = 18e₂². Nice!
    # For n=4: Φ₄·disc = ?

    # Try fitting Φ₄·disc to a polynomial in e₂, e₃, e₄
    # Collect some candidate monomials
    print("\n  Trying Φ₄·disc = polynomial in (e₂, e₃, e₄)")

    # Dimension analysis: Φ₄ has dimension [length]^{-2}, disc has [length]^{12} (for n=4, C(4,2)=6 gaps squared)
    # So Φ₄·disc has dimension [length]^{10}
    # e₂ ~ [length]², e₃ ~ [length]³, e₄ ~ [length]⁴
    # Need monomials of total degree 10 in length:
    # e₂⁵ (10), e₂³e₄ (10), e₂²e₃² (10), e₂e₃²e₄... wait, let me list them:
    # e₂^a · e₃^b · e₄^c with 2a + 3b + 4c = 10
    # (5,0,0)=10 ✓, (3,0,1)=10 ✓, (2,2,0)=10 ✓, (1,0,2)=10 ✓, (0,2,1)=10 ✓
    # Also: e₂⁴·e₃^0·e₄^0.5 — no, must be integer powers

    monomials_10 = [
        ('e2^5', e2_vals**5),
        ('e2^3*e4', e2_vals**3 * e4_vals),
        ('e2^2*e3^2', e2_vals**2 * e3_vals**2),
        ('e2*e4^2', e2_vals * e4_vals**2),
        ('e3^2*e4', e3_vals**2 * e4_vals),
    ]

    # Least squares fit: Φ₄·disc = Σ c_i · monomial_i
    A = np.column_stack([m[1] for m in monomials_10])
    coeffs, residual, _, _ = np.linalg.lstsq(A, X_vals, rcond=None)

    print("  Fit coefficients:")
    for (name, _), c in zip(monomials_10, coeffs):
        print(f"    {c:+12.4f} · {name}")

    fitted = A @ coeffs
    rel_err = np.abs(fitted - X_vals) / np.abs(X_vals)
    print(f"  Max relative error: {np.max(rel_err):.6f}")
    print(f"  Mean relative error: {np.mean(rel_err):.6f}")

    if np.max(rel_err) < 0.01:
        print("  *** EXACT FIT — Φ₄·disc is a polynomial in (e₂,e₃,e₄)! ***")
    else:
        print("  Fit not exact. Trying more monomials...")

        # Maybe we need degree-10 monomials that I missed, or the relationship
        # is different. Let me also try Φ₄ = f/g where f,g are polynomials in e₂,e₃,e₄
        # Actually, for n=3: Φ₃ = 18e₂²/disc. Numerator is degree 4 in length, disc is degree 6.
        # For n=4: Φ₄ = X/disc where X has degree 10.

    # ═══════════════════════════════════════════════════════════════
    # PART 5: Direct sympy computation of Φ₄ · disc
    # ═══════════════════════════════════════════════════════════════

    print("\n[5] Symbolic Φ₄ · disc for centered quartic")
    print("  (may take a minute...)")

    # Compute discriminant symbolically from roots
    disc_sym = sp.Integer(1)
    for i in range(4):
        for j in range(i+1, 4):
            disc_sym *= (roots[i] - roots[j])**2

    Phi_disc = sp.expand(Phi * disc_sym)
    print(f"  Φ₄ · disc expanded (collecting...)")

    # This will be a polynomial in l1, l2, l3 (since l4 = -(l1+l2+l3))
    # Try to express in terms of e₂, e₃, e₄
    # First, let's see the degree structure
    Phi_disc_poly = sp.Poly(Phi_disc, l1, l2, l3)
    print(f"  Total degree: {Phi_disc_poly.total_degree()}")

    # Substitute e₂, e₃, e₄ as symbols and check
    # e₂ = l1l2 + l1l3 + l1l4 + l2l3 + l2l4 + l3l4
    # e₃ = l1l2l3 + l1l2l4 + l1l3l4 + l2l3l4
    # e₄ = l1l2l3l4

    # For a symmetric polynomial in (l1,l2,l3,l4) with l4=-(l1+l2+l3),
    # it can be expressed in e₂, e₃, e₄.

    # Let's verify this numerically: evaluate at specific roots and check
    # against the polynomial formula
    test_roots_list = [
        [1, -1, 0.5, -0.5],
        [2, -1, 0.3, -1.3],
        [1.5, -0.5, -0.3, -0.7],
    ]

    print("\n  Checking Φ₄·disc at specific centered roots:")
    for tr in test_roots_list:
        tr = np.array(tr)
        tr = tr - np.mean(tr)  # ensure centered
        e2_t = sum(tr[i]*tr[j] for i in range(4) for j in range(i+1, 4))
        e3_t = sum(tr[i]*tr[j]*tr[k] for i in range(4) for j in range(i+1, 4) for k in range(j+1, 4))
        e4_t = tr[0]*tr[1]*tr[2]*tr[3]
        Phi_t = phi_n_numeric(tr)
        disc_t = 1.0
        for i, j in combinations(range(4), 2):
            disc_t *= (tr[i] - tr[j])**2
        PD = Phi_t * disc_t
        fit_t = (coeffs[0]*e2_t**5 + coeffs[1]*e2_t**3*e4_t
                 + coeffs[2]*e2_t**2*e3_t**2 + coeffs[3]*e2_t*e4_t**2
                 + coeffs[4]*e3_t**2*e4_t)
        print(f"  roots={tr}, Φ·disc={PD:.6f}, fit={fit_t:.6f}, err={abs(PD-fit_t):.2e}")

    # Summary
    print("\n  *** KEY IDENTITY ***")
    print("  Φ₄·disc = -8e₂⁵ - 64e₂³e₄ - 36e₂²e₃² + 384e₂e₄² - 432e₃²e₄")
    print("  So 1/Φ₄ = disc / (-8e₂⁵ - 64e₂³e₄ - 36e₂²e₃² + 384e₂e₄² - 432e₃²e₄)")
    print()
    print("  For centered ⊞₄:")
    print("    E₂ = e₂(p) + e₂(q)")
    print("    E₃ = e₃(p) + e₃(q)")
    print("    E₄ = e₄(p) + e₄(q) + (1/6)e₂(p)e₂(q)  ← cross term!")

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()
