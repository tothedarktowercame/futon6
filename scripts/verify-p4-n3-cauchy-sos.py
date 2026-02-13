#!/usr/bin/env python3
"""Task D (P4): explicit n=3 coefficient-addition route with SOS-ready surplus.

This script does four things:
1. Builds p ⊞_3 q directly from the 6 matchings and verifies MSS coefficient
   formulas in elementary symmetric coordinates.
2. Specializes to centered cubics and confirms coefficient-wise addition.
3. Verifies the centered cubic identity
      1/Phi_3 = (-4*e2^3 - 27*e3^2) / (18*e2^2)
   as a symbolic identity from roots.
4. Derives the 4-variable surplus polynomial and gives an explicit SOS-style
   decomposition of its numerator.
"""

from itertools import permutations

import sympy as sp


def symmetric_sums_3(r1, r2, r3):
    """Return (e1, e2, e3) for three roots."""
    e1 = r1 + r2 + r3
    e2 = r1 * r2 + r1 * r3 + r2 * r3
    e3 = r1 * r2 * r3
    return e1, e2, e3


def verify_matching_average_mss_formulas():
    """Verify n=3 MSS coefficients from explicit averaging over S_3."""
    x = sp.symbols("x")
    g1, g2, g3, d1, d2, d3 = sp.symbols("g1 g2 g3 d1 d2 d3")
    g = [g1, g2, g3]
    d = [d1, d2, d3]

    match_polys = []
    for perm in permutations(range(3)):
        factors = [x - (g[i] + d[perm[i]]) for i in range(3)]
        match_polys.append(sp.expand(sp.prod(factors)))
    avg_poly = sp.expand(sp.Rational(1, 6) * sum(match_polys))
    poly = sp.Poly(avg_poly, x)

    # For monic cubic x^3 - E1 x^2 + E2 x - E3.
    E1 = -poly.nth(2)
    E2 = poly.nth(1)
    E3 = -poly.nth(0)

    e1p, e2p, e3p = symmetric_sums_3(g1, g2, g3)
    e1q, e2q, e3q = symmetric_sums_3(d1, d2, d3)

    E1_expected = e1p + e1q
    E2_expected = e2p + sp.Rational(2, 3) * e1p * e1q + e2q
    E3_expected = (
        e3p
        + sp.Rational(1, 3) * e2p * e1q
        + sp.Rational(1, 3) * e1p * e2q
        + e3q
    )

    checks = {
        "E1_formula_ok": sp.simplify(E1 - E1_expected) == 0,
        "E2_formula_ok": sp.simplify(E2 - E2_expected) == 0,
        "E3_formula_ok": sp.simplify(E3 - E3_expected) == 0,
    }

    # Center both cubics: e1(p)=e1(q)=0.
    centered_subs = {g3: -g1 - g2, d3: -d1 - d2}
    E1_centered = sp.expand(E1.subs(centered_subs))
    E2_centered = sp.expand(E2.subs(centered_subs))
    E3_centered = sp.expand(E3.subs(centered_subs))
    e2p_centered = sp.expand(e2p.subs(centered_subs))
    e3p_centered = sp.expand(e3p.subs(centered_subs))
    e2q_centered = sp.expand(e2q.subs(centered_subs))
    e3q_centered = sp.expand(e3q.subs(centered_subs))

    centered_checks = {
        "E1_centered_zero": sp.simplify(E1_centered) == 0,
        "E2_centered_additive": sp.simplify(E2_centered - (e2p_centered + e2q_centered)) == 0,
        "E3_centered_additive": sp.simplify(E3_centered - (e3p_centered + e3q_centered)) == 0,
    }

    return avg_poly, checks, centered_checks


def verify_inv_phi3_formula():
    """Verify centered cubic identity for 1/Phi_3 in (e2, e3)."""
    l1, l2 = sp.symbols("l1 l2")
    l3 = -l1 - l2

    e1, e2, e3 = symmetric_sums_3(l1, l2, l3)
    assert sp.simplify(e1) == 0

    f1 = 1 / (l1 - l2) + 1 / (l1 - l3)
    f2 = 1 / (l2 - l1) + 1 / (l2 - l3)
    f3 = 1 / (l3 - l1) + 1 / (l3 - l2)
    phi3 = sp.simplify(f1**2 + f2**2 + f3**2)

    inv_phi_from_roots = sp.simplify(1 / phi3)
    inv_phi_from_coeffs = sp.simplify((-4 * e2**3 - 27 * e3**2) / (18 * e2**2))
    inv_phi_simplified = sp.simplify(-sp.Rational(2, 9) * e2 - sp.Rational(3, 2) * e3**2 / e2**2)
    e2s, e3s = sp.symbols("e2 e3")
    inv_phi_e2e3 = -sp.Rational(2, 9) * e2s - sp.Rational(3, 2) * e3s**2 / e2s**2

    return {
        "identity_disc_form_ok": sp.simplify(inv_phi_from_roots - inv_phi_from_coeffs) == 0,
        "identity_simplified_form_ok": sp.simplify(inv_phi_from_coeffs - inv_phi_simplified) == 0,
        "inv_phi_expr": inv_phi_e2e3,
    }


def build_surplus_sos_certificate():
    """Build 4-variable surplus polynomial and explicit SOS-style decomposition."""
    s, t = sp.symbols("s t", positive=True)
    u, v = sp.symbols("u v", real=True)

    inv_phi_p = sp.Rational(2, 9) * s - sp.Rational(3, 2) * u**2 / s**2
    inv_phi_q = sp.Rational(2, 9) * t - sp.Rational(3, 2) * v**2 / t**2
    inv_phi_conv = sp.Rational(2, 9) * (s + t) - sp.Rational(3, 2) * (u + v) ** 2 / (s + t) ** 2
    surplus = sp.simplify(inv_phi_conv - inv_phi_p - inv_phi_q)

    # Strip the harmless positive factor (3/2).
    inner = sp.simplify(sp.Rational(2, 3) * surplus)
    inner_num, inner_den = sp.fraction(sp.together(inner))
    inner_num = sp.expand(inner_num)
    inner_den = sp.expand(inner_den)

    # Explicit SOS-style decomposition on the cone s,t>0:
    # N = (s^2 v - t^2 u)^2 + s t (s v + t u)^2 + s t (s v - t u)^2.
    sos_num_decomp = (
        (s**2 * v - t**2 * u) ** 2
        + s * t * (s * v + t * u) ** 2
        + s * t * (s * v - t * u) ** 2
    )
    sos_num_expanded = sp.expand(sos_num_decomp)
    sos_ok = sp.simplify(inner_num - sos_num_expanded) == 0

    # Real-rooted centered cubic constraints (depressed cubic discriminants).
    # p(x)=x^3-sx-u and q(x)=x^3-tx-v have 3 distinct real roots iff:
    g_p = 4 * s**3 - 27 * u**2
    g_q = 4 * t**3 - 27 * v**2
    g_conv = 4 * (s + t) ** 3 - 27 * (u + v) ** 2

    return {
        "surplus_expr": surplus,
        "inner_expr": inner,
        "inner_num": inner_num,
        "inner_den": inner_den,
        "sos_num_decomp": sos_num_decomp,
        "sos_num_expanded": sos_num_expanded,
        "sos_ok": sos_ok,
        "cone_constraints": (g_p, g_q, g_conv),
    }


def main():
    print("=" * 72)
    print("Task D: n=3 explicit coefficient-addition route (matching average + SOS)")
    print("=" * 72)

    avg_poly, checks, centered_checks = verify_matching_average_mss_formulas()
    print("\n[1] Matching-average expansion over S_3")
    print("Average polynomial:")
    print(f"  {avg_poly}")
    print("General n=3 coefficient checks:")
    for k, v in checks.items():
        print(f"  {k}: {v}")
    print("Centered checks (e1(p)=e1(q)=0):")
    for k, v in centered_checks.items():
        print(f"  {k}: {v}")

    phi_checks = verify_inv_phi3_formula()
    print("\n[2] Centered cubic identity for 1/Phi_3")
    print(f"  identity_disc_form_ok: {phi_checks['identity_disc_form_ok']}")
    print(f"  identity_simplified_form_ok: {phi_checks['identity_simplified_form_ok']}")
    print(f"  1/Phi_3(e2,e3) = {phi_checks['inv_phi_expr']}")

    cert = build_surplus_sos_certificate()
    print("\n[3] Surplus for p(x)=x^3-sx-u, q(x)=x^3-tx-v with s,t>0")
    print(f"  surplus = {cert['surplus_expr']}")
    print(f"  inner = (2/3)*surplus = {cert['inner_expr']}")
    print(f"  numerator N(s,t,u,v) = {cert['inner_num']}")
    print(f"  denominator D(s,t,u,v) = {cert['inner_den']}")
    print("  SOS-style decomposition N =")
    print(f"    {cert['sos_num_decomp']}")
    print("  expanded N from decomposition =")
    print(f"    {cert['sos_num_expanded']}")
    print(f"  sos_ok: {cert['sos_ok']}")

    gp, gq, gc = cert["cone_constraints"]
    print("\n[4] Real-rooted centered cubic constraints (discriminants)")
    print(f"  g_p = {gp}  >= 0")
    print(f"  g_q = {gq}  >= 0")
    print(f"  g_conv = {gc}  >= 0")
    print("\nConclusion: Task D closes symbolically for n=3 with an explicit")
    print("SOS-ready numerator decomposition on s>0, t>0.")


if __name__ == "__main__":
    main()
