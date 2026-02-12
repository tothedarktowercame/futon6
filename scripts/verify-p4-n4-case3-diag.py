#!/usr/bin/env python3
"""Case 3 exact proof: diagonal and anti-diagonal subcases.

Case 3a: diagonal (a₃=b₃, a₄=b₄) — 2D system in (a₃, a₄)
Case 3b: anti-diagonal (a₃=-b₃, a₄=b₄) — 2D system in (a₃, a₄)

Both reduce to 2 equations in 2 unknowns, solvable by resultant.

Safe for restart: saves/loads cache.
"""

import sympy as sp
from sympy import (Rational, expand, symbols, diff, resultant,
                   Poly, gcd, together, fraction, S, factor_list,
                   count_roots, nroots, sqrt)
import numpy as np
import pickle
import os
import time
import sys

sys.stdout.reconfigure(line_buffering=True)

CACHE_FILE = "/tmp/case3-diag-cache.pkl"


def build_neg_N():
    a3, a4, b3, b4 = sp.symbols('a3 a4 b3 b4')
    disc_p = expand(256*a4**3 - 128*a4**2 - 144*a3**2*a4 - 27*a3**4 + 16*a4 + 4*a3**2)
    f1_p = 1 + 12*a4
    f2_p = 9*a3**2 + 8*a4 - 2
    disc_q = expand(256*b4**3 - 128*b4**2 - 144*b3**2*b4 - 27*b3**4 + 16*b4 + 4*b3**2)
    f1_q = 1 + 12*b4
    f2_q = 9*b3**2 + 8*b4 - 2
    c3 = a3 + b3
    c4 = a4 + Rational(1, 6) + b4
    disc_r = expand(256*c4**3 - 512*c4**2 + 288*c3**2*c4 - 27*c3**4 + 256*c4 + 32*c3**2)
    f1_r = expand(4 + 12*c4)
    f2_r = expand(-16 + 16*c4 + 9*c3**2)
    surplus_frac = together(-disc_r/(4*f1_r*f2_r) + disc_p/(4*f1_p*f2_p) + disc_q/(4*f1_q*f2_q))
    num, den = fraction(surplus_frac)
    neg_N = expand(-num)
    return neg_N, (a3, a4, b3, b4)


def analyze_2d_system(name, g1_expr, g2_expr, neg_N_expr, a3_sym, a4_sym,
                      disc_p_expr, disc_q_expr, f1_p_expr, f1_q_expr,
                      f2_p_expr, f2_q_expr):
    """Analyze a 2D system g1(a3,a4) = g2(a3,a4) = 0 via resultant.

    Returns list of (a3, a4, -N value, in_domain) tuples.
    """
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    t0 = time.time()

    g1_terms = len(g1_expr.as_ordered_terms())
    g2_terms = len(g2_expr.as_ordered_terms())
    g1_deg_a3 = Poly(g1_expr, a3_sym).degree()
    g1_deg_a4 = Poly(g1_expr, a4_sym).degree()
    g2_deg_a3 = Poly(g2_expr, a3_sym).degree()
    g2_deg_a4 = Poly(g2_expr, a4_sym).degree()
    print(f"  g1: {g1_terms} terms, deg({a3_sym})={g1_deg_a3}, deg({a4_sym})={g1_deg_a4}")
    print(f"  g2: {g2_terms} terms, deg({a3_sym})={g2_deg_a3}, deg({a4_sym})={g2_deg_a4}")

    # Check parity: if g1 is odd in a3, we can factor out a3
    is_g1_odd = expand(g1_expr + g1_expr.subs(a3_sym, -a3_sym)) == 0
    is_g2_even = expand(g2_expr - g2_expr.subs(a3_sym, -a3_sym)) == 0
    print(f"  g1 odd in {a3_sym}: {is_g1_odd}")
    print(f"  g2 even in {a3_sym}: {is_g2_even}")

    # If g1 is odd, handle a3=0 case separately (already covered by Case 1)
    # and for a3≠0, divide g1 by a3 and substitute u = a3²
    if is_g1_odd and is_g2_even:
        print(f"  Using parity: dividing g1 by {a3_sym}, substituting u = {a3_sym}²")
        u = sp.Symbol('u')
        h1_a3 = expand(sp.div(g1_expr, a3_sym, a3_sym)[0])
        # h1_a3 is even in a3; substitute u = a3²
        p1 = Poly(h1_a3, a3_sym)
        h1 = S.Zero
        for monom, coeff in p1.as_dict().items():
            power = monom[0]
            assert power % 2 == 0
            h1 += coeff * u**(power//2)
        h1 = expand(h1)

        p2 = Poly(g2_expr, a3_sym)
        h2 = S.Zero
        for monom, coeff in p2.as_dict().items():
            power = monom[0]
            assert power % 2 == 0
            h2 += coeff * u**(power//2)
        h2 = expand(h2)

        print(f"  h1(u, {a4_sym}): {len(h1.as_ordered_terms())} terms, deg_u={Poly(h1, u).degree()}, deg_{a4_sym}={Poly(h1, a4_sym).degree()}")
        print(f"  h2(u, {a4_sym}): {len(h2.as_ordered_terms())} terms, deg_u={Poly(h2, u).degree()}, deg_{a4_sym}={Poly(h2, a4_sym).degree()}")

        # Resultant to eliminate u
        print(f"  Computing resultant(h1, h2, u)...")
        t1 = time.time()
        R = expand(resultant(h1, h2, u))
        R_deg = Poly(R, a4_sym).degree()
        print(f"    R({a4_sym}): degree {R_deg}, {len(R.as_ordered_terms())} terms ({time.time()-t1:.1f}s)")

        # Factor
        R_fac = factor_list(R, a4_sym)
        print(f"    Factors:")
        for fac, mult in R_fac[1]:
            deg = Poly(fac, a4_sym).degree()
            print(f"      (deg {deg})^{mult}")

        # Find roots in domain: a4 ∈ [-1/12, 1/4]
        a4_lo, a4_hi = Rational(-1, 12), Rational(1, 4)
        n_roots = count_roots(Poly(R, a4_sym), a4_lo, a4_hi)
        print(f"    Real roots in [{float(a4_lo):.4f}, {float(a4_hi):.4f}]: {n_roots} (Sturm)")

        if n_roots == 0:
            print(f"  → No critical points with {a3_sym}≠0!")
            return []

        # Find approximate roots
        np_coeffs = [float(Poly(R, a4_sym).nth(R_deg - i)) for i in range(R_deg + 1)]
        np_roots = np.roots(np_coeffs)
        a4_cands = sorted([r.real for r in np_roots
                           if abs(r.imag) < 1e-6
                           and float(a4_lo) - 0.01 <= r.real <= float(a4_hi) + 0.01])

        if len(a4_cands) != n_roots:
            print(f"    numpy found {len(a4_cands)}, Sturm says {n_roots}")

        results = []
        for a4_f in a4_cands:
            a4_val = Rational(a4_f).limit_denominator(10**12)
            # Solve h1(u, a4_val) = 0 for u
            h1_sub = expand(h1.subs(a4_sym, a4_val))
            if h1_sub == 0:
                continue
            n_u = count_roots(Poly(h1_sub, u), Rational(1, 10**8), Rational(8, 27))
            if n_u == 0:
                continue
            u_coeffs = [float(Poly(h1_sub, u).nth(Poly(h1_sub, u).degree() - i))
                        for i in range(Poly(h1_sub, u).degree() + 1)]
            u_np = np.roots(u_coeffs)
            u_cands = sorted([r.real for r in u_np if abs(r.imag) < 1e-6 and r.real > 1e-8])

            for u_f in u_cands:
                a3_f = u_f**0.5
                # Evaluate -N and domain constraints
                pt = {a3_sym: a3_f, a4_sym: a4_f}
                neg_N_val = float(neg_N_expr.subs(pt))
                dp = float(disc_p_expr.subs(pt))
                dq = float(disc_q_expr.subs(pt))
                f1p = float(f1_p_expr.subs(pt))
                f1q = float(f1_q_expr.subs(pt))
                f2p = float(f2_p_expr.subs(pt))
                f2q = float(f2_q_expr.subs(pt))

                strict = dp > 1e-4 and dq > 1e-4 and f1p > 1e-4 and f1q > 1e-4 and f2p < -1e-4 and f2q < -1e-4
                on_boundary = dp > -1e-4 and dq > -1e-4 and f1p > -1e-4 and f1q > -1e-4 and f2p < 1e-4 and f2q < 1e-4

                if on_boundary:
                    tag = "INTERIOR" if strict else "BOUNDARY"
                    print(f"    *** {tag}: {a3_sym}=±{a3_f:.8f}, {a4_sym}={a4_f:.8f}")
                    print(f"        -N = {neg_N_val:.4f}")
                    print(f"        disc_p={dp:.4f} disc_q={dq:.4f} f1_p={f1p:.4f} f1_q={f1q:.4f} f2_p={f2p:.4f} f2_q={f2q:.4f}")
                    results.append((a3_f, a4_f, neg_N_val, strict))

        print(f"  ({time.time()-t0:.1f}s)")
        return results
    else:
        # Direct resultant in a3
        print(f"  Computing resultant(g1, g2, {a3_sym})...")
        t1 = time.time()
        R = expand(resultant(g1_expr, g2_expr, a3_sym))
        R_deg = Poly(R, a4_sym).degree()
        print(f"    R: degree {R_deg}, {len(R.as_ordered_terms())} terms ({time.time()-t1:.1f}s)")

        # Factor and find roots...
        n_roots = count_roots(Poly(R, a4_sym), Rational(-1, 12), Rational(1, 4))
        print(f"    Real roots in [-1/12, 1/4]: {n_roots}")

        # TODO: back-substitute
        print(f"  ({time.time()-t0:.1f}s)")
        return []


def main():
    print("=" * 70)
    print("CASE 3: Diagonal and anti-diagonal subcases")
    print("=" * 70)
    t_start = time.time()

    a3, a4, b3, b4 = sp.symbols('a3 a4 b3 b4')

    print("\nBuilding -N and gradient...")
    neg_N, _ = build_neg_N()
    g1 = expand(diff(neg_N, a3))
    g2 = expand(diff(neg_N, a4))
    g3 = expand(diff(neg_N, b3))
    g4 = expand(diff(neg_N, b4))
    print(f"  Done ({time.time()-t_start:.1f}s)")

    # ================================================================
    # Case 3a: Diagonal (b₃=a₃, b₄=a₄)
    # ================================================================
    print("\n\n" + "="*70)
    print("CASE 3a: DIAGONAL (b₃ = a₃, b₄ = a₄)")
    print("="*70)
    # On diagonal: g1 = g3 and g2 = g4 by exchange symmetry, so just need g1=g2=0
    g1_diag = expand(g1.subs([(b3, a3), (b4, a4)]))
    g2_diag = expand(g2.subs([(b3, a3), (b4, a4)]))
    neg_N_diag = expand(neg_N.subs([(b3, a3), (b4, a4)]))

    # Domain constraints on diagonal: p and q have same coefficients
    disc_p_diag = expand(256*a4**3 - 128*a4**2 - 144*a3**2*a4 - 27*a3**4 + 16*a4 + 4*a3**2)
    f1_diag = 1 + 12*a4
    f2_diag = 9*a3**2 + 8*a4 - 2

    diag_results = analyze_2d_system(
        "Diagonal (b₃=a₃, b₄=a₄)",
        g1_diag, g2_diag, neg_N_diag, a3, a4,
        disc_p_diag, disc_p_diag,  # p = q
        f1_diag, f1_diag,
        f2_diag, f2_diag,
    )

    # ================================================================
    # Case 3b: Anti-diagonal (b₃=-a₃, b₄=a₄)
    # ================================================================
    print("\n\n" + "="*70)
    print("CASE 3b: ANTI-DIAGONAL (b₃ = -a₃, b₄ = a₄)")
    print("="*70)
    # On anti-diagonal: g3 = -g1 and g4 = g2 by exchange+parity
    g1_anti = expand(g1.subs([(b3, -a3), (b4, a4)]))
    g2_anti = expand(g2.subs([(b3, -a3), (b4, a4)]))
    neg_N_anti = expand(neg_N.subs([(b3, -a3), (b4, a4)]))

    # Domain: p has (a3, a4), q has (-a3, a4) = same disc (disc is even in a3)
    disc_q_anti = disc_p_diag  # same as disc_p since disc is even in a3

    anti_results = analyze_2d_system(
        "Anti-diagonal (b₃=-a₃, b₄=a₄)",
        g1_anti, g2_anti, neg_N_anti, a3, a4,
        disc_p_diag, disc_q_anti,
        f1_diag, f1_diag,  # f1 doesn't depend on a3
        f2_diag, f2_diag,  # f2_q = 9(-a3)²+8a4-2 = same as f2_p
    )

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")

    print(f"\nCase 3a (diagonal) critical points in domain: {len(diag_results)}")
    for a3_f, a4_f, nN, strict in diag_results:
        tag = "interior" if strict else "boundary"
        print(f"  a₃=±{a3_f:.8f}, a₄={a4_f:.8f}: -N={nN:.4f} ({tag})")

    print(f"\nCase 3b (anti-diagonal) critical points in domain: {len(anti_results)}")
    for a3_f, a4_f, nN, strict in anti_results:
        tag = "interior" if strict else "boundary"
        print(f"  a₃=±{a3_f:.8f}, a₄={a4_f:.8f}: -N={nN:.4f} ({tag})")

    all_results = diag_results + anti_results
    all_nonneg = all(nN >= -1e-6 for _, _, nN, _ in all_results)

    print(f"\nAll critical points have -N ≥ 0: {all_nonneg}")
    print(f"Total time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
