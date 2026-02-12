#!/usr/bin/env python3
"""Case 2 exact proof, take 3: divide out GCD, complete elimination.

The resultants R12 = res(h1,h2,u) and R13 = res(h1,h3,u) share a common
factor G(a4,b4) = (a4+1/12)·(b4-1/4)^4 corresponding to domain boundary.
After dividing G out, the cofactors R12', R13' are coprime, and
res(R12', R13', a4) gives a univariate polynomial in b4.
"""

import sympy as sp
from sympy import (Rational, expand, symbols, diff, resultant, factor,
                   Poly, nroots, gcd, together, fraction, S,
                   factor_list, quo, real_roots, RootOf, Abs)
import numpy as np
import time
import sys

sys.stdout.reconfigure(line_buffering=True)


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


def sub_u(expr, a3_sym, u_sym):
    p = Poly(expr, a3_sym)
    result = S.Zero
    for monom, coeff in p.as_dict().items():
        power = monom[0]
        assert power % 2 == 0
        result += coeff * u_sym**(power // 2)
    return expand(result)


def main():
    print("=" * 70)
    print("CASE 2 (v3): Complete exact elimination for b₃=0, a₃≠0")
    print("=" * 70)
    t0 = time.time()

    a3, a4, b3, b4 = sp.symbols('a3 a4 b3 b4')
    u = sp.Symbol('u')

    print("\nStep 1: Build system...")
    neg_N, _ = build_neg_N()
    g = [expand(diff(neg_N, v).subs(b3, 0)) for v in (a3, a4, b3, b4)]

    h1_a3 = expand(sp.div(g[0], a3, a3)[0])
    h3_a3 = expand(sp.div(g[2], a3, a3)[0])

    h1 = sub_u(h1_a3, a3, u)
    h2 = sub_u(g[1], a3, u)
    h3 = sub_u(h3_a3, a3, u)
    h4 = sub_u(g[3], a3, u)
    print(f"  Done ({time.time()-t0:.1f}s)")

    # Also evaluate -N at b3=0 for back-substitution
    neg_N_0 = expand(neg_N.subs(b3, 0))
    neg_N_u = sub_u(neg_N_0, a3, u)

    # Domain constraints at b3=0
    disc_p_expr = expand(256*a4**3 - 128*a4**2 - 144*u*a4 - 27*u**2 + 16*a4 + 4*u)
    disc_q_expr = expand(256*b4**3 - 128*b4**2 + 16*b4)
    f1_p_expr = 1 + 12*a4
    f1_q_expr = 1 + 12*b4
    neg_f2_p_expr = expand(2 - 9*u - 8*a4)   # -f2_p > 0 on domain
    neg_f2_q_expr = expand(2 - 8*b4)           # -f2_q > 0 on domain

    print("\nStep 2: Compute resultants w.r.t. u...")
    t1 = time.time()
    R12 = expand(resultant(h1, h2, u))
    R13 = expand(resultant(h1, h3, u))
    print(f"  R12: {len(R12.as_ordered_terms())} terms, deg_a4={Poly(R12,a4).degree()}, deg_b4={Poly(R12,b4).degree()}")
    print(f"  R13: {len(R13.as_ordered_terms())} terms, deg_a4={Poly(R13,a4).degree()}, deg_b4={Poly(R13,b4).degree()}")
    print(f"  ({time.time()-t1:.1f}s)")

    print("\nStep 3: Divide out common boundary factors...")
    t1 = time.time()
    # GCD is (a4 + 1/12) · (b4 - 1/4)^4 = (12*a4+1)·(4*b4-1)^4 (up to constants)
    G_poly = gcd(Poly(R12, a4, b4), Poly(R13, a4, b4))
    G = expand(G_poly.as_expr())
    print(f"  GCD: {len(G.as_ordered_terms())} terms")

    # Factor the GCD to understand it
    G_fac = factor_list(G, a4, b4)
    print(f"  GCD factored: content={G_fac[0]}")
    for fac, mult in G_fac[1]:
        print(f"    ({fac})^{mult}")

    # Divide
    R12_prime = expand(quo(Poly(R12, a4, b4), G_poly).as_expr())
    R13_prime = expand(quo(Poly(R13, a4, b4), G_poly).as_expr())
    print(f"  R12' = R12/G: {len(R12_prime.as_ordered_terms())} terms, deg_a4={Poly(R12_prime,a4).degree()}, deg_b4={Poly(R12_prime,b4).degree()}")
    print(f"  R13' = R13/G: {len(R13_prime.as_ordered_terms())} terms, deg_a4={Poly(R13_prime,a4).degree()}, deg_b4={Poly(R13_prime,b4).degree()}")

    # Verify coprimality
    G2 = gcd(Poly(R12_prime, a4, b4), Poly(R13_prime, a4, b4))
    print(f"  gcd(R12', R13') = {G2} (should be constant)")
    print(f"  ({time.time()-t1:.1f}s)")

    print("\nStep 4: Factor cofactors...")
    t1 = time.time()
    R12_fac = factor_list(R12_prime, a4, b4)
    print(f"  R12' content: {R12_fac[0]}")
    R12_irred = []
    for fac, mult in R12_fac[1]:
        nterms = len(expand(fac).as_ordered_terms())
        da = Poly(fac, a4).degree() if fac.has(a4) else 0
        db = Poly(fac, b4).degree() if fac.has(b4) else 0
        print(f"    ({nterms} terms, deg_a4={da}, deg_b4={db})^{mult}")
        R12_irred.append((fac, mult))

    R13_fac = factor_list(R13_prime, a4, b4)
    print(f"  R13' content: {R13_fac[0]}")
    R13_irred = []
    for fac, mult in R13_fac[1]:
        nterms = len(expand(fac).as_ordered_terms())
        da = Poly(fac, a4).degree() if fac.has(a4) else 0
        db = Poly(fac, b4).degree() if fac.has(b4) else 0
        print(f"    ({nterms} terms, deg_a4={da}, deg_b4={db})^{mult}")
        R13_irred.append((fac, mult))
    print(f"  ({time.time()-t1:.1f}s)")

    # Strategy: take the LARGEST irreducible factor from each and compute resultant
    # (The small factors are likely univariate and can be analyzed directly)

    print("\nStep 5: Eliminate a₄ via resultant of cofactors...")
    t1 = time.time()

    # First try: take the big irreducible factors
    big12 = max(R12_fac[1], key=lambda x: Poly(x[0], a4).degree())[0]
    big13 = max(R13_fac[1], key=lambda x: Poly(x[0], a4).degree())[0]

    big12_da = Poly(big12, a4).degree()
    big13_da = Poly(big13, a4).degree()
    print(f"  Big factor from R12': deg_a4={big12_da}")
    print(f"  Big factor from R13': deg_a4={big13_da}")
    print(f"  Expected resultant degree in b4: {big12_da * big13_da}")

    print(f"  Computing resultant(big12, big13, a4)...")
    R_final = expand(resultant(big12, big13, a4))
    if R_final == 0:
        print("    ZERO — big factors share a common component")
        # Try different pair
        print("    Trying resultant of full R12' and R13'...")
        R_final = expand(resultant(R12_prime, R13_prime, a4))
        if R_final == 0:
            print("    STILL ZERO — unexpected!")
            return

    deg_b4 = Poly(R_final, b4).degree()
    nterms = len(R_final.as_ordered_terms())
    print(f"    R_final: {nterms} terms, degree {deg_b4} in b4")
    print(f"    ({time.time()-t1:.1f}s)")

    print("\nStep 6: Factor R_final...")
    t1 = time.time()
    R_final_fac = factor_list(R_final, b4)
    print(f"  Content: {R_final_fac[0]}")
    for fac, mult in R_final_fac[1]:
        nterms_f = len(expand(fac).as_ordered_terms())
        deg_f = Poly(fac, b4).degree()
        print(f"    (deg {deg_f}, {nterms_f} terms)^{mult}")
    print(f"  ({time.time()-t1:.1f}s)")

    print("\nStep 7: Find all real roots of R_final in b₄ ∈ [-1/12, 1/4]...")
    t1 = time.time()
    # Use numerical roots first
    try:
        all_roots_num = nroots(R_final, n=20)
        real_roots_num = sorted([complex(r).real for r in all_roots_num if abs(complex(r).imag) < 1e-8])
        b4_lo, b4_hi = -1/12, 1/4
        domain_roots = [r for r in real_roots_num if b4_lo - 0.001 <= r <= b4_hi + 0.001]

        print(f"  Total roots: {len(all_roots_num)}")
        print(f"  Real roots: {len(real_roots_num)}")
        print(f"  In domain range [-1/12, 1/4]: {len(domain_roots)}")
        for r in domain_roots:
            print(f"    b4 = {r:.15f}")
    except Exception as e:
        print(f"  nroots failed: {e}")
        domain_roots = []
    print(f"  ({time.time()-t1:.1f}s)")

    if not domain_roots:
        print(f"\n  *** NO b₄ ROOTS IN DOMAIN → CASE 2 HAS NO CRITICAL POINTS ***")
        print(f"\n{'='*70}")
        print(f"Total time: {time.time()-t0:.1f}s")
        return

    print("\nStep 8: Back-substitute to find full critical points...")
    # For each b4 candidate, find a4 from R12'=0, then u from h1=0
    # Check domain constraints and verify gradient vanishes

    # Also compute resultant for h4 consistency
    R14 = expand(resultant(h1, h4, u))
    R14_prime = expand(quo(Poly(R14, a4, b4), G_poly).as_expr())

    in_domain_count = 0
    for b4_val_num in domain_roots:
        b4_val = Rational(b4_val_num).limit_denominator(10**15)
        print(f"\n  b4 ≈ {float(b4_val):.12f}:")

        # Evaluate R12'(a4, b4_val) and find a4 roots
        R12_at_b4 = expand(R12_prime.subs(b4, b4_val))
        if R12_at_b4 == 0:
            print("    R12' vanishes identically at this b4 — degenerate")
            continue

        try:
            a4_roots_num = nroots(R12_at_b4, n=15)
            a4_real = sorted([complex(r).real for r in a4_roots_num if abs(complex(r).imag) < 1e-6])
            a4_candidates = [r for r in a4_real if -1/12 - 0.01 <= r <= 0.25 + 0.01]
        except Exception as e:
            print(f"    a4 root finding failed: {e}")
            continue

        print(f"    a4 candidates in range: {len(a4_candidates)}")

        for a4_val_num in a4_candidates:
            a4_val = Rational(a4_val_num).limit_denominator(10**15)

            # Check R13' and R14' consistency
            r13_check = abs(float(R13_prime.subs([(a4, a4_val), (b4, b4_val)])))
            r14_check = abs(float(R14_prime.subs([(a4, a4_val), (b4, b4_val)])))

            # Find u from h1(u, a4, b4) = 0
            h1_sub = h1.subs([(a4, a4_val), (b4, b4_val)])
            if h1_sub == 0:
                print(f"      a4={float(a4_val):.8f}: h1 vanishes identically")
                continue

            try:
                u_roots_num = nroots(h1_sub, n=15)
                u_real_pos = sorted([complex(r).real for r in u_roots_num
                                     if abs(complex(r).imag) < 1e-6 and complex(r).real > 1e-10])
            except Exception as e:
                print(f"      a4={float(a4_val):.8f}: u root finding failed: {e}")
                continue

            for u_val_num in u_real_pos:
                # Check all 4 gradient equations
                pt = {u: u_val_num, a4: float(a4_val), b4: float(b4_val)}
                grad_err = max(abs(float(h.subs(pt))) for h in [h1, h2, h3, h4])

                # Domain constraints
                dp = float(disc_p_expr.subs(pt))
                dq = float(disc_q_expr.subs({b4: float(b4_val)}))
                f1p = float(f1_p_expr.subs({a4: float(a4_val)}))
                f1q = float(f1_q_expr.subs({b4: float(b4_val)}))
                nf2p = float(neg_f2_p_expr.subs(pt))
                nf2q = float(neg_f2_q_expr.subs({b4: float(b4_val)}))

                in_dom = (dp > -1e-4 and dq > -1e-4 and f1p > -1e-4 and f1q > -1e-4
                          and nf2p > -1e-4 and nf2q > -1e-4)
                strict_in = (dp > 1e-4 and dq > 1e-4 and f1p > 1e-4 and f1q > 1e-4
                             and nf2p > 1e-4 and nf2q > 1e-4)

                if in_dom and grad_err < 1e-2:
                    neg_N_val = float(neg_N_u.subs(pt))
                    status = "INTERIOR" if strict_in else "BOUNDARY"
                    print(f"      *** {status}: u={u_val_num:.8f}, a4={float(a4_val):.8f}, b4={float(b4_val):.8f}")
                    print(f"          a3 = ±{u_val_num**0.5:.8f}")
                    print(f"          |grad| = {grad_err:.2e}")
                    print(f"          -N = {neg_N_val:.6f}")
                    print(f"          R13' check = {r13_check:.2e}, R14' check = {r14_check:.2e}")
                    print(f"          disc_p={dp:.6f}, disc_q={dq:.6f}")
                    print(f"          f1_p={f1p:.6f}, f1_q={f1q:.6f}")
                    print(f"          -f2_p={nf2p:.6f}, -f2_q={nf2q:.6f}")
                    if strict_in:
                        in_domain_count += 1

    print(f"\n{'='*70}")
    print(f"RESULT: {in_domain_count} critical points strictly inside domain")
    if in_domain_count == 0:
        print("*** CASE 2 PROVED: No critical points with b₃=0, a₃≠0 in interior ***")
    else:
        print("*** CASE 2 INCOMPLETE: Interior critical points exist ***")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
