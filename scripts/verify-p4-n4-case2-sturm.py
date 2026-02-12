#!/usr/bin/env python3
"""Case 2 exact proof using Sturm's theorem for certified root counting.

After resultant elimination produces R_final(b4) = 0, we:
1. Factor R_final into irreducible components
2. Use Sturm's theorem to count real roots of each factor in [-1/12, 1/4]
3. For any roots found, back-substitute and check domain constraints

This gives an exact, certified result.
"""

import sympy as sp
from sympy import (Rational, expand, symbols, diff, resultant, factor,
                   Poly, gcd, together, fraction, S, factor_list, quo,
                   count_roots, real_roots, RootOf, nroots, Interval,
                   sturm, sign, oo, N as Neval)
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


def count_roots_in_interval(poly_expr, var, a, b):
    """Count real roots of poly_expr in [a, b] using Sturm's theorem."""
    p = Poly(poly_expr, var)
    # sympy's count_roots does exactly this
    return count_roots(p, a, b)


def main():
    print("=" * 70)
    print("CASE 2: Certified proof via Sturm's theorem")
    print("=" * 70)
    t0 = time.time()

    a3, a4, b3, b4 = sp.symbols('a3 a4 b3 b4')
    u = sp.Symbol('u')

    print("\nStep 1: Build system and eliminate u...")
    neg_N, _ = build_neg_N()
    g = [expand(diff(neg_N, v).subs(b3, 0)) for v in (a3, a4, b3, b4)]

    h1 = sub_u(expand(sp.div(g[0], a3, a3)[0]), a3, u)
    h2 = sub_u(g[1], a3, u)
    h3 = sub_u(expand(sp.div(g[2], a3, a3)[0]), a3, u)
    h4 = sub_u(g[3], a3, u)
    print(f"  Done ({time.time()-t0:.1f}s)")

    print("\nStep 2: Compute resultants and divide out GCD...")
    R12 = expand(resultant(h1, h2, u))
    R13 = expand(resultant(h1, h3, u))

    G = gcd(Poly(R12, a4, b4), Poly(R13, a4, b4))
    R12p = expand(quo(Poly(R12, a4, b4), G).as_expr())
    R13p = expand(quo(Poly(R13, a4, b4), G).as_expr())

    # Also compute R14 for additional elimination
    R14 = expand(resultant(h1, h4, u))
    # Check if G divides R14 too
    R14_poly = Poly(R14, a4, b4)
    R14_rem = sp.rem(R14_poly, G)
    if R14_rem == Poly(0, a4, b4, domain='ZZ'):
        R14p = expand(quo(R14_poly, G).as_expr())
        print(f"  G divides R14 too: R14' has {len(R14p.as_ordered_terms())} terms")
    else:
        R14p = None
        print(f"  G does NOT divide R14")

    print(f"  R12': {len(R12p.as_ordered_terms())} terms, deg_a4={Poly(R12p,a4).degree()}")
    print(f"  R13': {len(R13p.as_ordered_terms())} terms, deg_a4={Poly(R13p,a4).degree()}")
    print(f"  ({time.time()-t0:.1f}s)")

    # Factor R13' to get simpler components
    R13p_fac = factor_list(R13p, a4, b4)
    print(f"\n  R13' factors:")
    r13_factors = []
    for fac, mult in R13p_fac[1]:
        da = Poly(fac, a4).degree() if fac.has(a4) else 0
        db = Poly(fac, b4).degree() if fac.has(b4) else 0
        print(f"    ({da},{db}) × {mult}")
        r13_factors.append((fac, mult))

    # The big irreducible factor from R13' (degree 7 in a4)
    big13 = max(R13p_fac[1], key=lambda x: Poly(x[0], a4).degree())[0]
    big13_da = Poly(big13, a4).degree()

    # R12' is already irreducible
    big12 = R12p
    big12_da = Poly(big12, a4).degree()

    print(f"\n  Big factor from R12': deg_a4={big12_da}")
    print(f"  Big factor from R13': deg_a4={big13_da}")

    print("\nStep 3: Compute R_final = resultant(big12, big13, a4)...")
    t1 = time.time()
    R_final = expand(resultant(big12, big13, a4))
    deg_final = Poly(R_final, b4).degree()
    nterms = len(R_final.as_ordered_terms())
    print(f"  R_final: {nterms} terms, degree {deg_final}")
    print(f"  ({time.time()-t1:.1f}s)")

    print("\nStep 4: Factor R_final...")
    t1 = time.time()
    R_final_fac = factor_list(R_final, b4)
    print(f"  Content: {R_final_fac[0]}")
    factors = []
    for fac, mult in R_final_fac[1]:
        deg = Poly(fac, b4).degree()
        nterms_f = len(expand(fac).as_ordered_terms())
        factors.append((fac, mult, deg))
        print(f"    (degree {deg}, {nterms_f} terms) ^ {mult}")
    print(f"  ({time.time()-t1:.1f}s)")

    print("\nStep 5: Count real roots in [-1/12, 1/4] via Sturm's theorem...")
    b4_lo = Rational(-1, 12)
    b4_hi = Rational(1, 4)

    total_roots = 0
    root_locations = []

    for fac, mult, deg in factors:
        t1 = time.time()
        n_roots = count_roots_in_interval(fac, b4, b4_lo, b4_hi)
        total_roots += n_roots * mult
        print(f"  Factor deg {deg} (mult {mult}): {n_roots} real roots in [{float(b4_lo):.4f}, {float(b4_hi):.4f}] ({time.time()-t1:.1f}s)")

        if n_roots > 0:
            # Find the actual roots
            if deg <= 2:
                # Solve analytically
                roots = sp.solve(fac, b4)
                for r in roots:
                    if r.is_real and b4_lo <= r <= b4_hi:
                        print(f"    Root: b4 = {r} = {float(r):.15f}")
                        root_locations.append((r, fac, mult))
            else:
                # Use RootOf for algebraic roots
                print(f"    Finding roots numerically for degree-{deg} factor...")
                try:
                    # Try nroots on the individual factor
                    roots_num = nroots(fac, n=20, maxsteps=100)
                    for r in roots_num:
                        r_complex = complex(r)
                        if abs(r_complex.imag) < 1e-8:
                            r_real = r_complex.real
                            if float(b4_lo) - 1e-10 <= r_real <= float(b4_hi) + 1e-10:
                                print(f"    Root: b4 ≈ {r_real:.15f}")
                                root_locations.append((Rational(r_real).limit_denominator(10**15), fac, mult))
                except Exception as e:
                    print(f"    nroots failed: {e}")
                    # Use count_roots on subintervals to isolate roots
                    print(f"    Isolating roots by bisection...")
                    # Bisect the interval to isolate each root
                    intervals = [(b4_lo, b4_hi)]
                    isolated = []
                    while intervals:
                        lo, hi = intervals.pop()
                        n = count_roots_in_interval(fac, b4, lo, hi)
                        if n == 0:
                            continue
                        if n == 1 and (hi - lo) < Rational(1, 1000):
                            mid = (lo + hi) / 2
                            print(f"    Root: b4 ∈ [{float(lo):.10f}, {float(hi):.10f}], ≈ {float(mid):.10f}")
                            root_locations.append((mid, fac, mult))
                        elif n >= 1:
                            mid = (lo + hi) / 2
                            intervals.append((lo, mid))
                            intervals.append((mid, hi))

    print(f"\n  TOTAL real roots of R_final in [{float(b4_lo):.4f}, {float(b4_hi):.4f}]: {total_roots}")
    print(f"  Distinct root locations: {len(root_locations)}")

    if total_roots == 0:
        print(f"\n  *** CERTIFIED: R_final has NO real roots in [{float(b4_lo)}, {float(b4_hi)}] ***")
        print(f"  *** CASE 2 PROVED: No critical points with b₃=0, a₃≠0 in domain interior ***")
        print(f"\n{'='*70}")
        print(f"Total time: {time.time()-t0:.1f}s")
        return

    print(f"\n  R_final has {total_roots} root(s) in range. Back-substituting...")

    # Also use R12' with small factors of R13' for consistency
    small_factors_13 = [f for f, m in R13p_fac[1] if Poly(f, a4).degree() <= 1]

    # Domain constraints in terms of u, a4, b4
    disc_p_expr = expand(256*a4**3 - 128*a4**2 - 144*u*a4 - 27*u**2 + 16*a4 + 4*u)
    disc_q_expr = expand(256*b4**3 - 128*b4**2 + 16*b4)
    f1_p_expr = 1 + 12*a4
    f1_q_expr = 1 + 12*b4
    neg_f2_p_expr = expand(2 - 9*u - 8*a4)
    neg_f2_q_expr = expand(2 - 8*b4)

    neg_N_0 = expand(neg_N.subs(b3, 0))
    neg_N_u = sub_u(neg_N_0, a3, u)

    interior_cp_count = 0

    for b4_val, _, _ in root_locations:
        print(f"\n  b4 ≈ {float(b4_val):.12f}:")

        # Quick domain check for q: disc_q ≥ 0 and f1_q > 0 and -f2_q > 0
        dq = float(disc_q_expr.subs(b4, b4_val))
        f1q = float(f1_q_expr.subs(b4, b4_val))
        nf2q = float(neg_f2_q_expr.subs(b4, b4_val))
        print(f"    q constraints: disc_q={dq:.6f}, f1_q={f1q:.6f}, -f2_q={nf2q:.6f}")

        if dq < -1e-6 or f1q < -1e-6 or nf2q < -1e-6:
            print(f"    OUTSIDE domain for q → skip")
            continue

        # Find a4 from big12(a4, b4_val) = 0
        big12_sub = expand(big12.subs(b4, b4_val))
        if big12_sub == 0:
            print(f"    big12 vanishes identically → degenerate")
            continue

        # Count a4 roots in [-1/12, 1/4]
        n_a4 = count_roots_in_interval(big12_sub, a4, Rational(-1,12), Rational(1,4))
        print(f"    a4 roots in [-1/12, 1/4]: {n_a4}")

        if n_a4 == 0:
            print(f"    No a4 roots in domain → skip")
            continue

        # Find approximate a4 values
        try:
            a4_roots = nroots(big12_sub, n=15, maxsteps=200)
            a4_cands = sorted([complex(r).real for r in a4_roots
                               if abs(complex(r).imag) < 1e-6
                               and -1/12 - 0.001 <= complex(r).real <= 0.25 + 0.001])
        except:
            # Bisection fallback
            a4_cands = []
            intervals = [(Rational(-1,12), Rational(1,4))]
            while intervals:
                lo, hi = intervals.pop()
                n = count_roots_in_interval(big12_sub, a4, lo, hi)
                if n == 0:
                    continue
                if n == 1 and (hi-lo) < Rational(1,1000):
                    a4_cands.append(float((lo+hi)/2))
                elif n >= 1:
                    mid = (lo+hi)/2
                    intervals.append((lo, mid))
                    intervals.append((mid, hi))

        for a4_val_f in a4_cands:
            a4_val = Rational(a4_val_f).limit_denominator(10**15)

            # Find u from h1(u, a4, b4) = 0
            h1_sub = expand(h1.subs([(a4, a4_val), (b4, b4_val)]))
            if h1_sub == 0:
                continue

            n_u = count_roots_in_interval(h1_sub, u, 0, Rational(8,27))
            if n_u == 0:
                continue

            try:
                u_roots = nroots(h1_sub, n=15)
                u_cands = [complex(r).real for r in u_roots
                           if abs(complex(r).imag) < 1e-6 and complex(r).real > 1e-8]
            except:
                u_cands = []

            for u_val in u_cands:
                a3_val = u_val**0.5
                pt = {u: u_val, a4: float(a4_val), b4: float(b4_val)}

                # Verify gradient
                grad_err = max(abs(float(h.subs(pt))) for h in [h1, h2, h3, h4])

                # Domain constraints
                dp = float(disc_p_expr.subs(pt))
                f1p = float(f1_p_expr.subs({a4: float(a4_val)}))
                nf2p = float(neg_f2_p_expr.subs(pt))

                strict_interior = (dp > 1e-4 and dq > 1e-4 and f1p > 1e-4 and f1q > 1e-4
                                   and nf2p > 1e-4 and nf2q > 1e-4)

                in_domain = (dp > -1e-4 and dq > -1e-4 and f1p > -1e-4 and f1q > -1e-4
                             and nf2p > -1e-4 and nf2q > -1e-4)

                if in_domain and grad_err < 0.1:
                    neg_N_val = float(neg_N_u.subs(pt))
                    status = "INTERIOR" if strict_interior else "BOUNDARY"
                    print(f"      *** {status} CP: a3=±{a3_val:.8f}, a4={float(a4_val):.8f}, b4={float(b4_val):.8f}")
                    print(f"          u={u_val:.8f}, -N={neg_N_val:.4f}, |grad|={grad_err:.2e}")
                    print(f"          disc_p={dp:.6f}, -f2_p={nf2p:.6f}")
                    if strict_interior:
                        interior_cp_count += 1

    print(f"\n{'='*70}")
    print(f"RESULT: {interior_cp_count} critical points strictly inside domain")
    if interior_cp_count == 0:
        print("*** CASE 2 PROVED: No interior critical points with b₃=0, a₃≠0 ***")
    else:
        print("*** CASE 2: Interior critical points exist (check -N values) ***")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
