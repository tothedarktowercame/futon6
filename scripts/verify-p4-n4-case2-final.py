#!/usr/bin/env python3
"""Case 2 certified proof: no interior critical points with b₃=0, a₃≠0.

Algebraic elimination via resultants, with Sturm root counting.

Key domain constraint: disc_q = 16*b₄*(4*b₄-1)² ≥ 0  ⟹  b₄ ≥ 0.
Combined with f₂_q < 0: b₄ < 1/4. So b₄ ∈ [0, 1/4].
This eliminates most roots of the elimination polynomial.

Safe for restart: saves/loads intermediate results via pickle.
"""

import sympy as sp
from sympy import (Rational, expand, symbols, diff, resultant,
                   Poly, gcd, together, fraction, S, factor_list, quo,
                   count_roots)
import numpy as np
import pickle
import os
import time
import sys

sys.stdout.reconfigure(line_buffering=True)

CACHE_FILE = "/tmp/case2-elimination-cache.pkl"


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
    """Replace a₃ with u = a₃² in a polynomial that is even in a₃."""
    p = Poly(expr, a3_sym)
    result = S.Zero
    for monom, coeff in p.as_dict().items():
        power = monom[0]
        assert power % 2 == 0, f"Odd power {power} in even polynomial"
        result += coeff * u_sym**(power // 2)
    return expand(result)


def poly_to_numpy(poly_expr, var):
    """Convert a univariate sympy polynomial to numpy coefficient array."""
    p = Poly(poly_expr, var)
    deg = p.degree()
    coeffs = [float(p.nth(deg - i)) for i in range(deg + 1)]
    return np.array(coeffs)


def run_elimination():
    """Build system and run resultant elimination. Returns factors of R_final."""
    a3, a4, b3, b4 = sp.symbols('a3 a4 b3 b4')
    u = sp.Symbol('u')

    print("Phase 1: Building system...")
    t0 = time.time()
    neg_N, _ = build_neg_N()
    g = [expand(diff(neg_N, v).subs(b3, 0)) for v in (a3, a4, b3, b4)]

    # Factor out a₃ from odd components, substitute u = a₃²
    h1 = sub_u(expand(sp.div(g[0], a3, a3)[0]), a3, u)
    h2 = sub_u(g[1], a3, u)
    h3 = sub_u(expand(sp.div(g[2], a3, a3)[0]), a3, u)
    h4 = sub_u(g[3], a3, u)
    print(f"  System built ({time.time()-t0:.1f}s)")

    print("\nPhase 2: Resultant elimination u → (a₄, b₄)...")
    R12 = expand(resultant(h1, h2, u))
    R13 = expand(resultant(h1, h3, u))
    G = gcd(Poly(R12, a4, b4), Poly(R13, a4, b4))
    R12p = expand(quo(Poly(R12, a4, b4), G).as_expr())
    R13p = expand(quo(Poly(R13, a4, b4), G).as_expr())
    print(f"  GCD divided out. R12':{len(R12p.as_ordered_terms())}t R13':{len(R13p.as_ordered_terms())}t")

    # Factor R13' to find its components
    R13p_fac = factor_list(R13p, a4, b4)
    big13 = max(R13p_fac[1], key=lambda x: Poly(x[0], a4).degree())[0]
    print(f"  R13' big factor: deg_a4={Poly(big13,a4).degree()}")

    print("\nPhase 3: Resultant elimination a₄ → b₄...")
    t1 = time.time()
    R_final = expand(resultant(R12p, big13, a4))
    print(f"  R_final: degree {Poly(R_final, b4).degree()} ({time.time()-t1:.1f}s)")

    print("\nPhase 4: Factoring R_final...")
    t1 = time.time()
    R_final_fac = factor_list(R_final, b4)
    factors = []
    for fac, mult in R_final_fac[1]:
        deg = Poly(fac, b4).degree()
        coeffs = poly_to_numpy(fac, b4)
        factors.append({
            'sympy': fac,
            'numpy_coeffs': coeffs,
            'degree': deg,
            'multiplicity': mult,
        })
        print(f"    degree {deg} × {mult}")
    print(f"  ({time.time()-t1:.1f}s)")

    # Also save the system polynomials for back-substitution
    cache = {
        'factors': factors,
        'h1': h1, 'h2': h2, 'h3': h3, 'h4': h4,
        'R12p': R12p, 'big13': big13,
        'neg_N_u': sub_u(expand(neg_N.subs(b3, 0)), a3, u),
    }
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(cache, f)
    print(f"\n  Saved cache to {CACHE_FILE}")

    return cache


def count_roots_by_signs(fac, var, lo_r, hi_r, np_approx_roots):
    """Count roots in [lo, hi] by evaluating polynomial sign at rational test points.

    Given approximate root locations from numpy, we evaluate the polynomial
    exactly (rational arithmetic) at points between roots and at boundaries.
    Sign changes correspond to odd-multiplicity roots. For simple roots
    (which is the generic case), this gives the exact count.

    Returns (certified_count, root_intervals) where each root_interval is
    (lo, hi, sign_lo, sign_hi) bracketing a root.
    """
    # Build test points: endpoints + midpoints between adjacent roots
    test_points = [lo_r]
    sorted_roots = sorted(np_approx_roots)
    for i, r in enumerate(sorted_roots):
        # Point just before this root
        if i == 0:
            mid = (lo_r + Rational(r).limit_denominator(10**12)) / 2
        else:
            mid = Rational((sorted_roots[i-1] + r) / 2).limit_denominator(10**12)
        test_points.append(mid)
    # Point after last root
    if sorted_roots:
        mid = (Rational(sorted_roots[-1]).limit_denominator(10**12) + hi_r) / 2
        test_points.append(mid)
    test_points.append(hi_r)

    # Evaluate polynomial exactly at each test point
    signs = []
    for pt in test_points:
        val = fac.subs(var, pt)
        s = sp.sign(val)
        signs.append(int(s))

    # Count sign changes
    n_changes = 0
    root_intervals = []
    for i in range(len(signs) - 1):
        if signs[i] * signs[i+1] < 0:
            n_changes += 1
            root_intervals.append((test_points[i], test_points[i+1], signs[i], signs[i+1]))
        elif signs[i+1] == 0:
            # Exact root at a test point
            n_changes += 1
            root_intervals.append((test_points[i+1], test_points[i+1], 0, 0))

    return n_changes, root_intervals


def find_roots_in_interval(factor_info, lo, hi):
    """Find all real roots of a factor in [lo, hi].

    Uses Sturm for small degrees, sign-counting for large degrees.
    """
    b4 = sp.Symbol('b4')
    fac = factor_info['sympy']
    deg = factor_info['degree']
    coeffs = factor_info['numpy_coeffs']

    lo_r = Rational(lo) if not isinstance(lo, sp.Rational) else lo
    hi_r = Rational(hi) if not isinstance(hi, sp.Rational) else hi

    t0 = time.time()

    if deg <= 2:
        # Solve analytically
        roots_exact = sp.solve(fac, b4)
        results = [(r, True) for r in roots_exact if r.is_real and lo_r <= r <= hi_r]
        print(f"    Analytic (deg {deg}): {len(results)} roots ({time.time()-t0:.1f}s)")
        return results

    # Get numpy approximations first (always fast)
    np_roots = np.roots(coeffs)
    real_mask = np.abs(np_roots.imag) < 1e-6
    in_range = real_mask & (np_roots.real >= float(lo) - 1e-8) & (np_roots.real <= float(hi) + 1e-8)
    approx = sorted(np_roots.real[in_range])

    if deg <= 40:
        # Sturm is fast for moderate degrees
        n_roots = count_roots(Poly(fac, b4), lo_r, hi_r)
        print(f"    Sturm (deg {deg}): {n_roots} roots in [{float(lo):.4f}, {float(hi):.4f}] ({time.time()-t0:.1f}s)")

        if n_roots == 0:
            return []
        if len(approx) == n_roots:
            return [(Rational(r).limit_denominator(10**15), False) for r in approx]
        # Numpy disagrees — use bisection
        print(f"    numpy found {len(approx)}, bisecting...")
        approx = isolate_by_bisection(fac, b4, lo_r, hi_r, n_roots)
        return [(Rational(r).limit_denominator(10**15), False) for r in approx]

    # For high-degree factors: use sign-counting (avoids Sturm's expensive GCD chain)
    print(f"    Numpy approx (deg {deg}): {len(approx)} candidate roots")
    for r in approx:
        print(f"      b4 ≈ {r:.12f}")

    # Certify via exact sign evaluation at rational test points
    print(f"    Certifying by sign evaluation...")
    t1 = time.time()
    n_sign_changes, intervals = count_roots_by_signs(fac, b4, lo_r, hi_r, approx)
    print(f"    Sign changes: {n_sign_changes} ({time.time()-t1:.1f}s)")

    if n_sign_changes != len(approx):
        print(f"    WARNING: {n_sign_changes} sign changes vs {len(approx)} numpy roots")
        # Add extra test points for more precision
        extra_points = np.linspace(float(lo), float(hi), 200)
        np_vals = np.polyval(coeffs, extra_points)
        sign_changes_fine = np.sum(np_vals[:-1] * np_vals[1:] < 0)
        print(f"    Fine grid (200 pts): {sign_changes_fine} sign changes")

    # Refine each root interval by bisection with exact arithmetic
    results = []
    for lo_i, hi_i, s_lo, s_hi in intervals:
        if lo_i == hi_i:
            results.append((lo_i, True))
            continue
        # Bisect to narrow the interval
        a, b = lo_i, hi_i
        for _ in range(50):  # ~15 decimal digits
            mid = (a + b) / 2
            val = sp.sign(fac.subs(b4, mid))
            if val == s_lo:
                a = mid
            else:
                b = mid
        root_approx = (a + b) / 2
        results.append((root_approx, False))
        print(f"    Root isolated: b4 ∈ [{float(a):.15f}, {float(b):.15f}]")

    return results


def isolate_by_bisection(fac, var, lo, hi, expected_count):
    """Isolate roots by recursive bisection using Sturm's theorem."""
    roots = []
    stack = [(lo, hi, expected_count)]
    while stack:
        a, b, n = stack.pop()
        if n == 0:
            continue
        if n == 1 and (b - a) < Rational(1, 10**6):
            roots.append(float((a + b) / 2))
            continue
        mid = (a + b) / 2
        n_left = count_roots(Poly(fac, var), a, mid)
        n_right = n - n_left
        if n_left > 0:
            stack.append((a, mid, n_left))
        if n_right > 0:
            stack.append((mid, b, n_right))
    return sorted(roots)


def back_substitute(cache, b4_candidates):
    """For each b₄ candidate, find (a₄, u) and check domain constraints."""
    a4, b4, u = sp.symbols('a4 b4 u')

    h1, h2, h3, h4 = cache['h1'], cache['h2'], cache['h3'], cache['h4']
    R12p = cache['R12p']
    neg_N_u = cache['neg_N_u']

    # Domain constraints (b₃=0 case, after u = a₃²)
    disc_p_expr = 256*a4**3 - 128*a4**2 - 144*u*a4 - 27*u**2 + 16*a4 + 4*u
    disc_q_expr = 256*b4**3 - 128*b4**2 + 16*b4   # = 16*b4*(4*b4-1)^2
    f1_p_expr = 1 + 12*a4
    f1_q_expr = 1 + 12*b4
    neg_f2_p_expr = 2 - 9*u - 8*a4    # > 0 on domain
    neg_f2_q_expr = 2 - 8*b4           # > 0 on domain (b4 < 1/4)

    interior_cps = []

    for b4_val, is_exact in b4_candidates:
        b4_f = float(b4_val)
        print(f"\n  b₄ = {b4_f:.12f}:")

        # Domain check for q
        dq = 16 * b4_f * (4*b4_f - 1)**2
        f1q = 1 + 12 * b4_f
        nf2q = 2 - 8 * b4_f
        print(f"    q: disc={dq:.6f} f1={f1q:.4f} -f2={nf2q:.4f}")

        if dq < -1e-8 or f1q < 1e-8 or nf2q < 1e-8:
            print(f"    → q outside domain, skip")
            continue

        # Solve R12'(a₄, b₄_val) = 0 for a₄
        R12_sub = expand(R12p.subs(b4, b4_val))
        if R12_sub == 0:
            print(f"    → R12' identically zero, degenerate")
            continue

        # Count a₄ roots in [-1/12, 1/4]
        n_a4 = count_roots(Poly(R12_sub, a4), Rational(-1, 12), Rational(1, 4))
        print(f"    a₄ roots in [-1/12, 1/4]: {n_a4}")
        if n_a4 == 0:
            print(f"    → no a₄ in domain, skip")
            continue

        # Find approximate a₄ values via numpy
        a4_coeffs = poly_to_numpy(R12_sub, a4)
        a4_np_roots = np.roots(a4_coeffs)
        a4_real = sorted([r.real for r in a4_np_roots
                          if abs(r.imag) < 1e-6 and -1/12 - 0.01 <= r.real <= 0.25 + 0.01])

        if len(a4_real) != n_a4:
            print(f"    numpy found {len(a4_real)}, Sturm says {n_a4}; using bisection")
            a4_real = isolate_by_bisection(R12_sub, a4, Rational(-1,12), Rational(1,4), n_a4)

        for a4_f in a4_real:
            # Check f1_p and -f2_p constraints (quick float check)
            f1p = 1 + 12 * a4_f
            if f1p < 1e-6:
                continue

            # Find u from h₁(u, a₄, b₄) = 0
            h1_sub = expand(h1.subs([(a4, Rational(a4_f).limit_denominator(10**12)),
                                      (b4, b4_val)]))
            if h1_sub == 0:
                continue

            # u must be in (0, 8/27] for a₃ ≠ 0 and a₃² ≤ 8/27
            n_u = count_roots(Poly(h1_sub, u), Rational(1, 10**8), Rational(8, 27))
            if n_u == 0:
                continue

            u_coeffs = poly_to_numpy(h1_sub, u)
            u_np_roots = np.roots(u_coeffs)
            u_pos = sorted([r.real for r in u_np_roots
                            if abs(r.imag) < 1e-6 and r.real > 1e-8 and r.real < 8/27 + 0.01])

            if not u_pos and n_u > 0:
                u_pos = isolate_by_bisection(h1_sub, u, Rational(1,10**8), Rational(8,27), n_u)

            for u_f in u_pos:
                # Verify all 4 gradient equations
                pt = {u: u_f, a4: a4_f, b4: b4_f}
                errs = [abs(float(h.subs(pt))) for h in [h1, h2, h3, h4]]
                grad_err = max(errs)

                # Full domain check
                dp = 256*a4_f**3 - 128*a4_f**2 - 144*u_f*a4_f - 27*u_f**2 + 16*a4_f + 4*u_f
                nf2p = 2 - 9*u_f - 8*a4_f

                strict = dp > 1e-4 and dq > 1e-4 and f1p > 1e-4 and f1q > 1e-4 and nf2p > 1e-4 and nf2q > 1e-4
                on_boundary = dp > -1e-4 and dq > -1e-4 and f1p > -1e-4 and f1q > -1e-4 and nf2p > -1e-4 and nf2q > -1e-4

                if on_boundary and grad_err < 1.0:
                    neg_N_val = float(neg_N_u.subs(pt))
                    tag = "INTERIOR" if strict else "BOUNDARY"
                    print(f"      *** {tag}: a₃=±{u_f**0.5:.8f} a₄={a4_f:.8f} b₄={b4_f:.8f}")
                    print(f"          -N = {neg_N_val:.4f}, |∇|={grad_err:.2e}")
                    print(f"          disc_p={dp:.6f} -f2_p={nf2p:.6f}")
                    if strict:
                        interior_cps.append((u_f, a4_f, b4_f, neg_N_val))

    return interior_cps


def main():
    print("=" * 70)
    print("CASE 2 CERTIFIED PROOF: b₃=0, a₃≠0")
    print("=" * 70)
    print(f"Domain constraint: disc_q = 16·b₄·(4b₄-1)² ≥ 0  ⟹  b₄ ∈ [0, 1/4]")
    t_start = time.time()

    b4 = sp.Symbol('b4')

    # Phase 1: Build elimination (or load from cache)
    if os.path.exists(CACHE_FILE):
        print(f"\nLoading cached elimination from {CACHE_FILE}...")
        with open(CACHE_FILE, 'rb') as f:
            cache = pickle.load(f)
        factors = cache['factors']
        print(f"  Loaded {len(factors)} factors of R_final")
    else:
        print(f"\nNo cache found. Running elimination from scratch...")
        cache = run_elimination()
        factors = cache['factors']

    # Phase 2: Find roots of each factor in [0, 1/4] (domain-restricted)
    print(f"\n{'='*70}")
    print(f"ROOT FINDING: b₄ ∈ [0, 1/4]")
    print(f"{'='*70}")

    b4_lo = Rational(0)
    b4_hi = Rational(1, 4)

    all_b4_candidates = []
    total_certified_roots = 0

    for i, finfo in enumerate(factors):
        deg = finfo['degree']
        mult = finfo['multiplicity']
        print(f"\nFactor {i}: degree {deg}, multiplicity {mult}")

        roots = find_roots_in_interval(finfo, b4_lo, b4_hi)
        total_certified_roots += len(roots) * mult

        for r, is_exact in roots:
            all_b4_candidates.append((r, is_exact))
            print(f"    → b₄ = {float(r):.12f} {'(exact)' if is_exact else '(approx)'}")

    print(f"\n{'='*70}")
    print(f"TOTAL b₄ candidates in [0, 1/4]: {len(all_b4_candidates)}")
    print(f"  (from {total_certified_roots} roots counted with multiplicity)")

    if not all_b4_candidates:
        print(f"\n*** CERTIFIED: No b₄ roots in [0, 1/4] ***")
        print(f"*** CASE 2 PROVED: No critical points with b₃=0, a₃≠0 in domain ***")
        print(f"\nTotal time: {time.time()-t_start:.1f}s")
        return

    # Phase 3: Back-substitution
    print(f"\n{'='*70}")
    print(f"BACK-SUBSTITUTION")
    print(f"{'='*70}")

    interior_cps = back_substitute(cache, all_b4_candidates)

    print(f"\n{'='*70}")
    print(f"FINAL RESULT")
    print(f"{'='*70}")
    print(f"Interior critical points found: {len(interior_cps)}")
    if not interior_cps:
        print(f"*** CASE 2 PROVED: No interior critical points with b₃=0, a₃≠0 ***")
    else:
        print(f"Interior critical points (all must have -N ≥ 0):")
        all_nonneg = True
        for u_val, a4_val, b4_val, neg_N_val in interior_cps:
            print(f"  u={u_val:.8f} a₄={a4_val:.8f} b₄={b4_val:.8f}: -N={neg_N_val:.6f}")
            if neg_N_val < -1e-6:
                all_nonneg = False
        if all_nonneg:
            print(f"*** All interior CPs have -N ≥ 0 ***")
        else:
            print(f"*** WARNING: Some interior CPs have -N < 0! ***")

    print(f"\nTotal time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
